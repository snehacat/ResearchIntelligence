"""Verify paper metadata against scholarly registries and inspect OA listings."""

import difflib
import json
import re
import time
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from scholarly_search.crossref_client import ScholarlySearchError


class PaperMetadataVerifier:
    """Compare supplied metadata with Crossref and OpenAlex registry records.

    The verifier contacts only the registry endpoints. It never follows URLs
    supplied in paper metadata or returned by a registry.
    """

    CROSSREF_URL = "https://api.crossref.org/works"
    OPENALEX_URL = "https://api.openalex.org/works"
    REQUEST_TIMEOUT = 10.0
    MAX_RETRIES = 2
    MAX_RESPONSE_BYTES = 2_000_000
    MAX_DOI_LENGTH = 300
    TITLE_MATCH_THRESHOLD = 0.9
    USER_AGENT = "ResearchIntelligence/0.1 (metadata verification)"

    def __init__(
        self,
        timeout: float = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0 or max_retries > 5:
            raise ValueError("max_retries must be between 0 and 5")
        self.timeout = timeout
        self.max_retries = max_retries

    def verify(self, paper: Mapping[str, Any]) -> Dict[str, Any]:
        """Return registry comparisons and open-access metadata for a paper."""
        if not isinstance(paper, Mapping):
            raise ScholarlySearchError("paper must be a mapping")

        title = self._clean_text(paper.get("title"))
        doi = self._clean_doi(paper.get("doi"))
        if paper.get("doi") and (
            not doi or len(doi) > self.MAX_DOI_LENGTH
        ):
            raise ScholarlySearchError("paper DOI is not a valid DOI")
        if not title and not doi:
            raise ScholarlySearchError("paper must include a title or DOI")
        if len(title) > 500:
            raise ScholarlySearchError("paper title must not exceed 500 characters")

        provider_errors = []
        try:
            crossref_work = (
                self._crossref_by_doi(doi)
                if doi
                else self._crossref_by_title(title)
            )
        except ScholarlySearchError as error:
            crossref_work = None
            provider_errors.append({"provider": "Crossref", "error": str(error)})
        matched_doi = self._clean_doi(
            crossref_work.get("DOI") if crossref_work else None
        )
        try:
            openalex_work = self._openalex_by_doi(doi or matched_doi)
        except ScholarlySearchError as error:
            openalex_work = None
            provider_errors.append({"provider": "OpenAlex", "error": str(error)})

        crossref_record = self._normalize_crossref(crossref_work)
        openalex_record = self._normalize_openalex(openalex_work)
        expected_doi = doi or matched_doi
        openalex_doi = (
            openalex_record.get("doi") if openalex_record else None
        )
        if openalex_record and expected_doi:
            openalex_record["doi_matches_lookup"] = (
                bool(openalex_doi)
                and expected_doi.casefold() == openalex_doi.casefold()
            )
        field_checks = self._compare_metadata(
            title=title,
            doi=doi,
            publication_year=paper.get("publication_year", paper.get("year")),
            authors=paper.get("authors"),
            registry_record=crossref_record,
        )
        if expected_doi and openalex_record:
            field_checks["openalex_doi"] = {
                "provided": expected_doi,
                "registry": openalex_doi,
                "matches": (
                    bool(openalex_doi)
                    and expected_doi.casefold() == openalex_doi.casefold()
                ),
            }

        warnings = []
        warnings.extend(
            f"{error['provider']} lookup failed: {error['error']}"
            for error in provider_errors
        )
        if not crossref_record and not any(
            error["provider"] == "Crossref" for error in provider_errors
        ):
            warnings.append("No matching Crossref record was found.")
        if not openalex_record and not any(
            error["provider"] == "OpenAlex" for error in provider_errors
        ):
            warnings.append("No matching OpenAlex record was found.")
        if field_checks.get("title", {}).get("matches") is False:
            warnings.append("The supplied title differs from the Crossref title.")
        if field_checks.get("doi", {}).get("matches") is False:
            warnings.append("The supplied DOI differs from the Crossref DOI.")
        if field_checks.get("publication_year", {}).get("matches") is False:
            warnings.append(
                "The supplied publication year differs from the Crossref year."
            )
        if field_checks.get("openalex_doi", {}).get("matches") is False:
            warnings.append("OpenAlex returned a different DOI for the matched work.")

        if provider_errors and not crossref_record and not openalex_record:
            status = "unavailable"
        elif not crossref_record and not openalex_record:
            status = "not_found"
        elif any(check.get("matches") is False for check in field_checks.values()):
            status = "mismatch"
        elif crossref_record and openalex_record:
            status = "verified"
        else:
            status = "partial"

        return {
            "status": status,
            "input": {
                "title": title or None,
                "doi": doi or None,
                "publication_year": paper.get(
                    "publication_year", paper.get("year")
                ),
            },
            "field_checks": field_checks,
            "crossref": crossref_record,
            "openalex": openalex_record,
            "open_access": self._open_access_summary(
                openalex_work,
                expected_doi=expected_doi,
            ),
            "warnings": warnings,
            "provider_errors": provider_errors,
            "verification_scope": (
                "Registry metadata consistency only; this does not verify "
                "paper contents, peer review, or live URL accessibility."
            ),
        }

    def _crossref_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        escaped_doi = quote(doi, safe="/-._;()")
        payload = self._request_json(
            "Crossref",
            f"{self.CROSSREF_URL}/{escaped_doi}",
            not_found_is_empty=True,
        )
        if payload is None:
            return None
        message = payload.get("message")
        if not isinstance(message, dict):
            raise ScholarlySearchError("Crossref returned an invalid work record")
        return message

    def _crossref_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        query = urlencode({"query.bibliographic": title, "rows": 5})
        payload = self._request_json("Crossref", f"{self.CROSSREF_URL}?{query}")
        message = payload.get("message")
        items = message.get("items") if isinstance(message, dict) else None
        if not isinstance(items, list):
            raise ScholarlySearchError("Crossref returned an invalid search result")

        normalized_input = self._normalize_title(title)
        best_work = None
        best_similarity = 0.0
        for item in items:
            if not isinstance(item, dict):
                continue
            titles = item.get("title")
            candidate_title = (
                self._clean_text(titles[0])
                if isinstance(titles, list) and titles
                else ""
            )
            if not candidate_title:
                continue
            similarity = difflib.SequenceMatcher(
                None,
                normalized_input,
                self._normalize_title(candidate_title),
            ).ratio()
            if similarity > best_similarity:
                best_similarity = similarity
                best_work = item
        if best_similarity < self.TITLE_MATCH_THRESHOLD:
            return None
        return best_work

    def _openalex_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        if not doi:
            return None
        escaped_doi = quote(doi, safe="/-._;()")
        payload = self._request_json(
            "OpenAlex",
            f"{self.OPENALEX_URL}/doi:{escaped_doi}",
            not_found_is_empty=True,
        )
        if payload is not None and not isinstance(payload, dict):
            raise ScholarlySearchError("OpenAlex returned an invalid work record")
        if payload is not None and not any(
            self._clean_text(payload.get(field))
            for field in ("id", "doi", "title")
        ):
            raise ScholarlySearchError("OpenAlex returned an invalid work record")
        return payload

    def _request_json(
        self,
        provider: str,
        url: str,
        not_found_is_empty: bool = False,
    ) -> Optional[Dict[str, Any]]:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self.USER_AGENT,
            },
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = response.read(self.MAX_RESPONSE_BYTES + 1)
                if len(body) > self.MAX_RESPONSE_BYTES:
                    raise ScholarlySearchError(
                        f"{provider} response exceeded the size limit"
                    )
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ScholarlySearchError(
                        f"{provider} returned an invalid JSON object"
                    )
                return payload
            except HTTPError as error:
                if error.code == 404 and not_found_is_empty:
                    return None
                retryable = error.code == 429 or 500 <= error.code < 600
                if not retryable or attempt == self.max_retries:
                    raise ScholarlySearchError(
                        f"{provider} verification failed with HTTP {error.code}"
                    ) from error
                retry_after = (
                    error.headers.get("Retry-After") if error.headers else None
                )
                delay = self._retry_after(retry_after)
            except (URLError, TimeoutError, OSError) as error:
                if attempt == self.max_retries:
                    raise ScholarlySearchError(
                        f"Could not reach {provider}: {error}"
                    ) from error
                delay = min(0.25 * (2 ** attempt), 2.0)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ScholarlySearchError(
                    f"{provider} returned invalid JSON"
                ) from error
            time.sleep(delay)
        raise ScholarlySearchError(f"{provider} verification exhausted retries")

    @staticmethod
    def _retry_after(value: Optional[str]) -> float:
        if value:
            try:
                return min(max(float(value), 0.0), 2.0)
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(value)
                    return min(max(retry_at.timestamp() - time.time(), 0.0), 2.0)
                except (TypeError, ValueError, OverflowError):
                    pass
        return 0.5

    @classmethod
    def _normalize_crossref(
        cls, work: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not work:
            return None
        titles = work.get("title")
        title = (
            cls._clean_text(titles[0])
            if isinstance(titles, list) and titles
            else None
        )
        raw_authors = work.get("author")
        authors = []
        if isinstance(raw_authors, list):
            for author in raw_authors:
                if not isinstance(author, dict):
                    continue
                name = " ".join(
                    part for part in (
                        cls._clean_text(author.get("given")),
                        cls._clean_text(author.get("family")),
                    ) if part
                )
                if name:
                    authors.append(name)
        venue = work.get("container-title")
        return {
            "source": "Crossref",
            "doi": cls._clean_doi(work.get("DOI")) or None,
            "title": title,
            "authors": authors,
            "publication_year": cls._crossref_year(work),
            "venue": (
                cls._clean_text(venue[0])
                if isinstance(venue, list) and venue
                else None
            ),
            "type": cls._clean_text(work.get("type")) or None,
            "registry_url": (
                f"https://doi.org/{cls._clean_doi(work.get('DOI'))}"
                if cls._clean_doi(work.get("DOI"))
                else None
            ),
        }

    @classmethod
    def _normalize_openalex(
        cls, work: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not work:
            return None
        oa = work.get("open_access")
        oa = oa if isinstance(oa, dict) else {}
        location = work.get("best_oa_location")
        location = location if isinstance(location, dict) else {}
        oa_url = cls._safe_https_url(oa.get("oa_url"))
        return {
            "source": "OpenAlex",
            "id": cls._clean_text(work.get("id")) or None,
            "doi": cls._clean_doi(work.get("doi")) or None,
            "title": cls._clean_text(work.get("title")) or None,
            "publication_year": (
                work.get("publication_year")
                if isinstance(work.get("publication_year"), int)
                else None
            ),
            "open_access": oa.get("is_oa") if isinstance(oa.get("is_oa"), bool) else None,
            "oa_status": cls._clean_text(oa.get("oa_status")) or None,
            "landing_page_url": (
                cls._safe_https_url(location.get("landing_page_url")) or oa_url
            ),
            "pdf_url": cls._safe_https_url(location.get("pdf_url")),
            "oa_url": oa_url,
            "license": cls._clean_text(location.get("license")) or None,
            "version": cls._clean_text(location.get("version")) or None,
        }

    @staticmethod
    def _open_access_summary(
        work: Optional[Dict[str, Any]],
        expected_doi: str = "",
    ) -> Dict[str, Any]:
        if not work:
            return {
                "status": "unknown",
                "source": "OpenAlex",
                "pdf_url": None,
                "landing_page_url": None,
                "url_accessibility": "not_checked",
            }
        openalex = PaperMetadataVerifier._normalize_openalex(work) or {}
        if expected_doi and (
            (openalex.get("doi") or "").casefold() != expected_doi.casefold()
        ):
            return {
                "status": "unknown",
                "source": "OpenAlex",
                "pdf_url": None,
                "landing_page_url": None,
                "url_accessibility": "not_checked",
                "reason": "OpenAlex record DOI did not match the requested DOI.",
            }
        is_oa = openalex.get("open_access")
        return {
            "status": (
                "open_access" if is_oa is True
                else "not_listed_open_access" if is_oa is False
                else "unknown"
            ),
            "source": "OpenAlex",
            "oa_status": openalex.get("oa_status"),
            "pdf_url": openalex.get("pdf_url"),
            "landing_page_url": openalex.get("landing_page_url"),
            "oa_url": openalex.get("oa_url"),
            "license": openalex.get("license"),
            "version": openalex.get("version"),
            "url_accessibility": "provider_listed_not_checked",
        }

    @classmethod
    def _compare_metadata(
        cls,
        title: str,
        doi: str,
        publication_year: Any,
        authors: Any,
        registry_record: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if registry_record is None:
            return {
                field: {"provided": value or None, "registry": None, "matches": None}
                for field, value in (
                    ("title", title),
                    ("doi", doi),
                    ("publication_year", publication_year),
                    ("authors", authors),
                )
                if value
            }

        checks: Dict[str, Any] = {}
        registry_title = registry_record.get("title")
        if title:
            similarity = (
                difflib.SequenceMatcher(
                    None,
                    cls._normalize_title(title),
                    cls._normalize_title(registry_title or ""),
                ).ratio()
                if registry_title
                else 0.0
            )
            checks["title"] = {
                "provided": title,
                "registry": registry_title,
                "similarity": round(similarity, 3),
                "matches": similarity >= cls.TITLE_MATCH_THRESHOLD,
            }
        registry_doi = registry_record.get("doi")
        if doi:
            checks["doi"] = {
                "provided": doi,
                "registry": registry_doi,
                "matches": bool(registry_doi)
                and doi.casefold() == registry_doi.casefold(),
            }
        registry_year = registry_record.get("publication_year")
        supplied_year = cls._year_value(publication_year)
        if supplied_year is not None:
            checks["publication_year"] = {
                "provided": supplied_year,
                "registry": registry_year,
                "matches": registry_year == supplied_year
                if registry_year is not None
                else None,
            }
        if authors:
            supplied_authors = cls._author_names(authors)
            registry_authors = cls._author_names(registry_record.get("authors"))
            checks["authors"] = {
                "provided": supplied_authors,
                "registry": registry_authors,
                "matches": (
                    bool(supplied_authors)
                    and bool(registry_authors)
                    and cls._normalize_author(supplied_authors[0])
                    == cls._normalize_author(registry_authors[0])
                )
                if supplied_authors and registry_authors
                else None,
            }
        return checks

    @staticmethod
    def _crossref_year(work: Dict[str, Any]) -> Optional[int]:
        for key in ("published-print", "published-online", "published", "created"):
            date_data = work.get(key)
            if not isinstance(date_data, dict):
                continue
            date_parts = date_data.get("date-parts")
            if (
                isinstance(date_parts, list)
                and date_parts
                and isinstance(date_parts[0], list)
                and date_parts[0]
                and isinstance(date_parts[0][0], int)
            ):
                return date_parts[0][0]
        return None

    @staticmethod
    def _year_value(value: Any) -> Optional[int]:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            match = re.search(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b", value)
            if match:
                return int(match.group(1))
        return None

    @classmethod
    def _author_names(cls, value: Any) -> list:
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple)):
            values = value
        else:
            return []
        names = []
        for item in values:
            if isinstance(item, str):
                name = cls._clean_text(item)
            elif isinstance(item, dict):
                name = " ".join(
                    part for part in (
                        cls._clean_text(item.get("given")),
                        cls._clean_text(item.get("family")),
                    ) if part
                )
            else:
                name = ""
            if name:
                names.append(name)
        return names

    @classmethod
    def _normalize_author(cls, name: str) -> str:
        if "," in name:
            family, given = name.split(",", 1)
            name = f"{given} {family}"
        return " ".join(re.findall(r"[a-z0-9]+", name.casefold()))

    @staticmethod
    def _clean_text(value: Any) -> str:
        return re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else ""

    @staticmethod
    def _normalize_title(title: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", title.casefold()))

    @staticmethod
    def _clean_doi(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        doi = value.strip()
        doi = re.sub(
            r"^https?://(?:dx\.)?doi\.org/",
            "",
            doi,
            flags=re.IGNORECASE,
        )
        return doi if re.fullmatch(r"10\.\d{4,9}/\S+", doi, flags=re.IGNORECASE) else ""

    @staticmethod
    def _safe_https_url(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        url = value.strip()
        return url if url.startswith("https://") and len(url) <= 2048 else None
