"""Search OpenAlex for papers related to an analyzed research paper."""

import difflib
import json
import re
import time
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from scholarly_search.crossref_client import ScholarlySearchError


class OpenAlexSearchClient:
    """Find related scholarly works using OpenAlex's public works API."""

    API_URL = "https://api.openalex.org/works"
    MAX_RESULTS = 50
    MAX_RETRIES = 2
    REQUEST_TIMEOUT = 10.0
    USER_AGENT = "ResearchIntelligence/0.1 (scholarly paper discovery)"
    TOPIC_STOPWORDS = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "into", "is", "it", "of", "on", "or", "the", "to", "with",
        "study", "research", "paper", "analysis", "using", "based",
        "approach", "method", "methods", "result", "results",
        "related", "work", "find", "use", "new", "include", "includes",
        "based",
    })

    def __init__(
        self,
        timeout: float = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        api_key: Optional[str] = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0 or max_retries > 5:
            raise ValueError("max_retries must be between 0 and 5")
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key

    def search_related_papers(
        self,
        paper_analysis: Mapping[str, Any],
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search by title and research content, then deduplicate and rank results.

        Only bounded metadata-derived search terms are sent to OpenAlex; the PDF
        text and full abstract are never transmitted.
        """
        if not isinstance(paper_analysis, Mapping):
            raise ScholarlySearchError("paper_analysis must be a mapping")
        if not isinstance(max_results, int) or isinstance(max_results, bool):
            raise ScholarlySearchError("max_results must be an integer")
        if not 1 <= max_results <= self.MAX_RESULTS:
            raise ScholarlySearchError(
                f"max_results must be between 1 and {self.MAX_RESULTS}"
            )

        queries = self._build_queries(paper_analysis)
        if not queries:
            raise ScholarlySearchError(
                "Paper analysis must include a title or research-content fields"
            )

        original_title = self._clean_text(paper_analysis.get("title"))
        original_doi = self._clean_doi(paper_analysis.get("doi"))
        candidates: Dict[str, Dict[str, Any]] = {}
        rows = min(max_results * 3, 100)

        for query in queries:
            message = self._fetch_works(query, rows)
            results = message.get("results", [])
            if not isinstance(results, list):
                raise ScholarlySearchError(
                    "OpenAlex response is missing a valid results list"
                )
            for item in results:
                if not isinstance(item, dict):
                    continue
                result = self._normalize_work(item, query)
                if result is None or self._is_uploaded_paper(
                    result, original_title, original_doi
                ):
                    continue
                result["topic_overlap_score"] = self._topic_overlap_score(
                    paper_analysis, result
                )
                result["topic_overlap_terms"] = self._topic_overlap_terms(
                    paper_analysis, result
                )

                dedupe_key = (result.get("doi") or "").lower()
                if not dedupe_key:
                    dedupe_key = self._normalize_title(result["title"])
                existing = candidates.get(dedupe_key)
                if existing is None or result["relevance_score"] > existing["relevance_score"]:
                    candidates[dedupe_key] = result

        ranked = sorted(
            candidates.values(),
            key=lambda result: (
                result["topic_overlap_score"],
                result["relevance_score"],
            ),
            reverse=True,
        )
        return self._deduplicate_title_author(ranked)[:max_results]

    @classmethod
    def _build_queries(cls, analysis: Mapping[str, Any]) -> List[str]:
        title = cls._clean_text(analysis.get("title"))
        topical_fields = (
            ("research_problem", 180),
            ("methodology", 140),
            ("dataset", 80),
            ("evaluation_metrics", 60),
        )
        topical_terms: List[str] = []
        for field, limit in topical_fields:
            value = analysis.get(field)
            if isinstance(value, (list, tuple)):
                text = " ".join(
                    cleaned for item in value
                    if (cleaned := cls._clean_text(item))
                )
            else:
                text = cls._clean_text(value)
            if text:
                topical_terms.append(text[:limit])

        queries = []
        if title:
            queries.append(title[:240])
        if topical_terms:
            topical_query = " ".join(topical_terms)[:400]
            if topical_query.casefold() not in {
                query.casefold() for query in queries
            }:
                queries.append(topical_query)
        return queries

    def _fetch_works(self, query: str, rows: int) -> Dict[str, Any]:
        params = urlencode({"search": query, "per_page": rows})
        headers = {
            "Accept": "application/json",
            "User-Agent": self.USER_AGENT,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        request = Request(f"{self.API_URL}?{params}", headers=headers)
        
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ScholarlySearchError(
                        "OpenAlex returned an invalid response structure"
                    )
                return payload
            except HTTPError as error:
                retryable = error.code == 429 or 500 <= error.code < 600
                if not retryable or attempt == self.max_retries:
                    raise ScholarlySearchError(
                        f"OpenAlex search failed with HTTP {error.code}"
                    ) from error
                retry_after = (
                    error.headers.get("Retry-After") if error.headers else None
                )
                delay = self._retry_after(retry_after)
            except (URLError, TimeoutError, OSError) as error:
                if attempt == self.max_retries:
                    raise ScholarlySearchError(
                        f"Could not reach OpenAlex: {error}"
                    ) from error
                delay = min(0.25 * (2 ** attempt), 2.0)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ScholarlySearchError(
                    "OpenAlex returned invalid JSON"
                ) from error

            time.sleep(delay)

        raise ScholarlySearchError("OpenAlex search exhausted its retry limit")

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
    def _normalize_work(
        cls, item: Dict[str, Any], matched_query: str
    ) -> Optional[Dict[str, Any]]:
        title = cls._clean_text(item.get("title"))
        if not title:
            return None

        authors: List[str] = []
        raw_authors = item.get("authorships", [])
        if isinstance(raw_authors, list):
            for authorship in raw_authors:
                if not isinstance(authorship, dict):
                    continue
                author = authorship.get("author")
                if not isinstance(author, dict):
                    continue
                name = " ".join(
                    part for part in (
                        cls._clean_text(author.get("given_name")),
                        cls._clean_text(author.get("family_name")),
                    ) if part
                )
                if name:
                    authors.append(name)

        doi = cls._clean_doi(item.get("doi"))
        
        # Get open access location for PDF URL
        location = item.get("best_oa_location", {})
        location = location if isinstance(location, dict) else {}
        pdf_url = cls._safe_https_url(location.get("pdf_url"))
        if not pdf_url:
            pdf_url = cls._safe_https_url(location.get("landing_page_url"))
        
        return {
            "title": title,
            "authors": authors,
            "year": item.get("publication_year") if isinstance(item.get("publication_year"), int) else None,
            "venue": cls._clean_text(
                ((item.get("primary_location") or {}).get("source") or {}).get("display_name")
            ) or None,
            "doi": doi,
            "url": pdf_url or None,
            "abstract": cls._clean_text(item.get("abstract")) or None,
            "relevance_score": cls._relevance_score(item.get("relevance_score")),
            "topic_overlap_score": 0.0,
            "topic_overlap_terms": [],
            "matched_query": matched_query,
            "source": "OpenAlex",
        }

    @classmethod
    def _topic_overlap_terms(
        cls,
        paper_analysis: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> List[str]:
        source_text = " ".join(
            cls._field_text(paper_analysis.get(field))
            for field in (
                "title",
                "research_problem",
                "methodology",
                "dataset",
                "evaluation_metrics",
            )
        )
        candidate_text = " ".join(
            cls._clean_text(candidate.get(field))
            for field in ("title", "abstract")
        )
        source_tokens = set(cls._topic_tokens(source_text))
        candidate_tokens = set(cls._topic_tokens(candidate_text))
        return sorted(source_tokens.intersection(candidate_tokens))

    @classmethod
    def _topic_overlap_score(
        cls,
        paper_analysis: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> float:
        source_text = " ".join(
            cls._field_text(paper_analysis.get(field))
            for field in (
                "title",
                "research_problem",
                "methodology",
                "dataset",
                "evaluation_metrics",
            )
        )
        title_tokens = cls._topic_tokens(cls._clean_text(candidate.get("title")))
        if not title_tokens:
            return 0.0
        source_tokens = set(cls._topic_tokens(source_text))
        return len(set(title_tokens).intersection(source_tokens)) / len(
            set(title_tokens)
        )

    @classmethod
    def _field_text(cls, value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return " ".join(
                cleaned for item in value
                if (cleaned := cls._clean_text(item))
            )
        return cls._clean_text(value)

    @classmethod
    def _topic_tokens(cls, value: str) -> List[str]:
        tokens = []
        for token in re.findall(r"[a-z0-9]+", value.casefold()):
            if token in cls.TOPIC_STOPWORDS:
                continue
            if len(token) > 4 and token.endswith("ies"):
                token = token[:-3] + "y"
            elif len(token) > 3 and token.endswith("s"):
                token = token[:-1]
            if token and token not in cls.TOPIC_STOPWORDS:
                tokens.append(token)
        return tokens

    @classmethod
    def _deduplicate_title_author(
        cls,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        unique: Dict[tuple, Dict[str, Any]] = {}
        for result in results:
            title = cls._normalize_title(result["title"])
            authors = result.get("authors") or []
            first_author = (
                cls._normalize_title(authors[0]) if authors else ""
            )
            key = (title, first_author)
            existing = unique.get(key)
            if existing is None:
                result["alternate_dois"] = []
                unique[key] = result
                continue

            doi = result.get("doi")
            if doi and doi != existing.get("doi") and doi not in existing[
                "alternate_dois"
            ]:
                existing["alternate_dois"].append(doi)
            if result.get("year") and result["year"] != existing.get("year"):
                alternate_years = existing.setdefault("alternate_years", [])
                if result["year"] not in alternate_years:
                    alternate_years.append(result["year"])
            if cls._metadata_completeness(result) > cls._metadata_completeness(
                existing
            ):
                alternate_dois = [
                    doi
                    for doi in existing["alternate_dois"]
                    if doi != result.get("doi")
                ]
                existing_doi = existing.get("doi")
                if (
                    existing_doi
                    and existing_doi != result.get("doi")
                    and existing_doi not in alternate_dois
                ):
                    alternate_dois.append(existing_doi)
                alternate_years = existing.get("alternate_years", [])
                result["alternate_dois"] = alternate_dois
                if alternate_years:
                    result["alternate_years"] = alternate_years
                unique[key] = result
        return list(unique.values())

    @staticmethod
    def _metadata_completeness(result: Mapping[str, Any]) -> int:
        return sum(
            bool(result.get(field))
            for field in ("doi", "authors", "year", "venue", "abstract")
        )

    @staticmethod
    def _relevance_score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return score if score >= 0 else 0.0

    @staticmethod
    def _clean_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _clean_doi(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        doi = value.strip()
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
        return doi if re.match(r"^10\.\d{4,9}/\S+$", doi, flags=re.IGNORECASE) else ""

    @staticmethod
    def _normalize_title(title: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", title.casefold()))

    @staticmethod
    def _safe_https_url(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        url = value.strip()
        if url.startswith("https://"):
            return url
        return None

    @classmethod
    def _is_uploaded_paper(
        cls, result: Dict[str, Any], original_title: str, original_doi: str
    ) -> bool:
        if original_doi and result.get("doi", "").casefold() == original_doi.casefold():
            return True
        if not original_title:
            return False
        normalized_original = cls._normalize_title(original_title)
        normalized_result = cls._normalize_title(result["title"])
        return (
            normalized_original == normalized_result
            or difflib.SequenceMatcher(
                None, normalized_original, normalized_result
            ).ratio() >= 0.96
        )
