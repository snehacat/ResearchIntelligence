"""Safely retrieve and analyze provider-listed open-access related papers."""

import hashlib
import http.client
import ipaddress
import socket
import time
from datetime import datetime, timezone
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)

import pymupdf

from embeddings.embedding_generator import EmbeddingError
from chunking.text_chunker import ChunkingError
from pdf_processing.text_extractor import PDFExtractionError, TextExtractor
from paper_analysis_errors import ExtractionError, PaperAnalysisError
from qa.answer_generator import AnswerGenerationError
from retrieval.semantic_retriever import RetrievalError
from scholarly_search.crossref_client import ScholarlySearchError
from scholarly_search.source_verifier import PaperMetadataVerifier


class PaperDownloadError(Exception):
    """Raised when a provider-listed PDF cannot be downloaded safely."""


class _PaperAnalyzer(Protocol):
    def analyze_paper(
        self, extraction_result: Dict[str, Any], document_id: str
    ) -> Dict[str, Any]: ...


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self, host: str, validated_addresses: set, **kwargs: Any
    ) -> None:
        self.validated_addresses = tuple(sorted(validated_addresses))
        if not self.validated_addresses:
            raise PaperDownloadError("PDF host has no validated addresses")
        super().__init__(host, **kwargs)

    def connect(self) -> None:
        errors = []
        for address in self.validated_addresses:
            raw_socket = None
            try:
                raw_socket = socket.create_connection(
                    (address, self.port), self.timeout, self.source_address
                )
                self.sock = self._context.wrap_socket(
                    raw_socket, server_hostname=self.host
                )
                return
            except OSError as error:
                errors.append(error)
                if raw_socket is not None:
                    raw_socket.close()
        raise OSError("Could not connect to any validated PDF host address") from errors[-1]


class _PinnedHTTPSHandler(HTTPSHandler):
    def __init__(self, addresses_by_host: Dict[str, set]) -> None:
        super().__init__()
        self.addresses_by_host = addresses_by_host

    def https_open(self, request: Request):
        hostname = urlsplit(request.full_url).hostname
        addresses = self.addresses_by_host.get(hostname.lower()) if hostname else None
        if not addresses:
            raise PaperDownloadError("PDF host was not validated before connection")
        connection_class = partial(
            _PinnedHTTPSConnection, validated_addresses=addresses
        )
        return self.do_open(
            connection_class,
            request,
            context=self._context,
            check_hostname=self._check_hostname,
        )


class _SafeRedirectHandler(HTTPRedirectHandler):
    def __init__(
        self, max_redirects: int, addresses_by_host: Dict[str, set]
    ) -> None:
        super().__init__()
        self.max_redirects = max_redirects
        self.redirect_count = 0
        self.addresses_by_host = addresses_by_host

    def redirect_request(
        self,
        request: Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> Optional[Request]:
        self.redirect_count += 1
        if self.redirect_count > self.max_redirects:
            raise PaperDownloadError("PDF download exceeded the redirect limit")
        hostname, redirect_addresses = (
            RelatedPaperEvidenceExtractor._validate_public_https_url(new_url)
        )
        self.addresses_by_host[hostname.lower()] = redirect_addresses
        return super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )


class RelatedPaperEvidenceExtractor:
    """Verify, download, extract, and analyze a bounded number of OA PDFs."""

    MAX_PAPERS = 10
    MAX_CANDIDATES = 50
    MAX_PDF_BYTES = 25 * 1024 * 1024
    MAX_PAGES = 300
    MAX_EXTRACTED_CHARACTERS = 2_000_000
    MAX_REDIRECTS = 3
    REQUEST_TIMEOUT = 20.0
    ACCEPTED_CONTENT_TYPES = {
        "",
        "application/pdf",
        "application/octet-stream",
        "application/x-pdf",
        "binary/octet-stream",
    }

    def __init__(
        self,
        analyzer_factory: Callable[[], _PaperAnalyzer],
        metadata_verifier: Optional[PaperMetadataVerifier] = None,
        text_extractor: Optional[TextExtractor] = None,
        timeout: float = REQUEST_TIMEOUT,
        max_pdf_bytes: int = MAX_PDF_BYTES,
        max_pages: int = MAX_PAGES,
        max_extracted_characters: int = MAX_EXTRACTED_CHARACTERS,
    ) -> None:
        if not callable(analyzer_factory):
            raise ValueError("analyzer_factory must be callable")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            raise ValueError("timeout must be a positive number")
        if not 0 < timeout <= 120:
            raise ValueError("timeout must be greater than 0 and at most 120 seconds")
        for value, name, maximum in (
            (max_pdf_bytes, "max_pdf_bytes", self.MAX_PDF_BYTES),
            (max_pages, "max_pages", self.MAX_PAGES),
            (
                max_extracted_characters,
                "max_extracted_characters",
                self.MAX_EXTRACTED_CHARACTERS,
            ),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= maximum
            ):
                raise ValueError(f"{name} must be between 1 and {maximum}")
        self.analyzer_factory = analyzer_factory
        self.metadata_verifier = metadata_verifier or PaperMetadataVerifier(
            timeout=timeout
        )
        self.text_extractor = text_extractor or TextExtractor()
        self.timeout = timeout
        self.max_pdf_bytes = max_pdf_bytes
        self.max_pages = max_pages
        self.max_extracted_characters = max_extracted_characters

    def extract_related_papers(
        self,
        papers: List[Dict[str, Any]],
        max_papers: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return one explicit status record for every candidate paper."""
        if not isinstance(papers, list):
            raise PaperDownloadError("papers must be a list")
        if len(papers) > self.MAX_CANDIDATES:
            raise PaperDownloadError(
                f"papers cannot contain more than {self.MAX_CANDIDATES} candidates"
            )
        if (
            not isinstance(max_papers, int)
            or isinstance(max_papers, bool)
            or not 1 <= max_papers <= self.MAX_PAPERS
        ):
            raise PaperDownloadError(
                f"max_papers must be between 1 and {self.MAX_PAPERS}"
            )
        if any(not isinstance(paper, Mapping) for paper in papers):
            raise PaperDownloadError("each paper candidate must be a mapping")

        results = []
        seen_dois = set()
        for index, paper in enumerate(papers):
            doi = PaperMetadataVerifier._clean_doi(paper.get("doi"))
            if doi and doi.casefold() in seen_dois:
                results.append(self._status_record(
                    paper, "skipped", "duplicate_doi"
                ))
                continue
            if doi:
                seen_dois.add(doi.casefold())

            if index >= max_papers:
                results.append(self._status_record(
                    paper, "not_processed", "per_request_limit"
                ))
                continue
            results.append(self._process_paper(paper, doi))
        return results

    def _process_paper(
        self, paper: Mapping[str, Any], doi: str
    ) -> Dict[str, Any]:
        try:
            verification = self.metadata_verifier.verify(paper)
        except ScholarlySearchError as error:
            return self._status_record(
                paper, "verification_failed", str(error)
            )

        open_access = verification.get("open_access", {})
        openalex = verification.get("openalex") or {}
        if verification.get("status") in {"mismatch", "not_found", "unavailable"}:
            return self._status_record(
                paper,
                "skipped",
                f"metadata_verification_{verification['status']}",
                verification=verification,
            )
        if openalex.get("doi_matches_lookup") is not True:
            return self._status_record(
                paper,
                "skipped",
                "openalex_doi_not_verified",
                verification=verification,
            )
        if open_access.get("status") != "open_access":
            return self._status_record(
                paper,
                "skipped",
                "not_listed_open_access",
                verification=verification,
            )
        pdf_url = open_access.get("pdf_url")
        if not pdf_url:
            return self._status_record(
                paper,
                "skipped",
                "no_provider_listed_pdf",
                verification=verification,
            )

        try:
            pdf_bytes = self._download_pdf(pdf_url)
        except PaperDownloadError as error:
            return self._status_record(
                paper,
                "download_failed",
                str(error),
                verification=verification,
            )

        try:
            extraction = self.text_extractor.extract_from_bytes(
                pdf_bytes,
                max_pages=self.max_pages,
                max_characters=self.max_extracted_characters,
            )
        except PDFExtractionError as error:
            return self._status_record(
                paper,
                "extraction_failed",
                str(error),
                verification=verification,
            )

        document_id = self._document_id(doi or paper.get("title", "paper"))
        try:
            analysis = self.analyzer_factory().analyze_paper(
                extraction,
                document_id=document_id,
            )
        except (
            PaperDownloadError,
            PDFExtractionError,
            EmbeddingError,
            RetrievalError,
            AnswerGenerationError,
            ScholarlySearchError,
            PaperAnalysisError,
            ExtractionError,
            ChunkingError,
        ) as error:
            return self._status_record(
                paper,
                "analysis_failed",
                str(error),
                verification=verification,
                document_id=document_id,
            )
        if not isinstance(analysis, dict):
            return self._status_record(
                paper,
                "analysis_failed",
                "paper analyzer returned an invalid result",
                verification=verification,
                document_id=document_id,
            )

        return {
            "status": "analyzed",
            "paper": dict(paper),
            "verification": verification,
            "document_id": document_id,
            "pdf": {
                "source_url": pdf_url,
                "bytes": len(pdf_bytes),
                "sha256": hashlib.sha256(pdf_bytes).hexdigest(),
                "pages": extraction["total_pages"],
                "characters": extraction["total_characters"],
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            },
            "analysis": analysis,
        }

    def _download_pdf(self, url: str) -> bytes:
        hostname, validated_addresses = self._validate_public_https_url(url)
        addresses_by_host = {hostname.lower(): validated_addresses}
        
        request = Request(
            url,
            headers={
                "Accept": "application/pdf,application/octet-stream;q=0.9",
                "User-Agent": "ResearchIntelligence/0.1 (open-access PDF retrieval)",
            },
        )
        opener = build_opener(
            ProxyHandler({}),
            _SafeRedirectHandler(self.MAX_REDIRECTS, addresses_by_host),
            _PinnedHTTPSHandler(addresses_by_host),
        )
        started_at = time.perf_counter()
        try:
            with opener.open(request, timeout=self.timeout) as response:
                final_url = response.geturl()
                final_hostname = urlsplit(final_url).hostname
                if (
                    not final_hostname
                    or final_hostname.lower() not in addresses_by_host
                ):
                    raise PaperDownloadError(
                        "Final PDF host was not validated before connection"
                    )
                content_type = response.headers.get(
                    "Content-Type", ""
                ).split(";", 1)[0].strip().lower()
                if content_type not in self.ACCEPTED_CONTENT_TYPES:
                    raise PaperDownloadError(
                        f"Unexpected PDF content type: {content_type or 'unknown'}"
                    )
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        if int(content_length) > self.max_pdf_bytes:
                            raise PaperDownloadError(
                                "PDF exceeds the configured download size limit"
                            )
                    except ValueError as error:
                        raise PaperDownloadError(
                            "PDF response has an invalid Content-Length"
                        ) from error
                pdf_bytes = response.read(self.max_pdf_bytes + 1)
        except PaperDownloadError:
            raise
        except HTTPError as error:
            raise PaperDownloadError(
                f"PDF provider returned HTTP {error.code}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise PaperDownloadError(f"PDF download failed: {error}") from error
        finally:
            self._last_download_time_seconds = time.perf_counter() - started_at

        if len(pdf_bytes) > self.max_pdf_bytes:
            raise PaperDownloadError(
                "PDF exceeds the configured download size limit"
            )
        if b"%PDF-" not in pdf_bytes[:1024]:
            raise PaperDownloadError("Provider response is not a PDF file")
        self._validate_pdf_limits(pdf_bytes)
        return pdf_bytes

    def _validate_pdf_limits(self, pdf_bytes: bytes) -> None:
        document = None
        try:
            document = pymupdf.Document(stream=pdf_bytes, filetype="pdf")
        except Exception as error:
            raise PaperDownloadError("Downloaded file is not a readable PDF") from error
        try:
            if document.needs_pass:
                raise PaperDownloadError("Password-protected PDFs are not supported")
            if document.page_count > self.max_pages:
                raise PaperDownloadError(
                    f"PDF has {document.page_count} pages; limit is {self.max_pages}"
                )
        finally:
            document.close()

    @staticmethod
    def _validate_public_https_url(url: str) -> Tuple[str, set]:
        """
        Validate that a URL is public HTTPS and return hostname with validated addresses.

        Args:
            url: URL to validate

        Returns:
            Tuple of (hostname, set of validated IP addresses)

        Raises:
            PaperDownloadError: If URL is invalid or resolves to private address
        """
        if not isinstance(url, str) or len(url) > 2048:
            raise PaperDownloadError("PDF URL is invalid or too long")
        try:
            parsed = urlsplit(url)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError as error:
            raise PaperDownloadError("PDF URL is malformed") from error
        if (
            parsed.scheme.lower() != "https"
            or not hostname
            or parsed.username
            or parsed.password
            or (port is not None and port != 443)
        ):
            raise PaperDownloadError("Only public HTTPS PDF URLs are allowed")

        try:
            addresses = {
                record[4][0].split("%", 1)[0]
                for record in socket.getaddrinfo(
                    hostname,
                    port or 443,
                    type=socket.SOCK_STREAM,
                )
            }
        except OSError as error:
            raise PaperDownloadError(
                f"Could not validate PDF host {hostname}: {error}"
            ) from error
        if not addresses:
            raise PaperDownloadError(f"PDF host {hostname} has no address records")
        for address in addresses:
            try:
                is_global = ipaddress.ip_address(address).is_global
            except ValueError as error:
                raise PaperDownloadError(
                    f"PDF host {hostname} resolved to an invalid address"
                ) from error
            if not is_global:
                raise PaperDownloadError(
                    "PDF host resolved to a non-public network address"
                )
        return hostname, addresses

    @staticmethod
    def _document_id(identifier: Any) -> str:
        digest = hashlib.sha256(str(identifier).encode("utf-8")).hexdigest()
        return f"related_{digest[:20]}"

    @staticmethod
    def _status_record(
        paper: Mapping[str, Any],
        status: str,
        reason: str,
        verification: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        record = {
            "status": status,
            "paper": dict(paper),
            "reason": reason,
        }
        if verification is not None:
            record["verification"] = verification
        if document_id is not None:
            record["document_id"] = document_id
        return record
