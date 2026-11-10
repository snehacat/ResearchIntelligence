"""Tests for guarded retrieval and analysis of related-paper PDFs."""

import io
from unittest.mock import Mock
from urllib.request import Request

import pymupdf
import pytest

from scholarly_search.related_paper_extractor import (
    PaperDownloadError,
    RelatedPaperEvidenceExtractor,
    _SafeRedirectHandler,
)
from paper_analysis_errors import ExtractionError


DOI = "10.1234/related.paper"
PDF_URL = "https://repository.example/related-paper.pdf"


class FakeResponse(io.BytesIO):
    def __init__(self, content, content_type="application/pdf", url=PDF_URL):
        super().__init__(content)
        self.headers = {"Content-Type": content_type}
        self.url = url

    def geturl(self):
        return self.url


class FakeOpener:
    def __init__(self, response):
        self.response = response
        self.request = None
        self.timeout = None

    def open(self, request, timeout):
        self.request = request
        self.timeout = timeout
        return self.response


def create_pdf_bytes(page_texts=("Related paper evidence on page one.",)):
    document = pymupdf.Document()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((50, 72), text)
    content = document.tobytes()
    document.close()
    return content


def verified_open_access():
    return {
        "status": "verified",
        "openalex": {"doi_matches_lookup": True},
        "open_access": {
            "status": "open_access",
            "pdf_url": PDF_URL,
        },
    }


@pytest.fixture
def extractor():
    verifier = Mock()
    verifier.verify.return_value = verified_open_access()
    return RelatedPaperEvidenceExtractor(
        analyzer_factory=Mock(),
        metadata_verifier=verifier,
        max_pdf_bytes=1024 * 1024,
        max_pages=5,
        max_extracted_characters=1000,
    )


class TestRelatedPaperWorkflow:
    def test_verifies_downloads_extracts_and_analyzes(self, extractor, monkeypatch):
        content = create_pdf_bytes()
        analysis = {
            "title": "Related paper",
            "evidence_citations": {
                "methodology": {
                    "citations": [{"start_page": 1, "end_page": 1}]
                }
            },
        }
        analyzer = Mock()
        analyzer.analyze_paper.return_value = analysis
        extractor.analyzer_factory = Mock(return_value=analyzer)
        extractor._download_pdf = Mock(return_value=content)
        paper = {"title": "Related paper", "doi": DOI}

        results = extractor.extract_related_papers([paper])

        assert len(results) == 1
        result = results[0]
        assert result["status"] == "analyzed"
        assert result["paper"] == paper
        assert result["pdf"]["bytes"] == len(content)
        assert result["pdf"]["pages"] == 1
        assert len(result["pdf"]["sha256"]) == 64
        assert result["analysis"] == analysis
        assert result["document_id"].startswith("related_")
        extractor.metadata_verifier.verify.assert_called_once_with(paper)
        extractor._download_pdf.assert_called_once_with(PDF_URL)
        analyzer.analyze_paper.assert_called_once()
        extraction = analyzer.analyze_paper.call_args.args[0]
        document_id = analyzer.analyze_paper.call_args.kwargs["document_id"]
        assert "Related paper evidence" in extraction["pages"][0]["text"]
        assert document_id == result["document_id"]

    @pytest.mark.parametrize(
        ("verification", "reason"),
        [
            ({"status": "mismatch"}, "metadata_verification_mismatch"),
            (
                {
                    "status": "verified",
                    "openalex": {"doi_matches_lookup": True},
                    "open_access": {"status": "not_listed_open_access"},
                },
                "not_listed_open_access",
            ),
            (
                {
                    "status": "verified",
                    "openalex": {"doi_matches_lookup": True},
                    "open_access": {"status": "open_access", "pdf_url": None},
                },
                "no_provider_listed_pdf",
            ),
            (
                {
                    "status": "verified",
                    "openalex": {"doi_matches_lookup": False},
                    "open_access": {
                        "status": "open_access",
                        "pdf_url": PDF_URL,
                    },
                },
                "openalex_doi_not_verified",
            ),
        ],
    )
    def test_skips_unverified_or_non_open_access_papers(
        self, extractor, verification, reason
    ):
        extractor.metadata_verifier.verify.return_value = verification
        extractor._download_pdf = Mock()

        result = extractor.extract_related_papers(
            [{"title": "Candidate", "doi": DOI}]
        )[0]

        assert result["status"] == "skipped"
        assert result["reason"] == reason
        extractor._download_pdf.assert_not_called()

    def test_reports_duplicate_and_limit_statuses(self, extractor):
        papers = [
            {"title": "First", "doi": DOI},
            {"title": "Duplicate", "doi": f"https://doi.org/{DOI}"},
            {"title": "Beyond limit", "doi": "10.1234/another"},
        ]

        results = extractor.extract_related_papers(papers, max_papers=1)

        assert [result["status"] for result in results] == [
            "download_failed",
            "skipped",
            "not_processed",
        ]
        assert results[1]["reason"] == "duplicate_doi"
        assert results[2]["reason"] == "per_request_limit"

    def test_returns_explicit_download_and_analysis_failures(self, extractor):
        paper = {"title": "Candidate", "doi": DOI}
        extractor._download_pdf = Mock(
            side_effect=PaperDownloadError("unsafe host")
        )
        result = extractor.extract_related_papers([paper])[0]
        assert result["status"] == "download_failed"
        assert result["reason"] == "unsafe host"

        extractor._download_pdf = Mock(return_value=create_pdf_bytes())
        analyzer = Mock()
        analyzer.analyze_paper.side_effect = RuntimeError("unexpected failure")
        extractor.analyzer_factory = Mock(return_value=analyzer)
        with pytest.raises(RuntimeError, match="unexpected failure"):
            extractor.extract_related_papers([paper])

    def test_isolates_known_analysis_errors_per_candidate(self, extractor):
        paper = {"title": "Candidate", "doi": DOI}
        extractor._download_pdf = Mock(return_value=create_pdf_bytes())
        analyzer = Mock()
        analyzer.analyze_paper.side_effect = ExtractionError("LLM extraction failed")
        extractor.analyzer_factory = Mock(return_value=analyzer)

        result = extractor.extract_related_papers([paper])[0]

        assert result["status"] == "analysis_failed"
        assert result["reason"] == "LLM extraction failed"

    def test_reports_pdf_page_limit_failure(self, extractor):
        paper = {"title": "Candidate", "doi": DOI}
        extractor.max_pages = 1
        extractor._download_pdf = Mock(
            return_value=create_pdf_bytes(("Page one.", "Page two."))
        )

        result = extractor.extract_related_papers([paper])[0]

        assert result["status"] == "extraction_failed"
        assert "pages; limit is 1" in result["reason"]

    @pytest.mark.parametrize(
        ("papers", "max_papers"),
        [
            ("not a list", 1),
            ([{"title": "Candidate"}], 0),
            ([{"title": "Candidate"}], True),
            ([None], 1),
        ],
    )
    def test_validates_batch_inputs(self, extractor, papers, max_papers):
        with pytest.raises(PaperDownloadError):
            extractor.extract_related_papers(papers, max_papers=max_papers)


class TestSafePDFDownload:
    def _allow_public_dns(self, monkeypatch):
        monkeypatch.setattr(
            "scholarly_search.related_paper_extractor.socket.getaddrinfo",
            lambda *args, **kwargs: [
                (2, 1, 6, "", ("93.184.216.34", 443))
            ],
        )

    def test_downloads_only_pdf_and_respects_request_timeout(
        self, extractor, monkeypatch
    ):
        content = create_pdf_bytes()
        response = FakeResponse(content)
        opener = FakeOpener(response)
        self._allow_public_dns(monkeypatch)
        monkeypatch.setattr(
            "scholarly_search.related_paper_extractor.build_opener",
            lambda handler: opener,
        )

        downloaded = extractor._download_pdf(PDF_URL)

        assert downloaded == content
        assert opener.timeout == extractor.timeout
        assert opener.request.headers["Accept"].startswith("application/pdf")

    @pytest.mark.parametrize(
        ("url", "addresses", "message"),
        [
            ("http://repository.example/paper.pdf", None, "HTTPS"),
            (
                "https://repository.example/paper.pdf",
                [("127.0.0.1", 443)],
                "non-public",
            ),
            (
                "https://user:password@repository.example/paper.pdf",
                None,
                "HTTPS",
            ),
        ],
    )
    def test_rejects_non_https_or_private_hosts(
        self, extractor, monkeypatch, url, addresses, message
    ):
        if addresses is not None:
            monkeypatch.setattr(
                "scholarly_search.related_paper_extractor.socket.getaddrinfo",
                lambda *args, **kwargs: [
                    (2, 1, 6, "", (addresses[0][0], addresses[0][1]))
                ],
            )
        else:
            self._allow_public_dns(monkeypatch)

        with pytest.raises(PaperDownloadError, match=message):
            extractor._download_pdf(url)

    def test_url_validation_returns_hostname_and_addresses(self):
        """A11: URL validation returns hostname and validated addresses for connection pinning."""
        from unittest.mock import patch

        with patch("scholarly_search.related_paper_extractor.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("93.184.216.34", 443)),
                (2, 1, 6, "", ("93.184.216.34", 443)),
            ]

            hostname, addresses = RelatedPaperEvidenceExtractor._validate_public_https_url(
                "https://repository.example/paper.pdf"
            )

            assert hostname == "repository.example"
            assert addresses == {"93.184.216.34"}

    def test_redirect_handler_rejects_insecure_redirects(self):
        handler = _SafeRedirectHandler(max_redirects=3, validated_addresses=set())
        with pytest.raises(PaperDownloadError, match="HTTPS"):
            handler.redirect_request(
                request=Request(PDF_URL),
                file_pointer=None,
                code=302,
                message="Found",
                headers={},
                new_url="http://internal.example/paper.pdf",
            )

    def test_redirect_handler_rejects_redirect_to_unvalidated_address(self):
        """A11: Redirect to address not in original validated set is rejected."""
        # Simulate original validation returned 8.8.8.8
        validated_addresses = {"8.8.8.8"}
        handler = _SafeRedirectHandler(max_redirects=3, validated_addresses=validated_addresses)
        
        # Mock DNS to return different public address for redirect target
        from unittest.mock import patch
        with patch("scholarly_search.related_paper_extractor.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("93.184.216.34", 443))  # Different public address
            ]
            
            with pytest.raises(PaperDownloadError, match="not in the originally validated set"):
                handler.redirect_request(
                    request=Request(PDF_URL),
                    file_pointer=None,
                    code=302,
                    message="Found",
                    headers={},
                    new_url="https://different.example/paper.pdf",
                )

    @pytest.mark.parametrize(
        ("content", "content_type", "max_bytes", "message"),
        [
            (b"<html>not a pdf</html>", "text/html", 1024, "content type"),
            (b"not a pdf", "application/pdf", 1024, "not a PDF"),
        ],
    )
    def test_rejects_html_or_invalid_pdf_response(
        self, extractor, monkeypatch, content, content_type, max_bytes, message
    ):
        self._allow_public_dns(monkeypatch)
        opener = FakeOpener(FakeResponse(content, content_type))
        monkeypatch.setattr(
            "scholarly_search.related_paper_extractor.build_opener",
            lambda handler: opener,
        )

        with pytest.raises(PaperDownloadError, match=message):
            extractor._download_pdf(PDF_URL)

    def test_rejects_pdf_over_configured_byte_limit(self, extractor, monkeypatch):
        content = create_pdf_bytes()
        extractor.max_pdf_bytes = len(content) - 1
        self._allow_public_dns(monkeypatch)
        opener = FakeOpener(FakeResponse(content))
        monkeypatch.setattr(
            "scholarly_search.related_paper_extractor.build_opener",
            lambda handler: opener,
        )

        with pytest.raises(PaperDownloadError, match="size limit"):
            extractor._download_pdf(PDF_URL)
