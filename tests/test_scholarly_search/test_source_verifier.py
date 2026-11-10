"""Tests for scholarly-registry metadata verification."""

import io
import json
from urllib.error import HTTPError

import pytest

from scholarly_search.crossref_client import ScholarlySearchError
from scholarly_search.source_verifier import PaperMetadataVerifier


DOI = "10.1234/example.paper"
TITLE = "A Robust Method for Research"


def json_response(value):
    return io.BytesIO(json.dumps(value).encode("utf-8"))


def crossref_work(**overrides):
    work = {
        "DOI": DOI,
        "title": [TITLE],
        "author": [{"given": "Jane", "family": "Doe"}],
        "published-print": {"date-parts": [[2024, 3]]},
        "container-title": ["Example Journal"],
        "type": "journal-article",
    }
    work.update(overrides)
    return work


def openalex_work(**overrides):
    work = {
        "id": "https://openalex.org/W123",
        "doi": f"https://doi.org/{DOI}",
        "title": TITLE,
        "publication_year": 2024,
        "open_access": {"is_oa": True, "oa_status": "gold"},
        "best_oa_location": {
            "landing_page_url": "https://publisher.example/article",
            "pdf_url": "https://repository.example/article.pdf",
        },
    }
    work.update(overrides)
    return work


class TestPaperMetadataVerifier:
    def test_verifies_doi_title_year_author_and_open_access_listing(self, monkeypatch):
        requests = []

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            if request.full_url.startswith(PaperMetadataVerifier.CROSSREF_URL):
                return json_response({"message": crossref_work()})
            if request.full_url.startswith(PaperMetadataVerifier.OPENALEX_URL):
                return json_response(openalex_work())
            raise AssertionError("Verifier contacted an unexpected URL")

        monkeypatch.setattr(
            "scholarly_search.source_verifier.urlopen", fake_urlopen
        )
        result = PaperMetadataVerifier().verify({
            "title": TITLE,
            "doi": f"https://doi.org/{DOI}",
            "publication_year": "2024",
            "authors": ["Jane Doe"],
            "url": "https://user-controlled.example/should-not-be-requested",
        })

        assert result["status"] == "verified"
        assert result["field_checks"]["title"]["matches"] is True
        assert result["field_checks"]["doi"]["matches"] is True
        assert result["field_checks"]["publication_year"]["matches"] is True
        assert result["field_checks"]["authors"]["matches"] is True
        assert result["field_checks"]["openalex_doi"]["matches"] is True
        assert result["crossref"]["venue"] == "Example Journal"
        assert result["open_access"]["status"] == "open_access"
        assert result["open_access"]["pdf_url"].endswith(".pdf")
        assert result["open_access"]["url_accessibility"] == "provider_listed_not_checked"
        assert all(
            request.full_url.startswith(
                (PaperMetadataVerifier.CROSSREF_URL, PaperMetadataVerifier.OPENALEX_URL)
            )
            for request, _ in requests
        )

    def test_title_lookup_requires_close_match_and_checks_year(self, monkeypatch):
        requested = []

        def fake_urlopen(request, timeout):
            requested.append(request.full_url)
            if request.full_url.startswith(PaperMetadataVerifier.CROSSREF_URL):
                return json_response({
                    "message": {
                        "items": [
                            crossref_work(
                                title=["Entirely Different Paper"],
                                DOI="10.1234/unrelated",
                            ),
                            crossref_work(),
                        ]
                    }
                })
            return json_response(openalex_work())

        monkeypatch.setattr(
            "scholarly_search.source_verifier.urlopen", fake_urlopen
        )
        result = PaperMetadataVerifier().verify({
            "title": TITLE,
            "publication_year": 2022,
        })

        assert result["status"] == "mismatch"
        assert result["crossref"]["doi"] == DOI
        assert result["field_checks"]["title"]["matches"] is True
        assert result["field_checks"]["publication_year"]["matches"] is False
        assert any("query.bibliographic" in url for url in requested)
        assert any(url.endswith("/works/doi:10.1234/example.paper") for url in requested)

    def test_reports_not_found_for_both_registries(self, monkeypatch):
        def not_found(request, timeout):
            raise HTTPError(request.full_url, 404, "Not found", {}, None)

        monkeypatch.setattr(
            "scholarly_search.source_verifier.urlopen", not_found
        )
        result = PaperMetadataVerifier(max_retries=0).verify({"doi": DOI})

        assert result["status"] == "not_found"
        assert result["crossref"] is None
        assert result["openalex"] is None
        assert len(result["warnings"]) == 2
        assert result["open_access"]["status"] == "unknown"

    def test_reports_partial_when_only_crossref_has_record(self, monkeypatch):
        def fake_urlopen(request, timeout):
            if request.full_url.startswith(PaperMetadataVerifier.CROSSREF_URL):
                return json_response({"message": crossref_work()})
            raise HTTPError(request.full_url, 404, "Not found", {}, None)

        monkeypatch.setattr(
            "scholarly_search.source_verifier.urlopen", fake_urlopen
        )
        result = PaperMetadataVerifier(max_retries=0).verify({"doi": DOI})

        assert result["status"] == "partial"
        assert result["crossref"]["doi"] == DOI
        assert result["openalex"] is None
        assert result["open_access"]["status"] == "unknown"

    def test_preserves_crossref_result_when_openalex_is_unavailable(self, monkeypatch):
        def fake_urlopen(request, timeout):
            if request.full_url.startswith(PaperMetadataVerifier.CROSSREF_URL):
                return json_response({"message": crossref_work()})
            raise HTTPError(request.full_url, 503, "Unavailable", {}, None)

        monkeypatch.setattr(
            "scholarly_search.source_verifier.urlopen", fake_urlopen
        )
        result = PaperMetadataVerifier(max_retries=0).verify({"doi": DOI})

        assert result["status"] == "partial"
        assert result["crossref"]["doi"] == DOI
        assert result["provider_errors"][0]["provider"] == "OpenAlex"
        assert "HTTP 503" in result["provider_errors"][0]["error"]

    def test_reports_unavailable_when_all_registries_fail(self, monkeypatch):
        def unavailable(request, timeout):
            raise HTTPError(request.full_url, 503, "Unavailable", {}, None)

        monkeypatch.setattr(
            "scholarly_search.source_verifier.urlopen", unavailable
        )
        result = PaperMetadataVerifier(max_retries=0).verify({"doi": DOI})

        assert result["status"] == "unavailable"
        assert {error["provider"] for error in result["provider_errors"]} == {
            "Crossref",
            "OpenAlex",
        }
        assert not any("No matching" in warning for warning in result["warnings"])

    def test_retries_rate_limited_registry_request(self, monkeypatch):
        requests = []

        def fake_urlopen(request, timeout):
            requests.append(request.full_url)
            if len(requests) == 1:
                raise HTTPError(
                    request.full_url, 429, "Rate limited",
                    {"Retry-After": "0"}, None
                )
            if request.full_url.startswith(PaperMetadataVerifier.CROSSREF_URL):
                return json_response({"message": crossref_work()})
            return json_response(openalex_work())

        monkeypatch.setattr(
            "scholarly_search.source_verifier.urlopen", fake_urlopen
        )
        monkeypatch.setattr(
            "scholarly_search.source_verifier.time.sleep", lambda _: None
        )
        result = PaperMetadataVerifier(max_retries=1).verify({"doi": DOI})

        assert result["status"] == "verified"
        assert len(requests) == 3

    def test_rejects_oversized_provider_responses(self, monkeypatch):
        monkeypatch.setattr(
            "scholarly_search.source_verifier.urlopen",
            lambda request, timeout: io.BytesIO(b'{"message": {}}'),
        )
        verifier = PaperMetadataVerifier(max_retries=0)
        verifier.MAX_RESPONSE_BYTES = 5

        result = verifier.verify({"doi": DOI})

        assert result["status"] == "unavailable"
        assert all(
            "size limit" in error["error"]
            for error in result["provider_errors"]
        )

    def test_sanitizes_provider_listed_urls_and_disclaims_accessibility(self, monkeypatch):
        def fake_urlopen(request, timeout):
            if request.full_url.startswith(PaperMetadataVerifier.CROSSREF_URL):
                return json_response({"message": crossref_work()})
            return json_response(openalex_work(
                best_oa_location={
                    "landing_page_url": "javascript:alert(1)",
                    "pdf_url": "http://insecure.example/paper.pdf",
                }
            ))

        monkeypatch.setattr(
            "scholarly_search.source_verifier.urlopen", fake_urlopen
        )
        result = PaperMetadataVerifier().verify({"doi": DOI})

        assert result["open_access"]["status"] == "open_access"
        assert result["open_access"]["pdf_url"] is None
        assert result["open_access"]["landing_page_url"] is None
        assert result["open_access"]["url_accessibility"] == "provider_listed_not_checked"

    def test_does_not_attribute_oa_links_to_a_different_doi(self, monkeypatch):
        def fake_urlopen(request, timeout):
            if request.full_url.startswith(PaperMetadataVerifier.CROSSREF_URL):
                raise HTTPError(request.full_url, 404, "Not found", {}, None)
            return json_response(openalex_work(
                doi="https://doi.org/10.9999/different-paper"
            ))

        monkeypatch.setattr(
            "scholarly_search.source_verifier.urlopen", fake_urlopen
        )
        result = PaperMetadataVerifier(max_retries=0).verify({"doi": DOI})

        assert result["status"] == "mismatch"
        assert result["field_checks"]["openalex_doi"]["matches"] is False
        assert result["openalex"]["doi_matches_lookup"] is False
        assert result["open_access"]["status"] == "unknown"
        assert result["open_access"]["pdf_url"] is None

    @pytest.mark.parametrize(
        "paper",
        [
            {},
            {"title": "Title", "doi": "not a DOI"},
            {"title": "x" * 501},
            "not a mapping",
        ],
    )
    def test_validates_inputs(self, paper):
        with pytest.raises(ScholarlySearchError):
            PaperMetadataVerifier().verify(paper)

    def test_surfaces_provider_errors_and_invalid_json(self, monkeypatch):
        verifier = PaperMetadataVerifier(max_retries=0)

        def fail_http(request, timeout):
            raise HTTPError(request.full_url, 503, "Unavailable", {}, None)

        monkeypatch.setattr(
            "scholarly_search.source_verifier.urlopen", fail_http
        )
        unavailable = verifier.verify({"doi": DOI})
        assert unavailable["status"] == "unavailable"
        assert all(
            "HTTP 503" in error["error"]
            for error in unavailable["provider_errors"]
        )

        monkeypatch.setattr(
            "scholarly_search.source_verifier.urlopen",
            lambda request, timeout: io.BytesIO(b"invalid json"),
        )
        invalid_response = verifier.verify({"doi": DOI})
        assert invalid_response["status"] == "unavailable"
        assert all(
            "invalid JSON" in error["error"]
            for error in invalid_response["provider_errors"]
        )
