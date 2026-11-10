"""Tests for Crossref-based related paper discovery."""

import io
import json
from urllib.error import HTTPError

import pytest

from scholarly_search.crossref_client import (
    CrossrefSearchClient,
    ScholarlySearchError,
)


@pytest.fixture
def search_client():
    return CrossrefSearchClient(max_retries=0)


class TestQueryConstruction:
    def test_builds_title_and_research_content_queries(self, search_client):
        queries = search_client._build_queries({
            "title": "Vision models for medical imaging",
            "research_problem": "Improve tumor detection",
            "methodology": "Convolutional neural networks",
            "dataset": "Public MRI scans",
            "evaluation_metrics": ["F1", "recall"],
            "abstract": "This private full abstract is not transmitted.",
        })

        assert queries[0] == "Vision models for medical imaging"
        assert "Improve tumor detection" in queries[1]
        assert "Convolutional neural networks" in queries[1]
        assert "Public MRI scans" in queries[1]
        assert "F1 recall" in queries[1]
        assert all("private full abstract" not in query for query in queries)
        assert len(queries[1]) <= 400

    def test_requires_searchable_fields(self, search_client):
        with pytest.raises(ScholarlySearchError, match="title or research-content"):
            search_client.search_related_papers({"abstract": "not transmitted"})

    @pytest.mark.parametrize("value", [0, 51, True, 1.5, "5"])
    def test_validates_result_count(self, search_client, value):
        with pytest.raises(ScholarlySearchError, match="max_results"):
            search_client.search_related_papers({"title": "Paper"}, value)


class TestCrossrefResults:
    def test_normalizes_deduplicates_and_excludes_uploaded_paper(
        self, search_client, monkeypatch
    ):
        requests = []

        def fake_fetch(query, rows):
            requests.append((query, rows))
            return {
                "items": [
                    {
                        "title": ["Original Paper"],
                        "DOI": "10.1234/original",
                        "score": 200,
                    },
                    {
                        "title": ["Related Paper"],
                        "author": [{"given": "Jane", "family": "Doe"}],
                        "published-print": {"date-parts": [[2023, 5]]},
                        "container-title": ["Research Journal"],
                        "DOI": "https://doi.org/10.5678/related",
                        "abstract": "<jats:p>Useful &amp; relevant.</jats:p>",
                        "score": 15,
                        "URL": "https://api.crossref.org/works/10.5678/related",
                    },
                    {
                        "title": ["Related Paper"],
                        "DOI": "10.5678/related",
                        "score": 10,
                    },
                    {
                        "title": ["Another Related Work"],
                        "score": 8,
                        "URL": "https://example.org/paper",
                    },
                    {"title": [], "score": 500},
                ]
            }

        monkeypatch.setattr(search_client, "_fetch_works", fake_fetch)
        results = search_client.search_related_papers(
            {
                "title": "Original Paper",
                "doi": "10.1234/original",
                "research_problem": "Find related work",
            },
            max_results=5,
        )

        assert [paper["title"] for paper in results] == [
            "Related Paper",
            "Another Related Work",
        ]
        assert results[0]["authors"] == ["Jane Doe"]
        assert results[0]["year"] == 2023
        assert results[0]["venue"] == "Research Journal"
        assert results[0]["doi"] == "10.5678/related"
        assert results[0]["url"] == "https://doi.org/10.5678/related"
        assert results[0]["abstract"] == "Useful & relevant."
        assert results[0]["source"] == "Crossref"
        assert len(requests) == 2
        assert all(rows == 15 for _, rows in requests)

    def test_ranks_by_topic_overlap_and_merges_same_title_author_versions(
        self, search_client, monkeypatch
    ):
        monkeypatch.setattr(
            search_client,
            "_fetch_works",
            lambda query, rows: {
                "items": [
                    {
                        "title": ["Credit Card Debt Survey"],
                        "author": [{"given": "S. Kathi", "family": "Brown"}],
                        "DOI": "10.26419/res.00929.002",
                        "score": 1,
                    },
                    {
                        "title": ["Credit Card Debt Survey"],
                        "author": [{"given": "S. Kathi", "family": "Brown"}],
                        "DOI": "10.26419/res.00929.003",
                        "score": 1,
                        "abstract": "Questionnaire by age for credit-card debt.",
                    },
                    {
                        "title": ["A Guide to Digital Image Processing"],
                        "DOI": "10.1234/unrelated",
                        "score": 10000,
                    },
                ]
            },
        )

        results = search_client.search_related_papers(
            {
                "title": (
                    "A Study of Electronic Banking and Credit Card Services "
                    "in Shimla"
                ),
                "research_problem": (
                    "Assess credit card use and satisfaction among residents"
                ),
                "methodology": "Survey questionnaire",
            },
            max_results=10,
        )

        assert len(results) == 2
        assert results[0]["title"] == "Credit Card Debt Survey"
        assert results[0]["topic_overlap_score"] > results[1][
            "topic_overlap_score"
        ]
        assert {"credit", "card"}.issubset(
            set(results[0]["topic_overlap_terms"])
        )
        assert results[0]["alternate_dois"] == [
            "10.26419/res.00929.002"
        ]
        assert results[0]["doi"] != results[0]["alternate_dois"][0]

    def test_banking_topic_overlap_uses_credit_card_and_local_study_terms(self):
        from scholarly_search.crossref_client import CrossrefSearchClient

        score = CrossrefSearchClient._topic_overlap_score(
            {
                "title": (
                    "A Study of Electronic Banking Regarding Credit Card "
                    "Services in Shimla"
                ),
                "research_problem": (
                    "Assess credit card use and satisfaction among residents"
                ),
                "methodology": "Survey questionnaire",
                "dataset": "100 participants in Shimla",
            },
            {
                "title": "Credit Card Debt Survey: Annotated Questionnaire by Age",
            },
        )

        assert score > 0.3

    def test_skips_malformed_items_and_normalizes_missing_metadata(
        self, search_client, monkeypatch
    ):
        monkeypatch.setattr(
            search_client,
            "_fetch_works",
            lambda query, rows: {
                "items": [
                    None,
                    {
                        "title": ["Paper without optional metadata"],
                        "score": "not-a-number",
                        "DOI": "invalid",
                    },
                ]
            },
        )

        results = search_client.search_related_papers({"title": "Uploaded Work"})

        assert len(results) == 1
        assert results[0]["authors"] == []
        assert results[0]["year"] is None
        assert results[0]["doi"] == ""
        assert results[0]["url"] is None
        assert results[0]["relevance_score"] == 0

    def test_rejects_malformed_works_list(self, search_client, monkeypatch):
        monkeypatch.setattr(
            search_client, "_fetch_works", lambda query, rows: {"items": None}
        )

        with pytest.raises(ScholarlySearchError, match="works list"):
            search_client.search_related_papers({"title": "Uploaded Work"})


class TestCrossrefTransport:
    def test_retries_rate_limit_then_returns_json(self, monkeypatch):
        client = CrossrefSearchClient(max_retries=1)
        calls = []

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            if len(calls) == 1:
                raise HTTPError(
                    request.full_url, 429, "Rate limited",
                    {"Retry-After": "0"}, None
                )
            return io.BytesIO(
                json.dumps({"message": {"items": []}}).encode("utf-8")
            )

        monkeypatch.setattr("scholarly_search.crossref_client.urlopen", fake_urlopen)
        monkeypatch.setattr("scholarly_search.crossref_client.time.sleep", lambda _: None)
        message = client._fetch_works("search terms", rows=10)

        assert message == {"items": []}
        assert len(calls) == 2
        assert calls[0][1] == client.timeout
        assert "query.bibliographic=search+terms" in calls[0][0].full_url

    def test_reports_http_and_invalid_json_failures(self, monkeypatch):
        client = CrossrefSearchClient(max_retries=0)

        def fail_http(request, timeout):
            raise HTTPError(request.full_url, 400, "Bad request", {}, None)

        monkeypatch.setattr("scholarly_search.crossref_client.urlopen", fail_http)
        with pytest.raises(ScholarlySearchError, match="HTTP 400"):
            client._fetch_works("query", rows=1)

        monkeypatch.setattr(
            "scholarly_search.crossref_client.urlopen",
            lambda request, timeout: io.BytesIO(b"not json"),
        )
        with pytest.raises(ScholarlySearchError, match="invalid JSON"):
            client._fetch_works("query", rows=1)

    @pytest.mark.parametrize(
        ("timeout", "retries"),
        [(0, 0), (1, -1), (1, 6)],
    )
    def test_validates_transport_configuration(self, timeout, retries):
        with pytest.raises(ValueError):
            CrossrefSearchClient(timeout=timeout, max_retries=retries)
