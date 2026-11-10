"""Tests for OpenAlex search client."""

import json
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import pytest

from scholarly_search.openalex_client import OpenAlexSearchClient
from scholarly_search.crossref_client import ScholarlySearchError


class TestOpenAlexSearchClient:
    """Tests for OpenAlexSearchClient."""

    def test_initialization_valid(self):
        """Test client initialization with valid parameters."""
        client = OpenAlexSearchClient(timeout=10.0, max_retries=2)
        assert client.timeout == 10.0
        assert client.max_retries == 2
        assert client.api_key is None

    def test_initialization_with_api_key(self):
        """Test client initialization with API key."""
        client = OpenAlexSearchClient(api_key="test_key")
        assert client.api_key == "test_key"

    def test_initialization_invalid_timeout(self):
        """Test initialization rejects invalid timeout."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            OpenAlexSearchClient(timeout=0)

    def test_initialization_invalid_max_retries(self):
        """Test initialization rejects invalid max_retries."""
        with pytest.raises(ValueError, match="max_retries must be between 0 and 5"):
            OpenAlexSearchClient(max_retries=6)

    def test_build_queries_with_title(self):
        """Test query building with title only."""
        analysis = {"title": "Test Paper Title"}
        queries = OpenAlexSearchClient._build_queries(analysis)
        assert len(queries) == 1
        assert "Test Paper Title" in queries[0]

    def test_build_queries_with_topical_fields(self):
        """Test query building with topical fields."""
        analysis = {
            "title": "Test Paper",
            "research_problem": "This is a research problem description",
            "methodology": "We use a novel method",
            "dataset": "Dataset name",
            "evaluation_metrics": ["accuracy", "precision"],
        }
        queries = OpenAlexSearchClient._build_queries(analysis)
        assert len(queries) == 2
        assert any("Test Paper" in q for q in queries)
        assert any("research problem" in q.lower() for q in queries)

    def test_build_queries_empty_analysis(self):
        """Test query building with empty analysis."""
        analysis = {}
        queries = OpenAlexSearchClient._build_queries(analysis)
        assert queries == []

    @patch("scholarly_search.openalex_client.urlopen")
    def test_search_related_papers_success(self, mock_urlopen):
        """Test successful search returns papers."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({
            "results": [
                {
                    "title": "Related Paper 1",
                    "authorships": [
                        {"author": {"given_name": "John", "family_name": "Doe"}}
                    ],
                    "publication_year": 2020,
                    "doi": "10.1234/test1",
                    "primary_location": {
                        "source": {"display_name": "Test Journal"}
                    },
                    "best_oa_location": {
                        "pdf_url": "https://example.com/paper1.pdf"
                    },
                    "abstract": "This is an abstract",
                    "relevance_score": 0.95,
                }
            ],
            "meta": {"count": 1}
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        client = OpenAlexSearchClient()
        analysis = {
            "title": "Test Paper",
            "research_problem": "Test problem",
        }
        results = client.search_related_papers(analysis, max_results=10)

        assert len(results) == 1
        assert results[0]["title"] == "Related Paper 1"
        assert results[0]["authors"] == ["John Doe"]
        assert results[0]["year"] == 2020
        assert results[0]["doi"] == "10.1234/test1"
        assert results[0]["source"] == "OpenAlex"

    @patch("scholarly_search.openalex_client.urlopen")
    def test_search_related_papers_empty_results(self, mock_urlopen):
        """Test search with no matching papers."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({
            "results": [],
            "meta": {"count": 0}
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        client = OpenAlexSearchClient()
        analysis = {"title": "Test Paper"}
        results = client.search_related_papers(analysis, max_results=10)

        assert results == []

    @patch("scholarly_search.openalex_client.urlopen")
    def test_search_related_papers_filters_uploaded_paper(self, mock_urlopen):
        """Test that the uploaded paper is filtered from results."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({
            "results": [
                {
                    "title": "Test Paper",  # Same as uploaded
                    "authorships": [
                        {"author": {"given_name": "John", "family_name": "Doe"}}
                    ],
                    "publication_year": 2020,
                    "doi": "10.1234/test",
                    "relevance_score": 0.95,
                },
                {
                    "title": "Related Paper",
                    "authorships": [
                        {"author": {"given_name": "Jane", "family_name": "Smith"}}
                    ],
                    "publication_year": 2021,
                    "doi": "10.1234/related",
                    "relevance_score": 0.85,
                }
            ],
            "meta": {"count": 2}
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        client = OpenAlexSearchClient()
        analysis = {"title": "Test Paper", "doi": "10.1234/test"}
        results = client.search_related_papers(analysis, max_results=10)

        assert len(results) == 1
        assert results[0]["title"] == "Related Paper"

    @patch("scholarly_search.openalex_client.urlopen")
    def test_search_related_papers_handles_missing_metadata(self, mock_urlopen):
        """Test normalization handles missing optional fields."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({
            "results": [
                {
                    "title": "Minimal Paper",
                    "authorships": [],
                    "publication_year": None,
                    "doi": None,
                    "relevance_score": 0.8,
                }
            ],
            "meta": {"count": 1}
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        client = OpenAlexSearchClient()
        analysis = {"title": "Test Paper"}
        results = client.search_related_papers(analysis, max_results=10)

        assert len(results) == 1
        assert results[0]["title"] == "Minimal Paper"
        assert results[0]["authors"] == []
        assert results[0]["year"] is None
        assert results[0]["doi"] == ""
        assert results[0]["venue"] is None
        assert results[0]["abstract"] is None

    @patch("scholarly_search.openalex_client.urlopen")
    def test_search_related_papers_invalid_max_results(self, mock_urlopen):
        """Test search rejects invalid max_results."""
        client = OpenAlexSearchClient()
        analysis = {"title": "Test Paper"}
        
        with pytest.raises(ScholarlySearchError, match="max_results must be between 1 and 50"):
            client.search_related_papers(analysis, max_results=0)

    @patch("scholarly_search.openalex_client.urlopen")
    def test_search_related_papers_no_query_possible(self, mock_urlopen):
        """Test search fails when no query can be built."""
        client = OpenAlexSearchClient()
        analysis = {}
        
        with pytest.raises(ScholarlySearchError, match="must include a title or research-content"):
            client.search_related_papers(analysis, max_results=10)

    @patch("scholarly_search.openalex_client.urlopen")
    def test_search_related_papers_malformed_response(self, mock_urlopen):
        """Test search handles malformed JSON response."""
        mock_response = Mock()
        mock_response.read.return_value = b"not valid json"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        client = OpenAlexSearchClient()
        analysis = {"title": "Test Paper"}
        
        with pytest.raises(ScholarlySearchError, match="invalid JSON"):
            client.search_related_papers(analysis, max_results=10)

    def test_retry_after_calculation(self):
        """Test retry-after delay calculation."""
        assert OpenAlexSearchClient._retry_after("0.5") == 0.5
        assert OpenAlexSearchClient._retry_after(None) == 0.5
        assert OpenAlexSearchClient._retry_after("invalid") == 0.5

    @patch("scholarly_search.openalex_client.urlopen")
    def test_search_related_papers_deduplicates_by_doi(self, mock_urlopen):
        """Test deduplication by DOI."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({
            "results": [
                {
                    "title": "Paper A",
                    "authorships": [{"author": {"given_name": "John", "family_name": "Doe"}}],
                    "publication_year": 2020,
                    "doi": "10.1234/paper",
                    "relevance_score": 0.9,
                },
                {
                    "title": "Paper A",  # Same DOI
                    "authorships": [{"author": {"given_name": "John", "family_name": "Doe"}}],
                    "publication_year": 2020,
                    "doi": "10.1234/paper",
                    "relevance_score": 0.85,
                }
            ],
            "meta": {"count": 2}
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        client = OpenAlexSearchClient()
        analysis = {"title": "Test Paper"}
        results = client.search_related_papers(analysis, max_results=10)

        assert len(results) == 1
        assert results[0]["relevance_score"] == 0.9  # Higher score kept

    def test_clean_text(self):
        """Test text cleaning."""
        assert OpenAlexSearchClient._clean_text(None) == ""
        assert OpenAlexSearchClient._clean_text(123) == ""
        assert OpenAlexSearchClient._clean_text("  Test  Text  ") == "Test Text"
        assert OpenAlexSearchClient._clean_text("Test\nText") == "Test Text"

    def test_clean_doi(self):
        """Test DOI cleaning."""
        assert OpenAlexSearchClient._clean_doi(None) == ""
        assert OpenAlexSearchClient._clean_doi(123) == ""
        assert OpenAlexSearchClient._clean_doi("10.1234/test") == "10.1234/test"
        assert OpenAlexSearchClient._clean_doi("https://doi.org/10.1234/test") == "10.1234/test"
        assert OpenAlexSearchClient._clean_doi("invalid") == ""

    def test_normalize_title(self):
        """Test title normalization."""
        assert OpenAlexSearchClient._normalize_title("Test Paper Title") == "test paper title"
        assert OpenAlexSearchClient._normalize_title("Test-Paper_Title!") == "test paper title"

    def test_safe_https_url(self):
        """Test HTTPS URL validation."""
        assert OpenAlexSearchClient._safe_https_url("https://example.com") == "https://example.com"
        assert OpenAlexSearchClient._safe_https_url("http://example.com") is None
        assert OpenAlexSearchClient._safe_https_url(None) is None
        assert OpenAlexSearchClient._safe_https_url(123) is None

    def test_topic_tokens(self):
        """Test topic token extraction."""
        tokens = OpenAlexSearchClient._topic_tokens("Machine learning research paper analysis")
        assert "machine" in tokens
        assert "learning" in tokens
        assert "research" not in tokens  # Stopword
        assert "paper" not in tokens  # Stopword

    def test_relevance_score(self):
        """Test relevance score normalization."""
        assert OpenAlexSearchClient._relevance_score(0.95) == 0.95
        assert OpenAlexSearchClient._relevance_score(-1.0) == 0.0
        assert OpenAlexSearchClient._relevance_score("invalid") == 0.0

    def test_normalize_work_with_venue(self):
        """Test work normalization with complete venue metadata."""
        item = {
            "title": "Test Paper",
            "publication_year": 2024,
            "doi": "10.1234/test",
            "primary_location": {
                "source": {
                    "display_name": "Test Conference"
                }
            },
            "best_oa_location": {
                "pdf_url": "https://example.com/paper.pdf"
            },
            "authorships": [
                {
                    "author": {
                        "given_name": "John",
                        "family_name": "Doe"
                    }
                }
            ],
            "abstract": "This is an abstract."
        }
        
        result = OpenAlexSearchClient._normalize_work(item, "test query")
        
        assert result["title"] == "Test Paper"
        assert result["year"] == 2024
        assert result["doi"] == "10.1234/test"
        assert result["venue"] == "Test Conference"
        assert result["url"] == "https://example.com/paper.pdf"
        assert result["abstract"] == "This is an abstract."
        assert result["source"] == "OpenAlex"
    
    def test_normalize_work_with_none_primary_location(self):
        """Test work normalization when primary_location is None (regression test)."""
        item = {
            "title": "Test Paper",
            "publication_year": 2024,
            "doi": "10.1234/test",
            "primary_location": None,  # This caused the AttributeError
            "authorships": [],
            "abstract": None
        }
        
        # Should not raise AttributeError
        result = OpenAlexSearchClient._normalize_work(item, "test query")
        
        assert result["title"] == "Test Paper"
        assert result["year"] == 2024
        assert result["doi"] == "10.1234/test"
        assert result["venue"] is None  # Should be None, not error
        assert result["url"] is None
        assert result["abstract"] is None
        assert result["source"] == "OpenAlex"
    
    def test_normalize_work_with_missing_primary_location(self):
        """Test work normalization when primary_location key is missing."""
        item = {
            "title": "Test Paper",
            "publication_year": 2024,
            "doi": "10.1234/test",
            # primary_location key missing entirely
            "authorships": [],
            "abstract": None
        }
        
        result = OpenAlexSearchClient._normalize_work(item, "test query")
        
        assert result["title"] == "Test Paper"
        assert result["venue"] is None
        assert result["url"] is None
    
    def test_normalize_work_with_none_source_in_primary_location(self):
        """Test work normalization when primary_location.source is None (regression)."""
        item = {
            "title": "Test Paper",
            "publication_year": 2024,
            "doi": "10.1234/test",
            "primary_location": {"source": None},  # This is the actual failure case
            "authorships": [],
            "abstract": None
        }
        
        # Should not raise AttributeError
        result = OpenAlexSearchClient._normalize_work(item, "test query")
        
        assert result["title"] == "Test Paper"
        assert result["year"] == 2024
        assert result["doi"] == "10.1234/test"
        assert result["venue"] is None  # Should be None, not error
        assert result["url"] is None
        assert result["abstract"] is None
        assert result["source"] == "OpenAlex"
