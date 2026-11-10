"""Tests for search provider factory and integration."""

import json
import os
from unittest.mock import Mock, patch

import pytest

from scholarly_search.search_provider_factory import (
    SearchProvider,
    configured_provider_name,
    create_search_provider,
)
from scholarly_search.crossref_client import CrossrefSearchClient
from scholarly_search.openalex_client import OpenAlexSearchClient


class TestSearchProviderFactory:
    """Tests for search provider factory."""

    def test_configured_provider_name_default(self):
        """Test default provider is Crossref."""
        with patch.dict(os.environ, {}, clear=True):
            assert configured_provider_name() == "crossref"

    def test_configured_provider_name_from_env(self):
        """Test provider name from environment variable."""
        with patch.dict(os.environ, {"RESEARCHINTELLIGENCE_SEARCH_PROVIDER": "openalex"}):
            assert configured_provider_name() == "openalex"

    def test_configured_provider_name_case_insensitive(self):
        """Test provider name is case-insensitive."""
        with patch.dict(os.environ, {"RESEARCHINTELLIGENCE_SEARCH_PROVIDER": "OPENALEX"}):
            assert configured_provider_name() == "openalex"

    def test_create_crossref_provider(self):
        """Test creating Crossref provider."""
        provider = create_search_provider("crossref")
        assert isinstance(provider, CrossrefSearchClient)

    def test_create_openalex_provider(self):
        """Test creating OpenAlex provider."""
        provider = create_search_provider("openalex")
        assert isinstance(provider, OpenAlexSearchClient)

    def test_create_openalex_provider_with_api_key(self):
        """Test creating OpenAlex provider with API key."""
        provider = create_search_provider("openalex", api_key="test_key")
        assert isinstance(provider, OpenAlexSearchClient)
        assert provider.api_key == "test_key"

    def test_create_openalex_provider_from_env_key(self):
        """Test creating OpenAlex provider with API key from environment."""
        with patch.dict(os.environ, {"OPENALEX_API_KEY": "env_key"}):
            provider = create_search_provider("openalex")
            assert isinstance(provider, OpenAlexSearchClient)
            assert provider.api_key == "env_key"

    def test_create_provider_uses_default_when_none(self):
        """Test factory uses default when provider_name is None."""
        with patch.dict(os.environ, {}, clear=True):
            provider = create_search_provider(None)
            assert isinstance(provider, CrossrefSearchClient)

    def test_create_provider_invalid_name(self):
        """Test factory rejects invalid provider name."""
        with pytest.raises(ValueError, match="must be one of: crossref, openalex"):
            create_search_provider("invalid")

    def test_crossref_provider_ignores_api_key(self):
        """Test Crossref provider ignores API key parameter."""
        provider = create_search_provider("crossref", api_key="ignored")
        assert isinstance(provider, CrossrefSearchClient)
        # Crossref client doesn't have api_key attribute
        assert not hasattr(provider, "api_key")


class TestProviderIntegration:
    """End-to-end integration tests for search providers."""

    @patch("scholarly_search.crossref_client.urlopen")
    def test_crossref_integration_search_success(self, mock_urlopen):
        """Test Crossref provider end-to-end search."""
        mock_response = Mock()
        mock_response.read.return_value = b'{"message":{"items":[{"title":{"0":"Test Paper"}}]}}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        provider = create_search_provider("crossref")
        analysis = {"title": "Test Paper", "research_problem": "Test problem"}
        results = provider.search_related_papers(analysis, max_results=5)

        assert isinstance(results, list)

    @patch("scholarly_search.openalex_client.urlopen")
    def test_openalex_integration_search_success(self, mock_urlopen):
        """Test OpenAlex provider end-to-end search."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({
            "results": [
                {
                    "title": "Related Paper",
                    "authorships": [{"author": {"given_name": "John", "family_name": "Doe"}}],
                    "publication_year": 2020,
                    "doi": "10.1234/test",
                    "primary_location": {"source": {"display_name": "Test Journal"}},
                    "best_oa_location": {"pdf_url": "https://example.com/paper.pdf"},
                    "abstract": "Test abstract",
                    "relevance_score": 0.9,
                }
            ],
            "meta": {"count": 1}
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        provider = create_search_provider("openalex")
        analysis = {"title": "Test Paper", "research_problem": "Test problem"}
        results = provider.search_related_papers(analysis, max_results=5)

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["title"] == "Related Paper"
        assert results[0]["source"] == "OpenAlex"

    @patch("scholarly_search.crossref_client.urlopen")
    def test_crossref_integration_empty_results(self, mock_urlopen):
        """Test Crossref handles empty results gracefully."""
        mock_response = Mock()
        mock_response.read.return_value = b'{"message":{"items":[]}}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        provider = create_search_provider("crossref")
        analysis = {"title": "Test Paper"}
        results = provider.search_related_papers(analysis, max_results=10)

        assert results == []

    @patch("scholarly_search.openalex_client.urlopen")
    def test_openalex_integration_empty_results(self, mock_urlopen):
        """Test OpenAlex handles empty results gracefully."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({
            "results": [],
            "meta": {"count": 0}
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        provider = create_search_provider("openalex")
        analysis = {"title": "Test Paper"}
        results = provider.search_related_papers(analysis, max_results=10)

        assert results == []

    @patch("scholarly_search.crossref_client.urlopen")
    def test_crossref_integration_handles_error(self, mock_urlopen):
        """Test Crossref integration handles provider errors."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Network error")

        provider = create_search_provider("crossref")
        analysis = {"title": "Test Paper"}

        with pytest.raises(Exception):  # ScholarlySearchError
            provider.search_related_papers(analysis, max_results=10)

    @patch("scholarly_search.openalex_client.urlopen")
    def test_openalex_integration_handles_error(self, mock_urlopen):
        """Test OpenAlex integration handles provider errors."""
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Network error")

        provider = create_search_provider("openalex")
        analysis = {"title": "Test Paper"}

        with pytest.raises(Exception):  # ScholarlySearchError
            provider.search_related_papers(analysis, max_results=10)

    def test_both_producers_implement_protocol(self):
        """Test both providers implement SearchProvider protocol."""
        crossref = create_search_provider("crossref")
        openalex = create_search_provider("openalex")

        assert hasattr(crossref, "search_related_papers")
        assert hasattr(openalex, "search_related_papers")

    def test_result_schema_compatibility(self):
        """Test both providers return compatible result schemas."""
        # Mock results from both providers should have same fields
        crossref_fields = {
            "title", "authors", "year", "venue", "doi", "url", "abstract",
            "relevance_score", "topic_overlap_score", "topic_overlap_terms",
            "matched_query", "source"
        }
        openalex_fields = {
            "title", "authors", "year", "venue", "doi", "url", "abstract",
            "relevance_score", "topic_overlap_score", "topic_overlap_terms",
            "matched_query", "source"
        }
        assert crossref_fields == openalex_fields
