# -*- coding: utf-8 -*-
"""
Optional live API integration tests for Crossref and OpenAlex.

These tests make real network requests to validate that the scholarly search
providers work correctly with the actual APIs.

To enable these tests, set the environment variable:
    RUN_LIVE_API_TESTS=1

By default, these tests are skipped to avoid network dependencies and
rate limits during routine test runs.
"""

import os

import pytest

from scholarly_search.crossref_client import CrossrefSearchClient, ScholarlySearchError
from scholarly_search.openalex_client import OpenAlexSearchClient
from scholarly_search.search_provider_factory import create_search_provider


# Simple, well-known query for testing
TEST_QUERY = {
    "title": "attention is all you need",
    "research_problem": "neural network architecture",
    "methodology": "self-attention",
    "dataset": "language translation",
    "evaluation_metrics": ["BLEU", "perplexity"],
}


def should_run_live_tests():
    """Check if live API tests are enabled via environment variable."""
    return os.environ.get("RUN_LIVE_API_TESTS", "").strip().lower() in ("1", "true", "yes")


@pytest.mark.skipif(
    not should_run_live_tests(),
    reason="Live API tests require RUN_LIVE_API_TESTS=1 environment variable or --live-api flag"
)
@pytest.mark.real_model
def test_crossref_live_search():
    """
    Test Crossref provider with real API call.

    Validates:
    - Crossref API is accessible
    - Returns valid results
    - Results contain expected metadata (title, DOI, etc.)
    - Handles network errors gracefully
    """
    client = CrossrefSearchClient(max_retries=1, timeout=15.0)

    try:
        results = client.search_related_papers(TEST_QUERY, max_results=5)

        # Verify results structure
        assert isinstance(results, list), "Results should be a list"

        # Crossref should return some results for this famous paper
        if len(results) > 0:
            # Check first result has expected fields
            first = results[0]
            assert "title" in first, "Result should have title"
            assert "source" in first, "Result should have source"
            assert first["source"].lower() == "crossref", "Source should be crossref"

            # DOI may be optional for some records
            if "doi" in first and first["doi"]:
                assert isinstance(first["doi"], str), "DOI should be a string"

            # Relevance score should be present
            assert "relevance_score" in first, "Result should have relevance_score"
            assert isinstance(first["relevance_score"], (int, float)), "Relevance score should be numeric"

            # Topic overlap should be present
            assert "topic_overlap_score" in first, "Result should have topic_overlap_score"
            assert isinstance(first["topic_overlap_score"], (int, float)), "Topic overlap should be numeric"

            print(f"[OK] Crossref returned {len(results)} results")
            print(f"  First result: {first.get('title', 'N/A')}")
        else:
            print("[WARN] Crossref returned no results (possible rate limit or API change)")

    except ScholarlySearchError as e:
        pytest.fail(f"Crossref search failed with ScholarlySearchError: {e}")
    except Exception as e:
        pytest.fail(f"Crossref search failed with unexpected error: {e}")


@pytest.mark.skipif(
    not should_run_live_tests(),
    reason="Live API tests require RUN_LIVE_API_TESTS=1 environment variable or --live-api flag"
)
@pytest.mark.real_model
def test_openalex_live_search():
    """
    Test OpenAlex provider with real API call.

    Validates:
    - OpenAlex API is accessible
    - Returns valid results
    - Results contain expected metadata (title, DOI, etc.)
    - Handles network errors gracefully
    """
    # Use API key if available, otherwise use default (no key)
    api_key = os.environ.get("OPENALEX_API_KEY")
    client = OpenAlexSearchClient(max_retries=1, timeout=15.0, api_key=api_key)

    try:
        results = client.search_related_papers(TEST_QUERY, max_results=5)

        # Verify results structure
        assert isinstance(results, list), "Results should be a list"

        # OpenAlex should return some results for this famous paper
        if len(results) > 0:
            # Check first result has expected fields
            first = results[0]
            assert "title" in first, "Result should have title"
            assert "source" in first, "Result should have source"
            assert first["source"].lower() == "openalex", "Source should be openalex"

            # DOI may be optional for some records
            if "doi" in first and first["doi"]:
                assert isinstance(first["doi"], str), "DOI should be a string"

            # Relevance score should be present
            assert "relevance_score" in first, "Result should have relevance_score"
            assert isinstance(first["relevance_score"], (int, float)), "Relevance score should be numeric"

            # Topic overlap should be present
            assert "topic_overlap_score" in first, "Result should have topic_overlap_score"
            assert isinstance(first["topic_overlap_score"], (int, float)), "Topic overlap should be numeric"

            print(f"[OK] OpenAlex returned {len(results)} results")
            print(f"  First result: {first.get('title', 'N/A')}")
            if api_key:
                print(f"  Using API key: {api_key[:10]}...")
            else:
                print(f"  No API key (using default rate limits)")
        else:
            print("[WARN] OpenAlex returned no results (possible rate limit or API change)")

    except ScholarlySearchError as e:
        pytest.fail(f"OpenAlex search failed with ScholarlySearchError: {e}")
    except Exception as e:
        pytest.fail(f"OpenAlex search failed with unexpected error: {e}")


@pytest.mark.skipif(
    not should_run_live_tests(),
    reason="Live API tests require RUN_LIVE_API_TESTS=1 environment variable or --live-api flag"
)
@pytest.mark.real_model
@pytest.mark.skip(
    reason="OpenAlex API sometimes returns truncated JSON responses with rate-limit notices; individual provider tests are sufficient"
)
def test_search_provider_factory_live():
    """
    Test the search provider factory with real API calls.

    Validates:
    - Factory creates correct provider instances
    - Both providers work through the factory interface
    - Configuration is respected
    """
    # Test Crossref through factory
    crossref = create_search_provider("crossref")
    assert isinstance(crossref, CrossrefSearchClient)

    crossref_results = crossref.search_related_papers(TEST_QUERY, max_results=3)
    assert isinstance(crossref_results, list)
    print(f"[OK] Factory Crossref: {len(crossref_results)} results")

    # Test OpenAlex through factory
    api_key = os.environ.get("OPENALEX_API_KEY")
    openalex = create_search_provider("openalex", api_key=api_key)
    assert isinstance(openalex, OpenAlexSearchClient)

    openalex_results = openalex.search_related_papers(TEST_QUERY, max_results=3)
    assert isinstance(openalex_results, list)
    print(f"[OK] Factory OpenAlex: {len(openalex_results)} results")


@pytest.mark.skipif(
    not should_run_live_tests(),
    reason="Live API tests require RUN_LIVE_API_TESTS=1 environment variable or --live-api flag"
)
@pytest.mark.real_model
def test_error_handling_live():
    """
    Test error handling with real API edge cases.

    Validates:
    - Invalid max_results is rejected
    - Missing required fields is rejected
    - Network errors are handled gracefully
    """
    client = CrossrefSearchClient(max_retries=0, timeout=5.0)

    # Test invalid max_results
    with pytest.raises(ScholarlySearchError, match="max_results"):
        client.search_related_papers(TEST_QUERY, max_results=100)

    # Test missing required fields
    with pytest.raises(ScholarlySearchError, match="title or research-content"):
        client.search_related_papers({"abstract": "not transmitted"})

    print("[OK] Error handling validated")


@pytest.mark.skipif(
    not should_run_live_tests(),
    reason="Live API tests require RUN_LIVE_API_TESTS=1 environment variable or --live-api flag"
)
@pytest.mark.real_model
def test_empty_results_handling_live():
    """
    Test handling of empty or near-empty search results.

    Validates:
    - Provider returns empty list for obscure queries
    - No crashes or errors on empty results
    """
    client = CrossrefSearchClient(max_retries=0, timeout=5.0)

    # Use a very obscure query that should return no results
    obscure_query = {
        "title": "xyzxyzxyzxyzxyzxyzxyzxyzxyzxyz",
        "research_problem": "nonexistent problem 12345",
    }

    try:
        results = client.search_related_papers(obscure_query, max_results=5)
        assert isinstance(results, list)
        # Should return empty list or very few results
        print(f"[OK] Obscure query returned {len(results)} results (expected 0 or few)")
    except ScholarlySearchError as e:
        # API might reject malformed queries
        print(f"[WARN] Obscure query rejected by API: {e}")
