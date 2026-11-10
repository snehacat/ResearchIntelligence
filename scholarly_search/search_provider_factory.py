"""Select a scholarly search provider from application configuration."""

import os
from typing import Optional, Protocol

from scholarly_search.crossref_client import CrossrefSearchClient
from scholarly_search.openalex_client import OpenAlexSearchClient


class SearchProvider(Protocol):
    """Protocol for scholarly search providers."""
    
    def search_related_papers(
        self,
        paper_analysis,
        max_results: int = 10,
    ) -> list:
        """Search for related papers."""
        ...


DEFAULT_PROVIDER = "crossref"


def configured_provider_name() -> str:
    """Return the configured search provider name, defaulting to Crossref."""
    return os.environ.get(
        "RESEARCHINTELLIGENCE_SEARCH_PROVIDER", DEFAULT_PROVIDER
    ).strip().casefold()


def create_search_provider(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> SearchProvider:
    """Construct one of the supported search providers.
    
    Args:
        provider_name: Name of the provider ('crossref' or 'openalex').
            If None, uses environment variable or default.
        api_key: Optional API key for OpenAlex. Ignored for Crossref.
    
    Returns:
        An instance of the configured search provider.
    
    Raises:
        ValueError: If provider_name is not supported.
    """
    if provider_name is None:
        provider_name = configured_provider_name()
    
    if provider_name == "crossref":
        return CrossrefSearchClient()
    
    if provider_name == "openalex":
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("OPENALEX_API_KEY")
        return OpenAlexSearchClient(api_key=api_key)
    
    raise ValueError(
        "RESEARCHINTELLIGENCE_SEARCH_PROVIDER must be one of: crossref, openalex"
    )
