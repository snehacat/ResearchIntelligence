"""Scholarly paper discovery integrations."""

from scholarly_search.crossref_client import CrossrefSearchClient, ScholarlySearchError
from scholarly_search.openalex_client import OpenAlexSearchClient
from scholarly_search.source_verifier import PaperMetadataVerifier
from scholarly_search.related_paper_extractor import (
    PaperDownloadError,
    RelatedPaperEvidenceExtractor,
)

__all__ = [
    "CrossrefSearchClient",
    "OpenAlexSearchClient",
    "PaperMetadataVerifier",
    "PaperDownloadError",
    "RelatedPaperEvidenceExtractor",
    "ScholarlySearchError",
]
