"""Offline evaluation infrastructure for related-paper search quality.

This module provides metric calculations for evaluating search provider performance
using deterministic, mocked datasets. Metrics are for offline testing and comparison;
they do not represent real-world provider performance.
"""

from typing import Any, Dict, List, Mapping, Set


class SearchMetrics:
    """Container for search quality metrics."""

    def __init__(
        self,
        precision_at_5: float,
        duplicate_rate: float,
        metadata_completeness: float,
        average_topic_overlap: float,
        provider_coverage: int,
        total_results: int,
    ):
        self.precision_at_5 = precision_at_5
        self.duplicate_rate = duplicate_rate
        self.metadata_completeness = metadata_completeness
        self.average_topic_overlap = average_topic_overlap
        self.provider_coverage = provider_coverage
        self.total_results = total_results

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary for reporting."""
        return {
            "precision_at_5": self.precision_at_5,
            "duplicate_rate": self.duplicate_rate,
            "metadata_completeness": self.metadata_completeness,
            "average_topic_overlap": self.average_topic_overlap,
            "provider_coverage": self.provider_coverage,
            "total_results": self.total_results,
        }


def calculate_precision_at_5(
    results: List[Mapping[str, Any]],
    relevant_dois: Set[str],
) -> float:
    """Calculate precision@5 using human-defined relevance labels.

    Args:
        results: List of search results ranked by relevance.
        relevant_dois: Set of DOIs labeled as truly relevant.

    Returns:
        Precision@5 score (0.0 to 1.0). If fewer than 5 results exist,
        calculates precision using available results.
    """
    if not results:
        return 0.0

    top_k = min(5, len(results))
    relevant_count = 0

    for result in results[:top_k]:
        doi = result.get("doi", "").lower()
        if doi in relevant_dois:
            relevant_count += 1

    return relevant_count / top_k


def calculate_duplicate_rate(
    results: List[Mapping[str, Any]],
    original_candidate_count: int,
) -> float:
    """Calculate duplicate rate after deduplication.

    Args:
        results: Deduplicated search results.
        original_candidate_count: Number of candidates before deduplication.

    Returns:
        Duplicate rate (0.0 to 1.0). Denominator is original_candidate_count.
        Returns 0.0 if original_candidate_count is 0.
    """
    if original_candidate_count == 0:
        return 0.0

    if not results:
        return 1.0  # All results were duplicates

    duplicates_removed = original_candidate_count - len(results)
    return duplicates_removed / original_candidate_count


def calculate_metadata_completeness(
    results: List[Mapping[str, Any]],
) -> float:
    """Calculate average metadata completeness across results.

    Args:
        results: List of search results.

    Returns:
        Average completeness (0.0 to 1.0) across 6 fields:
        title, authors, year, doi, abstract, url.
    """
    if not results:
        return 0.0

    fields_to_check = ["title", "authors", "year", "doi", "abstract", "url"]
    total_fields = len(results) * len(fields_to_check)
    complete_fields = 0

    for result in results:
        for field in fields_to_check:
            value = result.get(field)
            if value is not None and value != "" and value != []:
                complete_fields += 1

    return complete_fields / total_fields


def calculate_average_topic_overlap(
    results: List[Mapping[str, Any]],
) -> float:
    """Calculate average topic overlap score across results.

    Args:
        results: List of search results with topic_overlap_score field.

    Returns:
        Average topic overlap score (0.0 to 1.0).
    """
    if not results:
        return 0.0

    total_score = 0.0
    count = 0

    for result in results:
        # Only count if the field explicitly exists
        if "topic_overlap_score" in result:
            score = result["topic_overlap_score"]
            if isinstance(score, (int, float)):
                total_score += float(score)
                count += 1

    return total_score / count if count > 0 else 0.0


def calculate_metrics(
    results: List[Mapping[str, Any]],
    relevant_dois: Set[str],
    original_candidate_count: int,
) -> SearchMetrics:
    """Calculate all search quality metrics.

    Args:
        results: Deduplicated and ranked search results.
        relevant_dois: Set of DOIs labeled as truly relevant.
        original_candidate_count: Number of candidates before deduplication.

    Returns:
        SearchMetrics object with all calculated metrics.
    """
    precision_at_5 = calculate_precision_at_5(results, relevant_dois)
    duplicate_rate = calculate_duplicate_rate(results, original_candidate_count)
    metadata_completeness = calculate_metadata_completeness(results)
    average_topic_overlap = calculate_average_topic_overlap(results)
    provider_coverage = original_candidate_count
    total_results = len(results)

    return SearchMetrics(
        precision_at_5=precision_at_5,
        duplicate_rate=duplicate_rate,
        metadata_completeness=metadata_completeness,
        average_topic_overlap=average_topic_overlap,
        provider_coverage=provider_coverage,
        total_results=total_results,
    )
