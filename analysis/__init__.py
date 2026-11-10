"""Public exports for structured paper analysis and reporting."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "PaperAnalyzer": ("analysis.paper_analyzer", "PaperAnalyzer"),
    "PaperAnalysisError": ("analysis.paper_analyzer", "PaperAnalysisError"),
    "ExtractionError": ("analysis.paper_analyzer", "ExtractionError"),
    "LLMBackend": ("analysis.llm_backend", "LLMBackend"),
    "LLMGenerationResult": ("analysis.llm_backend", "LLMGenerationResult"),
    "GeminiBackend": ("analysis.gemini_backend", "GeminiBackend"),
    "GroqBackend": ("analysis.groq_backend", "GroqBackend"),
    "AnalysisJobManager": (
        "analysis.analysis_job_manager",
        "AnalysisJobManager",
    ),
    "AnalysisJobStore": ("analysis.job_store", "AnalysisJobStore"),
    "AnalysisJobStoreError": (
        "analysis.job_store",
        "AnalysisJobStoreError",
    ),
    "AnalysisJobStateError": (
        "analysis.job_store",
        "AnalysisJobStateError",
    ),
    "LiteratureComparator": (
        "analysis.literature_comparator",
        "LiteratureComparator",
    ),
    "LiteratureComparisonError": (
        "analysis.literature_comparator",
        "LiteratureComparisonError",
    ),
    "ResearchGapAnalyzer": (
        "analysis.research_gap_analyzer",
        "ResearchGapAnalyzer",
    ),
    "ResearchGapAnalysisError": (
        "analysis.research_gap_analyzer",
        "ResearchGapAnalysisError",
    ),
    "ResearchReportGenerator": (
        "analysis.research_report_generator",
        "ResearchReportGenerator",
    ),
    "ResearchReportError": (
        "analysis.research_report_generator",
        "ResearchReportError",
    ),
    "CrossrefSearchClient": (
        "scholarly_search.crossref_client",
        "CrossrefSearchClient",
    ),
    "OpenAlexSearchClient": (
        "scholarly_search.openalex_client",
        "OpenAlexSearchClient",
    ),
    "create_search_provider": (
        "scholarly_search.search_provider_factory",
        "create_search_provider",
    ),
    "ScholarlySearchError": (
        "scholarly_search.crossref_client",
        "ScholarlySearchError",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, symbol_name = _EXPORTS[name]
    except KeyError:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from None
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value
