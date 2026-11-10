"""Evaluation infrastructure for Crossref vs OpenAlex search comparison
and the Phase 1 extraction / retrieval / citation harness."""

from evaluation.gold_set_schema import (
    GoldSetValidationError,
    load_annotation,
    load_gold_set,
    validate_annotation,
)
from evaluation.metrics import (
    FieldResult,
    HarnessResult,
    PaperResult,
    evaluate_paper,
)

# Import scholarly_search evaluation separately to avoid circular import
try:
    from scholarly_search.evaluation import (
        SearchMetrics,
        calculate_metrics,
    )
except ImportError:
    # pymupdf may not be installed in all environments
    pass
