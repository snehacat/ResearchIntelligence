"""
Phase 1 evaluation metrics.

Three metrics are computed per field, then aggregated across all annotated
papers in the gold set.

─────────────────────────────────────────────────────────────────────────────
1. Field-level extraction accuracy
─────────────────────────────────────────────────────────────────────────────
Definition
  A field is "correct" when the predicted value is an *approximate match*
  to the gold value.  We use token-F1 (macro-averaged over list items) so
  that minor wording differences still score partially rather than 0/1.

  For list fields (authors, evaluation_metrics, limitations) token-F1 is
  computed per gold item vs. the best-matching predicted item, then averaged.

  For string fields token-F1 is computed directly between gold and predicted.

  A field where the gold value is null (information genuinely absent) is
  "correct" when the predicted value is also null/missing/empty; any
  non-empty prediction is penalised.

Limitations (stated, not hidden)
  - Token-F1 is a surface-form metric.  Semantically equivalent phrasings
    with different words will score below 1.0.
  - It cannot detect factually incorrect values that happen to share tokens
    with the gold value.
  - It is an *approximation*; human review of each extracted field remains
    necessary.

─────────────────────────────────────────────────────────────────────────────
2. Retrieval hit-rate@k
─────────────────────────────────────────────────────────────────────────────
Definition
  For a field with evidence_pages annotation [p1, p2, ...], at least one of
  the top-k retrieved chunks for that field must have start_page or end_page
  overlapping any annotated page.  Hit = 1 if satisfied, 0 otherwise.

  hit_rate@k = (fields with ≥1 hit) / (fields with non-empty evidence_pages)

k is configurable (default 3, matching the production retriever top_k).

─────────────────────────────────────────────────────────────────────────────
3. Citation-support rate
─────────────────────────────────────────────────────────────────────────────
Definition
  For each field, the analysis result carries evidence_citations[field].
  A field's citation is "supported" when ALL of the following hold:
    a. evidence_sufficiency ∈ {"sufficient", "partial"}  (not "insufficient")
    b. At least one citation entry is present
    c. At least one citation entry has a start_page that appears in the
       annotated evidence_pages for that field  (page-level traceability)

  citation_support_rate = (fields with supported citation) /
                          (fields with non-null gold value)

  When the gold value is null, citation support is not expected; those fields
  are excluded from the denominator.

Unmeasured items
  Factual correctness of the extracted text vs. the gold annotation is not
  measured automatically — token-F1 is a proxy.  Semantic equivalence,
  numeric precision, and claim-level fact-checking require human review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from evaluation.gold_set_schema import ALL_FIELDS, LIST_FIELDS, STRING_FIELDS


# ── Token helpers ─────────────────────────────────────────────────────────────

_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "with",
}


def _tokenize(text: str) -> List[str]:
    """Lower-case word/number tokens, stopwords removed."""
    tokens = re.findall(r"\d+(?:[.,]\d+)*|[^\W\d_]+", text.casefold())
    return [t for t in tokens if t not in _STOPWORDS]


def _token_f1(predicted: str, gold: str) -> float:
    """Compute token-level F1 between two strings.  Returns 0–1."""
    pred_tokens = _tokenize(predicted)
    gold_tokens = _tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    common = pred_set & gold_set
    precision = len(common) / len(pred_set)
    recall = len(common) / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _list_token_f1(predicted: Any, gold: List[str]) -> float:
    """Macro-average token-F1 for list fields.

    For each gold item find the best-matching predicted item.  Average over
    gold items.  If predicted is None/empty, all gold items score 0.
    """
    pred_items: List[str] = []
    if isinstance(predicted, list):
        pred_items = [str(v) for v in predicted if isinstance(v, str) and v.strip()]
    elif isinstance(predicted, str) and predicted.strip():
        pred_items = [predicted]

    scores: List[float] = []
    for gold_item in gold:
        if not pred_items:
            scores.append(0.0)
        else:
            scores.append(max(_token_f1(p, gold_item) for p in pred_items))
    return sum(scores) / len(scores) if scores else 0.0


# ── Per-field result dataclass ────────────────────────────────────────────────

@dataclass
class FieldResult:
    field_name: str
    # Extraction accuracy
    token_f1: float                   # 0–1
    gold_is_null: bool                # True → gold says field absent
    predicted_is_null: bool           # True → model returned null/empty
    correct_null: bool                # gold_is_null AND predicted_is_null
    # Retrieval hit-rate
    retrieval_hit: Optional[bool]     # None when no evidence_pages annotated
    retrieved_pages: List[int] = field(default_factory=list)
    annotated_pages: List[int] = field(default_factory=list)
    # Citation support
    citation_supported: Optional[bool] = None  # None when gold is null


@dataclass
class PaperResult:
    document_id: str
    field_results: List[FieldResult]

    # Aggregated over non-null fields
    @property
    def mean_token_f1(self) -> float:
        scores = [r.token_f1 for r in self.field_results if not r.gold_is_null]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def hit_rate_at_k(self) -> Optional[float]:
        eligible = [r for r in self.field_results if r.retrieval_hit is not None]
        if not eligible:
            return None
        return sum(1 for r in eligible if r.retrieval_hit) / len(eligible)

    @property
    def citation_support_rate(self) -> Optional[float]:
        eligible = [r for r in self.field_results if r.citation_supported is not None]
        if not eligible:
            return None
        return sum(1 for r in eligible if r.citation_supported) / len(eligible)


@dataclass
class HarnessResult:
    paper_results: List[PaperResult]
    top_k: int

    @property
    def macro_token_f1(self) -> float:
        vals = [p.mean_token_f1 for p in self.paper_results]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_hit_rate(self) -> Optional[float]:
        vals = [p.hit_rate_at_k for p in self.paper_results if p.hit_rate_at_k is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def macro_citation_support_rate(self) -> Optional[float]:
        vals = [
            p.citation_support_rate
            for p in self.paper_results
            if p.citation_support_rate is not None
        ]
        return sum(vals) / len(vals) if vals else None

    def to_dict(self) -> Dict[str, Any]:
        per_paper = []
        for pr in self.paper_results:
            per_field = []
            for fr in pr.field_results:
                per_field.append({
                    "field": fr.field_name,
                    "token_f1": round(fr.token_f1, 4),
                    "gold_is_null": fr.gold_is_null,
                    "predicted_is_null": fr.predicted_is_null,
                    "correct_null": fr.correct_null,
                    "retrieval_hit": fr.retrieval_hit,
                    "annotated_pages": fr.annotated_pages,
                    "retrieved_pages": fr.retrieved_pages,
                    "citation_supported": fr.citation_supported,
                })
            per_paper.append({
                "document_id": pr.document_id,
                "mean_token_f1": round(pr.mean_token_f1, 4),
                "hit_rate_at_k": (
                    round(pr.hit_rate_at_k, 4)
                    if pr.hit_rate_at_k is not None else None
                ),
                "citation_support_rate": (
                    round(pr.citation_support_rate, 4)
                    if pr.citation_support_rate is not None else None
                ),
                "fields": per_field,
            })
        return {
            "top_k": self.top_k,
            "summary": {
                "macro_token_f1": (
                    round(self.macro_token_f1, 4)
                    if self.macro_token_f1 is not None else None
                ),
                "macro_hit_rate_at_k": (
                    round(self.macro_hit_rate, 4)
                    if self.macro_hit_rate is not None else None
                ),
                "macro_citation_support_rate": (
                    round(self.macro_citation_support_rate, 4)
                    if self.macro_citation_support_rate is not None else None
                ),
                "num_papers": len(self.paper_results),
            },
            "unmeasured": (
                "Factual correctness requires human review. "
                "Token-F1 is a surface-form proxy only."
            ),
            "per_paper": per_paper,
        }


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate_paper(
    annotation: Dict[str, Any],
    analysis: Dict[str, Any],
    top_k: int = 3,
) -> PaperResult:
    """Compute all three metrics for one paper.

    Args:
        annotation: Validated gold-set annotation dict (from load_annotation).
        analysis:   Output of PaperAnalyzer.analyze_paper for the same paper.
        top_k:      Number of retrieved chunks to consider for hit-rate.

    Returns:
        PaperResult with per-field breakdown.
    """
    document_id = annotation["document_id"]
    gold_fields = annotation["fields"]
    evidence_citations: Dict[str, Any] = analysis.get("evidence_citations") or {}

    field_results: List[FieldResult] = []

    for field_name in ALL_FIELDS:
        fa = gold_fields[field_name]
        gold_value = fa["value"]
        evidence_pages: List[int] = fa.get("evidence_pages") or []

        predicted_value = analysis.get(field_name)

        # ── 1. Extraction accuracy ────────────────────────────────────────────
        gold_is_null = gold_value is None
        predicted_is_null = _is_empty(predicted_value)

        if gold_is_null:
            # Correct iff model also returns null/empty
            correct_null = predicted_is_null
            token_f1 = 1.0 if correct_null else 0.0
        elif predicted_is_null:
            correct_null = False
            token_f1 = 0.0
        else:
            correct_null = False
            if field_name in LIST_FIELDS:
                gold_list = gold_value if isinstance(gold_value, list) else [gold_value]
                token_f1 = _list_token_f1(predicted_value, gold_list)
            else:
                token_f1 = _token_f1(str(predicted_value), str(gold_value))

        # ── 2. Retrieval hit-rate@k ───────────────────────────────────────────
        retrieval_hit: Optional[bool] = None
        retrieved_pages: List[int] = []

        if evidence_pages:
            citation = evidence_citations.get(field_name, {}) or {}
            retrieved_chunks = citation.get("evidence_chunks") or []
            # Collect page numbers from top-k chunks
            retrieved_pages = _pages_from_chunks(retrieved_chunks[:top_k])
            annotated_set = set(evidence_pages)
            retrieved_set = set(retrieved_pages)
            retrieval_hit = bool(annotated_set & retrieved_set)

        # ── 3. Citation-support rate ──────────────────────────────────────────
        citation_supported: Optional[bool] = None

        if not gold_is_null:
            citation = evidence_citations.get(field_name, {}) or {}
            sufficiency = citation.get("evidence_sufficiency", "insufficient")
            citations_list = citation.get("citations") or []
            has_citation = bool(citations_list)
            # Page-level traceability: at least one cited page in annotated pages
            if evidence_pages and has_citation:
                cited_pages = _pages_from_citations(citations_list)
                page_match = bool(set(cited_pages) & set(evidence_pages))
            elif not evidence_pages:
                # No page annotation → cannot check page traceability;
                # fall back to sufficiency + presence
                page_match = has_citation
            else:
                page_match = False

            citation_supported = (
                sufficiency in ("sufficient", "partial")
                and has_citation
                and page_match
            )

        field_results.append(
            FieldResult(
                field_name=field_name,
                token_f1=token_f1,
                gold_is_null=gold_is_null,
                predicted_is_null=predicted_is_null,
                correct_null=correct_null,
                retrieval_hit=retrieval_hit,
                retrieved_pages=retrieved_pages,
                annotated_pages=evidence_pages,
                citation_supported=citation_supported,
            )
        )

    return PaperResult(document_id=document_id, field_results=field_results)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return all(
            (not isinstance(v, str) or not v.strip()) for v in value
        )
    return False


def _pages_from_chunks(chunks: List[Dict[str, Any]]) -> List[int]:
    pages: List[int] = []
    for chunk in chunks:
        for key in ("start_page", "end_page"):
            page = chunk.get(key)
            if isinstance(page, int) and not isinstance(page, bool) and page >= 1:
                pages.append(page)
    return sorted(set(pages))


def _pages_from_citations(citations: List[Dict[str, Any]]) -> List[int]:
    pages: List[int] = []
    for c in citations:
        for key in ("start_page", "end_page"):
            page = c.get(key)
            if isinstance(page, int) and not isinstance(page, bool) and page >= 1:
                pages.append(page)
    return sorted(set(pages))
