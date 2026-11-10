"""
Phase 1 evaluation metrics.

Five metric families are computed per field, then aggregated across all
annotated papers in the gold set.

─────────────────────────────────────────────────────────────────────────────
ITEM 5 — Per-field metrics; set-based for list fields
─────────────────────────────────────────────────────────────────────────────
  Free-text fields (title, publication_year, abstract, research_problem,
  methodology, dataset, reported_results):
    Token-F1 — stopwords removed, macro-averaged per field.
    Labelled "rough surface-form proxy" in report output.

  List fields (authors, evaluation_metrics, limitations):
    Set precision, set recall, set F1 over normalized items
    (lowercase, whitespace-trimmed, punctuation stripped to word-chars).
    Token-F1 is NOT used for list fields (item 5 requirement).

  Null gold / null predicted:
    Correct-null scores 1.0 for that field; non-null prediction on a
    null-gold field scores 0.0.

─────────────────────────────────────────────────────────────────────────────
ITEM 4 — Retrieval hit-rate@k for k = 1, 3, 5; MRR
─────────────────────────────────────────────────────────────────────────────
  hit_rate@k: fraction of fields (with annotated evidence_pages) where at
    least one of the top-k chunks has start_page or end_page overlapping
    any annotated page.

  MRR (Mean Reciprocal Rank): for each eligible field, find the rank r
    (1-based) of the first chunk whose page overlaps an annotated page;
    score = 1/r if found within MAX_RANK chunks, else 0.  Average over
    eligible fields then over papers.

─────────────────────────────────────────────────────────────────────────────
ITEM 2 — Citation traceability rate  (renamed from "citation support rate")
─────────────────────────────────────────────────────────────────────────────
  Formerly called "citation support rate".  Renamed to "citation
  traceability rate" because page overlap plus a sufficiency label only
  demonstrates that the system points to the right region of the paper.
  It does NOT verify that the cited passage semantically supports the
  extracted claim.  This limitation is stated in the report footnote.

  A field's citation is "traceable" when ALL of:
    a. evidence_sufficiency ∈ {"sufficient", "partial"}
    b. At least one citation entry is present
    c. At least one cited page appears in annotated evidence_pages
       (or citation is non-empty when no pages are annotated)

─────────────────────────────────────────────────────────────────────────────
ITEM 3 — Human-labeled citation agreement
─────────────────────────────────────────────────────────────────────────────
  When a human-labeled citation sample file is provided
  (evaluation/citation_sample/), the harness computes:
    - Agreement rate: fraction of pairs where auto-traceability label
      matches the human label (supported → traceable, unsupported → not).
    - Confusion counts: TP, FP, FN, TN.

Unmeasured items (documented, not hidden)
  Factual correctness requires human review.  Token-F1 and set-F1 are
  surface-form proxies.  "Traceable" ≠ "factually supported".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from evaluation.gold_set_schema import ALL_FIELDS, LIST_FIELDS, STRING_FIELDS

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_K_VALUES: Tuple[int, ...] = (1, 3, 5)
_MRR_MAX_RANK = 20   # only consider first 20 chunks for MRR

_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "with",
}

# ── Token / normalization helpers ─────────────────────────────────────────────


def _tokenize(text: str) -> List[str]:
    """Lower-case word/number tokens; stopwords removed."""
    tokens = re.findall(r"\d+(?:[.,]\d+)*|[^\W\d_]+", text.casefold())
    return [t for t in tokens if t not in _STOPWORDS]


def _normalize_item(text: str) -> str:
    """Lowercase, strip, collapse punctuation to spaces, strip again."""
    lowered = text.casefold().strip()
    cleaned = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _token_f1(predicted: str, gold: str) -> float:
    """Compute token-level F1 between two strings.  Returns 0–1.

    This is a *rough surface-form proxy*.  Semantically equivalent
    phrasings with different words will score below 1.0.
    """
    pred_tokens = _tokenize(predicted)
    gold_tokens = _tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    precision = len(common) / len(set(pred_tokens))
    recall = len(common) / len(set(gold_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _list_token_f1(predicted: Any, gold: List[str]) -> float:
    """Legacy macro-average token-F1 for list fields.

    Retained for backward compatibility and tests.  For new code, use
    _set_metrics() instead (item 5 requirement).
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


def _set_metrics(predicted: Any, gold: List[str]) -> Tuple[float, float, float]:
    """Compute set-based precision, recall, F1 for list fields.

    Normalization: lowercase, stripped, punctuation → space.
    Returns (precision, recall, f1).
    """
    pred_set: Set[str] = set()
    if isinstance(predicted, list):
        pred_set = {_normalize_item(str(v)) for v in predicted if isinstance(v, str) and v.strip()}
    elif isinstance(predicted, str) and predicted.strip():
        pred_set = {_normalize_item(predicted)}

    gold_set = {_normalize_item(g) for g in gold if isinstance(g, str) and g.strip()}

    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# ── Per-field result dataclass ────────────────────────────────────────────────

@dataclass
class FieldResult:
    field_name: str
    is_list_field: bool

    # Extraction accuracy
    token_f1: float                   # 0–1  (string fields only; 0 for list fields)
    set_precision: Optional[float]    # list fields only
    set_recall: Optional[float]       # list fields only
    set_f1: Optional[float]           # list fields only

    gold_is_null: bool
    predicted_is_null: bool
    correct_null: bool

    # Retrieval — per-k
    retrieval_hits: Dict[int, Optional[bool]]   # k → True/False/None
    retrieved_pages: List[int]
    annotated_pages: List[int]
    mrr: Optional[float]              # None when no evidence_pages annotated

    # Citation traceability (item 2 — renamed)
    citation_traceable: Optional[bool]  # None when gold is null


@dataclass
class PaperResult:
    document_id: str
    field_results: List[FieldResult]

    # ── Extraction ────────────────────────────────────────────────────────────
    @property
    def mean_token_f1(self) -> float:
        """Average token-F1 over non-null string fields only."""
        scores = [
            r.token_f1
            for r in self.field_results
            if not r.gold_is_null and not r.is_list_field
        ]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def mean_set_f1(self) -> Optional[float]:
        """Average set-F1 over non-null list fields."""
        scores = [
            r.set_f1
            for r in self.field_results
            if not r.gold_is_null and r.is_list_field and r.set_f1 is not None
        ]
        return sum(scores) / len(scores) if scores else None

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def hit_rate_at_k(self, k: int) -> Optional[float]:
        eligible = [
            r for r in self.field_results
            if r.retrieval_hits.get(k) is not None
        ]
        if not eligible:
            return None
        return sum(1 for r in eligible if r.retrieval_hits[k]) / len(eligible)

    @property
    def mrr(self) -> Optional[float]:
        vals = [r.mrr for r in self.field_results if r.mrr is not None]
        return sum(vals) / len(vals) if vals else None

    # ── Citation traceability ─────────────────────────────────────────────────
    @property
    def citation_traceability_rate(self) -> Optional[float]:
        """Fraction of non-null fields with traceable citations.

        NOTE: Traceability means page overlap + non-empty citation.
        It does NOT verify that the cited passage semantically supports
        the extracted claim.
        """
        eligible = [r for r in self.field_results if r.citation_traceable is not None]
        if not eligible:
            return None
        return sum(1 for r in eligible if r.citation_traceable) / len(eligible)


@dataclass
class HarnessResult:
    paper_results: List[PaperResult]
    k_values: Tuple[int, ...] = _DEFAULT_K_VALUES

    # ── Extraction ────────────────────────────────────────────────────────────
    @property
    def macro_token_f1(self) -> float:
        vals = [p.mean_token_f1 for p in self.paper_results]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def macro_set_f1(self) -> Optional[float]:
        vals = [p.mean_set_f1 for p in self.paper_results if p.mean_set_f1 is not None]
        return sum(vals) / len(vals) if vals else None

    # ── Retrieval ─────────────────────────────────────────────────────────────
    def macro_hit_rate(self, k: int) -> Optional[float]:
        vals = [p.hit_rate_at_k(k) for p in self.paper_results if p.hit_rate_at_k(k) is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def macro_mrr(self) -> Optional[float]:
        vals = [p.mrr for p in self.paper_results if p.mrr is not None]
        return sum(vals) / len(vals) if vals else None

    # ── Citation traceability ─────────────────────────────────────────────────
    @property
    def macro_citation_traceability_rate(self) -> Optional[float]:
        vals = [
            p.citation_traceability_rate
            for p in self.paper_results
            if p.citation_traceability_rate is not None
        ]
        return sum(vals) / len(vals) if vals else None

    def to_dict(self) -> Dict[str, Any]:
        per_paper = []
        for pr in self.paper_results:
            per_field = []
            for fr in pr.field_results:
                entry: Dict[str, Any] = {
                    "field": fr.field_name,
                    "is_list_field": fr.is_list_field,
                    "gold_is_null": fr.gold_is_null,
                    "predicted_is_null": fr.predicted_is_null,
                    "correct_null": fr.correct_null,
                    "annotated_pages": fr.annotated_pages,
                    "retrieved_pages": fr.retrieved_pages,
                    "mrr": round(fr.mrr, 4) if fr.mrr is not None else None,
                    "citation_traceable": fr.citation_traceable,
                }
                if fr.is_list_field:
                    entry["set_precision"] = (
                        round(fr.set_precision, 4) if fr.set_precision is not None else None
                    )
                    entry["set_recall"] = (
                        round(fr.set_recall, 4) if fr.set_recall is not None else None
                    )
                    entry["set_f1"] = (
                        round(fr.set_f1, 4) if fr.set_f1 is not None else None
                    )
                else:
                    entry["token_f1"] = round(fr.token_f1, 4)
                    entry["token_f1_note"] = "rough surface-form proxy"
                for k in self.k_values:
                    entry[f"hit@{k}"] = fr.retrieval_hits.get(k)
                per_field.append(entry)

            per_paper.append({
                "document_id": pr.document_id,
                "mean_token_f1": round(pr.mean_token_f1, 4),
                "mean_set_f1": (
                    round(pr.mean_set_f1, 4) if pr.mean_set_f1 is not None else None
                ),
                **{
                    f"hit_rate@{k}": (
                        round(pr.hit_rate_at_k(k), 4)
                        if pr.hit_rate_at_k(k) is not None else None
                    )
                    for k in self.k_values
                },
                "mrr": round(pr.mrr, 4) if pr.mrr is not None else None,
                "citation_traceability_rate": (
                    round(pr.citation_traceability_rate, 4)
                    if pr.citation_traceability_rate is not None else None
                ),
                "fields": per_field,
            })

        summary: Dict[str, Any] = {
            "macro_token_f1": round(self.macro_token_f1, 4),
            "macro_set_f1": (
                round(self.macro_set_f1, 4) if self.macro_set_f1 is not None else None
            ),
            **{
                f"macro_hit_rate@{k}": (
                    round(self.macro_hit_rate(k), 4)
                    if self.macro_hit_rate(k) is not None else None
                )
                for k in self.k_values
            },
            "macro_mrr": (
                round(self.macro_mrr, 4) if self.macro_mrr is not None else None
            ),
            "macro_citation_traceability_rate": (
                round(self.macro_citation_traceability_rate, 4)
                if self.macro_citation_traceability_rate is not None else None
            ),
            "num_papers": len(self.paper_results),
        }

        return {
            "k_values": list(self.k_values),
            "summary": summary,
            "footnotes": {
                "token_f1": (
                    "Token-F1 is a rough surface-form proxy for free-text fields. "
                    "Semantically equivalent but differently worded extractions "
                    "score below 1.0. It cannot detect factually wrong values "
                    "that share tokens with the gold annotation."
                ),
                "set_f1": (
                    "Set-F1 is computed over normalized items (lowercase, trimmed, "
                    "punctuation stripped). It measures item-level overlap, not "
                    "semantic equivalence or factual correctness."
                ),
                "citation_traceability_rate": (
                    "Citation traceability rate measures whether the system points "
                    "to the correct region of the paper (page overlap + non-empty "
                    "citation + non-insufficient sufficiency). "
                    "IT DOES NOT VERIFY that the cited passage semantically "
                    "supports the extracted claim. Human review is always required."
                ),
                "mrr": (
                    f"MRR is computed over the first {_MRR_MAX_RANK} retrieved "
                    "chunks per field, using the same page-overlap rule as hit-rate."
                ),
            },
            "per_paper": per_paper,
        }


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate_paper(
    annotation: Dict[str, Any],
    analysis: Dict[str, Any],
    k_values: Tuple[int, ...] = _DEFAULT_K_VALUES,
) -> PaperResult:
    """Compute all metrics for one paper.

    Args:
        annotation: Validated gold-set annotation dict (from load_annotation).
        analysis:   Output of PaperAnalyzer.analyze_paper for the same paper.
        k_values:   k values for hit-rate computation (default: 1, 3, 5).

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
        is_list = field_name in LIST_FIELDS

        # ── 1. Extraction accuracy ────────────────────────────────────────────
        gold_is_null = gold_value is None
        predicted_is_null = _is_empty(predicted_value)

        token_f1 = 0.0
        set_precision: Optional[float] = None
        set_recall: Optional[float] = None
        set_f1: Optional[float] = None

        if gold_is_null:
            correct_null = predicted_is_null
            token_f1 = 1.0 if correct_null else 0.0
            if is_list:
                set_precision = 1.0 if correct_null else 0.0
                set_recall = 1.0 if correct_null else 0.0
                set_f1 = 1.0 if correct_null else 0.0
        elif predicted_is_null:
            correct_null = False
            token_f1 = 0.0
            if is_list:
                set_precision, set_recall, set_f1 = 0.0, 0.0, 0.0
        else:
            correct_null = False
            if is_list:
                gold_list = gold_value if isinstance(gold_value, list) else [gold_value]
                set_precision, set_recall, set_f1 = _set_metrics(predicted_value, gold_list)
                # token_f1 stays 0 for list fields (reported as N/A)
            else:
                token_f1 = _token_f1(str(predicted_value), str(gold_value))

        # ── 2. Retrieval hit-rate@k and MRR ──────────────────────────────────
        retrieval_hits: Dict[int, Optional[bool]] = {k: None for k in k_values}
        retrieved_pages: List[int] = []
        mrr: Optional[float] = None

        if evidence_pages:
            citation = evidence_citations.get(field_name, {}) or {}
            retrieved_chunks = citation.get("evidence_chunks") or []
            retrieved_pages = _pages_from_chunks(retrieved_chunks)
            annotated_set = set(evidence_pages)

            for k in k_values:
                top_k_pages = set(_pages_from_chunks(retrieved_chunks[:k]))
                retrieval_hits[k] = bool(annotated_set & top_k_pages)

            # MRR — first rank (1-based) where a chunk's page overlaps
            mrr = 0.0
            for rank_0, chunk in enumerate(retrieved_chunks[:_MRR_MAX_RANK]):
                chunk_pages = set(_pages_from_chunks([chunk]))
                if annotated_set & chunk_pages:
                    mrr = 1.0 / (rank_0 + 1)
                    break

        # ── 3. Citation traceability (item 2) ─────────────────────────────────
        citation_traceable: Optional[bool] = None

        if not gold_is_null:
            citation = evidence_citations.get(field_name, {}) or {}
            sufficiency = citation.get("evidence_sufficiency", "insufficient")
            citations_list = citation.get("citations") or []
            has_citation = bool(citations_list)

            if evidence_pages and has_citation:
                cited_pages = _pages_from_citations(citations_list)
                page_match = bool(set(cited_pages) & set(evidence_pages))
            elif not evidence_pages:
                # No page annotation → fall back to citation presence
                page_match = has_citation
            else:
                page_match = False

            citation_traceable = (
                sufficiency in ("sufficient", "partial")
                and has_citation
                and page_match
            )

        field_results.append(
            FieldResult(
                field_name=field_name,
                is_list_field=is_list,
                token_f1=token_f1,
                set_precision=set_precision,
                set_recall=set_recall,
                set_f1=set_f1,
                gold_is_null=gold_is_null,
                predicted_is_null=predicted_is_null,
                correct_null=correct_null,
                retrieval_hits=retrieval_hits,
                retrieved_pages=retrieved_pages,
                annotated_pages=evidence_pages,
                mrr=mrr,
                citation_traceable=citation_traceable,
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
