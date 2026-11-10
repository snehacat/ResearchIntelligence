"""
Citation sample schema and validator for Phase 1 item 3.

Format
------
A citation sample file is a JSON object:

{
  "document_id": "<str>  — matches the gold-set document_id",
  "annotator":   "<str>  — non-empty, no placeholder",
  "annotation_date": "<str>  — ISO 8601 date",
  "annotated_without_model_output": <bool>,
  "notes": "<str | null>",
  "pairs": [
    {
      "pair_id":    "<str>  — unique within this file, e.g. 'title_p1'",
      "field_name": "<str>  — one of the 10 schema fields",
      "claim":      "<str>  — the extracted claim text as returned by the analyzer",
      "passage":    "<str>  — the cited passage text from the PDF chunk",
      "chunk_id":   "<str>  — the chunk_id from the analyzer's citation",
      "page":       <int | null>,  — 1-based page; null if unavailable
      "label":      "supported" | "partially_supported" | "unsupported"
    },
    ...
  ]
}

Label definitions
-----------------
  supported           : The passage clearly and directly supports the claim.
  partially_supported : The passage is related but incomplete; the claim
                        requires inference or additional context.
  unsupported         : The passage does not support the claim, or the claim
                        contains information absent from the passage.

Agreement metric
----------------
  Auto-traceability label (from metrics.py) is "traceable" / "not_traceable".
  Human label is mapped:
    supported | partially_supported → "traceable"   (lenient mapping)
    unsupported                     → "not_traceable"

  Agreement rate = fraction of pairs where auto and human agree.
  Confusion matrix: TP, FP, FN, TN reported in harness output.
  
  Design note: the lenient mapping (partially_supported → traceable) is chosen
  because auto-traceability only checks page overlap, not semantic support.
  Using strict mapping (only "supported" → traceable) would systematically
  under-count agreement for passages that partially support a claim.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluation.gold_set_schema import ALL_FIELDS, _PLACEHOLDER_PATTERNS, _DATE_PATTERN

VALID_LABELS: Tuple[str, ...] = ("supported", "partially_supported", "unsupported")

# Human labels that map to "traceable" for agreement comparison
TRACEABLE_LABELS = {"supported", "partially_supported"}


class CitationSampleValidationError(ValueError):
    """Raised when a citation sample file fails validation."""


def _reject_placeholder(value: str, location: str) -> None:
    if _PLACEHOLDER_PATTERNS.match(value.strip()):
        raise CitationSampleValidationError(
            f"{location}: placeholder value '{value}' — fill in the actual annotation."
        )


def validate_citation_sample(data: Dict[str, Any], filename_stem: str) -> None:
    """Validate a parsed citation sample dict.

    Args:
        data:          Parsed JSON dict.
        filename_stem: Stem of the file (without .json) for cross-check.

    Raises:
        CitationSampleValidationError: On any schema or value violation.
    """
    for key in ("document_id", "annotator", "annotation_date",
                "annotated_without_model_output", "pairs"):
        if key not in data:
            raise CitationSampleValidationError(f"Missing top-level key: '{key}'")

    doc_id = data["document_id"]
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise CitationSampleValidationError("'document_id' must be a non-empty string.")
    if doc_id != filename_stem:
        raise CitationSampleValidationError(
            f"'document_id' ('{doc_id}') does not match filename stem ('{filename_stem}')."
        )

    if not isinstance(data["annotator"], str) or not data["annotator"].strip():
        raise CitationSampleValidationError("'annotator' must be a non-empty string.")
    _reject_placeholder(data["annotator"], "annotator")

    if not isinstance(data["annotation_date"], str) or not _DATE_PATTERN.match(
        data["annotation_date"]
    ):
        raise CitationSampleValidationError(
            "'annotation_date' must be an ISO 8601 date string (YYYY-MM-DD)."
        )

    awmo = data["annotated_without_model_output"]
    if not isinstance(awmo, bool):
        raise CitationSampleValidationError(
            "'annotated_without_model_output' must be a boolean."
        )

    notes = data.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise CitationSampleValidationError("'notes' must be a string or null.")

    pairs = data["pairs"]
    if not isinstance(pairs, list):
        raise CitationSampleValidationError("'pairs' must be a list.")
    if len(pairs) < 1:
        raise CitationSampleValidationError("'pairs' must contain at least one entry.")

    seen_ids: set = set()
    for i, pair in enumerate(pairs):
        loc = f"pairs[{i}]"
        if not isinstance(pair, dict):
            raise CitationSampleValidationError(f"{loc}: must be a dict.")
        for key in ("pair_id", "field_name", "claim", "passage", "chunk_id", "label"):
            if key not in pair:
                raise CitationSampleValidationError(f"{loc}: missing key '{key}'.")

        pair_id = pair["pair_id"]
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise CitationSampleValidationError(f"{loc}.pair_id must be a non-empty string.")
        _reject_placeholder(pair_id, f"{loc}.pair_id")
        if pair_id in seen_ids:
            raise CitationSampleValidationError(
                f"{loc}.pair_id '{pair_id}' is not unique within this file."
            )
        seen_ids.add(pair_id)

        field_name = pair["field_name"]
        if field_name not in ALL_FIELDS:
            raise CitationSampleValidationError(
                f"{loc}.field_name '{field_name}' is not a valid schema field. "
                f"Valid fields: {list(ALL_FIELDS)}"
            )

        for text_key in ("claim", "passage"):
            v = pair[text_key]
            if not isinstance(v, str) or not v.strip():
                raise CitationSampleValidationError(
                    f"{loc}.{text_key} must be a non-empty string."
                )
            _reject_placeholder(v, f"{loc}.{text_key}")

        chunk_id = pair["chunk_id"]
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise CitationSampleValidationError(
                f"{loc}.chunk_id must be a non-empty string."
            )
        _reject_placeholder(chunk_id, f"{loc}.chunk_id")

        page = pair.get("page")
        if page is not None:
            if not isinstance(page, int) or isinstance(page, bool) or page < 1:
                raise CitationSampleValidationError(
                    f"{loc}.page must be a positive integer or null."
                )

        label = pair["label"]
        if label not in VALID_LABELS:
            raise CitationSampleValidationError(
                f"{loc}.label '{label}' is not valid. Must be one of {VALID_LABELS}."
            )
        _reject_placeholder(str(label), f"{loc}.label")


def load_citation_sample(path: Path) -> Dict[str, Any]:
    """Load and validate a citation sample from disk."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    validate_citation_sample(data, path.stem)
    return data


def build_citation_sample_template(document_id: str, num_pairs: int = 5) -> Dict[str, Any]:
    """Return a template citation sample with placeholder pairs."""
    from datetime import date

    pairs = []
    for i in range(1, num_pairs + 1):
        pairs.append({
            "pair_id": f"TODO_pair_{i}",
            "field_name": "title",
            "claim": "TODO",
            "passage": "TODO",
            "chunk_id": "TODO",
            "page": None,
            "label": "supported",
        })
    return {
        "document_id": document_id,
        "annotator": "TODO",
        "annotation_date": date.today().isoformat(),
        "annotated_without_model_output": False,
        "notes": None,
        "pairs": pairs,
    }


def compute_citation_agreement(
    citation_sample: Dict[str, Any],
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute agreement between auto-traceability and human labels.

    Auto-traceability logic (mirrors metrics.py):
      A citation is "traceable" when:
        - evidence_sufficiency ∈ {sufficient, partial}
        - at least one citation entry is present
        - (page match when page is annotated, else citation presence)

    Human label mapping (lenient):
      supported | partially_supported → traceable
      unsupported → not_traceable

    Args:
        citation_sample: Validated citation sample dict.
        analysis:        Output of PaperAnalyzer.analyze_paper.

    Returns:
        Dict with agreement_rate, confusion counts, and per-pair breakdown.
    """
    evidence_citations: Dict[str, Any] = analysis.get("evidence_citations") or {}
    pairs = citation_sample.get("pairs", [])

    tp = fp = fn = tn = 0
    per_pair = []

    for pair in pairs:
        field_name = pair["field_name"]
        human_label = pair["label"]
        human_traceable = human_label in TRACEABLE_LABELS

        # Compute auto-traceability for this field
        citation = evidence_citations.get(field_name, {}) or {}
        sufficiency = citation.get("evidence_sufficiency", "insufficient")
        citations_list = citation.get("citations") or []
        has_citation = bool(citations_list)
        page = pair.get("page")

        if page is not None and has_citation:
            from evaluation.metrics import _pages_from_citations
            cited_pages = _pages_from_citations(citations_list)
            page_match = page in set(cited_pages)
        else:
            page_match = has_citation

        auto_traceable = (
            sufficiency in ("sufficient", "partial")
            and has_citation
            and page_match
        )

        # Confusion matrix
        if auto_traceable and human_traceable:
            tp += 1
        elif auto_traceable and not human_traceable:
            fp += 1
        elif not auto_traceable and human_traceable:
            fn += 1
        else:
            tn += 1

        per_pair.append({
            "pair_id": pair["pair_id"],
            "field_name": field_name,
            "human_label": human_label,
            "human_traceable": human_traceable,
            "auto_traceable": auto_traceable,
            "agree": auto_traceable == human_traceable,
        })

    total = tp + fp + fn + tn
    agreement_rate = (tp + tn) / total if total > 0 else None

    return {
        "document_id": citation_sample["document_id"],
        "num_pairs": total,
        "agreement_rate": round(agreement_rate, 4) if agreement_rate is not None else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "footnote": (
            "Human label mapping: supported|partially_supported → traceable (lenient). "
            "Agreement measures whether auto-traceability and human labeling agree on "
            "whether the system pointed to the right passage. "
            "It does NOT measure whether the passage factually supports the claim."
        ),
        "per_pair": per_pair,
    }
