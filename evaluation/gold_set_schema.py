"""
Gold-set annotation schema and validator for Phase 1 evaluation.

Schema rules
------------
- document_id   : str, must match filename stem
- source_pdf    : str, relative path to the PDF from project root
- annotator     : str, non-empty
- annotation_date: str, ISO 8601 date (YYYY-MM-DD)
- notes         : str | None
- fields        : dict mapping each of the 10 schema field names to a
                  FieldAnnotation with 'value' and 'evidence_pages'

String fields  : title, publication_year, abstract, research_problem,
                 methodology, dataset, reported_results
  value        : str | None   (None only when genuinely absent in paper)

List fields    : authors, evaluation_metrics, limitations
  value        : list[str] | None

evidence_pages : list[int]  — 1-based page numbers; [] allowed only when
                 information is spread throughout or not applicable

Placeholder values such as "TODO", "...", "unknown" are rejected so that
accidental un-filled templates never silently pass validation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The 10 fields PaperAnalyzer extracts
STRING_FIELDS: Tuple[str, ...] = (
    "title",
    "publication_year",
    "abstract",
    "research_problem",
    "methodology",
    "dataset",
    "reported_results",
)

LIST_FIELDS: Tuple[str, ...] = (
    "authors",
    "evaluation_metrics",
    "limitations",
)

ALL_FIELDS: Tuple[str, ...] = STRING_FIELDS + LIST_FIELDS

_PLACEHOLDER_PATTERNS = re.compile(
    r"^(?:todo|\.{2,}|n/a|na|unknown|placeholder|fill ?in|tbd)$",
    re.IGNORECASE,
)

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class GoldSetValidationError(ValueError):
    """Raised when a gold-set annotation file fails validation."""


def _reject_placeholder(value: str, location: str) -> None:
    if _PLACEHOLDER_PATTERNS.match(value.strip()):
        raise GoldSetValidationError(
            f"{location}: placeholder value '{value}' — fill in the actual annotation."
        )


def validate_annotation(data: Dict[str, Any], filename_stem: str) -> None:
    """Validate a parsed gold-set annotation dict.

    Args:
        data: Parsed JSON dict.
        filename_stem: Stem of the file (without .json) for cross-check.

    Raises:
        GoldSetValidationError: On any schema or value violation.
    """
    # Top-level required keys
    for key in ("document_id", "source_pdf", "annotator", "annotation_date", "fields"):
        if key not in data:
            raise GoldSetValidationError(f"Missing top-level key: '{key}'")

    doc_id = data["document_id"]
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise GoldSetValidationError("'document_id' must be a non-empty string.")
    if doc_id != filename_stem:
        raise GoldSetValidationError(
            f"'document_id' ('{doc_id}') does not match filename stem ('{filename_stem}')."
        )

    if not isinstance(data["source_pdf"], str) or not data["source_pdf"].strip():
        raise GoldSetValidationError("'source_pdf' must be a non-empty string path.")

    if not isinstance(data["annotator"], str) or not data["annotator"].strip():
        raise GoldSetValidationError("'annotator' must be a non-empty string.")
    _reject_placeholder(data["annotator"], "annotator")

    if not isinstance(data["annotation_date"], str) or not _DATE_PATTERN.match(
        data["annotation_date"]
    ):
        raise GoldSetValidationError(
            "'annotation_date' must be an ISO 8601 date string (YYYY-MM-DD)."
        )

    # notes is optional
    notes = data.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise GoldSetValidationError("'notes' must be a string or null.")

    # fields
    fields = data["fields"]
    if not isinstance(fields, dict):
        raise GoldSetValidationError("'fields' must be a dict.")

    missing = set(ALL_FIELDS) - set(fields.keys())
    if missing:
        raise GoldSetValidationError(f"Missing field annotations: {sorted(missing)}")

    extra = set(fields.keys()) - set(ALL_FIELDS)
    if extra:
        raise GoldSetValidationError(f"Unknown fields in annotation: {sorted(extra)}")

    for field_name in ALL_FIELDS:
        fa = fields[field_name]
        if not isinstance(fa, dict):
            raise GoldSetValidationError(
                f"fields.{field_name}: must be a dict with 'value' and 'evidence_pages'."
            )
        if "value" not in fa:
            raise GoldSetValidationError(f"fields.{field_name}: missing 'value'.")
        if "evidence_pages" not in fa:
            raise GoldSetValidationError(f"fields.{field_name}: missing 'evidence_pages'.")

        value = fa["value"]
        evidence_pages = fa["evidence_pages"]

        # evidence_pages validation
        if not isinstance(evidence_pages, list):
            raise GoldSetValidationError(
                f"fields.{field_name}.evidence_pages must be a list of ints."
            )
        for page in evidence_pages:
            if not isinstance(page, int) or isinstance(page, bool) or page < 1:
                raise GoldSetValidationError(
                    f"fields.{field_name}.evidence_pages: '{page}' is not a positive integer."
                )

        # value validation
        if value is None:
            # Explicitly absent — allowed
            continue

        if field_name in STRING_FIELDS:
            if not isinstance(value, str):
                raise GoldSetValidationError(
                    f"fields.{field_name}.value must be a string or null."
                )
            _reject_placeholder(value, f"fields.{field_name}.value")

        elif field_name in LIST_FIELDS:
            if not isinstance(value, list):
                raise GoldSetValidationError(
                    f"fields.{field_name}.value must be a list of strings or null."
                )
            if not value:
                raise GoldSetValidationError(
                    f"fields.{field_name}.value must not be an empty list; use null if absent."
                )
            for i, item in enumerate(value):
                if not isinstance(item, str) or not item.strip():
                    raise GoldSetValidationError(
                        f"fields.{field_name}.value[{i}]: must be a non-empty string."
                    )
                _reject_placeholder(item, f"fields.{field_name}.value[{i}]")


def load_annotation(path: Path) -> Dict[str, Any]:
    """Load and validate a gold-set annotation from disk.

    Args:
        path: Path to the .json annotation file.

    Returns:
        Validated annotation dict.

    Raises:
        GoldSetValidationError: On schema violations.
        json.JSONDecodeError: On malformed JSON.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    validate_annotation(data, path.stem)
    return data


def load_gold_set(gold_dir: Path) -> List[Dict[str, Any]]:
    """Load every valid annotation from a directory.

    Args:
        gold_dir: Directory containing .json annotation files.

    Returns:
        List of validated annotation dicts (alphabetical order by document_id).

    Raises:
        GoldSetValidationError: If any file fails validation (stops loading).
    """
    paths = sorted(gold_dir.glob("*.json"))
    if not paths:
        raise GoldSetValidationError(
            f"No .json annotation files found in {gold_dir}"
        )
    return [load_annotation(p) for p in paths]
