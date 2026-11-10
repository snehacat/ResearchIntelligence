"""Tests for Phase 1 evaluation harness: schema validator, metrics, and runner.

All tests are offline (no LLM or network calls).  The conftest autouse fixture
already blocks external services.
"""

from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from evaluation.citation_sample_schema import (
    CitationSampleValidationError,
    build_citation_sample_template,
    compute_citation_agreement,
    validate_citation_sample,
)
from evaluation.generate_template import build_template
from evaluation.gold_set_schema import (
    ALL_FIELDS,
    GoldSetValidationError,
    load_annotation,
    load_gold_set,
    validate_annotation,
)
from evaluation.metrics import (
    FieldResult,
    HarnessResult,
    PaperResult,
    _DEFAULT_K_VALUES,
    _is_empty,
    _list_token_f1,
    _normalize_item,
    _set_metrics,
    _token_f1,
    evaluate_paper,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def valid_annotation() -> Dict[str, Any]:
    """A minimal but fully valid annotation for testing."""
    return {
        "document_id": "test_paper",
        "source_pdf": "validation/step5/papers/test_paper.pdf",
        "annotator": "Test Annotator",
        "annotation_date": "2026-11-10",
        "split": "dev",
        "annotated_without_model_output": True,
        "notes": None,
        "fields": {
            "title": {
                "value": "Attention Is All You Need",
                "evidence_pages": [1],
            },
            "authors": {
                "value": ["Ashish Vaswani", "Noam Shazeer"],
                "evidence_pages": [1],
            },
            "publication_year": {
                "value": "2017",
                "evidence_pages": [1],
            },
            "abstract": {
                "value": "The dominant sequence transduction models are based on complex recurrent networks.",
                "evidence_pages": [1],
            },
            "research_problem": {
                "value": "Improving sequence transduction without recurrence.",
                "evidence_pages": [1],
            },
            "methodology": {
                "value": "Transformer architecture with self-attention.",
                "evidence_pages": [2, 3],
            },
            "dataset": {
                "value": "WMT 2014 English-German and English-French translation tasks.",
                "evidence_pages": [4],
            },
            "evaluation_metrics": {
                "value": ["BLEU", "training cost"],
                "evidence_pages": [5],
            },
            "reported_results": {
                "value": "28.4 BLEU on WMT 2014 English-to-German.",
                "evidence_pages": [5],
            },
            "limitations": {
                "value": ["Restricted to fixed-length sequences."],
                "evidence_pages": [6],
            },
        },
    }


@pytest.fixture()
def valid_analysis(valid_annotation) -> Dict[str, Any]:
    """A synthetic analysis matching the gold annotation above."""
    doc_id = valid_annotation["document_id"]
    chunks = [
        {
            "chunk_id": f"{doc_id}_chunk_0",
            "document_id": doc_id,
            "chunk_index": 0,
            "start_page": 1,
            "end_page": 1,
            "text": "Attention Is All You Need Ashish Vaswani Noam Shazeer 2017",
        },
        {
            "chunk_id": f"{doc_id}_chunk_1",
            "document_id": doc_id,
            "chunk_index": 1,
            "start_page": 2,
            "end_page": 3,
            "text": "Transformer architecture with self-attention mechanism.",
        },
        {
            "chunk_id": f"{doc_id}_chunk_2",
            "document_id": doc_id,
            "chunk_index": 2,
            "start_page": 4,
            "end_page": 4,
            "text": "WMT 2014 English-German and English-French translation tasks.",
        },
        {
            "chunk_id": f"{doc_id}_chunk_3",
            "document_id": doc_id,
            "chunk_index": 3,
            "start_page": 5,
            "end_page": 5,
            "text": "BLEU score 28.4 training cost results.",
        },
        {
            "chunk_id": f"{doc_id}_chunk_4",
            "document_id": doc_id,
            "chunk_index": 4,
            "start_page": 6,
            "end_page": 6,
            "text": "Restricted to fixed-length sequences limitation.",
        },
    ]

    def _make_citation(field_name, chunk_index, sufficiency="sufficient"):
        chunk = chunks[chunk_index]
        return {
            "field_name": field_name,
            "evidence_chunks": [chunk],
            "citations": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "start_page": chunk["start_page"],
                    "end_page": chunk["end_page"],
                }
            ],
            "evidence_type": "explicit_fact",
            "evidence_sufficiency": sufficiency,
        }

    return {
        "document_id": doc_id,
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "publication_year": "2017",
        "abstract": "The dominant sequence transduction models are based on complex recurrent networks.",
        "research_problem": "Improving sequence transduction without recurrence.",
        "methodology": "Transformer architecture with self-attention.",
        "dataset": "WMT 2014 English-German and English-French translation tasks.",
        "evaluation_metrics": ["BLEU", "training cost"],
        "reported_results": "28.4 BLEU on WMT 2014 English-to-German.",
        "limitations": ["Restricted to fixed-length sequences."],
        "evidence_citations": {
            "title":             _make_citation("title", 0),
            "authors":           _make_citation("authors", 0),
            "publication_year":  _make_citation("publication_year", 0),
            "abstract":          _make_citation("abstract", 0),
            "research_problem":  _make_citation("research_problem", 0),
            "methodology":       _make_citation("methodology", 1),
            "dataset":           _make_citation("dataset", 2),
            "evaluation_metrics": _make_citation("evaluation_metrics", 3),
            "reported_results":  _make_citation("reported_results", 3),
            "limitations":       _make_citation("limitations", 4),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Item 1 — Gold-set schema validator (including new fields)
# ─────────────────────────────────────────────────────────────────────────────

class TestGoldSetSchema:

    def test_valid_annotation_passes(self, valid_annotation):
        validate_annotation(valid_annotation, "test_paper")

    def test_missing_top_level_key_raises(self, valid_annotation):
        del valid_annotation["annotator"]
        with pytest.raises(GoldSetValidationError, match="annotator"):
            validate_annotation(valid_annotation, "test_paper")

    def test_document_id_mismatch_raises(self, valid_annotation):
        with pytest.raises(GoldSetValidationError, match="document_id"):
            validate_annotation(valid_annotation, "wrong_stem")

    def test_placeholder_annotator_raises(self, valid_annotation):
        valid_annotation["annotator"] = "TODO"
        with pytest.raises(GoldSetValidationError, match="placeholder"):
            validate_annotation(valid_annotation, "test_paper")

    def test_placeholder_field_value_raises(self, valid_annotation):
        valid_annotation["fields"]["title"]["value"] = "TODO"
        with pytest.raises(GoldSetValidationError, match="placeholder"):
            validate_annotation(valid_annotation, "test_paper")

    def test_placeholder_list_item_raises(self, valid_annotation):
        valid_annotation["fields"]["authors"]["value"] = ["TODO"]
        with pytest.raises(GoldSetValidationError, match="placeholder"):
            validate_annotation(valid_annotation, "test_paper")

    def test_null_value_allowed(self, valid_annotation):
        valid_annotation["fields"]["dataset"]["value"] = None
        validate_annotation(valid_annotation, "test_paper")

    def test_empty_list_value_rejected(self, valid_annotation):
        valid_annotation["fields"]["authors"]["value"] = []
        with pytest.raises(GoldSetValidationError, match="empty list"):
            validate_annotation(valid_annotation, "test_paper")

    def test_non_positive_evidence_page_raises(self, valid_annotation):
        valid_annotation["fields"]["title"]["evidence_pages"] = [0]
        with pytest.raises(GoldSetValidationError, match="positive integer"):
            validate_annotation(valid_annotation, "test_paper")

    def test_missing_all_fields_raises(self, valid_annotation):
        del valid_annotation["fields"]["title"]
        with pytest.raises(GoldSetValidationError, match="Missing field"):
            validate_annotation(valid_annotation, "test_paper")

    def test_extra_field_raises(self, valid_annotation):
        valid_annotation["fields"]["unknown_field"] = {"value": "x", "evidence_pages": []}
        with pytest.raises(GoldSetValidationError, match="Unknown fields"):
            validate_annotation(valid_annotation, "test_paper")

    def test_invalid_date_format_raises(self, valid_annotation):
        valid_annotation["annotation_date"] = "10/11/2026"
        with pytest.raises(GoldSetValidationError, match="ISO 8601"):
            validate_annotation(valid_annotation, "test_paper")

    def test_string_field_as_list_raises(self, valid_annotation):
        valid_annotation["fields"]["title"]["value"] = ["list", "not", "string"]
        with pytest.raises(GoldSetValidationError, match="string"):
            validate_annotation(valid_annotation, "test_paper")

    def test_list_field_as_string_raises(self, valid_annotation):
        valid_annotation["fields"]["authors"]["value"] = "Single Author String"
        with pytest.raises(GoldSetValidationError, match="list"):
            validate_annotation(valid_annotation, "test_paper")

    # Item 7 — split field
    def test_valid_split_dev(self, valid_annotation):
        valid_annotation["split"] = "dev"
        validate_annotation(valid_annotation, "test_paper")

    def test_valid_split_test(self, valid_annotation):
        valid_annotation["split"] = "test"
        validate_annotation(valid_annotation, "test_paper")

    def test_invalid_split_raises(self, valid_annotation):
        valid_annotation["split"] = "train"
        with pytest.raises(GoldSetValidationError, match="split"):
            validate_annotation(valid_annotation, "test_paper")

    def test_missing_split_raises(self, valid_annotation):
        del valid_annotation["split"]
        with pytest.raises(GoldSetValidationError, match="split"):
            validate_annotation(valid_annotation, "test_paper")

    # Item 8 — annotated_without_model_output
    def test_awmo_true_accepted(self, valid_annotation):
        valid_annotation["annotated_without_model_output"] = True
        validate_annotation(valid_annotation, "test_paper")

    def test_awmo_false_accepted(self, valid_annotation):
        valid_annotation["annotated_without_model_output"] = False
        validate_annotation(valid_annotation, "test_paper")

    def test_awmo_missing_raises(self, valid_annotation):
        del valid_annotation["annotated_without_model_output"]
        with pytest.raises(GoldSetValidationError, match="annotated_without_model_output"):
            validate_annotation(valid_annotation, "test_paper")

    def test_awmo_non_bool_raises(self, valid_annotation):
        valid_annotation["annotated_without_model_output"] = "yes"
        with pytest.raises(GoldSetValidationError, match="boolean"):
            validate_annotation(valid_annotation, "test_paper")

    def test_load_annotation_from_disk(self, valid_annotation, tmp_path):
        path = tmp_path / "test_paper.json"
        path.write_text(json.dumps(valid_annotation), encoding="utf-8")
        loaded = load_annotation(path)
        assert loaded["document_id"] == "test_paper"

    def test_load_gold_set_no_files_raises(self, tmp_path):
        with pytest.raises(GoldSetValidationError, match="No .json"):
            load_gold_set(tmp_path)

    def test_load_gold_set_multiple_files(self, valid_annotation, tmp_path):
        for suffix in ("_a", "_b"):
            ann = deepcopy(valid_annotation)
            ann["document_id"] = f"test_paper{suffix}"
            ann["source_pdf"] = f"validation/step5/papers/test_paper{suffix}.pdf"
            (tmp_path / f"test_paper{suffix}.json").write_text(
                json.dumps(ann), encoding="utf-8"
            )
        result = load_gold_set(tmp_path)
        assert len(result) == 2
        assert result[0]["document_id"] == "test_paper_a"

    def test_load_gold_set_stops_on_invalid_file(self, valid_annotation, tmp_path):
        good = deepcopy(valid_annotation)
        (tmp_path / "test_paper.json").write_text(json.dumps(good), encoding="utf-8")
        bad = deepcopy(valid_annotation)
        bad["document_id"] = "bad_paper"
        bad["source_pdf"] = "validation/step5/papers/bad_paper.pdf"
        bad["annotator"] = "TODO"
        (tmp_path / "bad_paper.json").write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(GoldSetValidationError):
            load_gold_set(tmp_path)

    # Item 7 — split filter in load_gold_set
    def test_load_gold_set_split_filter(self, valid_annotation, tmp_path):
        dev_ann = deepcopy(valid_annotation)
        dev_ann["split"] = "dev"
        (tmp_path / "test_paper.json").write_text(json.dumps(dev_ann), encoding="utf-8")

        test_ann = deepcopy(valid_annotation)
        test_ann["document_id"] = "test_paper2"
        test_ann["source_pdf"] = "validation/step5/papers/test_paper2.pdf"
        test_ann["split"] = "test"
        (tmp_path / "test_paper2.json").write_text(json.dumps(test_ann), encoding="utf-8")

        dev_only = load_gold_set(tmp_path, split="dev")
        assert len(dev_only) == 1
        assert dev_only[0]["split"] == "dev"

        test_only = load_gold_set(tmp_path, split="test")
        assert len(test_only) == 1
        assert test_only[0]["split"] == "test"


# ─────────────────────────────────────────────────────────────────────────────
# Template generator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTemplateGenerator:

    def test_build_template_has_all_fields(self):
        tmpl = build_template("my_paper", "validation/step5/papers/my_paper.pdf")
        assert tmpl["document_id"] == "my_paper"
        assert set(tmpl["fields"].keys()) == set(ALL_FIELDS)

    def test_build_template_has_split_and_awmo(self):
        tmpl = build_template("my_paper", "x.pdf")
        assert "split" in tmpl
        assert "annotated_without_model_output" in tmpl

    def test_build_template_values_are_todo(self):
        tmpl = build_template("x", "x.pdf")
        for fa in tmpl["fields"].values():
            v = fa["value"]
            if isinstance(v, list):
                assert all(item == "TODO" for item in v)
            else:
                assert v == "TODO"

    def test_build_template_fails_schema_validation(self):
        tmpl = build_template("my_paper", "validation/step5/papers/my_paper.pdf")
        with pytest.raises(GoldSetValidationError, match="placeholder"):
            validate_annotation(tmpl, "my_paper")


# ─────────────────────────────────────────────────────────────────────────────
# Token-F1 and normalization helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenF1:

    def test_identical_strings_score_one(self):
        assert _token_f1("BLEU score", "BLEU score") == pytest.approx(1.0)

    def test_completely_different_score_zero(self):
        assert _token_f1("apple orange", "cat dog fish") == pytest.approx(0.0)

    def test_partial_overlap_between_zero_and_one(self):
        score = _token_f1("BLEU ROUGE", "BLEU accuracy")
        assert 0.0 < score < 1.0

    def test_empty_predicted_scores_zero(self):
        assert _token_f1("", "BLEU") == pytest.approx(0.0)

    def test_both_empty_scores_one(self):
        assert _token_f1("", "") == pytest.approx(1.0)

    def test_stopwords_ignored(self):
        assert _token_f1("the BLEU score", "a BLEU score") == pytest.approx(1.0)

    # Legacy list helper (still used in tests that pre-date item 5)
    def test_list_token_f1_exact_match(self):
        assert _list_token_f1(["BLEU", "ROUGE"], ["BLEU", "ROUGE"]) == pytest.approx(1.0)

    def test_list_token_f1_partial_match(self):
        score = _list_token_f1(["BLEU"], ["BLEU", "ROUGE"])
        assert 0.0 < score < 1.0

    def test_list_token_f1_no_predicted(self):
        assert _list_token_f1(None, ["BLEU"]) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Item 5 — set-based metrics for list fields
# ─────────────────────────────────────────────────────────────────────────────

class TestSetMetrics:

    def test_exact_match_all_one(self):
        p, r, f1 = _set_metrics(["BLEU", "ROUGE"], ["BLEU", "ROUGE"])
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_no_overlap_all_zero(self):
        p, r, f1 = _set_metrics(["BLEU"], ["ROUGE"])
        assert p == pytest.approx(0.0)
        assert r == pytest.approx(0.0)
        assert f1 == pytest.approx(0.0)

    def test_partial_overlap(self):
        p, r, f1 = _set_metrics(["BLEU", "ROUGE"], ["BLEU", "F1"])
        assert p == pytest.approx(0.5)
        assert r == pytest.approx(0.5)
        assert f1 == pytest.approx(0.5)

    def test_empty_predicted_zero(self):
        p, r, f1 = _set_metrics([], ["BLEU"])
        assert p == pytest.approx(0.0)
        assert f1 == pytest.approx(0.0)

    def test_none_predicted_zero(self):
        p, r, f1 = _set_metrics(None, ["BLEU"])
        assert f1 == pytest.approx(0.0)

    def test_both_empty_all_one(self):
        p, r, f1 = _set_metrics([], [])
        assert f1 == pytest.approx(1.0)

    def test_case_insensitive(self):
        p, r, f1 = _set_metrics(["bleu"], ["BLEU"])
        assert f1 == pytest.approx(1.0)

    def test_punctuation_stripped(self):
        # "BLEU." vs "BLEU" should match after normalization
        p, r, f1 = _set_metrics(["BLEU."], ["BLEU"])
        assert f1 == pytest.approx(1.0)

    def test_normalize_item_lowercases_and_strips(self):
        assert _normalize_item("  BLEU.  ") == "bleu"

    def test_normalize_item_collapses_punctuation(self):
        assert _normalize_item("multi-head") == "multi head"

    def test_string_predicted_treated_as_single_item(self):
        # A string prediction on a list gold
        p, r, f1 = _set_metrics("BLEU", ["BLEU"])
        assert f1 == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# _is_empty helper
# ─────────────────────────────────────────────────────────────────────────────

class TestIsEmpty:

    def test_none_is_empty(self):
        assert _is_empty(None)

    def test_empty_string_is_empty(self):
        assert _is_empty("")

    def test_whitespace_only_is_empty(self):
        assert _is_empty("   ")

    def test_empty_list_is_empty(self):
        assert _is_empty([])

    def test_list_of_blanks_is_empty(self):
        assert _is_empty(["", "  "])

    def test_non_empty_string_not_empty(self):
        assert not _is_empty("BLEU")

    def test_non_empty_list_not_empty(self):
        assert not _is_empty(["BLEU"])


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_paper — string fields use token-F1, list fields use set metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluatePaper:

    def test_perfect_match_gives_high_scores(self, valid_annotation, valid_analysis):
        result = evaluate_paper(valid_annotation, valid_analysis)
        assert result.mean_token_f1 > 0.8
        assert result.mean_set_f1 is not None
        assert result.mean_set_f1 > 0.8

    # Item 5 — list fields use set metrics, not token-F1
    def test_list_field_has_set_metrics_not_token_f1(self, valid_annotation, valid_analysis):
        result = evaluate_paper(valid_annotation, valid_analysis)
        authors_result = next(r for r in result.field_results if r.field_name == "authors")
        assert authors_result.is_list_field
        assert authors_result.set_f1 is not None
        assert authors_result.token_f1 == pytest.approx(0.0)  # not used for list fields

    def test_string_field_has_token_f1_not_set_metrics(self, valid_annotation, valid_analysis):
        result = evaluate_paper(valid_annotation, valid_analysis)
        title_result = next(r for r in result.field_results if r.field_name == "title")
        assert not title_result.is_list_field
        assert title_result.token_f1 > 0.0
        assert title_result.set_f1 is None

    def test_null_gold_null_predicted_correct_null(self, valid_annotation, valid_analysis):
        valid_annotation["fields"]["dataset"]["value"] = None
        valid_analysis["dataset"] = None
        result = evaluate_paper(valid_annotation, valid_analysis)
        ds = next(r for r in result.field_results if r.field_name == "dataset")
        assert ds.gold_is_null
        assert ds.predicted_is_null
        assert ds.correct_null
        assert ds.token_f1 == pytest.approx(1.0)

    def test_null_gold_non_null_predicted_penalised(self, valid_annotation, valid_analysis):
        valid_annotation["fields"]["dataset"]["value"] = None
        valid_analysis["dataset"] = "Some spurious dataset"
        result = evaluate_paper(valid_annotation, valid_analysis)
        ds = next(r for r in result.field_results if r.field_name == "dataset")
        assert not ds.correct_null
        assert ds.token_f1 == pytest.approx(0.0)

    def test_all_null_predicted_gives_zero_f1(self, valid_annotation):
        from analysis.paper_analyzer import PaperAnalyzer
        empty_analysis = {
            "document_id": valid_annotation["document_id"],
            "evidence_citations": {},
            **{f: None for f in PaperAnalyzer.SCHEMA_FIELDS},
        }
        result = evaluate_paper(valid_annotation, empty_analysis)
        for fr in result.field_results:
            if not fr.gold_is_null:
                if fr.is_list_field:
                    assert fr.set_f1 == pytest.approx(0.0)
                else:
                    assert fr.token_f1 == pytest.approx(0.0)

    # Item 4 — hit-rate@k for k=1,3,5
    def test_retrieval_hit_at_k1_when_first_chunk_matches(
        self, valid_annotation, valid_analysis
    ):
        result = evaluate_paper(valid_annotation, valid_analysis)
        title = next(r for r in result.field_results if r.field_name == "title")
        assert title.retrieval_hits[1] is True

    def test_retrieval_hit_at_k3_when_first_chunk_matches(
        self, valid_annotation, valid_analysis
    ):
        result = evaluate_paper(valid_annotation, valid_analysis)
        title = next(r for r in result.field_results if r.field_name == "title")
        assert title.retrieval_hits[3] is True

    def test_retrieval_miss_at_all_k_when_no_overlap(self, valid_annotation, valid_analysis):
        # Move title citation chunk to page 99
        cit = valid_analysis["evidence_citations"]["title"]
        cit["evidence_chunks"][0] = {**cit["evidence_chunks"][0], "start_page": 99, "end_page": 99}
        cit["citations"][0] = {**cit["citations"][0], "start_page": 99, "end_page": 99}
        result = evaluate_paper(valid_annotation, valid_analysis)
        title = next(r for r in result.field_results if r.field_name == "title")
        for k in [1, 3, 5]:
            assert title.retrieval_hits[k] is False

    def test_retrieval_hit_none_when_no_evidence_pages(self, valid_annotation, valid_analysis):
        valid_annotation["fields"]["title"]["evidence_pages"] = []
        result = evaluate_paper(valid_annotation, valid_analysis)
        title = next(r for r in result.field_results if r.field_name == "title")
        for k in [1, 3, 5]:
            assert title.retrieval_hits[k] is None

    # Item 4 — MRR
    def test_mrr_is_one_when_first_chunk_hits(self, valid_annotation, valid_analysis):
        result = evaluate_paper(valid_annotation, valid_analysis)
        title = next(r for r in result.field_results if r.field_name == "title")
        assert title.mrr == pytest.approx(1.0)

    def test_mrr_is_zero_when_no_hit(self, valid_annotation, valid_analysis):
        cit = valid_analysis["evidence_citations"]["title"]
        cit["evidence_chunks"][0] = {**cit["evidence_chunks"][0], "start_page": 99, "end_page": 99}
        result = evaluate_paper(valid_annotation, valid_analysis)
        title = next(r for r in result.field_results if r.field_name == "title")
        assert title.mrr == pytest.approx(0.0)

    def test_mrr_none_when_no_evidence_pages(self, valid_annotation, valid_analysis):
        valid_annotation["fields"]["title"]["evidence_pages"] = []
        result = evaluate_paper(valid_annotation, valid_analysis)
        title = next(r for r in result.field_results if r.field_name == "title")
        assert title.mrr is None

    # Item 2 — citation traceability (renamed)
    def test_citation_traceable_when_page_in_annotated_pages(
        self, valid_annotation, valid_analysis
    ):
        result = evaluate_paper(valid_annotation, valid_analysis)
        title = next(r for r in result.field_results if r.field_name == "title")
        assert title.citation_traceable is True

    def test_citation_not_traceable_when_insufficient_sufficiency(
        self, valid_annotation, valid_analysis
    ):
        valid_analysis["evidence_citations"]["title"]["evidence_sufficiency"] = "insufficient"
        result = evaluate_paper(valid_annotation, valid_analysis)
        title = next(r for r in result.field_results if r.field_name == "title")
        assert title.citation_traceable is False

    def test_citation_traceability_none_for_null_gold(self, valid_annotation, valid_analysis):
        valid_annotation["fields"]["dataset"]["value"] = None
        valid_analysis["dataset"] = None
        result = evaluate_paper(valid_annotation, valid_analysis)
        ds = next(r for r in result.field_results if r.field_name == "dataset")
        assert ds.citation_traceable is None

    def test_citation_not_traceable_when_no_citations(self, valid_annotation, valid_analysis):
        valid_analysis["evidence_citations"]["title"]["citations"] = []
        result = evaluate_paper(valid_annotation, valid_analysis)
        title = next(r for r in result.field_results if r.field_name == "title")
        assert title.citation_traceable is False

    def test_missing_evidence_citations_key_does_not_crash(self, valid_annotation, valid_analysis):
        del valid_analysis["evidence_citations"]
        result = evaluate_paper(valid_annotation, valid_analysis)
        assert isinstance(result, PaperResult)

    # PaperResult aggregation
    def test_paper_result_hit_rate_at_k(self, valid_annotation, valid_analysis):
        result = evaluate_paper(valid_annotation, valid_analysis)
        for k in [1, 3, 5]:
            hr = result.hit_rate_at_k(k)
            assert hr is not None
            assert 0.0 <= hr <= 1.0

    def test_paper_result_mrr(self, valid_annotation, valid_analysis):
        result = evaluate_paper(valid_annotation, valid_analysis)
        assert result.mrr is not None
        assert result.mrr > 0.0

    def test_paper_result_citation_traceability_rate(self, valid_annotation, valid_analysis):
        result = evaluate_paper(valid_annotation, valid_analysis)
        assert result.citation_traceability_rate is not None
        assert result.citation_traceability_rate > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# HarnessResult aggregation and to_dict
# ─────────────────────────────────────────────────────────────────────────────

class TestHarnessResult:

    def _make_field_result(
        self, name: str, is_list: bool, f1: float, hit: bool, cited: bool
    ) -> FieldResult:
        hits = {k: hit for k in _DEFAULT_K_VALUES}
        return FieldResult(
            field_name=name,
            is_list_field=is_list,
            token_f1=0.0 if is_list else f1,
            set_precision=f1 if is_list else None,
            set_recall=f1 if is_list else None,
            set_f1=f1 if is_list else None,
            gold_is_null=False,
            predicted_is_null=False,
            correct_null=False,
            retrieval_hits=hits,
            retrieved_pages=[1],
            annotated_pages=[1],
            mrr=1.0 if hit else 0.0,
            citation_traceable=cited,
        )

    def _make_paper_result(self, doc_id: str, f1: float, hit: bool, cited: bool) -> PaperResult:
        # one string field, one list field
        frs = [
            self._make_field_result("title", False, f1, hit, cited),
            self._make_field_result("authors", True, f1, hit, cited),
        ]
        return PaperResult(document_id=doc_id, field_results=frs)

    def test_macro_averages_over_papers(self):
        pr1 = self._make_paper_result("p1", 1.0, True, True)
        pr2 = self._make_paper_result("p2", 0.0, False, False)
        h = HarnessResult(paper_results=[pr1, pr2])
        assert h.macro_token_f1 == pytest.approx(0.5)
        assert h.macro_set_f1 == pytest.approx(0.5)
        assert h.macro_hit_rate(3) == pytest.approx(0.5)
        assert h.macro_mrr == pytest.approx(0.5)
        assert h.macro_citation_traceability_rate == pytest.approx(0.5)

    def test_to_dict_has_expected_keys(self):
        pr = self._make_paper_result("p1", 0.8, True, True)
        h = HarnessResult(paper_results=[pr])
        d = h.to_dict()
        assert "summary" in d
        assert "per_paper" in d
        assert "k_values" in d
        assert "footnotes" in d

    # Item 2 — renamed key in report
    def test_to_dict_uses_citation_traceability_key(self):
        pr = self._make_paper_result("p1", 0.9, True, True)
        h = HarnessResult(paper_results=[pr])
        d = h.to_dict()
        assert "macro_citation_traceability_rate" in d["summary"]
        assert "macro_citation_support_rate" not in d["summary"]

    def test_to_dict_footnote_states_no_semantic_verification(self):
        pr = self._make_paper_result("p1", 0.9, True, True)
        h = HarnessResult(paper_results=[pr])
        d = h.to_dict()
        footnote = d["footnotes"]["citation_traceability_rate"]
        assert "DOES NOT VERIFY" in footnote or "does not verify" in footnote.lower()

    # Item 4 — multiple k values in output
    def test_to_dict_has_hit_rate_for_all_k(self):
        pr = self._make_paper_result("p1", 0.8, True, True)
        h = HarnessResult(paper_results=[pr])
        d = h.to_dict()
        for k in [1, 3, 5]:
            assert f"macro_hit_rate@{k}" in d["summary"]
        assert "macro_mrr" in d["summary"]

    def test_no_papers_macro_f1_is_zero(self):
        h = HarnessResult(paper_results=[])
        assert h.macro_token_f1 == pytest.approx(0.0)

    def test_hit_rate_none_when_no_evidence_pages(self):
        fr = FieldResult(
            field_name="title", is_list_field=False, token_f1=0.9,
            set_precision=None, set_recall=None, set_f1=None,
            gold_is_null=False, predicted_is_null=False, correct_null=False,
            retrieval_hits={1: None, 3: None, 5: None},
            retrieved_pages=[], annotated_pages=[],
            mrr=None, citation_traceable=True,
        )
        pr = PaperResult(document_id="p1", field_results=[fr])
        h = HarnessResult(paper_results=[pr])
        assert h.macro_hit_rate(3) is None
        assert h.macro_mrr is None


# ─────────────────────────────────────────────────────────────────────────────
# Item 3 — Citation sample schema and agreement metric
# ─────────────────────────────────────────────────────────────────────────────

class TestCitationSampleSchema:

    @pytest.fixture()
    def valid_sample(self) -> Dict[str, Any]:
        return {
            "document_id": "test_paper",
            "annotator": "Test Annotator",
            "annotation_date": "2026-11-10",
            "annotated_without_model_output": True,
            "notes": None,
            "pairs": [
                {
                    "pair_id": "title_p1",
                    "field_name": "title",
                    "claim": "Attention Is All You Need",
                    "passage": "Attention Is All You Need Ashish Vaswani",
                    "chunk_id": "test_paper_chunk_0",
                    "page": 1,
                    "label": "supported",
                },
                {
                    "pair_id": "abstract_p1",
                    "field_name": "abstract",
                    "claim": "Models based on recurrent networks",
                    "passage": "Some unrelated text about convolutions",
                    "chunk_id": "test_paper_chunk_5",
                    "page": 2,
                    "label": "unsupported",
                },
            ],
        }

    def test_valid_sample_passes(self, valid_sample):
        validate_citation_sample(valid_sample, "test_paper")

    def test_missing_key_raises(self, valid_sample):
        del valid_sample["annotator"]
        with pytest.raises(CitationSampleValidationError, match="annotator"):
            validate_citation_sample(valid_sample, "test_paper")

    def test_placeholder_annotator_raises(self, valid_sample):
        valid_sample["annotator"] = "TODO"
        with pytest.raises(CitationSampleValidationError, match="placeholder"):
            validate_citation_sample(valid_sample, "test_paper")

    def test_duplicate_pair_id_raises(self, valid_sample):
        valid_sample["pairs"][1]["pair_id"] = valid_sample["pairs"][0]["pair_id"]
        with pytest.raises(CitationSampleValidationError, match="unique"):
            validate_citation_sample(valid_sample, "test_paper")

    def test_invalid_label_raises(self, valid_sample):
        valid_sample["pairs"][0]["label"] = "maybe"
        with pytest.raises(CitationSampleValidationError, match="label"):
            validate_citation_sample(valid_sample, "test_paper")

    def test_invalid_field_name_raises(self, valid_sample):
        valid_sample["pairs"][0]["field_name"] = "not_a_field"
        with pytest.raises(CitationSampleValidationError, match="field_name"):
            validate_citation_sample(valid_sample, "test_paper")

    def test_placeholder_claim_raises(self, valid_sample):
        valid_sample["pairs"][0]["claim"] = "TODO"
        with pytest.raises(CitationSampleValidationError, match="placeholder"):
            validate_citation_sample(valid_sample, "test_paper")

    def test_non_positive_page_raises(self, valid_sample):
        valid_sample["pairs"][0]["page"] = 0
        with pytest.raises(CitationSampleValidationError, match="positive integer"):
            validate_citation_sample(valid_sample, "test_paper")

    def test_null_page_allowed(self, valid_sample):
        valid_sample["pairs"][0]["page"] = None
        validate_citation_sample(valid_sample, "test_paper")

    def test_empty_pairs_raises(self, valid_sample):
        valid_sample["pairs"] = []
        with pytest.raises(CitationSampleValidationError, match="at least one"):
            validate_citation_sample(valid_sample, "test_paper")

    def test_build_template_has_placeholder_pair_ids(self):
        tmpl = build_citation_sample_template("my_paper", num_pairs=3)
        assert len(tmpl["pairs"]) == 3
        for pair in tmpl["pairs"]:
            assert "TODO" in pair["pair_id"]

    def test_build_template_fails_validation(self):
        tmpl = build_citation_sample_template("my_paper", num_pairs=2)
        with pytest.raises(CitationSampleValidationError):
            validate_citation_sample(tmpl, "my_paper")


class TestCitationAgreement:

    @pytest.fixture()
    def sample_with_supported(self, valid_analysis) -> Dict[str, Any]:
        """Sample with one 'supported' pair for the title field."""
        return {
            "document_id": valid_analysis["document_id"],
            "annotator": "Test",
            "annotation_date": "2026-11-10",
            "annotated_without_model_output": True,
            "notes": None,
            "pairs": [
                {
                    "pair_id": "title_p1",
                    "field_name": "title",
                    "claim": "Attention Is All You Need",
                    "passage": "Attention Is All You Need Ashish Vaswani",
                    "chunk_id": "test_paper_chunk_0",
                    "page": 1,
                    "label": "supported",
                },
            ],
        }

    def test_agreement_tp_when_both_traceable(self, sample_with_supported, valid_analysis):
        # Title citation is traceable (page 1 in annotated, sufficient sufficiency)
        result = compute_citation_agreement(sample_with_supported, valid_analysis)
        assert result["confusion"]["tp"] == 1
        assert result["confusion"]["fp"] == 0
        assert result["confusion"]["fn"] == 0
        assert result["confusion"]["tn"] == 0
        assert result["agreement_rate"] == pytest.approx(1.0)

    def test_agreement_fn_when_auto_not_traceable_human_supported(
        self, sample_with_supported, valid_analysis
    ):
        # Make title citation not traceable
        valid_analysis["evidence_citations"]["title"]["evidence_sufficiency"] = "insufficient"
        result = compute_citation_agreement(sample_with_supported, valid_analysis)
        assert result["confusion"]["fn"] == 1
        assert result["agreement_rate"] == pytest.approx(0.0)

    def test_partially_supported_maps_to_traceable(self, sample_with_supported, valid_analysis):
        sample_with_supported["pairs"][0]["label"] = "partially_supported"
        result = compute_citation_agreement(sample_with_supported, valid_analysis)
        # partially_supported → traceable; auto is also traceable → TP
        assert result["confusion"]["tp"] == 1

    def test_unsupported_maps_to_not_traceable(self, sample_with_supported, valid_analysis):
        sample_with_supported["pairs"][0]["label"] = "unsupported"
        # auto is traceable (page match, sufficient) → FP
        result = compute_citation_agreement(sample_with_supported, valid_analysis)
        assert result["confusion"]["fp"] == 1

    def test_footnote_present_in_result(self, sample_with_supported, valid_analysis):
        result = compute_citation_agreement(sample_with_supported, valid_analysis)
        assert "footnote" in result
        assert len(result["footnote"]) > 20


# ─────────────────────────────────────────────────────────────────────────────
# Harness runner (dry-run) integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHarnessRunnerDryRun:

    def _write_annotation(self, path: Path, annotation: dict) -> None:
        path.write_text(json.dumps(annotation), encoding="utf-8")

    def test_dry_run_produces_report_json_and_md(self, valid_annotation, tmp_path):
        gold_dir = tmp_path / "gold_set"
        gold_dir.mkdir()
        self._write_annotation(gold_dir / "test_paper.json", valid_annotation)
        output_dir = tmp_path / "output"

        import sys
        args = [
            "run_harness.py",
            "--gold-dir", str(gold_dir),
            "--output-dir", str(output_dir),
            "--dry-run",
        ]
        with patch.object(sys, "argv", args):
            from evaluation.run_harness import main
            main()

        assert (output_dir / "report.json").exists()
        assert (output_dir / "report.md").exists()

        with open(output_dir / "report.json", encoding="utf-8") as fh:
            report = json.load(fh)
        assert report["aggregate_summary"]["num_papers"] == 1

    def test_dry_run_all_metrics_zero_for_non_null_fields(self, valid_annotation, tmp_path):
        gold_dir = tmp_path / "gold_set"
        gold_dir.mkdir()
        self._write_annotation(gold_dir / "test_paper.json", valid_annotation)
        output_dir = tmp_path / "output"

        import sys
        args = [
            "run_harness.py",
            "--gold-dir", str(gold_dir),
            "--output-dir", str(output_dir),
            "--dry-run",
        ]
        with patch.object(sys, "argv", args):
            from evaluation.run_harness import main
            main()

        with open(output_dir / "report.json", encoding="utf-8") as fh:
            report = json.load(fh)
        pp = report["runs"][0]["per_paper"][0]
        assert pp["mean_token_f1"] == pytest.approx(0.0)

    def test_dry_run_no_groq_key_required(self, valid_annotation, tmp_path, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        gold_dir = tmp_path / "gold_set"
        gold_dir.mkdir()
        self._write_annotation(gold_dir / "test_paper.json", valid_annotation)
        output_dir = tmp_path / "output"

        import sys
        args = [
            "run_harness.py",
            "--gold-dir", str(gold_dir),
            "--output-dir", str(output_dir),
            "--dry-run",
        ]
        with patch.object(sys, "argv", args):
            from evaluation.run_harness import main
            main()  # Must not raise

    def test_requires_confirm_costs_without_dry_run(self, valid_annotation, tmp_path):
        gold_dir = tmp_path / "gold_set"
        gold_dir.mkdir()
        self._write_annotation(gold_dir / "test_paper.json", valid_annotation)

        import sys
        args = [
            "run_harness.py",
            "--gold-dir", str(gold_dir),
            "--output-dir", str(tmp_path / "out"),
        ]
        with patch.object(sys, "argv", args):
            from evaluation.run_harness import main
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code != 0

    def test_invalid_gold_set_exits_with_error(self, valid_annotation, tmp_path):
        gold_dir = tmp_path / "gold_set"
        gold_dir.mkdir()
        bad = deepcopy(valid_annotation)
        bad["annotator"] = "TODO"
        self._write_annotation(gold_dir / "test_paper.json", bad)

        import sys
        args = [
            "run_harness.py",
            "--gold-dir", str(gold_dir),
            "--output-dir", str(tmp_path / "out"),
            "--dry-run",
        ]
        with patch.object(sys, "argv", args):
            from evaluation.run_harness import main
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code != 0

    def test_document_id_filter(self, valid_annotation, tmp_path):
        gold_dir = tmp_path / "gold_set"
        gold_dir.mkdir()
        self._write_annotation(gold_dir / "test_paper.json", valid_annotation)
        second = deepcopy(valid_annotation)
        second["document_id"] = "other_paper"
        second["source_pdf"] = "validation/step5/papers/other_paper.pdf"
        self._write_annotation(gold_dir / "other_paper.json", second)

        output_dir = tmp_path / "output"
        import sys
        args = [
            "run_harness.py",
            "--gold-dir", str(gold_dir),
            "--output-dir", str(output_dir),
            "--dry-run",
            "--document-id", "test_paper",
        ]
        with patch.object(sys, "argv", args):
            from evaluation.run_harness import main
            main()

        with open(output_dir / "report.json", encoding="utf-8") as fh:
            report = json.load(fh)
        assert report["aggregate_summary"]["num_papers"] == 1
        assert report["runs"][0]["per_paper"][0]["document_id"] == "test_paper"

    # Item 6 — multiple runs
    def test_multiple_runs_produce_aggregate(self, valid_annotation, tmp_path):
        gold_dir = tmp_path / "gold_set"
        gold_dir.mkdir()
        self._write_annotation(gold_dir / "test_paper.json", valid_annotation)
        output_dir = tmp_path / "output"

        import sys
        args = [
            "run_harness.py",
            "--gold-dir", str(gold_dir),
            "--output-dir", str(output_dir),
            "--dry-run",
            "--runs", "3",
        ]
        with patch.object(sys, "argv", args):
            from evaluation.run_harness import main
            main()

        with open(output_dir / "report.json", encoding="utf-8") as fh:
            report = json.load(fh)
        assert report["num_runs"] == 3
        assert len(report["runs"]) == 3
        # With >1 run, aggregate values are dicts with mean/min/max
        agg = report["aggregate_summary"]
        assert isinstance(agg["macro_token_f1"], dict)
        assert "mean" in agg["macro_token_f1"]
        assert "min" in agg["macro_token_f1"]
        assert "max" in agg["macro_token_f1"]

    # Item 7 — split filter
    def test_split_filter_dev_only(self, valid_annotation, tmp_path):
        gold_dir = tmp_path / "gold_set"
        gold_dir.mkdir()
        dev_ann = deepcopy(valid_annotation)
        dev_ann["split"] = "dev"
        self._write_annotation(gold_dir / "test_paper.json", dev_ann)

        test_ann = deepcopy(valid_annotation)
        test_ann["document_id"] = "test_paper2"
        test_ann["source_pdf"] = "validation/step5/papers/test_paper2.pdf"
        test_ann["split"] = "test"
        self._write_annotation(gold_dir / "test_paper2.json", test_ann)

        output_dir = tmp_path / "output"
        import sys
        args = [
            "run_harness.py",
            "--gold-dir", str(gold_dir),
            "--output-dir", str(output_dir),
            "--dry-run",
            "--split", "dev",
        ]
        with patch.object(sys, "argv", args):
            from evaluation.run_harness import main
            main()

        with open(output_dir / "report.json", encoding="utf-8") as fh:
            report = json.load(fh)
        assert report["split"] == "dev"
        assert report["aggregate_summary"]["num_papers"] == 1

    def test_test_split_writes_audit_log(self, valid_annotation, tmp_path, monkeypatch):
        """Test-split runs must write an entry to test_split_audit.log."""
        gold_dir = tmp_path / "gold_set"
        gold_dir.mkdir()
        test_ann = deepcopy(valid_annotation)
        test_ann["split"] = "test"
        self._write_annotation(gold_dir / "test_paper.json", test_ann)

        output_dir = tmp_path / "output"
        audit_log = tmp_path / "audit.log"

        import sys
        # Patch the audit log path
        import evaluation.run_harness as rh
        monkeypatch.setattr(rh, "_AUDIT_LOG", audit_log)

        args = [
            "run_harness.py",
            "--gold-dir", str(gold_dir),
            "--output-dir", str(output_dir),
            "--dry-run",
            "--split", "test",
        ]
        with patch.object(sys, "argv", args):
            main_fn = rh.main
            main_fn()

        assert audit_log.exists()
        content = audit_log.read_text(encoding="utf-8")
        assert str(output_dir) in content

    # Item 8 — annotation independence summary in report
    def test_report_includes_annotation_independence_summary(
        self, valid_annotation, tmp_path
    ):
        valid_annotation["annotated_without_model_output"] = True
        gold_dir = tmp_path / "gold_set"
        gold_dir.mkdir()
        self._write_annotation(gold_dir / "test_paper.json", valid_annotation)
        output_dir = tmp_path / "output"

        import sys
        args = [
            "run_harness.py",
            "--gold-dir", str(gold_dir),
            "--output-dir", str(output_dir),
            "--dry-run",
        ]
        with patch.object(sys, "argv", args):
            from evaluation.run_harness import main
            main()

        with open(output_dir / "report.json", encoding="utf-8") as fh:
            report = json.load(fh)
        ai = report["annotation_independence"]
        assert ai["total"] == 1
        assert ai["without_model_output_count"] == 1
        assert ai["with_model_output_count"] == 0

    # Item 9 — report contains split and num_runs
    def test_report_json_has_split_and_num_runs(self, valid_annotation, tmp_path):
        gold_dir = tmp_path / "gold_set"
        gold_dir.mkdir()
        self._write_annotation(gold_dir / "test_paper.json", valid_annotation)
        output_dir = tmp_path / "output"

        import sys
        args = [
            "run_harness.py",
            "--gold-dir", str(gold_dir),
            "--output-dir", str(output_dir),
            "--dry-run",
            "--split", "dev",
        ]
        with patch.object(sys, "argv", args):
            from evaluation.run_harness import main
            main()

        with open(output_dir / "report.json", encoding="utf-8") as fh:
            report = json.load(fh)
        assert report["split"] == "dev"
        assert "num_runs" in report
        assert "generated_at" in report
