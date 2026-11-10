"""Tests for Phase 1 evaluation harness: schema validator, metrics, and runner.

All tests are offline (no LLM or network calls).  The conftest autouse fixture
already blocks external services; these tests additionally mock PaperAnalyzer
where needed.
"""

from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from evaluation.gold_set_schema import (
    ALL_FIELDS,
    GoldSetValidationError,
    load_annotation,
    load_gold_set,
    validate_annotation,
)
from evaluation.generate_template import build_template
from evaluation.metrics import (
    FieldResult,
    HarnessResult,
    PaperResult,
    _is_empty,
    _list_token_f1,
    _token_f1,
    evaluate_paper,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def valid_annotation() -> Dict[str, Any]:
    """A minimal but fully valid annotation for testing."""
    return {
        "document_id": "test_paper",
        "source_pdf": "validation/step5/papers/test_paper.pdf",
        "annotator": "Test Annotator",
        "annotation_date": "2026-11-10",
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
            "evaluation_metrics":_make_citation("evaluation_metrics", 3),
            "reported_results":  _make_citation("reported_results", 3),
            "limitations":       _make_citation("limitations", 4),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Schema validator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGoldSetSchema:

    def test_valid_annotation_passes(self, valid_annotation):
        """A fully valid annotation raises no errors."""
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
        """dataset is optional; null value must be accepted."""
        valid_annotation["fields"]["dataset"]["value"] = None
        validate_annotation(valid_annotation, "test_paper")

    def test_empty_list_value_rejected(self, valid_annotation):
        """An empty list is not allowed; use null instead."""
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

    def test_load_annotation_from_disk(self, valid_annotation, tmp_path):
        """load_annotation reads, parses, and validates a file."""
        path = tmp_path / "test_paper.json"
        path.write_text(json.dumps(valid_annotation), encoding="utf-8")
        loaded = load_annotation(path)
        assert loaded["document_id"] == "test_paper"

    def test_load_gold_set_no_files_raises(self, tmp_path):
        """load_gold_set raises when the directory has no .json files."""
        with pytest.raises(GoldSetValidationError, match="No .json"):
            load_gold_set(tmp_path)

    def test_load_gold_set_multiple_files(self, valid_annotation, tmp_path):
        """load_gold_set loads all valid files alphabetically."""
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
        """load_gold_set raises immediately when any file is invalid."""
        # write one good file
        good = deepcopy(valid_annotation)
        good["document_id"] = "test_paper"
        (tmp_path / "test_paper.json").write_text(json.dumps(good), encoding="utf-8")
        # write one bad file (placeholder annotator)
        bad = deepcopy(valid_annotation)
        bad["document_id"] = "bad_paper"
        bad["source_pdf"] = "validation/step5/papers/bad_paper.pdf"
        bad["annotator"] = "TODO"
        (tmp_path / "bad_paper.json").write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(GoldSetValidationError):
            load_gold_set(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Template generator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTemplateGenerator:

    def test_build_template_has_all_fields(self):
        tmpl = build_template("my_paper", "validation/step5/papers/my_paper.pdf")
        assert tmpl["document_id"] == "my_paper"
        assert set(tmpl["fields"].keys()) == set(ALL_FIELDS)

    def test_build_template_values_are_todo(self):
        tmpl = build_template("x", "x.pdf")
        for fa in tmpl["fields"].values():
            v = fa["value"]
            if isinstance(v, list):
                assert all(item == "TODO" for item in v)
            else:
                assert v == "TODO"

    def test_build_template_fails_schema_validation(self):
        """The generated template must FAIL validation (all TODO values)."""
        tmpl = build_template("my_paper", "validation/step5/papers/my_paper.pdf")
        with pytest.raises(GoldSetValidationError, match="placeholder"):
            validate_annotation(tmpl, "my_paper")


# ─────────────────────────────────────────────────────────────────────────────
# Token-F1 metric tests
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
        # "the" and "a" are stopwords; score should equal 1.0
        assert _token_f1("the BLEU score", "a BLEU score") == pytest.approx(1.0)

    def test_list_token_f1_exact_match(self):
        assert _list_token_f1(["BLEU", "ROUGE"], ["BLEU", "ROUGE"]) == pytest.approx(1.0)

    def test_list_token_f1_partial_match(self):
        score = _list_token_f1(["BLEU"], ["BLEU", "ROUGE"])
        assert 0.0 < score < 1.0

    def test_list_token_f1_no_predicted(self):
        assert _list_token_f1(None, ["BLEU"]) == pytest.approx(0.0)

    def test_list_token_f1_empty_predicted(self):
        assert _list_token_f1([], ["BLEU"]) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# _is_empty helper tests
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
# evaluate_paper metric tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluatePaper:

    def test_perfect_match_gives_high_f1(self, valid_annotation, valid_analysis):
        result = evaluate_paper(valid_annotation, valid_analysis, top_k=3)
        assert result.mean_token_f1 > 0.8

    def test_null_gold_null_predicted_correct_null(self, valid_annotation, valid_analysis):
        valid_annotation["fields"]["dataset"]["value"] = None
        valid_analysis["dataset"] = None
        result = evaluate_paper(valid_annotation, valid_analysis, top_k=3)
        ds_result = next(r for r in result.field_results if r.field_name == "dataset")
        assert ds_result.gold_is_null
        assert ds_result.predicted_is_null
        assert ds_result.correct_null
        assert ds_result.token_f1 == pytest.approx(1.0)

    def test_null_gold_non_null_predicted_penalised(self, valid_annotation, valid_analysis):
        valid_annotation["fields"]["dataset"]["value"] = None
        valid_analysis["dataset"] = "Some dataset that wasn't in the paper"
        result = evaluate_paper(valid_annotation, valid_analysis, top_k=3)
        ds_result = next(r for r in result.field_results if r.field_name == "dataset")
        assert ds_result.gold_is_null
        assert not ds_result.predicted_is_null
        assert not ds_result.correct_null
        assert ds_result.token_f1 == pytest.approx(0.0)

    def test_all_null_predicted_gives_zero_f1_for_non_null_fields(
        self, valid_annotation
    ):
        """An analysis with all fields null should score 0 on non-null gold fields."""
        empty_analysis = {
            "document_id": valid_annotation["document_id"],
            "evidence_citations": {},
            **{f: None for f in ALL_FIELDS},
        }
        result = evaluate_paper(valid_annotation, empty_analysis, top_k=3)
        # All gold fields are non-null, so token_f1 should be 0 for every field
        for fr in result.field_results:
            if not fr.gold_is_null:
                assert fr.token_f1 == pytest.approx(0.0)

    def test_retrieval_hit_when_pages_overlap(self, valid_annotation, valid_analysis):
        """title evidence_pages=[1]; analysis citation chunk has start_page=1 → hit."""
        result = evaluate_paper(valid_annotation, valid_analysis, top_k=3)
        title_result = next(r for r in result.field_results if r.field_name == "title")
        assert title_result.retrieval_hit is True

    def test_retrieval_miss_when_no_page_overlap(self, valid_annotation, valid_analysis):
        """Move citation chunk to page 99; annotated page is 1 → miss."""
        citation = valid_analysis["evidence_citations"]["title"]
        citation["evidence_chunks"][0] = {
            **citation["evidence_chunks"][0],
            "start_page": 99,
            "end_page": 99,
        }
        citation["citations"][0] = {
            **citation["citations"][0],
            "start_page": 99,
            "end_page": 99,
        }
        result = evaluate_paper(valid_annotation, valid_analysis, top_k=3)
        title_result = next(r for r in result.field_results if r.field_name == "title")
        assert title_result.retrieval_hit is False

    def test_retrieval_hit_none_when_no_evidence_pages_annotated(
        self, valid_annotation, valid_analysis
    ):
        """Fields with no evidence_pages annotation have retrieval_hit=None."""
        valid_annotation["fields"]["title"]["evidence_pages"] = []
        result = evaluate_paper(valid_annotation, valid_analysis, top_k=3)
        title_result = next(r for r in result.field_results if r.field_name == "title")
        assert title_result.retrieval_hit is None

    def test_citation_supported_when_page_in_annotated_pages(
        self, valid_annotation, valid_analysis
    ):
        result = evaluate_paper(valid_annotation, valid_analysis, top_k=3)
        title_result = next(r for r in result.field_results if r.field_name == "title")
        assert title_result.citation_supported is True

    def test_citation_not_supported_when_insufficient_sufficiency(
        self, valid_annotation, valid_analysis
    ):
        valid_analysis["evidence_citations"]["title"]["evidence_sufficiency"] = "insufficient"
        result = evaluate_paper(valid_annotation, valid_analysis, top_k=3)
        title_result = next(r for r in result.field_results if r.field_name == "title")
        assert title_result.citation_supported is False

    def test_citation_support_none_for_null_gold_fields(
        self, valid_annotation, valid_analysis
    ):
        valid_annotation["fields"]["dataset"]["value"] = None
        valid_analysis["dataset"] = None
        result = evaluate_paper(valid_annotation, valid_analysis, top_k=3)
        ds_result = next(r for r in result.field_results if r.field_name == "dataset")
        assert ds_result.citation_supported is None

    def test_citation_not_supported_when_no_citations(
        self, valid_annotation, valid_analysis
    ):
        valid_analysis["evidence_citations"]["title"]["citations"] = []
        result = evaluate_paper(valid_annotation, valid_analysis, top_k=3)
        title_result = next(r for r in result.field_results if r.field_name == "title")
        assert title_result.citation_supported is False

    def test_missing_evidence_citations_key(self, valid_annotation, valid_analysis):
        """analysis without evidence_citations key doesn't crash."""
        del valid_analysis["evidence_citations"]
        result = evaluate_paper(valid_annotation, valid_analysis, top_k=3)
        assert isinstance(result, PaperResult)

    def test_top_k_limits_chunks_considered(self, valid_annotation, valid_analysis):
        """top_k=1 considers only the first chunk — page hit depends on it."""
        # title chunk is page 1; annotated page is 1 → still a hit with k=1
        result_k1 = evaluate_paper(valid_annotation, valid_analysis, top_k=1)
        title_k1 = next(r for r in result_k1.field_results if r.field_name == "title")
        assert title_k1.retrieval_hit is True


# ─────────────────────────────────────────────────────────────────────────────
# HarnessResult aggregation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHarnessResult:

    def _make_paper_result(self, doc_id: str, f1: float, hit: bool, cited: bool) -> PaperResult:
        fr = FieldResult(
            field_name="title",
            token_f1=f1,
            gold_is_null=False,
            predicted_is_null=False,
            correct_null=False,
            retrieval_hit=hit,
            annotated_pages=[1],
            retrieved_pages=[1],
            citation_supported=cited,
        )
        return PaperResult(document_id=doc_id, field_results=[fr])

    def test_macro_averages_over_papers(self):
        pr1 = self._make_paper_result("p1", 1.0, True, True)
        pr2 = self._make_paper_result("p2", 0.0, False, False)
        harness = HarnessResult(paper_results=[pr1, pr2], top_k=3)
        assert harness.macro_token_f1 == pytest.approx(0.5)
        assert harness.macro_hit_rate == pytest.approx(0.5)
        assert harness.macro_citation_support_rate == pytest.approx(0.5)

    def test_to_dict_has_expected_keys(self):
        pr = self._make_paper_result("p1", 0.8, True, True)
        harness = HarnessResult(paper_results=[pr], top_k=3)
        d = harness.to_dict()
        assert "summary" in d
        assert "per_paper" in d
        assert "top_k" in d
        assert "unmeasured" in d

    def test_to_dict_summary_values(self):
        pr = self._make_paper_result("p1", 0.9, True, True)
        harness = HarnessResult(paper_results=[pr], top_k=5)
        d = harness.to_dict()
        assert d["top_k"] == 5
        assert d["summary"]["num_papers"] == 1
        assert d["summary"]["macro_token_f1"] == pytest.approx(0.9, abs=1e-3)

    def test_to_dict_per_paper_has_fields(self):
        pr = self._make_paper_result("paper_x", 0.7, True, False)
        harness = HarnessResult(paper_results=[pr], top_k=3)
        d = harness.to_dict()
        pp = d["per_paper"][0]
        assert pp["document_id"] == "paper_x"
        assert "fields" in pp
        assert len(pp["fields"]) == 1

    def test_no_papers_macro_f1_is_zero(self):
        harness = HarnessResult(paper_results=[], top_k=3)
        assert harness.macro_token_f1 == pytest.approx(0.0)

    def test_hit_rate_none_when_no_evidence_pages(self):
        fr = FieldResult(
            field_name="title",
            token_f1=0.9,
            gold_is_null=False,
            predicted_is_null=False,
            correct_null=False,
            retrieval_hit=None,  # no evidence_pages annotated
            annotated_pages=[],
            retrieved_pages=[],
            citation_supported=True,
        )
        pr = PaperResult(document_id="p1", field_results=[fr])
        harness = HarnessResult(paper_results=[pr], top_k=3)
        assert harness.macro_hit_rate is None


# ─────────────────────────────────────────────────────────────────────────────
# Harness runner (dry-run) integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHarnessRunnerDryRun:
    """Test the CLI runner in --dry-run mode so no LLM or PDF access is needed."""

    def _write_annotation(self, path: Path, annotation: dict) -> None:
        path.write_text(json.dumps(annotation), encoding="utf-8")

    def test_dry_run_produces_report_json(self, valid_annotation, tmp_path):
        """--dry-run writes report.json and report_summary.txt."""
        gold_dir = tmp_path / "gold_set"
        gold_dir.mkdir()
        self._write_annotation(gold_dir / "test_paper.json", valid_annotation)

        output_dir = tmp_path / "output"

        import sys
        from unittest.mock import patch

        # Patch sys.argv for argparse
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
        assert (output_dir / "report_summary.txt").exists()

        with open(output_dir / "report.json", encoding="utf-8") as fh:
            report = json.load(fh)
        assert report["summary"]["num_papers"] == 1

    def test_dry_run_all_metrics_zero_or_null(self, valid_annotation, tmp_path):
        """Dry-run sets all predictions to null, so F1=0 for non-null gold fields."""
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
        # All gold fields are non-null, so token_f1 should be 0 for every field
        pp = report["per_paper"][0]
        assert pp["mean_token_f1"] == pytest.approx(0.0)

    def test_dry_run_no_groq_key_required(self, valid_annotation, tmp_path, monkeypatch):
        """Dry-run should work even without GROQ_API_KEY in environment."""
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
        """Running without --dry-run and without --confirm-groq-costs exits with error."""
        gold_dir = tmp_path / "gold_set"
        gold_dir.mkdir()
        self._write_annotation(gold_dir / "test_paper.json", valid_annotation)

        import sys
        args = [
            "run_harness.py",
            "--gold-dir", str(gold_dir),
            "--output-dir", str(tmp_path / "out"),
            # No --dry-run, no --confirm-groq-costs
        ]
        with patch.object(sys, "argv", args):
            from evaluation.run_harness import main
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code != 0

    def test_invalid_gold_set_exits_with_error(self, valid_annotation, tmp_path):
        """Invalid annotation (placeholder) causes non-zero exit."""
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
        """--document-id evaluates only the specified paper."""
        gold_dir = tmp_path / "gold_set"
        gold_dir.mkdir()
        # Write two annotations
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
        assert report["summary"]["num_papers"] == 1
        assert report["per_paper"][0]["document_id"] == "test_paper"
