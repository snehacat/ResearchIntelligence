"""Edge-case unit tests for evaluation metrics (Task 3)."""

from pathlib import Path

import pytest
from evaluation.gold_set_schema import load_annotation
from evaluation.report_metrics import count_filled_annotations, load_gold_annotations
from evaluation.metrics import (
    _token_f1,
    _set_metrics,
    _list_token_f1,
    _tokenize,
    _normalize_item,
)


class TestTokenF1EdgeCases:
    """Edge case tests for token-F1 metric."""

    def test_empty_both_returns_1(self):
        """Both empty strings should return perfect F1 (1.0)."""
        assert _token_f1("", "") == 1.0

    def test_empty_predicted_returns_0(self):
        """Empty prediction with non-empty gold should return 0.0."""
        assert _token_f1("", "some text") == 0.0

    def test_empty_gold_returns_0(self):
        """Non-empty prediction with empty gold should return 0.0."""
        assert _token_f1("some text", "") == 0.0

    def test_identical_strings_returns_1(self):
        """Identical strings should return perfect F1 (1.0)."""
        assert _token_f1("transformer architecture", "transformer architecture") == 1.0

    def test_case_insensitive(self):
        """Token-F1 should be case-insensitive."""
        assert _token_f1("Transformer Architecture", "transformer architecture") == 1.0

    def test_stopwords_removed(self):
        """Stopwords should be removed from both strings."""
        assert _token_f1("the transformer and the attention", "transformer attention") == 1.0

    def test_no_overlap_returns_0(self):
        """No token overlap should return 0.0."""
        assert _token_f1("transformer attention", "convolutional pooling") == 0.0

    def test_partial_overlap(self):
        """Partial token overlap should return fractional F1."""
        score = _token_f1("transformer attention mechanism", "transformer pooling")
        assert 0 < score < 1

    def test_punctuation_handling(self):
        """Punctuation should be ignored (tokens are word/number sequences)."""
        assert _token_f1("transformer, attention!", "transformer attention") == 1.0

    def test_numbers_preserved(self):
        """Numbers should be preserved as tokens."""
        assert _token_f1("accuracy 95.5", "accuracy 95.5") == 1.0

    def test_duplicates_in_strings(self):
        """Duplicate tokens should not affect F1 (set-based comparison)."""
        assert _token_f1("transformer transformer", "transformer") == 1.0


class TestSetF1EdgeCases:
    """Edge case tests for set-F1 metric (authors, evaluation_metrics)."""

    def test_empty_both_lists_returns_1(self):
        """Both empty lists should return perfect precision/recall/F1 (1.0)."""
        precision, recall, f1 = _set_metrics([], [])
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_empty_predicted_returns_0(self):
        """Empty prediction with non-empty gold should return 0.0."""
        precision, recall, f1 = _set_metrics([], ["Author One", "Author Two"])
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_empty_gold_returns_0(self):
        """Non-empty prediction with empty gold should return 0.0."""
        precision, recall, f1 = _set_metrics(["Author One"], [])
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_identical_lists_returns_1(self):
        """Identical lists should return perfect precision/recall/F1 (1.0)."""
        precision, recall, f1 = _set_metrics(
            ["Author One", "Author Two"],
            ["Author One", "Author Two"]
        )
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_case_insensitive(self):
        """Set-F1 should be case-insensitive."""
        precision, recall, f1 = _set_metrics(
            ["Author One", "Author Two"],
            ["author one", "author two"]
        )
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_whitespace_stripped(self):
        """Whitespace should be stripped from items."""
        precision, recall, f1 = _set_metrics(
            ["  Author One  ", "Author Two"],
            ["Author One", "  Author Two  "]
        )
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_punctuation_normalized(self):
        """Punctuation should be normalized to spaces."""
        precision, recall, f1 = _set_metrics(
            ["Author, One", "Author Two"],
            ["Author One", "Author Two"]
        )
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_no_overlap_returns_0(self):
        """No item overlap should return 0.0."""
        precision, recall, f1 = _set_metrics(
            ["Author One", "Author Two"],
            ["Author Three", "Author Four"]
        )
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_partial_overlap(self):
        """Partial item overlap should return fractional precision/recall/F1."""
        precision, recall, f1 = _set_metrics(
            ["Author One", "Author Two", "Author Three"],
            ["Author One", "Author Four"]
        )
        assert 0 < precision < 1
        assert 0 < recall < 1
        assert 0 < f1 < 1

    def test_duplicate_items_in_prediction(self):
        """Duplicate items in prediction should be deduplicated (set-based)."""
        precision, recall, f1 = _set_metrics(
            ["Author One", "Author One", "Author Two"],
            ["Author One", "Author Two"]
        )
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_duplicate_items_in_gold(self):
        """Duplicate items in gold should be deduplicated (set-based)."""
        precision, recall, f1 = _set_metrics(
            ["Author One", "Author Two"],
            ["Author One", "Author One", "Author Two"]
        )
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_string_instead_of_list(self):
        """String input should be treated as single-item list."""
        precision, recall, f1 = _set_metrics("Author One", ["Author One"])
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_non_string_items_ignored(self):
        """Non-string items in lists should be ignored."""
        precision, recall, f1 = _set_metrics(
            ["Author One", None, 123, "Author Two"],
            ["Author One", "Author Two"]
        )
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_empty_string_items_ignored(self):
        """Empty string items should be ignored."""
        precision, recall, f1 = _set_metrics(
            ["Author One", "", "   ", "Author Two"],
            ["Author One", "Author Two"]
        )
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0


class TestListTokenF1EdgeCases:
    """Edge case tests for list token-F1 (limitations field)."""

    def test_empty_both_returns_0(self):
        """Both empty should return 0.0 (no gold items to average)."""
        assert _list_token_f1([], []) == 0.0

    def test_empty_predicted_returns_0(self):
        """Empty prediction with non-empty gold should return 0.0."""
        assert _list_token_f1([], ["limitation one"]) == 0.0

    def test_empty_gold_returns_0(self):
        """Non-empty prediction with empty gold should return 0.0."""
        assert _list_token_f1(["limitation one"], []) == 0.0

    def test_identical_sentences_returns_1(self):
        """Identical sentences should return 1.0."""
        assert _list_token_f1(
            ["The sample size was small."],
            ["The sample size was small."]
        ) == 1.0

    def test_best_match_per_gold_item(self):
        """Each gold item should match the best predicted item."""
        # Gold: ["A", "B"]
        # Predicted: ["A", "C"]
        # "A" matches perfectly, "B" matches nothing -> average 0.5
        score = _list_token_f1(["A", "C"], ["A", "B"])
        assert score == 0.5

    def test_string_instead_of_list(self):
        """String input should be treated as single-item list."""
        assert _list_token_f1("limitation one", ["limitation one"]) == 1.0

    def test_non_string_items_ignored(self):
        """Non-string items should be ignored."""
        assert _list_token_f1(
            ["limitation one", None, 123],
            ["limitation one"]
        ) == 1.0


class TestTokenizeEdgeCases:
    """Edge case tests for tokenization."""

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert _tokenize("") == []

    def test_only_stopwords(self):
        """String with only stopwords should return empty list."""
        assert _tokenize("the and is of") == []

    def test_only_punctuation(self):
        """String with only punctuation should return empty list."""
        assert _tokenize("!@#$%^&*()") == []

    def test_mixed_content(self):
        """Mixed words, numbers, and punctuation should tokenize correctly."""
        tokens = _tokenize("The accuracy was 95.5% in 2024.")
        assert "accuracy" in tokens
        assert "95.5" in tokens
        assert "2024" in tokens
        assert "the" not in tokens  # stopword removed

    def test_numbers_with_decimals(self):
        """Numbers with decimals should be preserved as single tokens."""
        tokens = _tokenize("accuracy 95.5 precision 0.85")
        assert "95.5" in tokens
        assert "0.85" in tokens

    def test_numbers_with_commas(self):
        """Numbers with commas should be tokenized without commas."""
        tokens = _tokenize("1,000 10,000")
        assert "1,000" in tokens
        assert "10,000" in tokens


class TestNormalizeItemEdgeCases:
    """Edge case tests for item normalization."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert _normalize_item("") == ""

    def test_whitespace_only(self):
        """Whitespace-only string should return empty string."""
        assert _normalize_item("   ") == ""

    def test_leading_trailing_whitespace(self):
        """Leading and trailing whitespace should be stripped."""
        assert _normalize_item("  Author One  ") == "author one"

    def test_case_conversion(self):
        """Uppercase should be converted to lowercase."""
        assert _normalize_item("AUTHOR ONE") == "author one"

    def test_punctuation_to_spaces(self):
        """Punctuation should be converted to spaces."""
        assert _normalize_item("Author, One!") == "author one"

    def test_multiple_spaces_collapsed(self):
        """Multiple spaces should be collapsed to single space."""
        assert _normalize_item("Author  One   Two") == "author one two"

    def test_trailing_whitespace_after_normalize(self):
        """Trailing whitespace should be stripped after normalization."""
        assert _normalize_item("Author One  ") == "author one"


class TestGoldAnnotationCounts:
    def test_empty_templates_do_not_increase_filled_paper_count(self):
        project_root = Path(__file__).parents[2]
        annotated = load_annotation(
            project_root
            / "evaluation"
            / "gold_set"
            / "attention_is_all_you_need.json"
        )
        annotations = {
            "attention_is_all_you_need": annotated,
            "empty_template_one": {
                "fields": {"title": {"value": ""}, "authors": {"value": []}}
            },
            "empty_template_two": {
                "fields": {"title": {"value": None}, "authors": {"value": []}}
            },
        }

        assert count_filled_annotations(annotations) == 1

    def test_owner_choice_todo_files_are_not_loaded_as_gold(self, tmp_path, capsys):
        todo_file = tmp_path / "placeholder_paper_1.json"
        todo_file.write_text(
            '{"TODO": "The owner chooses which paper to add to the gold set."}',
            encoding="utf-8",
        )

        assert load_gold_annotations(tmp_path) == {}
        assert "The owner chooses which paper" in capsys.readouterr().out
