"""Tests for query generation behavior and improvements."""

import pytest

from scholarly_search.crossref_client import CrossrefSearchClient
from scholarly_search.openalex_client import OpenAlexSearchClient


class TestQueryGenerationBehavior:
    """Tests to understand current query generation behavior."""

    def test_transformer_query_current_behavior(self):
        """Test query generation for Transformer architecture."""
        analysis = {
            "title": "Attention Is All You Need",
            "research_problem": "Develop a sequence transduction model relying only on attention",
            "methodology": "Transformer architecture using multi-head self-attention",
            "dataset": "WMT 2014 English-to-German and English-to-French",
            "evaluation_metrics": ["BLEU"],
        }

        queries = CrossrefSearchClient._build_queries(analysis)

        # Current behavior should have title query and topical query
        assert len(queries) == 2
        assert "Attention Is All You Need" in queries[0]
        # Check if technical phrases are preserved
        assert "multi-head" in queries[1] or "multi head" in queries[1].lower()

    def test_bert_query_current_behavior(self):
        """Test query generation for BERT."""
        analysis = {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "research_problem": "Pre-train deep bidirectional representations",
            "methodology": "Masked language modeling and next sentence prediction",
            "dataset": "BookCorpus and Wikipedia",
            "evaluation_metrics": ["GLUE", "SQuAD"],
        }

        queries = CrossrefSearchClient._build_queries(analysis)

        assert len(queries) == 2
        assert "BERT" in queries[0] or "Pre-training" in queries[0]

    def test_field_truncation_methodology(self):
        """Test if methodology field is being truncated."""
        long_methodology = (
            "We propose a novel approach combining convolutional neural networks "
            "with attention mechanisms for improved performance on sequence modeling tasks. "
            "The method uses a multi-head attention layer followed by position-wise feed-forward networks."
        )

        analysis = {
            "title": "Test Paper",
            "methodology": long_methodology,
        }

        queries = CrossrefSearchClient._build_queries(analysis)

        # Current limit is 140 characters
        topical_query = queries[1] if len(queries) > 1 else ""
        assert len(topical_query) <= 140  # Check truncation

    def test_field_truncation_dataset(self):
        """Test if dataset field is being truncated."""
        long_dataset = (
            "We evaluate on the Stanford Sentiment Treebank, Multi-Genre Natural Language Inference, "
            "Question Natural Language Inference, and the Quora Question Pairs dataset."
        )

        analysis = {
            "title": "Test Paper",
            "dataset": long_dataset,
        }

        queries = CrossrefSearchClient._build_queries(analysis)

        # Current limit is 80 characters
        topical_query = queries[1] if len(queries) > 1 else ""
        assert len(topical_query) <= 80  # Check truncation

    def test_technical_phrase_preservation(self):
        """Test if technical phrases are preserved in queries."""
        analysis = {
            "title": "Multi-Head Attention for Sequence Modeling",
            "methodology": "Uses multi-head self-attention mechanism",
        }

        queries = CrossrefSearchClient._build_queries(analysis)

        # _clean_text preserves hyphens, so the query should contain "multi-head"
        combined = " ".join(queries)
        assert "multi-head" in combined.lower() or "Multi-Head" in combined

    def test_self_attention_preservation(self):
        """Test that self-attention phrase is preserved."""
        analysis = {
            "title": "Self-Attention Networks",
            "methodology": "Uses self-attention for sequence modeling",
        }

        queries = CrossrefSearchClient._build_queries(analysis)

        combined = " ".join(queries)
        assert "self-attention" in combined.lower() or "Self-Attention" in combined

    def test_sequence_to_sequence_preservation(self):
        """Test that sequence-to-sequence phrase is preserved."""
        analysis = {
            "title": "Sequence-to-Sequence Models",
            "methodology": "Uses sequence-to-sequence architecture",
        }

        queries = CrossrefSearchClient._build_queries(analysis)

        combined = " ".join(queries)
        assert "sequence-to-sequence" in combined.lower() or "Sequence-to-Sequence" in combined

    def test_stopword_removal(self):
        """Test if stopwords are being removed."""
        analysis = {
            "title": "A Study on Research Paper Analysis Using Deep Learning",
            "research_problem": "Research paper analysis using deep learning methods",
        }

        queries = CrossrefSearchClient._build_queries(analysis)

        # Check if stopwords like "study", "research", "paper" are in queries
        # Note: _build_queries doesn't apply stopwords; that's done in _topic_tokens
        # So the query should contain these words
        combined = " ".join(queries).lower()
        assert "study" in combined or "paper" in combined  # Current behavior

    def test_empty_fields_handling(self):
        """Test query generation with empty fields."""
        analysis = {
            "title": "Test Paper",
            "research_problem": None,
            "methodology": "",
            "dataset": None,
            "evaluation_metrics": [],
        }

        queries = CrossrefSearchClient._build_queries(analysis)

        # Should only have title query
        assert len(queries) == 1
        assert queries[0] == "Test Paper"

    def test_duplicate_terms(self):
        """Test if duplicate terms are handled."""
        analysis = {
            "title": "Deep Learning for Deep Learning",
            "research_problem": "Deep learning approaches",
            "methodology": "Deep neural networks",
        }

        queries = CrossrefSearchClient._build_queries(analysis)

        # Current behavior doesn't deduplicate terms within query
        # This test documents current behavior
        combined = " ".join(queries).lower()
        # "deep" likely appears multiple times
        assert combined.count("deep") >= 2

    def test_both_providers_identical_queries(self):
        """Test that both providers generate identical queries."""
        analysis = {
            "title": "Test Paper",
            "research_problem": "Test problem",
            "methodology": "Test method",
        }

        crossref_queries = CrossrefSearchClient._build_queries(analysis)
        openalex_queries = OpenAlexSearchClient._build_queries(analysis)

        assert crossref_queries == openalex_queries


class TestCurrentCharacterLimits:
    """Document current character limits."""

    def test_title_limit(self):
        """Current title limit is 240 characters."""
        long_title = "A" * 250
        analysis = {"title": long_title}

        queries = CrossrefSearchClient._build_queries(analysis)
        assert len(queries[0]) == 240

    def test_research_problem_limit(self):
        """Current research_problem limit is 180 characters."""
        long_text = "A" * 200
        analysis = {"research_problem": long_text}

        queries = CrossrefSearchClient._build_queries(analysis)
        topical_query = queries[0] if len(queries) > 0 else ""
        assert len(topical_query) <= 180

    def test_methodology_limit(self):
        """Current methodology limit is 140 characters."""
        long_text = "A" * 150
        analysis = {"methodology": long_text}

        queries = CrossrefSearchClient._build_queries(analysis)
        topical_query = queries[0] if len(queries) > 0 else ""
        assert len(topical_query) <= 140

    def test_dataset_limit(self):
        """Current dataset limit is 80 characters."""
        long_text = "A" * 90
        analysis = {"dataset": long_text}

        queries = CrossrefSearchClient._build_queries(analysis)
        topical_query = queries[0] if len(queries) > 0 else ""
        assert len(topical_query) <= 80

    def test_evaluation_metrics_limit(self):
        """Current evaluation_metrics limit is 60 characters."""
        long_text = "A" * 70
        analysis = {"evaluation_metrics": long_text}

        queries = CrossrefSearchClient._build_queries(analysis)
        topical_query = queries[0] if len(queries) > 0 else ""
        assert len(topical_query) <= 60
