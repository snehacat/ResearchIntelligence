"""Tests for search evaluation metrics and offline evaluation infrastructure."""

import pytest

from scholarly_search.evaluation import (
    SearchMetrics,
    calculate_precision_at_5,
    calculate_duplicate_rate,
    calculate_metadata_completeness,
    calculate_average_topic_overlap,
    calculate_metrics,
)


class TestPrecisionAt5:
    """Tests for precision@5 calculation."""

    def test_precision_at_5_perfect(self):
        """Test perfect precision (all top 5 are relevant)."""
        results = [
            {"doi": "10.1234/1"},
            {"doi": "10.1234/2"},
            {"doi": "10.1234/3"},
            {"doi": "10.1234/4"},
            {"doi": "10.1234/5"},
        ]
        relevant_dois = {"10.1234/1", "10.1234/2", "10.1234/3", "10.1234/4", "10.1234/5"}
        assert calculate_precision_at_5(results, relevant_dois) == 1.0

    def test_precision_at_5_partial(self):
        """Test partial precision (3/5 relevant)."""
        results = [
            {"doi": "10.1234/1"},
            {"doi": "10.1234/2"},
            {"doi": "10.1234/3"},
            {"doi": "10.1234/4"},
            {"doi": "10.1234/5"},
        ]
        relevant_dois = {"10.1234/1", "10.1234/3", "10.1234/5"}
        assert calculate_precision_at_5(results, relevant_dois) == 0.6

    def test_precision_at_5_zero(self):
        """Test zero precision (none are relevant)."""
        results = [
            {"doi": "10.1234/1"},
            {"doi": "10.1234/2"},
            {"doi": "10.1234/3"},
            {"doi": "10.1234/4"},
            {"doi": "10.1234/5"},
        ]
        relevant_dois = {"10.1234/10", "10.1234/11"}
        assert calculate_precision_at_5(results, relevant_dois) == 0.0

    def test_precision_at_5_case_insensitive(self):
        """Test DOI matching is case-insensitive."""
        results = [
            {"doi": "10.1234/ABC"},
            {"doi": "10.1234/DEF"},
        ]
        relevant_dois = {"10.1234/abc"}
        assert calculate_precision_at_5(results, relevant_dois) == 0.5

    def test_precision_at_5_missing_doi(self):
        """Test results without DOI are never relevant."""
        results = [
            {"doi": "10.1234/1"},
            {"title": "Paper without DOI"},
        ]
        relevant_dois = {"10.1234/1"}
        assert calculate_precision_at_5(results, relevant_dois) == 0.5

    def test_precision_at_5_fewer_than_5_results(self):
        """Test precision with fewer than 5 results."""
        results = [
            {"doi": "10.1234/1"},
            {"doi": "10.1234/2"},
        ]
        relevant_dois = {"10.1234/1"}
        assert calculate_precision_at_5(results, relevant_dois) == 0.5

    def test_precision_at_5_empty_results(self):
        """Test precision with empty results."""
        results = []
        relevant_dois = {"10.1234/1"}
        assert calculate_precision_at_5(results, relevant_dois) == 0.0


class TestDuplicateRate:
    """Tests for duplicate rate calculation."""

    def test_duplicate_rate_no_duplicates(self):
        """Test no duplicates removed."""
        results = [{"doi": "10.1234/1"}, {"doi": "10.1234/2"}]
        original_count = 2
        assert calculate_duplicate_rate(results, original_count) == 0.0

    def test_duplicate_rate_half_duplicates(self):
        """Test half of results were duplicates."""
        results = [{"doi": "10.1234/1"}]
        original_count = 2
        assert calculate_duplicate_rate(results, original_count) == 0.5

    def test_duplicate_rate_all_duplicates(self):
        """Test all results were duplicates."""
        results = []
        original_count = 5
        assert calculate_duplicate_rate(results, original_count) == 1.0

    def test_duplicate_rate_zero_original(self):
        """Test zero original candidates."""
        results = []
        original_count = 0
        assert calculate_duplicate_rate(results, original_count) == 0.0


class TestMetadataCompleteness:
    """Tests for metadata completeness calculation."""

    def test_metadata_completeness_perfect(self):
        """Test all fields complete."""
        results = [
            {
                "title": "Paper 1",
                "authors": ["Author"],
                "year": 2020,
                "doi": "10.1234/1",
                "abstract": "Abstract",
                "url": "https://example.com",
            }
        ]
        assert calculate_metadata_completeness(results) == 1.0

    def test_metadata_completeness_partial(self):
        """Test partial completeness (4/6 fields)."""
        results = [
            {
                "title": "Paper 1",
                "authors": ["Author"],
                "year": 2020,
                "doi": "10.1234/1",
                "abstract": None,
                "url": None,
            }
        ]
        assert calculate_metadata_completeness(results) == 4/6

    def test_metadata_completeness_empty_string(self):
        """Test empty string counts as missing."""
        results = [
            {
                "title": "Paper 1",
                "authors": [],
                "year": 2020,
                "doi": "",
                "abstract": "Abstract",
                "url": "https://example.com",
            }
        ]
        # title (complete), authors (empty list, missing), year (complete),
        # doi (empty string, missing), abstract (complete), url (complete)
        # = 4/6 complete
        assert calculate_metadata_completeness(results) == 4/6

    def test_metadata_completeness_multiple_results(self):
        """Test average across multiple results."""
        results = [
            {
                "title": "Paper 1",
                "authors": ["Author"],
                "year": 2020,
                "doi": "10.1234/1",
                "abstract": "Abstract",
                "url": "https://example.com",
            },
            {
                "title": "Paper 2",
                "authors": None,
                "year": None,
                "doi": None,
                "abstract": None,
                "url": None,
            }
        ]
        # First: 6/6, Second: 1/6 = 7/12
        assert calculate_metadata_completeness(results) == 7/12

    def test_metadata_completeness_empty_results(self):
        """Test empty results."""
        results = []
        assert calculate_metadata_completeness(results) == 0.0


class TestAverageTopicOverlap:
    """Tests for average topic overlap calculation."""

    def test_average_topic_overlap_perfect(self):
        """Test perfect topic overlap."""
        results = [
            {"topic_overlap_score": 1.0},
            {"topic_overlap_score": 1.0},
        ]
        assert calculate_average_topic_overlap(results) == 1.0

    def test_average_topic_overlap_mixed(self):
        """Test mixed topic overlap scores."""
        results = [
            {"topic_overlap_score": 0.5},
            {"topic_overlap_score": 0.7},
            {"topic_overlap_score": 0.3},
        ]
        assert calculate_average_topic_overlap(results) == 0.5

    def test_average_topic_overlap_zero(self):
        """Test zero topic overlap."""
        results = [
            {"topic_overlap_score": 0.0},
            {"topic_overlap_score": 0.0},
        ]
        assert calculate_average_topic_overlap(results) == 0.0

    def test_average_topic_overlap_missing_score(self):
        """Test results without topic_overlap_score are not counted."""
        results = [
            {"topic_overlap_score": 0.5},
            {"title": "Paper without score"},
        ]
        # Only first result has the field, so it's the only one counted
        assert calculate_average_topic_overlap(results) == 0.5

    def test_average_topic_overlap_empty_results(self):
        """Test empty results."""
        results = []
        assert calculate_average_topic_overlap(results) == 0.0


class TestCalculateMetrics:
    """Tests for combined metrics calculation."""

    def test_calculate_metrics_full(self):
        """Test full metrics calculation."""
        results = [
            {
                "doi": "10.1234/1",
                "title": "Paper 1",
                "authors": ["Author"],
                "year": 2020,
                "doi": "10.1234/1",
                "abstract": "Abstract",
                "url": "https://example.com",
                "topic_overlap_score": 0.8,
            },
            {
                "doi": "10.1234/2",
                "title": "Paper 2",
                "authors": ["Author"],
                "year": 2020,
                "doi": "10.1234/2",
                "abstract": "Abstract",
                "url": "https://example.com",
                "topic_overlap_score": 0.6,
            },
        ]
        relevant_dois = {"10.1234/1"}
        original_count = 3

        metrics = calculate_metrics(results, relevant_dois, original_count)

        assert metrics.precision_at_5 == 0.5  # 1/2 relevant
        assert metrics.duplicate_rate == 1/3  # 1 duplicate removed
        assert metrics.metadata_completeness == 1.0  # All fields complete
        assert metrics.average_topic_overlap == 0.7  # (0.8 + 0.6) / 2
        assert metrics.provider_coverage == 3
        assert metrics.total_results == 2

    def test_calculate_metrics_to_dict(self):
        """Test metrics serialization."""
        results = [
            {
                "doi": "10.1234/1",
                "title": "Paper 1",
                "authors": ["Author"],
                "year": 2020,
                "doi": "10.1234/1",
                "abstract": "Abstract",
                "url": "https://example.com",
                "topic_overlap_score": 0.8,
            }
        ]
        relevant_dois = {"10.1234/1"}
        original_count = 1

        metrics = calculate_metrics(results, relevant_dois, original_count)
        metrics_dict = metrics.to_dict()

        assert metrics_dict["precision_at_5"] == 1.0
        assert metrics_dict["duplicate_rate"] == 0.0
        assert metrics_dict["metadata_completeness"] == 1.0
        assert metrics_dict["average_topic_overlap"] == 0.8
        assert metrics_dict["provider_coverage"] == 1
        assert metrics_dict["total_results"] == 1


class TestEvaluationFixtures:
    """Tests using mocked evaluation datasets."""

    def test_transformer_topic_dataset(self):
        """Test evaluation with transformer-related topic."""
        # Mock results for "Attention Is All You Need" query
        results = [
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors": ["Devlin et al."],
                "year": 2019,
                "doi": "10.1038/s41586-019-0318-v5",
                "abstract": "Abstract about BERT",
                "url": "https://example.com/bert.pdf",
                "topic_overlap_score": 0.7,
                "relevance_score": 0.9,
            },
            {
                "title": "Attention in Cognitive Psychology",
                "authors": ["Smith et al."],
                "year": 2010,
                "doi": "10.1000/psych-attention",
                "abstract": "Abstract about psychology",
                "url": "https://example.com/psych.pdf",
                "topic_overlap_score": 0.3,
                "relevance_score": 0.8,
            },
            {
                "title": "Efficient Transformers: A Survey",
                "authors": ["Tay et al."],
                "year": 2020,
                "doi": "10.1109/efficient-transformers",
                "abstract": "Abstract about transformer variants",
                "url": "https://example.com/efficient.pdf",
                "topic_overlap_score": 0.8,
                "relevance_score": 0.95,
            },
        ]
        relevant_dois = {
            "10.1038/s41586-019-0318-v5",  # BERT
            "10.1109/efficient-transformers",  # Efficient Transformers
        }
        original_count = 5  # Simulate 2 duplicates removed

        metrics = calculate_metrics(results, relevant_dois, original_count)

        # Top 5 with 3 results: BERT (relevant), Psych (irrelevant), Efficient (relevant)
        # Precision@5 = 2/3 = 0.667
        assert metrics.precision_at_5 == 2/3
        assert metrics.duplicate_rate == 2/5
        assert metrics.metadata_completeness == 1.0
        assert 0.6 <= metrics.average_topic_overlap <= 0.7

    def test_nlp_topic_dataset(self):
        """Test evaluation with general NLP topic."""
        results = [
            {
                "title": "Word2Vec: Distributed Representations",
                "authors": ["Mikolov et al."],
                "year": 2013,
                "doi": "10.1000/word2vec",
                "abstract": None,  # Missing abstract
                "url": "https://example.com/word2vec.pdf",
                "topic_overlap_score": 0.5,
            },
            {
                "title": "GloVe: Global Vectors",
                "authors": ["Pennington et al."],
                "year": 2014,
                "doi": "10.1000/glove",
                "abstract": "Abstract about GloVe",
                "url": None,  # Missing URL
                "topic_overlap_score": 0.6,
            },
        ]
        relevant_dois = {"10.1000/word2vec"}
        original_count = 2

        metrics = calculate_metrics(results, relevant_dois, original_count)

        assert metrics.precision_at_5 == 0.5
        assert metrics.duplicate_rate == 0.0
        # First: 5/6 (missing abstract), Second: 5/6 (missing URL) = 10/12
        assert metrics.metadata_completeness == 10/12
        assert metrics.average_topic_overlap == 0.55
