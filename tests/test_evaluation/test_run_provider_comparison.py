"""Tests for the evaluation runner."""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from evaluation.run_provider_comparison import (
    run_provider_search,
    save_results,
    create_relevance_label_template,
    calculate_metrics_from_labels,
    extract_minimal_analysis,
)


class TestExtractMinimalAnalysis:
    """Tests for minimal paper analysis extraction."""
    
    @patch("pdf_processing.text_extractor.TextExtractor")
    def test_extract_minimal_analysis_basic(self, mock_extractor_class):
        """Test basic extraction returns title from first page."""
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        
        # Mock extraction result with first page text
        mock_extractor.extract_from_file.return_value = {
            'pages': [
                {'page_number': 1, 'text': 'Attention Is All You Need\nAbstract\nThis is the abstract.'}
            ],
            'total_pages': 10,
            'pages_with_text': 10,
            'pages_without_text': [],
            'total_characters': 5000,
            'success': True
        }
        
        analysis = extract_minimal_analysis("test.pdf")
        
        assert analysis["title"] == "Attention Is All You Need"
        assert analysis["research_problem"] is None
        assert analysis["methodology"] is None
        assert analysis["dataset"] is None
        assert analysis["evaluation_metrics"] is None
        
        mock_extractor.extract_from_file.assert_called_once_with("test.pdf")
    
    @patch("pdf_processing.text_extractor.TextExtractor")
    def test_extract_minimal_analysis_skips_permission_header(self, mock_extractor_class):
        """Test that permission headers are skipped and actual title is extracted."""
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        
        # Mock extraction result with permission header (actual Attention Is All You Need PDF structure)
        # The permission notice spans 3 lines, then the title appears
        mock_extractor.extract_from_file.return_value = {
            'pages': [
                {'page_number': 1, 'text': 'Provided proper attribution is provided, Google hereby grants permission to\nreproduce the tables and figures in this paper solely for use in journalistic or\nscholarly works.\nAttention Is All You Need\nAbstract\nThis is the abstract.'}
            ],
            'total_pages': 10,
            'pages_with_text': 10,
            'pages_without_text': [],
            'total_characters': 5000,
            'success': True
        }
        
        analysis = extract_minimal_analysis("attention_is_all_you_need.pdf")
        
        # Should extract "Attention Is All You Need", not the permission notice
        assert analysis["title"] == "Attention Is All You Need"
    
    @patch("pdf_processing.text_extractor.TextExtractor")
    def test_extract_minimal_analysis_uses_filename_fallback(self, mock_extractor_class):
        """Test filename fallback when first page has no non-empty lines after skipping headers."""
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        
        # Mock extraction result with only permission text
        mock_extractor.extract_from_file.return_value = {
            'pages': [
                {'page_number': 1, 'text': 'Provided proper attribution is provided\n© 2024 All rights reserved\n\n'}
            ],
            'total_pages': 10,
            'pages_with_text': 9,
            'pages_without_text': [1],
            'total_characters': 5000,
            'success': True
        }
        
        analysis = extract_minimal_analysis("test.pdf")
        
        # Should use filename as fallback
        assert analysis["title"] == "test"
    
    @patch("pdf_processing.text_extractor.TextExtractor")
    @patch("sys.exit")
    def test_extract_minimal_analysis_handles_extraction_error(self, mock_exit, mock_extractor_class):
        """Test extraction error exits with error message."""
        from pdf_processing.text_extractor import PDFExtractionError
        
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        
        # Mock extraction error
        mock_extractor.extract_from_file.side_effect = PDFExtractionError("PDF not found")
        
        extract_minimal_analysis("missing.pdf")
        
        # Should exit with error
        mock_exit.assert_called_once_with(1)
        # Verify error message was printed
        # (The test would capture stdout, but we just verify the exit code)


class TestRunProviderSearch:
    """Tests for provider search execution."""
    
    @patch("evaluation.run_provider_comparison.create_search_provider")
    def test_run_provider_search_success(self, mock_create):
        """Test successful provider search."""
        mock_provider = Mock()
        mock_provider.search_related_papers.return_value = [
            {"title": "Paper 1", "doi": "10.1234/1"},
            {"title": "Paper 2", "doi": "10.1234/2"},
        ]
        mock_create.return_value = mock_provider
        
        paper_analysis = {"title": "Test"}
        results = run_provider_search("crossref", paper_analysis, max_results=10)
        
        assert results["success"] is True
        assert results["provider"] == "crossref"
        assert results["result_count"] == 2
        assert results["error"] is None
        assert len(results["results"]) == 2
    
    @patch("evaluation.run_provider_comparison.create_search_provider")
    def test_run_provider_search_error(self, mock_create):
        """Test provider search error handling."""
        mock_provider = Mock()
        mock_provider.search_related_papers.side_effect = Exception("API error")
        mock_create.return_value = mock_provider
        
        paper_analysis = {"title": "Test"}
        results = run_provider_search("crossref", paper_analysis, max_results=10)
        
        assert results["success"] is False
        assert results["provider"] == "crossref"
        assert results["result_count"] == 0
        assert results["error"] == "API error"
        assert results["results"] == []


class TestSaveResults:
    """Tests for result saving."""
    
    def test_save_results_creates_directory(self):
        """Test that save_results creates output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "nested", "dir")
            results = {"test": "data"}
            
            save_results(results, output_dir, "test.json")
            
            assert os.path.exists(output_dir)
            assert os.path.exists(os.path.join(output_dir, "test.json"))
    
    def test_save_results_content(self):
        """Test that save_results writes correct content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = {"key": "value", "number": 42}
            
            save_results(results, temp_dir, "test.json")
            
            with open(os.path.join(temp_dir, "test.json"), 'r') as f:
                loaded = json.load(f)
            
            assert loaded == results


class TestRelevanceLabelTemplate:
    """Tests for relevance label template creation."""
    
    def test_create_template_top_5_only(self):
        """Test template includes only top 5 results."""
        crossref_results = [
            {"title": f"Paper {i}", "doi": f"10.1234/{i}"}
            for i in range(10)
        ]
        openalex_results = [
            {"title": f"Paper {i}", "doi": f"10.5678/{i}"}
            for i in range(10)
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            create_relevance_label_template(crossref_results, openalex_results, temp_dir)
            
            template_path = os.path.join(temp_dir, "relevance_labels_template.json")
            with open(template_path, 'r') as f:
                template = json.load(f)
            
            assert len(template["crossref"]) == 5
            assert len(template["openalex"]) == 5
            assert "instructions" in template
    
    def test_create_template_handles_missing_doi(self):
        """Test template handles results without DOI."""
        crossref_results = [
            {"title": "Paper without DOI"}
        ]
        openalex_results = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            create_relevance_label_template(crossref_results, openalex_results, temp_dir)
            
            template_path = os.path.join(temp_dir, "relevance_labels_template.json")
            with open(template_path, 'r') as f:
                template = json.load(f)
            
            assert "no_doi_0" in template["crossref"]
            assert template["crossref"]["no_doi_0"]["title"] == "Paper without DOI"


class TestCalculateMetricsFromLabels:
    """Tests for metric calculation from labels."""
    
    def test_calculate_metrics_requires_labels(self):
        """Test calculation fails without labels file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create results but no labels
            crossref_data = {"results": [{"title": "Paper 1", "doi": "10.1234/1"}]}
            openalex_data = {"results": [{"title": "Paper 2", "doi": "10.5678/1"}]}
            
            with open(os.path.join(temp_dir, "crossref_final.json"), 'w') as f:
                json.dump(crossref_data, f)
            with open(os.path.join(temp_dir, "openalex_final.json"), 'w') as f:
                json.dump(openalex_data, f)
            
            metrics = calculate_metrics_from_labels(temp_dir)
            
            assert "error" in metrics
            assert "relevance_labels.json not found" in metrics["error"]
    
    def test_calculate_metrics_with_labels(self):
        """Test calculation with complete labels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create results
            crossref_data = {
                "results": [
                    {"title": "Paper 1", "doi": "10.1234/1", "topic_overlap_score": 0.8},
                    {"title": "Paper 2", "doi": "10.1234/2", "topic_overlap_score": 0.6},
                ]
            }
            openalex_data = {
                "results": [
                    {"title": "Paper 3", "doi": "10.5678/1", "topic_overlap_score": 0.7},
                    {"title": "Paper 4", "doi": "10.5678/2", "topic_overlap_score": 0.5},
                ]
            }
            
            with open(os.path.join(temp_dir, "crossref_final.json"), 'w') as f:
                json.dump(crossref_data, f)
            with open(os.path.join(temp_dir, "openalex_final.json"), 'w') as f:
                json.dump(openalex_data, f)
            
            # Create labels
            labels = {
                "crossref": {
                    "10.1234/1": {"title": "Paper 1", "relevant": True},
                    "10.1234/2": {"title": "Paper 2", "relevant": False},
                },
                "openalex": {
                    "10.5678/1": {"title": "Paper 3", "relevant": True},
                    "10.5678/2": {"title": "Paper 4", "relevant": False},
                },
            }
            
            with open(os.path.join(temp_dir, "relevance_labels.json"), 'w') as f:
                json.dump(labels, f)
            
            metrics = calculate_metrics_from_labels(temp_dir)
            
            assert metrics["crossref"]["precision_at_5"] == 0.5  # 1/2 relevant
            assert metrics["openalex"]["precision_at_5"] == 0.5  # 1/2 relevant
            assert metrics["crossref"]["duplicate_rate"] is None  # Unavailable
            assert metrics["openalex"]["duplicate_rate"] is None  # Unavailable
            assert "notes" in metrics
            assert "comparison" in metrics
    
    def test_calculate_metrics_marks_duplicate_rate_unavailable(self):
        """Test that duplicate rate is marked as unavailable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            crossref_data = {"results": [{"title": "Paper 1", "doi": "10.1234/1"}]}
            openalex_data = {"results": [{"title": "Paper 2", "doi": "10.5678/1"}]}
            
            with open(os.path.join(temp_dir, "crossref_final.json"), 'w') as f:
                json.dump(crossref_data, f)
            with open(os.path.join(temp_dir, "openalex_final.json"), 'w') as f:
                json.dump(openalex_data, f)
            
            labels = {
                "crossref": {"10.1234/1": {"title": "Paper 1", "relevant": True}},
                "openalex": {"10.5678/1": {"title": "Paper 2", "relevant": True}},
            }
            
            with open(os.path.join(temp_dir, "relevance_labels.json"), 'w') as f:
                json.dump(labels, f)
            
            metrics = calculate_metrics_from_labels(temp_dir)
            
            assert metrics["notes"]["duplicate_rate"] == "Unavailable - providers do not expose raw candidates before deduplication"
            assert metrics["crossref"]["duplicate_rate"] is None
            assert metrics["openalex"]["duplicate_rate"] is None
    
    def test_calculate_metrics_no_labels_comparison_unavailable(self):
        """Test comparison is marked unavailable when no labels provided."""
        with tempfile.TemporaryDirectory() as temp_dir:
            crossref_data = {"results": [{"title": "Paper 1", "doi": "10.1234/1"}]}
            openalex_data = {"results": [{"title": "Paper 2", "doi": "10.5678/1"}]}
            
            with open(os.path.join(temp_dir, "crossref_final.json"), 'w') as f:
                json.dump(crossref_data, f)
            with open(os.path.join(temp_dir, "openalex_final.json"), 'w') as f:
                json.dump(openalex_data, f)
            
            labels = {
                "crossref": {},
                "openalex": {},
            }
            
            with open(os.path.join(temp_dir, "relevance_labels.json"), 'w') as f:
                json.dump(labels, f)
            
            metrics = calculate_metrics_from_labels(temp_dir)
            
            assert metrics["comparison"]["better_precision"] == "unavailable"
    
    def test_calculate_metrics_with_all_null_labels(self):
        """Test calculation when all labels are null (unlabeled)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            crossref_data = {
                "results": [
                    {"title": "Paper 1", "doi": "10.1234/1", "topic_overlap_score": 0.8},
                ]
            }
            openalex_data = {
                "results": [
                    {"title": "Paper 2", "doi": "10.5678/1", "topic_overlap_score": 0.7},
                ]
            }
            
            with open(os.path.join(temp_dir, "crossref_final.json"), 'w') as f:
                json.dump(crossref_data, f)
            with open(os.path.join(temp_dir, "openalex_final.json"), 'w') as f:
                json.dump(openalex_data, f)
            
            # All labels are null (unlabeled)
            labels = {
                "crossref": {"10.1234/1": {"title": "Paper 1", "relevant": None}},
                "openalex": {"10.5678/1": {"title": "Paper 2", "relevant": None}},
            }
            
            with open(os.path.join(temp_dir, "relevance_labels.json"), 'w') as f:
                json.dump(labels, f)
            
            metrics = calculate_metrics_from_labels(temp_dir)
            
            # Should calculate metrics but with no relevant results
            assert "error" not in metrics
            assert metrics["crossref"]["precision_at_5"] == 0.0
            assert metrics["openalex"]["precision_at_5"] == 0.0
            assert metrics["comparison"]["better_precision"] == "unavailable"
    
    def test_calculate_metrics_with_all_null_labels(self):
        """Test calculation when all labels are null (unlabeled)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            crossref_data = {
                "results": [
                    {"title": "Paper 1", "doi": "10.1234/1", "topic_overlap_score": 0.8},
                ]
            }
            openalex_data = {
                "results": [
                    {"title": "Paper 2", "doi": "10.5678/1", "topic_overlap_score": 0.7},
                ]
            }
            
            with open(os.path.join(temp_dir, "crossref_final.json"), 'w') as f:
                json.dump(crossref_data, f)
            with open(os.path.join(temp_dir, "openalex_final.json"), 'w') as f:
                json.dump(openalex_data, f)
            
            # All labels are null (unlabeled)
            labels = {
                "crossref": {"10.1234/1": {"title": "Paper 1", "relevant": None}},
                "openalex": {"10.5678/1": {"title": "Paper 2", "relevant": None}},
            }
            
            with open(os.path.join(temp_dir, "relevance_labels.json"), 'w') as f:
                json.dump(labels, f)
            
            metrics = calculate_metrics_from_labels(temp_dir)
            
            # Should calculate metrics but with no relevant results
            assert "error" not in metrics
            assert metrics["crossref"]["precision_at_5"] == 0.0
            assert metrics["openalex"]["precision_at_5"] == 0.0
            assert metrics["comparison"]["better_precision"] == "unavailable"
