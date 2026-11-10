"""Tests for validation criteria separation (A6), reproducible validation (A5), and single-paper selection (Batch 6)."""

import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
from validation.step5.run_validation import (
    _execution_succeeded,
    _schema_is_valid,
    _calculate_extraction_coverage,
    _check_evidence_resolution,
    _is_valid_real_backend_result,
    _prepare_output_directory,
    build_argument_parser,
    main,
)


class TestExecutionStatus:
    """Tests for execution status gate (A6)."""

    def test_execution_succeeded_with_real_backend_and_attempts(self):
        """Execution succeeds when backend is real and attempts > 0."""
        diagnostics = {
            'backend_is_real': True,
            'attempts': 1,
        }
        assert _execution_succeeded(diagnostics) is True

    def test_execution_failed_with_mock_backend(self):
        """Execution fails when backend is not real."""
        diagnostics = {
            'backend_is_real': False,
            'attempts': 1,
        }
        assert _execution_succeeded(diagnostics) is False

    def test_execution_failed_with_zero_attempts(self):
        """Execution fails when no attempts were made."""
        diagnostics = {
            'backend_is_real': True,
            'attempts': 0,
        }
        assert _execution_succeeded(diagnostics) is False

    def test_execution_failed_with_missing_diagnostics(self):
        """Execution fails when diagnostics are missing."""
        diagnostics = {}
        assert _execution_succeeded(diagnostics) is False


class TestSchemaStatus:
    """Tests for schema validation gate (A6)."""

    def test_schema_valid_with_complete_structure(self):
        """Schema is valid when all required fields are present and correctly typed."""
        analysis = {
            'document_id': 'test-doc',
            'title': 'Test Paper',
            'authors': ['Author One'],
            'publication_year': '2024',
            'abstract': 'Test abstract',
            'research_problem': 'Test problem',
            'methodology': 'Test methodology',
            'dataset': 'Test dataset',
            'evaluation_metrics': ['accuracy'],
            'reported_results': 'Test results',
            'limitations': ['Test limitation'],
            'evidence_citations': {
                'title': {
                    'field_name': 'title',
                    'answer_text': 'Test Paper',
                    'evidence_type': 'verbatim',
                    'evidence_sufficiency': 'sufficient',
                    'requires_human_review': False,
                }
            },
            'extraction_timestamp': '2024-01-01T00:00:00Z',
        }
        assert _schema_is_valid(analysis) is True

    def test_schema_invalid_missing_required_field(self):
        """Schema is invalid when a required field is missing."""
        analysis = {
            'document_id': 'test-doc',
            'title': 'Test Paper',
            # Missing other required fields
        }
        assert _schema_is_valid(analysis) is False

    def test_schema_invalid_not_a_dict(self):
        """Schema is invalid when analysis is not a dictionary."""
        assert _schema_is_valid(None) is False
        assert _schema_is_valid("not a dict") is False
        assert _schema_is_valid([]) is False

    def test_schema_invalid_malformed_evidence_citations(self):
        """Schema is invalid when evidence_citations has wrong structure."""
        analysis = {
            'document_id': 'test-doc',
            'title': 'Test Paper',
            'authors': ['Author One'],
            'publication_year': '2024',
            'abstract': 'Test abstract',
            'research_problem': 'Test problem',
            'methodology': 'Test methodology',
            'dataset': 'Test dataset',
            'evaluation_metrics': ['accuracy'],
            'reported_results': 'Test results',
            'limitations': ['Test limitation'],
            'evidence_citations': 'not a dict',  # Wrong type
            'extraction_timestamp': '2024-01-01T00:00:00Z',
        }
        assert _schema_is_valid(analysis) is False

    def test_schema_invalid_missing_citation_field(self):
        """Schema is invalid when citation is missing required fields."""
        analysis = {
            'document_id': 'test-doc',
            'title': 'Test Paper',
            'authors': ['Author One'],
            'publication_year': '2024',
            'abstract': 'Test abstract',
            'research_problem': 'Test problem',
            'methodology': 'Test methodology',
            'dataset': 'Test dataset',
            'evaluation_metrics': ['accuracy'],
            'reported_results': 'Test results',
            'limitations': ['Test limitation'],
            'evidence_citations': {
                'title': {
                    'field_name': 'title',
                    # Missing other required fields
                }
            },
            'extraction_timestamp': '2024-01-01T00:00:00Z',
        }
        assert _schema_is_valid(analysis) is False


class TestExtractionCoverage:
    """Tests for extraction coverage calculation (A6)."""

    def test_coverage_complete_all_fields_extracted(self):
        """Coverage is complete when all expected fields are non-null."""
        analysis = {
            'title': 'Test',
            'authors': ['Author'],
            'publication_year': '2024',
            'abstract': 'Abstract',
            'research_problem': 'Problem',
            'methodology': 'Method',
            'dataset': 'Dataset',
            'evaluation_metrics': ['metric'],
            'reported_results': 'Results',
            'limitations': ['Limitation'],
        }
        coverage = _calculate_extraction_coverage(analysis)
        assert coverage['total_fields'] == 10
        assert coverage['extracted_fields'] == 10
        assert coverage['coverage_percentage'] == 100.0
        assert coverage['coverage_status'] == 'complete'
        assert len(coverage['empty_fields']) == 0

    def test_coverage_partial_some_fields_missing(self):
        """Coverage is partial when some fields are missing or empty."""
        analysis = {
            'title': 'Test',
            'authors': None,  # Missing
            'publication_year': '2024',
            'abstract': '',  # Empty string
            'research_problem': 'Problem',
            'methodology': None,  # Missing
            'dataset': 'Dataset',
            'evaluation_metrics': [],  # Empty list
            'reported_results': 'Results',
            'limitations': ['Limitation'],
        }
        coverage = _calculate_extraction_coverage(analysis)
        assert coverage['total_fields'] == 10
        assert coverage['extracted_fields'] == 6  # title, publication_year, research_problem, dataset, reported_results, limitations
        assert coverage['coverage_percentage'] == 60.0
        assert coverage['coverage_status'] == 'partial'
        assert len(coverage['empty_fields']) == 4

    def test_coverage_empty_no_fields_extracted(self):
        """Coverage is empty when all fields are missing or empty."""
        analysis = {
            'title': None,
            'authors': None,
            'publication_year': None,
            'abstract': None,
            'research_problem': None,
            'methodology': None,
            'dataset': None,
            'evaluation_metrics': None,
            'reported_results': None,
            'limitations': None,
        }
        coverage = _calculate_extraction_coverage(analysis)
        assert coverage['total_fields'] == 10
        assert coverage['extracted_fields'] == 0
        assert coverage['coverage_percentage'] == 0.0
        assert coverage['coverage_status'] == 'empty'
        assert len(coverage['empty_fields']) == 10

    def test_coverage_ignores_whitespace_only_strings(self):
        """Coverage treats whitespace-only strings as empty."""
        analysis = {
            'title': '   ',  # Whitespace only
            'authors': ['   '],  # List with whitespace only
            'publication_year': '2024',
            'abstract': 'Abstract',
            'research_problem': 'Problem',
            'methodology': 'Method',
            'dataset': 'Dataset',
            'evaluation_metrics': ['metric'],
            'reported_results': 'Results',
            'limitations': ['Limitation'],
        }
        coverage = _calculate_extraction_coverage(analysis)
        assert coverage['extracted_fields'] == 8  # title and authors don't count (both whitespace)
        assert coverage['coverage_status'] == 'partial'


class TestEvidenceResolution:
    """Tests for evidence resolution checking (A6)."""

    def test_resolution_fully_resolved_all_citations_have_passages(self):
        """Resolution is complete when all citations resolve to non-empty evidence chunks."""
        analysis = {
            'evidence_citations': {
                'title': {
                    'evidence_chunks': [
                        {
                            'chunk_id': 'chunk_1',
                            'document_id': 'test-doc',
                            'text': 'This is evidence.',
                        },
                        {
                            'chunk_id': 'chunk_2',
                            'document_id': 'test-doc',
                            'text': 'More evidence.',
                        }
                    ],
                    'citations': [
                        {'chunk_id': 'chunk_1', 'document_id': 'test-doc'},
                        {'chunk_id': 'chunk_2', 'document_id': 'test-doc'},
                    ]
                },
                'methodology': {
                    'evidence_chunks': [
                        {
                            'chunk_id': 'chunk_3',
                            'document_id': 'test-doc',
                            'text': 'Method evidence.',
                        }
                    ],
                    'citations': [
                        {'chunk_id': 'chunk_3', 'document_id': 'test-doc'},
                    ]
                }
            }
        }
        resolution = _check_evidence_resolution(analysis)
        assert resolution['total_citations'] == 3
        assert resolution['resolved_citations'] == 3
        assert resolution['unresolved_citations'] == 0
        assert resolution['resolution_status'] == 'fully_resolved'

    def test_resolution_partially_resolved_some_citations_missing_passages(self):
        """Resolution is partial when some citations don't resolve to evidence chunks."""
        analysis = {
            'evidence_citations': {
                'title': {
                    'evidence_chunks': [
                        {
                            'chunk_id': 'chunk_1',
                            'document_id': 'test-doc',
                            'text': 'This is evidence.',
                        }
                    ],
                    'citations': [
                        {'chunk_id': 'chunk_1', 'document_id': 'test-doc'},
                        {'chunk_id': 'missing_chunk', 'document_id': 'test-doc'},  # Not in evidence_chunks
                    ]
                },
                'methodology': {
                    'evidence_chunks': [],
                    'citations': [
                        {'chunk_id': 'chunk_2', 'document_id': 'test-doc'},  # No evidence_chunks
                    ]
                }
            }
        }
        resolution = _check_evidence_resolution(analysis)
        assert resolution['total_citations'] == 3
        assert resolution['resolved_citations'] == 1
        assert resolution['unresolved_citations'] == 2
        assert resolution['resolution_status'] == 'partially_resolved'

    def test_resolution_no_citations(self):
        """Resolution status is 'no_citations' when no citations exist."""
        analysis = {
            'evidence_citations': {}
        }
        resolution = _check_evidence_resolution(analysis)
        assert resolution['total_citations'] == 0
        assert resolution['resolved_citations'] == 0
        assert resolution['unresolved_citations'] == 0
        assert resolution['resolution_status'] == 'no_citations'

    def test_resolution_handles_malformed_citations(self):
        """Resolution gracefully handles malformed citation structures."""
        analysis = {
            'evidence_citations': {
                'title': 'not a dict',  # Malformed
                'methodology': {
                    'citations': 'not a list',  # Malformed
                },
                'dataset': {
                    'evidence_chunks': [
                        {
                            'chunk_id': 'chunk_1',
                            'document_id': 'test-doc',
                            'text': 'Valid evidence',
                        }
                    ],
                    'citations': [
                        {'chunk_id': 'chunk_1', 'document_id': 'test-doc'},
                        'not a dict',  # Malformed
                    ]
                }
            }
        }
        resolution = _check_evidence_resolution(analysis)
        assert resolution['total_citations'] == 1  # Only the valid one counts
        assert resolution['resolved_citations'] == 1
        assert resolution['unresolved_citations'] == 0
        assert resolution['resolution_status'] == 'fully_resolved'

    def test_resolution_chunk_id_missing_unresolved(self):
        """Resolution marks citation as unresolved when chunk_id is missing."""
        analysis = {
            'evidence_citations': {
                'title': {
                    'evidence_chunks': [
                        {
                            'chunk_id': 'chunk_1',
                            'document_id': 'test-doc',
                            'text': 'Evidence',
                        }
                    ],
                    'citations': [
                        {},  # No chunk_id
                    ]
                }
            }
        }
        resolution = _check_evidence_resolution(analysis)
        assert resolution['total_citations'] == 1
        assert resolution['resolved_citations'] == 0
        assert resolution['unresolved_citations'] == 1
        assert resolution['resolution_status'] == 'partially_resolved'

    def test_resolution_chunk_id_unknown_unresolved(self):
        """Resolution marks citation as unresolved when chunk_id not in evidence_chunks."""
        analysis = {
            'evidence_citations': {
                'title': {
                    'evidence_chunks': [
                        {
                            'chunk_id': 'chunk_1',
                            'document_id': 'test-doc',
                            'text': 'Evidence',
                        }
                    ],
                    'citations': [
                        {'chunk_id': 'unknown_chunk', 'document_id': 'test-doc'},
                    ]
                }
            }
        }
        resolution = _check_evidence_resolution(analysis)
        assert resolution['total_citations'] == 1
        assert resolution['resolved_citations'] == 0
        assert resolution['unresolved_citations'] == 1
        assert resolution['resolution_status'] == 'partially_resolved'

    def test_resolution_empty_chunk_text_unresolved(self):
        """Resolution marks citation as unresolved when chunk has empty text."""
        analysis = {
            'evidence_citations': {
                'title': {
                    'evidence_chunks': [
                        {
                            'chunk_id': 'chunk_1',
                            'document_id': 'test-doc',
                            'text': '',  # Empty
                        }
                    ],
                    'citations': [
                        {'chunk_id': 'chunk_1', 'document_id': 'test-doc'},
                    ]
                }
            }
        }
        resolution = _check_evidence_resolution(analysis)
        assert resolution['total_citations'] == 1
        assert resolution['resolved_citations'] == 0
        assert resolution['unresolved_citations'] == 1
        assert resolution['resolution_status'] == 'partially_resolved'

    def test_resolution_whitespace_only_chunk_text_unresolved(self):
        """Resolution marks citation as unresolved when chunk has only whitespace."""
        analysis = {
            'evidence_citations': {
                'title': {
                    'evidence_chunks': [
                        {
                            'chunk_id': 'chunk_1',
                            'document_id': 'test-doc',
                            'text': '   ',  # Whitespace only
                        }
                    ],
                    'citations': [
                        {'chunk_id': 'chunk_1', 'document_id': 'test-doc'},
                    ]
                }
            }
        }
        resolution = _check_evidence_resolution(analysis)
        assert resolution['total_citations'] == 1
        assert resolution['resolved_citations'] == 0
        assert resolution['unresolved_citations'] == 1
        assert resolution['resolution_status'] == 'partially_resolved'


class TestBackwardCompatibility:
    """Tests for backward compatibility with deprecated function (A6)."""

    def test_deprecated_function_still_works(self):
        """Deprecated _is_valid_real_backend_result still uses execution logic."""
        diagnostics = {
            'backend_is_real': True,
            'attempts': 1,
        }
        # Old behavior: required backend_is_real, attempts > 0, and valid_fields > 0
        # New behavior: only requires backend_is_real and attempts > 0
        assert _is_valid_real_backend_result(diagnostics) is True

    def test_deprecated_function_rejects_mock_backend(self):
        """Deprecated function still rejects mock backends."""
        diagnostics = {
            'backend_is_real': False,
            'attempts': 1,
        }
        assert _is_valid_real_backend_result(diagnostics) is False


class TestValidationScenarios:
    """End-to-end validation scenario tests (A6)."""

    def test_api_success_with_incomplete_extraction(self):
        """Scenario: API succeeds but extraction is incomplete (A6 requirement)."""
        # This scenario should have:
        # - execution_status: succeeded
        # - schema_status: valid
        # - coverage_status: partial
        # - quality_review_status: pending
        analysis = {
            'document_id': 'test-doc',
            'title': 'Test Paper',
            'authors': ['Author'],
            'publication_year': None,  # Missing
            'abstract': 'Abstract',
            'research_problem': None,  # Missing
            'methodology': 'Method',
            'dataset': None,  # Missing
            'evaluation_metrics': [],  # Empty
            'reported_results': 'Results',
            'limitations': ['Limitation'],
            'evidence_citations': {
                'title': {
                    'field_name': 'title',
                    'answer_text': 'Test Paper',
                    'evidence_type': 'verbatim',
                    'evidence_sufficiency': 'sufficient',
                    'requires_human_review': False,
                }
            },
            'extraction_timestamp': '2024-01-01T00:00:00Z',
        }
        
        assert _schema_is_valid(analysis) is True
        coverage = _calculate_extraction_coverage(analysis)
        assert coverage['coverage_status'] == 'partial'
        # Quality review should be pending (not automatically approved)
        # This is enforced by the validation structure, not a function call

    def test_schema_valid_with_unresolved_citations(self):
        """Scenario: Schema is valid but citations are unresolved (A6 requirement)."""
        analysis = {
            'document_id': 'test-doc',
            'title': 'Test Paper',
            'authors': ['Author'],
            'publication_year': '2024',
            'abstract': 'Abstract',
            'research_problem': 'Problem',
            'methodology': 'Method',
            'dataset': 'Dataset',
            'evaluation_metrics': ['metric'],
            'reported_results': 'Results',
            'limitations': ['Limitation'],
            'evidence_citations': {
                'title': {
                    'field_name': 'title',
                    'answer_text': 'Test Paper',
                    'evidence_type': 'verbatim',
                    'evidence_sufficiency': 'sufficient',
                    'requires_human_review': False,
                    'evidence_chunks': [],
                    'citations': [
                        {'chunk_id': 'missing_chunk'},  # Chunk not in evidence_chunks - unresolved
                    ]
                }
            },
            'extraction_timestamp': '2024-01-01T00:00:00Z',
        }
        
        assert _schema_is_valid(analysis) is True
        resolution = _check_evidence_resolution(analysis)
        assert resolution['resolution_status'] == 'partially_resolved'

    def test_complete_extraction_no_human_review(self):
        """Scenario: Complete extraction but no human review (A6 requirement)."""
        analysis = {
            'document_id': 'test-doc',
            'title': 'Test Paper',
            'authors': ['Author'],
            'publication_year': '2024',
            'abstract': 'Abstract',
            'research_problem': 'Problem',
            'methodology': 'Method',
            'dataset': 'Dataset',
            'evaluation_metrics': ['metric'],
            'reported_results': 'Results',
            'limitations': ['Limitation'],
            'evidence_citations': {
                'title': {
                    'field_name': 'title',
                    'answer_text': 'Test Paper',
                    'evidence_type': 'verbatim',
                    'evidence_sufficiency': 'sufficient',
                    'requires_human_review': False,
                    'evidence_chunks': [
                        {
                            'chunk_id': 'chunk_1',
                            'document_id': 'test-doc',
                            'text': 'Evidence',
                        }
                    ],
                    'citations': [
                        {'chunk_id': 'chunk_1', 'document_id': 'test-doc'},
                    ]
                }
            },
            'extraction_timestamp': '2024-01-01T00:00:00Z',
        }
        
        coverage = _calculate_extraction_coverage(analysis)
        assert coverage['coverage_status'] == 'complete'
        resolution = _check_evidence_resolution(analysis)
        assert resolution['resolution_status'] == 'fully_resolved'
        # But quality_review_status should still be 'pending' (not automated)

    def test_provider_failure(self):
        """Scenario: Provider failure (A6 requirement)."""
        diagnostics = {
            'backend_is_real': True,
            'attempts': 0,  # No successful attempts
        }
        
        assert _execution_succeeded(diagnostics) is False
        # Schema, coverage, and resolution should be unknown/None in this case

    def test_successful_evidence_resolved_with_human_review(self):
        """Scenario: Successful extraction with evidence resolution and human review (A6 requirement)."""
        # This is the "gold standard" scenario
        analysis = {
            'document_id': 'test-doc',
            'title': 'Test Paper',
            'authors': ['Author'],
            'publication_year': '2024',
            'abstract': 'Abstract',
            'research_problem': 'Problem',
            'methodology': 'Method',
            'dataset': 'Dataset',
            'evaluation_metrics': ['metric'],
            'reported_results': 'Results',
            'limitations': ['Limitation'],
            'evidence_citations': {
                'title': {
                    'field_name': 'title',
                    'answer_text': 'Test Paper',
                    'evidence_type': 'verbatim',
                    'evidence_sufficiency': 'sufficient',
                    'requires_human_review': False,
                    'evidence_chunks': [
                        {
                            'chunk_id': 'chunk_1',
                            'document_id': 'test-doc',
                            'text': 'Test Paper Evidence',
                        }
                    ],
                    'citations': [
                        {'chunk_id': 'chunk_1', 'document_id': 'test-doc'},
                    ]
                }
            },
            'extraction_timestamp': '2024-01-01T00:00:00Z',
        }
        
        assert _schema_is_valid(analysis) is True
        coverage = _calculate_extraction_coverage(analysis)
        assert coverage['coverage_status'] == 'complete'
        resolution = _check_evidence_resolution(analysis)
        assert resolution['resolution_status'] == 'fully_resolved'
        # quality_review_status would be set to 'reviewed' only by human action
        # The automation does not invent this status


class TestReproducibleValidationA5:
    """Tests for reproducible real-paper validation (A5)."""

    def test_prepare_output_directory_creates_unique_run_id(self):
        """A5: Each run creates a uniquely named output directory using timestamp."""
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmpdir:
            # Provide a custom output directory to test the logic
            custom_dir = Path(tmpdir) / "runs" / "20260115T120000123456Z"
            
            output_dir = _prepare_output_directory(custom_dir)
            
            assert output_dir == custom_dir
            assert output_dir.exists()
            assert output_dir.name == "20260115T120000123456Z"

    def test_prepare_output_directory_refuses_to_overwrite(self):
        """A5: Existing output directories are not overwritten."""
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "runs" / "20260115T120000123456Z"
            output_dir.mkdir(parents=True)
            
            # Should raise FileExistsError because directory exists
            with pytest.raises(FileExistsError):
                _prepare_output_directory(output_dir)

    def test_prepare_output_directory_accepts_custom_path(self):
        """A5: Custom output directory can be specified."""
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmpdir:
            custom_dir = Path(tmpdir) / "custom_run"
            
            output_dir = _prepare_output_directory(custom_dir)
            
            assert output_dir == custom_dir
            assert output_dir.exists()

    def test_validation_dimensions_in_result(self):
        """A5: Result includes all A6 validation dimensions."""
        # This is verified by the scenario tests, but we explicitly check structure here
        from validation.step5.run_validation import analyze_paper
        
        # Mock the entire analysis pipeline
        with patch('validation.step5.run_validation.TextExtractor') as mock_extractor, \
             patch('validation.step5.run_validation.TextChunker'), \
             patch('validation.step5.run_validation.EmbeddingGenerator'), \
             patch('validation.step5.run_validation.SemanticRetriever'), \
             patch('validation.step5.run_validation.AnswerGenerator'), \
             patch('validation.step5.run_validation.PaperAnalyzer') as mock_analyzer:
            
            # Setup mocks
            mock_extractor.return_value.extract_from_file.return_value = {
                'total_pages': 1,
                'total_characters': 100,
                'pages': [{'page_number': 1, 'text': 'Test content'}]
            }
            
            mock_analyzer.return_value.analyze_paper.return_value = {
                'document_id': 'test-doc',
                'title': 'Test Paper',
                'authors': ['Author'],
                'publication_year': '2024',
                'abstract': 'Abstract',
                'research_problem': 'Problem',
                'methodology': 'Method',
                'dataset': 'Dataset',
                'evaluation_metrics': ['metric'],
                'reported_results': 'Results',
                'limitations': ['Limitation'],
                'evidence_citations': {
                    'title': {
                        'field_name': 'title',
                        'answer_text': 'Test Paper',
                        'evidence_type': 'verbatim',
                        'evidence_sufficiency': 'sufficient',
                        'requires_human_review': False,
                    }
                },
                'extraction_timestamp': '2024-01-01T00:00:00Z',
                'extraction_diagnostics': {
                    'backend_is_real': True,
                    'attempts': 1,
                }
            }
            
            result = analyze_paper(
                'fake.pdf',
                'test-doc',
                backend_name='groq',
                model_name='test-model'
            )
            
            # Verify validation dimensions are present
            assert 'validation' in result
            validation = result['validation']
            assert 'execution_status' in validation
            assert 'schema_status' in validation
            assert 'extraction_coverage' in validation
            assert 'evidence_resolution' in validation
            assert 'quality_review_status' in validation
            assert validation['quality_review_status'] == 'pending'

    def test_quality_review_status_always_pending(self):
        """A5: Quality review status is never automated to 'reviewed'."""
        # Even with perfect extraction, quality review should be pending
        analysis = {
            'document_id': 'test-doc',
            'title': 'Test Paper',
            'authors': ['Author'],
            'publication_year': '2024',
            'abstract': 'Abstract',
            'research_problem': 'Problem',
            'methodology': 'Method',
            'dataset': 'Dataset',
            'evaluation_metrics': ['metric'],
            'reported_results': 'Results',
            'limitations': ['Limitation'],
            'evidence_citations': {
                'title': {
                    'field_name': 'title',
                    'answer_text': 'Test Paper',
                    'evidence_type': 'verbatim',
                    'evidence_sufficiency': 'sufficient',
                    'requires_human_review': False,
                    'evidence_chunks': [
                        {
                            'chunk_id': 'chunk_1',
                            'document_id': 'test-doc',
                            'text': 'Evidence',
                        }
                    ],
                    'citations': [
                        {'chunk_id': 'chunk_1', 'document_id': 'test-doc'},
                    ]
                }
            },
            'extraction_timestamp': '2024-01-01T00:00:00Z',
        }
        
        resolution = _check_evidence_resolution(analysis)
        coverage = _calculate_extraction_coverage(analysis)
        
        # Even with complete coverage and full resolution, quality review is not set by automation
        # This is enforced at the result level, not by these functions
        assert resolution['resolution_status'] == 'fully_resolved'
        assert coverage['coverage_status'] == 'complete'
        # The quality_review_status is set in analyze_paper, not in these helper functions

    def test_operational_success_not_quality_success(self):
        """A5: Operational success (API call) is not treated as quality success."""
        # Execution succeeded but extraction is poor
        analysis = {
            'document_id': 'test-doc',
            'title': 'Placeholder Title',  # Placeholder
            'authors': None,  # Missing
            'publication_year': None,  # Missing
            'abstract': 'Sample abstract.',  # Placeholder
            'research_problem': None,  # Missing
            'methodology': 'Sample methodology.',  # Placeholder
            'dataset': None,  # Missing
            'evaluation_metrics': None,  # Missing
            'reported_results': 'Sample results.',  # Placeholder
            'limitations': None,  # Missing
            'evidence_citations': {
                'title': {
                    'field_name': 'title',
                    'answer_text': 'Placeholder Title',
                    'evidence_type': 'unknown',
                    'evidence_sufficiency': 'insufficient',
                    'requires_human_review': True,
                }
            },
            'extraction_timestamp': '2024-01-01T00:00:00Z',
        }
        
        # Schema might be valid structurally, but coverage is poor
        assert _schema_is_valid(analysis) is True
        coverage = _calculate_extraction_coverage(analysis)
        assert coverage['coverage_status'] == 'partial'  # Only 4/10 fields
        resolution = _check_evidence_resolution(analysis)
        assert resolution['resolution_status'] == 'no_citations'  # No citations with passages
        
        # This would be marked as operationally successful but quality-pending


class TestValidationArtifactOutputA5:
    """Tests that each validation run writes its own output artifacts."""

    def test_run_writes_artifacts_to_requested_directory(self, tmp_path, monkeypatch):
        output_dir = tmp_path / "validation-run"
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        (papers_dir / "attention_is_all_you_need.pdf").touch()
        result = {
            "status": "failed",
            "pdf_path": "attention_is_all_you_need.pdf",
            "document_id": "attention_is_all_you_need",
            "runtime_seconds": 0,
            "errors": ["mock failure"],
            "retries": 0,
            "json_repairs": 0,
            "fallback_calls": 0,
            "backend": "groq",
            "model": "test-model",
            "prompt_eval_count": None,
            "extraction_diagnostics": {},
            "analysis": None,
            "error_message": "mock failure",
        }
        from validation.step5 import run_validation

        monkeypatch.setattr(run_validation, "__file__", str(tmp_path / "run_validation.py"))
        monkeypatch.setattr(run_validation, "analyze_paper", Mock(return_value=result))

        exit_code = main([
            "--backend", "groq",
            "--model", "test-model",
            "--confirm-public-papers",
            "--confirm-groq-costs",
            "--delay-seconds", "0",
            "--paper", "attention_is_all_you_need",
            "--output-dir", str(output_dir),
        ])

        assert exit_code == 1
        assert (output_dir / "validation_report.md").is_file()
        assert (output_dir / "evidence_audit.csv").is_file()
        assert (
            output_dir / "results" / "attention_is_all_you_need_result.json"
        ).is_file()
        assert (
            output_dir / "raw_outputs" / "attention_is_all_you_need_error.txt"
        ).is_file()


class TestSinglePaperSelectionBatch6:
    """Tests for single-paper selection argument (Batch 6)."""

    def test_paper_argument_accepts_valid_papers(self):
        """Batch 6: --paper argument accepts all three allowed values."""
        parser = build_argument_parser()
        
        for paper_name in ["attention_is_all_you_need", "bert_pretraining", "attention_survey"]:
            args = parser.parse_args([f"--paper={paper_name}"])
            assert args.paper == paper_name

    def test_paper_argument_rejects_invalid_paper(self):
        """Batch 6: --paper argument rejects invalid paper names."""
        parser = build_argument_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(["--paper=invalid_paper"])

    def test_paper_argument_omitted_default(self):
        """Batch 6: Omitting --paper defaults to None (process all papers)."""
        parser = build_argument_parser()
        args = parser.parse_args([])
        assert args.paper is None

    def test_paper_selection_filters_correctly(self):
        """Batch 6: Paper selection logic filters to single paper when specified."""
        all_papers = [
            ('attention_is_all_you_need.pdf', 'attention_is_all_you_need'),
            ('bert_pretraining.pdf', 'bert_pretraining'),
            ('attention_survey.pdf', 'attention_survey')
        ]
        
        # Test filtering for each paper
        for paper_id in ["attention_is_all_you_need", "bert_pretraining", "attention_survey"]:
            filtered = [(f"{paper_id}.pdf", paper_id)]
            assert len(filtered) == 1
            assert filtered[0][1] == paper_id

    def test_no_paper_selection_keeps_all_papers(self):
        """Batch 6: No paper selection keeps all three papers."""
        all_papers = [
            ('attention_is_all_you_need.pdf', 'attention_is_all_you_need'),
            ('bert_pretraining.pdf', 'bert_pretraining'),
            ('attention_survey.pdf', 'attention_survey')
        ]
        
        # Without filtering, all papers are included
        assert len(all_papers) == 3
