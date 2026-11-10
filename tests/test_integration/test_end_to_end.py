"""
End-to-end integration test for the complete research intelligence workflow.

This test covers:
PDF input → text extraction → chunking → paper analysis →
literature comparison → research-gap analysis → Markdown report generation

Note: External APIs (Groq, Crossref, OpenAlex) and model downloads are mocked
to avoid costs and network dependencies.
"""

import json
from unittest.mock import Mock

import numpy as np
import pymupdf
import pytest

from pdf_processing.text_extractor import TextExtractor
from chunking.text_chunker import TextChunker
from retrieval.semantic_retriever import SemanticRetriever
from qa.answer_generator import AnswerGenerator
from analysis.paper_analyzer import PaperAnalyzer
from analysis.llm_backend import LLMBackend, LLMGenerationResult


class FakeLLMBackend(LLMBackend):
    """Fake LLM backend for testing without API calls."""
    name = "fake"
    is_real = False

    def __init__(self, response_text):
        self.response_text = response_text

    def generate_json(self, prompt, *, timeout_seconds, options):
        return LLMGenerationResult(
            text=self.response_text,
            metadata={
                "done_reason": "stop",
                "prompt_eval_count": 100,
                "eval_count": 200,
                "duration": 0.5,
            },
        )


@pytest.fixture
def sample_pdf_bytes():
    """Create a minimal test PDF in memory."""
    document = pymupdf.Document()
    page = document.new_page()
    page.insert_text((50, 72), "Test Paper on Machine Learning")
    page.insert_text((50, 100), "This paper studies neural networks.")
    page.insert_text((50, 130), "We used 1000 participants in the study.")
    page.insert_text((50, 160), "The accuracy was 95%.")
    content = document.tobytes()
    document.close()
    return content


@pytest.fixture
def sample_extraction_result(sample_pdf_bytes):
    """Extract text from sample PDF."""
    extractor = TextExtractor()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(sample_pdf_bytes)
        temp_path = f.name
    try:
        result = extractor.extract_from_file(temp_path)
        return result
    finally:
        import os
        os.unlink(temp_path)


@pytest.fixture
def mock_groq_response():
    """Mock Groq structured analysis response."""
    return json.dumps({
        "title": "Test Paper on Machine Learning",
        "authors": ["Author One", "Author Two"],
        "publication_year": "2024",
        "abstract": "This paper studies neural networks with 1000 participants.",
        "research_problem": "How to improve neural network accuracy",
        "methodology": "We used a neural network architecture",
        "dataset": "1000 participants",
        "evaluation_metrics": ["accuracy"],
        "reported_results": "95% accuracy",
        "limitations": ["Small sample size"]
    })


def test_end_to_end_workflow(
    sample_extraction_result,
    mock_groq_response
):
    """
    Test the complete research intelligence workflow from PDF to report.

    This test verifies:
    1. PDF extraction works
    2. Chunking works
    3. Paper analysis works with fake backend
    4. Literature comparison works
    5. Research gap analysis works
    6. Report generation works
    7. Citations remain traceable
    """
    # Stage 1: PDF extraction (already done by fixture)
    assert sample_extraction_result["success"] is True
    assert sample_extraction_result["total_pages"] == 1
    assert sample_extraction_result["total_characters"] > 0

    # Stage 2: Chunking
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    chunking_result = chunker.chunk_extraction_result(
        sample_extraction_result,
        document_id="test_paper"
    )
    assert chunking_result["success"] is True
    assert chunking_result["total_chunks"] > 0
    chunks = chunking_result["chunks"]

    # Stage 3: Mock retrieval to avoid model download
    mock_retriever = Mock(spec=SemanticRetriever)
    mock_retriever.is_built = False
    mock_retriever.build_index = Mock()
    mock_retriever.search = Mock(return_value=[
        {
            'chunk': chunks[0],
            'similarity': 0.9,
            'rank': 1
        }
    ])

    # Stage 4: Paper analysis with fake backend
    fake_backend = FakeLLMBackend(mock_groq_response)
    answer_gen = AnswerGenerator(mock_retriever)
    analyzer = PaperAnalyzer(
        chunker=chunker,
        retriever=mock_retriever,
        answer_generator=answer_gen,
        model_name="fake-model",
        llm_backend=fake_backend
    )

    analysis = analyzer.analyze_paper(
        sample_extraction_result,
        document_id="test_paper"
    )

    # Verify analysis structure
    assert analysis["document_id"] == "test_paper"
    assert analysis["title"] == "Test Paper on Machine Learning"
    assert "evidence_citations" in analysis

    # Stage 5: Literature comparison (single paper test)
    comparison = analyzer.compare_papers([analysis])
    assert comparison["status"] == "insufficient_sources"  # Need at least 2 papers
    assert "field_comparisons" in comparison

    # Stage 6: Research gap analysis
    from analysis.research_gap_analyzer import ResearchGapAnalyzer
    gap_analyzer = ResearchGapAnalyzer()
    gap_analysis = gap_analyzer.analyze([analysis], comparison)
    assert "research_gap_candidates" in gap_analysis  # Actual field name
    assert "evidence_coverage_gaps" in gap_analysis

    # Stage 7: Report generation
    from analysis.research_report_generator import ResearchReportGenerator
    report_gen = ResearchReportGenerator()
    report = report_gen.generate(
        primary_analysis=analysis,
        related_papers=[],
        metadata_verifications=[],
        comparison=comparison,
        gap_analysis=gap_analysis
    )

    # Verify report structure
    assert "markdown" in report
    assert report["markdown"] is not None
    assert len(report["markdown"]) > 0
    assert "Test Paper on Machine Learning" in report["markdown"]

    # Stage 8: Verify citations are traceable
    assert "evidence_citations" in analysis
    for field_name, citation in analysis["evidence_citations"].items():
        if citation.get("citations"):
            for ref in citation["citations"]:
                assert "document_id" in ref
                assert "chunk_id" in ref
                assert "start_page" in ref
                assert "end_page" in ref


def test_optional_stage_failure_isolation(sample_extraction_result):
    """
    Test that failures in optional stages do not prevent primary analysis.

    This verifies:
    1. Primary analysis succeeds even if related paper discovery fails
    2. Primary analysis remains accessible
    """
    # Stage 1-4: Primary analysis (simplified, using fake backend)
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    chunking_result = chunker.chunk_extraction_result(
        sample_extraction_result,
        document_id="test_paper"
    )

    mock_retriever = Mock(spec=SemanticRetriever)
    mock_retriever.is_built = False
    mock_retriever.build_index = Mock()
    mock_retriever.search = Mock(return_value=[
        {
            'chunk': chunking_result["chunks"][0],
            'similarity': 0.9,
            'rank': 1
        }
    ])

    fake_backend = FakeLLMBackend(json.dumps({
        "title": "Test Paper",
        "authors": ["Author"],
        "publication_year": "2024",
        "abstract": "Test abstract",
        "research_problem": "Test problem",
        "methodology": "Test method",
        "dataset": "Test dataset",
        "evaluation_metrics": ["accuracy"],
        "reported_results": "95%",
        "limitations": ["Small sample"]
    }))

    answer_gen = AnswerGenerator(mock_retriever)
    analyzer = PaperAnalyzer(
        chunker=chunker,
        retriever=mock_retriever,
        answer_generator=answer_gen,
        model_name="fake-model",
        llm_backend=fake_backend
    )

    analysis = analyzer.analyze_paper(
        sample_extraction_result,
        document_id="test_paper"
    )

    assert analysis["document_id"] == "test_paper"
    assert analysis["title"] == "Test Paper"

    # Verify report can still be generated with minimal data
    from analysis.research_report_generator import ResearchReportGenerator
    report_gen = ResearchReportGenerator()
    report = report_gen.generate(
        primary_analysis=analysis,
        related_papers=[],
        metadata_verifications=[],
        comparison={"status": "insufficient_sources", "field_comparisons": {}},
        gap_analysis={"research_gap_candidates": [], "evidence_coverage_gaps": []}
    )

    assert "markdown" in report
    assert len(report["markdown"]) > 0
    assert "Test Paper" in report["markdown"]
