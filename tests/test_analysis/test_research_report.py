import pytest

from analysis.literature_comparator import LiteratureComparator
from analysis.research_gap_analyzer import (
    ResearchGapAnalysisError,
    ResearchGapAnalyzer,
)
from analysis.research_report_generator import (
    ResearchReportError,
    ResearchReportGenerator,
)


def paper(document_id, title, *, limitations, chunk_id, doi=None):
    return {
        "document_id": document_id,
        "title": title,
        "doi": doi,
        "research_problem": "A bounded test problem",
        "methodology": "A comparative method",
        "dataset": "A sample dataset",
        "evaluation_metrics": ["accuracy"],
        "reported_results": "A reported result",
        "limitations": limitations,
        "abstract": "A short paper summary.",
        "authors": ["A. Author"],
        "publication_year": "2024",
        "analysis_coverage": {
            "is_partial": True,
            "omitted_chunks": [{"chunk_id": "omitted-1"}],
        },
        "evidence_citations": {
            "abstract": {
                "evidence_chunks": [{
                    "document_id": document_id,
                    "chunk_id": f"{chunk_id}-abstract",
                    "start_page": 1,
                    "end_page": 1,
                    "text": "The paper studies a bounded sample.",
                }],
                "citations": [{
                    "document_id": document_id,
                    "chunk_id": f"{chunk_id}-abstract",
                    "start_page": 1,
                    "end_page": 1,
                }],
            },
            "limitations": {
                "evidence_type": "explicit_fact",
                "evidence_sufficiency": "sufficient",
                "evidence_chunks": [{
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "start_page": 4,
                    "end_page": 5,
                    "text": "The study was limited by a small sample size.",
                }],
                "citations": [{
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "start_page": 4,
                    "end_page": 5,
                }],
            },
            "reported_results": {
                "evidence_chunks": [{
                    "document_id": document_id,
                    "chunk_id": f"{chunk_id}-results",
                    "start_page": 6,
                    "end_page": 6,
                    "text": "The paper reports a test result.",
                }],
                "citations": [{
                    "document_id": document_id,
                    "chunk_id": f"{chunk_id}-results",
                    "start_page": 6,
                    "end_page": 6,
                }],
            },
        },
    }


def test_gap_analysis_emits_limited_follow_up_candidates_with_citations():
    papers = [
        paper(
            "doc-a",
            "Study A",
            limitations=["Small sample size"],
            chunk_id="chunk-a",
        ),
        paper(
            "doc-b",
            "Study B",
            limitations=["Limited geographic coverage"],
            chunk_id="chunk-b",
        ),
    ]
    comparison = LiteratureComparator().compare_papers(papers)

    result = ResearchGapAnalyzer().analyze(papers, comparison)

    assert result["candidate_count"] == 2
    assert result["global_novelty_claim"] is False
    assert result["requires_human_review"] is True
    assert result["research_gap_candidates"][0]["type"] == (
        "follow_up_from_stated_limitation"
    )
    assert result["research_gap_candidates"][0]["citations"] == [{
        "document_id": "doc-a",
        "chunk_id": "chunk-a",
        "start_page": 4,
        "end_page": 5,
    }]
    assert result["research_gap_candidates"][0]["evidence_status"] == "supported"
    assert result["research_gap_candidates"][1]["evidence_status"] == "unsupported"
    assert "not an exhaustive search" in result["scope_warning"]


def test_gap_analysis_does_not_treat_missing_fields_as_scientific_gaps():
    analysis = paper(
        "doc-a",
        "Study A",
        limitations=None,
        chunk_id="chunk-a",
    )
    analysis["dataset"] = None
    comparison = LiteratureComparator().compare_papers([analysis])

    result = ResearchGapAnalyzer().analyze([analysis], comparison)

    assert result["research_gap_candidates"] == []
    assert any(
        gap["field"] == "dataset"
        and gap["status"] == "not_extracted_or_unavailable"
        for gap in result["evidence_coverage_gaps"]
    )
    assert "not evidence" in result["evidence_coverage_gaps"][0]["interpretation"]


def test_gap_analysis_ignores_invalid_cross_document_limitation_citation():
    analysis = paper(
        "doc-a",
        "Study A",
        limitations="Short follow-up duration",
        chunk_id="chunk-a",
    )
    analysis["evidence_citations"]["limitations"]["citations"][0][
        "document_id"
    ] = "doc-b"
    comparison = LiteratureComparator().compare_papers([analysis])

    candidate = ResearchGapAnalyzer().analyze(
        [analysis], comparison
    )["research_gap_candidates"][0]

    assert candidate["citations"] == []
    assert candidate["evidence_status"] == "unresolved"
    assert candidate["requires_human_review"] is True


@pytest.mark.parametrize(
    ("reference_chunk_id", "passage", "expected_status"),
    [
        ("missing-chunk", "The study was limited by a small sample size.", "unresolved"),
        ("chunk-a", "   ", "unresolved"),
        ("chunk-a", "The study was limited by the available budget.", "unsupported"),
        ("chunk-a", "The study was limited by a small sample size.", "supported"),
    ],
)
def test_gap_candidate_requires_exact_resolved_passage(
    reference_chunk_id, passage, expected_status
):
    analysis = paper(
        "doc-a",
        "Study A",
        limitations=["Small sample size"],
        chunk_id="chunk-a",
    )
    citation = analysis["evidence_citations"]["limitations"]
    citation["citations"][0]["chunk_id"] = reference_chunk_id
    citation["evidence_chunks"][0]["text"] = passage

    candidate = ResearchGapAnalyzer().analyze(
        [analysis], LiteratureComparator().compare_papers([analysis])
    )["research_gap_candidates"][0]

    assert candidate["evidence_status"] == expected_status
    assert candidate["requires_human_review"] is True
    assert bool(candidate["citations"]) is (expected_status != "unresolved")


def test_gap_analyzer_validates_input_limits():
    with pytest.raises(ResearchGapAnalysisError, match="papers must"):
        ResearchGapAnalyzer().analyze("not papers", {})
    with pytest.raises(ResearchGapAnalysisError, match="mapping"):
        ResearchGapAnalyzer().analyze([], [])


def test_report_includes_traceable_analysis_and_review_warnings():
    primary = paper(
        "doc-a",
        "A *sample* study",
        limitations=["Small sample size"],
        chunk_id="chunk-a",
    )
    related = [{
        "status": "metadata_only",
        "paper": {
            "title": "Related study",
            "authors": ["B. Researcher"],
            "year": 2023,
            "doi": "10.1234/related",
            "url": "https://example.org/paper",
        },
    }]
    verifications = [{
        "doi": "10.1234/related",
        "status": "partial",
    }]
    comparison = LiteratureComparator().compare_papers([primary])
    gaps = ResearchGapAnalyzer().analyze([primary], comparison)

    report = ResearchReportGenerator().generate(
        primary,
        related_papers=related,
        metadata_verifications=verifications,
        comparison=comparison,
        gap_analysis=gaps,
    )

    markdown = report["markdown"]
    assert report["status"] == "generated"
    assert report["contains_global_novelty_claim"] is False
    assert r"A \*sample\* study" in markdown
    assert "doc-a/chunk-a p.4-5" in markdown
    assert "https://doi.org/10.1234/related" in markdown
    assert "Metadata verification: partial" in markdown
    assert "Coverage warning" in markdown
    assert "not validated research gaps" in markdown
    assert "[evidence: supported]" in markdown
    assert "not factual confidence" in markdown
    assert "The study was limited by a small sample size." in markdown
    assert "Summary source: model-extracted abstract." in markdown
    assert "The paper studies a bounded sample." in markdown


def test_report_rejects_non_mapping_inputs_and_unsafe_source_links():
    generator = ResearchReportGenerator()
    with pytest.raises(ResearchReportError, match="primary_analysis"):
        generator.generate([])
    with pytest.raises(ResearchReportError, match="related_papers"):
        generator.generate({}, related_papers=["not a mapping"])

    report = generator.generate(
        {"title": "Example"},
        related_papers=[{
            "paper": {
                "title": "Unsafe link",
                "url": "javascript:alert(1)",
            },
        }],
    )
    assert "javascript:" not in report["markdown"]


def test_streamlit_workflow_composes_report_without_network_calls(
    monkeypatch, tmp_path
):
    import app

    analysis = paper(
        "uploaded-doc",
        "Uploaded study",
        limitations=["Small sample size"],
        chunk_id="uploaded-chunk",
    )

    class FakeExtractor:
        def extract_from_file(self, _path):
            return {"total_characters": 100, "pages": []}

    class FakeAnalyzer:
        def __init__(self, *_args, **_kwargs):
            pass

        def analyze_paper(self, *_args, **_kwargs):
            return analysis

        def compare_papers(self, inputs):
            return LiteratureComparator().compare_papers(inputs)

    monkeypatch.setattr(app, "TextExtractor", FakeExtractor)
    monkeypatch.setattr(app, "TextChunker", lambda: object())
    monkeypatch.setattr(app, "EmbeddingGenerator", lambda: object())
    monkeypatch.setattr(app, "SemanticRetriever", lambda _embedder: object())
    monkeypatch.setattr(app, "AnswerGenerator", lambda _retriever: object())
    monkeypatch.setattr(app, "PaperAnalyzer", FakeAnalyzer)

    bundle = app.run_analysis(
        str(tmp_path / "sample.pdf"),
        "uploaded-doc",
        discover_related=False,
        analyze_related=False,
    )

    assert bundle["analysis"]["document_id"] == "uploaded-doc"
    assert bundle["related_papers"] == []
    assert bundle["comparison"]["source_count"] == 1
    assert bundle["gap_analysis"]["candidate_count"] == 1
    assert bundle["report"]["status"] == "generated"
    assert "Small sample size" in bundle["report"]["markdown"]


def test_related_paper_screen_hides_low_topic_overlap_from_processing():
    import app

    included, excluded = app._partition_related_papers([
        {
            "title": "Credit Card Debt Survey",
            "topic_overlap_score": 0.4,
        },
        {
            "title": "An Unrelated Image Processing Method",
            "topic_overlap_score": 0.0,
        },
    ])

    assert [paper["title"] for paper in included] == [
        "Credit Card Debt Survey"
    ]
    assert excluded[0]["screening_status"] == "screened_out"
    assert "keyword overlap" in excluded[0]["screening_reason"]


def test_related_metadata_is_not_compared_unless_paper_was_analyzed(
    monkeypatch, tmp_path
):
    import app

    primary = paper(
        "uploaded-doc",
        "Banking and credit card services",
        limitations=[],
        chunk_id="uploaded-chunk",
    )
    related = {
        "title": "Credit Card Debt Survey",
        "doi": "10.1234/related",
        "topic_overlap_score": 0.5,
    }
    unrelated = {
        "title": "Unrelated image processing",
        "doi": "10.1234/unrelated",
        "topic_overlap_score": 0.0,
    }
    analyzed_inputs = []

    class FakeExtractor:
        def extract_from_file(self, _path):
            return {"total_characters": 100, "pages": []}

    class FakeAnalyzer:
        def __init__(self, *_args, **_kwargs):
            pass

        def analyze_paper(self, *_args, **_kwargs):
            return primary

        def search_related_papers(self, *_args, **_kwargs):
            return [related.copy(), unrelated.copy()]

        def verify_paper_metadata(self, candidate):
            return {"status": "verified", "doi": candidate.get("doi")}

        def extract_related_paper_evidence(self, candidates, max_papers):
            assert [item["title"] for item in candidates] == [
                "Credit Card Debt Survey"
            ]
            analyzed = {
                "status": "analyzed",
                "analysis": paper(
                    "related-doc",
                    "Credit Card Debt Survey",
                    limitations=[],
                    chunk_id="related-chunk",
                ),
            }
            return [analyzed]

        def compare_papers(self, inputs):
            analyzed_inputs.extend(inputs)
            return LiteratureComparator().compare_papers(inputs)

    monkeypatch.setattr(app, "TextExtractor", FakeExtractor)
    monkeypatch.setattr(app, "TextChunker", lambda: object())
    monkeypatch.setattr(app, "EmbeddingGenerator", lambda: object())
    monkeypatch.setattr(app, "SemanticRetriever", lambda _embedder: object())
    monkeypatch.setattr(app, "AnswerGenerator", lambda _retriever: object())
    monkeypatch.setattr(app, "PaperAnalyzer", FakeAnalyzer)

    result = app.run_analysis(
        str(tmp_path / "sample.pdf"),
        "uploaded-doc",
        discover_related=True,
        analyze_related=True,
    )

    assert [item["title"] for item in result["related_papers"]] == [
        "Credit Card Debt Survey"
    ]
    assert [item["title"] for item in result["screened_out_papers"]] == [
        "Unrelated image processing"
    ]
    assert len(analyzed_inputs) == 2
    assert result["comparison"]["source_count"] == 2


def test_evidence_review_checks_chunk_page_and_numbers_without_asserting_truth():
    import app

    citation = {
        "citations": [{
            "document_id": "paper-a",
            "chunk_id": "chunk-1",
            "start_page": 3,
            "end_page": 3,
        }],
        "evidence_chunks": [{
            "document_id": "paper-a",
            "chunk_id": "chunk-1",
            "start_page": 3,
            "end_page": 3,
            "text": "The survey collected responses from 100 participants.",
        }],
    }

    matching = app._review_field_evidence(
        "Responses from 100 participants", citation
    )
    missing_number = app._review_field_evidence(
        "Responses from 150 participants", citation
    )
    citation["citations"][0]["start_page"] = 4
    mismatched_page = app._review_field_evidence(
        "Responses from 100 participants", citation
    )

    assert matching["reference_status"] == "consistent_with_retrieved_chunks"
    assert matching["numeric_status"] == "all_numeric_values_found"
    assert matching["human_review_required"] is True
    assert missing_number["numeric_status"] == "numeric_values_missing"
    assert missing_number["missing_numbers"] == ["150"]
    assert mismatched_page["reference_status"] == "missing_or_inconsistent"


def test_streamlit_workflow_fails_before_backend_initialization_on_empty_pdf(
    monkeypatch,
):
    import app

    class FakeExtractor:
        def extract_from_file(self, _path):
            return {"total_characters": 0}

    monkeypatch.setattr(app, "TextExtractor", FakeExtractor)
    monkeypatch.setattr(
        app,
        "EmbeddingGenerator",
        lambda: pytest.fail("No model should be initialized for an empty PDF"),
    )

    with pytest.raises(ValueError, match="no extractable text"):
        app.run_analysis(
            "empty.pdf",
            "empty-doc",
            discover_related=False,
            analyze_related=False,
        )
