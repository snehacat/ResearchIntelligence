"""Tests for deterministic multi-paper comparison."""

import pytest

from analysis.literature_comparator import (
    LiteratureComparator,
    LiteratureComparisonError,
)


@pytest.fixture
def comparator():
    return LiteratureComparator()


def analysis(document_id, title, **fields):
    return {
        "document_id": document_id,
        "title": title,
        "publication_year": "2024",
        "evidence_citations": {
            field_name: {
                "evidence_type": "explicit_fact",
                "evidence_sufficiency": "sufficient",
                "retrieval_similarity": 0.9,
                "confidence": 0.9,
                "requires_human_review": False,
                "evidence_chunks": [{
                    "chunk_id": f"{document_id}_chunk_1",
                    "document_id": document_id,
                    "text": " ".join(
                        str(item) for item in (
                            value if isinstance(value, list) else [value]
                        )
                    ),
                }],
                "citations": [
                    {
                        "chunk_id": f"{document_id}_chunk_1",
                        "document_id": document_id,
                        "start_page": 2,
                        "end_page": 3,
                    }
                ],
            }
            for field_name, value in fields.items()
        },
        **fields,
    }


def related(doi, paper_analysis, status="analyzed"):
    return {
        "status": status,
        "paper": {"title": paper_analysis["title"], "doi": doi},
        "verification": {"status": "verified"},
        "document_id": paper_analysis["document_id"],
        "analysis": paper_analysis,
    }


class TestLiteratureComparator:
    def test_compares_fields_and_preserves_citations(self, comparator):
        uploaded = analysis(
            "uploaded",
            "Uploaded paper",
            research_problem="Predicting flood risk",
            methodology="Graph neural network",
            dataset="River measurements",
            evaluation_metrics=["F1", "Recall"],
            reported_results="F1 of 0.82",
            limitations=["Small sample"],
        )
        related_paper = analysis(
            "related",
            "Related paper",
            research_problem="Predicting flood risk",
            methodology="Graph neural network",
            dataset="River measurements",
            evaluation_metrics=["F1", "Precision"],
            reported_results="F1 of 0.79",
            limitations=["Small sample", "Single region"],
        )

        result = comparator.compare_papers(
            [uploaded, related("10.1234/related", related_paper)]
        )

        assert result["status"] == "completed"
        assert result["source_count"] == 2
        assert result["field_comparisons"]["methodology"]["status"] == (
            "exact_agreement"
        )
        metrics = result["field_comparisons"]["evaluation_metrics"]
        assert metrics["status"] == "partial_overlap"
        assert metrics["shared_values"][0]["value"] == "F1"
        assert metrics["shared_values"][0]["source_ids"] == [
            "uploaded",
            "related",
        ]
        evidence = metrics["values"][0]["evidence"]
        assert evidence["status"] == "available"
        assert evidence["retrieval_similarity"] == 0.9
        assert evidence["factual_confidence"] is None
        assert evidence["confidence"] is None
        assert evidence["citations"] == [
            {
                "chunk_id": "uploaded_chunk_1",
                "document_id": "uploaded",
                "start_page": 2,
                "end_page": 3,
                "passage_excerpt": "F1 Recall",
            }
        ]

    def test_distinct_results_are_not_labeled_contradictory(self, comparator):
        papers = [
            analysis("one", "Paper one", reported_results="Accuracy was 91%."),
            analysis("two", "Paper two", reported_results="Accuracy was 87%."),
        ]

        result = comparator.compare_papers(papers)

        field_result = result["field_comparisons"]["reported_results"]
        assert field_result["status"] == "distinct_reported_values"
        assert "not automatically contradictions" in result["interpretation_note"]

    def test_missing_and_invalid_citations_require_human_review(self, comparator):
        first = analysis("one", "Paper one", methodology="Transformer")
        second = analysis("two", "Paper two", methodology="CNN")
        first["evidence_citations"].pop("methodology")
        second["evidence_citations"]["methodology"] = {
            "citations": [
                {"chunk_id": "wrong_chunk", "document_id": "different-paper"}
            ],
            "requires_human_review": False,
        }
        third = analysis("three", "Paper three", methodology="LSTM")
        third["evidence_citations"]["methodology"]["citations"] = [
            {"chunk_id": "chunk-without-document"}
        ]

        result = comparator.compare_papers([first, second, third])
        comparison = result["field_comparisons"]["methodology"]

        assert comparison["values"][0]["evidence"]["status"] == "missing"
        assert (
            comparison["values"][1]["evidence"]["status"]
            == "invalid_document_reference"
        )
        assert comparison["values"][2]["evidence"]["status"] == (
            "incomplete_reference"
        )
        assert comparison["requires_human_review"] is True
        assert comparison["citation_coverage"] == 0.0

    @pytest.mark.parametrize(
        ("passage", "chunk_document_id", "chunk_id", "expected_status"),
        [
            ("The exact source passage.", "paper", "paper_chunk_1", "available"),
            ("The exact source passage.", "paper", "missing_chunk", "unresolved_citation"),
            ("The exact source passage.", "other-paper", "paper_chunk_1", "unresolved_citation"),
            ("   ", "paper", "paper_chunk_1", "unresolved_citation"),
        ],
    )
    def test_citation_available_only_for_resolved_nonempty_passage(
        self,
        comparator,
        passage,
        chunk_document_id,
        chunk_id,
        expected_status,
    ):
        source = analysis("paper", "Paper", methodology="A method")
        citation = source["evidence_citations"]["methodology"]
        citation["citations"] = [{
            "document_id": "paper",
            "chunk_id": chunk_id,
        }]
        citation["evidence_chunks"] = [{
            "document_id": chunk_document_id,
            "chunk_id": "paper_chunk_1",
            "text": passage,
        }]

        result = comparator.compare_papers([source, source | {
            "document_id": "second",
            "title": "Second paper",
            "evidence_citations": {},
        }])

        evidence = result["field_comparisons"]["methodology"]["values"][0][
            "evidence"
        ]
        assert evidence["status"] == expected_status
        if expected_status == "available":
            assert evidence["citations"][0]["passage_excerpt"] == passage
        else:
            assert "passage_excerpt" not in evidence["citations"][0]

    def test_excludes_unanalyzed_and_duplicate_doi_records(self, comparator):
        first = analysis("one", "Paper one", dataset="Dataset A")
        second = analysis("two", "Paper two", dataset="Dataset B")
        duplicate = analysis("duplicate", "Duplicate", dataset="Dataset C")

        result = comparator.compare_papers(
            [
                related("https://doi.org/10.1/first", first),
                related("10.1/second", second),
                related("10.1/first", duplicate),
                {"status": "download_failed", "paper": {"title": "Failed"}},
            ]
        )

        assert result["source_count"] == 2
        assert [item["reason"] for item in result["excluded_papers"]] == [
            "duplicate_source",
            "paper_not_analyzed",
        ]

    def test_returns_insufficient_status_when_fewer_than_two_analyzed(self, comparator):
        result = comparator.compare_papers(
            [{"status": "skipped", "paper": {"title": "Unavailable"}}]
        )

        assert result["status"] == "insufficient_sources"
        assert result["source_count"] == 0
        assert all(
            field["status"] == "insufficient_data"
            for field in result["field_comparisons"].values()
        )

    def test_rejects_bad_inputs_and_values(self, comparator):
        with pytest.raises(LiteratureComparisonError, match="papers must"):
            comparator.compare_papers("not a list")
        with pytest.raises(LiteratureComparisonError, match="more than 20"):
            comparator.compare_papers([{}] * 21)
        with pytest.raises(LiteratureComparisonError, match="each paper input"):
            comparator.compare_papers([{}, None])
        with pytest.raises(LiteratureComparisonError, match="methodology"):
            comparator.compare_papers(
                [
                    analysis("one", "Paper one", methodology={"model": "CNN"}),
                    analysis("two", "Paper two", methodology="Transformer"),
                ]
            )

    def test_duplicate_document_ids_are_not_counted_twice(self, comparator):
        first = analysis("same-id", "Paper one", dataset="Dataset A")
        second = analysis("same-id", "Paper two", dataset="Dataset B")

        result = comparator.compare_papers([first, second])

        assert result["source_count"] == 1
        assert result["status"] == "insufficient_sources"
        assert result["excluded_papers"][0]["reason"] == "duplicate_source"
