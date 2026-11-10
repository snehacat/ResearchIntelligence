"""Conservative, evidence-linked research follow-up candidates."""

from collections.abc import Mapping, Sequence
import re
from typing import Any, Dict, List


class ResearchGapAnalysisError(ValueError):
    """Raised when research-gap input has an invalid shape."""


class ResearchGapAnalyzer:
    """Surface stated limitations and missing comparison evidence as candidates."""

    MAX_PAPERS = 20
    COVERAGE_FIELDS = (
        "research_problem",
        "methodology",
        "dataset",
        "evaluation_metrics",
        "reported_results",
        "limitations",
    )

    def analyze(
        self,
        papers: Sequence[Mapping[str, Any]],
        comparison: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(papers, (list, tuple)):
            raise ResearchGapAnalysisError("papers must be a list or tuple")
        if len(papers) > self.MAX_PAPERS:
            raise ResearchGapAnalysisError(
                f"papers cannot contain more than {self.MAX_PAPERS} candidates"
            )
        if not isinstance(comparison, Mapping):
            raise ResearchGapAnalysisError("comparison must be a mapping")

        candidates: List[Dict[str, Any]] = []
        seen_candidates = set()
        coverage_gaps = []
        sources = comparison.get("sources", [])
        if not isinstance(sources, list):
            sources = []
        source_by_document = {
            source.get("document_id"): source
            for source in sources
            if isinstance(source, Mapping) and source.get("document_id")
        }

        for index, paper in enumerate(papers):
            if not isinstance(paper, Mapping):
                raise ResearchGapAnalysisError(
                    f"paper at index {index} must be a mapping"
                )
            if (
                "status" in paper
                and paper.get("status") != "analyzed"
            ):
                continue
            analysis = paper.get("analysis", paper)
            if not isinstance(analysis, Mapping):
                continue
            document_id = analysis.get("document_id") or paper.get(
                "document_id"
            )
            source = source_by_document.get(document_id, {})
            title = (
                source.get("title")
                or analysis.get("title")
                or paper.get("title")
                or f"Source {index + 1}"
            )
            limitation_value = analysis.get("limitations")
            limitations = self._as_text_list(limitation_value)
            citation_data = analysis.get("evidence_citations")
            if not isinstance(citation_data, Mapping):
                citation_data = {}
            limitation_citation = citation_data.get("limitations")
            references, evidence_status = self._references(
                limitation_citation, document_id, limitations
            )

            for limitation in limitations:
                key = (str(document_id), limitation.casefold())
                if key in seen_candidates:
                    continue
                seen_candidates.add(key)
                candidates.append(
                    {
                        "type": "follow_up_from_stated_limitation",
                        "candidate_question": (
                            "Investigate the scope and impact of this "
                            f"reported limitation: {limitation}"
                        ),
                        "stated_limitation": limitation,
                        "source": {
                            "document_id": document_id,
                            "title": title,
                        },
                        "citations": references,
                        "evidence_status": evidence_status,
                        "requires_human_review": True,
                        "novelty_claim": False,
                    }
                )

            for field_name in self.COVERAGE_FIELDS:
                value = analysis.get(field_name)
                if self._is_missing(value):
                    coverage_gaps.append(
                        {
                            "source_id": source.get("source_id"),
                            "document_id": document_id,
                            "title": title,
                            "field": field_name,
                            "status": "not_extracted_or_unavailable",
                            "interpretation": (
                                "Missing structured output is not evidence "
                                "that the paper omits this research topic."
                            ),
                            "requires_human_review": True,
                        }
                    )

        return {
            "status": (
                "candidate_follow_ups_found"
                if candidates
                else "no_evidence_based_candidates"
            ),
            "candidate_count": len(candidates),
            "research_gap_candidates": candidates,
            "evidence_coverage_gaps": coverage_gaps,
            "source_count": comparison.get("source_count", 0),
            "scope_warning": (
                "These are review candidates derived from extracted "
                "limitations and missing fields, not validated research "
                "gaps. The supplied papers are not an exhaustive search, "
                "and no global novelty claim is made."
            ),
            "global_novelty_claim": False,
            "requires_human_review": True,
        }

    @staticmethod
    def _as_text_list(value: Any) -> List[str]:
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [
                item.strip()
                for item in value
                if isinstance(item, str) and item.strip()
            ]
        return []

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, list):
            return not any(
                isinstance(item, str) and item.strip() for item in value
            )
        return False

    @staticmethod
    def _references(
        citation: Any,
        document_id: Any,
        limitations: List[str],
    ) -> tuple[List[Dict[str, Any]], str]:
        if not isinstance(citation, Mapping):
            return [], "unresolved"
        raw_references = citation.get("citations")
        evidence_chunks = citation.get("evidence_chunks")
        if not isinstance(raw_references, list) or not isinstance(
            evidence_chunks, list
        ):
            return [], "unresolved"
        chunks_by_id = {
            (chunk.get("document_id"), chunk.get("chunk_id")): chunk
            for chunk in evidence_chunks
            if isinstance(chunk, Mapping)
        }
        references = []
        passages = []
        for reference in raw_references:
            if not isinstance(reference, Mapping):
                continue
            if (
                not reference.get("chunk_id")
                or not reference.get("document_id")
                or reference.get("document_id") != document_id
            ):
                continue
            chunk = chunks_by_id.get(
                (document_id, reference.get("chunk_id"))
            )
            passage = chunk.get("text") if isinstance(chunk, Mapping) else None
            if not isinstance(passage, str) or not passage.strip():
                continue
            references.append(
                {
                    key: reference[key]
                    for key in (
                        "document_id",
                        "chunk_id",
                        "start_page",
                        "end_page",
                    )
                    if key in reference
                }
            )
            passages.append(passage)

        if not passages:
            return [], "unresolved"

        normalized_passages = [
            " ".join(re.findall(r"\w+", passage.casefold()))
            for passage in passages
        ]
        normalized_limitations = [
            " ".join(re.findall(r"\w+", limitation.casefold()))
            for limitation in limitations
        ]
        if any(
            limitation and limitation in passage
            for limitation in normalized_limitations
            for passage in normalized_passages
        ):
            return references, "supported"
        return references, "unsupported"
