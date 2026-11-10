"""Deterministic, evidence-linked comparison of structured paper analyses."""

import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class LiteratureComparisonError(Exception):
    """Raised when comparison input is invalid or exceeds configured bounds."""


class LiteratureComparator:
    """Compare extracted findings without inventing semantic agreement."""

    MAX_PAPERS = 20
    COMPARISON_FIELDS = (
        "research_problem",
        "methodology",
        "dataset",
        "evaluation_metrics",
        "reported_results",
        "limitations",
    )
    LIST_FIELDS = frozenset({"evaluation_metrics", "limitations"})

    def compare_papers(
        self, papers: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        """Return per-paper findings, exact-match patterns, and source citations."""
        if not isinstance(papers, (list, tuple)):
            raise LiteratureComparisonError("papers must be a list or tuple")
        if len(papers) > self.MAX_PAPERS:
            raise LiteratureComparisonError(
                f"papers cannot contain more than {self.MAX_PAPERS} candidates"
            )
        if any(not isinstance(paper, Mapping) for paper in papers):
            raise LiteratureComparisonError(
                "each paper input must be a mapping"
            )

        sources: List[Dict[str, Any]] = []
        excluded_papers: List[Dict[str, Any]] = []
        seen_ids = set()
        for index, paper in enumerate(papers):
            normalized = self._normalize_paper(paper, index)
            if normalized is None:
                excluded_papers.append(
                    self._excluded_record(paper, index, "paper_not_analyzed")
                )
                continue
            source, analysis, identity_keys = normalized
            if any(identity_key in seen_ids for identity_key in identity_keys):
                excluded_papers.append(
                    self._excluded_record(paper, index, "duplicate_source")
                )
                continue
            seen_ids.update(identity_keys)
            sources.append(
                {
                    "source": source,
                    "analysis": analysis,
                }
            )

        field_comparisons = {
            field_name: self._compare_field(field_name, sources)
            for field_name in self.COMPARISON_FIELDS
        }
        return {
            "comparison_type": "deterministic_literature_comparison",
            "status": "completed" if len(sources) >= 2 else "insufficient_sources",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_count": len(sources),
            "sources": [entry["source"] for entry in sources],
            "excluded_papers": excluded_papers,
            "field_comparisons": field_comparisons,
            "interpretation_note": (
                "Shared and distinct values are based on exact normalized text, "
                "not semantic equivalence. Different reported results are not "
                "automatically contradictions. This comparison is limited to "
                "the supplied analyses and does not establish literature "
                "coverage, claim validity, or research novelty."
            ),
        }

    def _normalize_paper(
        self, paper: Mapping[str, Any], index: int
    ) -> Optional[Tuple[Dict[str, Any], Mapping[str, Any], List[str]]]:
        if "analysis" in paper or "status" in paper or "paper" in paper:
            if paper.get("status") != "analyzed":
                return None
            analysis = paper.get("analysis")
            metadata = paper.get("paper")
            if not isinstance(analysis, Mapping):
                raise LiteratureComparisonError(
                    f"analyzed paper at index {index} must contain an analysis mapping"
                )
            if not isinstance(metadata, Mapping):
                metadata = {}
        else:
            analysis = paper
            metadata = paper

        document_id = analysis.get("document_id") or paper.get("document_id")
        if document_id is not None:
            document_id = str(document_id).strip() or None
        doi = self._clean_doi(
            metadata.get("doi") or analysis.get("doi") or paper.get("doi")
        )
        if not document_id and not doi and not analysis.get("title"):
            raise LiteratureComparisonError(
                f"paper at index {index} requires a document_id, DOI, or title"
            )
        source_id = str(
            document_id or (f"doi:{doi}" if doi else f"source-{index + 1}")
        )
        identity_keys = []
        if document_id:
            identity_keys.append(f"document:{document_id.casefold()}")
        if doi:
            identity_keys.append(f"doi:{doi}")
        if not identity_keys:
            identity_keys.append(f"title:{self._normalize_text(str(analysis['title']))}")
        for field_name in self.COMPARISON_FIELDS:
            value = analysis.get(field_name)
            if value is None:
                continue
            if field_name in self.LIST_FIELDS:
                if isinstance(value, str):
                    continue
                if not isinstance(value, list) or any(
                    not isinstance(item, str) for item in value
                ):
                    raise LiteratureComparisonError(
                        f"{field_name} at paper index {index} must be a string "
                        "or a list of strings"
                    )
            elif not isinstance(value, str):
                raise LiteratureComparisonError(
                    f"{field_name} at paper index {index} must be a string"
                )
        verification = paper.get("verification")
        verification_status = (
            verification.get("status")
            if isinstance(verification, Mapping)
            else None
        )
        source = {
            "source_id": source_id,
            "document_id": document_id,
            "title": (
                metadata.get("title")
                or analysis.get("title")
                or paper.get("title")
            ),
            "doi": doi,
            "publication_year": (
                metadata.get("year")
                or analysis.get("publication_year")
                or paper.get("publication_year")
            ),
            "metadata_verification_status": verification_status,
        }
        return source, analysis, identity_keys

    def _compare_field(
        self,
        field_name: str,
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        values = []
        missing_source_ids = []
        value_groups: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "value": None,
                "source_ids": [],
                "document_ids": [],
                "citations": [],
            }
        )
        per_source_value_keys = []

        for entry in sources:
            source = entry["source"]
            analysis = entry["analysis"]
            value = analysis.get(field_name)
            if self._is_empty(value):
                missing_source_ids.append(source["source_id"])
                continue

            citation = self._citation_summary(
                analysis.get("evidence_citations", {}).get(field_name)
                if isinstance(analysis.get("evidence_citations"), Mapping)
                else None,
                source["document_id"],
            )
            values.append(
                {
                    "source_id": source["source_id"],
                    "document_id": source["document_id"],
                    "title": source["title"],
                    "value": value,
                    "evidence": citation,
                }
            )
            normalized_values = self._field_values(field_name, value)
            per_source_value_keys.append(set(normalized_values))
            for normalized_value, display_value in normalized_values.items():
                group = value_groups[normalized_value]
                if group["value"] is None:
                    group["value"] = display_value
                group["source_ids"].append(source["source_id"])
                if source["document_id"]:
                    group["document_ids"].append(source["document_id"])
                group["citations"].append(
                    {
                        "source_id": source["source_id"],
                        "document_id": source["document_id"],
                        **citation,
                    }
                )

        shared_values = [
            group
            for group in value_groups.values()
            if len(group["source_ids"]) >= 2
        ]
        if len(values) < 2:
            status = "insufficient_data"
        elif all(keys == per_source_value_keys[0] for keys in per_source_value_keys[1:]):
            status = "exact_agreement"
        elif shared_values:
            status = "partial_overlap"
        else:
            status = "distinct_reported_values"

        return {
            "status": status,
            "values": values,
            "shared_values": shared_values,
            "missing_source_ids": missing_source_ids,
            "citation_coverage": (
                sum(
                    value["evidence"]["status"] == "available"
                    for value in values
                )
                / len(values)
                if values
                else 0.0
            ),
            "requires_human_review": any(
                value["evidence"]["status"] != "available"
                or value["evidence"].get("requires_human_review", True)
                for value in values
            ),
        }

    def _field_values(self, field_name: str, value: Any) -> Dict[str, Any]:
        if field_name in self.LIST_FIELDS:
            if isinstance(value, str):
                items = [value]
            elif isinstance(value, list):
                items = value
            else:
                items = []
        else:
            items = [value]

        normalized: Dict[str, Any] = {}
        for item in items:
            if not isinstance(item, str) or not item.strip():
                continue
            key = self._normalize_text(item)
            if key:
                normalized.setdefault(key, item.strip())
        return normalized

    def _citation_summary(
        self,
        citation: Any,
        source_document_id: Optional[str],
    ) -> Dict[str, Any]:
        if not isinstance(citation, Mapping):
            return {
                "status": "missing",
                "evidence_type": "unknown",
                "evidence_sufficiency": "insufficient",
                "requires_human_review": True,
                "citations": [],
            }

        raw_citations = citation.get("citations")
        if not isinstance(raw_citations, list) or not raw_citations:
            evidence_chunks = citation.get("evidence_chunks")
            raw_citations = (
                evidence_chunks if isinstance(evidence_chunks, list) else []
            )
        evidence_chunks = citation.get("evidence_chunks")
        chunks_by_id = {
            (chunk.get("document_id"), chunk.get("chunk_id")): chunk
            for chunk in evidence_chunks
            if isinstance(chunk, Mapping)
        } if isinstance(evidence_chunks, list) else {}

        normalized_citations = []
        invalid_reference = False
        incomplete_reference = False
        unresolved_reference = False
        for raw in raw_citations:
            if not isinstance(raw, Mapping):
                invalid_reference = True
                continue
            document_id = raw.get("document_id")
            if (
                source_document_id
                and document_id
                and document_id != source_document_id
            ):
                invalid_reference = True
            if not raw.get("chunk_id") or not document_id:
                incomplete_reference = True
            reference = {
                key: raw[key]
                for key in ("chunk_id", "document_id", "start_page", "end_page")
                if key in raw
            }
            chunk = chunks_by_id.get((document_id, raw.get("chunk_id")))
            passage = chunk.get("text") if isinstance(chunk, Mapping) else None
            if not isinstance(passage, str) or not passage.strip():
                unresolved_reference = True
            else:
                normalized_passage = " ".join(passage.split())
                excerpt = normalized_passage[:800]
                if len(normalized_passage) > 800:
                    excerpt += "..."
                reference["passage_excerpt"] = excerpt
            normalized_citations.append(reference)

        if invalid_reference:
            status = "invalid_document_reference"
        elif incomplete_reference:
            status = "incomplete_reference"
        elif unresolved_reference:
            status = "unresolved_citation"
        elif not normalized_citations:
            status = "missing"
        else:
            status = "available"
        return {
            "status": status,
            "evidence_type": citation.get("evidence_type", "unknown"),
            "evidence_sufficiency": citation.get(
                "evidence_sufficiency", "unknown"
            ),
            "retrieval_similarity": citation.get("retrieval_similarity"),
            # Deprecated compatibility alias; never expose retrieval similarity
            # or an LLM self-rating as factual confidence.
            "factual_confidence": None,
            "confidence": None,
            "requires_human_review": citation.get(
                "requires_human_review", True
            ),
            "citations": normalized_citations,
        }

    @staticmethod
    def _excluded_record(
        paper: Mapping[str, Any], index: int, reason: str
    ) -> Dict[str, Any]:
        metadata = paper.get("paper")
        if not isinstance(metadata, Mapping):
            metadata = paper
        return {
            "input_index": index,
            "title": metadata.get("title") or paper.get("title"),
            "doi": metadata.get("doi") or paper.get("doi"),
            "status": paper.get("status"),
            "reason": reason,
        }

    @staticmethod
    def _clean_doi(doi: Any) -> Optional[str]:
        if not isinstance(doi, str):
            return None
        cleaned = doi.strip()
        cleaned = re.sub(
            r"(?i)^https?://(?:dx\.)?doi\.org/",
            "",
            cleaned,
        ).strip().casefold()
        return cleaned or None

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().casefold()

    @staticmethod
    def _is_empty(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, tuple, set)):
            return not any(
                isinstance(item, str) and item.strip() for item in value
            )
        return False
