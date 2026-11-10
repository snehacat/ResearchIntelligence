"""Generate a structured, traceable Markdown report from analysis records."""

import re
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlsplit


class ResearchReportError(ValueError):
    """Raised when required report inputs are invalid."""


class ResearchReportGenerator:
    """Render a report without asserting unsupported facts or novelty."""

    REPORT_FIELDS = (
        ("research_problem", "Research problem"),
        ("methodology", "Methodology"),
        ("dataset", "Dataset"),
        ("evaluation_metrics", "Evaluation metrics"),
        ("reported_results", "Reported results"),
        ("limitations", "Limitations"),
    )

    def generate(
        self,
        primary_analysis: Mapping[str, Any],
        *,
        related_papers: Optional[Sequence[Mapping[str, Any]]] = None,
        metadata_verifications: Optional[Sequence[Mapping[str, Any]]] = None,
        comparison: Optional[Mapping[str, Any]] = None,
        gap_analysis: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(primary_analysis, Mapping):
            raise ResearchReportError(
                "primary_analysis must be a mapping"
            )
        related = self._mapping_sequence(related_papers, "related_papers")
        verifications = self._mapping_sequence(
            metadata_verifications, "metadata_verifications"
        )
        if comparison is not None and not isinstance(comparison, Mapping):
            raise ResearchReportError("comparison must be a mapping")
        if gap_analysis is not None and not isinstance(gap_analysis, Mapping):
            raise ResearchReportError("gap_analysis must be a mapping")

        verification_by_doi = {
            self._clean_doi(item.get("doi")): item
            for item in verifications
            if self._clean_doi(item.get("doi"))
        }
        lines = ["# Research Analysis Report", ""]
        title = self._text(primary_analysis.get("title")) or "Untitled paper"
        lines.extend([f"## Uploaded paper: {self._escape(title)}", ""])
        lines.extend(self._primary_summary(primary_analysis))
        lines.extend(self._paper_fields(primary_analysis))
        lines.extend(self._coverage_warnings(primary_analysis))

        lines.extend(["## Related papers", ""])
        if not related:
            lines.extend(["No related-paper records were supplied.", ""])
        for index, paper in enumerate(related, 1):
            metadata = paper.get("paper")
            if not isinstance(metadata, Mapping):
                metadata = paper
            paper_title = self._text(metadata.get("title")) or "Untitled record"
            status = self._text(paper.get("status")) or "metadata_only"
            doi = self._clean_doi(metadata.get("doi"))
            verification = (
                verification_by_doi.get(doi)
                if doi
                else paper.get("verification")
            )
            verification_status = (
                verification.get("status")
                if isinstance(verification, Mapping)
                else "not_checked"
            )
            lines.append(f"### {index}. {self._escape(paper_title)}")
            lines.append(f"- Processing status: {self._escape(status)}")
            lines.append(
                "- Authors: "
                + self._escape(self._display_value(metadata.get("authors")))
            )
            lines.append(
                "- Publication year: "
                + self._escape(
                    self._display_value(
                        metadata.get("year")
                        or metadata.get("publication_year")
                    )
                )
            )
            lines.append(f"- Metadata verification: {self._escape(verification_status)}")
            topic_overlap = metadata.get("topic_overlap_score")
            if isinstance(topic_overlap, (int, float)) and not isinstance(
                topic_overlap, bool
            ):
                lines.append(
                    "- Exact title-term overlap: "
                    f"{topic_overlap:.0%} (keyword overlap only; not a relevance probability)"
                )
                terms = metadata.get("topic_overlap_terms")
                if isinstance(terms, list) and terms:
                    lines.append(
                        "- Matched title terms: "
                        + self._escape(", ".join(str(term) for term in terms))
                    )
            if doi:
                doi_url = f"https://doi.org/{quote(doi, safe='/')}"
                lines.append(f"- DOI: [{self._escape(doi)}]({doi_url})")
            else:
                lines.append("- DOI: unavailable")
            source_url = self._safe_url(metadata.get("url"))
            if source_url:
                lines.append(f"- Source link: {source_url}")
            lines.append("")

        lines.extend(self._comparison_section(comparison))
        lines.extend(self._gap_section(gap_analysis))
        warnings = self._report_warnings(
            primary_analysis, comparison, gap_analysis
        )
        lines.extend(["## Limitations and review warnings", ""])
        lines.extend(f"- {self._escape(warning)}" for warning in warnings)
        lines.append("")
        report_text = "\n".join(lines)
        return {
            "status": "generated",
            "title": title,
            "markdown": report_text,
            "warnings": warnings,
            "source_count": (
                comparison.get("source_count", 1)
                if isinstance(comparison, Mapping)
                else 1
            ),
            "contains_global_novelty_claim": False,
        }

    def _primary_summary(self, analysis: Mapping[str, Any]) -> List[str]:
        lines = ["### Summary", ""]
        abstract = self._text(analysis.get("abstract"))
        problem = self._text(analysis.get("research_problem"))
        summary = abstract or problem
        summary_field = "abstract" if abstract else "research_problem"
        lines.extend(
            [
                self._escape(summary)
                if summary
                else "No abstract or research-problem summary was extracted.",
                "",
                f"- Authors: {self._escape(self._display_value(analysis.get('authors')))}",
                f"- Publication year: {self._escape(self._display_value(analysis.get('publication_year')))}",
                "",
            ]
        )
        lines.append(
            "Summary source: "
            + ("model-extracted abstract." if abstract else "extracted research problem.")
        )
        lines.append("")
        lines.extend(self._field_evidence_lines(analysis, summary_field))
        return lines

    def _paper_fields(self, analysis: Mapping[str, Any]) -> List[str]:
        lines = ["### Extracted research details", ""]
        citations = analysis.get("evidence_citations")
        if not isinstance(citations, Mapping):
            citations = {}
        for field_name, label in self.REPORT_FIELDS:
            value = analysis.get(field_name)
            lines.append(f"#### {label}")
            if self._is_missing(value):
                lines.extend(["Not available in the extracted analysis.", ""])
                continue
            lines.append(self._escape(self._display_value(value)))
            lines.extend(self._field_evidence_lines(analysis, field_name))
        return lines

    def _field_evidence_lines(
        self,
        analysis: Mapping[str, Any],
        field_name: str,
    ) -> List[str]:
        citations = analysis.get("evidence_citations")
        if not isinstance(citations, Mapping):
            citations = {}
        field_citation = citations.get(field_name)
        if not isinstance(field_citation, Mapping):
            field_citation = {}
        references = field_citation.get("citations")
        if not isinstance(references, list):
            references = []
        evidence_chunks = field_citation.get("evidence_chunks")
        if not isinstance(evidence_chunks, list):
            evidence_chunks = []
        chunks_by_id = {
            (chunk.get("document_id"), chunk.get("chunk_id")): chunk
            for chunk in evidence_chunks
            if isinstance(chunk, Mapping)
        }
        lines = []
        valid_references = [
            reference for reference in references
            if isinstance(reference, Mapping)
            and reference.get("document_id")
            and reference.get("chunk_id")
        ]
        if not valid_references:
            return ["", "Source passage unavailable; verify manually.", ""]

        lines.extend(["", "Retrieved source passages (review support manually):"])
        for reference in valid_references:
            citation = (
                f"- document `{self._escape(str(reference['document_id']))}`, "
                f"chunk `{self._escape(str(reference['chunk_id']))}`"
            )
            start_page = reference.get("start_page")
            end_page = reference.get("end_page", start_page)
            if start_page is not None:
                citation += f", pages {start_page}-{end_page or start_page}"
            lines.append(citation)
            chunk = chunks_by_id.get(
                (reference.get("document_id"), reference.get("chunk_id"))
            )
            passage = (
                self._text(chunk.get("text"))
                if isinstance(chunk, Mapping)
                else ""
            )
            if passage:
                excerpt = passage[:800].replace("\n", " ").strip()
                if len(passage) > 800:
                    excerpt += "..."
                lines.extend(["", f"> {self._escape(excerpt)}"])
            else:
                lines.append("  - Retrieved passage text is unavailable.")
        lines.append("")
        return lines

    def _coverage_warnings(self, analysis: Mapping[str, Any]) -> List[str]:
        coverage = analysis.get("analysis_coverage")
        if not isinstance(coverage, Mapping) or not coverage.get("is_partial"):
            return []
        omitted = coverage.get("omitted_chunks")
        omitted_count = len(omitted) if isinstance(omitted, list) else 0
        return [
            "### Coverage warning",
            "",
            (
                "The extraction used selected excerpts, not the complete paper. "
                f"Recorded omitted chunks: {omitted_count}; review the source "
                "PDF for unprocessed sections."
            ),
            "",
        ]

    def _comparison_section(
        self, comparison: Optional[Mapping[str, Any]]
    ) -> List[str]:
        lines = ["## Comparison", ""]
        if not isinstance(comparison, Mapping):
            return lines + ["No cross-paper comparison was supplied.", ""]
        lines.append(
            f"Comparison status: {self._escape(str(comparison.get('status', 'unknown')))}."
        )
        lines.append("")
        field_comparisons = comparison.get("field_comparisons")
        if not isinstance(field_comparisons, Mapping):
            return lines + ["Comparison details are unavailable.", ""]
        for field_name, result in field_comparisons.items():
            if not isinstance(result, Mapping):
                continue
            label = self._escape(str(field_name.replace("_", " ").title()))
            lines.append(f"### {label}")
            lines.append(
                f"Pattern: {self._escape(str(result.get('status', 'unknown')))}."
            )
            values = result.get("values")
            if isinstance(values, list) and values:
                for value in values:
                    if not isinstance(value, Mapping):
                        continue
                    source_title = self._text(value.get("title")) or self._text(
                        value.get("source_id")
                    )
                    evidence = value.get("evidence")
                    refs = []
                    if isinstance(evidence, Mapping):
                        raw_refs = evidence.get("citations")
                        if isinstance(raw_refs, list):
                            refs = [
                                self._reference_label(ref)
                                for ref in raw_refs
                                if isinstance(ref, Mapping)
                            ]
                    line = (
                        f"- {self._escape(source_title)}: "
                        f"{self._escape(self._display_value(value.get('value')))}"
                    )
                    line += (
                        f" (references: {self._escape('; '.join(refs))})"
                        if refs
                        else " (citation reference unavailable)"
                    )
                    if isinstance(evidence, Mapping):
                        evidence_status = self._text(
                            evidence.get("status")
                        ) or "unresolved"
                        line += (
                            f" [evidence passage: "
                            f"{self._escape(evidence_status)}]"
                        )
                    lines.append(line)
                    if isinstance(evidence, Mapping):
                        raw_refs = evidence.get("citations")
                        if isinstance(raw_refs, list):
                            for reference in raw_refs:
                                if not isinstance(reference, Mapping):
                                    continue
                                passage = self._text(
                                    reference.get("passage_excerpt")
                                )
                                if passage:
                                    lines.append(
                                        f"  > {self._escape(passage)}"
                                    )
            else:
                lines.append("No comparable extracted values.")
            lines.append("")
        lines.append(
            "Differences are reported as distinct values, not factual "
            "contradictions or proof of a research gap."
        )
        lines.append("")
        return lines

    def _gap_section(
        self, gap_analysis: Optional[Mapping[str, Any]]
    ) -> List[str]:
        lines = ["## Candidate research follow-ups", ""]
        if not isinstance(gap_analysis, Mapping):
            return lines + ["No candidate analysis was supplied.", ""]
        candidates = gap_analysis.get("research_gap_candidates")
        if not isinstance(candidates, list) or not candidates:
            lines.extend(["No candidates were derived from the supplied evidence.", ""])
        else:
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                question = self._text(candidate.get("candidate_question"))
                if question:
                    evidence_status = self._text(
                        candidate.get("evidence_status")
                    ) or "unresolved"
                    lines.append(
                        f"- {self._escape(question)} "
                        f"[evidence: {self._escape(evidence_status)}]"
                    )
                    refs = candidate.get("citations")
                    labels = [
                        self._reference_label(ref)
                        for ref in refs
                        if isinstance(ref, Mapping)
                    ] if isinstance(refs, list) else []
                    if labels:
                        lines.append(
                            f"  - References: {self._escape('; '.join(labels))}"
                        )
                    else:
                        lines.append(
                            "  - No valid source reference; manual review required."
                        )
        lines.append("")
        warning = self._text(gap_analysis.get("scope_warning"))
        lines.extend(
            [
                self._escape(warning)
                if warning
                else "Candidates are not validated research gaps.",
                "",
            ]
        )
        return lines

    def _report_warnings(
        self,
        analysis: Mapping[str, Any],
        comparison: Optional[Mapping[str, Any]],
        gap_analysis: Optional[Mapping[str, Any]],
    ) -> List[str]:
        warnings = [
            "Extracted statements and citations require source-PDF review; a citation reference does not prove claim support.",
            "Retrieval similarity and evidence sufficiency are heuristic signals, not factual confidence.",
        ]
        coverage = analysis.get("analysis_coverage")
        if isinstance(coverage, Mapping) and coverage.get("is_partial"):
            warnings.append(
                "Extraction coverage is partial; selected chunks and omitted sections are recorded."
            )
        if isinstance(comparison, Mapping) and comparison.get("status") != "completed":
            warnings.append(
                "Cross-paper comparison has insufficient analyzed sources."
            )
        if isinstance(comparison, Mapping) and comparison.get("source_count", 0) < 2:
            warnings.append(
                "Crossref metadata-only records were not compared as evidence; "
                "at least two papers must be successfully analyzed."
            )
        if isinstance(gap_analysis, Mapping):
            warnings.append(
                "Research follow-ups are candidates only; novelty and completeness were not established."
            )
        return warnings

    @staticmethod
    def _mapping_sequence(
        values: Optional[Sequence[Mapping[str, Any]]],
        field_name: str,
    ) -> List[Mapping[str, Any]]:
        if values is None:
            return []
        if not isinstance(values, (list, tuple)) or any(
            not isinstance(value, Mapping) for value in values
        ):
            raise ResearchReportError(f"{field_name} must contain mappings")
        return list(values)

    @staticmethod
    def _reference_label(reference: Mapping[str, Any]) -> str:
        document_id = reference.get("document_id", "unknown")
        chunk_id = reference.get("chunk_id", "unknown")
        label = f"{document_id}/{chunk_id}"
        page = reference.get("start_page")
        if page is not None:
            label += f" p.{page}"
            end_page = reference.get("end_page")
            if end_page is not None and end_page != page:
                label += f"-{end_page}"
        return label

    @staticmethod
    def _safe_url(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        url = value.strip()
        if not url or urlsplit(url).scheme.casefold() not in {"https", "http"}:
            return None
        if not urlsplit(url).netloc:
            return None
        return quote(url, safe=":/?&=#%+;,@")

    @staticmethod
    def _clean_doi(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        doi = re.sub(
            r"(?i)^https?://(?:dx\.)?doi\.org/",
            "",
            value.strip(),
        ).strip()
        return doi or None

    @staticmethod
    def _is_missing(value: Any) -> bool:
        return value is None or value == "" or value == []

    @staticmethod
    def _text(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    @classmethod
    def _display_value(cls, value: Any) -> str:
        if value is None:
            return "unavailable"
        if isinstance(value, list):
            items = [cls._text(item) for item in value if cls._text(item)]
            return "; ".join(items) if items else "unavailable"
        if isinstance(value, str):
            return value.strip() or "unavailable"
        return str(value)

    @staticmethod
    def _escape(value: str) -> str:
        return re.sub(r"([\\`*_{}\[\]<>])", r"\\\1", value)
