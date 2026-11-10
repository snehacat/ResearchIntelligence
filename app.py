"""Streamlit interface for local research-paper analysis."""

import tempfile
import re
from pathlib import Path
from typing import Any, Dict

from analysis.paper_analyzer import PaperAnalyzer
from analysis.research_gap_analyzer import ResearchGapAnalyzer
from analysis.research_report_generator import ResearchReportGenerator
from chunking.text_chunker import TextChunker
from embeddings.embedding_generator import EmbeddingGenerator
from pdf_processing.text_extractor import TextExtractor
from qa.answer_generator import AnswerGenerator
from retrieval.semantic_retriever import SemanticRetriever


MIN_RELATED_TOPIC_OVERLAP = 0.15
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")


def _partition_related_papers(
    papers,
    minimum_overlap=MIN_RELATED_TOPIC_OVERLAP,
):
    """Keep title matches for downstream work and explain screened records."""
    relevant = []
    screened_out = []
    for paper in papers:
        score = paper.get("topic_overlap_score")
        if (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and score >= minimum_overlap
        ):
            paper["screening_status"] = "topic_match_candidate"
            relevant.append(paper)
        else:
            paper["screening_status"] = "screened_out"
            paper["screening_reason"] = (
                "The title has too little exact keyword overlap with the "
                "uploaded paper's title, research question, method, or data."
            )
            screened_out.append(paper)
    return relevant, screened_out


def _review_field_evidence(value, citation):
    """Check reference consistency and numeric presence, not factual support."""
    if not isinstance(citation, dict):
        citation = {}
    references = citation.get("citations")
    chunks = citation.get("evidence_chunks")
    references = references if isinstance(references, list) else []
    chunks = chunks if isinstance(chunks, list) else []
    chunks_by_id = {
        (chunk.get("document_id"), chunk.get("chunk_id")): chunk
        for chunk in chunks
        if isinstance(chunk, dict)
        and chunk.get("document_id")
        and chunk.get("chunk_id")
    }

    passages = []
    reference_consistent = bool(references)
    for reference in references:
        if not isinstance(reference, dict):
            reference_consistent = False
            continue
        key = (reference.get("document_id"), reference.get("chunk_id"))
        chunk = chunks_by_id.get(key)
        if chunk is None:
            reference_consistent = False
            continue
        for page_field in ("start_page", "end_page"):
            if (
                reference.get(page_field) is not None
                and chunk.get(page_field) != reference.get(page_field)
            ):
                reference_consistent = False
        text = chunk.get("text")
        if isinstance(text, str) and text.strip():
            passages.append({
                "document_id": key[0],
                "chunk_id": key[1],
                "start_page": chunk.get("start_page"),
                "end_page": chunk.get("end_page"),
                "text": text,
            })
        else:
            reference_consistent = False

    value_text = (
        " ".join(str(item) for item in value)
        if isinstance(value, list)
        else str(value or "")
    )
    expected_numbers = {
        number.replace(",", "")
        for number in NUMBER_PATTERN.findall(value_text)
    }
    source_text = " ".join(passage["text"] for passage in passages)
    source_numbers = {
        number.replace(",", "")
        for number in NUMBER_PATTERN.findall(source_text)
    }
    numeric_status = (
        "no_numeric_claims"
        if not expected_numbers
        else (
            "all_numeric_values_found"
            if expected_numbers.issubset(source_numbers)
            else "numeric_values_missing"
        )
    )
    return {
        "reference_status": (
            "consistent_with_retrieved_chunks"
            if reference_consistent
            else "missing_or_inconsistent"
        ),
        "numeric_status": numeric_status,
        "missing_numbers": sorted(expected_numbers - source_numbers),
        "passages": passages,
        "human_review_required": True,
    }


def run_analysis(
    pdf_path: str,
    document_id: str,
    *,
    discover_related: bool,
    analyze_related: bool,
    minimum_related_overlap: float = MIN_RELATED_TOPIC_OVERLAP,
    progress_callback=None,
) -> Dict[str, Any]:
    """Run the configured Groq workflow against a provided PDF path."""
    extraction_result = TextExtractor().extract_from_file(pdf_path)
    if not extraction_result.get("total_characters"):
        raise ValueError("The PDF contains no extractable text.")

    chunker = TextChunker()
    retriever = SemanticRetriever(EmbeddingGenerator())
    answer_generator = AnswerGenerator(retriever)
    analyzer = PaperAnalyzer(
        chunker,
        retriever,
        answer_generator,
    )
    primary = analyzer.analyze_paper(
        extraction_result,
        document_id,
        progress_callback=progress_callback,
    )

    # Optional stages with isolated error handling (A8)
    related_papers = []
    related_candidates = []
    screened_out_papers = []
    verifications = []
    related_evidence = []
    discovery_status = "skipped"
    discovery_error = None
    metadata_verification_status = "skipped"
    related_analysis_status = "skipped"

    if discover_related:
        try:
            related_candidates = analyzer.search_related_papers(
                primary, max_results=20
            )
            related_papers, screened_out_papers = _partition_related_papers(
                related_candidates, minimum_related_overlap
            )
            discovery_status = "completed"
        except Exception as e:
            discovery_status = "failed"
            discovery_error = str(e)
            # Preserve primary analysis even if discovery fails

        if related_papers:
            try:
                for paper in related_papers:
                    verification = analyzer.verify_paper_metadata(paper)
                    paper["verification"] = verification
                    verifications.append(
                        {"doi": paper.get("doi"), **verification}
                    )
                metadata_verification_status = "completed"
            except Exception as e:
                metadata_verification_status = "failed"
                # Preserve primary analysis even if verification fails

        if analyze_related and related_papers:
            try:
                related_evidence = analyzer.extract_related_paper_evidence(
                    related_papers,
                    max_papers=3,
                )
                related_analysis_status = "completed"
            except Exception as e:
                related_analysis_status = "failed"
                # Preserve primary analysis even if related analysis fails

    comparison_inputs = [primary, *related_evidence]
    comparison = analyzer.compare_papers(comparison_inputs)
    gap_analysis = ResearchGapAnalyzer().analyze(
        comparison_inputs, comparison
    )
    report = ResearchReportGenerator().generate(
        primary,
        related_papers=related_papers,
        metadata_verifications=verifications,
        comparison=comparison,
        gap_analysis=gap_analysis,
    )
    return {
        "analysis": primary,
        "related_papers": related_papers,
        "related_paper_candidates": related_candidates,
        "screened_out_papers": screened_out_papers,
        "metadata_verifications": verifications,
        "related_evidence": related_evidence,
        "comparison": comparison,
        "gap_analysis": gap_analysis,
        "report": report,
        "discovery_status": discovery_status,
        "discovery_error": discovery_error,
        "metadata_verification_status": metadata_verification_status,
        "related_analysis_status": related_analysis_status,
    }


def main():
    import streamlit as st

    st.set_page_config(
        page_title="AI Research Intelligence",
        page_icon="📚",
        layout="wide",
    )
    st.title("Research Paper Assistant")
    st.caption(
        "Extract a paper's key details, inspect the source passages, and "
        "optionally discover and compare related research."
    )
    st.info(
        "AI-generated details and retrieval citations are review aids, not "
        "proof. Analysis uses selected paper excerpts and may incur Groq "
        "charges. Check the source PDF before relying on any claim."
    )

    with st.sidebar:
        st.header("Analysis options")
        discover_related = st.checkbox(
            "Find related papers using Crossref", value=True
        )
        minimum_related_overlap = MIN_RELATED_TOPIC_OVERLAP
        if discover_related:
            minimum_related_overlap = st.slider(
                "Minimum title keyword overlap",
                min_value=0.0,
                max_value=1.0,
                value=MIN_RELATED_TOPIC_OVERLAP,
                step=0.05,
                help=(
                    "This is a simple term-match filter, not a relevance "
                    "probability. Lower it to see broader candidates."
                ),
            )
        analyze_related = st.checkbox(
            "Analyze eligible related-paper PDFs",
            disabled=not discover_related,
            help=(
                "Processes up to three topic-matched open-access papers. "
                "Their selected text is also sent to Groq."
            ),
        )

    with st.form("paper_analysis_form"):
        uploaded_file = st.file_uploader(
            "1. Choose a text-based research-paper PDF", type=["pdf"]
        )
        confirm_primary = st.checkbox(
            "2. I authorize sending selected excerpts from this paper to "
            "Groq and understand this may incur charges."
        )
        confirm_related = True
        if analyze_related:
            confirm_related = st.checkbox(
                "I also authorize sending selected excerpts from related "
                "papers to Groq and understand this may incur charges."
            )
        run_clicked = st.form_submit_button(
            "Analyze paper",
            type="primary",
        )

    if run_clicked and uploaded_file is None:
        st.error("Choose a research-paper PDF before starting the analysis.")
    elif run_clicked and not confirm_primary:
        st.error(
            "Confirm that you authorize sending selected paper excerpts to "
            "Groq before starting the analysis."
        )
    elif run_clicked and analyze_related and not confirm_related:
        st.error(
            "Confirm that you authorize sending selected related-paper "
            "excerpts to Groq, or turn off related-paper analysis."
        )

    if (
        run_clicked
        and uploaded_file is not None
        and confirm_primary
        and confirm_related
    ):
        temporary_path = None
        try:
            if uploaded_file.size > 25 * 1024 * 1024:
                raise ValueError("PDF is too large; maximum upload size is 25 MiB.")
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False
            ) as temporary_file:
                temporary_file.write(uploaded_file.getvalue())
                temporary_path = Path(temporary_file.name)

            stage_placeholder = st.empty()

            def progress(stage: str, message: str) -> None:
                stage_placeholder.info(f"{stage.replace('_', ' ').title()}: {message}")

            with st.spinner("Analyzing. This can take a few minutes..."):
                bundle = run_analysis(
                    str(temporary_path),
                    document_id=Path(uploaded_file.name).stem,
                    discover_related=discover_related,
                    analyze_related=analyze_related and confirm_related,
                    minimum_related_overlap=minimum_related_overlap,
                    progress_callback=progress,
                )
            st.session_state["research_bundle"] = bundle
            st.session_state["current_document_id"] = Path(uploaded_file.name).stem
            st.success("Analysis is ready. Review the summary and source evidence below.")
        except Exception as error:
            # A7: Clear stale results on failure
            st.session_state["research_bundle"] = None
            st.session_state["current_document_id"] = None
            # A10: Sanitize user-facing errors
            error_message = "Analysis failed. Please check the PDF format and try again."
            st.error(error_message)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    bundle = st.session_state.get("research_bundle")
    if not isinstance(bundle, dict):
        return

    # A7: Only render results if they belong to the current document
    current_document_id = st.session_state.get("current_document_id")
    if current_document_id is None:
        return

    analysis = bundle["analysis"]
    # Verify bundle belongs to current document
    if analysis.get("document_id") != current_document_id:
        st.warning("Previous analysis results are not available for this upload.")
        return

    st.divider()
    st.header(analysis.get("title") or "Untitled paper")
    author_values = analysis.get("authors")
    author_text = (
        ", ".join(author_values)
        if isinstance(author_values, list) and author_values
        else "Authors not extracted"
    )
    st.caption(
        f"{author_text} · {analysis.get('publication_year') or 'Year unavailable'}"
    )
    coverage = analysis.get("analysis_coverage", {})
    if coverage.get("is_partial"):
        st.warning(
            f"Partial coverage: {coverage.get('total_chunks', '?')} chunks "
            f"were available; {len(coverage.get('omitted_chunks', []))} were "
            "omitted. This is not a full-paper review."
        )
    tabs = st.tabs(
        ["Quick summary", "Evidence review", "Related papers", "Comparison", "Follow-ups", "Report"]
    )
    with tabs[0]:
        st.subheader("What this paper says")
        summary_fields = (
            ("Research question", "research_problem"),
            ("Method", "methodology"),
            ("Data", "dataset"),
            ("Main reported result", "reported_results"),
        )
        for label, field_name in summary_fields:
            value = analysis.get(field_name)
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.write(_format_value(value) if value else "Not extracted from the selected excerpts.")
        if analysis.get("limitations"):
            st.markdown("**Limitations reported by the model**")
            st.write(_format_value(analysis["limitations"]))
        st.caption(
            "These are extracted claims, not independently verified conclusions. "
            "Use Evidence review to inspect the retrieved source passages."
        )

    with tabs[1]:
        st.subheader("Extracted claims and source passages")
        st.caption(
            "A matching document/chunk reference means the citation points to "
            "a retrieved passage from this analysis. It does not prove that the "
            "passage supports the claim."
        )
        citations = analysis.get("evidence_citations", {})
        for field_name in PaperAnalyzer.SCHEMA_FIELDS:
            value = analysis.get(field_name)
            citation = citations.get(field_name, {}) if isinstance(citations, dict) else {}
            evidence_review = _review_field_evidence(value, citation)
            with st.expander(field_name.replace("_", " ").title()):
                st.markdown(f"**Extracted value:** {_format_value(value) if value else 'Not extracted'}")
                st.caption(
                    f"Reference check: {evidence_review['reference_status']} · "
                    f"Numeric check: {evidence_review['numeric_status']} · "
                    f"Evidence category: {citation.get('evidence_type', 'unknown')}"
                )
                if evidence_review["missing_numbers"]:
                    st.warning(
                        "These numbers were not found in the retrieved passages: "
                        + ", ".join(evidence_review["missing_numbers"])
                    )
                if not evidence_review["passages"]:
                    st.warning("No passage text is attached to this citation.")
                for passage in evidence_review["passages"]:
                    page = passage.get("start_page")
                    page_label = f" · page {page}" if page is not None else ""
                    st.markdown(
                        f"**{passage['document_id']} / {passage['chunk_id']}"
                        f"{page_label}**"
                    )
                    st.markdown(f"> {passage['text']}")
                st.caption("Human review is required for every extracted claim.")

    with tabs[2]:
        st.subheader("Related-paper search")
        related_papers = bundle.get("related_papers", [])
        discovery_status = bundle.get("discovery_status", "skipped")
        discovery_error = bundle.get("discovery_error")

        # A8: Show optional stage status
        if discovery_status == "failed":
            st.error(f"Related-paper search failed: {discovery_error}")
            st.info("Primary paper analysis is available above.")
        elif not bundle.get("related_paper_candidates"):
            st.info("Related-paper search was not run or returned no records.")
        elif not related_papers:
            st.warning(
                "Crossref returned records, but none met the minimum exact "
                "topic-overlap screen. Search terms are imperfect; inspect "
                "screened records below if you want to decide manually."
            )
        for paper in related_papers:
            with st.container(border=True):
                st.markdown(f"**{paper.get('title', 'Untitled record')}**")
                st.caption(
                    f"{', '.join(paper.get('authors') or []) or 'Authors unavailable'}"
                    f" · {paper.get('year') or 'Year unavailable'}"
                )
                st.write(
                    f"Topic keyword overlap: "
                    f"{paper.get('topic_overlap_score', 0):.0%} "
                    "(simple overlap, not a relevance probability)"
                )
                if paper.get("topic_overlap_terms"):
                    st.caption(
                        "Matched terms: "
                        + ", ".join(paper["topic_overlap_terms"])
                    )
                if paper.get("doi"):
                    st.markdown(f"DOI: https://doi.org/{paper['doi']}")
                if paper.get("alternate_dois"):
                    st.caption(
                        "Other DOI records merged for this same title/author: "
                        + ", ".join(paper["alternate_dois"])
                    )
                verification = paper.get("verification")
                if isinstance(verification, dict):
                    st.caption(
                        "Registry metadata check: "
                        f"{verification.get('status', 'not available')} "
                        "(metadata consistency only)"
                    )
                st.caption(
                    "Search metadata only; this record is not compared as "
                    "research evidence unless its PDF was successfully analyzed."
                )
        screened_out = bundle.get("screened_out_papers", [])
        if screened_out:
            with st.expander(f"Screened out ({len(screened_out)})"):
                st.caption("These are not included in comparison or related-PDF analysis.")
                st.dataframe(
                    [
                        {
                            "title": item.get("title"),
                            "topic overlap": item.get("topic_overlap_score"),
                            "matched terms": ", ".join(item.get("topic_overlap_terms", [])),
                            "reason": item.get("screening_reason"),
                        }
                        for item in screened_out
                    ],
                    use_container_width=True,
                )
        analyzed_count = sum(
            item.get("status") == "analyzed"
            for item in bundle.get("related_evidence", [])
        )
        st.info(
            f"{analyzed_count} related paper(s) successfully analyzed as "
            "evidence. Metadata-only search results are not part of the "
            "research comparison."
        )

    with tabs[3]:
        comparison = bundle.get("comparison", {})
        source_count = comparison.get("source_count", 0)
        st.subheader(f"Evidence comparison · {source_count} analyzed source(s)")
        if source_count < 2:
            st.info(
                "A cross-paper comparison needs at least two successfully "
                "analyzed papers. Finding metadata records alone is not enough."
            )
        for field_name, result in comparison.get("field_comparisons", {}).items():
            with st.expander(field_name.replace("_", " ").title()):
                st.caption(
                    f"Exact-text pattern: {result.get('status', 'unknown')} · "
                    "not a semantic equivalence or contradiction judgment"
                )
                values = result.get("values", [])
                if not values:
                    st.write("No comparable extracted values.")
                for item in values:
                    st.markdown(f"**{item.get('title') or item.get('source_id')}**")
                    st.write(_format_value(item.get("value")))
                    evidence = item.get("evidence", {})
                    st.caption(
                        f"Reference integrity: {evidence.get('status', 'unknown')}; "
                        "source passage review still required."
                    )
                    for ref in evidence.get("citations", []):
                        st.caption(
                            f"{ref.get('document_id')} / {ref.get('chunk_id')} "
                            f"/ page {ref.get('start_page', 'unknown')}"
                        )
                        if ref.get("passage_excerpt"):
                            st.markdown(f"> {ref['passage_excerpt']}")

    with tabs[4]:
        st.subheader("Follow-up questions to review")
        st.caption(
            "These prompts come from stated limitations. They are not confirmed "
            "research gaps or claims of novelty."
        )
        candidates = bundle.get("gap_analysis", {}).get("research_gap_candidates", [])
        if not candidates:
            st.info("No limitation-based follow-up candidates were extracted.")
        for candidate in candidates:
            with st.container(border=True):
                st.markdown(f"**{candidate.get('candidate_question')}**")
                st.write(
                    f"Reported limitation: "
                    f"{candidate.get('stated_limitation', 'unavailable')}"
                )
                for reference in candidate.get("citations", []):
                    st.caption(
                        f"{reference.get('document_id')} / "
                        f"{reference.get('chunk_id')} / "
                        f"page {reference.get('start_page', 'unknown')}"
                    )
        st.warning(bundle.get("gap_analysis", {}).get("scope_warning", "Review manually."))

    with tabs[5]:
        st.subheader("Downloadable report")
        st.warning("\n".join(f"- {warning}" for warning in bundle["report"]["warnings"]))
        st.download_button(
            "Download research report (Markdown)",
            data=bundle["report"]["markdown"],
            file_name="research-analysis-report.md",
            mime="text/markdown",
        )
        with st.expander("Preview report"):
            st.markdown(bundle["report"]["markdown"])


def _format_value(value):
    if value is None or value == "" or value == []:
        return "Not available"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


if __name__ == "__main__":
    main()
