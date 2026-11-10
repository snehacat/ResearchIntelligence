"""Streamlit interface for local research-paper analysis."""

import os
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

# ── Custom CSS injected once ──────────────────────────────────────────────────
_CSS = """
<style>
/* ─── Global typography & colour tokens ─── */
:root {
    --accent: #6C63FF;
    --accent-light: #EDE9FF;
    --success: #22C55E;
    --warning: #F59E0B;
    --danger: #EF4444;
    --muted: #6B7280;
    --card-bg: #FFFFFF;
    --card-border: #E5E7EB;
    --radius: 12px;
    --shadow: 0 2px 8px rgba(0,0,0,0.07);
}
[data-theme="dark"] {
    --card-bg: #1E1E2E;
    --card-border: #2D2D3F;
    --accent-light: #2D2857;
    --muted: #9CA3AF;
}

/* ─── Header banner ─── */
.ri-banner {
    background: linear-gradient(135deg, var(--accent) 0%, #9C6FFF 100%);
    color: white;
    padding: 2rem 2.5rem 1.5rem;
    border-radius: var(--radius);
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
}
.ri-banner h1 { font-size: 2rem; font-weight: 800; margin: 0 0 0.3rem; letter-spacing: -0.5px; }
.ri-banner p  { font-size: 0.95rem; opacity: 0.88; margin: 0; }

/* ─── Metric cards ─── */
.ri-metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.ri-metric {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 0.9rem 1.2rem;
    flex: 1 1 150px;
    box-shadow: var(--shadow);
}
.ri-metric .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.ri-metric .value { font-size: 1.5rem; font-weight: 700; color: var(--accent); margin: 0.2rem 0 0; }

/* ─── Evidence card ─── */
.ri-evidence-card {
    background: var(--accent-light);
    border-left: 4px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
    line-height: 1.6;
}
.ri-evidence-card .meta { font-size: 0.75rem; color: var(--muted); margin-top: 0.4rem; }

/* ─── Paper card ─── */
.ri-paper-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    margin-bottom: 0.9rem;
    box-shadow: var(--shadow);
}
.ri-paper-card h4 { font-size: 1rem; font-weight: 600; margin: 0 0 0.3rem; }
.ri-paper-card .meta { font-size: 0.8rem; color: var(--muted); }

/* ─── Status badges ─── */
.ri-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    margin-left: 0.4rem;
    vertical-align: middle;
}
.ri-badge-ok     { background: #DCFCE7; color: #166534; }
.ri-badge-warn   { background: #FEF3C7; color: #92400E; }
.ri-badge-error  { background: #FEE2E2; color: #991B1B; }
.ri-badge-info   { background: #EDE9FF; color: #4C1D95; }

/* ─── Overlap bar ─── */
.ri-bar-bg { background: #E5E7EB; border-radius: 999px; height: 8px; margin: 0.4rem 0; }
.ri-bar-fill { background: var(--accent); border-radius: 999px; height: 8px; }

/* ─── Upload zone ─── */
div[data-testid="stFileUploader"] > div { border-radius: var(--radius) !important; }

/* ─── Tab styling ─── */
div[data-testid="stTabs"] button {
    font-size: 0.88rem;
    font-weight: 500;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ─── Sidebar polish ─── */
section[data-testid="stSidebar"] {
    border-right: 1px solid var(--card-border);
}
.ri-sidebar-logo { font-size: 1.3rem; font-weight: 800; color: var(--accent); letter-spacing: -0.5px; }
</style>
"""


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
            discovery_error = "Crossref or network error occurred"

        if related_papers:
            try:
                for paper in related_papers:
                    verification = analyzer.verify_paper_metadata(paper)
                    paper["verification"] = verification
                    verifications.append(
                        {"doi": paper.get("doi"), **verification}
                    )
                metadata_verification_status = "completed"
            except Exception:
                metadata_verification_status = "failed"

        if analyze_related and related_papers:
            try:
                related_evidence = analyzer.extract_related_paper_evidence(
                    related_papers,
                    max_papers=3,
                )
                related_analysis_status = "completed"
            except Exception:
                related_analysis_status = "failed"

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


# ── UI helpers ─────────────────────────────────────────────────────────────────

def _badge(text: str, kind: str = "info") -> str:
    return f'<span class="ri-badge ri-badge-{kind}">{text}</span>'


def _overlap_bar(score: float) -> str:
    pct = max(0, min(100, int(score * 100)))
    color = "#22C55E" if pct >= 50 else "#F59E0B" if pct >= 25 else "#EF4444"
    return (
        f'<div class="ri-bar-bg"><div class="ri-bar-fill" '
        f'style="width:{pct}%;background:{color};"></div></div>'
    )


def _status_checks() -> list[str]:
    """Return a list of warning strings for missing environment setup."""
    warnings: list[str] = []
    if not os.environ.get("GROQ_API_KEY"):
        warnings.append(
            "GROQ_API_KEY is not set. Analysis will fail when it reaches the "
            "LLM step. Set the environment variable before running."
        )
    return warnings


def _render_sidebar(discover_related_default: bool) -> tuple[bool, float, bool]:
    """Render sidebar and return (discover_related, min_overlap, analyze_related)."""
    import streamlit as st

    with st.sidebar:
        st.markdown('<div class="ri-sidebar-logo">📚 ResearchIQ</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Analysis options**")

        discover_related = st.checkbox(
            "Find related papers (Crossref)",
            value=discover_related_default,
            help="Search Crossref for papers that share title terms with your upload.",
        )
        minimum_related_overlap = MIN_RELATED_TOPIC_OVERLAP
        if discover_related:
            minimum_related_overlap = st.slider(
                "Min. title keyword overlap",
                min_value=0.0,
                max_value=1.0,
                value=MIN_RELATED_TOPIC_OVERLAP,
                step=0.05,
                help=(
                    "Simple term-match filter — not a relevance probability. "
                    "Lower values broaden the candidate set."
                ),
            )
        analyze_related = st.checkbox(
            "Analyze eligible related PDFs",
            value=False,
            disabled=not discover_related,
            help=(
                "Downloads and processes up to 3 open-access related papers. "
                "Their excerpts are also sent to Groq."
            ),
        )

        st.markdown("---")
        st.markdown("**Environment**")
        api_key_set = bool(os.environ.get("GROQ_API_KEY"))
        if api_key_set:
            st.success("GROQ_API_KEY ✓", icon="✅")
        else:
            st.error("GROQ_API_KEY not set", icon="🔑")

        st.markdown("---")
        with st.expander("About"):
            st.markdown(
                "**AI Research Intelligence** extracts structured knowledge "
                "from research PDFs and cross-references related literature "
                "via Crossref / OpenAlex.\n\n"
                "All AI outputs require human review. Citations point to "
                "retrieved passages — they are not factual proof."
            )

    return discover_related, minimum_related_overlap, analyze_related


def _render_upload_form(analyze_related: bool):
    """Render the upload form and return (uploaded_file, confirm_primary, confirm_related, run_clicked)."""
    import streamlit as st

    with st.form("paper_analysis_form", clear_on_submit=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Upload a text-based research PDF (max 25 MB)",
                type=["pdf"],
                label_visibility="collapsed",
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_clicked = st.form_submit_button(
                "🔍 Analyze",
                type="primary",
                use_container_width=True,
            )

        st.caption("⚠️ Selected excerpts will be sent to Groq and may incur charges.")
        confirm_primary = st.checkbox(
            "I authorize sending excerpts from this paper to Groq.",
            key="confirm_primary",
        )
        confirm_related = True
        if analyze_related:
            confirm_related = st.checkbox(
                "I also authorize sending excerpts from related papers to Groq.",
                key="confirm_related",
            )

    return uploaded_file, confirm_primary, confirm_related, run_clicked


def _render_summary_tab(analysis: Dict[str, Any]) -> None:
    import streamlit as st

    coverage = analysis.get("analysis_coverage", {})
    if coverage.get("is_partial"):
        st.warning(
            f"⚠️ Partial coverage: {coverage.get('total_chunks', '?')} chunks available; "
            f"{len(coverage.get('omitted_chunks', []))} omitted. This is not a full-paper review."
        )

    # Metric row
    year = analysis.get("publication_year") or "—"
    author_values = analysis.get("authors") or []
    author_count = len(author_values) if isinstance(author_values, list) else "—"
    limitations = analysis.get("limitations")
    lim_count = len(limitations) if isinstance(limitations, list) else ("1" if limitations else "0")
    st.markdown(
        f"""
        <div class="ri-metric-row">
          <div class="ri-metric"><div class="label">Year</div><div class="value">{year}</div></div>
          <div class="ri-metric"><div class="label">Authors</div><div class="value">{author_count}</div></div>
          <div class="ri-metric"><div class="label">Limitations</div><div class="value">{lim_count}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_fields = [
        ("🔬 Research question", "research_problem"),
        ("⚙️ Method", "methodology"),
        ("📊 Data", "dataset"),
        ("📈 Main reported result", "reported_results"),
    ]
    for label, field_name in summary_fields:
        value = analysis.get(field_name)
        with st.container(border=True):
            st.markdown(f"**{label}**")
            if value:
                st.write(_format_value(value))
            else:
                st.caption("*Not extracted from the selected excerpts.*")

    if analysis.get("limitations"):
        with st.container(border=True):
            st.markdown("**⚠️ Limitations reported by the model**")
            st.write(_format_value(analysis["limitations"]))

    st.caption(
        "These are extracted claims, not independently verified conclusions. "
        "Use the Evidence Review tab to inspect retrieved source passages."
    )


def _render_evidence_tab(analysis: Dict[str, Any]) -> None:
    import streamlit as st

    st.caption(
        "A matching document/chunk reference confirms the citation points to a "
        "retrieved passage. It does **not** prove the passage supports the claim."
    )
    citations = analysis.get("evidence_citations", {})
    for field_name in PaperAnalyzer.SCHEMA_FIELDS:
        value = analysis.get(field_name)
        citation = citations.get(field_name, {}) if isinstance(citations, dict) else {}
        evidence_review = _review_field_evidence(value, citation)

        ref_ok = evidence_review["reference_status"] == "consistent_with_retrieved_chunks"
        num_ok = evidence_review["numeric_status"] in ("no_numeric_claims", "all_numeric_values_found")
        badge = _badge("✓ refs OK", "ok") if ref_ok else _badge("⚠ refs missing", "warn")
        num_badge = _badge("✓ nums OK", "ok") if num_ok else _badge("⚠ nums missing", "warn")

        with st.expander(
            f"{field_name.replace('_', ' ').title()}", expanded=False
        ):
            st.markdown(
                f"**Extracted value:** {_format_value(value) if value else '*Not extracted*'}"
            )
            st.markdown(badge + num_badge, unsafe_allow_html=True)
            st.caption(f"Evidence category: {citation.get('evidence_type', 'unknown')}")

            if evidence_review["missing_numbers"]:
                st.warning(
                    "Numbers not found in retrieved passages: "
                    + ", ".join(evidence_review["missing_numbers"])
                )
            if not evidence_review["passages"]:
                st.info("No passage text attached to this citation.")

            for passage in evidence_review["passages"]:
                page = passage.get("start_page")
                page_label = f" · p.{page}" if page is not None else ""
                st.markdown(
                    f'<div class="ri-evidence-card">'
                    f"{passage['text']}"
                    f'<div class="meta">📄 {passage["document_id"]} / {passage["chunk_id"]}{page_label}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.caption("🔍 Human review required for every extracted claim.")


def _render_related_tab(bundle: Dict[str, Any]) -> None:
    import streamlit as st

    related_papers = bundle.get("related_papers", [])
    discovery_status = bundle.get("discovery_status", "skipped")
    discovery_error = bundle.get("discovery_error")

    if discovery_status == "failed":
        st.error("Related-paper search failed. This can happen if Crossref is unavailable or a network issue occurred.")
        st.info("Primary paper analysis is still available in the other tabs.")
        return
    if not bundle.get("related_paper_candidates"):
        st.info("Related-paper search was not run, or Crossref returned no records.")
        return
    if not related_papers:
        st.warning(
            "Crossref returned records but none met the minimum topic-overlap threshold. "
            "Inspect screened records below if you want to lower the threshold."
        )

    # Summary stats
    analyzed_count = sum(
        item.get("status") == "analyzed"
        for item in bundle.get("related_evidence", [])
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Candidates found", len(bundle.get("related_paper_candidates", [])))
    col2.metric("Passed screening", len(related_papers))
    col3.metric("PDFs analyzed", analyzed_count)

    st.markdown("---")

    for paper in related_papers:
        score = paper.get("topic_overlap_score", 0)
        verification = paper.get("verification")
        ver_status = (
            verification.get("status", "not checked")
            if isinstance(verification, dict)
            else "not checked"
        )
        ver_badge = _badge(ver_status, "ok" if "verified" in ver_status else "warn")
        doi = paper.get("doi")

        st.markdown(
            f"""
            <div class="ri-paper-card">
              <h4>{paper.get('title', 'Untitled record')}</h4>
              <div class="meta">
                {', '.join(paper.get('authors') or []) or 'Authors unavailable'}
                &nbsp;·&nbsp; {paper.get('year') or 'Year unknown'}
                {ver_badge}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        bar_html = _overlap_bar(score)
        st.markdown(
            f"{bar_html}<small>Topic overlap: **{score:.0%}** "
            f"({'terms: ' + ', '.join(paper.get('topic_overlap_terms', [])) if paper.get('topic_overlap_terms') else 'no matched terms'})</small>",
            unsafe_allow_html=True,
        )
        if doi:
            st.markdown(f"🔗 [https://doi.org/{doi}](https://doi.org/{doi})")
        if paper.get("alternate_dois"):
            st.caption("Other DOI records for same title/author: " + ", ".join(paper["alternate_dois"]))
        st.markdown("")

    screened_out = bundle.get("screened_out_papers", [])
    if screened_out:
        with st.expander(f"🔽 Screened out ({len(screened_out)} records)"):
            st.caption("Not included in comparison or related-PDF analysis.")
            st.dataframe(
                [
                    {
                        "Title": item.get("title"),
                        "Overlap": f"{item.get('topic_overlap_score', 0):.0%}",
                        "Matched terms": ", ".join(item.get("topic_overlap_terms", [])),
                        "Reason": item.get("screening_reason"),
                    }
                    for item in screened_out
                ],
                use_container_width=True,
                hide_index=True,
            )

    if analyzed_count == 0:
        st.info(
            "No related-paper PDFs were analyzed. Enable **Analyze eligible related PDFs** "
            "in the sidebar to include open-access papers in the comparison."
        )


def _render_comparison_tab(bundle: Dict[str, Any]) -> None:
    import streamlit as st

    comparison = bundle.get("comparison", {})
    source_count = comparison.get("source_count", 0)

    col1, col2 = st.columns(2)
    col1.metric("Sources compared", source_count)
    col2.metric(
        "Fields compared",
        len(comparison.get("field_comparisons", {})),
    )

    if source_count < 2:
        st.info(
            "Cross-paper comparison needs at least **2 successfully analyzed papers**. "
            "Finding metadata records alone is not sufficient — enable 'Analyze eligible "
            "related PDFs' in the sidebar."
        )
        return

    st.markdown("---")
    for field_name, result in comparison.get("field_comparisons", {}).items():
        status = result.get("status", "unknown")
        badge = _badge(status, "ok" if status == "consistent" else "warn")
        with st.expander(f"{field_name.replace('_', ' ').title()}", expanded=False):
            st.markdown(badge, unsafe_allow_html=True)
            st.caption("Exact-text pattern only — not a semantic equivalence or contradiction judgment.")
            values = result.get("values", [])
            if not values:
                st.write("No comparable extracted values.")
            for item in values:
                st.markdown(f"**{item.get('title') or item.get('source_id')}**")
                st.write(_format_value(item.get("value")))
                evidence = item.get("evidence", {})
                st.caption(
                    f"Reference integrity: {evidence.get('status', 'unknown')} — "
                    "source passage review still required."
                )
                for ref in evidence.get("citations", []):
                    page = ref.get("start_page", "?")
                    st.caption(f"📄 {ref.get('document_id')} / {ref.get('chunk_id')} / p.{page}")
                    if ref.get("passage_excerpt"):
                        st.markdown(
                            f'<div class="ri-evidence-card">{ref["passage_excerpt"]}</div>',
                            unsafe_allow_html=True,
                        )


def _render_gaps_tab(bundle: Dict[str, Any]) -> None:
    import streamlit as st

    candidates = bundle.get("gap_analysis", {}).get("research_gap_candidates", [])

    st.caption(
        "These prompts come from **stated limitations** in the analyzed papers. "
        "They are not confirmed research gaps or claims of novelty."
    )

    if not candidates:
        st.info("No limitation-based follow-up candidates were extracted.")
    else:
        st.metric("Follow-up candidates", len(candidates))
        st.markdown("---")
        for i, candidate in enumerate(candidates, 1):
            with st.container(border=True):
                st.markdown(f"**{i}. {candidate.get('candidate_question', 'Untitled')}**")
                st.write(f"📝 Limitation: {candidate.get('stated_limitation', '—')}")
                for reference in candidate.get("citations", []):
                    st.caption(
                        f"📄 {reference.get('document_id')} / "
                        f"{reference.get('chunk_id')} / "
                        f"p.{reference.get('start_page', '?')}"
                    )

    scope_warning = bundle.get("gap_analysis", {}).get("scope_warning")
    if scope_warning:
        st.warning(scope_warning)


def _render_report_tab(bundle: Dict[str, Any]) -> None:
    import streamlit as st

    report = bundle.get("report", {})
    warnings_list = report.get("warnings", [])

    if warnings_list:
        with st.expander(f"⚠️ Report warnings ({len(warnings_list)})"):
            for w in warnings_list:
                st.markdown(f"- {w}")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.download_button(
            "⬇️ Download report (Markdown)",
            data=report.get("markdown", ""),
            file_name="research-analysis-report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        word_count = len(report.get("markdown", "").split())
        st.metric("Report length", f"{word_count:,} words")

    with st.expander("📄 Preview report", expanded=False):
        st.markdown(report.get("markdown", "*No report generated.*"))


# ── Main entry point ──────────────────────────────────────────────────────────

def main():
    import streamlit as st

    st.set_page_config(
        page_title="AI Research Intelligence",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": "**AI Research Intelligence** — PDF-to-report pipeline powered by Groq.",
        },
    )

    # Inject CSS
    st.markdown(_CSS, unsafe_allow_html=True)

    # Header banner
    st.markdown(
        """
        <div class="ri-banner">
          <h1>📚 AI Research Intelligence</h1>
          <p>Upload a research PDF → extract structured knowledge → discover & compare related literature</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Environment warnings (shown prominently if key is missing)
    env_issues = _status_checks()
    if env_issues:
        for issue in env_issues:
            st.warning(f"⚙️ Setup required: {issue}")

    # Sidebar
    discover_related, minimum_related_overlap, analyze_related = _render_sidebar(
        discover_related_default=True
    )

    # Upload form
    st.markdown("### Upload your paper")
    uploaded_file, confirm_primary, confirm_related, run_clicked = _render_upload_form(
        analyze_related
    )

    # Validation
    if run_clicked and uploaded_file is None:
        st.error("Please choose a research-paper PDF before starting the analysis.")
    elif run_clicked and not confirm_primary:
        st.error("Confirm that you authorize sending selected paper excerpts to Groq.")
    elif run_clicked and analyze_related and not confirm_related:
        st.error(
            "Confirm that you authorize sending related-paper excerpts to Groq, "
            "or disable related-paper analysis in the sidebar."
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
            progress_bar = st.progress(0, text="Starting analysis…")
            _stage_steps = {
                "extraction": 10,
                "chunking": 20,
                "embedding": 35,
                "analysis": 60,
                "related_search": 75,
                "comparison": 88,
                "report": 95,
            }

            def progress(stage: str, message: str) -> None:
                pct = _stage_steps.get(stage, 50)
                stage_placeholder.info(
                    f"**{stage.replace('_', ' ').title()}** — {message}"
                )
                progress_bar.progress(pct, text=message[:80])

            with st.spinner("Analyzing — this may take a few minutes…"):
                bundle = run_analysis(
                    str(temporary_path),
                    document_id=Path(uploaded_file.name).stem,
                    discover_related=discover_related,
                    analyze_related=analyze_related and confirm_related,
                    minimum_related_overlap=minimum_related_overlap,
                    progress_callback=progress,
                )

            progress_bar.progress(100, text="Done!")
            stage_placeholder.empty()
            st.session_state["research_bundle"] = bundle
            st.session_state["current_document_id"] = Path(uploaded_file.name).stem
            st.success("✅ Analysis complete — review the results below.")

        except ValueError as error:
            st.session_state["research_bundle"] = None
            st.session_state["current_document_id"] = None
            st.error(
                "Analysis failed. This can happen if the PDF has no extractable text. "
                "Please check the PDF format and try again."
            )
        except Exception:
            st.session_state["research_bundle"] = None
            st.session_state["current_document_id"] = None
            st.error(
                "Analysis failed. This can happen if the PDF has no extractable text, "
                "the Groq API key is invalid, or a network issue occurred. "
                "Check the sidebar for environment status and try again."
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    bundle = st.session_state.get("research_bundle")
    if not isinstance(bundle, dict):
        # Empty state: show instructions
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                "#### 1️⃣ Upload\nDrop a text-based research PDF (not scanned images)."
            )
        with c2:
            st.markdown(
                "#### 2️⃣ Configure\nChoose options in the sidebar — related-paper "
                "discovery is on by default."
            )
        with c3:
            st.markdown(
                "#### 3️⃣ Explore\nReview structured findings across 6 tabs: summary, "
                "evidence, related papers, comparison, follow-ups, and report."
            )
        return

    current_document_id = st.session_state.get("current_document_id")
    if current_document_id is None:
        return

    analysis = bundle["analysis"]
    if analysis.get("document_id") != current_document_id:
        st.warning("Previous analysis results are not available for this upload.")
        return

    # Results header
    st.markdown("---")
    author_values = analysis.get("authors")
    author_text = (
        ", ".join(author_values)
        if isinstance(author_values, list) and author_values
        else "Authors not extracted"
    )
    st.markdown(
        f"### {analysis.get('title') or 'Untitled paper'}\n"
        f"<small style='color:var(--muted)'>{author_text} &nbsp;·&nbsp; "
        f"{analysis.get('publication_year') or 'Year unavailable'}</small>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "📋 Quick Summary",
        "🔎 Evidence Review",
        "🔗 Related Papers",
        "⚖️ Comparison",
        "💡 Follow-ups",
        "📄 Report",
    ])

    with tabs[0]:
        _render_summary_tab(analysis)
    with tabs[1]:
        _render_evidence_tab(analysis)
    with tabs[2]:
        _render_related_tab(bundle)
    with tabs[3]:
        _render_comparison_tab(bundle)
    with tabs[4]:
        _render_gaps_tab(bundle)
    with tabs[5]:
        _render_report_tab(bundle)


def _format_value(value):
    if value is None or value == "" or value == []:
        return "Not available"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


if __name__ == "__main__":
    main()
