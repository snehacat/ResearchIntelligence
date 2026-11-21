"""Phase 1 evaluation harness.

Runs the full pipeline (PDF extraction → PaperAnalyzer) against every
annotated paper in the gold set, computes all Phase 1 metrics, and writes
a timestamped Markdown + JSON report to evaluation/results/.

Usage
-----
    python evaluation/run_harness.py --gold-dir evaluation/gold_set

Required environment variable
    GROQ_API_KEY or GEMINI_API_KEY, depending on the selected backend.

Optional
    RESEARCHINTELLIGENCE_LLM_MODEL   — override the selected provider's model
    RESEARCHINTELLIGENCE_LLM_BACKEND — override backend (groq | gemini)

Cost confirmation
    --confirm-api-costs  must be passed to allow LLM API calls.
    --confirm-groq-costs remains as a compatibility alias.

Offline / dry-run mode
    --dry-run  skips LLM extraction; all metrics are reported as N/A.
    Useful for testing the harness without spending API credits.

Split handling (item 7)
    --split dev|test   filters annotations by their 'split' field.
    Running the test split prints a prominent warning and writes an
    audit log entry to evaluation/results/test_split_audit.log.

Repeated runs (item 6)
    --runs N (default 1)  runs the full evaluation N times and reports
    mean ± range for each metric. Prints a baseline provider request estimate
    before running; retries and JSON recovery can add requests.

Citation sample (item 3)
    --citation-sample-dir  path to directory of citation sample .json files.
    When supplied, computes auto/human agreement for each matching file.

Output
------
    evaluation/results/harness_<TIMESTAMP>/
        report.json         — full metrics (all runs) in JSON
        report.md           — human-readable Markdown report
        per_paper/
            <document_id>_run<N>.json  — per-paper analysis output (audit)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.gold_set_schema import GoldSetValidationError, load_gold_set
from evaluation.metrics import (
    HarnessResult,
    PaperResult,
    _DEFAULT_K_VALUES,
    evaluate_paper,
)

_AUDIT_LOG = _PROJECT_ROOT / "evaluation" / "results" / "test_split_audit.log"


# ── Utilities ─────────────────────────────────────────────────────────────────

def _build_analyzer():
    """Construct a fully wired PaperAnalyzer using the same pattern as app.py."""
    from analysis.paper_analyzer import PaperAnalyzer
    from chunking.text_chunker import TextChunker
    from embeddings.embedding_generator import EmbeddingGenerator
    from qa.answer_generator import AnswerGenerator
    from retrieval.semantic_retriever import SemanticRetriever

    retriever = SemanticRetriever(EmbeddingGenerator())
    answer_generator = AnswerGenerator(retriever)
    return PaperAnalyzer(TextChunker(), retriever, answer_generator)


def _run_analysis(
    analyzer,
    source_pdf: str,
    document_id: str,
    allowed_root: Path = _PROJECT_ROOT,
) -> Dict[str, Any]:
    root = allowed_root.resolve()
    pdf_path = (root / source_pdf).resolve()
    try:
        pdf_path.relative_to(root)
    except ValueError:
        raise ValueError("PDF path must remain within the configured papers directory.") from None
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    from pdf_processing.text_extractor import TextExtractor

    extraction = TextExtractor().extract_from_file(str(pdf_path))
    return analyzer.analyze_paper(extraction, document_id=document_id)


def _run_dry(annotation: Dict[str, Any]) -> Dict[str, Any]:
    """Return a stub analysis with all fields null (dry-run / offline mode)."""
    from analysis.paper_analyzer import PaperAnalyzer
    return {
        "document_id": annotation["document_id"],
        "evidence_citations": {},
        **{f: None for f in PaperAnalyzer.SCHEMA_FIELDS},
    }


def _log_test_split_run(
    output_dir: Path,
    num_papers: int,
    num_runs: int,
    dry_run: bool,
) -> None:
    """Append an audit entry every time the test split is evaluated."""
    _AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = (
        f"{datetime.now(timezone.utc).isoformat()} | "
        f"output={output_dir} | "
        f"papers={num_papers} | "
        f"runs={num_runs} | "
        f"dry_run={dry_run}\n"
    )
    with open(_AUDIT_LOG, "a", encoding="utf-8") as fh:
        fh.write(entry)


# ── Report writers ─────────────────────────────────────────────────────────────

def _write_json_report(
    output_dir: Path,
    all_run_results: List[HarnessResult],
    timings: Dict[str, List[float]],
    split: Optional[str],
    dry_run: bool,
    citation_agreement_results: Optional[List[Dict[str, Any]]],
    annotation_independence_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Write report.json and return the dict."""
    runs_data = []
    for run_idx, result in enumerate(all_run_results):
        run_data = result.to_dict()
        if dry_run:
            for key in run_data["summary"]:
                if key != "num_papers":
                    run_data["summary"][key] = None
            for paper in run_data["per_paper"]:
                for key in (
                    "mean_token_f1",
                    "mean_set_f1",
                    "hit_rate@1",
                    "hit_rate@3",
                    "hit_rate@5",
                    "mrr",
                    "citation_traceability_rate",
                ):
                    paper[key] = None
                for field_result in paper["fields"]:
                    for key in (
                        "correct_null",
                        "mrr",
                        "citation_traceable",
                        "token_f1",
                        "set_precision",
                        "set_recall",
                        "set_f1",
                        "hit@1",
                        "hit@3",
                        "hit@5",
                    ):
                        if key in field_result:
                            field_result[key] = None
        runs_data.append(run_data)

    # Aggregate over runs
    num_runs = len(all_run_results)
    summary_keys = list(all_run_results[0].to_dict()["summary"].keys()) if all_run_results else []

    aggregate: Dict[str, Any] = {}
    if num_runs > 1:
        for key in summary_keys:
            if key == "num_papers":
                aggregate[key] = all_run_results[0].to_dict()["summary"][key]
                continue
            vals = [r.to_dict()["summary"].get(key) for r in all_run_results]
            nums = [v for v in vals if v is not None]
            if nums:
                aggregate[key] = {
                    "mean": round(sum(nums) / len(nums), 4),
                    "min": round(min(nums), 4),
                    "max": round(max(nums), 4),
                    "runs": nums,
                }
            else:
                aggregate[key] = None
    else:
        aggregate = all_run_results[0].to_dict()["summary"] if all_run_results else {}
    if dry_run:
        for key in aggregate:
            if key != "num_papers":
                aggregate[key] = None

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": split or "all",
        "num_runs": num_runs,
        "dry_run": dry_run,
        "aggregate_summary": aggregate,
        "annotation_independence": annotation_independence_summary,
        "runs": runs_data,
        "timings_seconds": {
            k: [round(v, 2) for v in vs] for k, vs in timings.items()
        },
    }
    if citation_agreement_results:
        report["citation_agreement"] = citation_agreement_results

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return report


def _write_markdown_report(
    output_dir: Path,
    report: Dict[str, Any],
    all_run_results: List[HarnessResult],
) -> None:
    """Write a human-readable Markdown report."""
    lines: List[str] = []
    lines += [
        "# Phase 1 Evaluation Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Split: `{report['split']}`  |  Runs: {report['num_runs']}  |  Dry-run: {report['dry_run']}",
        "",
    ]
    if report["dry_run"]:
        lines += [
            "> Dry run: no model analysis was performed; accuracy and retrieval metrics are N/A.",
            "",
        ]

    # Annotation independence
    ai = report.get("annotation_independence", {})
    lines += [
        "## Annotation Independence (item 8)",
        "",
        f"- Papers declaring `annotated_without_model_output: true`: "
        f"**{ai.get('without_model_output_count', 0)}** / {ai.get('total', 0)}",
        "",
    ]

    # Aggregate summary table
    agg = report.get("aggregate_summary", {})
    num_runs = report["num_runs"]
    lines += ["## Aggregate Metrics", ""]

    def _fmt(v: Any) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, dict):
            return f"{v.get('mean', 'N/A')} (min {v.get('min', 'N/A')} – max {v.get('max', 'N/A')})"
        return str(v)

    metric_rows = [
        ("Papers", "num_papers"),
        ("Macro token-F1 (string fields, rough proxy)", "macro_token_f1"),
        ("Macro set-F1 (list fields)", "macro_set_f1"),
        ("Macro hit-rate@1", "macro_hit_rate@1"),
        ("Macro hit-rate@3", "macro_hit_rate@3"),
        ("Macro hit-rate@5", "macro_hit_rate@5"),
        ("Macro MRR", "macro_mrr"),
        ("Macro citation traceability rate ⚠️", "macro_citation_traceability_rate"),
    ]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for label, key in metric_rows:
        lines.append(f"| {label} | {_fmt(agg.get(key))} |")
    lines.append("")

    # Citation traceability footnote
    if all_run_results:
        footnotes = all_run_results[0].to_dict().get("footnotes", {})
        ct_note = footnotes.get("citation_traceability_rate", "")
        lines += [
            "> ⚠️ **Citation traceability** — " + ct_note,
            "",
        ]

    # Citation agreement (item 3)
    ca = report.get("citation_agreement")
    if ca:
        lines += ["## Human-Labeled Citation Agreement (item 3)", ""]
        for entry in ca:
            lines += [
                f"### {entry['document_id']}",
                f"- Pairs: {entry['num_pairs']}",
                f"- Agreement rate: **{entry.get('agreement_rate', 'N/A')}**",
                f"- Confusion: TP={entry['confusion']['tp']}, "
                f"FP={entry['confusion']['fp']}, "
                f"FN={entry['confusion']['fn']}, "
                f"TN={entry['confusion']['tn']}",
                f"- {entry.get('footnote', '')}",
                "",
            ]

    # Per-paper summary (run 1 only if multiple runs)
    if all_run_results:
        lines += ["## Per-Paper Summary (run 1)", ""]
        first_run = report["runs"][0]
        lines.append("| Paper | token-F1 | set-F1 | HR@1 | HR@3 | HR@5 | MRR | Cite-Trace |")
        lines.append("|-------|----------|--------|------|------|------|-----|------------|")
        for pp in first_run.get("per_paper", []):
            lines.append(
                f"| {pp['document_id']} "
                f"| {_fmt(pp.get('mean_token_f1'))} "
                f"| {_fmt(pp.get('mean_set_f1'))} "
                f"| {_fmt(pp.get('hit_rate@1'))} "
                f"| {_fmt(pp.get('hit_rate@3'))} "
                f"| {_fmt(pp.get('hit_rate@5'))} "
                f"| {_fmt(pp.get('mrr'))} "
                f"| {_fmt(pp.get('citation_traceability_rate'))} |"
            )
        lines.append("")

    # Limitations
    lines += [
        "## Limitations and Notes",
        "",
        "- Token-F1 is a rough surface-form proxy for free-text fields.",
        "- Set-F1 measures item-level overlap; it does not detect semantic equivalence.",
        "- Citation traceability rate measures page-level pointer accuracy only.",
        "  It does NOT verify that the cited passage supports the extracted claim.",
        "- MRR and hit-rate use page-overlap; chunk boundaries affect results.",
        "- LLM non-determinism means repeated runs may produce different metrics.",
        "  Use --runs N to quantify variance.",
        "",
        f"Full JSON: `{output_dir / 'report.json'}`",
    ]

    with open(output_dir / "report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")
    print("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 1 evaluation harness — "
            "field accuracy, retrieval hit-rate@k, MRR, citation traceability."
        )
    )
    parser.add_argument(
        "--gold-dir",
        default="evaluation/gold_set",
        help="Directory containing gold-set .json annotation files.",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default=None,
        help=(
            "Evaluate only this split. "
            "Running 'test' prints a warning and writes an audit log entry."
        ),
    )
    parser.add_argument(
        "--papers-dir",
        default=None,
        help="Override PDF search directory (default: use source_pdf paths in annotations).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: evaluation/results/harness_<TIMESTAMP>).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip LLM analysis; all metrics are N/A. "
            "Validates gold set and harness logic without API calls."
        ),
    )
    parser.add_argument(
        "--confirm-groq-costs",
        "--confirm-api-costs",
        dest="confirm_api_costs",
        action="store_true",
        help=(
            "Required to allow real LLM API calls; "
            "--confirm-groq-costs is a compatibility alias."
        ),
    )
    parser.add_argument(
        "--document-id",
        default=None,
        help="Evaluate only this document_id instead of the full gold set.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help=(
            "Number of times to run the full evaluation (default: 1). "
            "Mean and min-max range are reported when runs > 1."
        ),
    )
    parser.add_argument(
        "--citation-sample-dir",
        default=None,
        help=(
            "Directory of human-labeled citation sample .json files "
            "(evaluation/citation_sample/ by default when the dir exists)."
        ),
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be a positive integer")
    backend_name = None
    if not args.dry_run:
        from analysis.backend_factory import (
            configured_backend_name,
            configured_model_name,
        )

        backend_name = configured_backend_name()
        if backend_name not in {"groq", "gemini"}:
            parser.error(
                "RESEARCHINTELLIGENCE_LLM_BACKEND must be one of: groq, gemini"
            )
        model_name = configured_model_name(backend_name)
        if not model_name:
            parser.error(
                f"Set RESEARCHINTELLIGENCE_LLM_MODEL when using {backend_name}."
            )

    # ── Load & validate gold set ──────────────────────────────────────────────
    gold_dir = _PROJECT_ROOT / args.gold_dir
    try:
        annotations = load_gold_set(gold_dir, split=args.split)
    except GoldSetValidationError as exc:
        print(f"Gold-set validation error: {exc}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Malformed JSON in gold set: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.document_id:
        annotations = [a for a in annotations if a["document_id"] == args.document_id]
        if not annotations:
            print(f"No annotation found for document_id='{args.document_id}'", file=sys.stderr)
            sys.exit(1)

    if not annotations:
        print(
            f"No annotations found (split='{args.split}', gold_dir='{gold_dir}').",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Gold set loaded: {len(annotations)} paper(s).")

    # ── Item 7: test-split warning and audit ──────────────────────────────────
    if args.split == "test":
        print(
            "\n" + "=" * 60,
            "\nWARNING: You are running on the TEST split.",
            "\nTest-split results should only be recorded for final reported numbers.",
            "\nTuning decisions must be made on the dev split only.",
            "\n" + "=" * 60 + "\n",
        )

    # ── Item 8: annotation independence summary ───────────────────────────────
    total_ann = len(annotations)
    without_model_count = sum(
        1 for a in annotations if a.get("annotated_without_model_output") is True
    )
    annotation_independence_summary = {
        "total": total_ann,
        "without_model_output_count": without_model_count,
        "with_model_output_count": total_ann - without_model_count,
    }
    print(
        f"Annotation independence: {without_model_count}/{total_ann} files declare "
        f"annotated_without_model_output=true."
    )

    # ── Safety gate for real API calls ───────────────────────────────────────
    if not args.dry_run and not args.confirm_api_costs:
        api_key_env_var = "GROQ_API_KEY" if backend_name == "groq" else "GEMINI_API_KEY"
        print(
            f"\nThis harness will send selected PDF excerpts to the {backend_name} API "
            f"({len(annotations)} paper(s) × {args.runs} run(s) = "
            f"{len(annotations) * args.runs} baseline request(s)), which may incur charges.\n"
            "Provider retries, JSON fallback, or malformed-response recovery can add "
            "requests; this baseline is not a maximum or a billing guarantee.\n"
            f"Re-run with --confirm-api-costs to proceed, or --dry-run to skip LLM calls. "
            f"Required key: {api_key_env_var}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.dry_run:
        estimated_requests = len(annotations) * args.runs
        print(
            f"Baseline {backend_name} request estimate (no retry/fallback): {estimated_requests} "
            f"({len(annotations)} paper(s) × {args.runs} run(s)). This is not a maximum "
            "or a billing guarantee; provider retries and JSON recovery can add requests."
        )
        if backend_name == "gemini":
            print(
                "WARNING: The Gemini integration is unverified and is not claimed as fully supported.",
                file=sys.stderr,
            )
        api_key_env_var = "GROQ_API_KEY" if backend_name == "groq" else "GEMINI_API_KEY"
        if not os.environ.get(api_key_env_var, "").strip():
            print(
                f"Error: {api_key_env_var} environment variable is not set.",
                file=sys.stderr,
            )
            sys.exit(1)

    # ── Build output dir ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = _PROJECT_ROOT / args.output_dir
    else:
        output_dir = _PROJECT_ROOT / "evaluation" / "results" / f"harness_{timestamp}"
    per_paper_dir = output_dir / "per_paper"
    if output_dir.exists():
        parser.error(f"Output directory already exists: {output_dir}")

    # ── Build analyzer once (reused across runs) ──────────────────────────────
    analyzer = None
    if not args.dry_run:
        print("Building PaperAnalyzer (embedding model may download on first run)…")
        analyzer = _build_analyzer()

    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output directory already exists: {output_dir}")

    # ── Item 7: audit log for test split ─────────────────────────────────────
    if args.split == "test":
        _log_test_split_run(output_dir, len(annotations), args.runs, args.dry_run)

    # ── Run N times (item 6) ──────────────────────────────────────────────────
    all_run_results: List[HarnessResult] = []
    timings: Dict[str, List[float]] = {}

    for run_idx in range(args.runs):
        if args.runs > 1:
            print(f"\n── Run {run_idx + 1}/{args.runs} ──────────────────────────")
        paper_results: List[PaperResult] = []

        for annotation in annotations:
            doc_id = annotation["document_id"]
            print(f"\nProcessing: {doc_id}")

            t0 = time.perf_counter()
            try:
                if args.dry_run:
                    analysis = _run_dry(annotation)
                else:
                    source_pdf = annotation["source_pdf"]
                    allowed_pdf_root = _PROJECT_ROOT
                    if args.papers_dir:
                        allowed_pdf_root = _PROJECT_ROOT / args.papers_dir
                        source_pdf = Path(source_pdf).name
                    analysis = _run_analysis(
                        analyzer,
                        source_pdf,
                        doc_id,
                        allowed_root=allowed_pdf_root,
                    )
            except FileNotFoundError as exc:
                print(f"  SKIP — {exc}")
                continue
            except Exception as exc:
                print(f"  ERROR — {exc}")
                continue

            elapsed = time.perf_counter() - t0
            timings.setdefault(doc_id, []).append(elapsed)
            print(f"  Analysis complete in {elapsed:.1f}s")

            # Save per-paper analysis for auditability
            per_paper_dir.mkdir(parents=True, exist_ok=True)
            with open(
                per_paper_dir / f"{doc_id}_run{run_idx + 1}.json",
                "w", encoding="utf-8",
            ) as fh:
                json.dump(analysis, fh, indent=2, ensure_ascii=False, default=str)
                fh.write("\n")

            result = evaluate_paper(annotation, analysis)
            paper_results.append(result)

            if args.dry_run:
                print("  Metrics: N/A (dry run; no model analysis performed)")
            else:
                hr_at_3 = result.hit_rate_at_k(3)
                print(
                    f"  token-F1={result.mean_token_f1:.3f}  "
                    f"set-F1={result.mean_set_f1!s}  "
                    f"HR@3={hr_at_3!s}  "
                    f"MRR={result.mrr!s}  "
                    f"cite-trace={result.citation_traceability_rate!s}"
                )

        if not paper_results:
            print(f"\nRun {run_idx + 1}: no papers evaluated.", file=sys.stderr)
            continue

        all_run_results.append(
            HarnessResult(paper_results=paper_results, k_values=_DEFAULT_K_VALUES)
        )

    if not all_run_results:
        print("\nNo runs completed successfully.", file=sys.stderr)
        sys.exit(1)

    # ── Citation agreement (item 3) ───────────────────────────────────────────
    citation_agreement_results: Optional[List[Dict[str, Any]]] = None
    citation_sample_dir = None
    if not args.dry_run and args.citation_sample_dir:
        citation_sample_dir = _PROJECT_ROOT / args.citation_sample_dir
    elif not args.dry_run:
        default_cs = _PROJECT_ROOT / "evaluation" / "citation_sample"
        if default_cs.is_dir():
            citation_sample_dir = default_cs

    if citation_sample_dir and citation_sample_dir.is_dir():
        from evaluation.citation_sample_schema import (
            CitationSampleValidationError,
            compute_citation_agreement,
            load_citation_sample,
        )
        citation_agreement_results = []
        for cs_path in sorted(citation_sample_dir.glob("*.json")):
            try:
                sample = load_citation_sample(cs_path)
            except (CitationSampleValidationError, json.JSONDecodeError) as exc:
                print(f"  Citation sample validation error ({cs_path.name}): {exc}")
                continue
            # Use run-1 analysis for the matching document
            doc_id = sample["document_id"]
            matching_analysis_path = per_paper_dir / f"{doc_id}_run1.json"
            if not matching_analysis_path.exists():
                print(f"  No analysis found for citation sample '{doc_id}'; skipping agreement.")
                continue
            with open(matching_analysis_path, encoding="utf-8") as fh:
                analysis = json.load(fh)
            agreement = compute_citation_agreement(sample, analysis)
            citation_agreement_results.append(agreement)
            print(
                f"  Citation agreement for {doc_id}: "
                f"{agreement.get('agreement_rate', 'N/A')} "
                f"({agreement['num_pairs']} pairs)"
            )

    # ── Write reports ─────────────────────────────────────────────────────────
    report = _write_json_report(
        output_dir,
        all_run_results,
        timings,
        args.split,
        args.dry_run,
        citation_agreement_results,
        annotation_independence_summary,
    )
    _write_markdown_report(output_dir, report, all_run_results)
    print(f"\nFull report: {output_dir}")


if __name__ == "__main__":
    main()
