"""Phase 1 evaluation harness.

Runs the full pipeline (PDF extraction → PaperAnalyzer) against every
annotated paper in the gold set, computes the three Phase 1 metrics, and
writes a timestamped report to evaluation/results/.

Usage
-----
    python evaluation/run_harness.py --gold-dir evaluation/gold_set

Required environment variable
    GROQ_API_KEY   — Groq LLM key (analysis calls the real Groq backend)

Optional environment variables
    RESEARCHINTELLIGENCE_LLM_MODEL   — override Groq model
    RESEARCHINTELLIGENCE_LLM_BACKEND — override backend (groq | gemini)

This script is intentionally explicit about costs:
  --confirm-groq-costs  must be passed to allow LLM API calls.

Dry-run / offline mode
    Pass --dry-run to skip LLM extraction and report metrics as N/A.
    Useful for testing the harness itself without spending API credits.

Output
------
    evaluation/results/harness_<TIMESTAMP>/
        report.json         — full metrics report
        report_summary.txt  — human-readable summary
        per_paper/
            <document_id>.json  — per-paper analysis output (for audit)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.gold_set_schema import GoldSetValidationError, load_gold_set
from evaluation.metrics import HarnessResult, evaluate_paper


def _build_analyzer():
    """Construct a fully wired PaperAnalyzer using the same pattern as app.py."""
    from analysis.paper_analyzer import PaperAnalyzer
    from chunking.text_chunker import TextChunker
    from embeddings.embedding_generator import EmbeddingGenerator
    from qa.answer_generator import AnswerGenerator
    from retrieval.semantic_retriever import SemanticRetriever

    retriever = SemanticRetriever(EmbeddingGenerator())
    answer_generator = AnswerGenerator(retriever)
    return PaperAnalyzer(
        TextChunker(),
        retriever,
        answer_generator,
    )


def _run_analysis(
    analyzer,
    source_pdf: str,
    document_id: str,
) -> Dict[str, Any]:
    """Run extraction against a single PDF; return the analysis dict."""
    from pdf_processing.text_extractor import TextExtractor

    pdf_path = _PROJECT_ROOT / source_pdf
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    extraction = TextExtractor().extract_from_file(str(pdf_path))
    return analyzer.analyze_paper(extraction, document_id=document_id)


def _run_dry(annotation: Dict[str, Any]) -> Dict[str, Any]:
    """Return a stub analysis with all fields null (dry-run / offline mode)."""
    from analysis.paper_analyzer import PaperAnalyzer

    return {
        "document_id": annotation["document_id"],
        "evidence_citations": {},
        **{field: None for field in PaperAnalyzer.SCHEMA_FIELDS},
    }


def _write_report(output_dir: Path, result: HarnessResult, timings: Dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_paper_dir = output_dir / "per_paper"
    per_paper_dir.mkdir(exist_ok=True)

    report = result.to_dict()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["timings_seconds"] = {k: round(v, 2) for k, v in timings.items()}

    with open(output_dir / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    # Human-readable summary
    summary_lines = [
        "Phase 1 Evaluation Harness — Summary",
        "=" * 50,
        f"Papers evaluated : {report['summary']['num_papers']}",
        f"top_k            : {report['top_k']}",
        "",
        "── Aggregate metrics ──────────────────────────",
    ]
    s = report["summary"]
    for label, key in [
        ("Macro token-F1 (extraction accuracy)", "macro_token_f1"),
        ("Macro hit-rate@k (retrieval)", "macro_hit_rate_at_k"),
        ("Macro citation-support rate", "macro_citation_support_rate"),
    ]:
        v = s.get(key)
        summary_lines.append(f"  {label:<42} {v if v is not None else 'N/A'}")
    summary_lines += [
        "",
        "── Per-paper summary ──────────────────────────",
    ]
    for pp in report["per_paper"]:
        summary_lines.append(
            f"  {pp['document_id']:<40} "
            f"F1={pp['mean_token_f1']!s:<6}  "
            f"HR={pp['hit_rate_at_k']!s:<6}  "
            f"CR={pp['citation_support_rate']!s:<6}"
        )
    summary_lines += [
        "",
        "NOTE: Token-F1 is a surface-form proxy.  Human review required.",
        f"Full report: {output_dir / 'report.json'}",
    ]
    summary_text = "\n".join(summary_lines)
    with open(output_dir / "report_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(summary_text)
        fh.write("\n")
    print(summary_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 evaluation harness — field accuracy, retrieval hit-rate, citation support."
    )
    parser.add_argument(
        "--gold-dir",
        default="evaluation/gold_set",
        help="Directory containing gold-set .json annotation files.",
    )
    parser.add_argument(
        "--papers-dir",
        default=None,
        help="Override PDF search directory (default: use source_pdf paths in annotations).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the report (default: evaluation/results/harness_<TIMESTAMP>).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks to consider for hit-rate@k (default: 3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip LLM analysis; report metrics as N/A.  "
            "Validates gold set and harness logic without API calls."
        ),
    )
    parser.add_argument(
        "--confirm-groq-costs",
        action="store_true",
        help="Required to allow real Groq API calls.  Each paper uses one LLM request.",
    )
    parser.add_argument(
        "--document-id",
        default=None,
        help="Evaluate only this document_id instead of the full gold set.",
    )
    args = parser.parse_args()

    # ── Load & validate gold set ──────────────────────────────────────────────
    gold_dir = _PROJECT_ROOT / args.gold_dir
    try:
        annotations = load_gold_set(gold_dir)
    except GoldSetValidationError as exc:
        print(f"Gold-set validation error: {exc}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Malformed JSON in gold set: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.document_id:
        annotations = [a for a in annotations if a["document_id"] == args.document_id]
        if not annotations:
            print(
                f"No annotation found for document_id='{args.document_id}'",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Gold set loaded: {len(annotations)} paper(s).")

    # ── Safety gate for real API calls ───────────────────────────────────────
    if not args.dry_run and not args.confirm_groq_costs:
        print(
            "\nThis harness will send selected PDF excerpts to the Groq API "
            "(one request per paper), which may incur charges.\n"
            "Re-run with --confirm-groq-costs to proceed, or --dry-run to skip LLM calls.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Build analyzer (once; reused across papers) ───────────────────────────
    analyzer = None
    if not args.dry_run:
        if not os.environ.get("GROQ_API_KEY"):
            print("Error: GROQ_API_KEY environment variable is not set.", file=sys.stderr)
            sys.exit(1)
        print("Building PaperAnalyzer (embedding model may download on first run)…")
        analyzer = _build_analyzer()

    # ── Run per paper ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = _PROJECT_ROOT / args.output_dir
    else:
        output_dir = _PROJECT_ROOT / "evaluation" / "results" / f"harness_{timestamp}"

    per_paper_dir = output_dir / "per_paper"

    paper_results = []
    timings: Dict[str, float] = {}

    for annotation in annotations:
        doc_id = annotation["document_id"]
        print(f"\nProcessing: {doc_id}")

        t0 = time.perf_counter()
        try:
            if args.dry_run:
                analysis = _run_dry(annotation)
            else:
                # Override source_pdf directory if --papers-dir given
                source_pdf = annotation["source_pdf"]
                if args.papers_dir:
                    source_pdf = str(
                        Path(args.papers_dir) / Path(source_pdf).name
                    )
                analysis = _run_analysis(analyzer, source_pdf, doc_id)

        except FileNotFoundError as exc:
            print(f"  SKIP — {exc}")
            continue
        except Exception as exc:
            print(f"  ERROR — {exc}")
            continue

        elapsed = time.perf_counter() - t0
        timings[doc_id] = elapsed
        print(f"  Analysis complete in {elapsed:.1f}s")

        # Save per-paper analysis for auditability
        per_paper_dir.mkdir(parents=True, exist_ok=True)
        with open(per_paper_dir / f"{doc_id}.json", "w", encoding="utf-8") as fh:
            json.dump(analysis, fh, indent=2, ensure_ascii=False, default=str)
            fh.write("\n")

        result = evaluate_paper(annotation, analysis, top_k=args.top_k)
        paper_results.append(result)
        print(
            f"  token-F1={result.mean_token_f1:.3f}  "
            f"hit-rate@{args.top_k}={result.hit_rate_at_k!s}  "
            f"citation-support={result.citation_support_rate!s}"
        )

    if not paper_results:
        print("\nNo papers were successfully evaluated.", file=sys.stderr)
        sys.exit(1)

    harness_result = HarnessResult(paper_results=paper_results, top_k=args.top_k)
    _write_report(output_dir, harness_result, timings)


if __name__ == "__main__":
    main()
