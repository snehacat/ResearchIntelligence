#!/usr/bin/env python3
"""
Metrics report script for evaluation harness (Task 3).

This script takes filled gold files and a validation run directory,
then prints a metrics report with n_papers and n_fields, plus a visible
warning when n_papers < 5. All metrics are labeled as surface-form proxies.

Usage:
    python evaluation/report_metrics.py --gold-dir evaluation/gold_set/ --run-dir validation/step5/runs/20261029_review_flag_check/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from evaluation.gold_set_schema import load_annotation, validate_annotation
from evaluation.metrics import evaluate_paper, HarnessResult


def load_gold_annotations(gold_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all gold annotations from a directory."""
    gold_dir = Path(gold_dir)
    if not gold_dir.exists():
        print(f"Error: Gold directory not found: {gold_dir}")
        sys.exit(1)

    annotations = {}
    for json_file in gold_dir.glob("*.json"):
        try:
            annotation = load_annotation(json_file)
            document_id = annotation["document_id"]
            annotations[document_id] = annotation
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return annotations


def load_validation_results(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all analysis results from a validation run directory."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    results_dir = run_dir / "results"
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    results = {}
    for result_file in results_dir.glob("*_result.json"):
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            document_id = result.get("document_id")
            if document_id:
                results[document_id] = result
        except Exception as e:
            print(f"Warning: Failed to load {result_file}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate metrics report from gold annotations and validation run results."
    )
    parser.add_argument(
        "--gold-dir",
        required=True,
        help="Directory containing gold annotation JSON files",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Validation run directory containing results/",
    )
    args = parser.parse_args()

    # Load gold annotations
    gold_annotations = load_gold_annotations(args.gold_dir)
    n_papers = len(gold_annotations)

    # Load validation results
    validation_results = load_validation_results(args.run_dir)

    # Count total fields
    n_fields = 0
    for annotation in gold_annotations.values():
        n_fields += len(annotation.get("fields", {}))

    # Print header
    print("=" * 70)
    print("EVALUATION METRICS REPORT")
    print("=" * 70)
    print()

    # Print sample size warning
    print(f"n_papers: {n_papers}")
    print(f"n_fields: {n_fields}")
    print()

    if n_papers < 5:
        print("=" * 70)
        print("WARNING: Small sample size")
        print("=" * 70)
        print(f"Only {n_papers} paper(s) in the evaluation set. Metrics may not be")
        print("representative of general system performance. Expand the gold set")
        print("to at least 5 papers before making broad accuracy claims.")
        print()

    # Evaluate each paper that has both gold and results
    paper_results = []
    for document_id, gold in gold_annotations.items():
        if document_id not in validation_results:
            print(f"Warning: No validation result found for {document_id}")
            continue

        analysis = validation_results[document_id]
        try:
            paper_result = evaluate_paper(
                gold=gold,
                analysis=analysis,
                document_id=document_id,
                k_values=(1, 3, 5),
            )
            paper_results.append(paper_result)
        except Exception as e:
            print(f"Warning: Failed to evaluate {document_id}: {e}")

    if not paper_results:
        print("Error: No papers could be evaluated (missing or mismatched results)")
        sys.exit(1)

    # Aggregate results
    harness_result = HarnessResult(paper_results=paper_results)

    # Print metrics
    print("=" * 70)
    print("EXTRACTION ACCURACY (SURFACE-FORM PROXIES)")
    print("=" * 70)
    print(f"Macro Token-F1: {harness_result.macro_token_f1:.4f}")
    print(f"  Note: {harness_result.footnotes['token_f1']}")
    print()

    if harness_result.macro_set_f1 is not None:
        print(f"Macro Set-F1: {harness_result.macro_set_f1:.4f}")
        print(f"  Note: {harness_result.footnotes['set_f1']}")
        print()

    print("=" * 70)
    print("RETRIEVAL METRICS (SURFACE-FORM PROXIES)")
    print("=" * 70)
    for k in (1, 3, 5):
        print(f"Hit Rate @{k}: {harness_result.macro_hit_rate(k):.4f}")
    print(f"MRR: {harness_result.macro_mrr:.4f}")
    print()

    print("=" * 70)
    print("CITATION TRACEABILITY (SURFACE-FORM PROXY)")
    print("=" * 70)
    print(f"Citation Traceability Rate: {harness_result.citation_traceability_rate:.4f}")
    print(f"  Note: {harness_result.footnotes['citation_traceability_rate']}")
    print()

    print("=" * 70)
    print("IMPORTANT DISCLAIMERS")
    print("=" * 70)
    print("All metrics above are SURFACE-FORM PROXIES:")
    print("- Token-F1 and Set-F1 measure lexical overlap, not factual correctness.")
    print("- Semantically equivalent but differently worded content scores below 1.0.")
    print("- Hit-rate@k and MRR measure page overlap, not semantic relevance.")
    print("- Citation traceability means the system points to the correct paper")
    print("  region; it does NOT verify that the cited passage semantically supports")
    print("  the extracted claim.")
    print("- Human review is required to establish factual correctness.")
    print()


if __name__ == "__main__":
    main()
