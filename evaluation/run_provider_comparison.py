"""Run controlled Crossref vs OpenAlex search comparison.

This script performs a real-world evaluation of both search providers using
the same paper, query inputs, and result limit. Results are saved to a timestamped
directory for manual relevance labeling and metric calculation.

Usage:
    python evaluation/run_provider_comparison.py --paper attention_is_all_you_need --max-results 10

After running:
1. Review the results in evaluation/results/<timestamp>/
2. Edit relevance_labels.json to mark top 5 results as relevant/irrelevant
3. Run: python evaluation/run_provider_comparison.py --calculate-metrics --results-dir <timestamp>
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scholarly_search.search_provider_factory import create_search_provider
from scholarly_search.evaluation import calculate_metrics


def extract_minimal_analysis(pdf_path: str) -> Dict[str, Any]:
    """Extract minimal paper analysis from PDF for search queries.
    
    Extracts only the fields needed for query generation (title, research_problem,
    methodology, dataset, evaluation_metrics). Does not perform full Groq analysis.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Dictionary with minimal analysis fields.
        
    Raises:
        SystemExit: If PDF extraction fails, with error message and non-zero exit status.
    """
    from pdf_processing.text_extractor import TextExtractor, PDFExtractionError
    
    extractor = TextExtractor()
    
    extraction_result = None
    try:
        extraction_result = extractor.extract_from_file(pdf_path)
    except PDFExtractionError as e:
        print(f"Error: PDF extraction failed: {e}")
        sys.exit(1)
    
    # Extract title from first page
    first_page_text = ""
    if extraction_result and extraction_result['pages']:
        first_page = extraction_result['pages'][0]
        first_page_text = first_page['text']
    
    # Extract title: skip permission headers and find first substantive line
    lines = first_page_text.strip().split('\n')
    
    # Common permission/copyright patterns to skip
    permission_patterns = [
        "provided proper attribution",
        "grants permission",
        "copyright",
        "©",
        "all rights reserved",
        "reproduce the tables",
        "solely for use in",
        "journalistic or",
        "scholarly works",
    ]
    
    # Find first non-empty line that doesn't look like a permission notice
    # Skip continuation lines after permission patterns
    skip_next = 0
    title = None
    for line in lines:
        if skip_next > 0:
            skip_next -= 1
            continue
        
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Check if line looks like a permission notice
        if any(pattern in line_stripped.lower() for pattern in permission_patterns):
            # Skip this line and the next 2 lines (continuation of permission)
            skip_next = 2
            continue
        
        title = line_stripped
        break
    
    # Fallback to filename if no title found
    if not title:
        title = os.path.basename(pdf_path).replace('.pdf', '')
    
    return {
        "title": title,
        "research_problem": None,  # Would come from Groq analysis
        "methodology": None,  # Would come from Groq analysis
        "dataset": None,  # Would come from Groq analysis
        "evaluation_metrics": None,  # Would come from Groq analysis
    }


def run_provider_search(
    provider_name: str,
    paper_analysis: Dict[str, Any],
    max_results: int,
) -> Dict[str, Any]:
    """Run search for a single provider.
    
    Args:
        provider_name: Name of the provider ('crossref' or 'openalex').
        paper_analysis: Paper analysis with search fields.
        max_results: Maximum results to return.
        
    Returns:
        Dictionary with search results and metadata.
    """
    provider = create_search_provider(provider_name)
    
    try:
        results = provider.search_related_papers(paper_analysis, max_results=max_results)
        
        return {
            "provider": provider_name,
            "success": True,
            "error": None,
            "results": results,
            "result_count": len(results),
            "requested_max_results": max_results,
        }
    except Exception as e:
        return {
            "provider": provider_name,
            "success": False,
            "error": str(e),
            "results": [],
            "result_count": 0,
            "requested_max_results": max_results,
        }


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    filename: str,
) -> None:
    """Save results to JSON file.
    
    Args:
        results: Results dictionary to save.
        output_dir: Output directory path.
        filename: Output filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def create_relevance_label_template(
    crossref_results: List[Dict[str, Any]],
    openalex_results: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """Create a template for manual relevance labeling.
    
    Args:
        crossref_results: Crossref search results.
        openalex_results: OpenAlex search results.
        output_dir: Output directory path.
    """
    template = {
        "crossref": {},
        "openalex": {},
        "instructions": "Mark DOIs as true (relevant), false (irrelevant), or null (uncertain). Only label top 5 results from each provider.",
    }
    
    # Add top 5 Crossref results
    for i, result in enumerate(crossref_results[:5]):
        doi = result.get("doi", f"no_doi_{i}")
        title = result.get("title", "Unknown")
        template["crossref"][doi] = {
            "title": title,
            "relevant": None,  # User will fill this
        }
    
    # Add top 5 OpenAlex results
    for i, result in enumerate(openalex_results[:5]):
        doi = result.get("doi", f"no_doi_{i}")
        title = result.get("title", "Unknown")
        template["openalex"][doi] = {
            "title": title,
            "relevant": None,  # User will fill this
        }
    
    save_results(template, output_dir, "relevance_labels_template.json")


def calculate_metrics_from_labels(
    results_dir: str,
) -> Dict[str, Any]:
    """Calculate metrics from manually labeled results.
    
    Args:
        results_dir: Directory containing results and labels.
        
    Returns:
        Dictionary with calculated metrics.
    """
    # Load results
    crossref_path = os.path.join(results_dir, "crossref_final.json")
    openalex_path = os.path.join(results_dir, "openalex_final.json")
    labels_path = os.path.join(results_dir, "relevance_labels.json")
    
    with open(crossref_path, 'r', encoding='utf-8') as f:
        crossref_data = json.load(f)
    with open(openalex_path, 'r', encoding='utf-8') as f:
        openalex_data = json.load(f)
    
    # Load labels
    try:
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
    except FileNotFoundError:
        return {
            "error": "relevance_labels.json not found. Please label results first.",
            "crossref": None,
            "openalex": None,
            "comparison": None,
        }
    
    # Extract relevant DOIs
    crossref_relevant = set()
    for doi, data in labels.get("crossref", {}).items():
        if data.get("relevant") is True:
            crossref_relevant.add(doi.lower())
    
    openalex_relevant = set()
    for doi, data in labels.get("openalex", {}).items():
        if data.get("relevant") is True:
            openalex_relevant.add(doi.lower())
    
    # Calculate metrics
    # Note: duplicate_rate is unavailable because providers don't expose raw candidates
    # We use original_candidate_count=0 to indicate this is unavailable
    crossref_metrics = calculate_metrics(
        crossref_data["results"],
        crossref_relevant,
        original_candidate_count=0,  # Unavailable
    )
    
    openalex_metrics = calculate_metrics(
        openalex_data["results"],
        openalex_relevant,
        original_candidate_count=0,  # Unavailable
    )
    
    # Prepare report
    report = {
        "crossref": crossref_metrics.to_dict(),
        "openalex": openalex_metrics.to_dict(),
        "notes": {
            "duplicate_rate": "Unavailable - providers do not expose raw candidates before deduplication",
            "precision_at_5": "Calculated from manual relevance labels",
            "metadata_completeness": "Calculated from result metadata",
            "topic_overlap": "Calculated from provider-assigned scores",
        },
        "comparison": {
            "better_precision": "unavailable" if len(crossref_relevant) == 0 and len(openalex_relevant) == 0 else "crossref" if crossref_metrics.precision_at_5 > openalex_metrics.precision_at_5 else "openalex" if openalex_metrics.precision_at_5 > crossref_metrics.precision_at_5 else "tie",
            "better_metadata": "crossref" if crossref_metrics.metadata_completeness > openalex_metrics.metadata_completeness else "openalex" if openalex_metrics.metadata_completeness > crossref_metrics.metadata_completeness else "tie",
            "higher_coverage": "unavailable" if crossref_metrics.provider_coverage == 0 or openalex_metrics.provider_coverage == 0 else "crossref" if crossref_metrics.provider_coverage > openalex_metrics.provider_coverage else "openalex" if openalex_metrics.provider_coverage > crossref_metrics.provider_coverage else "tie",
        }
    }
    
    # Mark duplicate_rate as unavailable
    report["crossref"]["duplicate_rate"] = None
    report["openalex"]["duplicate_rate"] = None
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run Crossref vs OpenAlex search comparison evaluation"
    )
    parser.add_argument(
        "--paper",
        default="attention_is_all_you_need",
        help="Paper filename (without .pdf extension)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum results to return from each provider"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: evaluation/results/TIMESTAMP)"
    )
    parser.add_argument(
        "--papers-dir",
        default="validation/step5/papers",
        help="Directory containing PDF papers"
    )
    parser.add_argument(
        "--calculate-metrics",
        action="store_true",
        help="Calculate metrics from labeled results instead of running new searches"
    )
    parser.add_argument(
        "--results-dir",
        help="Directory containing results for metric calculation (used with --calculate-metrics)"
    )
    
    args = parser.parse_args()
    
    if args.calculate_metrics:
        if not args.results_dir:
            print("Error: --results-dir required with --calculate-metrics")
            sys.exit(1)
        
        # Calculate metrics from existing results
        metrics = calculate_metrics_from_labels(args.results_dir)
        
        # Save metrics report
        output_path = os.path.join(args.results_dir, "metrics_report.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Metrics report saved to: {output_path}")
        
        # Check for error condition
        if metrics.get("error"):
            print(f"Error: {metrics['error']}")
            print("Please create relevance_labels.json from the template and label results first.")
            return
        
        # Print metrics only if available
        if metrics.get("crossref") and metrics.get("openalex"):
            print(f"Crossref precision@5: {metrics['crossref']['precision_at_5']}")
            print(f"OpenAlex precision@5: {metrics['openalex']['precision_at_5']}")
        else:
            print("Metrics not available - check error message above.")
        
        return
    
    # Run new searches
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join("evaluation", "results", f"comparison_{timestamp}")
    
    pdf_path = os.path.join(args.papers_dir, f"{args.paper}.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"Error: Paper not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Extracting minimal analysis from: {pdf_path}")
    paper_analysis = extract_minimal_analysis(pdf_path)
    print(f"Title: {paper_analysis['title']}")
    
    # Create output directory only after successful extraction
    os.makedirs(output_dir, exist_ok=True)
    
    # Run searches for both providers
    print(f"\nRunning Crossref search (max_results={args.max_results})...")
    crossref_results = run_provider_search("crossref", paper_analysis, args.max_results)
    
    print(f"Running OpenAlex search (max_results={args.max_results})...")
    openalex_results = run_provider_search("openalex", paper_analysis, args.max_results)
    
    # Save results
    save_results(crossref_results, output_dir, "crossref_final.json")
    save_results(openalex_results, output_dir, "openalex_final.json")
    
    # Save configuration
    config = {
        "timestamp": timestamp,
        "paper": args.paper,
        "pdf_path": pdf_path,
        "paper_analysis": paper_analysis,
        "max_results": args.max_results,
        "crossref": {
            "success": crossref_results["success"],
            "error": crossref_results["error"],
            "result_count": crossref_results["result_count"],
        },
        "openalex": {
            "success": openalex_results["success"],
            "error": openalex_results["error"],
            "result_count": openalex_results["result_count"],
        },
    }
    save_results(config, output_dir, "config.json")
    
    # Create relevance labeling template only if both providers succeeded
    if crossref_results["success"] and openalex_results["success"]:
        create_relevance_label_template(
            crossref_results["results"],
            openalex_results["results"],
            output_dir
        )
        print(f"\nRelevance labeling template created: {output_dir}/relevance_labels_template.json")
        print("Edit this file to mark top 5 results as relevant/irrelevant, then run:")
        print(f"  python evaluation/run_provider_comparison.py --calculate-metrics --results-dir {output_dir}")
    else:
        print(f"\nWarning: Relevance labeling template not created")
        if not crossref_results["success"]:
            print(f"  Crossref search failed: {crossref_results['error']}")
        if not openalex_results["success"]:
            print(f"  OpenAlex search failed: {openalex_results['error']}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Crossref: {crossref_results['result_count']} results")
    print(f"OpenAlex: {openalex_results['result_count']} results")


if __name__ == "__main__":
    main()
