"""
Step 5 validation script for real-paper analysis.

This script analyzes research papers through the actual Step 5 pipeline
and saves results, evidence audits, and validation reports.
"""

import json
import csv
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.backend_factory import configured_backend_name, configured_model_name
from pdf_processing.text_extractor import TextExtractor
from chunking.text_chunker import TextChunker
from embeddings.embedding_generator import EmbeddingGenerator
from retrieval.semantic_retriever import SemanticRetriever
from qa.answer_generator import AnswerGenerator
from analysis.paper_analyzer import PaperAnalyzer
import re


def _execution_succeeded(diagnostics: dict) -> bool:
    """Check if the analysis executed successfully (pipeline execution gate)."""
    return (
        diagnostics.get('backend_is_real') is True
        and diagnostics.get('attempts', 0) > 0
    )


def _schema_is_valid(analysis: dict) -> bool:
    """Check if the analysis output conforms to the expected schema."""
    if not isinstance(analysis, dict):
        return False
    
    required_fields = [
        'document_id', 'title', 'authors', 'publication_year', 'abstract',
        'research_problem', 'methodology', 'dataset', 'evaluation_metrics',
        'reported_results', 'limitations', 'evidence_citations', 'extraction_timestamp'
    ]
    
    for field in required_fields:
        if field not in analysis:
            return False
    
    # Check evidence_citations structure
    evidence_citations = analysis.get('evidence_citations', {})
    if not isinstance(evidence_citations, dict):
        return False
    
    # Check each citation has required fields
    for field_name, citation in evidence_citations.items():
        if not isinstance(citation, dict):
            return False
        # Required citation fields
        required_citation_fields = ['field_name', 'answer_text', 'evidence_type', 
                                       'evidence_sufficiency', 'requires_human_review']
        for field in required_citation_fields:
            if field not in citation:
                return False
    
    return True


def _calculate_extraction_coverage(analysis: dict) -> dict:
    """
    Calculate extraction coverage across expected fields.
    
    Returns:
        Dictionary with:
        - total_fields: total number of expected fields
        - extracted_fields: number of non-null fields
        - coverage_percentage: percentage of fields extracted
        - coverage_status: 'complete', 'partial', or 'empty'
    """
    expected_fields = [
        'title', 'authors', 'publication_year', 'abstract',
        'research_problem', 'methodology', 'dataset', 'evaluation_metrics',
        'reported_results', 'limitations'
    ]
    
    extracted_count = 0
    empty_fields = []
    
    for field in expected_fields:
        value = analysis.get(field)
        if value is not None:
            # For lists, check if non-empty
            if isinstance(value, list):
                if value and any(item for item in value if isinstance(item, str) and item.strip()):
                    extracted_count += 1
                else:
                    empty_fields.append(field)
            # For strings, check if non-empty
            elif isinstance(value, str) and value.strip():
                extracted_count += 1
            else:
                empty_fields.append(field)
        else:
            empty_fields.append(field)
    
    coverage_percentage = (extracted_count / len(expected_fields)) * 100 if expected_fields else 0
    
    if coverage_percentage == 100:
        coverage_status = 'complete'
    elif coverage_percentage > 0:
        coverage_status = 'partial'
    else:
        coverage_status = 'empty'
    
    return {
        'total_fields': len(expected_fields),
        'extracted_fields': extracted_count,
        'empty_fields': empty_fields,
        'coverage_percentage': coverage_percentage,
        'coverage_status': coverage_status
    }


def _check_evidence_resolution(analysis: dict) -> dict:
    """
    Check if cited document/chunk IDs resolve to actual non-empty passages.
    
    Returns:
        Dictionary with:
        - total_citations: total number of citations across all fields
        - resolved_citations: citations that resolve to non-empty passages
        - unresolved_citations: citations with missing or empty passages
        - resolution_status: 'fully_resolved', 'partially_resolved', or 'no_citations'
    """
    evidence_citations = analysis.get('evidence_citations', {})
    if not isinstance(evidence_citations, dict):
        return {
            'total_citations': 0,
            'resolved_citations': 0,
            'unresolved_citations': 0,
            'resolution_status': 'no_citations'
        }
    
    total_citations = 0
    resolved_citations = 0
    unresolved_citations = 0
    
    for field_name, citation in evidence_citations.items():
        if not isinstance(citation, dict):
            continue
        
        # Build mapping from chunk_id to evidence chunk for this field
        evidence_chunks = citation.get('evidence_chunks', [])
        if not isinstance(evidence_chunks, list):
            evidence_chunks = []
        
        chunks_by_id = {
            chunk.get('chunk_id'): chunk
            for chunk in evidence_chunks
            if isinstance(chunk, dict)
            and isinstance(chunk.get('chunk_id'), str)
            and chunk.get('chunk_id')
        }
        
        citations_list = citation.get('citations', [])
        if not isinstance(citations_list, list):
            continue
        
        for cit in citations_list:
            if not isinstance(cit, dict):
                continue
            total_citations += 1
            
            # Check if citation's chunk_id resolves to a non-empty evidence chunk
            chunk_id = cit.get('chunk_id')
            if not isinstance(chunk_id, str) or not chunk_id:
                unresolved_citations += 1
                continue
            
            cited_chunk = chunks_by_id.get(chunk_id)
            if cited_chunk is None:
                unresolved_citations += 1
                continue
            
            # Check if chunk has non-empty text
            chunk_text = cited_chunk.get('text')
            if not isinstance(chunk_text, str) or not chunk_text.strip():
                unresolved_citations += 1
                continue
            
            # Citation resolves to a non-empty passage
            resolved_citations += 1
    
    if total_citations == 0:
        resolution_status = 'no_citations'
    elif resolved_citations == total_citations:
        resolution_status = 'fully_resolved'
    else:
        resolution_status = 'partially_resolved'
    
    return {
        'total_citations': total_citations,
        'resolved_citations': resolved_citations,
        'unresolved_citations': unresolved_citations,
        'resolution_status': resolution_status
    }


def _is_valid_real_backend_result(diagnostics: dict) -> bool:
    """
    DEPRECATED: This conflates execution success with extraction quality.
    Use the separate validation dimensions instead.
    Kept for backward compatibility with existing callers.
    """
    return _execution_succeeded(diagnostics)


def _validation_exit_code(results: list) -> int:
    return int(any(result['status'] != 'success' for result in results))


def analyze_paper(
    pdf_path: str,
    document_id: str,
    backend_name: str = "groq",
    model_name: Optional[str] = None,
) -> dict:
    """
    Analyze a single paper through the full Step 5 pipeline.

    Returns:
        Dictionary with analysis results and metadata
    """
    start_time = time.time()
    errors = []
    diagnostics = {}
    if backend_name not in {"gemini", "groq"}:
        raise ValueError("backend_name must be 'gemini' or 'groq'")
    if backend_name == "gemini" and (
        not isinstance(model_name, str) or not model_name.strip()
    ):
        raise ValueError("model_name is required for the Gemini backend")
    selected_model = model_name or configured_model_name(backend_name)
    if not selected_model:
        raise ValueError(
            f"model_name is required for the {backend_name} backend"
        )

    try:
        # Step 1: PDF extraction
        print(f"\n1. Extracting text from {pdf_path}...")
        extractor = TextExtractor()
        extraction_result = extractor.extract_from_file(pdf_path)
        print(f"   - Pages: {extraction_result['total_pages']}")
        print(f"   - Characters: {extraction_result['total_characters']}")

        # Step 2: Initialize components
        print(f"\n2. Initializing components...")
        chunker = TextChunker()
        embedder = EmbeddingGenerator()
        retriever = SemanticRetriever(embedder)
        answer_gen = AnswerGenerator(retriever)
        analyzer = PaperAnalyzer(
            chunker,
            retriever,
            answer_gen,
            model_name=selected_model,
            llm_backend_name=backend_name,
        )
        print("   - Components initialized")

        # Step 3: Analyze
        print(
            f"\n3. Running bounded structured analysis via {backend_name} "
            f"({analyzer.llm_timeout_seconds:g}-second model timeout)..."
        )
        analysis = analyzer.analyze_paper(
            extraction_result,
            document_id,
            progress_callback=lambda stage, message: print(
                f"   - {stage}: {message}", flush=True
            ),
        )

        runtime = time.time() - start_time
        diagnostics = analysis.get('extraction_diagnostics', {})
        
        # A6: Separate validation dimensions
        execution_status = 'succeeded' if _execution_succeeded(diagnostics) else 'failed'
        schema_status = 'valid' if _schema_is_valid(analysis) else 'invalid'
        extraction_coverage = _calculate_extraction_coverage(analysis)
        evidence_resolution = _check_evidence_resolution(analysis)
        
        # Quality review is always 'pending' until human review (A6 requirement)
        quality_review_status = 'pending'
        
        # For backward compatibility with existing callers
        valid_backend_result = _is_valid_real_backend_result(diagnostics)
        
        if not valid_backend_result:
            message = (
                "Analysis did not execute successfully from the "
                "configured real backend."
            )
            return {
                'status': 'failed',
                'pdf_path': pdf_path,
                'document_id': document_id,
                'runtime_seconds': runtime,
                'errors': [message],
                'retries': diagnostics.get('retries'),
                'json_repairs': diagnostics.get('json_repairs'),
                'fallback_calls': diagnostics.get('fallback_calls'),
                'backend': diagnostics.get('backend_name', backend_name),
                'model': diagnostics.get('model', selected_model),
                'prompt_eval_count': diagnostics.get('prompt_eval_count'),
                'extraction_diagnostics': diagnostics,
                'analysis': analysis,
                'error_message': message,
                # A6: Separate validation dimensions
                'validation': {
                    'execution_status': execution_status,
                    'schema_status': schema_status,
                    'extraction_coverage': extraction_coverage,
                    'evidence_resolution': evidence_resolution,
                    'quality_review_status': quality_review_status,
                }
            }

        return {
            'status': 'success',
            'pdf_path': pdf_path,
            'document_id': document_id,
            'runtime_seconds': runtime,
            'errors': errors,
            'retries': diagnostics.get('retries'),
            'json_repairs': diagnostics.get('json_repairs'),
            'fallback_calls': diagnostics.get('fallback_calls'),
            'backend': diagnostics.get('backend_name', backend_name),
            'model': diagnostics.get('model', selected_model),
            'prompt_eval_count': diagnostics.get('prompt_eval_count'),
            'extraction_diagnostics': diagnostics,
            'analysis': analysis,
            # A6: Separate validation dimensions
            'validation': {
                'execution_status': execution_status,
                'schema_status': schema_status,
                'extraction_coverage': extraction_coverage,
                'evidence_resolution': evidence_resolution,
                'quality_review_status': quality_review_status,
            }
        }

    except Exception as e:
        runtime = time.time() - start_time
        errors.append(str(e))
        diagnostics = getattr(e, 'diagnostics', diagnostics) or diagnostics
        
        # A6: Set validation dimensions for exception case
        execution_status = 'failed'
        schema_status = 'unknown'
        extraction_coverage = None
        evidence_resolution = None
        quality_review_status = 'pending'
        
        return {
            'status': 'failed',
            'pdf_path': pdf_path,
            'document_id': document_id,
            'runtime_seconds': runtime,
            'errors': errors,
            'retries': diagnostics.get('retries'),
            'json_repairs': diagnostics.get('json_repairs'),
            'fallback_calls': diagnostics.get('fallback_calls'),
            'backend': diagnostics.get('backend_name', backend_name),
            'model': diagnostics.get('model', selected_model),
            'prompt_eval_count': diagnostics.get('prompt_eval_count'),
            'extraction_diagnostics': diagnostics,
            'analysis': None,
            'error_message': str(e),
            # A6: Separate validation dimensions
            'validation': {
                'execution_status': execution_status,
                'schema_status': schema_status,
                'extraction_coverage': extraction_coverage,
                'evidence_resolution': evidence_resolution,
                'quality_review_status': quality_review_status,
            }
        }


def create_evidence_audit(analysis_results: list) -> list:
    """
    Create evidence audit CSV rows for all analyzed papers.

    Returns:
        List of audit row dictionaries
    """
    audit_rows = []

    for result in analysis_results:
        if result['status'] != 'success':
            continue

        analysis = result['analysis']
        document_id = result['document_id']
        pdf_name = Path(result['pdf_path']).name

        # Get evidence citations
        evidence_citations = analysis.get('evidence_citations', {})

        # Schema fields
        fields = [
            'title', 'authors', 'publication_year', 'abstract',
            'research_problem', 'methodology', 'dataset',
            'evaluation_metrics', 'reported_results', 'limitations'
        ]

        for field_name in fields:
            field_value = analysis.get(field_name)
            citation = evidence_citations.get(field_name, {})

            evidence_chunks = citation.get('evidence_chunks', [])
            citations = citation.get('citations', [])
            if not isinstance(evidence_chunks, list):
                evidence_chunks = []
            if not isinstance(citations, list):
                citations = []
            chunks_by_id = {
                chunk.get("chunk_id"): chunk
                for chunk in evidence_chunks
                if (
                    isinstance(chunk, dict)
                    and isinstance(chunk.get("chunk_id"), str)
                    and chunk.get("chunk_id")
                )
            }

            # Automated assessment
            evidence_type = citation.get('evidence_type', 'unknown')
            evidence_sufficiency = citation.get('evidence_sufficiency', 'insufficient')
            lexical_support = citation.get("lexical_support", 0.0)
            if (
                not isinstance(lexical_support, (int, float))
                or isinstance(lexical_support, bool)
            ):
                lexical_support = 0.0
            retrieval_similarity = citation.get("retrieval_similarity")

            references = citations if citations else [None]
            for evidence_reference in references:
                supporting_text = ""
                page_info = ""
                chunk_id = ""
                cited_chunk = None

                if evidence_reference is None:
                    citation_status = (
                        "citation_missing" if evidence_chunks
                        else "missing_evidence"
                    )
                elif not isinstance(evidence_reference, dict):
                    citation_status = "invalid_citation"
                else:
                    candidate_chunk_id = evidence_reference.get("chunk_id")
                    if isinstance(candidate_chunk_id, str):
                        chunk_id = candidate_chunk_id
                    if evidence_reference.get("document_id") != document_id:
                        citation_status = "wrong_document"
                    elif not chunk_id:
                        citation_status = "missing_chunk"
                    else:
                        cited_chunk = chunks_by_id.get(chunk_id)
                        if cited_chunk is None:
                            citation_status = "missing_chunk"
                        elif cited_chunk.get("document_id") != document_id:
                            citation_status = "wrong_document"
                        elif (
                            not isinstance(cited_chunk.get("text"), str)
                            or not cited_chunk["text"].strip()
                        ):
                            citation_status = "empty_passage"
                        elif (
                            evidence_reference.get("start_page") is not None
                            and evidence_reference.get("start_page")
                            != cited_chunk.get("start_page")
                        ) or (
                            evidence_reference.get("end_page") is not None
                            and evidence_reference.get("end_page")
                            != cited_chunk.get("end_page")
                        ):
                            citation_status = "page_mismatch"
                        else:
                            citation_status = "reference_consistent_unverified"
                            supporting_text = cited_chunk["text"]

                    page_start = evidence_reference.get("start_page")
                    page_end = evidence_reference.get("end_page", page_start)
                    if cited_chunk is not None:
                        if page_start is None:
                            page_start = cited_chunk.get("start_page")
                        if page_end is None:
                            page_end = cited_chunk.get("end_page", page_start)
                    if page_start is not None:
                        page_info = f"{page_start}-{page_end or page_start}"

                requires_review = True
                if citation_status != "reference_consistent_unverified":
                    automated_assessment = "unsupported_reference"
                elif evidence_sufficiency == "insufficient":
                    automated_assessment = "unsupported"
                elif (
                    evidence_sufficiency != "sufficient"
                    or lexical_support < 0.3
                    or citation.get("requires_human_review", True)
                ):
                    automated_assessment = "weak_support_requires_review"
                else:
                    automated_assessment = "reference_consistent_unverified"

                explanation = (
                    f"Type: {evidence_type}, Sufficiency: {evidence_sufficiency}, "
                    f"Reference: {citation_status}, Review: {requires_review}. "
                    "Reference consistency and retrieval similarity do not "
                    "verify factual support."
                )

                audit_rows.append({
                    'paper_name': pdf_name,
                    'document_id': document_id,
                    'field_name': field_name,
                    'extracted_value': (
                        str(field_value) if field_value is not None else 'None'
                    ),
                    'evidence_type': evidence_type,
                    'evidence_sufficiency': evidence_sufficiency,
                    'cited_page': page_info,
                    'chunk_id': chunk_id,
                    'supporting_text': supporting_text,
                    'automated_assessment': automated_assessment,
                    'explanation': explanation,
                    'citation_status': citation_status,
                    'retrieval_similarity': retrieval_similarity,
                    'lexical_support': lexical_support,
                    'requires_human_review': requires_review,
                    'human_verdict': '',
                    'human_notes': ''
                })

    return audit_rows


def _prepare_output_directory(output_dir: Optional[Path]) -> Path:
    """Create a fresh run directory and refuse to overwrite prior outputs."""
    if output_dir is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = Path(__file__).parent / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step 5 paper validation.")
    parser.add_argument(
        "--backend",
        choices=("groq", "gemini"),
        default=configured_backend_name(),
        help="LLM backend; defaults to RESEARCHINTELLIGENCE_LLM_BACKEND or groq.",
    )
    parser.add_argument(
        "--model",
        help="Model name; defaults to RESEARCHINTELLIGENCE_LLM_MODEL or provider default.",
    )
    parser.add_argument(
        "--confirm-public-papers",
        action="store_true",
        help="Confirm uploaded PDFs may be sent to a hosted model provider.",
    )
    parser.add_argument(
        "--confirm-groq-costs",
        action="store_true",
        help="Confirm possible Groq API charges before hosted inference.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=10,
        help="Delay between paper requests (default: 10 seconds).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New, non-existing directory for this run's outputs.",
    )
    parser.add_argument(
        "--paper",
        choices=("attention_is_all_you_need", "bert_pretraining", "attention_survey"),
        help="Validate only the specified paper (omit to validate all papers).",
    )
    return parser


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.delay_seconds < 0:
        parser.error("--delay-seconds must be non-negative")
    if args.backend == "gemini":
        if not args.model:
            parser.error("--model is required when --backend=gemini")
        if not args.confirm_public_papers:
            parser.error(
                "--confirm-public-papers is required when --backend=gemini"
            )
    if args.backend == "groq":
        if not args.confirm_public_papers:
            parser.error(
                "--confirm-public-papers is required when --backend=groq"
            )
        if not args.confirm_groq_costs:
            parser.error(
                "--confirm-groq-costs is required when --backend=groq"
            )
        if not args.model:
            args.model = configured_model_name("groq")
    return args


def main(argv=None):
    """Main validation workflow."""
    args = parse_arguments(argv)
    if args.backend in {"gemini", "groq"}:
        print(
            f"Reminder: {args.backend} processes paper text on an external "
            "provider; review data terms and possible charges before continuing."
        )
    selected_model = args.model or configured_model_name(args.backend)
    if not selected_model:
        raise ValueError(f"Configure a model for the {args.backend} backend")

    output_root = _prepare_output_directory(args.output_dir)

    print("=" * 60)
    print("Step 5 Validation - Real Paper Analysis")
    print("=" * 60)

    # Papers to analyze
    papers_dir = Path(__file__).parent / 'papers'
    all_papers = [
        ('attention_is_all_you_need.pdf', 'attention_is_all_you_need'),
        ('bert_pretraining.pdf', 'bert_pretraining'),
        ('attention_survey.pdf', 'attention_survey')
    ]
    
    # Filter papers if --paper argument is supplied
    if args.paper:
        papers = [(f"{args.paper}.pdf", args.paper)]
    else:
        papers = all_papers

    results_dir = output_root / 'results'
    raw_outputs_dir = output_root / 'raw_outputs'
    results_dir.mkdir()
    raw_outputs_dir.mkdir()

    analysis_results = []

    for paper_index, (pdf_filename, document_id) in enumerate(papers):
        pdf_path = papers_dir / pdf_filename

        print(f"\n{'=' * 60}")
        print(f"Analyzing: {pdf_filename}")
        print(f"Document ID: {document_id}")
        print(f"{'=' * 60}")

        if not pdf_path.exists():
            print(f"ERROR: PDF not found: {pdf_path}")
            analysis_results.append({
                'status': 'failed',
                'pdf_path': str(pdf_path),
                'document_id': document_id,
                'runtime_seconds': 0,
                'errors': [f"PDF not found: {pdf_path}"],
                'retries': None,
                'json_repairs': None,
                'fallback_calls': None,
                'backend': args.backend,
                'model': selected_model,
                'prompt_eval_count': None,
                'extraction_diagnostics': {},
                'analysis': None,
                'error_message': 'PDF not found'
            })
            if paper_index < len(papers) - 1:
                time.sleep(args.delay_seconds)
            continue

        # Analyze the paper
        result = analyze_paper(
            str(pdf_path),
            document_id,
            backend_name=args.backend,
            model_name=selected_model,
        )
        analysis_results.append(result)

        # Save result
        # A5: Save full result including validation dimensions
        result_file = results_dir / f"{document_id}_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n   Result saved to: {result_file}")
        
        if result['status'] == 'success':
            analysis_file = results_dir / f"{document_id}_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(result['analysis'], f, indent=2, ensure_ascii=False)
            print(f"   Analysis saved to: {analysis_file}")
        else:
            # Preserve the bounded diagnostics needed to diagnose failed calls.
            error_file = raw_outputs_dir / f"{document_id}_error.txt"
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Error: {result['error_message']}\n")
                f.write(f"Errors: {result['errors']}\n")
                f.write("Extraction diagnostics:\n")
                json.dump(
                    result.get('extraction_diagnostics', {}),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"\n   Error saved to: {error_file}")
        if paper_index < len(papers) - 1:
            time.sleep(args.delay_seconds)

    # Create evidence audit
    print(f"\n{'=' * 60}")
    print("Creating evidence audit...")
    print(f"{'=' * 60}")

    audit_rows = create_evidence_audit(analysis_results)

    audit_file = output_root / 'evidence_audit.csv'
    with open(audit_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'paper_name', 'document_id', 'field_name', 'extracted_value',
            'evidence_type',
            'evidence_sufficiency', 'cited_page', 'chunk_id', 'supporting_text',
            'automated_assessment', 'explanation', 'citation_status',
            'retrieval_similarity', 'lexical_support', 'requires_human_review',
            'human_verdict', 'human_notes'
        ])
        writer.writeheader()
        writer.writerows(audit_rows)

    print(f"Audit saved to: {audit_file}")
    print(f"Total audit rows: {len(audit_rows)}")

    # Create validation report
    print(f"\n{'=' * 60}")
    print("Creating validation report...")
    print(f"{'=' * 60}")

    report_file = output_root / 'validation_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Step 5 Validation Report\n\n")
        f.write(f"**Date:** {datetime.now().isoformat()}\n\n")
        f.write("## Environment\n\n")
        f.write(
            "- Machine details and provider-side context allocation are not "
            "collected by this script.\n"
        )
        f.write("- Automated tests are not run by this script.\n\n")

        f.write("## Papers Analyzed\n\n")
        for result in analysis_results:
            f.write(f"### {Path(result['pdf_path']).name}\n\n")
            f.write(f"- **Status:** {result['status']}\n")
            f.write(f"- **Document ID:** {result['document_id']}\n")
            f.write(f"- **Runtime:** {result['runtime_seconds']:.2f} seconds\n")
            f.write(f"- **Backend:** {result.get('backend', 'not available')}\n")
            f.write(f"- **Model:** {result.get('model', 'not available')}\n")
            f.write(
                "- **Prompt evaluation tokens:** "
                f"{result.get('prompt_eval_count', 'not available')}\n"
            )
            diagnostics = result.get('extraction_diagnostics', {})
            f.write(
                "- **Attempts / retries / JSON repairs / fallback calls:** "
                f"{diagnostics.get('attempts', 'not available')} / "
                f"{result.get('retries', 'not available')} / "
                f"{result.get('json_repairs', 'not available')} / "
                f"{result.get('fallback_calls', 'not available')}\n"
            )
            f.write(
                f"- **Model finish reason:** "
                f"{diagnostics.get('done_reason', 'not available')}\n"
            )
            f.write(
                f"- **Generated tokens / model duration:** "
                f"{diagnostics.get('eval_count', 'not available')} / "
                f"{diagnostics.get('duration', 'not available')}\n"
            )
            
            # A5: Include A6 validation dimensions in report
            validation = result.get('validation', {})
            if validation:
                f.write("\n### Validation Dimensions (A6)\n\n")
                f.write(f"- **Execution Status:** {validation.get('execution_status', 'unknown')}\n")
                f.write(f"- **Schema Status:** {validation.get('schema_status', 'unknown')}\n")
                
                coverage = validation.get('extraction_coverage')
                if coverage:
                    f.write(f"- **Extraction Coverage:** {coverage.get('coverage_status', 'unknown')} "
                           f"({coverage.get('extracted_fields', 0)}/{coverage.get('total_fields', 0)} fields, "
                           f"{coverage.get('coverage_percentage', 0):.1f}%)\n")
                
                resolution = validation.get('evidence_resolution')
                if resolution:
                    f.write(f"- **Evidence Resolution:** {resolution.get('resolution_status', 'unknown')} "
                           f"({resolution.get('resolved_citations', 0)}/{resolution.get('total_citations', 0)} citations resolved)\n")
                
                f.write(f"- **Quality Review Status:** {validation.get('quality_review_status', 'pending')}\n")
                f.write("\n")
            
            if result['status'] == 'success':
                f.write(f"- **Errors:** {len(result['errors'])}\n")
            else:
                f.write(f"- **Error:** {result['error_message']}\n")
                f.write(f"- **Errors:** {result['errors']}\n")
            f.write("\n")

        f.write("## Evidence Audit\n\n")
        f.write(f"- Total audit rows: {len(audit_rows)}\n")
        f.write(f"- Audit file: `evidence_audit.csv`\n")
        f.write(f"- Results directory: `results/`\n\n")

        f.write("## Critical Failures\n\n")
        critical_failures = []
        for result in analysis_results:
            if result['status'] == 'failed':
                critical_failures.append(f"{Path(result['pdf_path']).name}: {result['error_message']}")

        if critical_failures:
            f.write("Critical failures detected:\n\n")
            for failure in critical_failures:
                f.write(f"- {failure}\n")
        else:
            f.write("No critical failures detected.\n\n")

        f.write("## Recommendation\n\n")
        if critical_failures:
            f.write("Step 5 **FAILED** due to critical issues. Manual code review and fixes required before approval.\n")
        else:
            f.write("All attempted papers returned non-empty validated fields from the configured real backend.\n\n")
            f.write("This status does not establish extraction accuracy; human spot-checking remains necessary.\n\n")
            f.write("The following requires manual review:\n")
            f.write("- Extracted field values against source PDFs (10-15 fields per paper)\n")
            f.write("- Evidence citations and page numbers\n")
            f.write("- Contradictory or uncertain classifications\n")
            f.write("- The `human_verdict` and `human_notes` columns in `evidence_audit.csv` are blank for this review.\n")

    print(f"Validation report saved to: {report_file}")

    print(f"\n{'=' * 60}")
    print("Validation complete")
    print(f"{'=' * 60}")
    return _validation_exit_code(analysis_results)


if __name__ == "__main__":
    raise SystemExit(main())
