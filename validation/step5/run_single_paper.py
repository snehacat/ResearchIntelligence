"""
Test single paper analysis to measure runtime.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pdf_processing.text_extractor import TextExtractor
from chunking.text_chunker import TextChunker
from embeddings.embedding_generator import EmbeddingGenerator
from retrieval.semantic_retriever import SemanticRetriever
from qa.answer_generator import AnswerGenerator
from analysis.paper_analyzer import PaperAnalyzer, ExtractionError

pdf_path = "validation/step5/papers/attention_is_all_you_need.pdf"
document_id = "attention_is_all_you_need"

print("=" * 60)
print("Single Paper Test - attention_is_all_you_need.pdf")
print("=" * 60)

start_time = time.time()

# Step 1: PDF extraction
print(f"\n1. Extracting text...")
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
analyzer = PaperAnalyzer(chunker, retriever, answer_gen)
print("   - Components initialized")

# Step 3: Analyze
print("\n3. Running bounded structured analysis (90-second model timeout)...")
analysis_start = time.time()
try:
    analysis = analyzer.analyze_paper(
        extraction_result,
        document_id,
        progress_callback=lambda stage, message: print(
            f"   - {stage}: {message}", flush=True
        ),
    )
except ExtractionError as error:
    print(f"\nAnalysis failed: {error}")
    raise SystemExit(1) from error
analysis_time = time.time() - analysis_start

total_time = time.time() - start_time

print(f"\n{'=' * 60}")
print(f"Analysis Complete")
print(f"{'=' * 60}")
print(f"Analysis time: {analysis_time:.2f} seconds")
print(f"Total time: {total_time:.2f} seconds")
print(f"\nExtracted title: {analysis.get('title', 'N/A')}")
print(f"Extracted authors: {analysis.get('authors', 'N/A')}")
print(f"Extracted year: {analysis.get('publication_year', 'N/A')}")
citations = analysis.get("evidence_citations", {})
print(f"Evidence citation fields: {len(citations)}")
print(f"Fields with citations: {sum(bool(value) for value in citations.values())}")
