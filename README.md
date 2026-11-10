# AI Research Intelligence System

A system for understanding research papers and discovering related literature.

## Project Objective

Build an AI Research Intelligence System that helps users understand a research paper and investigate related research. The current workflow analyzes uploaded PDFs, can discover related scholarly papers, compares available findings, surfaces cautious follow-up candidates, and generates a traceable Markdown report. A simple Streamlit demonstration interface is included.

## Current Status

**Development Phase:** Groq-backed research workflow and Streamlit demonstration (PDF-to-report pipeline implemented; human citation review remains pending)

**Implemented Functionality:**
- PDF text extraction using PyMuPDF
- Page-by-page text extraction with page number preservation
- Error handling for missing files, invalid PDFs, and PDFs with no extractable text
- Text chunking with configurable chunk size and overlap
- Metadata preservation for each chunk (chunk_id, document_id, chunk_index, page numbers, character positions)
- Page traceability across chunk boundaries
- Text embeddings using sentence-transformers/all-MiniLM-L6-v2 (384-dimensional)
- Semantic retrieval using FAISS with cosine similarity
- Top-k ranked results with chunk metadata
- Structured paper analysis with evidence traceability
- Related-paper discovery through Crossref or OpenAlex using extracted title and research-topic metadata
- Metadata verification against Crossref and OpenAlex
- Bounded open-access PDF retrieval, extraction, and per-paper evidence analysis
- Section-aware, character-bounded LLM input with a configurable response timeout
- Analysis-stage progress callbacks for UI integration
- SQLite-backed background analysis jobs with retained inputs/checkpoints and process-restart recovery
- Deterministic cross-paper comparison of findings, methods, datasets, metrics, and limitations with per-paper citations
- Groq-backed structured paper extraction and evidence-based question answering
- Conservative follow-up candidates derived only from extracted limitations, plus missing-evidence coverage warnings
- Streamlit results organized into a quick summary, evidence review, related-paper screening, analyzed-paper comparison, follow-up candidates, and report download
- Related-paper screening by exact title-term overlap, with near-duplicate title/author records merged while retaining alternate DOI identifiers
- Citation review that displays retrieved source passages and checks reference/page consistency and numeric presence without claiming factual verification
- Structured Markdown reports containing retrieved passages, related-paper topic-match notes, and explicit coverage/review warnings
- Automated tests for extraction, chunking, embeddings, retrieval, paper analysis, search, verification, PDF acquisition, analysis jobs, reports, and Groq integration

See `DECISION.md` for detailed development roadmap and decision log.

## Technology Stack

- Python for application logic
- PyMuPDF for PDF text extraction
- Streamlit for the demonstration interface
- OpenAlex for scholarly paper discovery
- Semantic Scholar as an optional complementary source
- Crossref for public scholarly work search and DOI metadata
- Groq API using the official Python SDK; requests may incur charges
- Sentence Transformers and FAISS for semantic retrieval
- SQLite for background job state
- pytest for automated testing
- Git for version control

## Key Principles

- Step-by-step development: Only implement what is explicitly requested
- No fabrication: Never hardcode fake papers, citations, or evaluation metrics
- Traceable citations: All claims must be supported by actual sources
- Distinguish evidence types: Explicit statements vs. synthesized conclusions vs. potential gaps
- Honest limitations: Never claim a research gap is novel without verification
- Open-source preference: Use free and open-source tools wherever practical

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Groq configuration

Groq is the configured default; local Ollama support has been removed. Install
the listed project dependencies and set `GROQ_API_KEY` in your local
operating-system environment. The key is read when a generation request is
made. Do not paste it into source code, chat, logs, or reports. The project
does not automatically load `.env` files.

The configured Groq model defaults to `openai/gpt-oss-20b`; override it with
`RESEARCHINTELLIGENCE_LLM_MODEL` or the validation script's `--model` option.
This model was selected from provider documentation describing strict
JSON-schema output. A synthetic request using this model and schema succeeded
in the configured environment; this does not verify extraction quality.

The validation harness requires explicit external-data and possible-cost
confirmations for Groq:

```powershell
python validation\step5\run_validation.py --backend groq --confirm-public-papers --confirm-groq-costs
```

The harness processes three bundled research papers, sends selected paper
excerpts (up to 10 chunks and 12,000 characters) to the provider, waits ten
seconds between papers by default, and writes each run to a new directory.
Review the provider's current data terms, model availability, rate limits, and
pricing before running it. Provider pricing and account terms may change. A synthetic
Groq health check succeeded, but it does not validate extraction quality or
real-paper behavior.

**Latest real-paper validation:** Single-paper validation (attention_is_all_you_need.pdf) completed the Step 5 analyzer with `openai/gpt-oss-20b` after the output cap was raised to 2,048 tokens. The earlier 768-token cap failed on the Transformer paper; the successful request generated 1,133 completion tokens. Measured full-analyzer result: Transformer, 1,803 prompt / 1,326 completion tokens, 9 non-empty fields, 24 citation entries, 68.01 seconds. Validation artifacts are retained in `validation/step5/runs/20261029_review_flag_check/`. This result confirms pipeline completion and parsing for one paper, not factual accuracy or citation support. Human review and full three-paper validation remain necessary.

For a one-request connectivity check with a synthetic prompt and no paper
content, run:

```powershell
python test_qa_smoke.py --confirm-costs
```

### Scholarly search provider configuration

The system supports two scholarly search providers for related-paper discovery:

- **Crossref** (default): Free public scholarly work search
- **OpenAlex**: Free public scholarly work search with open access coverage

To select a provider, set the `RESEARCHINTELLIGENCE_SEARCH_PROVIDER` environment variable:

```powershell
# Use Crossref (default)
set RESEARCHINTELLIGENCE_SEARCH_PROVIDER=crossref

# Use OpenAlex
set RESEARCHINTELLIGENCE_SEARCH_PROVIDER=openalex
```

For OpenAlex, you can optionally set an API key for 10× the daily budget:

```powershell
set OPENALEX_API_KEY=your_api_key_here
```

To get an OpenAlex API key:
1. Visit https://openalex.org/settings/api
2. Enter your email address
3. Click the link sent to your email to activate the key

Both providers return results in the same schema with the following fields:
- `title`, `authors`, `year`, `venue`, `doi`, `url`, `abstract`
- `relevance_score`, `topic_overlap_score`, `topic_overlap_terms`
- `source` (identifies which provider returned the result)

### Streamlit demonstration

Start the user interface with:

```powershell
streamlit run app.py
```

The interface requires a separate checkbox confirmation before sending an
uploaded paper's selected excerpts to Groq. Crossref discovery is optional and
requires network access. Processing related-paper PDFs is separately
confirmable and limited to three eligible papers. The uploaded PDF is held in
a temporary file during analysis and removed afterward; displayed results
remain in the Streamlit session. The first embedding initialization may
download the existing sentence-transformer model.

The results page separates metadata-only search results from papers that were
actually analyzed and used in comparison. Related-paper screening uses a
configurable exact keyword-overlap threshold; this is a filtering heuristic,
not a relevance probability. The evidence tab and report show retrieved
passages so users can compare them with each extracted claim. Automated
reference/numeric checks do not prove that a passage supports a claim; source
PDF review remains necessary.

`RESEARCHINTELLIGENCE_LLM_BACKEND` defaults to `groq`. Gemini remains an
optional hosted backend. `RESEARCHINTELLIGENCE_LLM_MODEL` can override the
Groq model or specify the Gemini model.

### Running Tests

Run all automated tests:

```bash
pytest tests/ -v
```

Run only PDF extraction tests:

```bash
pytest tests/test_pdf_processing/test_text_extractor.py -v
```

Run only chunking tests:

```bash
pytest tests/test_chunking/test_text_chunker.py -v
```

### Optional Live Scholarly API Tests

The project includes optional integration tests that make real API requests to Crossref and OpenAlex to verify scholarly search provider connectivity. These tests are located in `tests/test_integration/test_live_scholarly_search.py` and are **skipped by default** to avoid network dependencies and provider rate limits.

To enable and run these live API tests on Windows PowerShell:

```powershell
$env:RUN_LIVE_API_TESTS="1"
python -m pytest tests/test_integration/test_live_scholarly_search.py -v
```

**Warning:** These tests make real external API requests and may be affected by network availability, provider rate limits, or API changes. Do not enable them unless you want to verify actual Crossref/OpenAlex connectivity. These tests are optional and not required for basic project setup or usage.

### Example Usage

Run the example script to test PDF extraction interactively:

```bash
python example_usage.py
```

The script will prompt you to enter a path to a PDF file and will display the extracted text.

### Programmatic Usage

#### PDF Extraction

```python
from pdf_processing import TextExtractor

# Initialize the extractor
extractor = TextExtractor()

# Extract text from a PDF file
result = extractor.extract_from_file("path/to/your/file.pdf")

# Access extraction results
print(f"Total pages: {result['total_pages']}")
print(f"Pages with text: {result['pages_with_text']}")
print(f"Total characters: {result['total_characters']}")

# Get text from a specific page
page_1_text = extractor.get_text_by_page(result, 1)

# Get all text as a single string with page markers
full_text = extractor.get_text_as_string(result)
```

#### Text Chunking

```python
from chunking import TextChunker

# Initialize the chunker (configurable chunk size and overlap)
chunker = TextChunker(chunk_size=500, chunk_overlap=50)

# Chunk extraction result from PDF extraction
chunking_result = chunker.chunk_extraction_result(extraction_result, document_id="my_paper")

# Access chunking results
print(f"Total chunks: {chunking_result['total_chunks']}")
print(f"Document ID: {chunking_result['document_id']}")

# Iterate through chunks
for chunk in chunking_result['chunks']:
    print(f"Chunk {chunk['chunk_index']}:")
    print(f"  Pages {chunk['start_page']}-{chunk['end_page']}")
    print(f"  Characters {chunk['start_char']}-{chunk['end_char']}")
    print(f"  Text: {chunk['text'][:100]}...")  # First 100 chars

# Chunk simple text (not from PDF)
# Note: chunk_simple_text does not include page metadata
simple_text = "This is some text that needs to be chunked..."
chunking_result = chunker.chunk_simple_text(simple_text)

# Character offsets: start_char is inclusive, end_char is exclusive
# original_text[start_char:end_char] reconstructs the chunk text
```

#### Semantic Retrieval

```python
from embeddings import EmbeddingGenerator
from retrieval import SemanticRetriever

# Initialize embedding generator
generator = EmbeddingGenerator()

# Initialize semantic retriever
retriever = SemanticRetriever(generator, embedding_dimension=384)

# Build index from chunks (after chunking)
chunks = chunking_result['chunks']
retriever.build_index(chunks)

# Search for relevant chunks
query = "What are the research limitations?"
results = retriever.search(query, top_k=5)

# Access results
for result in results:
    print(f"Rank {result['rank']}:")
    print(f"  Similarity: {result['similarity']:.4f}")
    print(f"  Chunk ID: {result['chunk']['chunk_id']}")
    print(f"  Document ID: {result['chunk']['document_id']}")
    print(f"  Pages: {result['chunk']['start_page']}-{result['chunk']['end_page']}")
    print(f"  Text: {result['chunk']['text'][:100]}...")

# Results include all original chunk metadata for citation
```

## Known Limitations (Step 3)

- PDFs with no extractable text (potentially scanned or image-based) will raise `NoExtractableTextError`. OCR is not implemented.
- Empty PDFs (zero pages) cannot be programmatically created with PyMuPDF for testing; this test is skipped. The error handling code is implemented but requires a pre-made empty PDF file for automated testing.
- No OCR functionality for scanned PDFs.
- No support for password-protected PDFs.
- Test file locking: Windows occasionally locks test files during cleanup; the teardown method handles this gracefully with retry logic.
- Chunking uses character-based boundaries, which may split words (verified in tests). This is acceptable for Step 2 as word-aware chunking can be added later if needed.
- Page boundaries in chunks are determined by character position; this is accurate but not semantic (doesn't detect paragraphs or sections).
- Character positions are relative to concatenated text; `text[start_char:end_char]` reconstructs the chunk, but start_char/end_char are not absolute page positions (verified in tests).
- `chunk_simple_text()` does not include page metadata (start_page, end_page) because no page information is available (verified in tests).
- Embedding model is English-optimized; may have reduced quality for other languages.
- FAISS index is in-memory only; lost when process exits (acceptable for Step 3).
- No disk persistence for embeddings or index.
- CPU-based inference; slower than GPU but sufficient for Step 3 scale.
- Model download required on first run (~22MB for all-MiniLM-L6-v2).
- **Temporary file cleanup:** Uploaded PDFs are written to temporary files during analysis and cleaned up in a `finally` block. If the process crashes, is forcefully terminated, or the system shuts down before the cleanup executes, temporary files may remain on disk. The temporary files are created in the system's default temporary directory (managed by Python's `tempfile` module). The operating system may periodically clean this directory, but this is not guaranteed. Leftover temporary files do not affect system functionality but may consume disk space if crashes occur frequently.

## Documentation

- `DECISION.md` - Complete decision log and development roadmap
- `README.md` - This file (project overview)
- `pdf_processing/text_extractor.py` - PDF extraction module with docstrings
- `example_usage.py` - Interactive example script

## Related-paper search

After constructing `PaperAnalyzer`, use the combined workflow to analyze an
uploaded PDF and search for related works:

```python
analysis = analyzer.analyze_paper_with_related_papers(
    extraction_result,
    document_id="uploaded-paper",
    max_results=10,
)
related_papers = analysis["related_papers"]
```

The search uses Crossref's public works API and does not require an API key.
It searches by extracted title and by bounded research-problem, methodology,
dataset, and evaluation-metric terms. Only these metadata-derived terms are
sent to Crossref; the PDF text and full abstract are not sent. Search is explicit
through this method, so the existing `analyze_paper()` workflow remains offline.
Results include title, authors, year, venue, DOI/link, abstract when available,
the Crossref relevance score, and the query that found each record.

Crossref coverage and metadata completeness vary. Results are discovery leads,
not evidence that a paper supports a claim or is a perfect match. Network,
provider, and malformed-response errors are surfaced as `ScholarlySearchError`.

## Metadata and source verification

Check a discovered work against Crossref and OpenAlex registry records:

```python
for paper in related_papers:
    verification = analyzer.verify_paper_metadata(paper)
    print(verification["status"])
    print(verification["field_checks"])
    print(verification["open_access"])
```

Verification compares supplied title, DOI, year, and first author with available
Crossref metadata and checks the matched DOI against OpenAlex when available.
Statuses distinguish `verified`, `partial`, `mismatch`, `not_found`, and
`unavailable`. Provider failures are included explicitly in
`verification["provider_errors"]`; if one registry is unavailable, the other
registry's results are retained. The verifier rejects OpenAlex links when the
returned DOI does not match the DOI being checked.
Open-access status and PDF/landing-page links are provider-reported; link
accessibility is not tested, and the verifier never fetches a URL supplied by a
search result. A registry match does not prove that the paper's claims are true,
that it was peer reviewed, or that its full text is legally or technically
accessible. Network and malformed-response errors are surfaced explicitly.

## Related-paper evidence extraction

After discovery, process a bounded number of selected candidates:

```python
evidence_results = analyzer.extract_related_paper_evidence(
    related_papers,
    max_papers=3,
)
for result in evidence_results:
    print(result["status"], result.get("reason"))
```

Only records with a matching verified OpenAlex DOI, a provider-reported
open-access flag, and a provider-listed direct PDF URL are downloaded. The
downloader requires public HTTPS hosts, validates redirect targets, checks PDF
content, and enforces byte, page, character, candidate-count, and timeout
limits. Each paper uses an isolated retriever so its analysis citations retain
that paper's chunk IDs and page numbers. Results have explicit statuses such as
`analyzed`, `skipped`, `download_failed`, `extraction_failed`, and
`analysis_failed`; papers without direct PDF links remain metadata-only. HTML
landing pages, OCR, and arbitrary URL scraping are not used.

## Multi-paper literature comparison

Compare the uploaded paper's analysis with Step 8 result records:

```python
comparison = analyzer.compare_papers(
    [uploaded_analysis, *evidence_results],
)

print(comparison["status"], comparison["source_count"])
print(comparison["field_comparisons"]["reported_results"])
```

The comparator accepts direct `PaperAnalyzer.analyze_paper()` results and
Step 8 paper records. It includes analyzed papers only, excludes duplicates
by DOI or document ID, and reports skipped/failed candidates in
`excluded_papers`. It compares research problem, methodology, dataset,
evaluation metrics, reported results, and limitations. Results include
source-specific values and compact evidence references (chunk IDs and page
numbers); missing, incomplete, or mismatched citation references are flagged
and require human review.

Comparison is deterministic and uses exact, case/whitespace-normalized values:
it can identify exact matches and shared list entries, but does not infer
semantic equivalence. Different result statements are labelled as distinct,
not as contradictions. Two or more successfully analyzed sources are required
for a `completed` comparison; otherwise the result status is
`insufficient_sources`. At most 20 candidates may be supplied. This is a
bounded comparison of the available records, not an exhaustive literature
review, truth assessment, or research-gap/novelty claim.

## Responsive paper analysis

Structured extraction uses at most 10 selected chunks and 12,000 paper-text
characters. `analysis_coverage` records selected and omitted chunk identifiers,
page ranges, and whether the character limit truncated input. Selection
prioritizes detected abstract, methods, results, limitations, and conclusion
headings; it is a bounded heuristic, not a guarantee that every relevant
passage is selected.
Missing schema keys are kept as `null` rather than triggering separate QA
requests, preserving the single-call bound and avoiding unsupported recovery.
Groq is asked for schema-constrained JSON output capped at 2,048 generated
tokens. The cap was raised after a real-paper response needed 1,133 completion
tokens and failed with the earlier 768-token limit. The request timeout defaults
to 90 seconds and can be configured up to 30 minutes when constructing
`PaperAnalyzer`. A timeout raises an actionable `ExtractionError`; it is not
returned as a successful or partial analysis.

Evidence citations are assembled after extraction from semantic retrieval, not
from passages the model is known to have used. Similarity and
`evidence_sufficiency` are retrieval heuristics, not correctness probabilities;
the legacy `confidence` field is a deprecated compatibility alias for
`retrieval_similarity`.

An optional callback lets an application display workflow stages:

```python
analysis = analyzer.analyze_paper(
    extraction_result,
    document_id="uploaded-paper",
    progress_callback=lambda stage, message: print(stage, message),
)
```

Stages include `chunking`, `extracting`, `validating`, `indexing`, `complete`,
and `failed` for extraction failures. The callback reports stage transitions;
it does not provide a percentage or stream model tokens. This direct call is
synchronous; applications that need a responsive interface can submit work
through the SQLite-backed job service:

```python
from analysis import AnalysisJobManager, AnalysisJobStore

store = AnalysisJobStore()  # Windows: %LOCALAPPDATA%\ResearchIntelligence
manager = AnalysisJobManager(
    store,
    analyzer_factory=create_paper_analyzer,
)

job = manager.submit(extraction_result, document_id="uploaded-paper")
job_id = job["job_id"]  # Return this immediately to the UI.
current = manager.get_job(job_id)  # Poll status, stage, and message.
if current and current["status"] == "completed":
    analysis = current["result"]

# Call only after the user explicitly chooses to delete a non-running job.
manager.delete(job_id)
```

The manager runs one analysis at a time in a background worker, persists each
job's extracted page text and status, checkpoints the structured LLM output,
and stores successful results. Normal job status/list responses omit extracted
paper text and internal checkpoints; source text is only read by the worker.
On application restart, jobs owned by a process
that is no longer running are requeued; a saved LLM checkpoint avoids repeating
inference if interruption occurred after extraction completed. A job interrupted
during a hosted model request must repeat that request because partial
generations are not saved. Failed jobs can be retried with `manager.retry`
and completed/failed jobs can be deleted explicitly with `manager.delete`;
running jobs cannot be deleted. Inference and JSON inputs have finite limits,
and provider/model errors are visible in the persisted failed status.
The Step 5 validation harness also saves bounded extraction diagnostics for
failed model calls, including any available finish reason and token counts.

The local database retains source text, status, checkpoints, and results until
explicit deletion. Deletion removes the job row but is not a guaranteed secure
erase of SQLite/WAL storage. The database is not encrypted by the application;
protect the local account/device and database file accordingly. An extraction
payload larger than 25 MiB is rejected instead of being silently truncated. The job service
provides a backend API for polling. The included Streamlit page currently runs
analysis synchronously and displays stage messages; it does not yet poll the
background job API or stream tokens.

## License

*To be determined*

## Development Steps

1. ✅ Project Foundation and PDF Text Extraction (completed)
2. ✅ Document Chunking and Metadata (completed)
3. ✅ Embeddings and Semantic Retrieval (completed)
4. ✅ Evidence-Based Question Answering (completed)
5. ✅ Structured Research Paper Analysis (completed)
6. ✅ Scholarly Search and Related Paper Discovery (completed)
7. ✅ Paper Metadata and Source Verification (completed)
8. ✅ Evidence Extraction from Related Papers (implementation complete; live LLM output unverified)
9. ✅ Multi-Paper Literature Comparison (implemented)
10. ✅ Conservative limitation-based follow-up candidates (not a novelty search)
11. ✅ Structured Markdown report generation
12. ✅ Basic Streamlit PDF-analysis interface
13. Testing, Evaluation, and Robustness
14. Finalization and Resume Readiness

See `DECISION.md` for detailed breakdown of each step.
