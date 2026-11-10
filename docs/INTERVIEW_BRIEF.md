# Interview Brief

## Project Pitch (3 sentences)

This is a research-paper intelligence system that accepts PDFs, extracts structured information using an LLM, discovers related literature through Crossref/OpenAlex, and generates comparison reports with evidence citations. The system uses bounded extraction (max 10 chunks, 12,000 characters) with page-aware chunking and semantic retrieval to provide traceable evidence for each extracted claim. Current evaluation on one paper shows extraction coverage (9/10 fields) and citation traceability (90%), but these are surface-form proxies and broader accuracy requires human review and a larger gold set.

---

## Core vs Auxiliary Components

### Core Components (8)
These are essential for the minimal viable system:

1. **PDF extraction** (`pdf_processing/text_extractor.py`) - PyMuPDF-based text extraction with page tracking
2. **Chunking** (`chunking/text_chunker.py`) - Page-aware text chunking with overlap for context
3. **Embeddings** (`embeddings/embedding_generator.py`) - sentence-transformers (all-MiniLM-L6-v2) for semantic search
4. **Retrieval** (`retrieval/semantic_retriever.py`) - FAISS-based semantic retrieval over chunks
5. **Paper analysis** (`analysis/paper_analyzer.py`) - Bounded LLM extraction (10 chunks, 12,000 chars) with section-aware heading patterns
6. **LLM backends** (`analysis/groq_backend.py`, `analysis/gemini_backend.py`) - Hosted LLM providers (Groq primary, Gemini secondary)
7. **QA** (`qa/answer_generator.py`) - Evidence-based question answering using retrieval
8. **Streamlit UI** (`app.py`) - Demonstration interface for uploading PDFs and viewing results

### Auxiliary Components (10)
These provide additional functionality but are not required for minimal analysis:

1. **Job manager** (`analysis/job_manager.py`, `analysis/job_store.py`) - SQLite-backed job management (exists as backend API but not integrated into Streamlit UI)
2. **Search providers** (`scholarly_search/crossref_client.py`, `scholarly_search/openalex_client.py`) - Two providers for related-paper discovery
3. **Related paper extractor** (`scholarly_search/related_paper_extractor.py`) - Downloads and analyzes related PDFs with safety checks
4. **Source verifier** (`scholarly_search/source_verifier.py`) - Metadata verification using Crossref/OpenAlex
5. **Literature comparator** (`analysis/literature_comparator.py`) - Exact-match comparison of extracted fields
6. **Research gap analyzer** (`analysis/research_gap_analyzer.py`) - Conservative gap identification from limitations
7. **Report generator** (`analysis/research_report_generator.py`) - Markdown report generation with citations
8. **Validation harness** (`validation/step5/run_validation.py`) - CLI tool for running validation experiments
9. **Evaluation harness** (`evaluation/run_harness.py`) - Gold-set evaluation with metrics
10. **CI workflow** (`.github/workflows/test.yml`) - Automated testing (Python 3.11/3.12)

---

## Major Design Choices

### Choice 1: MiniLM Embedding Model (all-MiniLM-L6-v2)

**Rationale:**
- Small (384-dimensional, ~80MB) - fast inference, low memory
- Good quality for semantic search benchmarks
- No GPU required - runs on CPU
- Widely used and well-maintained

**Trade-off:**
- Smaller model than state-of-the-art (e.g., E5-large, BGE-large)
- May miss some semantic nuances that larger models capture
- Acceptable for this use case where retrieval is a filter, not final verification

**Limitation:**
- Retrieval quality depends on embedding model; larger models may perform better but cost more

---

### Choice 2: Character Chunking with Overlap

**Rationale:**
- Page-aware chunks retain document structure (page numbers)
- Overlap (50 characters by default) ensures context is preserved across chunk boundaries
- Character-based (not token-based) avoids tokenization complexity
- Simple and deterministic

**Trade-off:**
- Fixed overlap may not be optimal for all content types
- May split sentences in awkward places
- Character count doesn't account for information density

**Limitation:**
- Chunk boundaries are arbitrary; some semantic units may be split across chunks
- Default chunk size is 500 characters (configurable in TextChunker.__init__)

---

### Choice 3: 10-Chunk / 12,000-Character Cap

**Rationale:**
- Bounded extraction limits LLM input cost and latency
- Section-aware heading patterns select relevant chunks (methodology, experiments, etc.)
- Prevents sending entire paper to LLM (cost and context window)
- Focuses extraction on high-value sections

**Trade-off:**
- May miss relevant information in unselected chunks
- Partial coverage is inherent (not full-paper analysis)
- Requires heading pattern tuning for different paper formats

**Limitation:**
- Extraction is partial by design; claims about "full coverage" would be incorrect

---

### Choice 4: Groq Model (openai/gpt-oss-20b)

**Rationale:**
- Open-source model via Groq API
- Fast inference (Groq's specialized hardware)
- Lower cost than proprietary models (GPT-4, Claude)
- Sufficient capability for structured extraction task

**Trade-off:**
- Smaller than state-of-the-art proprietary models
- May have lower quality on complex reasoning tasks
- Depends on Groq API availability and pricing

**Limitation:**
- Model quality affects extraction accuracy; larger models may perform better but cost more

---

### Choice 5: Post-Hoc Citation Retrieval

**Rationale:**
- Citations are generated after extraction by retrieving relevant chunks
- Separates extraction from evidence verification
- Allows component-wise citation support checks
- Bounded retrieval (top-k chunks) is fast and deterministic

**Trade-off:**
- Citations are NOT the passages the LLM actually used during extraction
- Cannot prove what the LLM used internally
- Verbatim checks are lexical, not semantic

**Limitation:**
- "Traceable citations" means the system points to the correct paper region with page overlap, but does NOT verify that the cited passage semantically supports the extracted claim. Human review is required for semantic support verification.

---

### Choice 6: Exact Match Comparison

**Rationale:**
- Deterministic and reproducible
- No additional dependencies (embedding models, LLMs)
- Conservative (minimizes false positives)
- Fast to compute

**Trade-off:**
- Misses semantically equivalent but differently worded content
- Misses paraphrases and synonyms
- Requires verbatim matches which are rare in free-text LLM output

**Limitation:**
- Comparison only finds lexical matches; semantically equivalent but differently worded content will not match. Future alternatives (embedding similarity, LLM judge) would require labeled pairs for validation.

---

### Choice 7: Title-Overlap Screening for Related Papers

**Rationale:**
- Deterministic overlap fraction between distinctive title tokens
- Fast and no external dependencies
- Explained as "exact keyword overlap, not semantic relevance"
- User-adjustable threshold

**Trade-off:**
- Not semantic relevance (misses synonyms, domain context)
- May miss relevant papers with different terminology
- May include irrelevant papers with shared title terms

**Limitation:**
- Title-term overlap is a heuristic, not a relevance score. Actual relevance requires semantic comparison validated against labeled pairs.

---

## Evaluation Metrics

### What the Metrics Mean

**Token-F1 (0.7587 on one paper):**
- Rough surface-form proxy for string fields
- Measures lexical overlap of tokens (words/numbers) after stopword removal
- Does NOT measure factual correctness or semantic equivalence
- Semantically equivalent but differently worded content scores below 1.0

**Set-F1 (0.7000 on one paper):**
- Rough surface-form proxy for short-item list fields (authors, evaluation_metrics)
- Measures item-level overlap after normalization (lowercase, trimmed, punctuation stripped)
- Does NOT measure semantic equivalence
- Used for authors and evaluation_metrics only (not limitations)

**Citation Traceability Rate (0.9000 on one paper):**
- Fraction of fields where citations point to the correct paper region
- Requires: (a) non-empty citation, (b) page overlap with annotated evidence_pages, (c) evidence marked "sufficient" or "partial"
- Does NOT verify that the cited passage semantically supports the extracted claim
- "Traceable" ≠ "factually supported"

**Hit Rate @k (0.7000 @1, 0.9000 @3, 0.9000 @5 on one paper):**
- Fraction of fields where at least one of the top-k retrieved chunks has page overlap with annotated evidence
- Measures retrieval precision, not semantic relevance
- Page overlap is a necessary but not sufficient condition for relevance

**MRR (0.8000 on one paper):**
- Mean Reciprocal Rank: average of 1/rank for first chunk with page overlap
- Measures retrieval ranking quality
- Higher is better (1.0 means first chunk always overlaps)

### Important Caveats

- **Small sample size:** All metrics are based on ONE paper (attention_is_all_you_need.pdf). Not statistically significant.
- **Surface-form proxies:** Token-F1 and Set-F1 measure lexical overlap, not factual correctness.
- **Not semantic verification:** Citation traceability and hit-rate measure page overlap, not semantic support.
- **Requires human review:** Metrics do not establish factual correctness; human review is required.

---

## Known Limitations

### Verified Limitations

1. **Partial coverage:** Extraction is bounded to 10 chunks and 12,000 characters. Not full-paper analysis.

2. **Post-hoc citations:** Citations are retrieved after extraction, not emitted by the LLM. Cannot prove what the LLM used.

3. **Verbatim checks:** Component-wise citation support uses verbatim matching. Does not establish semantic support.

4. **Exact comparison:** Literature comparison uses exact normalized string matching. Misses semantically equivalent content.

5. **Title-overlap heuristic:** Related-paper screening uses exact title-term overlap. Not semantic relevance.

6. **Single-paper evaluation:** Current gold set has only one annotated paper. Metrics are not representative.

7. **No human-reviewed results:** No extractions have been human-reviewed for factual correctness.

8. **Job manager not integrated:** SQLite job manager exists as backend API but is not used by Streamlit UI (synchronous flow).

9. **DNS re-resolution gap (mitigated, not eliminated):** The related-paper downloader validates URLs before connection and checks that the final address is in the originally validated set. However, urllib re-resolves DNS at connect time, and the intersection check runs after the request was already sent. This is a mitigation that catches many cases but does not implement true IP pinning (connecting to the validated IP while preserving hostname for SNI and certificate verification). The remaining gap is that a very fast DNS change during the connection window could theoretically bypass the check.

### UNVERIFIED Components

1. **Live related-paper LLM output:** Related-paper extraction runs LLM analysis on downloaded PDFs, but this output has not been human-reviewed for accuracy.

2. **Broader extraction accuracy:** Only one paper has been evaluated against gold annotations. Accuracy on other papers is unknown.

3. **Related-paper relevance quality:** Title-overlap screening has not been validated against human judgments of relevance.

4. **Gap analysis quality:** Research gap candidates are generated from limitations but have not been validated as genuine research gaps.

---

## Validation Artifacts

### What the Retained Runs Show

**Successful run (20261029_review_flag_check):**
- Paper: attention_is_all_you_need.pdf
- Model: Groq openai/gpt-oss-20b
- Runtime: 68.01 seconds
- Tokens: 1,803 prompt / 1,326 completion
- Coverage: 9/10 fields extracted
- Citations: 24 entries resolved
- Status: Pipeline completed and parsed successfully

**What this proves:**
- The pipeline executes without errors for this paper
- Structured extraction returns valid JSON
- Citations can be resolved to chunks

**What this does NOT prove:**
- Factual correctness of extracted values
- Semantic support of cited passages
- Generalization to other papers
- That all fields are accurate

### Historical Failures

**Failed run (20261028T113758773602Z):**
- All three bundled PDFs failed with HTTP 400
- Cause: Output cap was 768 tokens (insufficient for structured extraction)
- Fix: Raised cap to 2,048 tokens (DECISION-030)

---

## 10 Likely Interview Questions

### Q1: What is the system's core workflow?

**Answer:**
1. User uploads PDF
2. PyMuPDF extracts text with page tracking
3. Text is chunked page-by-page with 50-character overlap
4. Chunks are embedded using sentence-transformers (all-MiniLM-L6-v2)
5. For each section heading (methodology, experiments, etc.), up to 10 chunks are selected
6. LLM (Groq openai/gpt-oss-20b) extracts structured fields from selected chunks
7. Citations are generated by retrieving relevant chunks for each field
8. Results are displayed in Streamlit UI with evidence passages

---

### Q2: Why is extraction bounded to 10 chunks and 12,000 characters?

**Answer:**
To limit LLM input cost and latency while focusing on high-value sections. Section-aware heading patterns select relevant chunks (methodology, experiments, dataset, etc.) rather than sending the entire paper. This is a deliberate design choice for partial coverage, not a bug. The trade-off is that some relevant information in unselected chunks may be missed.

---

### Q3: How do citations work, and what do they prove?

**Answer:**
Citations are generated AFTER extraction by retrieving relevant chunks using semantic search. The system checks if retrieved chunks contain verbatim matches for extracted claims and records page overlap. This is "traceable" because the system points to the correct paper region, but it does NOT prove that the LLM actually used those passages during extraction, nor does it verify semantic support. Citations are post-hoc retrieval results, not model-emitted citations.

---

### Q4: Why use exact string matching for comparison instead of semantic similarity?

**Answer:**
Exact matching is deterministic, reproducible, and has no additional dependencies (no embedding models or LLMs needed). It's conservative by design (minimizes false positives). The trade-off is that it misses semantically equivalent but differently worded content. I documented alternatives (embedding similarity, LLM judge, hybrid) in docs/COMPARISON_OPTIONS.md, but any semantic comparison would require labeled matched/unmatched pairs for validation before deployment.

---

### Q5: What are the evaluation metrics, and what do they mean?

**Answer:**
- Token-F1 (0.7587): Lexical overlap proxy for string fields. Does not measure factual correctness.
- Set-F1 (0.7000): Item-level overlap for authors and evaluation_metrics. Does not measure semantic equivalence.
- Citation traceability (0.9000): Fraction of fields where citations point to the correct paper region. Does not verify semantic support.
- Hit-rate@k: Fraction of fields where top-k retrieved chunks have page overlap with evidence.
- MRR (0.8000): Mean reciprocal rank of first overlapping chunk.

**Important:** All metrics are based on ONE paper (attention_is_all_you_need.pdf) and are surface-form proxies. They do not establish factual correctness. Human review is required.

---

### Q6: What is the difference between core and auxiliary components?

**Answer:**
Core components (8) are essential for minimal viable analysis: PDF extraction, chunking, embeddings, retrieval, paper analysis, LLM backends, QA, and Streamlit UI. Auxiliary components (10) provide additional functionality: job manager, two search providers, related-paper extractor, source verifier, literature comparator, gap analyzer, report generator, validation harness, evaluation harness, and CI workflow. The Streamlit app currently uses a synchronous flow and does not integrate the SQLite job manager.

---

### Q7: What accuracy does the system achieve?

**Answer:**
I can only report what has been measured: on one paper (attention_is_all_you_need.pdf), Token-F1 is 0.7587, Set-F1 is 0.7000, and citation traceability is 0.9000. These are surface-form proxies based on a single annotated paper and are not representative of general system performance. Broader accuracy requires a larger gold set (at least 5 papers) and human review of extraction correctness. I do not claim broader accuracy beyond what has been measured.

---

### Q8: How does the system handle errors in optional stages like related-paper discovery?

**Answer:**
The implementation isolates failures: related-paper discovery, metadata verification, and related-paper analysis are each wrapped in try/except blocks that set status fields to "failed" but do NOT re-raise. The primary paper analysis completes successfully even if optional stages fail. This ensures that a network error in Crossref does not destroy the user's primary analysis results.

---

### Q9: What are the known security considerations for PDF downloads?

**Answer:**
The related-paper downloader validates URLs before connection: only public HTTPS URLs are allowed, DNS resolution checks for non-public addresses, redirect targets are validated against the original address set, and the final connected address is checked to ensure it's in the originally validated set. This is a mitigation that defends against many DNS re-resolution attacks and SSRF. However, urllib re-resolves DNS at connect time, and the intersection check runs after the request was already sent, so a very fast DNS change during the connection window could theoretically bypass the check. True IP pinning (connecting to the validated IP while preserving hostname for SNI and certificate verification) is not implemented. Additionally, byte limits, page limits, and character limits prevent resource exhaustion.

---

### Q10: What remains unverified in the system?

**Answer:**
- Live related-paper LLM output has not been human-reviewed for accuracy
- Broader extraction accuracy beyond the single evaluated paper is unknown
- Related-paper relevance quality (title-overlap heuristic) has not been validated against human judgments
- Gap analysis candidates have not been validated as genuine research gaps
- No extractions have been human-reviewed for factual correctness
- The job manager exists but is not integrated into the Streamlit UI

These are explicitly marked as UNVERIFIED in the documentation. I distinguish between what the code does (pipeline execution) and what is actually correct (requires human review).
