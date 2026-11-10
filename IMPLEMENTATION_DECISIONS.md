# Implementation Decisions and Rationale

This document captures key implementation decisions and their rationale. It is intended to help explain "why X was done" without relying on AI-generated documentation.

## Component-Wise Citation Grounding

**Problem:** The original implementation extracted a methodology claim like "A Transformer architecture using multi-head self-attention, positional encodings, and feed-forward layers" and cited chunks 15, 18, and 27. However, the positional-encoding evidence was actually on page 6, section 3.5, which was not among the cited chunks. The methodology chunk selector excluded page 6 because "Positional Encoding" was not among the heading patterns.

**Solution:**
1. Expanded methodology section patterns to include `r"positional encoding"`, `r"self.?attention"`, `r"multi.?head"`, and `r"feed.?forward"`
2. Added component-wise retrieval: for multi-part claims, each component is retrieved separately with `top_k=2`
3. Added conservative verbatim matching: a component is "supported" only if its normalized phrase appears verbatim in at least one retrieved chunk
4. Added three-way status classification: `supported`, `not_supported_by_retrieved_evidence`, `unverified`
5. Preserved separate mappings: `component_retrieved_chunks` (all) vs `component_evidence_map` (verified only)

**Why this approach:**
- Pattern expansion improves chunk selection without blindly increasing prompt size
- Component-wise retrieval ensures each technical concept gets dedicated evidence search
- Verbatim matching prevents false positives from single-word overlap
- Three-way status distinguishes "not found in retrieved evidence" from "retrieval failed"
- Separate mappings prevent cross-component contamination (e.g., "attention" evidence cannot support "memory")
- Human review is triggered for any unsupported/unverified component, never cleared from true to false

**Trade-offs:**
- Increased retrieval calls for multi-component claims (N components × top_k=2)
- Ambiguous component splits possible (e.g., "Smith, Jones, and Lee" splits into 3 components)
- Verbatim matching is conservative but may miss semantically equivalent phrasing
- Duplicate components/chunks are not deduplicated (cosmetic issue, low impact)

## Token-F1 vs Set-F1 for List Fields

**Problem:** The evaluation system needs to score list fields (authors, evaluation_metrics, limitations). Set-F1 on full sentences would rarely match unless verbatim, which is too harsh for free-text limitations.

**Solution:**
- `SET_LIST_FIELDS` (authors, evaluation_metrics): Use set precision/recall/F1 with normalization (lowercase, strip, punctuation → space)
- `TOKEN_LIST_FIELDS` (limitations): Use token-F1 macro-averaged over gold items vs best-matching predicted item

**Why this distinction:**
- Authors and metric names are short discrete items (1-2 words) that can be normalized and compared as sets
- Limitations are full sentences; set-F1 would require verbatim matches and penalize correct but differently worded extractions
- Token-F1 is a rough surface-form proxy for limitations, explicitly labeled as such

**Trade-offs:**
- Token-F1 does not measure semantic equivalence
- Semantically equivalent but differently worded limitations score below 1.0
- This is a known limitation, acknowledged in documentation

## Exact String Matching for Comparison

**Problem:** Comparing findings across papers requires determining whether statements match. LLM output rarely uses identical phrasing.

**Current Solution:** Exact, case/whitespace-normalized string matching

**Why exact matching (current):**
- Deterministic and reproducible across runs
- No external dependencies beyond Python
- Easy to understand and debug
- Sufficient for identifying identical values (e.g., same BLEU score, same dataset name)
- Conservative: never claims a match when one doesn't exist

**Limitations:**
- Free-text LLM output rarely uses identical phrasing
- Semantically equivalent statements are marked as distinct
- Different units or formats won't match
- Reordered list items won't match

**Future improvements (documented):**
- Embedding similarity for semantic equivalence
- LLM judge to assess whether statements convey the same meaning
- Hybrid approach: exact matching for numeric values, embedding similarity for prose
- Validation against human-labeled examples

## Citation Mechanism (Post-Hoc Retrieval)

**Problem:** How to assign citations to extracted claims when the LLM does not output chunk IDs or page numbers.

**Solution:** Citations are generated via semantic retrieval AFTER extraction, not from what the LLM actually used during generation. For each extracted field, the system performs a separate semantic search over all PDF chunks.

**Why post-hoc retrieval:**
- LLMs do not natively output chunk IDs or page numbers
- Retrieval-based citation assignment is a standard approach in RAG systems
- FAISS provides efficient similarity search
- Cosine similarity scores provide a ranking mechanism

**Limitations (explicitly documented):**
- Citations do not reflect what the LLM actually used during generation
- "Traceable" (page overlap + non-empty citation + non-insufficient sufficiency) ≠ "factually supported"
- Page overlap only confirms the citation points to the right region of the paper
- Human review is required to verify semantic support
- Verbatim matching checks phrase presence, not semantic correctness

## Core vs Auxiliary Components

**Rationale for classification:**

**Core (essential for basic PDF-to-analysis):**
- PDF extraction, chunking, embeddings, retrieval, paper analysis, LLM backends, QA, Streamlit UI
- These form the minimal viable system: upload PDF → extract text → analyze → view results

**Auxiliary (supporting features):**
- Job manager: Not used by Streamlit app; exists for background processing workflows
- Two search providers: Redundancy and comparison, but system works with just one
- Related paper extractor: Optional enhancement for deeper analysis
- Literature comparator/gap analyzer/report generator: Valuable but not required for single-paper understanding
- Validation/evaluation harnesses: Development/testing, not end-user features

**Why this distinction matters:**
- Helps focus on what's essential for the core workflow
- Makes it clear which components can be removed without breaking basic functionality
- Provides guidance for future refactoring or simplification

## Gold Set Evaluation Design

**Problem:** Need to measure extraction accuracy without claiming factual correctness.

**Solution:**
- Token-F1 for string fields (rough surface-form proxy)
- Set-F1 for short discrete list fields (authors, evaluation_metrics)
- Token-F1 macro-averaged for free-text sentence lists (limitations)
- Citation traceability rate (page overlap + non-empty citation + non-insufficient sufficiency)
- Hit rate @k and MRR for retrieval quality

**Why these metrics:**
- Token-F1 and set-F1 are standard NLP evaluation metrics
- They measure surface-form similarity, not factual correctness
- Citation traceability measures whether the system points to the right region of the paper
- Hit rate and MRR measure retrieval quality (are relevant chunks ranked highly?)
- All metrics are explicitly labeled as proxies or heuristics

**Limitations (explicitly documented):**
- Token-F1 and set-F1 do not measure factual correctness
- "Traceable" ≠ "factually supported"
- Currently only 1 annotated paper (not statistically significant)
- Requires more annotated papers and human review for meaningful evaluation

## Handling Missing `paper_analysis_errors` Module

**Problem:** After cloning/pushing to GitHub, the application failed with `ModuleNotFoundError: No module named 'paper_analysis_errors'`. The module was never committed to the repository.

**Solution:** Created `paper_analysis_errors.py` with two exception classes:
- `PaperAnalysisError`: Base exception for overall paper analysis failures
- `ExtractionError`: Exception for LLM extraction failures with optional `diagnostics` dict

**Why separate module:**
- Shared exceptions used across multiple modules (paper_analyzer, groq_backend, gemini_backend, related_paper_extractor)
- Centralizes exception definitions for consistency
- Allows optional `diagnostics` parameter for detailed error information

**Fix:**
- Added the missing module with proper exception classes
- Ensured backward compatibility with all existing imports
- Preserved existing `diagnostics` argument usage in `paper_analyzer.py`
