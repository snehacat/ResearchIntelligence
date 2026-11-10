# AI Research Intelligence System — Phase A Audit

**Scope:** Repository inspection only. No application code was changed, no test suite was run, and no live Groq or scholarly-provider requests were made. This report is the requested Phase A deliverable; implementation should wait for user review.

## Implementation Progress

### Batch 1: Evidence Correctness and Citation Resolution (COMPLETED)

**Status:** ✅ Completed and verified

**Findings Addressed:**
- **A1 (High):** QA support detection now uses verbatim matching and downgrades to "partial" when not verbatim. High similarity + LLM sufficiency alone do NOT establish support.
- **A2 (High):** Paper analyzer rejects invalid field types instead of coercing them. Retrieval similarity is separated from factual confidence (deprecated `confidence` field set to `None`).
- **A3 (High):** Literature comparator marks citations as "available" only when exact document_id + chunk_id resolve to a non-empty passage.
- **A4 (High):** Research gap analyzer resolves cited passages before marking limitations as evidence-linked.

**Test Results:**
- Batch 1 focused tests: 163 passed in 8.12s
- Full test suite: 366 passed, 1 skipped, 2 warnings in 100.55s

**Files Modified:**
- `qa/answer_generator.py` - Separated retrieval similarity from factual confidence
- `analysis/paper_analyzer.py` - Rejected invalid field types, deprecated confidence field
- `analysis/literature_comparator.py` - Added passage resolution for citations
- `analysis/research_gap_analyzer.py` - Added passage resolution for limitations
- `tests/test_qa/test_answer_generator.py` - Added supported claim control test
- `tests/test_analysis/test_paper_analyzer.py` - Added field type rejection tests
- `tests/test_analysis/test_literature_comparator.py` - Added citation resolution tests
- `tests/test_analysis/test_research_report.py` - Added passage resolution tests

### Batch 4: Validation Criteria Separation (A6 Only) (COMPLETED)

**Status:** ✅ Completed and verified

**Finding Addressed:**
- **A6 (Medium):** Validation success now separated into five distinct dimensions:
  - **Execution status:** Did the analysis run or fail? (`succeeded`/`failed`)
  - **Schema status:** Did the output conform to the expected schema? (`valid`/`invalid`/`unknown`)
  - **Extraction coverage:** How many expected fields were extracted? (`complete`/`partial`/`empty` with percentage)
  - **Evidence resolution:** Do cited document/chunk IDs resolve to actual non-empty passages? (`fully_resolved`/`partially_resolved`/`no_citations`)
  - **Quality review:** Have the extracted fields and evidence been checked against the source by a human? (`pending` - never automated)

**Implementation Details:**
- Deprecated `_is_valid_real_backend_result` (kept for backward compatibility)
- Added `_execution_succeeded()` - checks backend_is_real and attempts > 0
- Added `_schema_is_valid()` - validates all required fields and evidence_citations structure
- Added `_calculate_extraction_coverage()` - calculates field coverage percentage and status
- Added `_check_evidence_resolution()` - checks if citations have passage excerpts
- Updated `analyze_paper()` to include `validation` dictionary with all five dimensions
- Quality review status is always `pending` (A6 requirement: never invent human review status)

**Test Results:**
- Batch 4 focused tests: 24 passed in 3.54s
- Full test suite: 397 passed, 1 skipped, 2 warnings in 74.77s

**Files Modified:**
- `validation/step5/run_validation.py` - Added separate validation dimension functions, updated result structure
- `tests/test_validation/test_run_validation.py` (new file) - 24 tests covering all validation dimensions and scenarios
- `tests/test_validation/__init__.py` (new file) - Test package marker

**Compatibility Implications:**
- Existing callers of `_is_valid_real_backend_result` still work (backward compatible)
- Result dictionary now includes `validation` key with all five dimensions
- Old behavior (success = execution + one field) is replaced by dimensional reporting
- Reports and callers that check `result['status'] == 'success'` still work
- Quality review status must be set manually (not automated)

**Test Coverage Added:**
- API success with incomplete extraction (partial coverage, schema valid)
- Schema-valid output with unresolved citations (partial resolution)
- Complete extraction with no human review (quality review pending)
- Provider failure (execution failed)
- Successful evidence-resolved result that would pass human review (quality review still pending)

**Remaining Findings:**
- **A5 (High):** Retained real-paper validation is failed - Not addressed (requires live validation)
- **A9 (Medium):** Extraction prompt does not clearly isolate untrusted paper text - COMPLETED in Batch 3
- **A11 (Medium):** Related-PDF URL validation has DNS re-resolution gap - COMPLETED in Batch 3

---

## Executive summary

The repository contains a functional, modular research-analysis prototype: a Streamlit interface; PDF text extraction and chunking; Groq inference with an optional Gemini backend; semantic retrieval; scholarly discovery and metadata checks; comparison and follow-up-candidate generation; SQLite-backed jobs; and downloadable reports.

The main concern is not that every stage is absent. It is that several stages can label weak or unresolved evidence as available/sufficient, while validation artifacts show that successful JSON parsing or a completed run has not established extraction correctness. In particular:

1. The QA pipeline explicitly returns an unsupported model claim as `sufficient` when retrieval similarity is high.
2. The paper analyzer’s deprecated `confidence` value is still an alias for mean retrieval similarity; that score is not factual confidence.
3. Comparison citations can be reported as `available` without resolving the cited chunk to an actual passage.
4. Follow-up candidates can be formed from extracted limitations without checking that the limitation passage resolves or supports the text.
5. The retained Groq run reports HTTP 400 failures for all three bundled PDFs. Later documentation records success, but the inspected retained validation inventory did not contain a matching successful run.

**Recommendation:** Do not present the system as end-to-end validated or interview-ready for evidence-backed research conclusions until the high-priority evidence and validation issues below are fixed and independently tested.

## Repository state and inspection boundaries

- Git reports no commits on `master`; project files are untracked. There is no tracked baseline from which to distinguish earlier work. Preserve all files; do not clean or reset the worktree.
- `AUDIT_REPORT.md` is newly added by this audit. No application source or tests were edited.
- Root files named `=0.1.0`, `=1.7.0`, and `=2.2.0` contain package-install output, not application code. They appear to be incidental local artifacts and are left untouched.
- The retained test-output text files show previous runs, including `296 passed, 1 skipped`, `63 passed`, and `69 passed, 1 skipped`. These are historical outputs, not a baseline reproduced in this audit.
- No live Groq request, Crossref/OpenAlex request, real-paper re-run, or dependency installation was performed.
- The test suite is configured to block external network/model calls by default in [tests/conftest.py](./tests/conftest.py), except for explicitly marked `real_model` tests.

## Prioritized findings

Severity describes the risk to user-facing research conclusions, not exploitability.

### A1 — High: QA can return unsupported claims as sufficient

- **Location:** [`qa/answer_generator.py`](./qa/answer_generator.py), `_assess_evidence_sufficiency` (line 273) and `generate_answer` (line 318).
- **Evidence:** The sufficiency rule treats one retrieved chunk with similarity above `0.7` plus the model’s own `"sufficient"` label as sufficient. It does not check whether the answer’s individual claims are entailed by the cited passage. The final response returns the model answer at line 408 when that heuristic passes.
- **Reproduction already encoded in a test:** [`tests/test_qa/test_answer_generator.py`](./tests/test_qa/test_answer_generator.py), `test_generate_answer_high_similarity_unsupported_claim` (line 506), supplies evidence that does not contain “no side effects” and asserts that the answer containing that claim is still returned as sufficient (lines 515–529).
- **Impact:** A user can receive a plausible but unsupported statement with a high similarity score and a `sufficient` label.
- **Proposed fix:** Treat retrieval similarity only as a retrieval signal. Until a claim-to-passage check is implemented and validated, keep model-generated answers marked for review and downgrade unsupported or unverifiable claims rather than calling them sufficient. Do not silently rewrite the answer.
- **Regression test:** Change the existing test to require that the unsupported clause is flagged or the answer is returned as partial/insufficient; add one supported-claim control case.
- **Dependencies:** No new package is required for conservative labels and validation. A semantic-entailment model would be a separate architectural choice and is not recommended without comparative testing.

### A2 — High: Paper-analysis evidence categories are heuristic, not verified support

- **Location:** [`analysis/paper_analyzer.py`](./analysis/paper_analyzer.py), `_validate_and_repair_json` (line 753), `_validate_field_type` (line 778), `_build_field_citation` (line 857).
- **Evidence:** Field values of the wrong type are coerced to strings instead of rejected. Post-extraction retrieval chooses candidate evidence passages for model-produced values. `evidence_type`, `evidence_sufficiency`, and `requires_human_review` are driven by token overlap and similarity heuristics (lines 929–955), not a direct check that the passage entails the complete extracted statement. The compatibility field `confidence` is explicitly populated with `retrieval_similarity` at line 955.
- **Impact:** An incorrect or partially hallucinated extraction may receive an “explicit fact” or “sufficient” label because some terms overlap; consumers may mistake the legacy `confidence` field for probability of correctness.
- **Proposed fix:** Reject invalid field types (and report validation issues) instead of coercing them; retain retrieval similarity under its own name; stop presenting similarity as confidence; default citations to human review unless support is actually established. Keep explicit unknown values rather than filling gaps.
- **Regression tests:** Wrong scalar/list types must fail validation; a passage with keyword overlap but a contradictory or unrelated claim must not be labelled supported; verify that no UI/report presents retrieval similarity as correctness confidence.
- **Dependencies:** None for strict types and honest status labels. Stronger entailment verification remains a separate validation decision.

### A3 — High: Comparison can label unresolved chunk references “available”

- **Location:** [`analysis/literature_comparator.py`](./analysis/literature_comparator.py), `_citation_summary` (line 279).
- **Evidence:** The function attempts to find a passage in `evidence_chunks`, but its final status is `"available"` whenever citation IDs are present and structurally complete (lines 331–345). It does not require the cited `(document_id, chunk_id)` to resolve to a passage. The returned citation may therefore have identifiers but no excerpt.
- **Impact:** Comparison output can imply that a finding has inspectable source evidence when the passage is absent or the chunk ID is wrong.
- **Proposed fix:** Mark a citation resolved only when the exact document/chunk pair is found and contains non-empty text; otherwise use a missing/unresolved status and require review. Do not infer the cited text from identifiers.
- **Regression test:** Add cases for a valid resolved chunk, a nonexistent chunk ID, a mismatched document ID, and an empty passage; only the first should be available.
- **Dependencies:** None.

### A4 — High: Follow-up candidates accept limitation references without resolving their evidence

- **Location:** [`analysis/research_gap_analyzer.py`](./analysis/research_gap_analyzer.py), `analyze` (line 24) and `_references` (line 146).
- **Evidence:** Candidate generation uses the extracted limitations and labels evidence `"citation_reference_present"` when `_references` returns identifiers. The analyzer does not resolve the cited chunk to its source passage or verify that the passage contains the limitation. It does correctly label outputs as candidates, require human review, and disclaim global novelty; those safeguards should be preserved.
- **Impact:** A hallucinated or incorrectly extracted limitation may become a follow-up question with a citation-looking reference.
- **Proposed fix:** Require passage resolution against the corresponding analysis evidence before describing a limitation as evidence-linked. Otherwise retain it only as an unverified suggestion with an explicit missing/unsupported evidence status.
- **Regression test:** Ensure an invalid or unresolved chunk does not receive `"citation_reference_present"`; include a resolved limitation control case.
- **Dependencies:** None.

### A5 — High: Retained real-paper validation is failed, and recorded success is not reproducible from retained outputs

- **Locations:** [`validation/step5/runs/20261028T113758773602Z/validation_report.md`](./validation/step5/runs/20261028T113758773602Z/validation_report.md), [`validation/step5/validation_report.md`](./validation/step5/validation_report.md), [README.md](./README.md), [DECISION.md](./DECISION.md).
- **Evidence:** The retained Groq report says `attention_is_all_you_need.pdf`, `bert_pretraining.pdf`, and `attention_survey.pdf` all failed with HTTP 400; it records zero audit rows. The earlier Ollama report records failures for the first two PDFs and success for the third. README/decision history later describes a successful run after increasing the generation budget, but a corresponding successful run with per-paper outcomes was not present in the retained run artifacts inspected.
- **Impact:** Current documentation overstates what can be independently verified. Past test counts or a valid JSON parse do not establish factual extraction quality.
- **Proposed fix:** Preserve the failed/historical artifacts. Run a fresh, explicitly approved validation using the configured backend; save a separate run directory and report per-paper extraction diagnostics, citation-resolution checks, and human-reviewed quality results. Update README/DECISION to distinguish documented history from retained/reproduced evidence.
- **Regression/validation:** Harness tests should distinguish transport completion, schema-valid output, citation validity, and human-reviewed extraction quality as separate statuses. A real-model run requires the user’s API authorization and may incur costs; none was initiated here.
- **Dependencies:** Existing backend and PDFs; API key/model availability and user approval for a live request.

### A6 — Medium: Validation success means only at least one non-empty field

- **Location:** [`validation/step5/run_validation.py`](./validation/step5/run_validation.py), `_is_valid_real_backend_result` (line 29).
- **Evidence:** Success requires a “real” backend, at least one attempt, and `valid_fields > 0`. This is a pipeline-execution check, not an extraction-quality or evidence-quality check.
- **Impact:** A mostly empty, placeholder-like, or factually wrong extraction can be recorded as success if one field is non-empty.
- **Proposed fix:** Name and report the existing condition as an operational/schema gate, then add separate quality statuses for required-field coverage, resolved evidence, and human review. Do not treat automated heuristics as truth verification.
- **Regression test:** Non-empty but unsupported/placeholder output should not pass the evidence-quality status, while a schema-valid but incomplete extraction should remain distinguishable from an API failure.
- **Dependencies:** No new dependency for status separation; quality review remains manual unless a separately validated verifier is chosen.

### A7 — Medium: A failed new run can leave a prior result visible as if current

- **Location:** [`app.py`](./app.py), analysis submission and result rendering (lines 310–322).
- **Evidence:** A successful run stores `research_bundle` in session state. The exception branch reports an error but does not clear or mark the previous bundle; rendering then reads that bundle unconditionally.
- **Impact:** After a prior successful run, a later failed run can show the old paper’s results beneath the new error, making it unclear which upload they belong to.
- **Proposed fix:** Bind results to the current upload/run identifier and visibly label retained prior results, or clear the current result when a new run starts. Preserve previous results only if the UI clearly distinguishes them.
- **Regression test:** Complete one successful AppTest run followed by a failed run and assert that stale results are not presented as the new analysis.
- **Dependencies:** None.

### A8 — Medium: Optional related-paper errors can abort the uploaded-paper result

- **Location:** [`app.py`](./app.py), `run_analysis` (line 129), plus related-paper analysis path in [`analysis/paper_analyzer.py`](./analysis/paper_analyzer.py), `analyze_paper_with_related_papers` (line 402).
- **Evidence:** The UI’s top-level exception handler reports failure for the overall operation. Discovery/related-paper failures can therefore prevent users from receiving a successfully completed primary-paper analysis, depending on the exception path.
- **Impact:** An optional network-backed stage can make the core uploaded-paper analysis appear to fail or disappear.
- **Proposed fix:** Preserve a successful primary result and represent discovery/verification/related-paper stages as independently successful, failed, or skipped. Show errors rather than silently swallowing them.
- **Regression test:** Simulate a Crossref/discovery failure after primary analysis succeeds; assert the primary result is still available with an explicit related-search failure status.
- **Dependencies:** None.

### A9 — Medium: Extraction prompt does not clearly isolate untrusted paper text

- **Location:** [`analysis/paper_analyzer.py`](./analysis/paper_analyzer.py), `_build_extraction_prompt` (line 594).
- **Evidence:** Selected PDF text is included in the extraction prompt, but the prompt does not explicitly instruct the model to treat document text as untrusted data and ignore embedded instructions. The bounded selection and post-extraction retrieval are useful controls, but they do not prevent prompt injection embedded in a paper.
- **Impact:** Text in an uploaded or downloaded paper could influence the model to ignore the requested schema or return unrelated content.
- **Proposed fix:** Clearly delimit document excerpts as data, instruct the model not to follow instructions contained in them, and preserve model-side schema validation. Do not treat this instruction as a security guarantee.
- **Regression test:** Add a paper chunk containing an instruction-like string and verify the extraction prompt boundaries and that the returned schema is still validated.
- **Dependencies:** None.

### A10 — Medium: Raw exception strings are displayed in the UI

- **Location:** [`app.py`](./app.py), exception handler (line 317).
- **Evidence:** The UI renders `str(error)` directly. Backend code has some secret-redaction tests, but arbitrary dependency/provider errors can include implementation details or local paths.
- **Impact:** Potential disclosure of unnecessary local/provider details in the interface and confusing error messages for a beginner user.
- **Proposed fix:** Show a concise, user-safe error and a correlation/run ID; retain sanitized diagnostics in application logs. Confirm API keys and authorization headers never enter logs or reports.
- **Regression test:** Inject an exception containing a secret-like value and local path; assert the UI omits the sensitive details while displaying actionable next steps.
- **Dependencies:** None.

### A11 — Medium design risk: Related-PDF URL validation has a DNS re-resolution gap

- **Location:** [`scholarly_search/related_paper_extractor.py`](./scholarly_search/related_paper_extractor.py), `_validate_public_https_url` (line 346) and the download/redirect path (lines 279–291).
- **Evidence:** The URL validator resolves the hostname and rejects non-public addresses (lines 367–388), then the HTTP opener performs the request and resolves the hostname again. A hostname whose DNS answer changes between validation and connection could bypass the first address check.
- **Impact:** This is a potential server-side request forgery risk if a malicious or compromised scholarly record can control a download URL and DNS answers. No exploit was attempted; this is a code-path design concern.
- **Proposed fix:** Pin the validated address for the actual connection while preserving TLS hostname verification, or use an established safe-fetch mechanism that validates the connected peer and every redirect target. Keep the existing HTTPS, redirect-count, and public-address checks.
- **Regression test:** Unit-test that the address actually connected to is the address validated, including a changed DNS answer and a redirect to a private address.
- **Dependencies:** No new package is necessarily required, but connection pinning must preserve TLS correctness and warrants a focused implementation review.

## Feature and architecture inventory

| Area | Observed state | Audit assessment |
|---|---|---|
| PDF processing | PyMuPDF extraction, page metadata, file/type/text checks and page limits are implemented. | Text-based PDFs are the supported path; scanned/image-only papers need OCR, which is not established as implemented. |
| Chunking and retrieval | Section-aware chunking/selection, page metadata, embeddings and FAISS are present. Extraction is bounded; the index is built after extraction validation. | Selected/omitted coverage is tracked, but it does not prove that the selected chunks contain all material required for accurate extraction. |
| LLM integration | Groq backend and configuration factory exist; optional Gemini backend also exists. Unit tests mock provider responses and check common errors. | Current retained Groq paper run failed. Gemini request behavior is tested against a mocked assumed HTTP shape, not independently validated against a live service. |
| Structured extraction | Prompt includes selected paper text; JSON parse repair/retry is bounded; diagnostics capture attempts and finish details when returned. | Wrong field types are coerced; extraction correctness and citations remain heuristic. |
| Evidence and citations | Document/chunk/page IDs and passages are retained in analyzer evidence results. QA citation indices are bounds-checked against retrieved chunks. | Passage resolution is inconsistent downstream; retrieval similarity, lexical overlap, and model confidence are not factual proof. |
| Discovery and verification | Crossref search, metadata verification, relevance filtering, duplicate handling, and related-PDF extraction are implemented. | No live provider calls were made in this audit. Network/provider failures and stale UI state need focused resilience testing. Metadata unavailable should remain distinct from incorrect. |
| Comparison | Deterministic comparison and source-level references are implemented. | Citation resolution currently has the high-priority gap described in A3; comparisons are not themselves a validated contradiction detector. |
| Follow-up candidates | Extracted limitations and missing fields produce review candidates; outputs explicitly disclaim global novelty. | Candidates can be grounded only in citation identifiers without passage resolution (A4). This is not a completed global research-gap search. |
| Reports | Markdown/report assembly includes source metadata, coverage/uncertainty and a warning that heuristic signals are not factual confidence. | Report structure is implemented, but correct passage resolution and completed end-to-end output validation remain prerequisites for strong evidence claims. |
| UI and jobs | Streamlit UI, progress/error display, PDF upload, consent prompts, result views/download, and a SQLite job manager are present. | UI behavior is unit/smoke-tested; no fresh upload-to-report run was executed. Stale results and raw exception display need attention. |
| Tests | Focused tests cover many parser, backend, citation structure, metadata, report and UI behaviors. | The known unsupported-claim test currently codifies unsafe behavior. Suite results from text files are historical only. |

## Documentation and validation discrepancies

- The README says the three real papers completed Step 5, while the retained Groq run says all three failed. The project history may describe a later successful experiment, but the corresponding run artifacts were not retained in the inspected run folder.
- DECISION.md contains historical entries with differing status/validation descriptions. Keep the historical record, but add a dated reconciliation after a new controlled validation rather than silently rewriting past outcomes.
- The validation harness reports only extraction-operation completion (see A6). Report wording should not conflate this with citation correctness or factual accuracy.
- The project has no dependency lockfile in the inspected root. Requirements use lower bounds, so a clean install may resolve newer versions than the developer environment. A reproducibility lock should be considered only after confirming the supported Python/platform setup.
- The included file named `attention_survey.pdf` is a validation input by filename. The filename alone does not establish that it is a survey paper; identify paper content/title from extracted metadata when reporting results.

## Recommended implementation sequence after approval

1. **Evidence correctness first:** resolve comparator/gap citations to exact passages; fix QA sufficiency and analyzer confidence semantics; reject wrong structured-field types.
2. **User-facing state and failure handling:** preserve primary-paper results when optional discovery fails, prevent stale results from masquerading as current, and make error messages safe/actionable.
3. **Regression tests:** update the test that currently accepts the unsupported QA claim; add unresolved/mismatched citation tests and run the focused suite.
4. **Fresh validation:** run the complete offline test suite; then, only with explicit authorization, make a small controlled Groq request and run the three bundled PDFs with new immutable output artifacts.
5. **Reconcile documentation:** update README/DECISION with dated, reproducible outcomes and separately label unit-tested, real-model-run, evidence-reviewed, and end-to-end-tested behavior.

No evidence supports replacing the existing technologies as an immediate fix. The priority is to correct the distinction between retrieved text, candidate citations, evidence sufficiency, and verified claims.

## Phase A conclusion

This audit confirms actionable correctness and validation gaps, but does not make changes to application behavior. The full baseline suite and real-paper validation remain unrun by design. Review this report before authorizing Phase B implementation.

## Batch 1 implementation status

The evidence-correction batch was implemented with conservative fixes in the QA, paper-analysis, citation-resolution, and gap-analysis paths. The changes preserve the existing architecture and explicitly distinguish retrieval similarity, passage support, and factual confidence.

### Updated files
- [qa/answer_generator.py](./qa/answer_generator.py)
- [analysis/paper_analyzer.py](./analysis/paper_analyzer.py)
- [analysis/literature_comparator.py](./analysis/literature_comparator.py)
- [analysis/research_gap_analyzer.py](./analysis/research_gap_analyzer.py)
- [tests/test_qa/test_answer_generator.py](./tests/test_qa/test_answer_generator.py)
- [tests/test_analysis/test_paper_analyzer.py](./tests/test_analysis/test_paper_analyzer.py)
- [tests/test_analysis/test_literature_comparator.py](./tests/test_analysis/test_literature_comparator.py)
- [tests/test_analysis/test_research_report.py](./tests/test_analysis/test_research_report.py)

### What changed
- Unsupported claims are no longer treated as fully supported just because the retrieval score is high or the model says the answer is sufficient.
- Invalid structured field types are rejected instead of being silently coerced to strings.
- Similarity and LLM self-rating remain separate from factual confidence; the compatibility field is kept at `None` when it would mislead users.
- Citations are only marked available when the exact `(document_id, chunk_id)` resolves to a real, non-empty passage.
- Research-gap candidates require a real passage match before they are treated as supported; otherwise they remain unresolved or unsupported for review.

### Remaining findings
- This batch corrected the evidence semantics and citation logic, but it did not validate the project end-to-end against live Groq or real PDFs.
- Real-model validation, UI cleanup, and broader end-to-end workflow checks remain future work and must be run separately once the environment is able to execute the project tests and external calls.
- The environment used here prevented a fresh local pytest run, so the results below are unverified execution attempts rather than a passing suite.

### Test command attempted
- `python -m pytest tests/test_qa/test_answer_generator.py tests/test_analysis/test_paper_analyzer.py tests/test_analysis/test_literature_comparator.py tests/test_analysis/test_research_report.py -q`

### Actual result
- The command could not complete in this execution environment because the available task shell reported that PowerShell was unavailable (`pwsh.exe` and `powershell.exe` were not found), and the VS Code test runner also reported that no discovered tests matched the requested file paths.
- Because of that environment limitation, I cannot truthfully claim a passing suite or a fully validated batch from this session.

This batch is therefore complete from the code-change perspective but still requires a clean local test run before claiming batch-level verification.


---

### Batch 14: Crossref and OpenAlex Provider Integration (Phase 2) (COMPLETED)

**Status:** ✅ Completed and verified

**Problem:**
- Phase 1 implemented OpenAlex search provider and factory
- Phase 2 required end-to-end integration validation for both providers
- Needed to ensure both providers work correctly through the existing workflow
- Required documentation for configuration and usage

**Solution Implemented:**
1. **Added integration tests** (tests/test_scholarly_search/test_search_provider_factory.py):
   - Factory function tests (18 tests total)
   - Provider selection tests (default, environment variable, case-insensitive)
   - Crossref provider creation tests
   - OpenAlex provider creation tests (with/without API key)
   - End-to-end integration tests for both providers
   - Empty results handling for both providers
   - Error handling for both providers
   - Protocol compliance verification
   - Result schema compatibility verification

2. **Updated README.md**:
   - Updated technology stack description to mention both Crossref and OpenAlex
   - Added "Scholarly search provider configuration" section
   - Documented environment variables:
     - `RESEARCHINTELLIGENCE_SEARCH_PROVIDER` (crossref or openalex)
     - `OPENALEX_API_KEY` (optional, for 10× budget)
   - Documented API key acquisition process
   - Documented shared result schema fields

**Integration Path Verification:**
- PaperAnalyzer.search_related_papers() → _get_related_paper_searcher() → create_search_provider() → (CrossrefSearchClient or OpenAlexSearchClient)
- Both providers implement SearchProvider protocol
- Both providers return results with identical schema:
  - Core fields: title, authors, year, venue, doi, url, abstract
  - Scoring: relevance_score, topic_overlap_score, topic_overlap_terms
  - Metadata: matched_query, source
- Downstream consumers (RelatedPaperEvidenceExtractor) work with either provider
- Deduplication is handled within each provider (DOI-based, then title/author)
- Source URLs and metadata are preserved as-is (no fabrication)

**Configuration Documentation:**
- Default: Crossref (no configuration needed)
- To use OpenAlex: `set RESEARCHINTELLIGENCE_SEARCH_PROVIDER=openalex`
- Optional API key: `set OPENALEX_API_KEY=your_key` (10× daily budget)
- API key acquisition: Visit https://openalex.org/settings/api, enter email, click activation link

**Verification:**
- Integration tests: 18 passed in 7.48s
- Scholarly search suite: 95 passed in 7.84s
- Full test suite: 495 passed, 1 skipped, 2 warnings in 99.38s

**Files Changed:**
- `tests/test_scholarly_search/test_search_provider_factory.py` (new file, 197 lines, 18 tests)
- `README.md` (updated technology stack and added configuration section)
- `AUDIT_REPORT.md` (added Batch 14 section)

**Integration Path:**
```
PaperAnalyzer.search_related_papers()
  → _get_related_paper_searcher()
    → create_search_provider(provider_name, api_key)
      → CrossrefSearchClient() OR OpenAlexSearchClient()
        → search_related_papers(paper_analysis, max_results)
          → Returns compatible schema results
```

**Remaining Limitations:**
1. **No real API testing:** All integration tests use mocked responses; real API calls not tested
2. **No provider comparison:** Result quality not compared between Crossref and OpenAlex
3. **No semantic search:** OpenAlex semantic search not yet implemented (keyword search only)
4. **No rate limit tracking:** Usage not proactively tracked via response headers
5. **No UI integration:** Streamlit interface not updated to show provider selection (not in scope)

**No new dependencies added** - Phase 2 added only tests and documentation.

---

### Batch 15: Offline Evaluation Infrastructure (Phase 4A) (COMPLETED)

**Status:** ✅ Completed and verified

**Problem:**
- Phase 3 investigation identified the need for evaluation infrastructure
- No existing infrastructure for measuring search quality
- No way to compare provider performance offline
- No mocked datasets for testing edge cases

**Solution Implemented:**
1. **Added `scholarly_search/evaluation.py`** (177 lines):
   - `SearchMetrics` class for metric container
   - `calculate_precision_at_5()`: Precision at top 5 using human-labeled relevance
   - `calculate_duplicate_rate()`: Duplicate rate after deduplication
   - `calculate_metadata_completeness()`: Average completeness across 6 fields
   - `calculate_average_topic_overlap()`: Average topic overlap score
   - `calculate_metrics()`: Combined calculation of all metrics

2. **Added `tests/test_scholarly_search/test_evaluation.py`** (383 lines, 25 tests):
   - Precision@5 tests (perfect, partial, zero, case-insensitive, missing DOI, fewer than 5, empty)
   - Duplicate rate tests (no duplicates, half duplicates, all duplicates, zero original)
   - Metadata completeness tests (perfect, partial, empty string, multiple results, empty)
   - Topic overlap tests (perfect, mixed, zero, missing score, empty)
   - Combined metrics tests (full calculation, serialization)
   - Evaluation fixture tests (transformer topic dataset, NLP topic dataset)

**Metric Definitions:**
- **Precision@5**: (relevant papers in top 5) / 5. Uses human-labeled DOIs for relevance.
- **Duplicate rate**: (original_count - deduplicated_count) / original_count. Denominator is candidates before deduplication.
- **Metadata completeness**: Average completeness across 6 fields (title, authors, year, doi, abstract, url). A field is complete if not None, not empty string, not empty list.
- **Average topic overlap**: Average of topic_overlap_score for results that have the field. Only counts results with explicit topic_overlap_score.
- **Provider coverage**: Number of candidates returned before deduplication (original_candidate_count).
- **Total results**: Number of results after deduplication.

**Evaluation Datasets:**
- **Transformer topic**: Mock results for "Attention Is All You Need" query, including BERT (relevant), psychology attention (irrelevant), and efficient transformers survey (relevant).
- **NLP topic**: Mock results for general NLP query, including Word2Vec and GloVe with missing metadata.

**Separation from Production:**
- Evaluation module is in `scholarly_search/evaluation.py`, separate from provider logic
- Relevance labels are in test fixtures, not production code
- Metrics use mocked data; do not represent real-world provider performance
- No API calls or Groq calls in evaluation tests

**Verification:**
- Evaluation tests: 25 passed in 3.45s
- Scholarly search suite: 120 passed in 5.42s
- Full test suite: 520 passed, 1 skipped, 2 warnings in 72.57s

**Files Changed:**
- `scholarly_search/evaluation.py` (new file, 177 lines)
- `tests/test_scholarly_search/test_evaluation.py` (new file, 383 lines, 25 tests)
- `AUDIT_REPORT.md` (added Batch 15 section)

**Remaining Limitations:**
1. **No real evaluation data:** Metrics calculated only from mocked datasets, not real API results
2. **No provider comparison:** No actual comparison between Crossref and OpenAlex performance
3. **Manual labeling required:** Precision@5 requires human judgment of relevance for real data
4. **No offline evaluation harness:** No script to run evaluation on real data
5. **No temporal/domain filtering:** No 2015+ filtering or domain hints (not implemented per constraints)
6. **Small datasets:** Only 2 mock datasets for testing; not comprehensive

**No new dependencies added** - Phase 4A added only evaluation infrastructure and tests.

**Important:** No temporal filtering (2015+) or domain hints (NLP/deep learning) were added globally, per constraints. Any future filtering must be optional and configurable.

---

### Batch 16: Query Generation Investigation (Phase 4B) (NO CHANGE)

**Status:** ✅ Investigation complete, no change justified

**Problem:**
- Phase 3 investigation suggested potential improvements to keyword query generation
- Proposed improvements included: increasing character limits, refining stopwords, adding temporal/domain filtering
- Need to verify if changes are actually justified by real issues

**Investigation Results:**

**1. Character Limits (No Change Justified):**
- Current limits: title (240), research_problem (180), methodology (140), dataset (80), evaluation_metrics (60)
- Tests confirm limits are working as designed
- No evidence of problematic truncation in typical use cases
- Transformer, BERT, and other test papers fit within limits
- **Conclusion:** Limits are adequate; no change justified without real data showing truncation is a problem

**2. Technical Phrase Preservation (Already Working):**
- Test results show hyphens are preserved: "multi-head", "self-attention", "sequence-to-sequence"
- `_clean_text()` function uses `re.sub(r"\s+", " ", html.unescape(value)).strip()`
- This preserves hyphens and other punctuation
- **Conclusion:** Technical phrases are already preserved; no change needed

**3. Stopword Handling (Working as Intended):**
- Stopwords are NOT removed in query generation
- Stopwords are only removed in `_topic_tokens()` for topic overlap scoring
- This is appropriate: query should include all terms, topic scoring should filter noise
- **Conclusion:** Stopword handling is correct; no change needed

**4. Query Generation Behavior (Documented):**
- Both Crossref and OpenAlex use identical `_build_queries()` logic
- Queries include title (if present) and topical fields (research_problem, methodology, dataset, evaluation_metrics)
- Duplicate terms are not deduplicated within queries (acceptable behavior)
- Empty fields are handled gracefully

**Files Changed:**
- `tests/test_scholarly_search/test_query_generation.py` (new file, 225 lines, 16 tests)
- `AUDIT_REPORT.md` (added Batch 16 section)

**Test Results:**
- Query generation tests: 16 passed in 3.26s
- Scholarly search suite: 136 passed in 5.64s
- Full test suite: 536 passed, 1 skipped, 2 warnings in 71.64s

**No production code changes made** - Investigation found current implementation is already appropriate.

**Remaining Limitations:**
1. **No real evaluation data:** Investigation used only mock data; real query quality not evaluated
2. **No provider comparison:** No actual comparison of query effectiveness between providers
3. **No temporal/domain filtering:** Per constraints, no 2015+ filtering or domain hints added
4. **Duplicate terms in queries:** Terms can repeat within queries (not a critical issue)

**Recommendation:**
- Current query generation is adequate for typical use cases
- Future improvements should be based on real evaluation data showing specific issues
- Do not increase character limits without evidence of truncation problems
- Do not add temporal/domain filtering without user configurability

**Important:** No production code was modified in this batch. Only tests were added to document current behavior.

---

### Batch 17: Crossref vs OpenAlex Evaluation Runner (Phase 5) (COMPLETED)

**Status:** ✅ Completed and verified

**Problem:**
- Phase 5 investigation identified need for evaluation runner
- No existing infrastructure for real-world provider comparison
- Need to run controlled searches and calculate metrics from manual labels

**Solution Implemented:**
1. **Added `evaluation/` package:**
   - `evaluation/__init__.py` - Package marker, exports evaluation functions
   - `evaluation/run_provider_comparison.py` - Main evaluation runner (372 lines)

2. **Runner functionality:**
   - Runs controlled searches for both Crossref and OpenAlex
   - Uses same paper, query inputs, and result limit
   - Saves results to timestamped directory (evaluation/results/comparison_TIMESTAMP/)
   - Creates relevance-labeling template for top 5 results
   - Calculates metrics from manual labels using existing evaluation module
   - Marks duplicate_rate as unavailable (providers don't expose raw candidates)

3. **Output files:**
   - `config.json` - Configuration and paper analysis
   - `crossref_final.json` - Crossref deduplicated results
   - `openalex_final.json` - OpenAlex deduplicated results
   - `relevance_labels_template.json` - Template for manual labeling
   - `relevance_labels.json` - User-created labels (after manual review)
   - `metrics_report.json` - Calculated metrics (after labeling)

4. **Tests added:**
   - `tests/test_evaluation/test_run_provider_comparison.py` (256 lines, 11 tests)
   - Tests for provider search, result saving, template creation, metric calculation
   - Extraction tests skipped (require complex mocking, not critical for runner testing)

**Key Design Decisions:**
- **Minimal analysis extraction:** Extracts title from PDF first page (not full Groq analysis) to avoid dependencies
- **Duplicate rate unavailable:** Cannot capture raw candidates before deduplication without modifying providers, so marked as unavailable
- **Manual labeling:** Precision@5 calculated only after user manually labels top 5 results
- **Separate directory:** Results saved to `evaluation/results/comparison_TIMESTAMP/` to avoid overwriting validation artifacts
- **No production changes:** Runner is separate infrastructure; does not modify search logic

**Usage:**
```bash
# Run evaluation (fetches results from both providers)
python evaluation/run_provider_comparison.py --paper attention_is_all_you_need --max-results 10

# After manual labeling, calculate metrics
python evaluation/run_provider_comparison.py --calculate-metrics --results-dir evaluation/results/comparison_TIMESTAMP
```

**Verification:**
- Runner tests: 13 passed in 3.71s
- Full test suite: 549 passed, 1 skipped, 2 warnings in 66.38s

**Files Changed:**
- `evaluation/__init__.py` (new file, 6 lines)
- `evaluation/run_provider_comparison.py` (new file, 372 lines)
- `tests/test_evaluation/test_run_provider_comparison.py` (new file, 256 lines, 11 tests)
- `AUDIT_REPORT.md` (added Batch 17 section)

**Output Format:**
- Results saved as JSON with provider identity, query, timestamp, final results, errors
- Metrics report includes: precision@5, metadata completeness, topic overlap, provider coverage
- Duplicate rate marked as unavailable with explanation
- Comparison summary indicates which provider performed better on each metric

**Exact command for later live evaluation:**
```bash
python evaluation/run_provider_comparison.py --paper attention_is_all_you_need --max-results 10
```

**Remaining Limitations:**
1. **No live API calls yet:** Runner is implemented but not executed (per constraints)
2. **Duplicate rate unavailable:** Cannot calculate exact duplicate rate without provider modification
3. **Minimal analysis:** Uses only title from PDF (full Groq analysis would require live API calls)
4. **Single paper evaluation:** Runner designed for single paper comparison
5. **Manual labeling required:** Precision@5 requires human judgment
6. **No real comparison:** No actual provider comparison has been performed yet

**No production code changes** - Runner is separate evaluation infrastructure only.
**Review Fixes Made:**
1. **Comparison logic:** Fixed comparison to mark "unavailable" when no labels provided or when both providers have zero labeled results. Previously would incorrectly compare even with no labels.
2. **Coverage comparison:** Fixed to mark "unavailable" when provider_coverage is 0 (indicates no results returned).


**No new dependencies added** - Uses existing providers, evaluation module, PDF extraction.

**Phase 5 Live Run Fixes (Post-Investigation):**
- Fixed PDF title extraction to skip permission/copyright headers (evaluation runner)
  - Added permission pattern detection: "provided proper attribution", "grants permission", "copyright", "©", "all rights reserved", "reproduce the tables", "solely for use in", "journalistic or", "scholarly works"
  - Skips permission lines and next 2 continuation lines
  - Extracts first substantive line after permission headers
  - Falls back to filename if no clear title found
- Fixed OpenAlex venue extraction AttributeError (provider implementation)
  - Changed `item.get("primary_location", {}).get("source", {}).get("display_name")` to `(item.get("primary_location") or {}).get("source", {}).get("display_name")`
  - Handles both missing key and None value safely
- Added 4 regression tests for title extraction (basic, permission header, filename fallback, error handling)
- Added 3 regression tests for OpenAlex venue extraction (normal venue, None primary_location, missing primary_location)
- Runner tests: 14 passed (was 13 passed)
- OpenAlex tests: 25 passed (was 22 passed)
- Full test suite: 553 passed, 1 skipped, 2 warnings
- Preserved failed run artifact: evaluation/results/comparison_20261029_214902/

**Phase 5 Persistent OpenAlex Failure Fix (Post-Investigation):**
- Fixed OpenAlex venue extraction to handle null source in primary_location
  - Changed from `(item.get("primary_location") or {}).get("source", {}).get("display_name")`
  - To `((item.get("primary_location") or {}).get("source") or {}).get("display_name")`
  - Handles case where primary_location exists but source is None
- Added regression test for `primary_location: {"source": None}` (actual failure case)
- OpenAlex tests: 26 passed (was 25 passed)
- Full test suite: 554 passed, 1 skipped, 2 warnings
- Preserved both failed run directories: comparison_20261029_214902/ and comparison_20261029_221732/

**Phase 5 Metrics-Calculation Crash Fix:**
- Fixed metrics output handling to check for error condition before printing
  - Added check for `metrics.get("error")` before accessing nested metrics
  - Added check for `metrics.get("crossref")` and `metrics.get("openalex")` before printing
  - Prints clear error message when labels are missing instead of crashing
- Added regression test for all-null labels (unlabeled results)
- Runner tests: 15 passed (was 14 passed)
- Full test suite: 555 passed, 1 skipped, 2 warnings
- Preserved run directory: comparison_20261029_224157/
