# Step 5 Fix Plan — Proposal Only

**Date:** 2026-10-28  
**State:** Proposal only. No production changes are made here.

## Priority 0 — Establish a reproducible baseline before changing behavior

| Change | Problem addressed | Files likely affected | Risk | Test | Measurable acceptance criterion |
|---|---|---|---|---|---|
| Capture exact prompt, request options, raw response, Ollama metadata, and per-stage wall time in a diagnostic harness. Keep paper text out of ordinary logs; write it only to an explicitly protected local diagnostic output when requested. | Existing failures have only a parse exception and aggregate runtime; root cause cannot be distinguished. | `validation/step5/run_validation.py` or a new diagnostic-only runner; `validation/step5/diagnosis/raw_outputs/` | Raw paper text may be sensitive; verbose logging can grow quickly. | Run against the same real model and PDFs; unit-test metadata extraction separately. | Every model attempt records model tag and digest, request options, prompt character count, estimated-token method, raw response, `done_reason`, prompt/eval counts, total duration, exception, and wall time, or explicitly marks a field unavailable. No secret values in logs. |
| Instrument actual retry, repair, and fallback event counters; remove counters from reports if not instrumented. | Current counters remain zero by initialization and do not measure events. | `analysis/paper_analyzer.py`; `validation/step5/run_validation.py`; related tests | Counter semantics may become ambiguous across nested calls. | Force each event through controlled unit tests, then inspect emitted report data. | Each counter increments exactly once per corresponding event in tests; no success or failure report displays a fabricated default. |

## Priority 1 — Stabilize bounded LLM calls and JSON handling

| Change | Problem addressed | Files likely affected | Risk | Test | Measurable acceptance criterion |
|---|---|---|---|---|---|
| Pass an explicit configurable `num_ctx` in the Ollama options, using a conservative default proven to fit the target machine/model; log the requested and runtime-reported context separately. | Current request leaves context to Ollama defaults; effective runtime context is not recorded. | `analysis/paper_analyzer.py`; analyzer configuration; tests; README/DECISION documentation | Too-large context increases RAM/VRAM usage and may slow inference; too-small context truncates information. | Unit-test that options carry the configured value; on the actual machine compare configured value with `ollama ps` after load. | Every run records configured `num_ctx`; for validation, runtime `CONTEXT` is at least the tokenized prompt plus generation reserve, or the test explicitly fails before extraction. No inferred context claim. |
| Make timeout a documented configuration value rather than rejecting all values above 90 seconds; use distinct connection/read or overall limits only if Ollama client behavior is verified. | Current API hard caps timeout at 90 seconds and can cause an otherwise progressing request to be cut off. | `analysis/paper_analyzer.py`; job-service configuration/callers; tests; docs | Raising the timeout can increase user wait and occupied worker time. Do not mask slow output by raising it alone. | Boundary tests for valid/invalid values and a controlled fake slow server/client test; separately measure real model timing. | Configuration accepts the documented supported range; each timed-out job reports elapsed time and explicit timeout status; UI/job polling stays responsive. Timeout must not silently retry indefinitely. |
| Add one bounded retry only for repairable/likely transient failures, with explicit policy and counters. Do not retry deterministic schema violations indefinitely. | One malformed/empty response currently ends extraction immediately. | `analysis/paper_analyzer.py`; tests; validation reporting | Retries can double inference time and duplicate provider load. | Simulate malformed-first/valid-second and two-malformed responses; run real-paper validation with retry count captured. | At most one retry per job; retry is visible in result/log; no more than two calls; repeated failure remains a failed job with the exact exception and raw responses preserved. |
| Implement a real JSON repair path for known syntax defects, parse strictly after repair, and retain the original response. Add repair counters. Avoid greedy extraction or constrain any extraction to a single valid JSON object. | `_validate_and_repair_json` currently runs after parsing and cannot repair malformed JSON; greedy brace matching can merge spans. | `analysis/paper_analyzer.py`; parser tests; `validation/step5/run_validation.py` | Repair can alter meaning or hide truncation. | Test trailing comma, invalid quotes, truncated JSON, braces inside strings, multiple objects, empty response, and valid JSON. | Only explicitly supported syntax repairs are applied; semantic content is not guessed; each repair is counted; truncated/empty/multiple-object cases fail explicitly rather than being marked successful. |
| Keep schema validation non-mutating and reject or sanitize invalid list members consistently. | Current validator mutates input and permits `None` within string lists. | `analysis/paper_analyzer.py`; tests | Stricter validation may turn formerly tolerated outputs into explicit failures/nulls. | Assert original input unchanged; cover mixed values including `[None]`, numbers, and nested objects. | Input dict is unchanged; every list item satisfies the documented element type or the entire field is explicitly null/invalid with a recorded validation issue. |

## Priority 2 — Make evidence claims verifiable

| Change | Problem addressed | Files likely affected | Risk | Test | Measurable acceptance criterion |
|---|---|---|---|---|---|
| Have the model return a verbatim quote and chunk ID per non-null field; verify the quote is an exact substring of that supplied chunk before emitting a citation. Keep original chunk text and page metadata linked. | Current citations are top-3 semantic retrieval results after extraction; they are not evidence that the model used or that the field is supported by those passages. | `analysis/paper_analyzer.py`; output schema; comparator if it consumes citations; tests; audit generation | Quotes may be absent, malformed, or fabricated; exact substring checks can reject harmless normalization differences. | Tests for exact quote, wrong chunk ID, quote absent from chunk, quote text with whitespace variation, and null fields. Real audit against source PDF. | Every non-null claim marked evidence-linked has a verified quote that is a substring of its cited chunk; otherwise it is explicitly unsupported and requires human review. |
| Rename `confidence` to `retrieval_similarity` (or equivalent) and document its numeric range/meaning; do not present similarity as calibrated confidence. | Current field is the average cosine similarity, not probability or calibrated certainty. | `analysis/paper_analyzer.py`; output consumers, tests, README/DECISION, validation CSV | Schema compatibility for existing saved JSON and clients. | Search all consumers; schema regression tests; confirm emitted key and range. | No output field called `confidence` represents raw cosine similarity; migration or compatibility behavior is explicit. |
| Remove the numeric-set contradiction heuristic or label it only as a numeric disagreement candidate; never promote disjoint number sets alone to contradiction. | Unrelated chunks with different numeric values can be falsely called contradictory; verbal/numeric conflicts may be missed. | `analysis/paper_analyzer.py`; tests; audit schema/docs | Removing a signal may make true numeric conflicts less visible. | Positive and negative cases with unrelated numbers, same result in different formats, and actual opposite claims. | No contradiction status is emitted solely from disjoint integer sets. If no reliable comparison is implemented, return `unknown`/human review. |

## Priority 3 — Improve coverage without unbounded prompts

| Change | Problem addressed | Files likely affected | Risk | Test | Measurable acceptance criterion |
|---|---|---|---|---|---|
| Replace the fixed evenly-spaced ten-chunk selection with deterministic section-aware selection that explicitly prioritizes abstract, method, results, limitations, conclusion, tables/captions, while reporting selected and omitted chunk IDs. Consider per-field extraction only if one bounded call remains unreliable. | Long papers and central tables can be omitted from one prompt; current selector has no table-aware policy. | `analysis/paper_analyzer.py`; chunk metadata if needed; validation tests; docs | More model calls increase runtime; section headings and table detection can fail on unusual layouts. | Synthetic selector tests for section/table regions and end-to-end papers in the protocol below. | Every output records selected chunk IDs and omitted coverage; all required categories are included when detectable; no prompt exceeds configured token budget; no paper content is silently claimed as covered if omitted. |
| Evaluate per-field or small-group extraction against the current single-call approach only after raw failure cause is known. | One response may omit fields, truncate JSON, or exceed the small model's reliable output budget. | Analyzer, prompt and tests; only if baseline evidence supports it | More calls may exceed the desired user wait and increase failure surface. | Compare on the same five papers/model/hardware with identical evidence and timing capture. | Keep per-field extraction only if it improves the predeclared field-evidence pass rate by at least 15 percentage points without exceeding the user-defined runtime budget; otherwise keep the bounded single call. |

## Priority 4 — Separate Step 5 from later network features

| Change | Problem addressed | Files likely affected | Risk | Test | Measurable acceptance criterion |
|---|---|---|---|---|---|
| Move related-paper search, verification, and extraction behind separate services/factories so constructing the Step 5 analyzer does not instantiate Crossref/OpenAlex clients unless those features are invoked. | The Step 5 constructor creates later-step clients, coupling offline analysis to the discovery/verification architecture. | `analysis/paper_analyzer.py`; service wiring; tests; package exports | Constructor/API changes could break callers or tests. | Constructor test that offline Step 5 can be created with network access disabled; integration tests for each explicit later-step service. | Step 5 extraction runs without constructing or calling scholarly clients; search/verification behavior remains explicitly callable and tested. |

## Alternatives considered

1. **Only increase timeout.** Rejected as the sole remedy. The source establishes a 90-second cap, but the existing validation report contains no LLM-call durations or raw responses; it does not establish timeout as the cause of the empty/invalid JSON.
2. **Only set a larger context.** Not justified until current prompt token count and runtime context are measured. A larger context consumes more memory and can worsen latency.
3. **Use a larger or different model.** Defer until same-model runs capture response metadata and the repair/context diagnosis is complete. Current evidence establishes two JSON parse failures and one historical success status, but not whether model capacity, timeout, context, transport, or response handling caused them. A larger model may improve output quality but increases hardware requirements and runtime; a different model would confound diagnosis.
4. **Use a hosted model.** Not proposed for this validation because it changes the privacy/dependency profile and the test request is to use the real local model. It can be a later user decision, not a diagnostic substitute.
5. **Replace embeddings or abandon retrieval.** Not indicated by the observed JSON parse exception. Retrieval/evidence issues should be repaired separately after extraction is reproducible.

## Re-validation protocol

### Corpus

Run the same pinned local model and configuration on five independently checked papers:

1. Empirical machine-learning paper.
2. Survey/review paper.
3. Short paper.
4. Table-heavy paper.
5. Multi-column paper.

Use the existing BERT paper only if its empirical-ML type is confirmed from the PDF. Do not assume `attention_survey.pdf` is a survey: the saved artifact contains a different-looking title and placeholder extraction values. The user should select/provide a survey, short, table-heavy, and multi-column PDF if no already-local paper has been checked for those traits. Do not download or substitute papers without user direction.

Before any real run, record OS, CPU, GPU, RAM, Python and Ollama server/CLI versions, model tag and digest, `ollama show` metadata, and runtime `ollama ps` context/offload after loading the model. Record one immutable configuration per run. Use the same model digest across all five papers.

### Per-paper run capture

- Capture the complete prompt in the protected local raw-output folder, its character count, and a named token-estimation method; record the actual Ollama `prompt_eval_count` separately.
- Capture exact request options and response fields: raw content, `done_reason`, `prompt_eval_count`, `eval_count`, `total_duration`, and process wall time.
- Save exact exceptions; preserve failed attempts and do not replace them with a later successful response.
- Run one baseline configuration, then one controlled explicit-`num_ctx` comparison on each original failure, changing no other parameter.
- Run the requested test suite once after changes and preserve its exact command, complete console output, exit code, duration, and pytest summary.

### Human audit and gates

- Use `step5_audit_template.csv`; it has ten fields for each of the five paper categories and blank `human_verdict` and `human_notes` cells.
- A candidate result only counts as field-supported when the extracted value is correct on the PDF spot-check and its quoted passage is present in the cited chunk/page.
- Report field-level extraction, quote-verification, citation-page, parse-failure, retry, repair, timeout, and human-review counts with numerators and denominators. Do not replace missing values with zero.
- The user must independently spot-check at least 10 fields per paper (at least 50 checks total) before deciding whether Step 5 is ready for further review. No system output or execution status substitutes for that review.
- Proposed measurable gate for submission to user review: all five papers finish or fail explicitly; zero malformed responses are labeled successful; at least 90% of non-null extracted fields pass the user's source check and at least 95% of emitted quotes validate as exact substrings; no silent truncation; all remaining unsupported fields are marked for human review. The user may change these thresholds before the protocol is run.

## Steps 6-9 remain frozen

Do not extend or refactor Steps 6-9 while Step 5 is being diagnosed. Once Step 5 has passed the above evidence gates and the user has reviewed it, re-review each later step before resuming:

- **Step 6:** Revisit Crossref as the sole current discovery provider versus the earlier plan listing OpenAlex as primary and Semantic Scholar as optional. Compare coverage, rate limits, query behavior, reproducibility, and API error handling.
- **Step 7:** Re-review DOI/title/year/author matching rules, source provenance, provider disagreement, and the distinction between metadata verification and paper-content verification.
- **Step 8:** Re-review open-access eligibility, licensing claims, safe PDF retrieval, page/character bounds, local retention/deletion, retry/restart behavior, and whether it uses the newly verified Step 5 evidence contract.
- **Step 9:** Re-review supported input shapes, duplicate handling, citation integrity, exact-match limits, and ensure comparisons do not overstate semantic agreement or contradictions.

Only the user decides whether these steps may resume.
