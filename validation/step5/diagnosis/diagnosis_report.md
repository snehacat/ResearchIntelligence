# Step 5 Diagnosis — 2026-10-28

## Scope and evidence labels

This diagnosis changes no production code and does not modify existing
`validation/step5` files. Claims are labeled as **Measured**, **Read from
code**, or **Assumed**. No accuracy claim is made from the older report's
single successful status.

## Part 1 — Decision-log correction

**Read from code / documentation:** `DECISION.md` now lists only Steps 1-4 as
completed. Step 5 is recorded as implemented with failed real-paper validation
and no user approval. Steps 6-9 are recorded as implemented, unreviewed, and
frozen pending Step 5. A dated correction preserves the historical summaries
while clarifying the earlier status claims and test counts.

## Part 2 — Requested suite and runtime diagnosis

### Full test suite

**Measured execution outcome:** The exact requested command,
`.\venv\Scripts\python.exe -m pytest tests/ -v --tb=short`, did **not** start.
The task runner reported that neither `pwsh.exe` nor `powershell.exe` was
available and that no terminal was found. The VS Code test runner separately
reported that no tests were available through its test extensions. No pytest
process ran, so there is no exit code, duration, pass/fail count, or test
interruption to report. This is not a passing test result and was not an
interrupted test run. The exact execution-tool messages are in
`raw_outputs/execution_blocked.txt`.

### Environment

| Item | Finding | Evidence label |
|---|---|---|
| OS | Windows | Read from the active workspace environment |
| CPU | Not measured | Python execution service unavailable |
| GPU | Not measured | Python execution service unavailable; `nvidia-smi` was not run |
| RAM | Not measured | Python execution service unavailable |
| Python | 3.12.10 is stated in the existing `validation/step5/validation_report.md`; not re-measured today | Read from existing report |
| Ollama Python client | 0.6.3 is stated in the existing report; this is not the Ollama server/CLI version and was not re-measured today | Read from existing report |
| Ollama server/CLI version | Unknown | Not queried |
| Model | The existing report says `llama3.2:3b`; the analyzer default also names it. Current model availability was not checked | Read from existing report/code |
| Model digest | Unknown | `ollama list` was not run |
| Effective runtime context | Unknown | `ollama ps` was not run while the model was loaded |

The existing report's `Context length: 131072` is an unverified historical
entry and must not be treated as the current effective context. **Measured from
official Ollama documentation:** the context-length page explains that defaults
vary with available VRAM and shows `ollama ps` reporting a running model's
allocated `CONTEXT`. The code-run service was unavailable, so neither the
requested `ollama show` check nor a runtime `ollama ps` check could be
performed. Documentation consulted:

- https://docs.ollama.com/context-length
- https://docs.ollama.com/cli
- https://docs.ollama.com/api/chat

The chat API documents response metadata including `done_reason`,
`total_duration`, `prompt_eval_count`, and `eval_count`; none of those values
was captured for a new run.

### Real-model reproduction

**Measured execution outcome:** Both requested real-paper reruns were blocked
before model availability could be checked. No model request was sent, and no
raw content, `done_reason`, `prompt_eval_count`, `eval_count`, `total_duration`,
prompt character count, prompt token estimate, or per-run duration is
available for this attempt. Per the instruction not to substitute a model or
use mocks, no substitute was used.

**Read from existing validation report (historical run, not reproduced):**

| Paper | Recorded status/error | Recorded runtime | Raw response and Ollama metadata |
|---|---|---:|---|
| `attention_is_all_you_need.pdf` | Failed; `Failed to parse JSON from LLM response: Expecting value: line 1 column 1 (char 0)` | 146.04 s | Not present in the report |
| `bert_pretraining.pdf` | Failed; `Failed to parse JSON from LLM response: Expecting value: line 1 column 1 (char 0)` | 56.71 s | Not present in the report |

The old durations are full-pipeline runtimes, not measured LLM request
durations. The old report contains no raw response body or Ollama response
metadata. These values are not new diagnostic runs.

### Context-length hypothesis

**Verdict: Inconclusive; neither supported nor refuted by runtime evidence.**
**Read from code:** the prompt includes up to 12,000 paper-text characters plus
instructions, while the Ollama request options set `num_predict` but do not set
`num_ctx` (`analysis/paper_analyzer.py`, lines 306-311 and 338-391).

No actual prompt was built for these runs, no prompt character count or token
estimate was captured, the effective context is unknown, and no
`prompt_eval_count` or raw response is available. The requested comparison run
with explicit `num_ctx=8192` was not performed. Therefore silent context
truncation cannot be concluded from the available evidence.

### 90-second timeout

**Read from code:** `PaperAnalyzer` caps `llm_timeout_seconds` at 90 seconds
and passes the value to the Ollama client (`analysis/paper_analyzer.py`, lines
49-50, 94-111, and 728-734).

**Read from existing validation report:** the two historical outcomes are
JSON parse errors; their recorded full-pipeline times are 146.04 s and
56.71 s. Since the LLM-call durations and raw responses were not recorded,
those full-pipeline times do not show whether the cap contributed. The
timeout's contribution is **unknown**; neither paper was re-run here.

## Part 3 — Suspected defects A-J

The source locations below refer to `analysis/paper_analyzer.py` unless noted.
Examples based on the saved survey JSON are explicitly artifact measurements,
not fresh LLM results; that JSON contains placeholder-looking values and is
not evidence of extraction accuracy.

| Item | Verdict | Evidence and small example |
|---|---|---|
| **A. SequenceMatcher against full chunk makes the 0.8 explicit-fact threshold unreachable for short values** | **Confirmed for short values; requested real-survey ratio distribution not measured** | **Read from code:** lines 47, 655-666 compare the full strings and require `overlap > 0.8`. **Measured from existing artifact:** `reported_results` is `"Sample results."` (15 characters) and its cited chunk records `chunk_size: 500` (`validation/step5/results/attention_survey_analysis.json`). Even with all 15 query characters matched, the SequenceMatcher ratio is at most `2*15/(15+500) = 0.0583`, far below 0.8. The saved JSON is placeholder-like; no fresh distribution was computed. |
| **B. `confidence` is average cosine similarity; sufficiency uses similarity thresholds** | **Confirmed** | **Read from code:** lines 590-593 assign the arithmetic mean of retriever `similarity` values to `confidence`; lines 668-690 classify sufficiency using only the maximum similarity with 0.7/0.3 thresholds. **Measured from existing artifact:** one saved citation reports `confidence: 0.10864995668331782` and `evidence_sufficiency: "insufficient"`; this is an old output value, not a calibrated probability. |
| **C. Contradiction means the first two chunks share no integers** | **Partly** | **Read from code:** lines 695-725 only return true when both first chunks have non-empty number sets and those sets are disjoint. A pair with `{2022}` and `{2019}` triggers the heuristic; a pair with no numbers does not. The rule does not establish semantic contradiction. |
| **D. Citations are retrieved after extraction, not passages the model actually used** | **Confirmed** | **Read from code:** lines 306-322 parse the model response without quote/chunk provenance. Lines 517-610 then query the retriever with the extracted field and use `top_k=3`. The reported chunks are post-hoc retrieval results, not known model-attended passages. |
| **E. At most ten chunks / 12,000 paper-text characters can omit middle results/tables** | **Confirmed as a possible omission** | **Read from code:** limits are at lines 53-54 and applied at lines 338-391; selection is capped at ten at lines 394-439. For a document with more than ten chunks, unselected material is absent from the prompt. No table-aware selection exists in this path. This establishes possible omission, not that a specific table was omitted in a new run. |
| **F. No retry or parse-failure JSON repair exists, despite DECISION-019** | **Confirmed; documentation overstates implementation** | **Read from code:** lines 288-336 make one `client.chat` call; JSON decode errors become `ExtractionError`. Lines 448-477 add missing keys and coerce field types only after successful JSON parsing. **Read from `DECISION.md`, DECISION-019:** the plan describes a controlled repair, a retry, and targeted QA fallback. These parse-failure recovery actions are not present in the extraction method. |
| **G. Lists can retain `None`; validation mutates input; greedy regex may capture wrong spans** | **Confirmed** | **Read from code:** lines 461-477 assign missing keys directly into the supplied dict. Lines 497-503 convert list members using `str(item) if item is not None else None`, so `[None]` remains `[None]`. Lines 317-319 use greedy `\{.*\}` with DOTALL; multiple brace-delimited spans can be consumed as one invalid JSON candidate. |
| **H. Constructor rejects timeout values above 90 seconds** | **Confirmed** | **Read from code:** lines 49-50 set the maximum to 90; lines 94-101 reject values above it. For example, `llm_timeout_seconds=91` follows the rejection branch. |
| **I. Constructor creates Crossref and metadata-verifier clients** | **Confirmed** | **Read from code:** lines 107-110 default to `CrossrefSearchClient()` and `PaperMetadataVerifier()`. This couples construction to the later discovery/verification types, although this source alone does not establish that network requests occur during construction. |
| **J. Validation counters are hardcoded** | **Confirmed** | **Read from code:** `validation/step5/run_validation.py`, lines 34-37 initializes `retries`, `json_repairs`, and `fallback_calls` to zero; success/failure records return those variables without any increment in the script. The saved report's zero counters are therefore not evidence that these events were instrumented. |

## Part 4 — Fix proposal

The requested proposal is in `fix_plan.md`. It is not implemented. Steps 6-9
remain frozen pending Step 5 review.
