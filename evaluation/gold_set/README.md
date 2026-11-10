# Gold-Set Annotation Format

This directory stores human-annotated ground-truth records used by the Phase 1
evaluation harness.  Each file is a single JSON object following the schema
below.  The harness validates every file against that schema before computing
any metrics; it refuses to run on malformed or placeholder annotations.

## File naming

    <document_id>.json

`document_id` must match the filename stem and the `document_id` field inside
the file.  Example: `attention_is_all_you_need.json`.

## Schema

```json
{
  "document_id":        "<str>  — matches filename stem",
  "source_pdf":         "<str>  — path relative to project root",
  "annotator":          "<str>  — your name / handle (no placeholder)",
  "annotation_date":    "<str>  — ISO 8601 date, e.g. 2026-11-10",
  "split":              "dev" | "test",
  "annotated_without_model_output": true | false,
  "notes":              "<str | null>  — optional free-text notes",

  "fields": {
    "title":             {"value": "<str>",           "evidence_pages": [<int>, ...]},
    "authors":           {"value": ["<str>", ...],    "evidence_pages": [<int>, ...]},
    "publication_year":  {"value": "<str>",           "evidence_pages": [<int>, ...]},
    "abstract":          {"value": "<str>",           "evidence_pages": [<int>, ...]},
    "research_problem":  {"value": "<str>",           "evidence_pages": [<int>, ...]},
    "methodology":       {"value": "<str>",           "evidence_pages": [<int>, ...]},
    "dataset":           {"value": "<str | null>",    "evidence_pages": [<int>, ...]},
    "evaluation_metrics":{"value": ["<str>", ...],    "evidence_pages": [<int>, ...]},
    "reported_results":  {"value": "<str>",           "evidence_pages": [<int>, ...]},
    "limitations":       {"value": ["<str>", ...],    "evidence_pages": [<int>, ...]}
  }
}
```

### New required fields (Phase 1 hardening)

**`split`** (`"dev"` or `"test"`)
- `dev` — tuning split.  Run freely to explore metric behaviour.
- `test` — held-out split.  Run only when recording final numbers.
  The harness prints a prominent warning and writes every test-split
  run to `evaluation/results/test_split_audit.log`.

**`annotated_without_model_output`** (`true` or `false`)
- **Must be set explicitly.**  The validator rejects the boolean placeholder
  value; you must decide and set this yourself.
- Set to `true` if you wrote every annotation by reading the PDF directly,
  without looking at any model output (Groq, ChatGPT, etc.).
- Set to `false` if you consulted model output at any point.  Describe what
  you used in the `notes` field.
- The harness reports how many gold files declare `true` in every run.

### Annotation independence rules (item 8)

**Annotations must be written from the paper itself, never copied from
model output.**

Rationale: if you copy the model's own extraction into the gold set, the
evaluation metric measures self-consistency rather than accuracy against
ground truth.  This would make the numbers meaningless for interview purposes.

Permitted workflow:
1. Open the PDF.
2. Read the relevant section.
3. Write the annotation in your own words (or as a direct quote from the PDF).
4. Set `annotated_without_model_output: true`.

If you need to compare your annotation against model output *after* you have
finished annotating, that is fine — but the annotation must be written first.

### Field rules

- `value` — the human-verified correct answer, copied or paraphrased from
  the PDF.  For list fields (`authors`, `evaluation_metrics`, `limitations`)
  the value is a JSON array of strings.
- `evidence_pages` — list of 1-based page numbers where the information
  appears.  Use `[]` only when information is spread throughout the paper.
- For fields where the paper genuinely does not state a value (e.g. `dataset`
  for a purely theoretical paper), set `"value": null` and
  `"evidence_pages": []`.
- **Do not leave placeholder strings** (`"TODO"`, `"..."`, `"unknown"`).
  The validator rejects them.

### Retrieval hit-rate annotation

Each field's `evidence_pages` list is used to compute retrieval hit-rate@1,
@3, and @5, plus MRR.  The harness checks whether a retrieved chunk for that
field covers one of the annotated pages.

## Citation sample files

For the human-labeled citation agreement metric (item 3), put citation sample
files in `evaluation/citation_sample/<document_id>.json`.  Generate a template:

```powershell
python evaluation/generate_template.py --paper attention_is_all_you_need --citation-sample
```

## Templates

Run the template generator to create a skeleton file you can fill in:

```powershell
python evaluation/generate_template.py --paper attention_is_all_you_need
```

The generated file will be placed at
`evaluation/gold_set/<document_id>.json` with every value set to `"TODO"` so
the validator catches unfilled entries.

## Running the harness

```powershell
# Validate the gold set offline (no Groq calls, no cost)
python evaluation/run_harness.py --gold-dir evaluation/gold_set --dry-run

# Run against the dev split (uses Groq — costs apply)
python evaluation/run_harness.py --gold-dir evaluation/gold_set --split dev --confirm-groq-costs

# Run 3 times to quantify LLM variance
python evaluation/run_harness.py --gold-dir evaluation/gold_set --split dev --runs 3 --confirm-groq-costs

# Run against the test split (final numbers only — adds audit log entry)
python evaluation/run_harness.py --gold-dir evaluation/gold_set --split test --confirm-groq-costs
```

Output is written to `evaluation/results/harness_<TIMESTAMP>/`:
- `report.json` — full metrics in JSON
- `report.md`   — human-readable Markdown
- `per_paper/`  — per-paper analysis outputs for audit
