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
  "source_pdf":         "<str>  — path relative to project root, e.g. validation/step5/papers/X.pdf",
  "annotator":          "<str>  — your name / handle",
  "annotation_date":    "<str>  — ISO 8601 date, e.g. 2026-11-10",
  "notes":              "<str | null>  — optional free-text annotation notes",

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

### Field rules

- `value` must be the **human-verified** correct answer, copied or paraphrased
  from the PDF.  For list fields (`authors`, `evaluation_metrics`,
  `limitations`) the value is a JSON array of strings.
- `evidence_pages` is a list of 1-based page numbers where the information
  appears.  Use an empty list `[]` only when the information is spread
  throughout the paper (e.g. the title appears on page 1 but also in headers).
- For fields where the paper genuinely does not state a value (e.g. `dataset`
  for a purely theoretical paper), set `"value": null` and
  `"evidence_pages": []`.
- Do **not** leave placeholder strings (`"TODO"`, `"..."`, `"unknown"`).  The
  validator rejects them.

### Retrieval hit-rate annotation

Each field's `evidence_pages` list is used to compute retrieval hit-rate@k:
the harness checks whether at least one retrieved chunk for that field covers
one of the annotated pages.

## Templates

Run the template generator to create a skeleton file you can fill in:

```powershell
python evaluation/generate_template.py --paper attention_is_all_you_need \
  --papers-dir validation/step5/papers
```

The generated file will be placed at
`evaluation/gold_set/<document_id>.json` with every value set to `"TODO"` so
the validator catches unfilled entries.
