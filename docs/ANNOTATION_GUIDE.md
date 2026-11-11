# Gold Set Annotation Guide

This guide explains how to annotate research papers for the evaluation gold set.

## IMPORTANT RULE

**ANNOTATE BEFORE VIEWING ANY MODEL OUTPUT**

1. Read the PDF first
2. Fill in the gold annotation
3. Only then may you compare with model output for verification
4. Set `annotated_without_model_output` to `true` if you followed this rule
5. Set `annotated_without_model_output` to `false` if you consulted model output first

The purpose of the gold set is to measure how well the system performs when the annotator has NOT seen the model's output. If you look at model output first, you cannot claim the gold set is independent.

---

## Schema Overview

Each gold annotation file is a JSON with the following structure:

```json
{
  "document_id": "paper_stem",
  "source_pdf": "path/to/paper.pdf",
  "annotator": "your_name",
  "annotation_date": "YYYY-MM-DD",
  "split": "dev",
  "annotated_without_model_output": true,
  "notes": "Optional notes about the annotation process",
  "fields": {
    "title": {"value": "...", "evidence_pages": [1]},
    "publication_year": {"value": "2017", "evidence_pages": [1]},
    ...
  }
}
```

### 10 Fields

**String fields (7):**
1. title
2. publication_year
3. abstract
4. research_problem
5. methodology
6. dataset
7. reported_results

**List fields (3):**
8. authors
9. evaluation_metrics
10. limitations

---

## Field-by-Field Annotation Guidelines

### 1. title

**What counts as correct:**
- The exact paper title as it appears on the first page
- Include subtitle if present
- Use capitalization as shown in the paper

**How to annotate from PDF:**
- Look at the first page of the PDF
- Copy the title text exactly as written
- Evidence pages: typically page 1

**Example:**
```
value: "Attention Is All You Need"
evidence_pages: [1]
```

---

### 2. publication_year

**What counts as correct:**
- The year the paper was published
- Usually found near the title or in the footer/header
- If the paper has a conference date, use the conference year
- If the paper has a journal publication date, use the journal year

**How to annotate from PDF:**
- Check the first page (title area) for publication year
- Check footer/header for copyright date
- Check references for self-citation with year
- Use the most authoritative year (journal publication > conference > preprint)

**Example:**
```
value: "2017"
evidence_pages: [1]
```

---

### 3. abstract

**What counts as correct:**
- The full abstract text as written in the paper
- Include all sentences
- Do not summarize or paraphrase
- Use exact wording from the paper

**How to annotate from PDF:**
- Find the abstract section (usually on page 1)
- Copy the abstract text exactly
- If the paper has no abstract, set value to null and evidence_pages to []

**Example:**
```
value: "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks..."
evidence_pages: [1]
```

---

### 4. research_problem

**What counts as correct:**
- The main problem the paper addresses
- Usually 1-3 sentences describing the research gap or motivation
- Focus on the "why" of the paper, not the "how"

**How to annotate from PDF:**
- Read the introduction section
- Look for sentences like "We address...", "This paper aims to...", "The problem is..."
- Summarize the problem in your own words, but stay close to the paper's language
- Do not include the methodology or solution

**Example:**
```
value: "Develop a sequence transduction model that relies only on attention mechanisms, eliminating recurrence and convolutions, to improve quality and training efficiency."
evidence_pages: [1, 2]
```

---

### 5. methodology

**What counts as correct:**
- The key technical approach or algorithm
- Model architecture, experimental setup, data processing pipeline
- 2-4 sentences covering the main components

**How to annotate from PDF:**
- Read the methods/experiments section
- Identify the core technical contribution
- List the main components (e.g., "Transformer architecture with multi-head self-attention, positional encodings, feed-forward layers")
- Do not include every detail; focus on the main design choices

**Example:**
```
value: "Transformer architecture with multi-head self-attention, positional encodings, feed-forward layers, encoder-decoder structure, and residual connections."
evidence_pages: [2, 3, 5, 6]
```

---

### 6. dataset

**What counts as correct:**
- The dataset(s) used for experiments
- Dataset name, size, domain if relevant
- If the paper uses multiple datasets, list all

**How to annotate from PDF:**
- Read the experiments section
- Look for "dataset", "data", "corpus", "training data"
- List dataset names with brief descriptions
- If the paper is theoretical with no experiments, set value to null and evidence_pages to []

**Example:**
```
value: "WMT 2014 English-to-German and English-to-French translation datasets"
evidence_pages: [7]
```

---

### 7. reported_results

**What counts as correct:**
- The main quantitative results reported in the paper
- Key metrics (accuracy, BLEU, F1, etc.) with values
- 2-4 sentences summarizing the main findings

**How to annotate from PDF:**
- Read the results section
- Identify the primary metrics
- Report the best results with context (e.g., "28.4 BLEU on WMT 2014 English-to-German")
- Do not include every result; focus on the main claims

**Example:**
```
value: "28.4 BLEU on WMT 2014 English-to-German translation task, 41.8 BLEU on WMT 2014 English-to-French translation task. The Transformer base model achieves 28.4 BLEU on WMT 2014 En-De, outperforming previous best models."
evidence_pages: [7, 8]
```

---

### 8. authors

**What counts as correct:**
- All authors listed on the paper
- Full names as shown in the paper
- Order matters (first author, second author, etc.)

**How to annotate from PDF:**
- Look at the first page (title area) for author list
- Copy author names exactly as written
- Preserve order
- Include all authors, not just the first author

**Example:**
```
value: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Łukasz Kaiser", "Illia Polosukhin"]
evidence_pages: [1]
```

---

### 9. evaluation_metrics

**What counts as correct:**
- The metrics used to evaluate the method
- Metric names (e.g., "BLEU", "accuracy", "F1", "precision", "recall")
- Short descriptions if needed

**How to annotate from PDF:**
- Read the experiments/results section
- Look for "evaluation metric", "metric", "measured by"
- List all metrics used
- If the paper reports no metrics (theoretical work), set value to null and evidence_pages to []

**Example:**
```
value: ["BLEU", "perplexity", "training time", "parameter count"]
evidence_pages: [7, 8]
```

---

### 10. limitations

**What counts as correct:**
- The limitations of the method as discussed by the authors
- Usually found in discussion, conclusion, or future work sections
- 2-4 sentences covering the main limitations

**How to annotate from PDF:**
- Read the discussion/conclusion section
- Look for "limitations", "drawbacks", "future work", "not addressed"
- Summarize the limitations in your own words
- Do not include limitations you infer; only what the authors explicitly state

**Example:**
```
value: [
  "The Transformer has a quadratic time complexity in sequence length due to the self-attention mechanism, which makes it less efficient for very long sequences compared to RNNs.",
  "The model requires large amounts of data and compute resources to train effectively.",
  "Positional encodings are fixed-length sinusoidal functions; while they work well, other approaches might be more flexible."
]
evidence_pages: [7, 8]
```

---

## evidence_pages Guidelines

**What counts as correct:**
- Page numbers where the information can be found in the PDF
- Use 1-based page numbers (page 1 is the first page)
- List all pages where the information appears

**How to annotate:**
- Open the PDF
- Navigate to the section containing the information
- Note the page number (shown in your PDF viewer)
- If information spans multiple pages, list all pages
- If information is spread throughout the paper (e.g., limitations discussed in multiple sections), use []

**Example:**
```
evidence_pages: [1]        # Found on page 1 only
evidence_pages: [2, 3]     # Found on pages 2 and 3
evidence_pages: []         # Spread throughout or not applicable
```

---

## Common Mistakes to Avoid

1. **Looking at model output first**
   - ❌ Read model output, then annotate
   - ✅ Read PDF, annotate, then check model output

2. **Summarizing instead of quoting**
   - ❌ Abstract: "This paper discusses attention mechanisms"
   - ✅ Abstract: "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks..."

3. **Including methodology in research_problem**
   - ❌ Research problem: "We use a Transformer to..."
   - ✅ Research problem: "Develop a sequence transduction model that relies only on attention mechanisms..."

4. **Missing authors beyond the first**
   - ❌ Authors: ["Ashish Vaswani"]
   - ✅ Authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", ...]

5. **Using placeholder values**
   - ❌ Title: "TODO"
   - ❌ Methodology: "unknown"
   - ✅ Leave value as null if genuinely absent, or fill in the actual value

6. **Incorrect page numbers**
   - ❌ Using 0-based page numbers (page 0 is invalid)
   - ✅ Using 1-based page numbers (page 1 is the first page)

---

## Validation

After filling in the gold annotation, run the validator:

```bash
python -c "import json,sys; from evaluation.gold_set_schema import validate_annotation; p=sys.argv[1]; validate_annotation(json.load(open(p, encoding='utf-8')), p.replace('\\','/').split('/')[-1][:-5]); print('VALID')" evaluation/gold_set/your_paper.json
```
Prints VALID on success; otherwise it raises GoldSetValidationError naming the problem.

The validator will check:
- All required fields are present
- Placeholder values are rejected
- Date format is correct (YYYY-MM-DD)
- Page numbers are 1-based (not 0)
- evidence_pages is a list of integers

If validation fails, fix the errors and re-run.

---

## Bundled PDFs

The repository includes three bundled PDFs for annotation:

1. **attention_is_all_you_need.pdf** - Already annotated (example reference)
2. **bert_pretraining.pdf** - Blank template, needs annotation
3. **attention_survey.pdf** - Blank template, needs annotation

**Note:** attention_survey.pdf is actually titled "Weighted Estimates for the Bergman Projection on the Hartogs Triangle" - this is a misleading filename, but annotate based on the actual PDF content.

---

## Split Assignment

- **dev split**: Use for tuning and initial evaluation. Run freely.
- **test split**: Held-out split. Harness will print a warning and log every test-split run.

For initial work, use `split: "dev"`. Move to `split: "test"` only when you have a stable evaluation pipeline and want to measure final performance.
