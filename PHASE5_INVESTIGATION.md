# Phase 5 Investigation: Real-World Crossref vs OpenAlex Search Evaluation

## Current State Inspection

### Existing Evaluation Module
**File:** `scholarly_search/evaluation.py`

**Available functions:**
- `calculate_precision_at_5(results, relevant_dois)` - Calculates precision using human-labeled DOIs
- `calculate_duplicate_rate(results, original_candidate_count)` - Calculates duplicate rate
- `calculate_metadata_completeness(results)` - Calculates average completeness across 6 fields
- `calculate_average_topic_overlap(results)` - Calculates average topic overlap score
- `calculate_metrics(results, relevant_dois, original_candidate_count)` - Combined metrics calculation

**Status:** ✅ Existing evaluation module is sufficient for calculating metrics from real results.

### Provider Interface
**File:** `scholarly_search/search_provider_factory.py`

**Interface:**
```python
provider = create_search_provider(provider_name="crossref" or "openalex")
results = provider.search_related_papers(paper_analysis, max_results=10)
```

**Status:** ✅ Factory allows direct provider selection without environment variables.

### Current Provider Behavior
**Both providers:**
- Fetch `max_results * 3` candidates before deduplication (rows = min(max_results * 3, 100))
- Deduplicate by DOI, then by normalized title
- Rank by topic_overlap_score (primary) and relevance_score (secondary)
- Return deduplicated results up to max_results

**Missing for evaluation:**
- No way to capture raw candidates before deduplication
- No way to separate deduplication step for duplicate rate calculation

### Existing Validation Structure
**Directory:** `validation/step5/`

**Contents:**
- `papers/` - PDF files (attention_is_all_you_need.pdf, bert_pretraining.pdf, attention_survey.pdf)
- `run_validation.py` - Full validation runner with Groq
- `run_single_paper.py` - Single paper validation
- `runs/` - Historical validation artifacts
- `diagnosis/`, `raw_outputs/`, `results/` - Result directories

**Status:** ⚠️ Existing validation is for full paper analysis, not search provider comparison. A separate evaluation runner is needed.

---

## Recommended Evaluation Procedure

### Step 1: Create Evaluation Runner
**File:** `evaluation/run_provider_comparison.py` (new file)

**Purpose:** Small Python script to run controlled searches for both providers and save results.

**Functionality:**
1. Load paper analysis from existing validation result or create minimal analysis from PDF
2. Run Crossref search with max_results=10
3. Run OpenAlex search with max_results=10
4. Save raw results to JSON files (before deduplication capture not needed - can use candidate count)
5. Save final results to JSON files (after deduplication)
6. Generate a relevance-labeling template file

**Dependencies:** Uses existing providers, evaluation module, and PDF extraction. No new dependencies.

### Step 2: Paper Selection
**Recommended paper:** `attention_is_all_you_need.pdf`

**Reasoning:**
- Well-known paper with clear related work
- Expected relevant papers: BERT, GPT, Efficient Transformers, Longformer, etc.
- Existing validation has analysis data available

**Alternative:** If full analysis is needed, can extract minimal fields (title, research_problem, methodology, dataset, evaluation_metrics) directly from PDF without Groq.

### Step 3: Execution Commands
```bash
# Run evaluation
python evaluation/run_provider_comparison.py --paper attention_is_all_you_need --max-results 10

# This will create:
# evaluation/results/crossref_raw.json
# evaluation/results/openalex_raw.json
# evaluation/results/crossref_final.json
# evaluation/results/openalex_final.json
# evaluation/results/relevance_labels_template.json
```

### Step 4: Manual Relevance Labeling
**File:** `evaluation/results/relevance_labels_template.json`

**Format:**
```json
{
  "crossref": {
    "10.1234/relevant_doi": true,
    "10.1234/irrelevant_doi": false,
    "10.1234/uncertain_doi": null
  },
  "openalex": {
    "10.5678/relevant_doi": true,
    "10.5678/irrelevant_doi": false
  }
}
```

**Instructions:**
- Review top 5 results from each provider
- Mark DOIs as true (relevant), false (irrelevant), or null (uncertain)
- Save as `evaluation/results/relevance_labels.json`

### Step 5: Calculate Metrics
**File:** `evaluation/calculate_metrics.py` (new file, optional)

**Purpose:** Load results and labels, calculate metrics using existing evaluation module.

**Alternative:** Add metric calculation to the runner script itself after labeling.

---

## Required Files for Implementation

### New Files
1. **`evaluation/run_provider_comparison.py`** - Main evaluation runner
2. **`evaluation/__init__.py`** - Package marker
3. **`evaluation/results/`** - Directory for results (not in version control)

### Existing Files to Use
1. **`scholarly_search/search_provider_factory.py`** - Provider creation
2. **`scholarly_search/crossref_client.py`** - Crossref search
3. **`scholarly_search/openalex_client.py`** - OpenAlex search
4. **`scholarly_search/evaluation.py`** - Metric calculations
5. **`pdf_processing/text_extractor.py`** - PDF text extraction (if minimal analysis needed)
6. **`validation/step5/papers/attention_is_all_you_need.pdf`** - Test paper

### Files to NOT Modify
- Production search logic (`crossref_client.py`, `openalex_client.py`)
- Provider factory (`search_provider_factory.py`)
- Evaluation module (`evaluation.py`)
- Historical validation artifacts (`validation/step5/runs/`)

---

## Data Storage Structure

### Directory: `evaluation/results/`

**Files created:**
```
evaluation/results/
├── crossref_raw.json          # Raw results (all candidates fetched)
├── openalex_raw.json          # Raw results (all candidates fetched)
├── crossref_final.json        # Deduplicated results (max_results)
├── openalex_final.json        # Deduplicated results (max_results)
├── relevance_labels_template.json  # Template for manual labeling
├── relevance_labels.json       # Human-labeled relevance (user creates)
└── metrics_report.json        # Calculated metrics (after labeling)
```

### Result JSON Schema
```json
{
  "provider": "crossref",
  "query_paper": {
    "title": "Attention Is All You Need",
    "research_problem": "...",
    "methodology": "...",
    "dataset": "...",
    "evaluation_metrics": ["..."]
  },
  "query_used": "...",
  "candidate_count": 30,  // For duplicate rate calculation
  "results": [
    {
      "title": "...",
      "authors": ["..."],
      "year": 2020,
      "venue": "...",
      "doi": "10.1234/...",
      "url": "https://...",
      "abstract": "...",
      "relevance_score": 0.95,
      "topic_overlap_score": 0.7,
      "topic_overlap_terms": ["..."],
      "matched_query": "...",
      "source": "crossref"
    }
  ]
}
```

### Metrics Report Schema
```json
{
  "crossref": {
    "precision_at_5": 0.8,
    "duplicate_rate": 0.1,
    "metadata_completeness": 0.9,
    "average_topic_overlap": 0.65,
    "provider_coverage": 30,
    "total_results": 10
  },
  "openalex": {
    "precision_at_5": 0.6,
    "duplicate_rate": 0.05,
    "metadata_completeness": 0.85,
    "average_topic_overlap": 0.55,
    "provider_coverage": 30,
    "total_results": 10
  },
  "comparison": {
    "better_precision": "crossref",
    "better_metadata": "crossref",
    "lower_duplicates": "openalex"
  }
}
```

---

## Implementation Plan

### Runner Script Design
**File:** `evaluation/run_provider_comparison.py`

**Key components:**
1. **Command-line arguments:**
   - `--paper`: Paper filename (default: attention_is_all_you_need)
   - `--max-results`: Max results to return (default: 10)
   - `--output-dir`: Output directory (default: evaluation/results)

2. **Paper analysis source:**
   - Option A: Use existing validation result from `validation/step5/runs/20261029_review_flag_check/results/attention_is_all_you_need_analysis.json`
   - Option B: Extract minimal fields from PDF directly (no Groq)
   - **Recommendation:** Option B for minimal dependencies

3. **Provider selection:**
   - Create Crossref client directly
   - Create OpenAlex client directly
   - No environment variable modification

4. **Result capture:**
   - Use `max_results * 3` to capture candidate count for duplicate rate
   - Save both raw candidate count and final deduplicated results
   - Preserve all metadata fields

5. **Output:**
   - JSON files for each provider
   - Relevance labeling template
   - Human-readable summary

### Duplicate Rate Calculation
**Current limitation:** Existing providers don't expose raw candidates before deduplication.

**Workaround:**
- Use `candidate_count = max_results * 3` (what providers fetch)
- Use `deduplicated_count = len(results)` (what providers return)
- Calculate `duplicate_rate = (candidate_count - deduplicated_count) / candidate_count`

**Note:** This is an approximation but sufficient for evaluation purposes.

---

## Safety Considerations

### API Usage
- **Crossref:** Free, no rate limit for typical use
- **OpenAlex:** Free tier is $1/day = 1,000 searches or 10,000 list requests
- **Safety:** Evaluation uses 2 searches (1 per provider), well within free tier

### Data Privacy
- Only sends title and research fields to providers (no full PDF text)
- Results are stored locally in `evaluation/results/`
- No personal data transmitted

### Overwrite Protection
- Use timestamped output directories: `evaluation/results/20261029_comparison/`
- Do not overwrite historical validation artifacts in `validation/step5/runs/`

### Error Handling
- Graceful failure if one provider fails
- Continue with available results
- Report errors in output

---

## Comparison Reporting

### Metrics to Calculate
1. **Precision@5:** (relevant papers in top 5) / 5
2. **Duplicate rate:** (candidate_count - deduplicated_count) / candidate_count
3. **Metadata completeness:** Average completeness across 6 fields
4. **Average topic overlap:** Average topic_overlap_score
5. **Provider coverage:** Number of candidates fetched

### Comparison Summary
- Side-by-side table of metrics
- Which provider has better precision
- Which provider has better metadata
- Which provider has lower duplicates
- Qualitative notes on result quality

### Important Distinction
- Report will clearly state: "Results from live API calls on 2026-10-29"
- Distinguish from mock-test results in Phase 4A
- Do not generalize from single evaluation

---

## Remaining Limitations

1. **Single evaluation:** Results from one paper are not generalizable
2. **Manual labeling:** Precision@5 requires human judgment
3. **Approximate duplicate rate:** Cannot capture exact duplicates without modifying providers
4. **No temporal comparison:** Evaluation is a snapshot; provider quality may change over time
5. **No cost analysis:** API costs not tracked (but negligible for single evaluation)
6. **No semantic search comparison:** Both providers use keyword search only

---

## Conclusion

**Implementation needed:** Small evaluation runner script (~200 lines)

**Files to create:**
- `evaluation/run_provider_comparison.py`
- `evaluation/__init__.py`
- `evaluation/results/` (directory, not in version control)

**Files to use (no modification):**
- Existing providers, evaluation module, PDF extraction

**Execution:**
1. Run script to fetch results from both providers
2. Manually label relevance for top 5 results
3. Calculate metrics using existing evaluation module
4. Generate comparison report

**Safety:**
- No production code changes
- No historical artifact modification
- Separate output directory
- Well within free API limits

**Phase 5 investigation complete. Stopping as requested.**
