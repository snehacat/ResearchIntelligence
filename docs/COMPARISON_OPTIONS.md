# Comparison Options Design Note

This document compares three alternative approaches for paper comparison and related-paper screening beyond the current exact string matching implementation.

## Current Implementation

The current `literature_comparator` uses **exact normalized string matching**:
- Fields are normalized (lowercase, whitespace-trimmed, punctuation stripped)
- Comparison requires verbatim match after normalization
- Related-paper screening uses exact title-term overlap

**Advantages:**
- Deterministic and reproducible
- No additional dependencies
- Conservative (minimizes false positives)
- Fast to compute

**Limitations:**
- Misses semantically equivalent but differently worded content
- Misses paraphrases and synonyms
- Requires verbatim matches which are rare in free-text LLM output
- Title-term overlap is not semantic relevance

---

## Option 1: Embedding Similarity with Threshold

### Description

Compute dense vector embeddings for field values (e.g., using sentence-transformers) and compare using cosine similarity. Values above a threshold are considered matches.

### Cost

**Compute:**
- Requires embedding model (e.g., sentence-transformers, 100-500MB)
- Inference time: ~10-50ms per comparison on CPU
- Memory: Model loaded in memory (~500MB - 2GB depending on model)

**Development:**
- Implementation complexity: Low to medium
- Model selection and threshold tuning required
- Existing embedding infrastructure can be reused

### Failure Modes

**False Positives:**
- Semantically similar but factually different claims (e.g., "accuracy 95%" vs "accuracy 90%")
- Domain-related but irrelevant matches
- Threshold too low -> many spurious matches

**False Negatives:**
- Semantically dissimilar but actually equivalent claims
- Threshold too high -> misses valid matches
- Model bias toward certain domains/languages

**Model Limitations:**
- Embedding models trained on general text may not capture technical nuance
- Short phrases (names, metrics) may not embed meaningfully
- Context loss: embeddings don't capture negation or conditions well

### Validation Approach

**Required: Labeled matched/unmatched pairs**

1. **Create a validation set** of field pairs:
   - Manually label pairs as "match" or "no match"
   - Cover all field types (title, methodology, dataset, etc.)
   - Include edge cases (similar but different, different but equivalent)

2. **Threshold tuning:**
   - Compute cosine similarity for all labeled pairs
   - Choose threshold that maximizes F1 on validation set
   - Report precision, recall, F1 at chosen threshold

3. **Cross-validation:**
   - Split labeled pairs into train/validation/test
   - Tune threshold on validation set
   - Report final metrics on held-out test set

4. **Per-field analysis:**
   - Different fields may require different thresholds
   - Report metrics per field type

**Failure to validate:** Without labeled pairs, threshold is arbitrary and performance is unknown.

---

## Option 2: LLM Judge

### Description

Use an LLM to determine whether two field values are semantically equivalent. The LLM is prompted with both values and asked to judge match/no-match.

### Cost

**Compute:**
- Requires LLM API calls (e.g., Groq, OpenAI) or local model
- Inference time: ~500ms - 5s per comparison (depending on model)
- API costs: $0.01 - $0.10 per 1000 comparisons (varies by provider)

**Development:**
- Implementation complexity: Medium
- Prompt engineering required
- Error handling for API failures
- Rate limiting considerations

### Failure Modes

**False Positives:**
- LLM hallucination (claims match when they don't)
- Prompt injection (maliciously crafted values)
- Model bias toward saying "yes"

**False Negatives:**
- LLM conservatism (claims no match when there is one)
- Context window limitations for long field values
- Model unfamiliarity with technical domain

**API Limitations:**
- Rate limits prevent batch processing
- API downtime blocks comparison
- Cost scaling with number of comparisons

**Prompt Sensitivity:**
- Small prompt changes change behavior
- Different models behave differently
- Hard to ensure consistency across runs

### Validation Approach

**Required: Labeled matched/unmatched pairs**

1. **Create a validation set** (same as Option 1)

2. **Prompt development:**
   - Develop prompts with few-shot examples
   - Test prompts on validation set
   - Measure precision, recall, F1

3. **Model comparison:**
   - Test multiple LLMs (e.g., Groq gpt-oss-20b, GPT-4, Claude)
   - Compare cost, latency, and accuracy
   - Select best model for use case

4. **Consistency testing:**
   - Run same pair multiple times
   - Measure inter-rater agreement (LLM should be consistent)
   - Report consistency metrics

5. **Error analysis:**
   - Manually review misclassified pairs
   - Identify patterns in failures
   - Refine prompts based on analysis

**Failure to validate:** Without labeled pairs, cannot measure accuracy or detect systematic biases.

---

## Option 3: Hybrid Approach

### Description

Combine multiple methods in a pipeline:
- **Stage 1:** Fast exact match (current implementation) - returns known matches
- **Stage 2:** Embedding similarity with high threshold - returns high-confidence matches
- **Stage 3:** LLM judge for ambiguous cases - resolves borderline cases

### Cost

**Compute:**
- Combines costs of all methods
- But only uses expensive methods when necessary
- Overall cost depends on distribution of cases

**Development:**
- Implementation complexity: High
- Requires coordinating multiple systems
- Threshold tuning for embedding stage
- Prompt engineering for LLM stage
- Fallback logic between stages

### Failure Modes

**All failure modes from individual methods apply**, plus:

**Pipeline errors:**
- Bugs in stage transitions
- Incorrect fallback logic
- Performance degradation if many cases reach expensive stages

**Threshold interaction:**
- Embedding threshold too high -> too many cases go to LLM (expensive)
- Embedding threshold too low -> false positives from embedding stage
- Requires careful balancing

**Complexity:**
- Harder to debug when something goes wrong
- More moving parts to maintain
- Difficult to reason about overall behavior

### Validation Approach

**Required: Labeled matched/unmatched pairs**

1. **Create a validation set** (same as Options 1 and 2)

2. **Stage-wise validation:**
   - Validate exact match stage (should be perfect)
   - Validate embedding stage independently
   - Validate LLM stage independently

3. **End-to-end validation:**
   - Run full pipeline on validation set
   - Measure overall precision, recall, F1
   - Report per-stage contribution

4. **Cost-benefit analysis:**
   - Measure what fraction of cases reach each stage
   - Report cost per comparison
   - Compare accuracy vs cost trade-off

5. **Ablation study:**
   - Test with and without each stage
   - Measure contribution of each stage to overall accuracy

**Failure to validate:** Without labeled pairs, cannot optimize thresholds or validate that pipeline adds value over simpler methods.

---

## Recommendation

**Do not implement any of these options without first creating a labeled validation set.**

The current exact matching approach is conservative and deterministic. Before replacing it with any of the alternatives:

1. **Create a labeled validation set** of at least 100 field-value pairs across all field types
2. **Measure baseline performance** of exact matching on this set
3. **Validate any alternative** against the same set with labeled ground truth
4. **Report metrics** (precision, recall, F1) for each method
5. **Justify the trade-off** between accuracy improvement and added complexity/cost

If labeled pairs show that exact matching has unacceptable false negative rates (missing valid matches), then:

- **Start with Option 1 (embedding similarity)** as the simplest alternative
- Validate with labeled pairs before deploying
- Only consider Option 2 (LLM judge) or Option 3 (hybrid) if Option 1 is insufficient

**Key principle:** Any semantic comparison method must be validated against labeled pairs to ensure it actually improves over the current conservative approach.
