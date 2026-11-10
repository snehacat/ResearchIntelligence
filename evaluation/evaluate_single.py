"""
Quick script to evaluate a single analysis against gold set.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.gold_set_schema import load_annotation
from evaluation.metrics import evaluate_paper

# Load gold set
gold_path = Path('evaluation/gold_set/attention_is_all_you_need.json')
annotation = load_annotation(gold_path)

# Load existing analysis
analysis_path = Path('validation/step5/runs/20261029_review_flag_check/results/attention_is_all_you_need_result.json')
with open(analysis_path, encoding='utf-8') as f:
    result = json.load(f)
analysis = result['analysis']

# Evaluate
paper_result = evaluate_paper(annotation, analysis)

print("=" * 60)
print(f"Evaluation for: {paper_result.document_id}")
print("=" * 60)
print(f"\nMean Token-F1: {paper_result.mean_token_f1:.4f}")
mean_set_f1 = paper_result.mean_set_f1
print(f"Mean Set-F1: {f'{mean_set_f1:.4f}' if mean_set_f1 is not None else 'N/A'}")
traceability = paper_result.citation_traceability_rate
print(f"Citation Traceability Rate: {f'{traceability:.4f}' if traceability is not None else 'N/A'}")
hit1 = paper_result.hit_rate_at_k(1)
print(f"Hit Rate @1: {f'{hit1:.4f}' if hit1 is not None else 'N/A'}")
hit3 = paper_result.hit_rate_at_k(3)
print(f"Hit Rate @3: {f'{hit3:.4f}' if hit3 is not None else 'N/A'}")
hit5 = paper_result.hit_rate_at_k(5)
print(f"Hit Rate @5: {f'{hit5:.4f}' if hit5 is not None else 'N/A'}")
mrr = paper_result.mrr
print(f"MRR: {f'{mrr:.4f}' if mrr is not None else 'N/A'}")

print("\n" + "=" * 60)
print("Per-Field Breakdown")
print("=" * 60)
for fr in paper_result.field_results:
    print(f"\n{fr.field_name}:")
    print(f"  Gold null: {fr.gold_is_null}")
    print(f"  Predicted null: {fr.predicted_is_null}")
    print(f"  Correct null: {fr.correct_null}")
    if fr.is_set_list_field:
        print(f"  Set Precision: {fr.set_precision:.4f}")
        print(f"  Set Recall: {fr.set_recall:.4f}")
        print(f"  Set F1: {fr.set_f1:.4f}")
    else:
        print(f"  Token-F1: {fr.token_f1:.4f}")
    print(f"  Citation traceable: {fr.citation_traceable}")
    print(f"  Annotated pages: {fr.annotated_pages}")
    print(f"  Retrieved pages: {fr.retrieved_pages}")
