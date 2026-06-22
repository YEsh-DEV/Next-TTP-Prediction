import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.query_pipeline import GraphRAGPipeline
from pipeline.markov_predictor import GlobalMarkovPredictor
from pipeline.deterministic_classifier_v2 import DeterministicClassifierV2

report_dir = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d"

def get_date(e):
    if e.date:
        try: return datetime.strptime(e.date, "%Y-%m-%d")
        except ValueError: pass
    return datetime.min

def run_phase_0():
    print("Extracting full benchmark dataset...")
    pipeline = GraphRAGPipeline(base_dir)
    pipeline.deterministic_classifier = DeterministicClassifierV2(base_dir)
    
    events = [e for e in pipeline.all_events.values() if e.event_type == "ReportEvent"]
    events.sort(key=lambda x: (get_date(x), x.event_id))
    
    transitions = []
    
    for i in range(len(events) - 1):
        curr_e = events[i]
        next_e = events[i+1]
        
        curr_det = pipeline.deterministic_classifier.classify_event(curr_e)
        next_det = pipeline.deterministic_classifier.classify_event(next_e)
        
        if curr_det['techniques'] and next_det['techniques']:
            transitions.append({
                "current_technique": curr_det['techniques'][0]['id'],
                "actual_next_technique": next_det['techniques'][0]['id'],
                "source_event": curr_e.event_id,
                "target_event": next_e.event_id,
                "date_source": curr_e.date,
                "date_target": next_e.date
            })
            
    df = pd.DataFrame(transitions)
    
    # 80/20 Chronological Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    train_df.to_csv(os.path.join(report_dir, "benchmark_train.csv"), index=False)
    test_df.to_csv(os.path.join(report_dir, "benchmark_test.csv"), index=False)
    
    print(f"Total: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)}")
    
    # Monkey-patch Markov Predictor to ONLY use the Train Split
    markov = GlobalMarkovPredictor(base_dir)
    markov.transition_counts.clear()
    markov.transition_probs.clear()
    markov.total_transitions = 0
    
    from collections import defaultdict
    new_counts = defaultdict(lambda: defaultdict(int))
    for _, row in train_df.iterrows():
        new_counts[row['current_technique']][row['actual_next_technique']] += 1
        
    for curr_state, next_states in new_counts.items():
        total = sum(next_states.values())
        for next_state, count in next_states.items():
            markov.transition_probs[curr_state][next_state] = count / total
            
    print("Re-evaluating Markov on Test Split...")
    
    hits_1 = 0
    hits_3 = 0
    mrr_sum = 0
    
    for _, row in test_df.iterrows():
        curr_t = row['current_technique']
        actual = row['actual_next_technique']
        
        preds = markov.top_k_predictions(curr_t, k=10)
        pred_states = [p['state'] for p in preds]
        
        if len(pred_states) == 0:
            continue
            
        top1 = pred_states[0]
        if actual == top1:
            hits_1 += 1
        if actual in pred_states[:3]:
            hits_3 += 1
        if actual in pred_states:
            rank = pred_states.index(actual) + 1
            mrr_sum += (1.0 / rank)
            
    total_t = len(test_df)
    mrr = mrr_sum / total_t
    hit1_pct = (hits_1 / total_t) * 100
    hit3_pct = (hits_3 / total_t) * 100
    
    print(f"Markov (Leakage-Free Test Split) - Hits@1: {hit1_pct:.2f}% | Hits@3: {hit3_pct:.2f}% | MRR: {mrr:.4f}")
    
    with open(os.path.join(report_dir, "evaluation_leakage_report.md"), "w", encoding="utf-8") as f:
        f.write("# EVALUATION LEAKAGE FIX REPORT\n\n")
        f.write("## Issue Identified\n")
        f.write("The previous evaluation iterated over the exact same 100% corpus that `GlobalMarkovPredictor` used to construct its transition matrix, resulting in massive data leakage (Test = Train). This caused an artificially high 45.16% Hits@1 rate because the model had already 'seen' the transition it was evaluating.\n\n")
        f.write("## Mitigation\n")
        f.write("The chronological corpus was split 80/20:\n")
        f.write(f"- **Train Transitions:** {len(train_df)}\n")
        f.write(f"- **Test Transitions:** {len(test_df)}\n\n")
        f.write("The `GlobalMarkovPredictor` transition matrix was overridden to be computed **strictly** from the Train slice. The evaluation was run **strictly** on the Test slice.\n\n")
        f.write("## Unbiased Markov Baseline Results\n")
        f.write("| Metric | Leaked Score (100% Overlap) | Unbiased Score (Test Split) |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| Hits@1 | 45.16% | {hit1_pct:.2f}% |\n")
        f.write(f"| Hits@3 | 74.19% | {hit3_pct:.2f}% |\n")
        f.write(f"| MRR | 0.6141 | {mrr:.4f} |\n\n")
        f.write("This unbiased baseline will be used for all subsequent ML model comparisons.\n")

if __name__ == "__main__":
    run_phase_0()
