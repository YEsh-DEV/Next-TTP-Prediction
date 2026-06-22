import os
import pandas as pd
from collections import defaultdict

report_dir = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d"

def verify_markov():
    train_df = pd.read_csv(os.path.join(report_dir, "benchmark_train.csv"))
    test_df = pd.read_csv(os.path.join(report_dir, "benchmark_test.csv"))
    
    # Build empirical matrix
    transitions = defaultdict(lambda: defaultdict(int))
    known_states = set()
    
    for _, row in train_df.iterrows():
        c = row['current_technique']
        t = row['actual_next_technique']
        transitions[c][t] += 1
        known_states.add(c)
        known_states.add(t)
        
    audit_lines = ["# MARKOV EVALUATION AUDIT\n"]
    error_lines = ["# ERROR ANALYSIS REPORT\n"]
    
    hits_1, hits_3 = 0, 0
    total = len(test_df)
    
    unseen_state_count = 0
    
    error_counts = {
        "State sparsity (Known state, empty transitions)": 0,
        "Unseen transition (Target state never followed source)": 0,
        "Unseen state (Source state not in training)": 0,
    }
    
    for idx, row in test_df.iterrows():
        c = row['current_technique']
        actual = row['actual_next_technique']
        
        audit_lines.append(f"### Transition {idx+1}")
        audit_lines.append(f"**Current State:** {c}")
        audit_lines.append(f"**Actual Next State:** {actual}\n")
        
        if c not in known_states:
            unseen_state_count += 1
            error_counts["Unseen state (Source state not in training)"] += 1
            preds = []
            reason = "Unseen state (Source state not in training)"
        else:
            if len(transitions[c]) == 0:
                preds = []
                reason = "State sparsity (Known state, empty transitions)"
                error_counts[reason] += 1
            else:
                sorted_t = sorted(transitions[c].items(), key=lambda x: x[1], reverse=True)
                preds = [x[0] for x in sorted_t]
                if actual not in preds:
                    reason = "Unseen transition (Target state never followed source)"
                    error_counts[reason] += 1
                else:
                    reason = "Predicted but not top-1"
        
        audit_lines.append("**Predictions:**")
        if not preds:
            audit_lines.append("No predictions possible (0.0% prob)")
        for i, p in enumerate(preds[:5]):
            audit_lines.append(f"{i+1}. {p}")
            
        c_at_1 = actual in preds[:1]
        c_at_3 = actual in preds[:3]
        
        if c_at_1: hits_1 += 1
        if c_at_3: hits_3 += 1
        
        audit_lines.append(f"\nCorrect@1: {c_at_1} | Correct@3: {c_at_3}\n---\n")
        
        if not c_at_1:
            error_lines.append(f"### Failed Transition {idx+1}")
            error_lines.append(f"**Current:** {c} | **Actual:** {actual} | **Top Pred:** {preds[0] if preds else 'None'}")
            error_lines.append(f"**Reason:** {reason}\n")
            
    with open(os.path.join(report_dir, "MARKOV_EVALUATION_AUDIT.md"), "w") as f:
        f.write("\n".join(audit_lines))
        
    with open(os.path.join(report_dir, "ERROR_ANALYSIS_REPORT.md"), "w") as f:
        f.write("\n".join(error_lines))
        f.write("\n## Error Distribution\n")
        for k, v in error_counts.items():
            f.write(f"- {k}: {v}\n")
            
    # COVERAGE REPORT
    cov_lines = ["# TRANSITION COVERAGE REPORT\n"]
    cov_lines.append(f"**Total Test Transitions:** {total}\n")
    
    known_pct = ((total - unseen_state_count) / total) * 100
    unseen_pct = (unseen_state_count / total) * 100
    
    cov_lines.append(f"- **Transitions with known source state:** {total - unseen_state_count} ({known_pct:.2f}%)")
    cov_lines.append(f"- **Transitions with UNSEEN source state:** {unseen_state_count} ({unseen_pct:.2f}%)")
    
    with open(os.path.join(report_dir, "TRANSITION_COVERAGE_REPORT.md"), "w") as f:
        f.write("\n".join(cov_lines))
        
    print("Tasks 1, 2, 5 complete.")

if __name__ == "__main__":
    verify_markov()
