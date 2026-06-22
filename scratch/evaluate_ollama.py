import os
import sys
import pandas as pd
import json
import requests
import time

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.query_pipeline import GraphRAGPipeline

report_dir = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d"

OLLAMA_URL = "http://localhost:11434/api/generate"

def get_ollama_model():
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=5)
        res.raise_for_status()
        models = res.json().get("models", [])
        for m in models:
            name = m.get("name", "")
            if "qwen" in name.lower():
                return name
    except Exception as e:
        print(f"Failed to auto-detect model: {e}")
    return "qwen:7b"

def evaluate_ollama():
    MODEL_NAME = get_ollama_model()
    print(f"Using Ollama Model: {MODEL_NAME}")
    print("Loading datasets for Ollama evaluation...")
    pipeline = GraphRAGPipeline(base_dir)
    test_df = pd.read_csv(os.path.join(report_dir, "benchmark_test.csv"))
    
    mitre_df = pd.read_excel(os.path.join(base_dir, "MitreEnterprise.xlsx"))
    mitre_info = {}
    for _, row in mitre_df.iterrows():
        mitre_info[str(row['Tactic ID'])] = {"name": str(row['Tactic Name']), "desc": str(row['Description'])}
        
    hits_1, hits_3, mrr_sum = 0, 0, 0
    tp_dict, fp_dict, fn_dict = {}, {}, {}
    
    import re
    
    audit_lines = ["# LLM AUDIT REPORT\n"]
    
    for idx, row in test_df.iterrows():
        curr_t = row['current_technique']
        actual_t = row['actual_next_technique']
        evt_id = row['source_event']
        
        evt = pipeline.all_events.get(evt_id)
        narrative = evt.narrative if evt else "No context available."
        
        t_info = mitre_info.get(curr_t, {"name": "Unknown", "desc": "Unknown"})
        
        prompt = f"""You are an advanced Cyber Threat Intelligence expert.
The attacker's current state is {curr_t} ({t_info['name']}).
Description of current state: {t_info['desc']}

The context of the current observed event is:
{narrative}

Predict the top 5 most likely next MITRE ATT&CK techniques the attacker will logically progress to.
Output exactly 5 MITRE technique IDs (e.g., T1059) inside a JSON array.
"""
        
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        }
        
        raw_text = ""
        preds = []
        try:
            print(f"Calling Ollama for Row {idx+1}...", flush=True)
            res = requests.post(OLLAMA_URL, json=payload, timeout=120)
            res.raise_for_status()
            
            resp_json = res.json()
            raw_text = resp_json.get("response", "").strip()
            
            # Robust extraction of Technique IDs
            matches = re.findall(r'T\d{4}(?:\.\d{3})?', raw_text)
            
            # Remove duplicates, keep order
            seen = set()
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    preds.append(m)
                    
        except Exception as e:
            print(f"Error for transition {idx}: {e}", flush=True)
            preds = []
            
        audit_lines.append(f"### Sample {idx+1}")
        audit_lines.append(f"**Current Technique:** {curr_t}")
        audit_lines.append(f"**Ground Truth Next Technique:** {actual_t}\n")
        audit_lines.append("**Prompt Sent To LLM:**\n```text\n" + prompt + "\n```\n")
        audit_lines.append("**Raw Response:**\n```text\n" + raw_text + "\n```\n")
        audit_lines.append("**Extracted Technique IDs:** " + str(matches if 'matches' in locals() else []))
        audit_lines.append("**Final Parsed Prediction:** " + str(preds))
        audit_lines.append("---\n")
        
        print(f"Row {idx+1}/{len(test_df)} | Actual: {actual_t} | Preds: {preds}", flush=True)
        
        if len(preds) > 0:
            top1 = preds[0]
            if actual_t == top1:
                hits_1 += 1
                tp_dict[actual_t] = tp_dict.get(actual_t, 0) + 1
            else:
                fn_dict[actual_t] = fn_dict.get(actual_t, 0) + 1
                fp_dict[top1] = fp_dict.get(top1, 0) + 1
                
            if actual_t in preds[:3]:
                hits_3 += 1
                
            if actual_t in preds:
                rank = preds.index(actual_t) + 1
                mrr_sum += (1.0 / rank)
        else:
            fn_dict[actual_t] = fn_dict.get(actual_t, 0) + 1
            
    total_t = len(test_df)
    mrr = mrr_sum / total_t
    hit1_pct = (hits_1 / total_t) * 100
    hit3_pct = (hits_3 / total_t) * 100
    
    with open(os.path.join(report_dir, "LLM_AUDIT_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(audit_lines))
    
    import numpy as np
    precisions = []
    recalls = []
    
    all_classes = set(test_df['actual_next_technique']).union(set(test_df['current_technique']))
    for c in all_classes:
        tp = tp_dict.get(c, 0)
        fp = fp_dict.get(c, 0)
        fn = fn_dict.get(c, 0)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        
    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f1 = 2 * (macro_p * macro_r) / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0
    
    print(f"Ollama {MODEL_NAME} Evaluation - Hits@1: {hit1_pct:.2f}% | Hits@3: {hit3_pct:.2f}% | MRR: {mrr:.4f} | F1: {macro_f1:.4f}")
    
    with open(os.path.join(report_dir, "LLM_BASELINE_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("# LLM BASELINE EVALUATION REPORT (Ollama)\n\n")
        f.write("## Methodology\n")
        f.write(f"Used local Ollama (`{MODEL_NAME}`) to evaluate the identical Test split. Input prompt included the current MITRE technique ID, technique description, and the retrieved narrative context of the source event.\n\n")
        f.write("## Results\n")
        f.write(f"- **Hits@1:** {hit1_pct:.2f}%\n")
        f.write(f"- **Hits@3:** {hit3_pct:.2f}%\n")
        f.write(f"- **MRR:** {mrr:.4f}\n")
        f.write(f"- **Precision:** {macro_p:.4f}\n")
        f.write(f"- **Recall:** {macro_r:.4f}\n")
        f.write(f"- **F1:** {macro_f1:.4f}\n\n")

if __name__ == "__main__":
    evaluate_ollama()
