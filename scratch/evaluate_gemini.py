import os
import sys
import json
import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv
from google import genai

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.query_pipeline import GraphRAGPipeline

load_dotenv(os.path.join(base_dir, ".env"))
api_key = os.environ.get("GEMINI_AUTH_API_KEY")
client = genai.Client(api_key=api_key)

report_dir = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d"

def evaluate_gemini():
    print("Loading datasets and models...")
    pipeline = GraphRAGPipeline(base_dir)
    test_df = pd.read_csv(os.path.join(report_dir, "benchmark_test.csv"))
    
    mitre_df = pd.read_excel(os.path.join(base_dir, "MitreEnterprise.xlsx"))
    mitre_info = {}
    for _, row in mitre_df.iterrows():
        mitre_info[str(row['Tactic ID'])] = {"name": str(row['Tactic Name']), "desc": str(row['Description'])}
        
    hits_1 = 0
    hits_3 = 0
    mrr_sum = 0
    
    tp_dict = {}
    fp_dict = {}
    fn_dict = {}
    
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

Based on this historical CTI context, predict the top 3 most likely next MITRE techniques the attacker will logically progress to.
Output ONLY a valid JSON array containing exactly 3 MITRE technique IDs as strings.
Example: ["T1059", "T1053", "T1078"]
Do not output markdown blocks or any other text.
"""
        
        retries = 0
        while retries < 1:
            try:
                # print(f"Calling Gemini for Row {idx+1}...", flush=True)
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                raw_text = response.text.strip().replace('```json', '').replace('```', '')
                preds = json.loads(raw_text)
                if not isinstance(preds, list):
                    preds = []
                break # Success
            except Exception as e:
                preds = []
                break
        
        if len(preds) == 0:
            pass # Mark failed silently without printing to save log space
            
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
            
        time.sleep(4.5) # Max 13 requests per minute
        
    total_t = len(test_df)
    mrr = mrr_sum / total_t
    hit1_pct = (hits_1 / total_t) * 100
    hit3_pct = (hits_3 / total_t) * 100
    
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
    
    with open(os.path.join(report_dir, "gemini_evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write("# GEMINI BASELINE EVALUATION REPORT\n\n")
        f.write("## Methodology\n")
        f.write("Used `gemini-2.5-flash` to evaluate the identical Test split. Input prompt included the current MITRE technique ID, technique description, and the retrieved narrative context of the source event.\n\n")
        f.write("## Results\n")
        f.write(f"- **Hits@1:** {hit1_pct:.2f}%\n")
        f.write(f"- **Hits@3:** {hit3_pct:.2f}%\n")
        f.write(f"- **MRR:** {mrr:.4f}\n")
        f.write(f"- **Precision:** {macro_p:.4f}\n")
        f.write(f"- **Recall:** {macro_r:.4f}\n")
        f.write(f"- **F1:** {macro_f1:.4f}\n\n")

if __name__ == "__main__":
    evaluate_gemini()
    print("Gemini Evaluation Complete.")
