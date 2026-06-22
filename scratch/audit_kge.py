import os
import sys
import pandas as pd
import torch
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

report_dir = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d"

def audit_kge():
    print("Training KGE for Audit...")
    triples_path = os.path.join(base_dir, "scratch", "neo4j_triples.tsv")
    tf = TriplesFactory.from_path(triples_path)
    
    # Increase epochs to ensure it actually learns something
    res = pipeline(
        training=tf,
        testing=tf,
        model="TransE",
        training_kwargs=dict(num_epochs=100),
        random_seed=42,
    )
    
    model = res.model
    entity_to_id = tf.entity_to_id
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    relation_id = tf.relation_to_id['USES']
    
    test_df = pd.read_csv(os.path.join(report_dir, "benchmark_test.csv"))
    triples_df = pd.read_csv(triples_path, sep='\t')
    
    audit_lines = ["# KGE AUDIT REPORT\n"]
    
    for idx, row in test_df.iterrows():
        if idx >= 50:
            break
            
        curr_t = row['current_technique']
        actual_t = row['actual_next_technique']
        
        audit_lines.append(f"### Test Sample {idx+1}")
        audit_lines.append(f"**Input Node:** {curr_t}")
        audit_lines.append(f"**Ground Truth Node:** {actual_t}")
        
        # In our graph, the structure is APTGroup -> USES -> Technique.
        # Thus, parent = APTGroup where tail == curr_t.
        parents = triples_df[(triples_df['tail'] == curr_t) & (triples_df['relation'] == 'USES')]['head'].unique()
        parent_ids = [entity_to_id.get(p) for p in parents if p in entity_to_id]
        
        if not parent_ids:
            audit_lines.append("**Predictions:** None (No structural parents in graph)\n")
            continue
            
        all_techs = [e for e in entity_to_id.keys() if e.startswith("T1")]
        tech_scores = {t: 0.0 for t in all_techs}
        
        for pid in parent_ids:
            with torch.no_grad():
                h = torch.tensor([pid] * len(all_techs))
                r = torch.tensor([relation_id] * len(all_techs))
                t = torch.tensor([entity_to_id[tech] for tech in all_techs])
                
                # score_t computes score for specified h, r, t
                scores = model.score_t(torch.stack([h, r, t], dim=1)).squeeze().numpy()
                if scores.ndim > 1:
                    scores = scores.flatten()
                
                for i, tech in enumerate(all_techs):
                    # score mapping based on index
                    tech_scores[tech] += float(scores[i])
                    
        ranked = sorted(tech_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = ranked[:5]
        
        audit_lines.append("**Top-5 Predictions:**")
        for i, (p, score) in enumerate(top_k):
            audit_lines.append(f"{i+1}. {p} (Score: {score:.4f})")
        
        # Verify the true target
        true_score = tech_scores.get(actual_t, "NOT IN GRAPH")
        audit_lines.append(f"\n**Ground Truth Score:** {true_score}")
        audit_lines.append("---\n")
        
    with open(os.path.join(report_dir, "KGE_AUDIT_REPORT.md"), "w") as f:
        f.write("\n".join(audit_lines))
        
    print("Task 2 (KGE Audit) complete.")

if __name__ == "__main__":
    audit_kge()
