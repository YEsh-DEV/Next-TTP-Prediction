import os
import sys
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

report_dir = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d"

def export_neo4j_triples():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(base_dir, ".env"))
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD", "password")
    
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    triples = []
    
    with driver.session() as session:
        # APTGroup -> USES -> Technique
        res = session.run("MATCH (a:APTGroup)-[:USES]->(t:Technique) RETURN a.name as h, 'USES' as r, t.technique_id as t")
        for rec in res:
            triples.append([rec['h'], rec['r'], rec['t']])
            
        # Software -> USES -> Technique
        res = session.run("MATCH (s:Software)-[:USES]->(t:Technique) RETURN s.software_id as h, 'USES' as r, t.technique_id as t")
        for rec in res:
            triples.append([rec['h'], rec['r'], rec['t']])
            
        # APTGroup -> USES -> Software
        res = session.run("MATCH (a:APTGroup)-[:USES]->(s:Software) RETURN a.name as h, 'USES' as r, s.software_id as t")
        for rec in res:
            triples.append([rec['h'], rec['r'], rec['t']])
            
    driver.close()
    
    if not triples:
        print("Neo4j database is empty or down! Cannot train KGE.")
        return None
        
    df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
    df.to_csv(os.path.join(base_dir, "scratch", "neo4j_triples.tsv"), sep='\t', index=False)
    return os.path.join(base_dir, "scratch", "neo4j_triples.tsv")

def evaluate_kge(model_name: str, triples_path: str):
    print(f"Training {model_name} via PyKEEN...")
    tf = TriplesFactory.from_path(triples_path)
    
    res = pipeline(
        training=tf,
        testing=tf, # We train and test on the static Neo4j graph, but we benchmark on the temporal transitions!
        model=model_name,
        training_kwargs=dict(num_epochs=100), # Minimal for speed
        random_seed=42,
    )
    
    model = res.model
    entity_to_id = tf.entity_to_id
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    relation_id = tf.relation_to_id['USES']
    
    # KGE Inference Strategy:
    # KGEs lack sequential awareness. To predict T_next from T_current:
    # 1. Identify which APTs or Softwares 'use' T_current in the training graph.
    # 2. For those APTs/Softwares, use the KGE model to rank ALL techniques T' for the link (APT, USES, T').
    # 3. Aggregate scores to pick the Top-K structural techniques.
    
    test_df = pd.read_csv(os.path.join(report_dir, "benchmark_test.csv"))
    
    triples_df = pd.read_csv(triples_path, sep='\t')
    
    hits_1, hits_3, mrr_sum = 0, 0, 0
    
    for idx, row in test_df.iterrows():
        curr_t = row['current_technique']
        actual_t = row['actual_next_technique']
        
        # Find structural parents of curr_t
        parents = triples_df[(triples_df['tail'] == curr_t) & (triples_df['relation'] == 'USES')]['head'].unique()
        
        if len(parents) == 0:
            # Dead-end structurally
            continue
            
        # Predict tail scores for (parent, USES, ?)
        parent_ids = [entity_to_id.get(p) for p in parents if p in entity_to_id]
        if not parent_ids:
            continue
            
        all_techs = [e for e in entity_to_id.keys() if e.startswith("T1")]
        tech_scores = {t: 0.0 for t in all_techs}
        
        for pid in parent_ids:
            with torch.no_grad():
                h = torch.tensor([pid] * len(all_techs))
                r = torch.tensor([relation_id] * len(all_techs))
                t = torch.tensor([entity_to_id[tech] for tech in all_techs])
                
                scores = model.score_t(torch.stack([h, r, t], dim=1)).squeeze().numpy()
                
                for tech, score in zip(all_techs, scores):
                    tech_scores[tech] += float(np.sum(score))
                    
        # Sort descending
        ranked = sorted(tech_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = [x[0] for x in ranked[:10]]
        
        if len(top_k) > 0:
            top1 = top_k[0]
            if actual_t == top1:
                hits_1 += 1
            if actual_t in top_k[:3]:
                hits_3 += 1
            if actual_t in top_k:
                mrr_sum += 1.0 / (top_k.index(actual_t) + 1)
                
    total = len(test_df)
    mrr = mrr_sum / total
    hit1_pct = (hits_1 / total) * 100
    hit3_pct = (hits_3 / total) * 100
    
    print(f"{model_name} Evaluation - Hits@1: {hit1_pct:.2f}% | Hits@3: {hit3_pct:.2f}% | MRR: {mrr:.4f}")
    
    with open(os.path.join(report_dir, f"{model_name.lower()}_report.md"), "w", encoding="utf-8") as f:
        f.write(f"# {model_name} EVALUATION REPORT\n\n")
        f.write("## Methodology\n")
        f.write(f"Trained `{model_name}` on the static Neo4j APT->USES->Technique schema. Predicted chronological transitions by aggregating structural affinity across all threat groups associated with the current technique.\n\n")
        f.write("## Results\n")
        f.write(f"- **Hits@1:** {hit1_pct:.2f}%\n")
        f.write(f"- **Hits@3:** {hit3_pct:.2f}%\n")
        f.write(f"- **MRR:** {mrr:.4f}\n")
        
    return hit1_pct, hit3_pct, mrr

if __name__ == "__main__":
    t_path = export_neo4j_triples()
    if t_path:
        evaluate_kge("TransE", t_path)
        evaluate_kge("RotatE", t_path)
