import os
import sys
import json

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from pipeline.query_pipeline import GraphRAGPipeline

def run_task_d():
    pipeline = GraphRAGPipeline(base_dir)
    query = "Burning Umbrella"
    
    print("=================================================")
    print("TASK D — REAL EXECUTION PROOF")
    print(f"Query: {query}")
    print("Mode: deterministic")
    print("=================================================")
    
    # 1. Retrieved Event IDs
    q_emb = pipeline.model.encode([query], show_progress_bar=False).tolist()[0]
    retrieval_res = pipeline.cti_collection.query(query_embeddings=[q_emb], n_results=3, include=["metadatas"])
    retrieved_events = retrieval_res['ids'][0]
    print("\n1. Retrieved Event IDs:")
    print(retrieved_events)
    
    # 2 & 3. Deterministic Mapping
    print("\n2. Retrieved MITRE Candidate IDs & 3. Final Top 3 Techniques (Per Event):")
    final_techniques = set()
    for eid in retrieved_events:
        evt = pipeline.all_events.get(eid)
        if evt:
            det_result = pipeline.deterministic_classifier.classify_event(evt)
            print(f"  - Event {eid} Top 3: {det_result['techniques']}")
            for t in det_result['techniques']:
                final_techniques.add(t['id'])
    
    final_techniques = list(final_techniques)
    print(f"  -> Combined Unique Final Techniques: {final_techniques}")
    
    # 4. Neo4j Traversal Results
    print("\n4. Neo4j Traversal Results:")
    related_soft = set()
    related_apts = set()
    with pipeline.neo4j_driver.session() as session:
        for t in final_techniques:
            s_res = session.run("MATCH (s:Software)-[:USES]->(t:Technique {technique_id: $tid}) RETURN s.software_id AS soft", tid=t)
            for r in s_res: related_soft.add(r['soft'])
            a_res = session.run("MATCH (a:APTGroup)-[:USES]->(t:Technique {technique_id: $tid}) RETURN a.name AS apt", tid=t)
            for r in a_res: related_apts.add(r['apt'])
            a2_res = session.run("MATCH (a:APTGroup)-[:USES]->(s:Software)-[:USES]->(t:Technique {technique_id: $tid}) RETURN a.name AS apt", tid=t)
            for r in a2_res: related_apts.add(r['apt'])
            
    print(f"  - Related Software: {list(related_soft)}")
    print(f"  - Related APT Groups: {list(related_apts)}")
    
    # 5. Returned JSON
    print("\n5. Returned JSON:")
    final_json = pipeline.execute_query(query, top_k=3, classification_mode="deterministic")
    print(json.dumps(final_json, indent=4))

if __name__ == "__main__":
    run_task_d()
