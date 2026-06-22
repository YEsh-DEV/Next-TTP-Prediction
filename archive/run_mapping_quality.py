import os
import sys
import chromadb
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from pipeline.xml_parser import parse_xml_file

def run_mapping_eval():
    chroma_dir = os.path.join(base_dir, "chroma_db")
    db_client = chromadb.PersistentClient(path=chroma_dir)
    cti_collection = db_client.get_collection(name="cti_events")
    mitre_collection = db_client.get_collection(name="mitre_techniques")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    xml_path = os.path.join(base_dir, "CTI_Report_Dataset", "CTIDataset_2018_ReportEvent.xml")
    all_events = {f"evt_{e.event_id}": e for e in parse_xml_file(xml_path)}
    
    from dotenv import load_dotenv
    load_dotenv(os.path.join(base_dir, ".env"))
    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI"), 
        auth=(os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ.get("NEO4J_PASSWORD"))
    )
    
    queries = ["Burning Umbrella", "Operation Kitty", "Lazarus Cryptocurrency"]
    
    report_path = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\mapping_quality_report.md"
    
    # Extract Neo4j Technique details
    tech_data = {}
    with driver.session() as session:
        res = session.run("MATCH (t:Technique) RETURN t.technique_id AS tid, t.name AS name, t.description AS desc")
        for r in res:
            tech_data[r['tid']] = {"name": r['name'], "desc": r['desc']}
            
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# TASK 1 & 2 — MAPPING QUALITY REPORT\n\n")
        f.write("Validating whether the mathematically ranked offline vectors actually represent semantically meaningful MITRE Techniques.\n\n")
        
        for q in queries:
            f.write(f"## Target Query: `{q}`\n\n")
            q_emb = model.encode([q], show_progress_bar=False).tolist()[0]
            
            cti_res = cti_collection.query(query_embeddings=[q_emb], n_results=3, include=["metadatas"])
            retrieved_eids = cti_res['ids'][0]
            
            f.write("### 1. Retrieved CTI Events\n")
            for eid in retrieved_eids:
                evt = all_events.get(eid)
                title = evt.info if evt else "Unknown"
                f.write(f"- **{eid}**: {title}\n")
            
            f.write("\n### 2. Top 10 Technique Candidates (per Event)\n")
            for eid in retrieved_eids:
                evt = all_events.get(eid)
                if not evt: continue
                
                categories = list(set([a.category for a in evt.attributes if a.category]))
                types = list(set([a.type for a in evt.attributes if a.type]))
                text = f"Observed Categories: {categories}\nObserved Types: {types}"
                
                evt_emb = model.encode([text], show_progress_bar=False).tolist()[0]
                m_res = mitre_collection.query(query_embeddings=[evt_emb], n_results=10, include=["metadatas", "distances"])
                
                cands = m_res['ids'][0]
                dists = m_res['distances'][0]
                metas = m_res['metadatas'][0]
                
                f.write(f"#### Candidates for {eid}\n")
                f.write("| Rank | Technique ID | Score | Name | Description Snippet (from Neo4j) |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- |\n")
                
                for i in range(10):
                    tid = cands[i]
                    dist = dists[i]
                    score = max(0.0, 1.0 - (dist / 2.0))
                    
                    name = metas[i].get('name', tech_data.get(tid, {}).get('name', 'Unknown'))
                    desc = tech_data.get(tid, {}).get('desc') or "None"
                    
                    short_desc = (desc[:80] + "...") if len(desc) > 80 else desc
                    short_desc = short_desc.replace("\n", " ").replace("|", " ")
                    
                    f.write(f"| {i+1} | `{tid}` | {score:.2f} | {name} | {short_desc} |\n")
                f.write("\n")
            f.write("---\n\n")

    print(f"Proof saved to {report_path}")

if __name__ == "__main__":
    run_mapping_eval()
