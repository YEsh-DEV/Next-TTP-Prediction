import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from pipeline.xml_parser import parse_xml_file
from pipeline.hybrid_classifier import HybridClassifier

def run_retrieval_audit():
    xml_path = os.path.join(base_dir, "CTI_Report_Dataset", "CTIDataset_2018_ReportEvent.xml")
    events = parse_xml_file(xml_path)
    
    evt_218 = next((e for e in events if e.event_id == 218), None)
    if not evt_218:
        print("Event 218 not found!")
        return

    classifier = HybridClassifier(base_dir)
    
    # 1. Format text
    categories = list(set([attr.category for attr in evt_218.attributes if attr.category]))
    types = list(set([attr.type for attr in evt_218.attributes if attr.type]))
    text = f"Observed Categories: {categories}\nObserved Types: {types}"
    
    # 2. Embed
    model = SentenceTransformer('all-MiniLM-L6-v2')
    event_embedding = model.encode([text], show_progress_bar=False).tolist()[0]
    
    # 3. Retrieve
    results = classifier.collection.query(
        query_embeddings=[event_embedding],
        n_results=20,
        include=["metadatas", "distances"]
    )
    
    output_path = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\candidate_retrieval_report.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# STEP 3 — Candidate Retrieval Audit\n\n")
        f.write("## 1. Retrieval Logic Inspection\n")
        f.write("- **Collection Queried:** `mitre_techniques`\n")
        f.write("- **Source of Collection:** `MitreEnterprise.xlsx` (Tactic ID, Tactic Name, Description)\n")
        f.write("- **Pollution Mechanism:** Because `MitreEnterprise.xlsx` contains over 4,600 Software/Actor IDs disguised in the `Tactic ID` column, the RAG candidate pool is massively polluted.\n\n")
        
        f.write("## 2. Event 218 Top 20 Candidates\n")
        f.write("| Rank | Retrieved ID | L2 Distance | Entity Type (Regex) | Name |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        import re
        pattern = re.compile(r"^T[0-9]{4}(\.[0-9]{3})?$")
        
        for i in range(len(results['ids'][0])):
            tid = results['ids'][0][i]
            dist = results['distances'][0][i]
            meta = results['metadatas'][0][i]
            name = meta.get('name', 'Unknown')
            
            entity_type = "Technique" if pattern.match(tid) else ("Software" if tid.startswith("S") else ("ThreatActor" if tid.startswith("G") else "Unknown"))
            
            f.write(f"| {i+1} | `{tid}` | {dist:.4f} | **{entity_type}** | {name} |\n")

    print(f"Proof saved to {output_path}")

if __name__ == "__main__":
    run_retrieval_audit()
