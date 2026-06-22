import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from pipeline.xml_parser import parse_xml_file
from pipeline.hybrid_classifier import HybridClassifier
from sentence_transformers import SentenceTransformer

def run_revalidation():
    xml_path = os.path.join(base_dir, "CTI_Report_Dataset", "CTIDataset_2018_ReportEvent.xml")
    events = parse_xml_file(xml_path)
    evt_218 = next((e for e in events if e.event_id == 218), None)
    
    classifier = HybridClassifier(base_dir)
    
    categories = list(set([attr.category for attr in evt_218.attributes if attr.category]))
    types = list(set([attr.type for attr in evt_218.attributes if attr.type]))
    text = f"Observed Categories: {categories}\nObserved Types: {types}"
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    event_embedding = model.encode([text], show_progress_bar=False).tolist()[0]
    
    # 1. Top 20 Candidates
    results = classifier.collection.query(
        query_embeddings=[event_embedding],
        n_results=20,
        include=["metadatas", "distances"]
    )
    
    # 2. Run HybridClassifier
    print("Executing Gemini Classification...")
    matches = classifier.classify_event(evt_218)
    
    output_path = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\event_218_final_validation.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# TASK 2 — REAL EVENT 218 REVALIDATION\n\n")
        
        f.write("## 1. Event Metadata\n")
        f.write(f"- **Event ID:** {evt_218.event_id}\n")
        f.write(f"- **Date:** {evt_218.date}\n")
        f.write(f"- **Info:** {evt_218.info}\n")
        f.write(f"- **Attribute Count:** {len(evt_218.attributes)}\n\n")
        
        f.write("## 2. Top 20 Retrieved MITRE Candidates\n")
        f.write("| Rank | Candidate ID | Score (L2) | Candidate Name |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        for i in range(len(results['ids'][0])):
            tid = results['ids'][0][i]
            dist = results['distances'][0][i]
            name = results['metadatas'][0][i].get('name', 'Unknown')
            f.write(f"| {i+1} | `{tid}` | {dist:.4f} | {name} |\n")
        f.write("\n")
        
        f.write("## 3. EXACT RAW OUTPUT (HybridClassifier)\n")
        f.write("```json\n")
        if not matches:
            f.write("[] // (API returned empty or failed to meet 0.65 threshold)\n")
        else:
            output = [{"technique_id": m.technique_id, "confidence": m.confidence} for m in matches]
            import json
            f.write(json.dumps(output, indent=2) + "\n")
        f.write("```\n")

    print(f"Proof saved to {output_path}")

if __name__ == "__main__":
    run_revalidation()
