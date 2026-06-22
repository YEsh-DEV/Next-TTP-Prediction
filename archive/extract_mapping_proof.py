import os
import sys
import json

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.xml_parser import parse_xml_file
from pipeline.mitre_mapper import MitreMapper

def extract_mapping_proof():
    xml_path = os.path.join(base_dir, "CTI_Report_Dataset", "CTIDataset_2018_ReportEvent.xml")
    events = parse_xml_file(xml_path)
    
    # Top 3 'Burning Umbrella' events
    target_ids = [218, 1295, 964]
    top_3 = [e for e in events if e.event_id in target_ids]
    
    mapper = MitreMapper(base_dir)
    
    output_path = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\submission_mapping_proof.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# TASK 4 — REAL MAPPING PROOF\n\n")
        f.write("Executing the Gemini Hybrid Classifier on the Top 3 retrieved GraphRAG events.\n\n")
        
        for evt in top_3:
            f.write(f"## EVENT: {evt.event_id} | {evt.info}\n")
            f.write("### 1. Retrieved Event\n")
            f.write("```json\n")
            f.write(json.dumps({
                "event_id": evt.event_id,
                "date": evt.date,
                "info": evt.info,
                "attribute_count": len(evt.attributes)
            }, indent=2) + "\n")
            f.write("```\n\n")
            
            f.write("### 2. Candidate Techniques & Confidence\n")
            try:
                matches = mapper.classifier.classify_event(evt)
                f.write("```text\n")
                if not matches:
                    f.write("(No techniques met the 0.65 confidence threshold.)\n")
                else:
                    for m in matches:
                        f.write(f"Predicted Technique: {m.technique_id} | Confidence: {m.confidence:.2f}\n")
                f.write("```\n\n")
            except Exception as e:
                f.write("```text\n")
                f.write(f"API Error: {str(e)}\n")
                f.write("```\n\n")

    print(f"Proof saved to {output_path}")

if __name__ == "__main__":
    extract_mapping_proof()
