import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.xml_parser import parse_xml_file
from pipeline.mitre_mapper import MitreMapper

def run_mapping_validation():
    print("--- TASK D: MAPPING VALIDATION ---")
    xml_path = os.path.join(base_dir, "CTI_Report_Dataset", "CTIDataset_2018_ReportEvent.xml")
    events = parse_xml_file(xml_path)
    
    # Retrieve the Top 3 events from the 'Burning Umbrella' query
    target_ids = [218, 1295, 964]
    top_3_events = [e for e in events if e.event_id in target_ids]
    
    mapper = MitreMapper(base_dir)
    
    for evt in top_3_events:
        print(f"\n======================================")
        print(f"EVENT: {evt.event_id} | {evt.info}")
        print(f"======================================")
        
        # We manually call the classifier to see the raw output
        print("-> Retrieving Candidate Techniques from ChromaDB...")
        matches = mapper.classifier.classify_event(evt)
        
        print("\n-> Gemini Predicted Techniques:")
        if not matches:
            print("  (No techniques met confidence threshold / API Rate Limit hit)")
        for m in matches:
            print(f"  [+] {m.technique_id} (Confidence: {m.confidence:.2f})")
            
            import pandas as pd
            attack_df = pd.read_excel(os.path.join(base_dir, 'attackmitre.xlsx'))
            
            # Find APT Groups
            candidate_apts = []
            if 'Group Techniques' in attack_df.columns:
                for idx, row in attack_df.dropna(subset=['Group Techniques']).iterrows():
                    if m.technique_id in str(row['Group Techniques']):
                        candidate_apts.append(row['APT Group Name'])
            
            # Find Software
            candidate_soft = []
            if 'Software Techniques' in attack_df.columns:
                for idx, row in attack_df.dropna(subset=['Software Techniques']).iterrows():
                    if m.technique_id in str(row['Software Techniques']):
                        candidate_soft.append(row['Software ID'])
                        
            print(f"      -> Candidate APT Groups (Graph Path): {list(set(candidate_apts))[:3]}...")
            print(f"      -> Candidate Software (Graph Path): {list(set(candidate_soft))[:3]}...")
            print("-" * 40)
            
if __name__ == "__main__":
    run_mapping_validation()
