import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.xml_parser import parse_xml_file
from pipeline.deterministic_classifier import DeterministicClassifier
from pipeline.deterministic_classifier_v2 import DeterministicClassifierV2

def run_comparison():
    xml_path = os.path.join(base_dir, "CTI_Report_Dataset", "CTIDataset_2018_ReportEvent.xml")
    events = parse_xml_file(xml_path)
    
    classifier_v1 = DeterministicClassifier(base_dir)
    classifier_v2 = DeterministicClassifierV2(base_dir)
    
    v1_techs = set()
    v2_techs = set()
    
    print("Executing V1 and V2 across the corpus...")
    
    for e in events:
        res1 = classifier_v1.classify_event(e)
        if res1['techniques']:
            v1_techs.add(res1['techniques'][0]['id'])
            
        res2 = classifier_v2.classify_event(e)
        if res2['techniques']:
            v2_techs.add(res2['techniques'][0]['id'])

    print("\n=================================================")
    print("V1 vs V2 DETERMINISTIC CLASSIFIER COMPARISON")
    print("=================================================")
    print(f"V1 Unique Techniques: {len(v1_techs)}")
    print(f"V2 Unique Techniques: {len(v2_techs)}")
    
    if len(v2_techs) > 20:
        print("VERDICT: V2 SUCCESSFULLY EXPANDED STATE SPACE.")
    else:
        print("VERDICT: V2 FAILED TO EXPAND STATE SPACE.")

if __name__ == "__main__":
    run_comparison()
