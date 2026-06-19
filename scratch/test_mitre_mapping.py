import os
import sys
import json
import pprint

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from pipeline.xml_parser import parse_xml_file
from pipeline.mitre_mapper import MitreMapper

def test_mapper():
    target_file = os.path.join(BASE_DIR, "CTI_Report_Dataset", "CTIDataset_2018_ReportEvent.xml")
    
    print("Parsing XML dataset...")
    events = parse_xml_file(target_file)
    
    # Find Event 218
    target_event = next((e for e in events if e.event_id == 218), None)
    if not target_event:
        print("Event 218 not found!")
        return
        
    print(f"Loaded Event 218 with {len(target_event.attributes)} attributes.")
    
    print("Initializing MitreMapper...")
    mapper = MitreMapper(BASE_DIR)
    
    print("Mapping event to ExtractionResult graph...")
    result = mapper.map_event(target_event)
    
    print("\n--- Mapping Results ---")
    print(f"Entities Created: {len(result.entities)}")
    print(f"Events Created: {len(result.events)}")
    print(f"Relationships Created: {len(result.relationships)}")
    
    # Let's see how many of each entity type
    type_counts = {}
    for e in result.entities:
        t = e.type.value
        type_counts[t] = type_counts.get(t, 0) + 1
        
    print("\nEntity Types:")
    for t, count in type_counts.items():
        print(f"  {t}: {count}")
        
    print("\nSample JSON output (first 1500 chars):")
    json_str = result.model_dump_json(indent=2)
    print(json_str[:1500])
    
    output_path = os.path.join(BASE_DIR, "scratch", "mapped_event_218.json")
    with open(output_path, "w") as f:
        f.write(json_str)
    print(f"\nSaved full mapped JSON to {output_path}")

if __name__ == "__main__":
    test_mapper()
