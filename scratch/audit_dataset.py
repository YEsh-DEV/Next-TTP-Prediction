import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
import pprint
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from pipeline.xml_parser import parse_xml_file
from pipeline.chunker import CTIChunker

def audit_dataset():
    target_file = os.path.join(BASE_DIR, "CTI_Report_Dataset", "CTIDataset_2018_ReportEvent.xml")
    
    print("--- Phase 1: Global Statistics ---")
    
    events = parse_xml_file(target_file)
    
    total_events = len(events)
    total_attributes = 0
    non_empty_comments = 0
    
    category_counter = Counter()
    type_counter = Counter()
    
    # We want to find a really "heavy" event for the deep dive
    heaviest_event = None
    max_attrs = 0
    
    for event in events:
        num_attrs = len(event.attributes)
        total_attributes += num_attrs
        
        if num_attrs > max_attrs:
            max_attrs = num_attrs
            heaviest_event = event
            
        for attr in event.attributes:
            category_counter[attr.category] += 1
            type_counter[attr.type] += 1
            if attr.comment and attr.comment.strip():
                non_empty_comments += 1
                
    avg_attrs = total_attributes / total_events if total_events > 0 else 0
    comment_pct = (non_empty_comments / total_attributes * 100) if total_attributes > 0 else 0
    
    print(f"Total Events: {total_events}")
    print(f"Average attribute count per event: {avg_attrs:.2f}")
    print(f"Percentage of attributes with non-empty comments: {comment_pct:.2f}%")
    
    print("\nTop 20 Attribute Categories:")
    for cat, count in category_counter.most_common(20):
        print(f"  {cat}: {count}")
        
    print("\nTop 20 Attribute Types:")
    for t, count in type_counter.most_common(20):
        print(f"  {t}: {count}")
        
    print("\n--- Phase 2: Single Event Deep Dive ---")
    if not heaviest_event:
        print("No events found.")
        return
        
    event_id = heaviest_event.event_id
    print(f"Selected Event ID: {event_id} (contains {len(heaviest_event.attributes)} attributes)")
    
    # 1. Raw XML
    tree = ET.parse(target_file)
    root = tree.getroot()
    raw_xml_string = ""
    for child in root:
        if child.find('id') is not None and child.find('id').text == str(event_id):
            raw_xml_string = ET.tostring(child, encoding='unicode')
            break
            
    print("\n1. Raw XML Event (truncated to 1500 chars for console):")
    print(raw_xml_string[:1500] + "\n...[truncated]")
    
    # 2. Parsed CTIEvent Object
    print("\n2. Parsed CTIEvent Object (JSON repr, truncated):")
    parsed_json = json.dumps(heaviest_event.model_dump(), indent=2, default=str)
    print(parsed_json[:1500] + "\n...[truncated]")
    
    # 3-6. Attributes
    categories = list(set(a.category for a in heaviest_event.attributes))
    types = list(set(a.type for a in heaviest_event.attributes))
    values = [a.value for a in heaviest_event.attributes]
    comments = [a.comment for a in heaviest_event.attributes if a.comment]
    
    print(f"\n3. All Attribute Categories in Event: {categories}")
    print(f"\n4. All Attribute Types in Event: {types}")
    print(f"\n5. Attribute Values (first 20): {values[:20]}")
    print(f"\n6. Attribute Comments (first 20): {comments[:20]}")
    
    # 7. Narrative Chunk
    chunker = CTIChunker()
    chunks = chunker.chunk_event(heaviest_event)
    
    print(f"\n7. Generated Narrative Chunk(s) count: {len(chunks)}")
    narrative_chunk_text = ""
    for c in chunks:
        if c['chunk_type'] == 'narrative':
            narrative_chunk_text = c['text']
            print("\nNarrative Chunk Preview:")
            print(narrative_chunk_text[:1000])
            break
            
    if not narrative_chunk_text and chunks:
        print("\nNo narrative chunk found! Preview of first chunk:")
        narrative_chunk_text = chunks[0]['text']
        print(narrative_chunk_text[:1000])
        
    # 8. Report
    print("\n--- 8. Data Loss Report ---")
    print(f"Raw XML character length: {len(raw_xml_string)}")
    print(f"Parsed Event character length: {len(parsed_json)}")
    print(f"Narrative Chunk character length: {len(narrative_chunk_text)}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    audit_dataset()
