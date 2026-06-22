import os
import sys
import json

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.xml_parser import parse_xml_file
from pipeline.mitre_mapper import MitreMapper
from pipeline.neo4j_ingester import Neo4jIngester

def run_test():
    print("Initializing Phase 4 Test Pipeline...")
    
    # 1. Parse XML
    xml_path = os.path.join(base_dir, "CTI_Report_Dataset", "CTIDataset_2018_ReportEvent.xml")
    events = parse_xml_file(xml_path)
    
    # Take a small chronological batch of 2 events to test Temporal Edges
    # We sort them to ensure we get a sequence of events
    events.sort(key=lambda x: x.date)
    test_batch = events[:2]
    
    print(f"Selected {len(test_batch)} events for ingestion test.")
    
    # 2. Map Events
    mapper = MitreMapper(base_dir)
    extractions = []
    
    print("\nMapping Events using Hybrid Classifier (ChromaDB + Gemini)...")
    for i, event in enumerate(test_batch):
        print(f"Mapping Event {i+1}/5 (ID: {event.event_id})...")
        ext = mapper.map_event(event)
        
        # Override the event ID so it's globally unique instead of just 'evt_01'
        ext.events[0].event_id = f"evt_{event.event_id}"
        
        # Update relationships to point to the new unique event ID
        for rel in ext.relationships:
            if rel.source_id == "evt_01":
                rel.source_id = f"evt_{event.event_id}"
                
        extractions.append(ext)
        
    print("\nMapping complete. Launching Neo4j Ingestion...")
    
    # 3. Ingest to Neo4j
    ingester = Neo4jIngester(base_dir)
    try:
        ingester.ingest(extractions)
    finally:
        ingester.close()
        
    print("\nTest completed successfully! The graph has been built in Neo4j.")

if __name__ == "__main__":
    run_test()
