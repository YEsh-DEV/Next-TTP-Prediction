import os
import sys
import json
from neo4j import GraphDatabase

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.xml_parser import parse_xml_file
from pipeline.mitre_mapper import MitreMapper
from pipeline.neo4j_ingester import Neo4jIngester
from dotenv import load_dotenv

load_dotenv(os.path.join(base_dir, ".env"))
uri = os.environ.get("NEO4J_URI")
user = os.environ.get("NEO4J_USERNAME", "neo4j")
password = os.environ.get("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(user, password))

def wipe_db():
    print("Wiping Neo4j Database for clean Pilot Validation...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

def run_ingestion():
    print("Initializing Pilot Batch...")
    xml_path = os.path.join(base_dir, "CTI_Report_Dataset", "CTIDataset_2018_ReportEvent.xml")
    events = parse_xml_file(xml_path)
    
    events.sort(key=lambda x: x.date)
    test_batch = events[:50]
    
    mapper = MitreMapper(base_dir)
    extractions = []
    
    print(f"\nMapping {len(test_batch)} Events...")
    for i, event in enumerate(test_batch):
        ext = mapper.map_event(event)
        
        if ext.events:
            ext.events[0].event_id = f"evt_{event.event_id}"
            for rel in ext.relationships:
                if rel.source_id == "evt_01":
                    rel.source_id = f"evt_{event.event_id}"
                    
        extractions.append(ext)
        if i % 10 == 0:
            print(f"Processed {i}/50 events...")
            
    print("\nIngesting to Neo4j...")
    ingester = Neo4jIngester(base_dir)
    ingester.ingest(extractions)
    ingester.close()

def run_validation():
    print("\n--- PILOT GRAPH VALIDATION RESULTS ---")
    with driver.session() as session:
        counts = {
            "Events": session.run("MATCH (e:Event) RETURN count(e) as c").single()["c"],
            "Techniques": session.run("MATCH (t:Technique) RETURN count(t) as c").single()["c"],
            "Infrastructure": session.run("MATCH (i:Infrastructure) RETURN count(i) as c").single()["c"],
            "OBSERVED_TECHNIQUE": session.run("MATCH ()-[r:OBSERVED_TECHNIQUE]->() RETURN count(r) as c").single()["c"],
            "PRECEDES": session.run("MATCH ()-[r:PRECEDES]->() RETURN count(r) as c").single()["c"]
        }
        
        print(json.dumps(counts, indent=2))
        
        print("\n--- SAMPLE EVENTS ---")
        evts = session.run("MATCH (e:Event) RETURN e.event_id as id, size(e.indicators) as ind_count LIMIT 5")
        for e in evts: print(e)
        
        print("\n--- SAMPLE INFRASTRUCTURE ---")
        infs = session.run("MATCH (i:Infrastructure) RETURN i.value, i.type LIMIT 5")
        for i in infs: print(i)
        
        print("\n--- SAMPLE OBSERVED_TECHNIQUE ---")
        rels = session.run("MATCH (e:Event)-[r:OBSERVED_TECHNIQUE]->(t:Technique) RETURN e.event_id, r.confidence, t.technique_id LIMIT 5")
        for r in rels: print(r)
        
        print("\n--- SAMPLE PRECEDES PATHS ---")
        paths = session.run("MATCH (e1:Event)-[:PRECEDES]->(e2:Event) RETURN e1.event_id, e1.timestamp, e2.event_id, e2.timestamp LIMIT 5")
        for p in paths: print(p)

if __name__ == "__main__":
    wipe_db()
    run_ingestion()
    run_validation()
    driver.close()
