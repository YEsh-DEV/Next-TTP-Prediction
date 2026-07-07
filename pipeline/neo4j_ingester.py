import os
from neo4j import GraphDatabase
from collections import Counter
from typing import List
from dotenv import load_dotenv

from schemas.extraction_schema import ExtractionResult

class Neo4jIngester:
    def __init__(self, base_dir: str):
        load_dotenv(os.path.join(base_dir, ".env"))
        
        self.uri = os.environ.get("NEO4J_URI")
        self.user = os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = os.environ.get("NEO4J_PASSWORD")
        
        if not all([self.uri, self.user, self.password]):
            raise ValueError("Neo4j credentials missing from .env")
            
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
    def close(self):                                                               
        self.driver.close()
        
    def _create_constraints(self):
        with self.driver.session() as session:
            # Create uniqueness constraints for idempotent MERGEs
            queries = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Technique) REQUIRE t.technique_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Infrastructure) REQUIRE i.value IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (a:APTGroup) REQUIRE a.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Software) REQUIRE s.name IS UNIQUE"
            ]
            for q in queries:
                session.run(q)

    def ingest(self, extractions: List[ExtractionResult]):
        print("Initializing Neo4j Constraints...")
        self._create_constraints()
        
        # --- PASS 1: Global Indicator Frequency ---
        print("Pass 1: Calculating Global Indicator Frequencies...")
        indicator_counter = Counter()
        for ext in extractions:
            for event in ext.events:
                # Add all indicators in this event to the global counter
                indicator_counter.update(event.indicators)
                
        # Identify high-frequency indicators (Shared Infrastructure)
        shared_infrastructure = {ind for ind, count in indicator_counter.items() if count >= 2}
        print(f"Identified {len(shared_infrastructure)} shared indicators to be promoted to physical Nodes.")

        # --- PASS 2: Node Ingestion ---
        print("Pass 2: Ingesting Graph Nodes...")
        with self.driver.session() as session:
            for ext in extractions:
                # 1. Ingest Entities (Threat Actors, Malware, CVEs)
                for ent in ext.entities:
                    if ent.type.value == "ThreatActor":
                        session.run("""
                            MERGE (a:APTGroup {name: $name})
                            SET a.description = $desc
                        """, name=ent.value, desc=ent.description)
                    elif ent.type.value == "Technique":
                        # Note: technique entity_id is like 'tech_T1043', value is the actual Tactic Name
                        tech_id = ent.entity_id.replace('tech_', '')
                        session.run("""
                            MERGE (t:Technique {technique_id: $tid})
                            SET t.name = $name, t.description = $desc
                        """, tid=tech_id, name=ent.value, desc=ent.description)
                    elif ent.type.value == "FileArtifact":
                        session.run("""
                            MERGE (s:Software {name: $name})
                        """, name=ent.value)
                        
                # 2. Ingest Events & Dynamic Infrastructure
                for event in ext.events:
                    # Filter rare indicators to store on the Event node itself
                    rare_indicators = [ind for ind in event.indicators if ind not in shared_infrastructure]
                    
                    session.run("""
                        MERGE (e:Event {event_id: $eid})
                        SET e.action = $action,
                            e.timestamp = $timestamp,
                            e.description = $desc,
                            e.indicators = $rare_inds
                    """, eid=event.event_id, action=event.action, 
                         timestamp=event.timestamp, desc=event.description,
                         rare_inds=rare_indicators)
                         
                    # Create physical Nodes for shared infrastructure in this event
                    event_shared_inds = [ind for ind in event.indicators if ind in shared_infrastructure]
                    for ind in event_shared_inds:
                        # Extract type and value (e.g. 'ip-src:192.168.1.1')
                        parts = ind.split(':', 1)
                        i_type = parts[0] if len(parts) > 1 else "unknown"
                        i_value = parts[1] if len(parts) > 1 else ind
                        
                        session.run("""
                            MERGE (i:Infrastructure {value: $val})
                            SET i.type = $type
                            WITH i
                            MATCH (e:Event {event_id: $eid})
                            MERGE (e)-[:COMMUNICATES_WITH]->(i)
                        """, val=i_value, type=i_type, eid=event.event_id)

        # --- PASS 3: Edge Ingestion ---
        print("Pass 3: Ingesting Graph Relationships...")
        with self.driver.session() as session:
            for ext in extractions:
                for rel in ext.relationships:
                    # OBSERVED_TECHNIQUE edges
                    if rel.type.value == "OBSERVED_TECHNIQUE":
                        tech_id = rel.target_id.replace('tech_', '')
                        session.run("""
                            MATCH (e:Event {event_id: $eid})
                            MATCH (t:Technique {technique_id: $tid})
                            MERGE (e)-[r:OBSERVED_TECHNIQUE]->(t)
                            SET r.confidence = $conf
                        """, eid=rel.source_id, tid=tech_id, conf=rel.confidence)
                        
                # 3b. Temporal Edges (PRECEDES)
                # Sort events globally across all extractions by timestamp
                all_events = []
                for ex in extractions:
                    all_events.extend(ex.events)
                    
                # Sort chronologically
                all_events.sort(key=lambda x: x.timestamp)
                
                # Draw PRECEDES edges
                for i in range(len(all_events) - 1):
                    e1 = all_events[i]
                    e2 = all_events[i+1]
                    # Calculate basic time delta (simplified)
                    session.run("""
                        MATCH (e1:Event {event_id: $eid1})
                        MATCH (e2:Event {event_id: $eid2})
                        MERGE (e1)-[:PRECEDES]->(e2)
                    """, eid1=e1.event_id, eid2=e2.event_id)
                    
        print("Neo4j Ingestion Complete!")
