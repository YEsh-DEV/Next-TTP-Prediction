"""
Push Frozen Benchmark Graph to Neo4j (AuraDB)

This script loads the official Top-2 Actor-aware benchmark
and uploads it to Neo4j so you can visualize the exact 
transitions used in the evaluation.
"""
import os, sys
from neo4j import GraphDatabase
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from scratch.final_production_run import build_frozen_benchmark
from pipeline.deterministic_classifier_v2 import DeterministicClassifierV2

# Load from .env file
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def push_to_neo4j():
    print("Loading official benchmark graph...")
    classifier = DeterministicClassifierV2(base_dir)
    train, test, all_nodes, all_techniques, timelines = build_frozen_benchmark(classifier)
    all_transitions = train + test

    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # Clear existing
        print("Clearing existing database...")
        session.run("MATCH (n) DETACH DELETE n")

        print(f"Pushing {len(all_nodes)} nodes and {len(all_transitions)} edges...")
        
        # Merge Nodes
        for node in all_nodes:
            if "::" in node:
                actor, tech = node.split("::", 1)
                session.run(
                    """
                    MERGE (n:ActorState {id: $id})
                    SET n.actor = $actor, n.technique = $tech
                    """,
                    id=node, actor=actor, tech=tech
                )
            else:
                session.run(
                    """
                    MERGE (n:State {id: $id})
                    SET n.technique = $id
                    """,
                    id=node
                )
        
        # Create Edges
        for t in all_transitions:
            session.run(
                """
                MATCH (s {id: $src})
                MATCH (t {id: $tgt})
                MERGE (s)-[r:NEXT_TTP]->(t)
                SET r.actor = $actor,
                    r.src_date = $src_date,
                    r.tgt_date = $tgt_date,
                    r.src_event = $src_event,
                    r.tgt_event = $tgt_event
                """,
                src=t["src_node"],
                tgt=t["tgt_node"],
                actor=t["actor"],
                src_date=t["src_date"],
                tgt_date=t["tgt_date"],
                src_event=t["src_event"],
                tgt_event=t.get("tgt_event", "")
            )

    driver.close()
    print("Graph successfully pushed to Neo4j!")
    print("You can now open Neo4j Bloom or Browser to visualize it.")

if __name__ == "__main__":
    if not NEO4J_PASSWORD:
        print("Please set NEO4J_PASSWORD in your .env file!")
        sys.exit(1)
    push_to_neo4j()
