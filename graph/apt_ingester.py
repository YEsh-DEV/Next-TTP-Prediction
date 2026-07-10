import os
import sys
import pandas as pd
import requests
from neo4j import GraphDatabase
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas.neo4j_schema import APTGroupSchema, SoftwareSchema, TechniqueSchema, TacticSchema  

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

def create_constraints(driver):
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:APTGroup) REQUIRE g.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Software) REQUIRE s.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Technique) REQUIRE t.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (ta:Tactic) REQUIRE ta.name IS UNIQUE")

def wipe_db(driver):
    print("Wiping existing graph...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

def fetch_stix_tactics():
    print("Fetching MITRE ATT&CK STIX data for Tactics mapping...")
    url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    data = requests.get(url).json()
    
    tactics_set = set()
    tech_to_tactic = []
    
    for obj in data["objects"]:
        if obj["type"] == "attack-pattern":
            ext_refs = obj.get("external_references", [])
            mitre_id = next((ext["external_id"] for ext in ext_refs if ext["source_name"] == "mitre-attack"), None)
            if mitre_id:
                for kp in obj.get("kill_chain_phases", []):
                    if kp["kill_chain_name"] == "mitre-attack":
                        tactic_name = kp["phase_name"].replace("-", " ").title()
                        tactics_set.add(tactic_name)
                        tech_to_tactic.append({'technique': mitre_id, 'tactic': tactic_name})
                        
    return list(tactics_set), tech_to_tactic

def ingest_data(driver, attackmitre_path, mitreenterprise_path):
    print("Loading datasets...")
    attackmitre = pd.read_excel(attackmitre_path)
    mitreenterprise = pd.read_excel(mitreenterprise_path)

    tactics_list, tech_to_tactic_rels = fetch_stix_tactics()
    
    # 0. Ingest Tactics
    print("Ingesting Tactics...")
    tactics_payload = [TacticSchema(name=t).model_dump() for t in tactics_list]
    with driver.session() as session:
        session.run("""
            UNWIND $tactics AS t
            MERGE (ta:Tactic {name: t.name})
        """, tactics=tactics_payload)

    # 1. Prepare Technique properties
    print("Preparing Technique properties...")
    techniques = []
    for _, row in mitreenterprise.iterrows():
        t_id = str(row['Tactic ID']).strip()
        if pd.isna(row['Tactic ID']) or not t_id:
            continue
            
        t_name = str(row['Tactic Name']).strip() if not pd.isna(row['Tactic Name']) else None
        desc = str(row['Description']).strip() if not pd.isna(row['Description']) else None
        miti = str(row['Mitigation Steps']).strip() if not pd.isna(row['Mitigation Steps']) else None
        
        # Pydantic validation
        tech = TechniqueSchema(id=t_id, name=t_name, description=desc, mitigation_steps=miti)
        techniques.append(tech.model_dump())

    # Ingest Techniques
    with driver.session() as session:
        session.run("""
            UNWIND $techniques AS t
            MERGE (tech:Technique {id: t.id})
            SET tech.name = t.name,
                tech.description = t.description,
                tech.mitigation_steps = t.mitigation_steps
        """, techniques=techniques)
        
        # Ingest Technique -> Tactic relationships
        print("Ingesting Technique -> Tactic relationships...")
        session.run("""
            UNWIND $rels AS r
            MATCH (tech:Technique {id: r.technique})
            MATCH (ta:Tactic {name: r.tactic})
            MERGE (tech)-[:BELONGS_TO_TACTIC]->(ta)
        """, rels=tech_to_tactic_rels)

    print("Processing attackmitre relationships...")
    
    # 2. Prepare APT Groups, Software, and Relationships
    groups_to_ingest = []
    softwares_to_ingest = []
    
    group_uses_technique = []
    group_uses_software = []
    software_uses_technique = []

    for _, row in attackmitre.iterrows():
        apt_name = str(row['APT Group Name']).strip()
        if pd.isna(row['APT Group Name']) or not apt_name:
            continue
            
        groups_to_ingest.append(APTGroupSchema(name=apt_name).model_dump())
        
        # Parse Group Techniques
        if not pd.isna(row['Group Techniques']):
            g_techs = [t.strip() for t in str(row['Group Techniques']).split(';') if t.strip()]
            for t in g_techs:
                group_uses_technique.append({'group': apt_name, 'technique': t})

        # Parse Software
        s_id = str(row['Software ID']).strip() if not pd.isna(row['Software ID']) else None
        if s_id:
            softwares_to_ingest.append(SoftwareSchema(id=s_id).model_dump())
            group_uses_software.append({'group': apt_name, 'software': s_id})
            
            # Parse Software Techniques
            if not pd.isna(row['Software Techniques']):
                s_techs = [t.strip() for t in str(row['Software Techniques']).split(';') if t.strip()]
                for t in s_techs:
                    software_uses_technique.append({'software': s_id, 'technique': t})

    # Deduplicate lists of dicts
    groups_to_ingest = [dict(t) for t in {tuple(d.items()) for d in groups_to_ingest}]
    softwares_to_ingest = [dict(t) for t in {tuple(d.items()) for d in softwares_to_ingest}]
    group_uses_technique = [dict(t) for t in {tuple(d.items()) for d in group_uses_technique}]
    group_uses_software = [dict(t) for t in {tuple(d.items()) for d in group_uses_software}]
    software_uses_technique = [dict(t) for t in {tuple(d.items()) for d in software_uses_technique}]

    with driver.session() as session:
        # Ingest Nodes
        session.run("UNWIND $groups AS g MERGE (n:APTGroup {name: g.name})", groups=groups_to_ingest)
        session.run("UNWIND $softwares AS s MERGE (n:Software {id: s.id})", softwares=softwares_to_ingest)
        
        # Relationships: Group -> Technique (Ensure Technique exists first) -> USES_TECHNIQUE
        session.run("""
            UNWIND $rels AS r
            MERGE (g:APTGroup {name: r.group})
            MERGE (t:Technique {id: r.technique})
            MERGE (g)-[:USES_TECHNIQUE]->(t)
        """, rels=group_uses_technique)
        
        # Relationships: Group -> Software -> USES
        session.run("""
            UNWIND $rels AS r
            MERGE (g:APTGroup {name: r.group})
            MERGE (s:Software {id: r.software})
            MERGE (g)-[:USES]->(s)
        """, rels=group_uses_software)
        
        # Relationships: Software -> Technique (Ensure Technique exists first) -> USES_TECHNIQUE
        session.run("""
            UNWIND $rels AS r
            MERGE (s:Software {id: r.software})
            MERGE (t:Technique {id: r.technique})
            MERGE (s)-[:USES_TECHNIQUE]->(t)
        """, rels=software_uses_technique)

    print("Ingestion complete.")

def verify_graph(driver):
    queries = [
        ("Query 1: MATCH (a:APTGroup) RETURN count(a)", "MATCH (a:APTGroup) RETURN count(a) as Count"),
        ("Query 2: MATCH (s:Software) RETURN count(s)", "MATCH (s:Software) RETURN count(s) as Count"),
        ("Query 3: MATCH (t:Technique) RETURN count(t)", "MATCH (t:Technique) RETURN count(t) as Count"),
        ("Query 4: MATCH (ta:Tactic) RETURN count(ta)", "MATCH (ta:Tactic) RETURN count(ta) as Count"),
        ("Query 5: MATCH p=(:APTGroup)-[:USES]->(:Software) RETURN p LIMIT 5", "MATCH p=(:APTGroup)-[:USES]->(:Software) RETURN p LIMIT 5"),
        ("Query 6: MATCH p=(:Technique)-[:BELONGS_TO_TACTIC]->(:Tactic) RETURN p LIMIT 5", "MATCH p=(:Technique)-[:BELONGS_TO_TACTIC]->(:Tactic) RETURN p LIMIT 5"),
        ("Query 7: MATCH p=(:APTGroup)-[:USES]->(:Software)-[:USES_TECHNIQUE]->(:Technique)-[:BELONGS_TO_TACTIC]->(:Tactic) RETURN p LIMIT 1", "MATCH p=(:APTGroup)-[:USES]->(:Software)-[:USES_TECHNIQUE]->(:Technique)-[:BELONGS_TO_TACTIC]->(:Tactic) RETURN p LIMIT 1")
    ]
    
    with driver.session() as session:
        print("\n--- User Requested Validations ---")
        for title, query in queries:
            print(f"\n{title}")
            res = session.run(query)
            for record in res:
                if 'Count' in record.keys():
                    print(record['Count'])
                else:
                    print(record['p'])

if __name__ == "__main__":
    if not URI or not USERNAME or not PASSWORD:
        print("Error: Missing Neo4j credentials in .env file.")
        sys.exit(1)
        
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    
    # Resolving absolute paths to data files
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    attack_path = os.path.join(base_dir, "attackmitre.xlsx")
    mitre_path = os.path.join(base_dir, "MitreEnterprise.xlsx")

    try:
        wipe_db(driver)
        print("Initializing constraints...")
        create_constraints(driver)
        ingest_data(driver, attack_path, mitre_path)
        verify_graph(driver)
    finally:
        driver.close()
