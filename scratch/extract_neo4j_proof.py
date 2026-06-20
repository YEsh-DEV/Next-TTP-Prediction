import os
import pandas as pd
from neo4j import GraphDatabase

def run():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(base_dir, ".env"))
    
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    
    attack_path = os.path.join(base_dir, "attackmitre.xlsx")
    df = pd.read_excel(attack_path)
    
    apt_names = []
    software_ids = []
    apt_to_software = []
    apt_to_tech = []
    soft_to_tech = []
    
    for _, row in df.iterrows():
        apt = str(row.get('APT Group Name', '')).strip()
        soft = str(row.get('Software ID', '')).strip()
        g_techs = str(row.get('Group Techniques', ''))
        s_techs = str(row.get('Software Techniques', ''))
        
        if apt and apt != 'nan': apt_names.append(apt)
        if soft and soft != 'nan': software_ids.append(soft)
        if apt and apt != 'nan' and soft and soft != 'nan': apt_to_software.append({"apt": apt, "soft": soft})
        
        if g_techs and g_techs != 'nan':
            for t in g_techs.split(';'):
                if t.strip(): apt_to_tech.append({"apt": apt, "tech": t.strip()})
                
        if s_techs and s_techs != 'nan':
            for t in s_techs.split(';'):
                if t.strip(): soft_to_tech.append({"soft": soft, "tech": t.strip()})
    
    with driver.session() as session:
        print("Ingesting APTGroups (Batched)...")
        session.run("UNWIND $apts AS a MERGE (:APTGroup {name: a})", apts=list(set(apt_names)))
        print("Ingesting Software (Batched)...")
        session.run("UNWIND $softs AS s MERGE (:Software {software_id: s})", softs=list(set(software_ids)))
        print("Linking APT->Software (Batched)...")
        session.run("UNWIND $rels AS r MATCH (a:APTGroup {name: r.apt}), (s:Software {software_id: r.soft}) MERGE (a)-[:USES]->(s)", rels=apt_to_software)
        print("Linking APT->Technique (Batched)...")
        session.run("UNWIND $rels AS r MATCH (a:APTGroup {name: r.apt}) MERGE (t:Technique {technique_id: r.tech}) MERGE (a)-[:USES]->(t)", rels=apt_to_tech)
        print("Linking Software->Technique (Batched)...")
        session.run("UNWIND $rels AS r MATCH (s:Software {software_id: r.soft}) MERGE (t:Technique {technique_id: r.tech}) MERGE (s)-[:USES]->(t)", rels=soft_to_tech)
        
        print("\nExtracting Cypher Validations...")
        output_path = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\submission_neo4j_proof.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# TASK 2 — REAL NEO4J PROOF\n\n")
            
            f.write("## 1. Node Counts\n")
            f.write("`MATCH (n) RETURN labels(n), count(*)`\n")
            res = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count")
            f.write("```text\n")
            for record in res: f.write(f"Label: {record['label']} | Count: {record['count']}\n")
            f.write("```\n\n")
            
            f.write("## 2. Edge Counts\n")
            f.write("`MATCH ()-[r]->() RETURN type(r), count(*)`\n")
            res = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count")
            f.write("```text\n")
            for record in res: f.write(f"Type: {record['type']} | Count: {record['count']}\n")
            f.write("```\n\n")
            
            f.write("## 3. Top 10 Paths: APTGroup -> Software\n")
            f.write("`MATCH p=(:APTGroup)-[:USES]->(:Software) RETURN p LIMIT 10`\n")
            res = session.run("MATCH (a:APTGroup)-[:USES]->(s:Software) RETURN a.name AS apt, s.software_id AS soft LIMIT 10")
            f.write("```text\n")
            for record in res: f.write(f"(APTGroup: {record['apt']})-[:USES]->(Software: {record['soft']})\n")
            f.write("```\n\n")
            
            f.write("## 4. Top 10 Paths: Software -> Technique\n")
            f.write("`MATCH p=(:Software)-[:USES]->(:Technique) RETURN p LIMIT 10`\n")
            res = session.run("MATCH (s:Software)-[:USES]->(t:Technique) RETURN s.software_id AS soft, t.technique_id AS tech LIMIT 10")
            f.write("```text\n")
            for record in res: f.write(f"(Software: {record['soft']})-[:USES]->(Technique: {record['tech']})\n")
            f.write("```\n\n")
            
        print(f"Proof saved to {output_path}")

if __name__ == "__main__":
    run()
