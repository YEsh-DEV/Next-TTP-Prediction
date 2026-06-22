import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

def validate_graph_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(base_dir, ".env"))
    
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    
    output_path = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\graph_path_validation.md"
    
    # We will trace paths for T1346 (Top 1 from cleansed retrieval) and T1071 (Top 5)
    target_techniques = ["T1346", "T1071"]
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# TASK 3 — REAL GRAPH PATH VALIDATION\n\n")
        f.write("Executing raw Cypher queries against AuraDB to trace `(:APTGroup)-[:USES]->(:Software)-[:USES]->(:Technique)`.\n\n")
        
        with driver.session() as session:
            for tech in target_techniques:
                f.write(f"## Path Trace for Technique: `{tech}`\n")
                
                # Query 1: Software -> Technique
                cypher_soft = "MATCH p=(s:Software)-[:USES]->(t:Technique {technique_id: $tid}) RETURN s.software_id AS soft LIMIT 5"
                soft_res = session.run(cypher_soft, tid=tech)
                f.write(f"### Direct Software Links (`(:Software)-[:USES]->(:Technique)`)\n")
                f.write("```text\n")
                soft_found = False
                for record in soft_res:
                    soft_found = True
                    f.write(f"(Software: {record['soft']}) -[:USES]-> (Technique: {tech})\n")
                if not soft_found:
                    f.write("No direct Software links found.\n")
                f.write("```\n\n")
                
                # Query 2: APT -> Software -> Technique
                cypher_apt = "MATCH p=(a:APTGroup)-[:USES]->(s:Software)-[:USES]->(t:Technique {technique_id: $tid}) RETURN a.name AS apt, s.software_id AS soft LIMIT 5"
                apt_res = session.run(cypher_apt, tid=tech)
                f.write(f"### Deep APT Links (`(:APTGroup)-[:USES]->(:Software)-[:USES]->(:Technique)`)\n")
                f.write("```text\n")
                apt_found = False
                for record in apt_res:
                    apt_found = True
                    f.write(f"(APTGroup: {record['apt']}) -[:USES]-> (Software: {record['soft']}) -[:USES]-> (Technique: {tech})\n")
                if not apt_found:
                    f.write("No deep APT-to-Software pathways found for this specific technique.\n")
                f.write("```\n\n")
                
                # Query 3: APT -> Technique
                cypher_apt_direct = "MATCH p=(a:APTGroup)-[:USES]->(t:Technique {technique_id: $tid}) RETURN a.name AS apt LIMIT 5"
                apt_direct_res = session.run(cypher_apt_direct, tid=tech)
                f.write(f"### Direct APT Links (`(:APTGroup)-[:USES]->(:Technique)`)\n")
                f.write("```text\n")
                apt_direct_found = False
                for record in apt_direct_res:
                    apt_direct_found = True
                    f.write(f"(APTGroup: {record['apt']}) -[:USES]-> (Technique: {tech})\n")
                if not apt_direct_found:
                    f.write("No direct APT pathways found.\n")
                f.write("```\n\n")
                
    print(f"Proof saved to {output_path}")

if __name__ == "__main__":
    validate_graph_paths()
