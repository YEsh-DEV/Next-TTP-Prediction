import os
import sys
import math
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

class StaticNeo4jIngester:
    def __init__(self, base_dir: str):
        load_dotenv(os.path.join(base_dir, ".env"))
        
        self.uri = os.environ.get("NEO4J_URI")
        self.user = os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = os.environ.get("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        self.enterprise_path = os.path.join(base_dir, "MitreEnterprise.xlsx")
        self.attack_path = os.path.join(base_dir, "attackmitre.xlsx")

    def wipe_database(self):
        print("Wiping AuraDB for clean static ingestion...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            
        print("Creating Core Constraints...")
        with self.driver.session() as session:
            queries = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Technique) REQUIRE t.technique_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (a:APTGroup) REQUIRE a.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Software) REQUIRE s.software_id IS UNIQUE"
            ]
            for q in queries:
                session.run(q)

    def ingest_enterprise_techniques(self):
        print("Ingesting MITRE Techniques from MitreEnterprise.xlsx...")
        df = pd.read_excel(self.enterprise_path)
        
        with self.driver.session() as session:
            for index, row in df.iterrows():
                tech_id = str(row.get('Tactic ID', '')).strip() # In this dataset, Technique IDs are stored in the 'Tactic ID' column
                if not tech_id or tech_id == 'nan': continue
                
                name = str(row.get('Tactic Name', '')).strip()
                desc = str(row.get('Description', '')).strip()
                mitigation = str(row.get('Mitigation Steps', '')).strip()
                
                if desc == 'nan': desc = ''
                if mitigation == 'nan': mitigation = ''
                
                session.run("""
                    MERGE (t:Technique {technique_id: $tid})
                    SET t.name = $name,
                        t.description = $desc,
                        t.mitigation = $mitig
                """, tid=tech_id, name=name, desc=desc, mitig=mitigation)
                
    def ingest_attack_groups_and_software(self):
        print("Ingesting APT Groups and Software from attackmitre.xlsx...")
        df = pd.read_excel(self.attack_path)
        
        with self.driver.session() as session:
            for index, row in df.iterrows():
                apt_name = str(row.get('APT Group Name', '')).strip()
                software_id = str(row.get('Software ID', '')).strip()
                group_techs_raw = row.get('Group Techniques')
                software_techs_raw = row.get('Software Techniques')
                
                if apt_name and apt_name != 'nan':
                    session.run("MERGE (a:APTGroup {name: $name})", name=apt_name)
                    
                if software_id and software_id != 'nan':
                    session.run("MERGE (s:Software {software_id: $sid})", sid=software_id)
                    
                # Link APT to Software
                if apt_name and apt_name != 'nan' and software_id and software_id != 'nan':
                    session.run("""
                        MATCH (a:APTGroup {name: $aname})
                        MATCH (s:Software {software_id: $sid})
                        MERGE (a)-[:USES]->(s)
                    """, aname=apt_name, sid=software_id)
                    
                # Link APT to Techniques
                if isinstance(group_techs_raw, str) and group_techs_raw != 'nan':
                    techs = [t.strip() for t in group_techs_raw.split(';') if t.strip()]
                    for t in techs:
                        session.run("""
                            MATCH (a:APTGroup {name: $aname})
                            MERGE (t:Technique {technique_id: $tid}) // Merge to ensure node exists even if missing from Enterprise file
                            MERGE (a)-[:USES]->(t)
                        """, aname=apt_name, tid=t)
                        
                # Link Software to Techniques
                if isinstance(software_techs_raw, str) and software_techs_raw != 'nan':
                    techs = [t.strip() for t in software_techs_raw.split(';') if t.strip()]
                    for t in techs:
                        session.run("""
                            MATCH (s:Software {software_id: $sid})
                            MERGE (t:Technique {technique_id: $tid})
                            MERGE (s)-[:USES]->(t)
                        """, sid=software_id, tid=t)

    def close(self):
        self.driver.close()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ingester = StaticNeo4jIngester(base_dir)
    try:
        ingester.wipe_database()
        ingester.ingest_enterprise_techniques()
        ingester.ingest_attack_groups_and_software()
        print("Static MITRE Knowledge Graph ingestion complete!")
    finally:
        ingester.close()
