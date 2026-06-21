import os
import json
import chromadb
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from pipeline.hybrid_classifier import HybridClassifier
from pipeline.deterministic_classifier import DeterministicClassifier
from pipeline.xml_parser import parse_xml_file
from dotenv import load_dotenv

class GraphRAGPipeline:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(base_dir, "chroma_db"))
        self.cti_collection = self.chroma_client.get_collection(name="cti_events")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.hybrid_classifier = HybridClassifier(base_dir)
        self.deterministic_classifier = DeterministicClassifier(base_dir)
        
        load_dotenv(os.path.join(base_dir, ".env"))
        self.neo4j_driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI"),
            auth=(os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ.get("NEO4J_PASSWORD"))
        )
        
        xml_path = os.path.join(base_dir, "CTI_Report_Dataset", "CTIDataset_2018_ReportEvent.xml")
        self.all_events = {f"evt_{e.event_id}": e for e in parse_xml_file(xml_path)}

    def execute_query(self, query: str, top_k: int = 3, classification_mode: str = "deterministic") -> dict:
        q_emb = self.model.encode([query], show_progress_bar=False).tolist()[0]
        
        # 1. Retrieval
        retrieval_res = self.cti_collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["metadatas"]
        )
        retrieved_ids = retrieval_res['ids'][0]
        
        # 2. Mapping
        mapped_techniques = set()
        for eid in retrieved_ids:
            evt = self.all_events.get(eid)
            if evt:
                if classification_mode == "gemini":
                    try:
                        matches = self.hybrid_classifier.classify_event(evt)
                        for m in matches: mapped_techniques.add(m.technique_id)
                    except Exception: pass
                elif classification_mode == "semantic":
                    m_res = self.hybrid_classifier.collection.query(query_embeddings=[q_emb], n_results=3, include=["distances"])
                    mapped_techniques.update(m_res['ids'][0])
                elif classification_mode == "deterministic":
                    det_result = self.deterministic_classifier.classify_event(evt)
                    for t in det_result['techniques']: mapped_techniques.add(t['id'])

        mapped_techniques = list(mapped_techniques)
        
        # 3. Traversal
        related_software = set()
        related_apts = set()
        
        with self.neo4j_driver.session() as session:
            for t in mapped_techniques:
                # Software -> Technique
                s_res = session.run("MATCH (s:Software)-[:USES]->(t:Technique {technique_id: $tid}) RETURN s.software_id AS soft", tid=t)
                for r in s_res: related_software.add(r['soft'])
                
                # APT -> Technique
                a_res = session.run("MATCH (a:APTGroup)-[:USES]->(t:Technique {technique_id: $tid}) RETURN a.name AS apt", tid=t)
                for r in a_res: related_apts.add(r['apt'])
                
                # APT -> Software -> Technique
                a2_res = session.run("MATCH (a:APTGroup)-[:USES]->(s:Software)-[:USES]->(t:Technique {technique_id: $tid}) RETURN a.name AS apt", tid=t)
                for r in a2_res: related_apts.add(r['apt'])
                
        # 4. JSON Output
        return {
            "query": query,
            "classification_mode": classification_mode,
            "retrieved_events": retrieved_ids,
            "mapped_techniques": mapped_techniques,
            "related_software": list(related_software),
            "related_apt_groups": list(related_apts)
        }
