import os
import chromadb
from sentence_transformers import SentenceTransformer
from schemas.cti_schema import CTIEvent

class DeterministicClassifierV2:
    def __init__(self, base_dir: str):
        self.chroma_dir = os.path.join(base_dir, "chroma_db")
        self.db_client = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection = self.db_client.get_collection(name="mitre_techniques")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def classify_event(self, cti_event: CTIEvent) -> dict:
        """
        Classifies a CTIEvent using dense, contextual embedding blocks (V2).
        Fixes the State Collapse by extracting rich natural language and indicators.
        """
        title = cti_event.info
        categories = list(set([a.category for a in cti_event.attributes if a.category]))
        types = list(set([a.type for a in cti_event.attributes if a.type]))
        comments = list(set([a.comment for a in cti_event.attributes if a.comment]))
        cves = [a.value for a in cti_event.attributes if a.type == "vulnerability"]
        filenames = [a.value for a in cti_event.attributes if a.type == "filename"]
        domains = [a.value for a in cti_event.attributes if a.type == "domain"]
        urls = [a.value for a in cti_event.attributes if a.type == "url"][:3]
        hashes = [a.value for a in cti_event.attributes if a.type in ("md5", "sha1", "sha256")][:3]
        
        # Build Semantic Context String
        text = f"Title: {title}\n"
        text += f"Categories: {', '.join(categories)}\n"
        text += f"Types: {', '.join(types)}\n"
        
        if comments: text += f"Context/Comments: {', '.join(comments)}\n"
        if cves: text += f"Vulnerabilities (CVEs): {', '.join(cves)}\n"
        if filenames: text += f"Targeted Files: {', '.join(filenames[:5])}\n"
        if domains: text += f"Network Domains: {', '.join(domains[:5])}\n"
        if urls: text += f"URLs: {', '.join(urls)}\n"
        if hashes: text += f"Malware Hashes: {', '.join(hashes)}\n"
        
        event_embedding = self.model.encode([text], show_progress_bar=False).tolist()[0]
        
        results = self.collection.query(
            query_embeddings=[event_embedding],
            n_results=20,
            include=["distances"]
        )
        
        candidates = results['ids'][0]
        distances = results['distances'][0]
        
        ranked_techniques = []
        for tid, dist in zip(candidates, distances):
            score = max(0.0, 1.0 - (dist / 2.0))
            ranked_techniques.append({"id": tid, "score": round(score, 2)})
            
        ranked_techniques.sort(key=lambda x: x['score'], reverse=True)
        top_5 = ranked_techniques[:5]
        
        return {
            "event_id": str(cti_event.event_id),
            "techniques": top_5
        }
