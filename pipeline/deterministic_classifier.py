import os
import chromadb
from sentence_transformers import SentenceTransformer
from schemas.cti_schema import CTIEvent

class DeterministicClassifier:
    def __init__(self, base_dir: str):
        self.chroma_dir = os.path.join(base_dir, "chroma_db")
        self.db_client = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection = self.db_client.get_collection(name="mitre_techniques")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def classify_event(self, cti_event: CTIEvent) -> dict:
        """
        Classifies a CTIEvent completely deterministically using localized ChromaDB.
        NO API CALLS ALLOWED.
        """
        categories = list(set([a.category for a in cti_event.attributes if a.category]))
        types = list(set([a.type for a in cti_event.attributes if a.type]))
        text = f"Observed Categories: {categories}\nObserved Types: {types}"
        
        event_embedding = self.model.encode([text], show_progress_bar=False).tolist()[0]
        
        results = self.collection.query(
            query_embeddings=[event_embedding],
            n_results=20,
            include=["distances"]
        )
        
        candidates = results['ids'][0]
        distances = results['distances'][0]
        
        # Rank by similarity and return Top 3. (Lower L2 distance = higher similarity)
        # Convert distance to a rough similarity score (1.0 - (dist / 2)) assuming normalized vectors
        
        ranked_techniques = []
        for tid, dist in zip(candidates, distances):
            score = max(0.0, 1.0 - (dist / 2.0))
            ranked_techniques.append({"id": tid, "score": round(score, 2)})
            
        ranked_techniques.sort(key=lambda x: x['score'], reverse=True)
        top_3 = ranked_techniques[:3]
        
        return {
            "event_id": str(cti_event.event_id),
            "techniques": top_3
        }
