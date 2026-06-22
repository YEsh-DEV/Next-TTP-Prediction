import os
import chromadb
from sentence_transformers import SentenceTransformer

def run_retrieval():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chroma_client = chromadb.PersistentClient(path=os.path.join(base_dir, "chroma_db"))
    collection = chroma_client.get_collection(name="cti_events")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    queries = [
        "Burning Umbrella",
        "Operation Kitty",
        "Spearphishing Campaign",
        "Network Activity",
        "CVE"
    ]
    
    print("--- GraphRAG Retrieval Validation ---\n")
    
    for q in queries:
        print(f"QUERY: '{q}'")
        print("-" * 40)
        
        q_emb = model.encode([q], show_progress_bar=False).tolist()[0]
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=5,
            include=["metadatas", "distances"]
        )
        
        for i in range(len(results['ids'][0])):
            event_id = results['ids'][0][i]
            dist = results['distances'][0][i]
            meta = results['metadatas'][0][i]
            
            # Convert cosine distance to a rough similarity score (1 - dist)
            # Note: ChromaDB distances depend on the metric (default L2), 
            # but all-MiniLM outputs normalized vectors, so L2 distance is related to cosine similarity.
            # lower distance = higher similarity
            
            print(f"  Rank {i+1}: {event_id} | L2 Distance: {dist:.4f}")
            print(f"  Metadata: {meta}")
            print()
        print("=" * 60 + "\n")

if __name__ == "__main__":
    run_retrieval()
