import os
import chromadb
from sentence_transformers import SentenceTransformer

def extract_retrieval_proof():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chroma_client = chromadb.PersistentClient(path=os.path.join(base_dir, "chroma_db"))
    collection = chroma_client.get_collection(name="cti_events")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    query = "Burning Umbrella"
    print(f"Querying ChromaDB for: '{query}'")
    
    q_emb = model.encode([query], show_progress_bar=False).tolist()[0]
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=10,
        include=["metadatas", "distances"]
    )
    
    output_path = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\submission_retrieval_proof.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# TASK 3 — REAL RETRIEVAL PROOF\n\n")
        f.write(f"**Target Query:** `{query}`\n\n")
        f.write("## Top 10 Retrieved Events\n\n")
        f.write("| Rank | Event ID | Date | Score (L2 Distance) | Title / Info |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        for i in range(len(results['ids'][0])):
            event_id = results['ids'][0][i]
            dist = results['distances'][0][i]
            meta = results['metadatas'][0][i]
            date = meta.get('date', 'N/A')
            info = meta.get('info', 'N/A')
            
            f.write(f"| {i+1} | `{event_id}` | {date} | {dist:.4f} | {info} |\n")
            
    print(f"Proof saved to {output_path}")

if __name__ == "__main__":
    extract_retrieval_proof()
