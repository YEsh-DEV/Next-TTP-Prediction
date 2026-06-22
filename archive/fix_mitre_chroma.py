import os
import re
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

def cleanse_chroma():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    enterprise_path = os.path.join(base_dir, "MitreEnterprise.xlsx")
    
    df = pd.read_excel(enterprise_path)
    
    pattern = re.compile(r"^T[0-9]{4}(\.[0-9]{3})?$")
    
    unique_rows = {}
    for _, row in df.iterrows():
        tid = str(row.get('Tactic ID', '')).strip()
        if pattern.match(tid) and tid not in unique_rows:
            unique_rows[tid] = {
                "id": tid,
                "name": str(row.get('Tactic Name', '')),
                "desc": str(row.get('Tactic Description', ''))
            }
            
    valid_rows = list(unique_rows.values())
            
    print(f"Extracted {len(valid_rows)} valid Techniques from Excel.")
    
    chroma_dir = os.path.join(base_dir, "chroma_db")
    db_client = chromadb.PersistentClient(path=chroma_dir)
    
    try:
        db_client.delete_collection("mitre_techniques")
        print("Purged existing polluted collection.")
    except Exception:
        pass
        
    collection = db_client.create_collection("mitre_techniques")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    docs = []
    metas = []
    ids = []
    
    for r in valid_rows:
        text = f"Technique: {r['name']}\nDescription: {r['desc']}"
        docs.append(text)
        metas.append({"name": r['name']})
        ids.append(r['id'])
        
    print("Embedding vectors...")
    embeddings = model.encode(docs, show_progress_bar=True).tolist()
    
    print("Injecting clean data into ChromaDB...")
    collection.add(
        embeddings=embeddings,
        documents=docs,
        metadatas=metas,
        ids=ids
    )
    print("Cleansing complete! Candidate pool is 100% pure.")

if __name__ == "__main__":
    cleanse_chroma()
