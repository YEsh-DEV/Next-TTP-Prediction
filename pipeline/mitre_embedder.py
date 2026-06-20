import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import pandas as pd
import chromadb
from pipeline.embedder import CTIEmbedder

class MitreEmbedder:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.mitre_enterprise_path = os.path.join(base_dir, "MitreEnterprise.xlsx")
        self.chroma_dir = os.path.join(base_dir, "chroma_db")
        
        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name="mitre_techniques",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = CTIEmbedder()

    def embed_mitre(self):
        try:
            df = pd.read_excel(self.mitre_enterprise_path)
            # Ensure required columns exist
            df = df[['Tactic ID', 'Tactic Name', 'Description']].dropna(subset=['Tactic ID'])
            df = df.drop_duplicates(subset=['Tactic ID'])
        except Exception as e:
            print(f"Failed to load MitreEnterprise.xlsx: {e}")
            return
            
        print(f"Loaded {len(df)} MITRE Techniques. Embedding...")
        
        ids = []
        documents = []
        metadatas = []
        
        for _, row in df.iterrows():
            tech_id = str(row['Tactic ID'])
            name = str(row['Tactic Name'])
            desc = str(row.get('Description', ''))
            
            # Create a rich semantic document for embedding
            text_to_embed = f"Technique ID: {tech_id}\nName: {name}\nDescription: {desc}"
            
            ids.append(tech_id)
            documents.append(text_to_embed)
            metadatas.append({
                "Tactic ID": tech_id,
                "Tactic Name": name
            })
            
        # Embed in batches to avoid memory issues
        embeddings = self.embedder.embed_texts(documents)
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Successfully embedded {self.collection.count()} MITRE techniques into ChromaDB.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embedder = MitreEmbedder(base_dir)
    embedder.embed_mitre()
