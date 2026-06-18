import chromadb

class ChromaManager:
    def __init__(self, persist_directory: str):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="cti_reports",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        if not chunks:
            return
            
        ids = [f"evt_{c['event_id']}_chunk_{c['chunk_index']}" for c in chunks]
        documents = [c['text'] for c in chunks]
        metadatas = [{"event_id": c["event_id"], "date": c["date"], "info": c["info"], "chunk_index": c["chunk_index"]} for c in chunks]
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def count(self):
        return self.collection.count()
