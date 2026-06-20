import os
import sys
import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.xml_parser import parse_xml_file

class CTIEmbedder:
    def __init__(self, base_dir: str):
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(base_dir, "chroma_db"))
        self.collection = self.chroma_client.get_or_create_collection(name="cti_events")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.xml_path = os.path.join(base_dir, "CTI_Report_Dataset", "CTIDataset_2018_ReportEvent.xml")

    def format_event_text(self, event) -> str:
        categories = list(set([attr.category for attr in event.attributes if attr.category]))
        types = list(set([attr.type for attr in event.attributes if attr.type]))
        # Keep descriptions brief so they fit in the semantic window
        text = f"Event ID: {event.event_id}\n"
        text += f"Report Date: {event.date}\n"
        text += f"Event Title/Info: {event.info}\n"
        text += f"Observed Indicator Categories: {categories}\n"
        text += f"Observed Indicator Types: {types}"
        return text

    def embed_dataset(self):
        print("Parsing XML dataset...")
        events = parse_xml_file(self.xml_path)
        print(f"Loaded {len(events)} events.")

        # Prepare documents, metadatas, ids
        documents = []
        metadatas = []
        ids = []

        for evt in events:
            doc_text = self.format_event_text(evt)
            documents.append(doc_text)
            metadatas.append({
                "event_id": str(evt.event_id),
                "date": str(evt.date),
                "info": str(evt.info),
                "attribute_count": len(evt.attributes)
            })
            ids.append(f"evt_{evt.event_id}")

        print("Embedding events into ChromaDB in batches...")
        batch_size = 5000
        for i in tqdm(range(0, len(documents), batch_size)):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]

            # Generate vectors
            embeddings = self.model.encode(batch_docs, show_progress_bar=False).tolist()

            # Upsert to ChromaDB
            self.collection.upsert(
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_metas,
                ids=batch_ids
            )

        print("Embedding complete!")

    def generate_validation_stats(self):
        count = self.collection.count()
        print("\n--- CHROMADB VALIDATION STATS ---")
        print(f"1. collection.count(): {count}")
        print(f"2. Number of unique events: {count} (1-to-1 mapping)")
        print(f"3. Number of chunks: {count} (No semantic chunking required due to brief MISP attribute text)")
        
        peek = self.collection.peek(5)
        print("\n4. collection.peek(5):")
        for i in range(len(peek['ids'])):
            print(f"  ID: {peek['ids'][i]}")
            print(f"  Metadata: {peek['metadatas'][i]}")
            print(f"  Document Snippet: {peek['documents'][i][:100]}...")
            
        print("\n5. Metadata schema:")
        print("  - event_id (str)")
        print("  - date (str)")
        print("  - info (str)")
        print("  - attribute_count (int)")
        print("-----------------------------------")

if __name__ == "__main__":
    embedder = CTIEmbedder(base_dir)
    # Check if we already embedded it to save time
    if embedder.collection.count() < 1000:
        embedder.embed_dataset()
    else:
        print("Dataset already embedded! Skipping encode phase.")
        
    embedder.generate_validation_stats()
