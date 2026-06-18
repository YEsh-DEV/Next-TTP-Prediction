import os
import glob
import json
from tqdm import tqdm
from pipeline.xml_parser import parse_xml_file
from pipeline.chunker import CTIChunker
from pipeline.embedder import CTIEmbedder
from pipeline.vector_store import ChromaManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "CTI_Report_Dataset")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
CACHE_FILE = os.path.join(BASE_DIR, "ingestion_cache.json")

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {"processed_files": []}

def save_cache(cache_data):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=4)

def run_ingestion(limit_files=None):
    print("--- CTI Dataset Infrastructure Orchestrator ---")
    
    xml_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.xml")))
    if not xml_files:
        print("No XML files found in CTI_Report_Dataset.")
        return
    
    cache = load_cache()
    chunker = CTIChunker(batch_size=20)
    embedder = CTIEmbedder()
    vector_store = ChromaManager(CHROMA_DIR)
    
    stats = {
        "files_processed": 0,
        "events_parsed": 0,
        "attributes_mapped": 0,
        "chunks_embedded": 0
    }
    
    files_to_process = [f for f in xml_files if os.path.basename(f) not in cache["processed_files"]]
    print(f"Found {len(xml_files)} total files. {len(files_to_process)} remaining to process.")
    
    if limit_files is not None:
        files_to_process = files_to_process[:limit_files]
        print(f"Limiting to {limit_files} files for trial run.")

    for filepath in files_to_process:
        filename = os.path.basename(filepath)
        print(f"\nProcessing: {filename}")
        
        events = parse_xml_file(filepath)
        stats["events_parsed"] += len(events)
        
        file_chunks = []
        for event in events:
            stats["attributes_mapped"] += len(event.attributes)
            chunks = chunker.chunk_event(event)
            file_chunks.extend(chunks)
            
        print(f"  Extracted {len(file_chunks)} chunks. Embedding...")
        
        if file_chunks:
            # Embed in batches
            batch_size = 256
            for i in tqdm(range(0, len(file_chunks), batch_size), desc="Embedding Batches"):
                batch = file_chunks[i:i+batch_size]
                texts = [c['text'] for c in batch]
                embeddings = embedder.embed_texts(texts)
                vector_store.add_chunks(batch, embeddings)
                stats["chunks_embedded"] += len(batch)
                
        cache["processed_files"].append(filename)
        save_cache(cache)
        stats["files_processed"] += 1

    print("\n--- Phase-2 Statistics ---")
    print(f"Files Processed: {stats['files_processed']}")
    print(f"Events Parsed:   {stats['events_parsed']}")
    print(f"Attributes:      {stats['attributes_mapped']}")
    print(f"Chunks in DB:    {vector_store.count()}")

if __name__ == "__main__":
    # We run a trial on 2 files to quickly validate
    run_ingestion(limit_files=2)
