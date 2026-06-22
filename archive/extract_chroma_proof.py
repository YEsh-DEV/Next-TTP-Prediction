import os
import chromadb

def extract_chroma_proof():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chroma_client = chromadb.PersistentClient(path=os.path.join(base_dir, "chroma_db"))
    collection = chroma_client.get_collection(name="cti_events")
    
    output_path = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\submission_chroma_proof.md"
    
    count = collection.count()
    peek = collection.peek(10)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# TASK 1 — REAL CHROMADB PROOF\n\n")
        
        f.write("## 1. collection.count()\n")
        f.write(f"```text\n{count}\n```\n\n")
        
        f.write("## 2. collection.peek(10)\n")
        f.write(f"Total returned in peek: {len(peek['ids'])}\n")
        f.write("```json\n")
        import json
        peek_subset = {"ids": peek["ids"], "metadatas": peek["metadatas"]}
        f.write(json.dumps(peek_subset, indent=2))
        f.write("\n```\n\n")
        
        f.write("## 3. 5 Random Records (Sampled from peek)\n")
        for i in range(5):
            f.write(f"**Record {i+1} (ID: {peek['ids'][i]})**\n")
            f.write(f"- **Metadata:** {peek['metadatas'][i]}\n")
            f.write(f"- **Document Snippet:** {peek['documents'][i][:150]}...\n\n")
            
        f.write("## 4. Metadata Schema\n")
        f.write("Extracted automatically from the first document's metadata keys:\n")
        f.write("```json\n")
        schema = {k: type(v).__name__ for k, v in peek['metadatas'][0].items()}
        f.write(json.dumps(schema, indent=2))
        f.write("\n```\n")
        
    print(f"Proof saved to {output_path}")

if __name__ == "__main__":
    extract_chroma_proof()
