import os
import sys
import json

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from pipeline.query_pipeline import GraphRAGPipeline

def run_deterministic_validation():
    print("Initializing Deterministic GraphRAG Pipeline...")
    pipeline = GraphRAGPipeline(base_dir)
    
    queries = [
        "Burning Umbrella",
        "Operation Kitty",
        "Lazarus Cryptocurrency"
    ]
    
    output_path = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\deterministic_system_validation.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# TASK 5 — DETERMINISTIC SYSTEM VALIDATION\n\n")
        f.write("Executing the fully assembled Query Pipeline in 100% Deterministic Mode (Offline Vectors ONLY) to extract Graph Subgraphs.\n\n")
        
        for q in queries:
            print(f"Executing Query: '{q}'...")
            f.write(f"## Target Query: `{q}`\n")
            
            try:
                result = pipeline.execute_query(q, top_k=3, classification_mode="deterministic")
                f.write("```json\n")
                f.write(json.dumps(result, indent=4) + "\n")
                f.write("```\n\n")
            except Exception as e:
                f.write("```json\n")
                f.write(json.dumps({"error": str(e)}, indent=4) + "\n")
                f.write("```\n\n")
                
    print(f"Proof saved to {output_path}")

if __name__ == "__main__":
    run_deterministic_validation()
