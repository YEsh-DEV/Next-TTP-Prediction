import os
import sys
import time
import json
from dotenv import load_dotenv
import chromadb

# Ensure correct path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from pipeline.extractor import Extractor, CTIValidationError

def run_sample_extraction():
    print("--- Phase-3 End-to-End Extraction Test ---")
    
    # 1. Load Environment Variables (including GEMINI_AUTH_API_KEY)
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    api_key = os.environ.get("GEMINI_AUTH_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_AUTH_API_KEY not found in .env")
        sys.exit(1)
        
    print("API Key loaded successfully.")

    # 2. Retrieve Context from ChromaDB
    print("\nConnecting to ChromaDB...")
    chroma_dir = os.path.join(BASE_DIR, "chroma_db")
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection("cti_reports")
    
    # Get a single chunk
    results = collection.get(limit=1)
    
    if not results['documents']:
        print("ERROR: No narrative chunks found in ChromaDB. Please run ingestion first.")
        sys.exit(1)
        
    chunk_text = results['documents'][0]
    metadata = results['metadatas'][0]
    
    print("\n--- A. Input Event Information ---")
    print(f"Event ID: {metadata.get('event_id')}")
    print(f"Source File: {metadata.get('source_file')}")
    print(f"Event Type: {metadata.get('event_type')}")
    print(f"Context Text Preview: {chunk_text[:150]}...")
    
    # 3. Send Context to Gemini 2.5 Flash
    extractor = Extractor(api_key=api_key)
    
    print("\n--- B. Gemini Request Statistics ---")
    print("Sending request to Gemini 2.5 Flash...")
    start_time = time.time()
    
    # We will wrap in try/except to catch validation errors
    validation_status = "PASSED"
    validation_error = None
    extracted_result = None
    
    try:
        extracted_result = extractor.extract_from_chunk(chunk_text)
    except CTIValidationError as e:
        validation_status = "FAILED"
        validation_error = str(e)
    except Exception as e:
        validation_status = "ERROR"
        validation_error = str(e)
        
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Assuming prompt size is roughly proportional to chunk length in characters
    prompt_size = len(chunk_text)
    response_size = len(extracted_result.model_dump_json()) if extracted_result else 0
    
    print(f"Prompt Size (Characters): ~{prompt_size}")
    print(f"Response Size (Characters): ~{response_size}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    
    print("\n--- F. Validation Checks ---")
    print(f"Validation Layer Status: {validation_status}")
    if validation_error:
        print(f"Details: {validation_error}")
        
    if not extracted_result:
        print("\nExtraction failed. Cannot provide statistics.")
        sys.exit(1)
        
    # 4. Extraction Statistics
    print("\n--- C. Extraction Statistics ---")
    print(f"Entities Extracted: {len(extracted_result.entities)}")
    print(f"Events Extracted: {len(extracted_result.events)}")
    print(f"Relationships Extracted: {len(extracted_result.relationships)}")
    print(f"Temporal Edges Extracted: {len(extracted_result.temporal_causal_edges)}")
    print(f"Evidence Count: {len(extracted_result.evidence)}")
    
    # 5. Save Final JSON Output
    output_path = os.path.join(os.environ.get("APPDATA_DIR", BASE_DIR), "sample_extraction.json")
    # For Antigravity artifacts, saving directly to the conversation's scratch space if we were given it, 
    # but we'll save it to the current directory so the agent can read it, and print it.
    output_path = os.path.join(BASE_DIR, "scratch", "sample_extraction.json")
    
    with open(output_path, "w") as f:
        f.write(extracted_result.model_dump_json(indent=4))
        
    print("\n--- E. Confirm Output File Location ---")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    run_sample_extraction()
