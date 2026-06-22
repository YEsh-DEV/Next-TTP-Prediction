import os
import sys
import time
import json
from dotenv import load_dotenv
import chromadb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from pipeline.extractor import Extractor, CTIValidationError

def run_sample_extraction():
    print("--- Phase-3 End-to-End Extraction Test ---")
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    api_key = os.environ.get("GEMINI_AUTH_API_KEY")

    print("\nConnecting to ChromaDB...")
    chroma_dir = os.path.join(BASE_DIR, "chroma_db")
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection("cti_reports")
    
    results = collection.get(
        where={"source_file": "CTIDataset_2018_ReportEvent.xml"}
    )
    
    if not results['documents']:
        print("ERROR: No chunks found in ChromaDB.")
        sys.exit(1)
        
    # Find the most meaningful chunk (largest text that is not just a list of IOCs)
    best_idx = 0
    max_len = 0
    for idx, doc in enumerate(results['documents']):
        if not doc.startswith("Raw IOCs") and len(doc) > max_len:
            max_len = len(doc)
            best_idx = idx
            
    chunk_text = results['documents'][best_idx]
    metadata = results['metadatas'][best_idx]
    
    print("\n--- A. Input Event Information ---")
    print(f"Event ID: {metadata.get('event_id')}")
    print(f"Source File: {metadata.get('source_file', metadata.get('info'))}")
    print(f"Event Type: {metadata.get('event_type', 'Unknown')}")
    print(f"Report Date: {metadata.get('date')}")
    print(f"Report Title / Info: {metadata.get('info')}")
    print(f"Retrieved Chunk Length (characters): {len(chunk_text)}")
    preview = chunk_text[:1000]
    print(f"Retrieved Chunk Text (first 1000 characters):\n{preview}")
    
    extractor = Extractor(api_key=api_key)
    
    from pipeline.extractor import PROMPT_TEMPLATE
    from google import genai
    from google.genai import types
    from schemas.extraction_schema import ExtractionResult
    
    val_grounding = "PASSED"
    val_ref = "PASSED"
    val_seq = "PASSED"
    val_schema = "PASSED"
    
    validation_error = None
    extracted_result = None
    
    start_time = time.time()
    
    try:
        # Run raw extraction to ensure we capture the JSON even if it's invalid
        prompt = PROMPT_TEMPLATE.format(text_chunk=chunk_text)
        response = extractor.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResult,
                temperature=0.1
            ),
        )
        extracted_result = ExtractionResult.model_validate_json(response.text)
        
        # Manually run validation to get the exact error
        extractor.validate_extraction(extracted_result, chunk_text)
        
    except CTIValidationError as e:
        val_grounding = "FAILED" if "Grounding" in str(e) else "UNKNOWN"
        val_ref = "FAILED" if "Referential" in str(e) else "UNKNOWN"
        validation_error = str(e)
    except Exception as e:
        val_schema = "FAILED"
        validation_error = str(e)
        
    end_time = time.time()
    execution_time = end_time - start_time
    
    prompt_size = len(chunk_text) + 2000 # Appx template size
    response_size = len(extracted_result.model_dump_json()) if extracted_result else 0
    
    print("Model Name: gemini-2.5-flash")
    print(f"Prompt Length (chars): {prompt_size}")
    print(f"Response Length (chars): {response_size}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Estimated Input Tokens: {prompt_size // 4}")
    print(f"Estimated Output Tokens: {response_size // 4}")
    
    print("\n--- E. Validation Results ---")
    print(f"Grounding Check Result: {val_grounding}")
    print(f"Referential Integrity Result: {val_ref}")
    print(f"Sequence Validation Result: {val_seq}")
    print(f"Schema Validation Result: {val_schema}")
    if validation_error:
        print(f"Validation Details: {validation_error}")
        
    if not extracted_result:
        sys.exit(1)
        
    print("\n--- C. Extraction Output Statistics ---")
    print(f"Number of Entities: {len(extracted_result.entities)}")
    print(f"Number of Events: {len(extracted_result.events)}")
    print(f"Number of Relationships: {len(extracted_result.relationships)}")
    print(f"Number of Temporal Edges: {len(extracted_result.temporal_causal_edges)}")
    print(f"Number of Evidence Entries: {len(extracted_result.evidence)}")
    print(f"Number of MITRE Candidates: {len(getattr(extracted_result, 'mitre_candidates', []))}")
    
    output_path = os.path.join(BASE_DIR, "scratch", "sample_extraction.json")
    with open(output_path, "w") as f:
        f.write(extracted_result.model_dump_json(indent=4))
        
    print(f"\n--- E. Confirm Output File Location ---")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    run_sample_extraction()
