import os
import json
import chromadb
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types
from dotenv import load_dotenv

from schemas.cti_schema import CTIEvent
from pipeline.embedder import CTIEmbedder

class TechniqueMatch(BaseModel):
    technique_id: str = Field(..., description="MITRE Technique ID (e.g., T1043)")
    confidence: float = Field(..., description="Probability match score from 0.0 to 1.0")

class ClassificationResult(BaseModel):
    techniques: List[TechniqueMatch] = Field(default_factory=list, description="Top matched MITRE Techniques with confidence scores.")

class HybridClassifier:
    def __init__(self, base_dir: str):
        load_dotenv(os.path.join(base_dir, ".env"))
        self.api_key = os.environ.get("GEMINI_AUTH_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_AUTH_API_KEY or GEMINI_API_KEY is not set.")
        
        self.client = genai.Client(api_key=self.api_key)
        
        # Initialize Vector Store for Candidate Retrieval
        self.chroma_dir = os.path.join(base_dir, "chroma_db")
        self.db_client = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection = self.db_client.get_collection(name="mitre_techniques")
        self.embedder = CTIEmbedder()
            
    def classify_event(self, cti_event: CTIEvent) -> List[TechniqueMatch]:
        """
        Passes the MISP attributes to ChromaDB for candidate retrieval,
        then asks Gemini to probabilistically rank the top candidates.
        """
        categories = list(set(a.category for a in cti_event.attributes if a.category))
        indicator_types = list(set(a.type for a in cti_event.attributes if a.type))
        
        event_summary = f"""
        Event Title/Info: {cti_event.info}
        Observed Indicator Categories: {categories}
        Observed Indicator Types: {indicator_types}
        """
        
        # Candidate Retrieval via ChromaDB
        query_embedding = self.embedder.embed_texts([event_summary])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=30
        )
        
        candidates = "\n".join(results['documents'][0]) if results['documents'] else "No candidates found."
        
        prompt = f"""You are a precise MITRE ATT&CK classifier.
        
You will be provided with a summary of a MISP Cyber Threat Intelligence event, and a list of the Top 10 candidate MITRE Techniques retrieved from our vector database.
Your task is to select the best matching MITRE Techniques and assign a mathematical probability (confidence score from 0.0 to 1.0).

### Top 10 Candidate MITRE Techniques:
{candidates}

### Event to Classify:
{event_summary}

### Instructions:
1. Review the Observed Indicator Categories (e.g., 'Network activity') and Types (e.g., 'url', 'ip-src').
2. Match this behavior against the provided Candidate MITRE Techniques.
3. Return the selected Technique IDs and your confidence score in a JSON array.
4. DO NOT hallucinate IDs. Only use IDs from the provided candidate list. If none match, return an empty list.
"""
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ClassificationResult,
                    temperature=0.1
                ),
            )
            result = ClassificationResult.model_validate_json(response.text)
            return result.techniques
        except Exception as e:
            print(f"Classification failed: {e}")
            return []
