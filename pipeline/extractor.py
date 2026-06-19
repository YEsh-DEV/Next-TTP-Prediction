import os
import json
from typing import Optional
from google import genai
from google.genai import types
from schemas.extraction_schema import ExtractionResult

PROMPT_TEMPLATE = """
You are an expert Cyber Threat Intelligence analyst.
Your task is to extract a strictly structured Attack Knowledge Graph from the provided CTI report chunk.

RULES:
1. NO HALLUCINATION. Only extract what is explicitly stated in the chunk.
2. EVENT SEQUENCING: Break down the attack into meaningful actions ("events"). Use the PRECEDES relationship to chronologically link events if a sequence is observed.
3. EVIDENCE: Every extracted event, relationship, and entity must be mapped to an exact `text_snippet` from the provided chunk.
4. CONFIDENCE: Assign a float (0.0 to 1.0) to relationships and events based on the explicitness of the text.

<REPORT_CHUNK>
{text_chunk}
</REPORT_CHUNK>

Generate the extraction matching the requested JSON schema.
"""

class Extractor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def extract_from_chunk(self, chunk_text: str) -> ExtractionResult:
        if not self.client:
            raise ValueError("GEMINI_API_KEY is missing. Cannot perform extraction.")
        
        prompt = PROMPT_TEMPLATE.format(text_chunk=chunk_text)
        
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResult,
                temperature=0.1
            ),
        )
        
        return ExtractionResult.model_validate_json(response.text)
