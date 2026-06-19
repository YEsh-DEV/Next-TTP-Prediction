import os
import json
from typing import Optional
from google import genai
from google.genai import types
from schemas.extraction_schema import ExtractionResult

PROMPT_TEMPLATE = """
You are an elite Cyber Threat Intelligence (CTI) analyst and Knowledge Graph extraction engine.
Your task is to parse the provided threat intelligence report chunk and extract a highly structured, temporal-causal Attack Knowledge Graph.

The output must conform strictly to the provided JSON schema.

### Core Objectives:
1. **Entity Extraction**: Identify threat actors, malware, tools, infrastructure, tactics, techniques, file artifacts, vulnerabilities, campaigns, and victims.
2. **Event Sequencing**: Identify specific actions (e.g., process injection, network connections) as "Events". Assign sequential `sequence_number` attributes representing chronological order of execution.
3. **Causal Mapping**: Map temporal-causal dependencies between Events using PRECEDES, CAUSES, or TRIGGERS.
4. **Relational Links**: Construct relationship triplets (USES, TARGETS, EXPLOITS, etc.) linking entities to other entities or events.
5. **MITRE Alignment**: Align event actions to standard MITRE ATT&CK technique IDs (e.g., T1059) based on behavior details.
6. **Strict Grounding**: For EVERY entity, event, relationship, or temporal-causal edge, extract the EXACT raw text snippet supporting it as evidence.

### Strict Extraction Rules:
* **NO HALLUCINATIONS**: Do not assume details. If details (like timestamps or technique mappings) are not mentioned or highly implied, leave them null.
* **Chronological Integrity**: Event sequence numbers must start at 1 and increment continuously.
* **Identifier Consistency**: Every ID must remain consistent throughout the document.
* **Exact Evidence**: The `text_snippet` must be present EXACTLY as a substring in the input report text.

<REPORT_CHUNK>
{text_chunk}
</REPORT_CHUNK>

Generate the extraction matching the requested JSON schema.
"""

class CTIValidationError(ValueError):
    """Custom exception raised when CTI extraction fails validation checks."""
    pass

class Extractor:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_AUTH_API_KEY") or os.environ.get("GEMINI_API_KEY")
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
        
        result = ExtractionResult.model_validate_json(response.text)
        return self.validate_extraction(result, chunk_text)

    def validate_extraction(self, result: ExtractionResult, chunk_text: str) -> ExtractionResult:
        """
        Validates the extracted CTI graph:
        1. Exact substring check for evidence grounding.
        2. Referential integrity check (relationships, edges, events reference existing IDs).
        3. Chronological sequence number consistency and normalization.
        """
        entity_ids = {e.entity_id for e in result.entities}
        event_ids = {e.event_id for e in result.events}
        relationship_ids = {r.relationship_id for r in result.relationships}
        edge_ids = {t.edge_id for t in result.temporal_causal_edges}
        
        all_element_ids = entity_ids | event_ids | relationship_ids | edge_ids
        
        # 1. Grounding check
        for ev in result.evidence:
            if ev.text_snippet not in chunk_text:
                raise CTIValidationError(
                    f"Grounding check failed: Snippet '{ev.text_snippet}' was not found as a verbatim substring in the source text."
                )
            if ev.target_id not in all_element_ids:
                raise CTIValidationError(
                    f"Referential integrity failed: Evidence {ev.evidence_id} references a non-existent target ID {ev.target_id}."
                )

        # 2. Referential integrity of relationships
        for r in result.relationships:
            if r.source_id not in all_element_ids or r.target_id not in all_element_ids:
                raise CTIValidationError(
                    f"Referential integrity failed: Relationship {r.relationship_id} references missing IDs (source: {r.source_id}, target: {r.target_id})"
                )

        # 3. Referential integrity of temporal edges
        for edge in result.temporal_causal_edges:
            if edge.source_event_id not in event_ids or edge.target_event_id not in event_ids:
                raise CTIValidationError(
                    f"Referential integrity failed: Temporal edge {edge.edge_id} references missing events (source: {edge.source_event_id}, target: {edge.target_event_id})"
                )

        # 4. Referential integrity of event actor and targets
        for event in result.events:
            if event.actor_id and event.actor_id not in entity_ids:
                raise CTIValidationError(
                    f"Referential integrity failed: Event {event.event_id} references missing actor {event.actor_id}"
                )
            for t_id in event.target_ids:
                if t_id not in entity_ids:
                    raise CTIValidationError(
                        f"Referential integrity failed: Event {event.event_id} references missing target {t_id}"
                    )

        # 5. Chronological sequence number normalization
        if result.events:
            sorted_events = sorted(result.events, key=lambda e: e.sequence_number)
            for idx, event in enumerate(sorted_events):
                event.sequence_number = idx + 1
            result.events = sorted_events

        return result
