from enum import Enum
from pydantic import BaseModel, Field
from typing import List

class EntityType(str, Enum):
    THREAT_ACTOR = "ThreatActor"
    MALWARE = "Malware"
    TOOL = "Tool"
    TECHNIQUE = "Technique"
    TACTIC = "Tactic"
    VICTIM = "Victim"
    INFRASTRUCTURE = "Infrastructure"
    VULNERABILITY = "Vulnerability"
    CAMPAIGN = "Campaign"
    ORGANIZATION = "Organization"
    FILE_ARTIFACT = "FileArtifact"

class RelationshipType(str, Enum):
    USES = "USES"
    EMPLOYS_TECHNIQUE = "EMPLOYS_TECHNIQUE"
    TARGETS = "TARGETS"
    EXPLOITS = "EXPLOITS"
    COMMUNICATES_WITH = "COMMUNICATES_WITH"
    PARTICIPATED_IN = "PARTICIPATED_IN"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"
    PRECEDES = "PRECEDES"

class Metadata(BaseModel):
    source_file: str = Field(..., description="The source XML filename")
    report_date: str = Field(..., description="Date of the report/event")

class Entity(BaseModel):
    entity_id: str = Field(..., description="Unique ID for this entity within the extraction (e.g., 'ent_1')")
    type: EntityType = Field(..., description="The type of entity, constrained to the specific enum")
    value: str = Field(..., description="The name or value of the entity (e.g., 'Patchwork', 'PowerShell')")

class Event(BaseModel):
    event_id: str = Field(..., description="Unique ID for this event within the extraction (e.g., 'evt_1')")
    action: str = Field(..., description="A meaningful attack action (e.g., 'PowerShell Execution')")
    entities: List[str] = Field(default_factory=list, description="List of entity_ids involved in this event")

class Relationship(BaseModel):
    source: str = Field(..., description="The ID of the source entity or event")
    type: RelationshipType = Field(..., description="The relationship type, constrained to the enum")
    target: str = Field(..., description="The ID of the target entity or event")
    confidence: float = Field(..., description="A confidence score between 0.0 and 1.0", ge=0.0, le=1.0)

class MitreCandidate(BaseModel):
    entity_id: str = Field(..., description="The ID of the entity this maps to")
    suggested_technique: str = Field(..., description="The MITRE ATT&CK Technique ID (e.g., 'T1059')")
    reasoning: str = Field(..., description="Brief reasoning for this mapping")

class Evidence(BaseModel):
    target_id: str = Field(..., description="The ID of the entity, event, or relationship this evidence supports")
    text_snippet: str = Field(..., description="The exact text snippet from the chunk")

class ExtractionResult(BaseModel):
    metadata: Metadata
    entities: List[Entity] = Field(default_factory=list)
    events: List[Event] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    mitre_candidates: List[MitreCandidate] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
