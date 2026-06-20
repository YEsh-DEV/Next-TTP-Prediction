from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

class EntityType(str, Enum):
    THREAT_ACTOR = "ThreatActor"
    CAMPAIGN = "Campaign"
    MALWARE = "Malware"
    TOOL = "Tool"
    VICTIM = "Victim"
    INFRASTRUCTURE = "Infrastructure"
    VULNERABILITY = "Vulnerability"
    FILE_ARTIFACT = "FileArtifact"
    TACTIC = "Tactic"
    TECHNIQUE = "Technique"

class RelationshipType(str, Enum):
    USES = "USES"                          # Actor/Malware USES Tool/Technique
    TARGETS = "TARGETS"                    # Actor/Malware TARGETS Victim/Infrastructure
    EXPLOITS = "EXPLOITS"                  # Malware/Actor EXPLOITS Vulnerability
    COMMUNICATES_WITH = "COMMUNICATES_WITH"# Malware/Infrastructure COMMUNICATES_WITH Infrastructure
    PARTICIPATED_IN = "PARTICIPATED_IN"    # Actor PARTICIPATED_IN Campaign
    CO_OCCURS_WITH = "CO_OCCURS_WITH"      # Entities found together without explicit action
    PRECEDES = "PRECEDES"                  # Event PRECEDES another event (can also be covered by TemporalCausalEdge)
    OBSERVED_TECHNIQUE = "OBSERVED_TECHNIQUE" # Event observed probabilistically mapping to a Technique

class TemporalCausalType(str, Enum):
    PRECEDES = "PRECEDES"                  # Event A is chronologically before Event B
    CAUSES = "CAUSES"                      # Event A logically/physically causes Event B to happen
    TRIGGERS = "TRIGGERS"                  # Event A triggers automated/instant execution of Event B

class Metadata(BaseModel):
    source_file: str = Field(..., description="The source XML filename")
    report_date: Optional[str] = Field(None, description="Absolute date/time of the report or events if known (ISO 8601)")
    extraction_timestamp: str = Field(..., description="Timestamp of when extraction was executed")

class Entity(BaseModel):
    entity_id: str = Field(..., description="Unique alphanumeric ID (e.g., 'ent_01', 'ent_02')")
    type: EntityType = Field(..., description="Type of entity classification")
    value: str = Field(..., description="Canonical name or identifier of the entity (e.g., 'APT41', 'Mimikatz', 'CVE-2020-0601')")
    aliases: List[str] = Field(default_factory=list, description="Alternative names discovered in the text")
    description: Optional[str] = Field(None, description="Short context description from the report text")

class Event(BaseModel):
    event_id: str = Field(..., description="Unique event identifier (e.g., 'evt_01')")
    action: str = Field(..., description="Core cyber attack action/behavior (e.g., 'LSASS Memory Dump', 'C2 Domain Beaconing')")
    timestamp: Optional[str] = Field(None, description="Absolute or relative timestamp of the event (ISO 8601 or offsets like 'T+12m')")
    sequence_number: int = Field(..., description="Chronological ordering index of the event, starting at 1")
    actor_id: Optional[str] = Field(None, description="The entity_id of the entity executing the action (e.g., ThreatActor or Malware)")
    target_ids: List[str] = Field(default_factory=list, description="List of entity_ids representing the targets of this action")
    mitre_technique_id: Optional[str] = Field(None, description="Standard MITRE Technique ID mapped to this event (e.g., 'T1003.001')")
    description: str = Field(..., description="Detailed description of what occurred during this event")
    indicators: List[str] = Field(default_factory=list, description="Array of single-use infrastructure indicators (IPs, URLs) compressed into the event to prevent graph explosion.")

class Relationship(BaseModel):
    relationship_id: str = Field(..., description="Unique relationship identifier (e.g., 'rel_01')")
    source_id: str = Field(..., description="The ID of the source entity or event")
    target_id: str = Field(..., description="The ID of the target entity or event")
    type: RelationshipType = Field(..., description="Relationship type enum")
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0 based on text clarity", ge=0.0, le=1.0)

class TemporalCausalEdge(BaseModel):
    edge_id: str = Field(..., description="Unique edge identifier (e.g., 'tc_01')")
    source_event_id: str = Field(..., description="The event_id of the cause/predecessor event")
    target_event_id: str = Field(..., description="The event_id of the effect/successor event")
    type: TemporalCausalType = Field(..., description="The temporal/causal relationship between events")
    time_delta: Optional[str] = Field(None, description="Time difference description if explicitly stated (e.g., '10 minutes later', 'immediately')")
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0", ge=0.0, le=1.0)

class Evidence(BaseModel):
    evidence_id: str = Field(..., description="Unique evidence ID (e.g., 'ev_01')")
    target_id: str = Field(..., description="The ID of the entity, event, relationship, or temporal edge this evidence supports")
    text_snippet: str = Field(..., description="The EXACT verbatim text quotation from the report justifying the element")

class ExtractionResult(BaseModel):
    metadata: Metadata
    entities: List[Entity] = Field(default_factory=list)
    events: List[Event] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    temporal_causal_edges: List[TemporalCausalEdge] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
