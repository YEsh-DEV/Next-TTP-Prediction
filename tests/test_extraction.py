import pytest
from schemas.extraction_schema import (
    ExtractionResult, Metadata, Entity, Event, Relationship,
    TemporalCausalEdge, Evidence, EntityType, RelationshipType, TemporalCausalType
)
from pipeline.extractor import Extractor, CTIValidationError

def test_valid_extraction_validation():
    chunk_text = "APT41 used Cobalt Strike to breach Global Tech Corp on 2026-06-19."
    
    metadata = Metadata(
        source_file="report.xml",
        extraction_timestamp="2026-06-20T00:00:00Z"
    )
    entities = [
        Entity(entity_id="ent_01", type=EntityType.THREAT_ACTOR, value="APT41"),
        Entity(entity_id="ent_02", type=EntityType.MALWARE, value="Cobalt Strike"),
        Entity(entity_id="ent_03", type=EntityType.VICTIM, value="Global Tech Corp")
    ]
    events = [
        Event(
            event_id="evt_01",
            action="Breach network",
            sequence_number=5, # Will test sorting and re-indexing!
            actor_id="ent_01",
            target_ids=["ent_03"],
            description="Breached Global Tech Corp"
        ),
        Event(
            event_id="evt_02",
            action="Deploy malware",
            sequence_number=2, # Will be re-indexed to 1, and evt_01 will be re-indexed to 2
            actor_id="ent_01",
            target_ids=["ent_02"],
            description="Used Cobalt Strike"
        )
    ]
    relationships = [
        Relationship(
            relationship_id="rel_01",
            source_id="ent_01",
            target_id="ent_02",
            type=RelationshipType.USES,
            confidence=0.95
        )
    ]
    temporal_edges = [
        TemporalCausalEdge(
            edge_id="tc_01",
            source_event_id="evt_02",
            target_event_id="evt_01",
            type=TemporalCausalType.PRECEDES,
            confidence=0.90
        )
    ]
    evidence = [
        Evidence(
            evidence_id="ev_01",
            target_id="ent_01",
            text_snippet="APT41"
        ),
        Evidence(
            evidence_id="ev_02",
            target_id="ent_02",
            text_snippet="Cobalt Strike"
        ),
        Evidence(
            evidence_id="ev_03",
            target_id="ent_03",
            text_snippet="Global Tech Corp"
        )
    ]
    
    result = ExtractionResult(
        metadata=metadata,
        entities=entities,
        events=events,
        relationships=relationships,
        temporal_causal_edges=temporal_edges,
        evidence=evidence
    )
    
    extractor = Extractor()
    validated = extractor.validate_extraction(result, chunk_text)
    
    # Check sequence re-indexing
    # evt_02 has sequence_number=2, evt_01 has sequence_number=5
    # After sort and re-index, evt_02 must be first (seq=1), evt_01 must be second (seq=2)
    assert validated.events[0].event_id == "evt_02"
    assert validated.events[0].sequence_number == 1
    assert validated.events[1].event_id == "evt_01"
    assert validated.events[1].sequence_number == 2
    
    # Check that they exist
    assert len(validated.entities) == 3
    assert len(validated.relationships) == 1
    assert len(validated.temporal_causal_edges) == 1
    assert len(validated.evidence) == 3

def test_grounding_failure():
    chunk_text = "APT41 breached Global Tech Corp."
    
    # We include Cobalt Strike in evidence but it is NOT in chunk_text
    metadata = Metadata(
        source_file="report.xml",
        extraction_timestamp="2026-06-20T00:00:00Z"
    )
    entities = [
        Entity(entity_id="ent_01", type=EntityType.THREAT_ACTOR, value="APT41"),
        Entity(entity_id="ent_02", type=EntityType.MALWARE, value="Cobalt Strike")
    ]
    evidence = [
        Evidence(
            evidence_id="ev_01",
            target_id="ent_01",
            text_snippet="APT41"
        ),
        Evidence(
            evidence_id="ev_02",
            target_id="ent_02",
            text_snippet="Cobalt Strike" # Fails grounding check!
        )
    ]
    
    result = ExtractionResult(
        metadata=metadata,
        entities=entities,
        evidence=evidence
    )
    
    extractor = Extractor()
    with pytest.raises(CTIValidationError) as excinfo:
        extractor.validate_extraction(result, chunk_text)
    assert "Grounding check failed" in str(excinfo.value)

def test_referential_integrity_relationship_failure():
    chunk_text = "APT41 breached Global Tech Corp."
    
    metadata = Metadata(
        source_file="report.xml",
        extraction_timestamp="2026-06-20T00:00:00Z"
    )
    entities = [
        Entity(entity_id="ent_01", type=EntityType.THREAT_ACTOR, value="APT41")
    ]
    # Relationship targets missing entity 'ent_99'
    relationships = [
        Relationship(
            relationship_id="rel_01",
            source_id="ent_01",
            target_id="ent_99",
            type=RelationshipType.TARGETS,
            confidence=0.90
        )
    ]
    
    result = ExtractionResult(
        metadata=metadata,
        entities=entities,
        relationships=relationships
    )
    
    extractor = Extractor()
    with pytest.raises(CTIValidationError) as excinfo:
        extractor.validate_extraction(result, chunk_text)
    assert "Referential integrity failed" in str(excinfo.value)

def test_referential_integrity_temporal_edge_failure():
    chunk_text = "APT41 breached Global Tech Corp."
    
    metadata = Metadata(
        source_file="report.xml",
        extraction_timestamp="2026-06-20T00:00:00Z"
    )
    events = [
        Event(
            event_id="evt_01",
            action="Breach network",
            sequence_number=1,
            description="Breached Global Tech Corp"
        )
    ]
    # Edge references missing event 'evt_99'
    temporal_edges = [
        TemporalCausalEdge(
            edge_id="tc_01",
            source_event_id="evt_01",
            target_event_id="evt_99",
            type=TemporalCausalType.PRECEDES,
            confidence=0.90
        )
    ]
    
    result = ExtractionResult(
        metadata=metadata,
        events=events,
        temporal_causal_edges=temporal_edges
    )
    
    extractor = Extractor()
    with pytest.raises(CTIValidationError) as excinfo:
        extractor.validate_extraction(result, chunk_text)
    assert "Referential integrity failed" in str(excinfo.value)
