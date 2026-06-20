import os
import pandas as pd
from datetime import datetime
from typing import List

from schemas.cti_schema import CTIEvent
from schemas.extraction_schema import (
    ExtractionResult, Metadata, Entity, EntityType, 
    Event, Relationship, RelationshipType, Evidence
)
from pipeline.hybrid_classifier import HybridClassifier

class MitreMapper:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.attack_mitre_path = os.path.join(base_dir, "attackmitre.xlsx")
        self.mitre_enterprise_path = os.path.join(base_dir, "MitreEnterprise.xlsx")
        self.classifier = HybridClassifier(base_dir)
        
        # Load datasets
        try:
            self.df_attack = pd.read_excel(self.attack_mitre_path)
            self.df_enterprise = pd.read_excel(self.mitre_enterprise_path)
        except Exception as e:
            print(f"Warning: Could not load MITRE Excel files. Ensure they exist. {e}")
            self.df_attack = pd.DataFrame(columns=['APT Group Name', 'Group Techniques', 'Software ID'])
            self.df_enterprise = pd.DataFrame(columns=['Tactic ID', 'Tactic Name', 'Description'])
            
        # Standardize columns
        if 'Tactic ID' in self.df_enterprise.columns:
            self.df_enterprise['Tactic ID'] = self.df_enterprise['Tactic ID'].astype(str)

    def map_event(self, cti_event: CTIEvent) -> ExtractionResult:
        """
        Deterministically maps a CTIEvent's attributes and info to an ExtractionResult graph.
        Uses HybridClassifier to determine techniques.
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        metadata = Metadata(
            source_file=cti_event.source_file,
            report_date=cti_event.date,
            extraction_timestamp=timestamp
        )
        
        entities: List[Entity] = []
        relationships: List[Relationship] = []
        evidence: List[Evidence] = []
        
        ent_counter = 1
        rel_counter = 1
        ev_counter = 1
        
        # 1. APT Group Identification
        apt_group_name = None
        info_lower = str(cti_event.info).lower()
        for _, row in self.df_attack.iterrows():
            group = str(row.get('APT Group Name', '')).strip()
            if group and group.lower() in info_lower:
                apt_group_name = group
                break
                
        actor_id = None
        if apt_group_name:
            actor_id = f"ent_{ent_counter:02d}"
            ent_counter += 1
            entities.append(Entity(
                entity_id=actor_id,
                type=EntityType.THREAT_ACTOR,
                value=apt_group_name,
                description=f"Identified via deterministic regex mapping from report info."
            ))
            evidence.append(Evidence(
                evidence_id=f"ev_{ev_counter:02d}",
                target_id=actor_id,
                text_snippet=cti_event.info
            ))
            ev_counter += 1

        # 2. Process Attributes (Indicators)
        event_indicators = []
        for attr in cti_event.attributes:
            entity_type = None
            
            # GRAPH COMPRESSION: Absorb infrastructure directly into the event property array
            if attr.type in ['url', 'ip-src', 'domain', 'email-src']:
                event_indicators.append(f"{attr.type}:{attr.value}")
                continue
                
            # Promote high-value IOCs to permanent Graph Nodes
            elif attr.type in ['filename', 'md5', 'sha1', 'sha256']:
                entity_type = EntityType.FILE_ARTIFACT
            elif attr.type == 'vulnerability':
                entity_type = EntityType.VULNERABILITY
                
            if entity_type:
                ent_id = f"ent_{ent_counter:02d}"
                ent_counter += 1
                
                # Create Entity
                entities.append(Entity(
                    entity_id=ent_id,
                    type=entity_type,
                    value=attr.value,
                    description=f"Category: {attr.category}, Type: {attr.type}"
                ))
                
                # Grounding
                evidence.append(Evidence(
                    evidence_id=f"ev_{ev_counter:02d}",
                    target_id=ent_id,
                    text_snippet=attr.value
                ))
                ev_counter += 1
                
                # Relate to Actor if exists
                if actor_id:
                    rel_id = f"rel_{rel_counter:02d}"
                    rel_counter += 1
                    
                    rel_type = RelationshipType.USES
                    if entity_type == EntityType.VULNERABILITY:
                        rel_type = RelationshipType.EXPLOITS
                        
                    relationships.append(Relationship(
                        relationship_id=rel_id,
                        source_id=actor_id,
                        target_id=ent_id,
                        type=rel_type,
                        confidence=1.0 # Deterministic
                    ))
                    
        # 3. Dynamic MITRE Technique Classification (ChromaDB + Gemini)
        technique_matches = self.classifier.classify_event(cti_event)
                
        # 4. Create the core Event Node
        events: List[Event] = []
        core_event_id = "evt_01"
        
        if not technique_matches and not entities and not event_indicators:
            # Completely empty event
            pass
        else:
            events.append(Event(
                event_id=core_event_id,
                action="Indicator Observation",
                timestamp=cti_event.date,
                sequence_number=1,
                actor_id=actor_id,
                target_ids=[e.entity_id for e in entities if e.entity_id != actor_id],
                mitre_technique_id=None,
                description="MISP indicators observed and compressed.",
                indicators=list(set(event_indicators)) # Compressed graph array
            ))
            
            # 5. Create Probabilistic OBSERVED_TECHNIQUE Edges
            for match in technique_matches:
                tech_id = match.technique_id
                confidence = match.confidence
                
                # Ensure the Technique node exists in the output so the edge is valid
                tech_ent_id = f"tech_{tech_id}"
                
                # Check if we already created this technique node
                if not any(e.entity_id == tech_ent_id for e in entities):
                    # Lookup official name
                    name = tech_id
                    enterprise_match = self.df_enterprise[self.df_enterprise['Tactic ID'] == tech_id]
                    if not enterprise_match.empty:
                        name = enterprise_match.iloc[0].get('Tactic Name', tech_id)
                        
                    entities.append(Entity(
                        entity_id=tech_ent_id,
                        type=EntityType.TECHNIQUE,
                        value=name,
                        description=f"MITRE Technique {tech_id}"
                    ))
                
                rel_id = f"rel_{rel_counter:02d}"
                rel_counter += 1
                
                relationships.append(Relationship(
                    relationship_id=rel_id,
                    source_id=core_event_id,
                    target_id=tech_ent_id,
                    type=RelationshipType.OBSERVED_TECHNIQUE,
                    confidence=confidence # Probabilistic fuzzy weighting for R-GCN
                ))

        return ExtractionResult(
            metadata=metadata,
            entities=entities,
            events=events,
            relationships=relationships,
            temporal_causal_edges=[],
            evidence=evidence
        )
