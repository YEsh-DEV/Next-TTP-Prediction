from schemas.cti_schema import CTIEvent
from typing import List, Dict, Any

class CTIChunker:
    def __init__(self, batch_size=20):
        self.batch_size = batch_size
        
    def chunk_event(self, event: CTIEvent) -> List[Dict[str, Any]]:
        chunks = []
        
        # Representation B: Narrative Event
        narrative_chunk = {
            "event_id": event.event_id,
            "source_file": event.source_file,
            "event_type": event.event_type,
            "report_year": event.report_year,
            "date": event.date,
            "info": event.info,
            "chunk_type": "narrative",
            "chunk_index": 0,
            "text": event.narrative
        }
        chunks.append(narrative_chunk)
        
        # Representation A: Raw Event
        if not event.attributes:
            return chunks

        for i in range(0, len(event.attributes), self.batch_size):
            batch = event.attributes[i:i + self.batch_size]
            
            chunk_text = f"Raw IOCs for {event.event_type} {event.event_id} ({event.info}):\n"
            for attr in batch:
                comment_str = f" (Comment: {attr.comment})" if attr.comment else ""
                chunk_text += f"- [{attr.category}] {attr.type}: {attr.value}{comment_str}\n"
            
            ioc_chunk = {
                "event_id": event.event_id,
                "source_file": event.source_file,
                "event_type": event.event_type,
                "report_year": event.report_year,
                "date": event.date,
                "info": event.info,
                "chunk_type": "ioc_raw",
                "chunk_index": (i // self.batch_size) + 1,
                "text": chunk_text
            }
            chunks.append(ioc_chunk)
            
        return chunks
