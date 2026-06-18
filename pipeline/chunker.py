from schemas.cti_schema import CTIEvent
from typing import List, Dict, Any

class CTIChunker:
    def __init__(self, batch_size=20):
        self.batch_size = batch_size
        
    def chunk_event(self, event: CTIEvent) -> List[Dict[str, Any]]:
        chunks = []
        # Base metadata text injects context into every chunk
        base_text = f"Report or Malware Info: {event.info} | Date: {event.date} | Event ID: {event.id}\nIndicators:"
        
        if not event.attributes:
            return chunks

        # Split attributes into semantic batches
        for i in range(0, len(event.attributes), self.batch_size):
            batch = event.attributes[i:i + self.batch_size]
            chunk_text = base_text + "\n"
            for attr in batch:
                comment_str = f" (Comment: {attr.comment})" if attr.comment else ""
                chunk_text += f"- [{attr.category}] {attr.type}: {attr.value}{comment_str}\n"
            
            chunk_meta = {
                "event_id": event.id,
                "date": event.date,
                "info": event.info,
                "chunk_index": i // self.batch_size,
                "text": chunk_text
            }
            chunks.append(chunk_meta)
            
        return chunks
