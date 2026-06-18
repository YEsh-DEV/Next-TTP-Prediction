import pytest
from schemas.cti_schema import CTIEvent
from pipeline.chunker import CTIChunker

def test_pydantic_schema():
    event_data = {
        "id": 1,
        "date": "2019-01-01",
        "info": "test_report.pdf",
        "attributes": [
            {"id": 100, "category": "Network activity", "type": "ip-src", "value": "1.1.1.1", "comment": None}
        ]
    }
    event = CTIEvent(**event_data)
    assert event.id == 1
    assert event.attributes[0].value == "1.1.1.1"

def test_chunker_logic():
    event = CTIEvent(
        id=2, 
        date="2020-01-01", 
        info="malware.pdf", 
        attributes=[
            {"id": i, "category": "cat", "type": "typ", "value": f"val{i}", "comment": None} for i in range(45)
        ]
    )
    
    chunker = CTIChunker(batch_size=20)
    chunks = chunker.chunk_event(event)
    
    # 45 attributes with batch size 20 should yield 3 chunks (20, 20, 5)
    assert len(chunks) == 3
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1
    assert chunks[2]["chunk_index"] == 2
    
    # Ensure metadata is injected into the text
    assert "Report or Malware Info: malware.pdf" in chunks[0]["text"]
