import pytest
from schemas.cti_schema import CTIEvent
from pipeline.chunker import CTIChunker

def test_pydantic_schema():
    event_data = {
        "event_id": 1,
        "source_file": "CTIDataset_2019_ReportEvent.xml",
        "event_type": "ReportEvent",
        "report_year": 2019,
        "date": "2019-01-01",
        "info": "test_report.pdf",
        "attributes": [
            {"id": 100, "category": "Network activity", "type": "ip-src", "value": "1.1.1.1", "comment": None}
        ]
    }
    event = CTIEvent(**event_data)
    assert event.event_id == 1
    assert event.attributes[0].value == "1.1.1.1"

def test_chunker_logic():
    event = CTIEvent(
        event_id=2, 
        source_file="CTIDataset_2020_MalwareEvent.xml",
        event_type="MalwareEvent",
        report_year=2020,
        date="2020-01-01", 
        info="malware.pdf", 
        attributes=[
            {"id": i, "category": "cat", "type": "typ", "value": f"val{i}", "comment": None} for i in range(45)
        ]
    )
    
    chunker = CTIChunker(batch_size=20)
    chunks = chunker.chunk_event(event)
    
    assert len(chunks) == 4
    assert chunks[0]["chunk_type"] == "narrative"
    assert chunks[0]["chunk_index"] == 0
    
    assert chunks[1]["chunk_type"] == "ioc_raw"
    assert chunks[1]["chunk_index"] == 1
    assert chunks[2]["chunk_index"] == 2
    assert chunks[3]["chunk_index"] == 3
    
    assert "CTI Malware from 2020 titled 'malware.pdf'" in chunks[0]["text"]
