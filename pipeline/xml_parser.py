import os
import xml.etree.ElementTree as ET
from schemas.cti_schema import CTIEvent, CTIAttribute

def parse_xml_file(filepath: str) -> list[CTIEvent]:
    events = []
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    filename = os.path.basename(filepath)
    
    event_type = "Unknown"
    report_year = 0
    parts = filename.split('_')
    if len(parts) >= 3:
        try:
            report_year = int(parts[1])
        except ValueError:
            pass
        event_type = parts[2].replace(".xml", "")
    
    for event_elem in root.findall('Event'):
        event_id = int(event_elem.findtext('id', '0'))
        date = event_elem.findtext('date', '')
        info = event_elem.findtext('info', '')
        
        attributes = []
        for attr_elem in event_elem.findall('Attribute'):
            attr_id_text = attr_elem.findtext('id')
            attr_id = int(attr_id_text) if attr_id_text and attr_id_text.strip() else hash(ET.tostring(attr_elem))
            category = attr_elem.findtext('category', '')
            attr_type = attr_elem.findtext('type', '')
            value = attr_elem.findtext('value', '')
            comment = attr_elem.findtext('comment', '')
            
            attr = CTIAttribute(
                id=attr_id,
                category=category,
                type=attr_type,
                value=value,
                comment=comment if comment else None
            )
            attributes.append(attr)
            
        event = CTIEvent(
            event_id=event_id,
            source_file=filename,
            event_type=event_type,
            report_year=report_year,
            date=date,
            info=info,
            attributes=attributes
        )
        events.append(event)
        
    return events
