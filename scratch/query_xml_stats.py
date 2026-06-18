import os
import sys
import xml.etree.ElementTree as ET

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "CTI_Report_Dataset")
FILE_PATH = os.path.join(DATA_DIR, "CTIDataset_2019_ReportEvent.xml")

def run():
    print(f"Analyzing {os.path.basename(FILE_PATH)}...")
    tree = ET.parse(FILE_PATH)
    root = tree.getroot()
    
    events = root.findall('Event')
    total_events = len(events)
    
    total_attributes = 0
    for event in events:
        attrs = event.findall('Attribute')
        total_attributes += len(attrs)
        
    avg_attributes = total_attributes / total_events if total_events > 0 else 0
    
    print(f"Event nodes: {total_events}")
    print(f"Attribute nodes: {total_attributes}")
    print(f"Average attributes per event: {avg_attributes:.2f}")

if __name__ == "__main__":
    run()
