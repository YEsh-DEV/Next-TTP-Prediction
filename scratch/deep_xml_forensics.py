import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import re

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

report_dir = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d"

def node_to_string(node):
    # Minimal stringification of an XML node
    # Remove children for brevity if it's too long, or just print outer XML
    try:
        s = ET.tostring(node, encoding='unicode', method='xml').strip()
        if len(s) > 500:
            return s[:500] + " ...</" + node.tag.split('}')[-1] + ">"
        return s
    except:
        return str(node)

def run_deep_forensics():
    xml_path = os.path.join(base_dir, "CTI_Report_Dataset", "CTIDataset_2018_ReportEvent.xml")
    if not os.path.exists(xml_path):
        print("XML not found.")
        return
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    audit_lines = ["# DEEP XML FORENSIC REPORT\n"]
    
    # PHASE 1 — FULL XML INVENTORY
    audit_lines.append("## PHASE 1 — FULL XML INVENTORY\n")
    
    tag_counts = Counter()
    
    # Also collect all elements by their clean tag
    all_elements = defaultdict(list)
    
    for elem in root.iter():
        t = elem.tag
        if '}' in t:
            t = t.split('}')[1]
        tag_counts[t] += 1
        all_elements[t].append(elem)
        
    audit_lines.append("| Tag Name | Count |")
    audit_lines.append("| --- | --- |")
    for k, v in tag_counts.most_common():
        audit_lines.append(f"| `{k}` | {v} |")
    audit_lines.append("\n")
    
    # PHASE 2 — RAW TAG EXTRACTION
    audit_lines.append("## PHASE 2 — RAW TAG EXTRACTION\n")
    
    targets = ['Tag', 'Galaxy', 'GalaxyCluster', 'Object', 'Attribute']
    for tgt in targets:
        audit_lines.append(f"### `{tgt}` Examples\n")
        nodes = all_elements.get(tgt, [])
        if not nodes:
            audit_lines.append(f"*No `{tgt}` elements found in the XML corpus.*\n")
        else:
            for i, n in enumerate(nodes[:5]): # 5 per category (25 total)
                audit_lines.append("```xml")
                audit_lines.append(node_to_string(n))
                audit_lines.append("```\n")
                
    # PHASE 3 & 4 — THREAT ACTOR DISCOVERY & DUPLICATE ANALYSIS
    audit_lines.append("## PHASE 3 & 4 — THREAT ACTOR DISCOVERY & DUPLICATE ANALYSIS\n")
    
    keywords = ['apt', 'threat actor', 'actor', 'intrusion set', 'campaign', 'malware', 'tool', 'group']
    
    entity_to_events = defaultdict(set)
    entity_counts = Counter()
    
    events = all_elements.get('Event', [])
    for ev in events:
        eid_node = ev.find('id')
        if eid_node is None:
            # Maybe namespace
            for child in ev:
                if 'id' in child.tag:
                    eid_node = child
                    break
        eid = eid_node.text if eid_node is not None else "Unknown"
        
        # Traverse all text inside this Event
        for child in ev.iter():
            text = child.text
            if not text: continue
            
            # Check for specific fields
            t = child.tag.lower()
            if '}' in t: t = t.split('}')[1]
            
            # Extract from known MISP galaxy or tag names
            if t == 'name' or t == 'value' or t == 'info':
                val = text.strip()
                val_lower = val.lower()
                
                # Heuristic: If the value mentions an actor or malware
                is_match = False
                if val.startswith('misp-galaxy:'):
                    # MISP Galaxy tag
                    parts = val.split('=')
                    if len(parts) > 1:
                        entity = parts[-1].replace('"', '')
                        # only keep if it's an actor, tool, or campaign
                        if 'threat-actor' in val or 'tool' in val or 'campaign' in val or 'ransomware' in val:
                            entity_to_events[entity].add(eid)
                            entity_counts[entity] += 1
                else:
                    for kw in keywords:
                        if kw in val_lower:
                            # We don't want to just add massive info paragraphs
                            if len(val) < 100: 
                                entity_to_events[val].add(eid)
                                entity_counts[val] += 1
                            else:
                                # Regex extraction for APT## or known patterns from text
                                apts = re.findall(r'APT\s?\d+|Lazarus|Turla|Sofacy|Cozy Bear|Fancy Bear|FIN\d', val, re.IGNORECASE)
                                for a in apts:
                                    a_clean = a.upper().replace(' ', '')
                                    entity_to_events[a_clean].add(eid)
                                    entity_counts[a_clean] += 1
                            break

    audit_lines.append("### Top 100 Discovered Entities\n")
    audit_lines.append("| Entity | Total Mentions | Unique Events |")
    audit_lines.append("| --- | --- | --- |")
    
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:100]
    for ent, count in sorted_entities:
        uniq_evts = len(entity_to_events[ent])
        audit_lines.append(f"| `{ent}` | {count} | {uniq_evts} |")
    audit_lines.append("\n")
    
    # PHASE 5 — RELATIONSHIP RECOVERY
    audit_lines.append("## PHASE 5 — RELATIONSHIP RECOVERY\n")
    
    multi_event_entities = {k: v for k, v in entity_to_events.items() if len(v) > 1}
    
    if not multi_event_entities:
        audit_lines.append("*(No entities found appearing in more than 1 unique event)*\n")
    else:
        for ent, eids in sorted(multi_event_entities.items(), key=lambda x: len(x[1]), reverse=True):
            audit_lines.append(f"**Entity:** `{ent}`")
            audit_lines.append(f"**Associated Event IDs ({len(eids)}):** {', '.join(sorted(list(eids)))}\n")
            
    # PHASE 6 — FINAL VERDICT
    audit_lines.append("## PHASE 6 — FINAL VERDICT\n")
    
    q1 = "YES" if all_elements.get('Tag') or all_elements.get('Attribute') else "NO"
    multi_count = len(multi_event_entities)
    
    audit_lines.append(f"**Q1: Did the previous audit miss metadata?**\nYes/No: Based on Phase 1, if Tags and Attributes exist, the parser dropped them. The previous audit successfully proved parser drops, but we are re-verifying if the manual extraction loop missed deeply nested text. Wait for Phase 3/4 evidence.\n")
    
    audit_lines.append(f"**Q2: Do repeated actors actually exist?**\nWe found {multi_count} entities that span multiple unique Event IDs.\n")
    
    audit_lines.append(f"**Q3: Can valid temporal sequences be recovered?**\nIf `{multi_count}` > 0, then we can recover at least some sequences. If `{multi_count}` == 0, then it is mathematically impossible to sequence this dataset.\n")
    
    audit_lines.append(f"**Q4: Should Experiment-3 continue using this dataset?**\nSee the numbers above. If multi-event actors are near zero, we must STOP and require a new dataset.\n")
    
    with open(os.path.join(report_dir, "DEEP_XML_FORENSIC_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(audit_lines))
        
    print("Deep Forensic XML Audit complete.")

if __name__ == "__main__":
    run_deep_forensics()
