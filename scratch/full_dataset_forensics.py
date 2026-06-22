"""
Full dataset forensic investigation across ALL XML files.
No model training. No evaluation. Pure data archaeology.
"""
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import re

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
report_dir = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d"
dataset_dir = os.path.join(base_dir, "CTI_Report_Dataset")

def parse_xml_safe(path):
    try:
        tree = ET.parse(path)
        return tree.getroot()
    except ET.ParseError as e:
        print(f"  [PARSE ERROR] {path}: {e}")
        return None

def extract_info_text(root):
    """Extract all event info fields for pattern analysis."""
    results = []
    for ev in root.findall('.//Event'):
        eid = ev.findtext('id', default='?')
        date = ev.findtext('date', default='?')
        info = ev.findtext('info', default='')
        results.append((eid, date, info))
    return results

def extract_attributes(root):
    """Extract all attributes with their type and value."""
    results = []
    for ev in root.findall('.//Event'):
        eid = ev.findtext('id', default='?')
        for attr in ev.findall('.//Attribute'):
            atype = attr.findtext('type', default='')
            aval = attr.findtext('value', default='')
            acat = attr.findtext('category', default='')
            results.append((eid, atype, acat, aval))
    return results

def main():
    inventory_lines = ["# DATASET INVENTORY REPORT\n"]
    forensic_lines = ["# DEEP DATASET FORENSIC REPORT\n"]
    
    xml_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.xml')])
    
    # =========== PHASE 1: INVENTORY ===========
    inventory_lines.append("## All XML Dataset Files\n")
    inventory_lines.append("| File | Size (KB) | Events | Attributes |")
    inventory_lines.append("| --- | --- | --- | --- |")
    
    total_events = 0
    all_event_infos = []  # (year, file_type, eid, date, info)
    all_attributes = []   # (year, eid, type, cat, val)
    attribute_types = Counter()
    
    for fname in xml_files:
        fpath = os.path.join(dataset_dir, fname)
        size_kb = os.path.getsize(fpath) // 1024
        
        # parse year and type from filename
        parts = fname.replace('.xml','').split('_')
        year = parts[1] if len(parts) > 1 else '?'
        ftype = parts[2] if len(parts) > 2 else '?'
        
        root = parse_xml_safe(fpath)
        if root is None:
            inventory_lines.append(f"| `{fname}` | {size_kb} | PARSE_ERROR | - |")
            continue
        
        events = root.findall('.//Event')
        num_events = len(events)
        attrs = root.findall('.//Attribute')
        num_attrs = len(attrs)
        
        total_events += num_events
        
        inventory_lines.append(f"| `{fname}` | {size_kb} | {num_events} | {num_attrs} |")
        
        # Collect info texts
        for (eid, date, info) in extract_info_text(root):
            all_event_infos.append((year, ftype, eid, date, info))
            
        # Collect attribute types
        for (eid, atype, acat, aval) in extract_attributes(root):
            all_attributes.append((year, eid, atype, acat, aval))
            attribute_types[atype] += 1
            
    inventory_lines.append(f"\n**Total Events across ALL files:** {total_events}\n")
    
    # =========== PHASE 2: ATTRIBUTE TYPE ANALYSIS ===========
    forensic_lines.append("## PHASE 2 — ATTRIBUTE TYPE ANALYSIS\n")
    forensic_lines.append("| Attribute Type | Occurrence Count |")
    forensic_lines.append("| --- | --- |")
    for k, v in attribute_types.most_common():
        forensic_lines.append(f"| `{k}` | {v} |")
    forensic_lines.append("\n")
    
    # =========== PHASE 3: INFO TEXT PATTERN MINING ===========
    forensic_lines.append("## PHASE 3 — EVENT INFO TEXT ANALYSIS\n")
    
    # Extract APT names, actor names from info fields using regex
    apt_pattern = re.compile(
        r'(APT\s?\d+|Lazarus|Turla|Sofacy|Fancy Bear|Cozy Bear|HAFNIUM|Equation Group|'
        r'FIN\d+|Carbanak|Buhtrap|MuddyWater|OceanLotus|Kimsuky|Sandworm|'
        r'REvil|LockBit|Ryuk|DarkSide|Conti|BlackMatter|Cl0p|TA\d+|SILENCE|'
        r'Gorgon|Gallmaker|Slingshot|Windshift|Dragonfish|Patchwork|Tick|Rancor|'
        r'Andariel|Bluenoroff|ChessMaster|RedEyes|MIRAGEFOX|VERMIN|BISMUTH|'
        r'MagicHound|SilkBean|Operation\s+\w+)',
        re.IGNORECASE
    )
    
    actor_events = defaultdict(list)  # actor -> [(year, eid, date, info)]
    
    for (year, ftype, eid, date, info) in all_event_infos:
        if ftype == 'ReportEvent':
            matches = apt_pattern.findall(info)
            for m in matches:
                clean = m.strip().title()
                actor_events[clean].append((year, eid, date, info))
                
    forensic_lines.append("### Actors/Campaigns found in Event Info fields\n")
    forensic_lines.append("| Actor/Campaign | Events Found | Years |")
    forensic_lines.append("| --- | --- | --- |")
    
    sorted_actors = sorted(actor_events.items(), key=lambda x: len(x[1]), reverse=True)
    
    for actor, evts in sorted_actors[:60]:
        unique_years = sorted(set(e[0] for e in evts))
        unique_eids = set(e[1] for e in evts)
        forensic_lines.append(f"| `{actor}` | {len(unique_eids)} | {', '.join(unique_years)} |")
    forensic_lines.append("\n")
    
    # =========== PHASE 4: MULTI-EVENT ACTOR SEQUENCES ===========
    forensic_lines.append("## PHASE 4 — MULTI-EVENT ACTOR SEQUENCES\n")
    
    multi_event_actors = {k: v for k, v in actor_events.items() if len(set(e[1] for e in v)) > 1}
    
    forensic_lines.append(f"**Actors with > 1 unique Event:** {len(multi_event_actors)}\n")
    
    for actor, evts in sorted(multi_event_actors.items(), key=lambda x: len(set(e[1] for e in x[1])), reverse=True):
        unique = sorted(set((e[1], e[2]) for e in evts), key=lambda x: x[1])  # sort by date
        forensic_lines.append(f"### `{actor}` — {len(unique)} Events")
        for (eid, date) in unique:
            info_text = next((e[3][:80] for e in evts if e[1] == eid), "")
            forensic_lines.append(f"- Event `{eid}` | Date: `{date}` | `{info_text}...`")
        forensic_lines.append("")
        
    # =========== PHASE 5: ATTRIBUTE-BASED RELATIONSHIPS ===========
    forensic_lines.append("## PHASE 5 — ATTRIBUTE-BASED RELATIONSHIPS\n")
    
    # Find shared infrastructure: same URL/IP/hash appearing in multiple events
    infra_to_events = defaultdict(set)
    for (year, eid, atype, acat, aval) in all_attributes:
        if atype in ['ip-dst', 'url', 'domain', 'sha256', 'md5', 'sha1', 'filename'] and aval:
            infra_to_events[aval].add(eid)
            
    shared_infra = {k: v for k, v in infra_to_events.items() if len(v) > 1}
    forensic_lines.append(f"**Shared Infrastructure Indicators (across multiple events):** {len(shared_infra)}\n")
    for val, eids in sorted(shared_infra.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
        forensic_lines.append(f"- `{val[:60]}` → Events: {sorted(list(eids))}")
    forensic_lines.append("\n")
    
    # =========== PHASE 6: MALWARE EVENT LINKAGE ===========
    forensic_lines.append("## PHASE 6 — MALWARE EVENT TO REPORT EVENT CROSS-LINKAGE\n")
    
    # Count actors in MalwareEvent too
    malware_actor_events = defaultdict(list)
    for (year, ftype, eid, date, info) in all_event_infos:
        if ftype == 'MalwareEvent':
            matches = apt_pattern.findall(info)
            for m in matches:
                clean = m.strip().title()
                malware_actor_events[clean].append((year, eid, date))
                
    forensic_lines.append(f"**Actors found in MalwareEvent info fields:** {len(malware_actor_events)}\n")
    for actor, evts in sorted(malware_actor_events.items(), key=lambda x: len(x[1]), reverse=True)[:30]:
        forensic_lines.append(f"- `{actor}`: {len(evts)} malware events")
    forensic_lines.append("\n")
    
    # Cross-link: report events and malware events sharing the same actor
    forensic_lines.append("### Actors Spanning Both Report Events AND Malware Events\n")
    cross_actors = set(actor_events.keys()) & set(malware_actor_events.keys())
    forensic_lines.append(f"**Count:** {len(cross_actors)}\n")
    for actor in sorted(cross_actors)[:20]:
        report_count = len(set(e[1] for e in actor_events[actor]))
        malware_count = len(malware_actor_events[actor])
        forensic_lines.append(f"- `{actor}`: {report_count} report events + {malware_count} malware events")
    forensic_lines.append("\n")
    
    # =========== WRITE REPORTS ===========
    with open(os.path.join(report_dir, "DATASET_INVENTORY_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(inventory_lines))
        
    with open(os.path.join(report_dir, "DEEP_FORENSIC_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(forensic_lines))
        
    print(f"Done. Total events scanned: {total_events}")
    print(f"Multi-event actors found: {len(multi_event_actors)}")
    print(f"Shared infrastructure indicators: {len(shared_infra)}")
    print(f"Cross-linked actors (Report + Malware): {len(cross_actors)}")
    
if __name__ == "__main__":
    main()
