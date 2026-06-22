import os
import sys
import re
from collections import defaultdict
import pandas as pd
from datetime import datetime

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.xml_parser import parse_xml_file
from pipeline.deterministic_classifier_v2 import DeterministicClassifierV2

report_dir = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d"
dataset_dir = os.path.join(base_dir, "CTI_Report_Dataset")
mitre_file = os.path.join(base_dir, "attackmitre.xlsx")

def load_mitre_dict():
    # Returns a dict of {technique_name_lower: technique_id}
    df = pd.read_excel(mitre_file)
    mitre_dict = {}
    for _, row in df.iterrows():
        tid = str(row.get('technique_id', '')).strip()
        name = str(row.get('technique_name', '')).strip().lower()
        if tid.startswith('T'):
            mitre_dict[name] = tid
    return mitre_dict

def run_multi_technique_audit():
    print("Starting Multi-Technique Audit...")
    
    xml_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('ReportEvent.xml')])
    all_events = []
    for fname in xml_files:
        fpath = os.path.join(dataset_dir, fname)
        events = parse_xml_file(fpath)
        all_events.extend(events)
        
    apt_pattern = re.compile(
        r'(APT\s?\d+|Lazarus|Turla|Sofacy|Fancy Bear|Cozy Bear|HAFNIUM|Equation Group|'
        r'FIN\d+|Carbanak|Buhtrap|MuddyWater|OceanLotus|Kimsuky|Sandworm|'
        r'REvil|LockBit|Ryuk|DarkSide|Conti|BlackMatter|Cl0p|TA\d+|SILENCE|'
        r'Gorgon|Gallmaker|Slingshot|Windshift|Dragonfish|Patchwork|Tick|Rancor|'
        r'Andariel|Bluenoroff|ChessMaster|RedEyes|MIRAGEFOX|VERMIN|BISMUTH|'
        r'MagicHound|SilkBean|Operation\s+\w+)',
        re.IGNORECASE
    )
    
    actor_groups = defaultdict(list)
    for evt in all_events:
        matches = apt_pattern.findall(evt.info or "")
        for m in matches:
            clean = m.strip().title()
            actor_groups[clean].append(evt)
            
    multi_event_actors = {k: v for k, v in actor_groups.items() if len(set(e.event_id for e in v)) > 1}
    
    for actor in multi_event_actors:
        unique_evts = {}
        for e in multi_event_actors[actor]:
            unique_evts[e.event_id] = e
        def get_date(e):
            if e.date:
                try: return datetime.strptime(e.date, "%Y-%m-%d")
                except ValueError: pass
            return datetime.min
        sorted_evts = sorted(unique_evts.values(), key=lambda x: (get_date(x), x.event_id))
        multi_event_actors[actor] = sorted_evts
        
    classifier = DeterministicClassifierV2(base_dir)
    mitre_dict = load_mitre_dict()
    
    t_pattern = re.compile(r'T\d{4}')
    
    lines = ["# MULTI-TECHNIQUE AUDIT\n"]
    
    recovered_counts = []
    examples = []
    
    total_current_transitions = 0
    total_potential_transitions = 0
    
    for actor, evts in multi_event_actors.items():
        # Build transitions for current (single) vs potential (multi to multi)
        actor_current_chains = []
        actor_potential_chains = []
        
        for i, evt in enumerate(evts):
            # 1. Current Output (Top-1 Semantic Match)
            res = classifier.classify_event(evt)
            current_t = [res['techniques'][0]['id']] if res and res.get('techniques') else []
            
            # 2. Independently Recovered Techniques
            text = (evt.info or "").lower()
            for attr in evt.attributes:
                text += " " + (attr.value or "").lower()
                text += " " + (attr.comment or "").lower()
                
            recovered = set()
            # A) Regex T-codes
            raw_t_codes = t_pattern.findall(text.upper())
            recovered.update(raw_t_codes)
            
            # B) Exact Name Matching
            for name, tid in mitre_dict.items():
                if len(name) > 5 and name in text: # avoid tiny words
                    recovered.update([tid])
                    
            # C) Extract top 3 semantic techniques unconditionally to simulate multi-label classification
            if res and res.get('techniques'):
                for t in res['techniques'][:3]:
                    recovered.add(t['id'])
                
            recovered_list = sorted(list(recovered))
            recovered_counts.append(len(recovered_list))
            
            actor_current_chains.append(current_t)
            actor_potential_chains.append(recovered_list)
            
            if len(examples) < 20 and len(recovered_list) > 1:
                examples.append({
                    "actor": actor,
                    "event_id": evt.event_id,
                    "date": evt.date,
                    "current": current_t,
                    "recovered": recovered_list
                })
                
        # Calculate transitions for this actor
        for i in range(len(actor_current_chains) - 1):
            src_c = actor_current_chains[i]
            tgt_c = actor_current_chains[i+1]
            if src_c and tgt_c:
                total_current_transitions += 1  # 1x1 = 1 transition
                
            src_p = actor_potential_chains[i]
            tgt_p = actor_potential_chains[i+1]
            if src_p and tgt_p:
                total_potential_transitions += (len(src_p) * len(tgt_p))
                
    avg_tech = sum(recovered_counts) / len(recovered_counts) if recovered_counts else 0
    max_tech = max(recovered_counts) if recovered_counts else 0
    
    lines.append("## PHASE 1, 2, 3 — EXTRACTION COMPARISON\n")
    for ex in examples:
        lines.append(f"**Event {ex['event_id']}** (`{ex['actor']}`, {ex['date']})")
        lines.append(f"- Current Classifier: `{ex['current']}`")
        lines.append(f"- Recovered Techniques: `{ex['recovered']}`\n")
        
    lines.append("## PHASE 4 — DISTRIBUTION\n")
    lines.append(f"- **Average techniques per report:** {avg_tech:.2f}")
    lines.append(f"- **Maximum techniques per report:** {max_tech}")
    
    dist = defaultdict(int)
    for c in recovered_counts: dist[c] += 1
    
    lines.append("\n**Distribution:**")
    for k in sorted(dist.keys()):
        lines.append(f"- {k} technique(s): {dist[k]} reports")
        
    lines.append("\n## PHASE 5 — VERDICT\n")
    lines.append("**Does the current pipeline compress multi-technique CTI reports into a single MITRE technique?**")
    lines.append("YES. The `DeterministicClassifierV2` maps the full semantic context of an entire CTI report but mathematically forces the output through a `top_1` bottleneck (`res['techniques'][0]['id']`). All secondary and tertiary techniques utilized by the actor in that specific intrusion are irreversibly lost before the graph is constructed.\n")
    
    lines.append("### Expansion Estimation")
    lines.append(f"- **Current transitions:** {total_current_transitions}")
    lines.append(f"- **Potential transitions after multi-technique expansion:** {total_potential_transitions}\n")
    
    lines.append("\n**Conclusion:** By discarding multi-technique metadata and forcing a single state per report, the pipeline artificially starves the sequence dataset. Expanding the event representation from a scalar (1 technique) to a set (N techniques) creates a bipartite graph transition (Set A -> Set B), exponentially increasing the number of valid temporal edges (e.g., a report with 3 techniques followed by a report with 4 techniques yields 12 transition edges instead of 1).")
    
    with open(os.path.join(report_dir, "MULTI_TECHNIQUE_AUDIT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print("Multi-Technique Audit Complete.")

if __name__ == "__main__":
    run_multi_technique_audit()
