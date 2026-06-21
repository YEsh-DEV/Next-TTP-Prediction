import os
import sys
import random
from collections import Counter, defaultdict
import datetime

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.xml_parser import parse_xml_file
from pipeline.deterministic_classifier_v2 import DeterministicClassifierV2
from pipeline.markov_predictor import GlobalMarkovPredictor
from pipeline.query_pipeline import GraphRAGPipeline

def get_date(e):
    if e.date:
        try:
            return datetime.datetime.strptime(e.date, "%Y-%m-%d")
        except ValueError:
            pass
    return datetime.datetime.min

def run_validation():
    xml_path = os.path.join(base_dir, "CTI_Report_Dataset", "CTIDataset_2018_ReportEvent.xml")
    events = parse_xml_file(xml_path)
    
    classifier = DeterministicClassifierV2(base_dir)
    
    print("Mapping 125 events using V2 Classifier...")
    sorted_events = sorted(events, key=lambda x: (get_date(x), x.event_id))
    
    all_mappings = []
    technique_sequence = []
    
    for e in sorted_events:
        res = classifier.classify_event(e)
        if res['techniques']:
            top1 = res['techniques'][0]['id']
            technique_sequence.append(top1)
            all_mappings.append({
                "event_id": str(e.event_id),
                "title": e.info,
                "techniques": res['techniques']
            })
            
    # Task 1: Distribution
    tech_counts = Counter(technique_sequence)
    total_valid = len(technique_sequence)
    sorted_techs = tech_counts.most_common()
    
    rep1 = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\final_technique_distribution.md"
    with open(rep1, "w", encoding="utf-8") as f:
        f.write("# TASK 1 — FINAL TECHNIQUE DISTRIBUTION\n\n")
        f.write("| Technique ID | Count | Percentage |\n")
        f.write("| :--- | :--- | :--- |\n")
        total = 0
        for tid, count in sorted_techs:
            pct = (count / total_valid) * 100
            f.write(f"| `{tid}` | {count} | {pct:.2f}% |\n")
            total += count
        f.write(f"\n**Total Count:** {total}\n")
        f.write(f"**Verification:** sum(counts) == {total_valid} -> {'PASS' if total == 125 else 'WARNING (Failed mappings)'}\n")

    # Task 2: Legitimacy
    results = classifier.collection.get()
    valid_ids = set(results['ids'])
    id_to_name = dict(zip(results['ids'], [m['name'] for m in results['metadatas']]))
    
    all_valid = True
    rep2 = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\technique_legitimacy_report.md"
    with open(rep2, "w", encoding="utf-8") as f:
        f.write("# TASK 2 — TECHNIQUE LEGITIMACY AUDIT\n\n")
        f.write("| Technique ID | Exists In MITRE (Y/N) | Technique Name |\n")
        f.write("| :--- | :--- | :--- |\n")
        for tid, _ in sorted_techs:
            exists = "Y" if tid in valid_ids else "N"
            name = id_to_name.get(tid, "UNKNOWN")
            f.write(f"| `{tid}` | {exists} | {name} |\n")
            if exists == "N":
                all_valid = False
        f.write(f"\n**Validation Status:** {'PASS' if all_valid else 'FAIL VALIDATION'}\n")

    # Task 3: Mapping Review
    sample_20 = random.sample(all_mappings, min(20, len(all_mappings)))
    rep3 = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\event_mapping_review.md"
    with open(rep3, "w", encoding="utf-8") as f:
        f.write("# TASK 3 — EVENT MAPPING REVIEW\n\n")
        for s in sample_20:
            f.write(f"### Event {s['event_id']}: {s['title']}\n")
            f.write("```text\n")
            for t in s['techniques'][:3]:
                f.write(f"-> {t['id']} (Score: {t['score']})\n")
            f.write("```\n")

    # Task 4: Markov Matrix
    transition_counts = defaultdict(int)
    for i in range(len(technique_sequence)-1):
        transition_counts[f"{technique_sequence[i]} -> {technique_sequence[i+1]}"] += 1
        
    sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)
    
    rep4 = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\markov_matrix_report.md"
    with open(rep4, "w", encoding="utf-8") as f:
        f.write("# TASK 4 — MARKOV MATRIX INSPECTION\n\n")
        f.write(f"- **Number of States:** {len(sorted_techs)}\n")
        f.write(f"- **Number of Transitions:** {sum(transition_counts.values())}\n\n")
        f.write("### Top 20 Transition Pairs\n")
        f.write("```text\n")
        for pair, count in sorted_transitions[:20]:
            f.write(f"{pair} : {count}\n")
        f.write("```\n")
        
    # Task 5: Query Diversity
    print("Running 6 target queries...")
    pipeline = GraphRAGPipeline(base_dir)
    pipeline.deterministic_classifier = classifier # Override with V2
    markov = GlobalMarkovPredictor(base_dir)
    
    queries = ["Burning Umbrella", "Operation Kitty", "Lazarus Cryptocurrency", "MuddyWater", "APT29", "Turla"]
    
    rep5 = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\query_diversity_report.md"
    with open(rep5, "w", encoding="utf-8") as f:
        f.write("# TASK 5 — QUERY DIVERSITY TEST\n\n")
        
        for q in queries:
            f.write(f"## Query: {q}\n")
            payload = pipeline.execute_query(q, top_k=3, classification_mode="deterministic")
            retrieved = payload["retrieved_events"]
            f.write(f"- **Retrieved Events:** {', '.join(retrieved)}\n")
            
            mapped = payload["mapped_techniques"]
            f.write(f"- **Mapped Techniques:** {', '.join(mapped) if mapped else 'None'}\n")
            
            if mapped:
                # Get the chronologically last event's technique
                # To be exact, we can use the top mapped technique or predict from all.
                # Let's predict from the first mapped technique returned.
                latest = mapped[0] 
                preds = markov.top_k_predictions(latest, k=3)
                pred_str = ", ".join([p['state'] for p in preds])
            else:
                pred_str = "None"
            f.write(f"- **Predicted Next Techniques:** {pred_str}\n\n")

if __name__ == "__main__":
    run_validation()
