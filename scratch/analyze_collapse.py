import os
import sys
import random
from collections import Counter

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.xml_parser import parse_xml_file
from pipeline.deterministic_classifier import DeterministicClassifier

def run_collapse_analysis():
    xml_path = os.path.join(base_dir, "CTI_Report_Dataset", "CTIDataset_2018_ReportEvent.xml")
    events = parse_xml_file(xml_path)
    classifier = DeterministicClassifier(base_dir)
    
    top1_list = []
    sample_data = []
    
    print(f"Executing Deterministic Classifier (V1) over {len(events)} events...")
    
    for e in events:
        res = classifier.classify_event(e)
        if res['techniques']:
            top1_list.append(res['techniques'][0]['id'])
            sample_data.append({
                "event_id": str(e.event_id),
                "title": e.info,
                "techniques": res['techniques']
            })
            
    # Task 1: Distribution
    tech_counts = Counter(top1_list)
    total_valid = len(top1_list)
    sorted_techs = tech_counts.most_common()
    
    rep1 = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\technique_distribution_report.md"
    with open(rep1, "w", encoding="utf-8") as f:
        f.write("# TASK 1 — TECHNIQUE DISTRIBUTION REPORT\n\n")
        f.write("| Technique ID | Count | Percentage |\n")
        f.write("| :--- | :--- | :--- |\n")
        for tid, count in sorted_techs:
            pct = (count / total_valid) * 100
            f.write(f"| `{tid}` | {count} | {pct:.2f}% |\n")
            
    # Task 2: Collapse Analysis
    unique_count = len(sorted_techs)
    rep2 = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\collapse_analysis_report.md"
    with open(rep2, "w", encoding="utf-8") as f:
        f.write("# TASK 2 — COLLAPSE ANALYSIS REPORT\n\n")
        f.write(f"- **Total Events Evaluated:** {total_valid}\n")
        f.write(f"- **Unique Top-1 Techniques:** {unique_count}\n\n")
        if unique_count < 20:
            f.write("## VERDICT: SEVERE STATE COLLAPSE\n")
        else:
            f.write("## VERDICT: HEALTHY STATE DISTRIBUTION\n")
            
    # Task 3: Coverage
    top10 = sorted_techs[:10]
    top10_sum = sum([c for t, c in top10])
    coverage_pct = (top10_sum / total_valid) * 100 if total_valid > 0 else 0
    
    rep3 = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\coverage_report.md"
    with open(rep3, "w", encoding="utf-8") as f:
        f.write("# TASK 3 — TOP-10 TECHNIQUE COVERAGE REPORT\n\n")
        f.write(f"The Top 10 techniques explain **{coverage_pct:.2f}%** of all events in the corpus.\n\n")
        f.write("### Top 10 Techniques by Frequency:\n")
        for i, (tid, count) in enumerate(top10):
            f.write(f"{i+1}. `{tid}` ({count} occurrences)\n")
            
    # Task 4: Sample Inspection
    sample_25 = random.sample(sample_data, min(25, len(sample_data)))
    rep4 = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\mapping_sample_report.md"
    with open(rep4, "w", encoding="utf-8") as f:
        f.write("# TASK 4 — EVENT MAPPING INSPECTION\n\n")
        for s in sample_25:
            f.write(f"### Event {s['event_id']}: {s['title']}\n")
            f.write("```text\n")
            for t in s['techniques']:
                f.write(f"-> {t['id']} (Score: {t['score']})\n")
            f.write("```\n")

    print("State Collapse Analytics Completed.")

if __name__ == "__main__":
    run_collapse_analysis()
