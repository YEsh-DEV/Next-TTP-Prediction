import os
import re
import pandas as pd

def run_audit():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    enterprise_path = os.path.join(base_dir, "MitreEnterprise.xlsx")
    
    df = pd.read_excel(enterprise_path)
    
    # Extract IDs from the 'Tactic ID' column
    ids = df['Tactic ID'].dropna().astype(str).str.strip().tolist()
    
    total_ids = len(ids)
    
    # Regex for Txxxx or Txxxx.xxx
    pattern = re.compile(r"^T[0-9]{4}(\.[0-9]{3})?$")
    
    valid_ids = []
    invalid_ids = []
    
    for i in ids:
        if pattern.match(i):
            valid_ids.append(i)
        else:
            invalid_ids.append(i)
            
    sample_20 = ids[:20]
    
    output_path = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\ontology_audit_report.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# STEP 1 — Ontology Audit (`MitreEnterprise.xlsx`)\n\n")
        f.write("## 1. Global ID Metrics\n")
        f.write(f"- **Total Extracted IDs:** {total_ids}\n")
        f.write(f"- **Valid Technique IDs (Matches Regex):** {len(valid_ids)}\n")
        f.write(f"- **Invalid IDs (Fails Regex):** {len(invalid_ids)}\n\n")
        
        if invalid_ids:
            f.write("### Identified Pollutants (Invalid IDs)\n")
            f.write("```text\n")
            for i in invalid_ids:
                f.write(f"- {i}\n")
            f.write("```\n\n")
            
        f.write("## 2. Validation Pattern\n")
        f.write("`^T[0-9]{4}(\.[0-9]{3})?$`\n\n")
        
        f.write("## 3. Sample 20 IDs\n")
        f.write("```text\n")
        for i in sample_20:
            f.write(f"{i}\n")
        f.write("```\n")

    print(f"Proof saved to {output_path}")

if __name__ == "__main__":
    run_audit()
