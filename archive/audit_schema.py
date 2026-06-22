import os
import pandas as pd

def run_audit():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    attack_path = os.path.join(base_dir, "attackmitre.xlsx")
    
    df = pd.read_excel(attack_path)
    
    apt_names = set()
    software_ids = set()
    technique_ids = set()
    
    paths_apt_to_soft = 0
    paths_apt_to_tech = 0
    paths_soft_to_tech = 0
    
    for _, row in df.iterrows():
        apt = str(row.get('APT Group Name', '')).strip()
        soft = str(row.get('Software ID', '')).strip()
        g_techs = str(row.get('Group Techniques', ''))
        s_techs = str(row.get('Software Techniques', ''))
        
        if apt and apt != 'nan':
            apt_names.add(apt)
        if soft and soft != 'nan':
            software_ids.add(soft)
            
        if apt and apt != 'nan' and soft and soft != 'nan':
            paths_apt_to_soft += 1
            
        if g_techs and g_techs != 'nan':
            for t in g_techs.split(';'):
                t_clean = t.strip()
                if t_clean:
                    technique_ids.add(t_clean)
                    paths_apt_to_tech += 1
                    
        if s_techs and s_techs != 'nan':
            for t in s_techs.split(';'):
                t_clean = t.strip()
                if t_clean:
                    technique_ids.add(t_clean)
                    paths_soft_to_tech += 1

    output_path = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\attackmitre_schema_report.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# STEP 2 — Schema Audit (`attackmitre.xlsx`)\n\n")
        f.write("## 1. Entity Counts\n")
        f.write(f"- **Unique APT Groups:** {len(apt_names)}\n")
        f.write(f"- **Unique Software IDs:** {len(software_ids)}\n")
        f.write(f"- **Unique Technique IDs Referenced:** {len(technique_ids)}\n\n")
        
        f.write("## 2. Structural Edge Counts (Graph Pathways)\n")
        f.write(f"- **`APTGroup` -> `Software` Edges:** {paths_apt_to_soft}\n")
        f.write(f"- **`APTGroup` -> `Technique` Edges:** {paths_apt_to_tech}\n")
        f.write(f"- **`Software` -> `Technique` Edges:** {paths_soft_to_tech}\n\n")
        
        f.write("## 3. Ontology Validation\n")
        import re
        pattern = re.compile(r"^T[0-9]{4}(\.[0-9]{3})?$")
        invalid_techs = [t for t in technique_ids if not pattern.match(t)]
        if invalid_techs:
            f.write(f"**WARNING:** Found {len(invalid_techs)} non-Txxxx IDs embedded inside Technique lists!\n")
            f.write("```text\n")
            for t in invalid_techs[:10]: f.write(f"- {t}\n")
            f.write("```\n")
        else:
            f.write("**PASSED:** All Technique IDs referenced in `attackmitre.xlsx` correctly match `^T[0-9]{4}(\\.[0-9]{3})?$`.\n")

    print(f"Proof saved to {output_path}")

if __name__ == "__main__":
    run_audit()
