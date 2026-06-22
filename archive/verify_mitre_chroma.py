import os
import re
import chromadb

def verify_clean_db():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chroma_dir = os.path.join(base_dir, "chroma_db")
    db_client = chromadb.PersistentClient(path=chroma_dir)
    collection = db_client.get_collection("mitre_techniques")
    
    data = collection.get()
    ids = data['ids']
    
    total = len(ids)
    
    pattern_t = re.compile(r"^T[0-9]{4}(\.[0-9]{3})?$")
    pattern_s = re.compile(r"^S[0-9]{4}$")
    pattern_g = re.compile(r"^G[0-9]{4}$")
    
    count_t = 0
    count_s = 0
    count_g = 0
    count_other = 0
    
    for tid in ids:
        if pattern_t.match(tid): count_t += 1
        elif pattern_s.match(tid): count_s += 1
        elif pattern_g.match(tid): count_g += 1
        else: count_other += 1
        
    output_path = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\mitre_vector_validation.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# TASK 1 — VERIFY CLEAN MITRE VECTOR DB\n\n")
        f.write("## 1. Global Metrics\n")
        f.write(f"- **`collection.count()`:** {total}\n\n")
        
        f.write("## 2. Regex Validation Statistics\n")
        f.write(f"- **Txxxx / Txxxx.xxx (Techniques):** {count_t}\n")
        f.write(f"- **Sxxxx (Software):** {count_s}\n")
        f.write(f"- **Gxxxx (APT Groups):** {count_g}\n")
        f.write(f"- **Other:** {count_other}\n\n")
        
        f.write("## 3. First 50 IDs Dump\n")
        f.write("```text\n")
        for i in ids[:50]:
            f.write(f"{i}\n")
        f.write("```\n")
        
    print(f"Validation saved to {output_path}")

if __name__ == "__main__":
    verify_clean_db()
