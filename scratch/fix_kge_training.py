"""
Fix TransE and RotatE training using correct PyKEEN split() API.
"""
import os, sys, csv, warnings
import numpy as np
warnings.filterwarnings("ignore")
os.environ["PYTHONIOENCODING"] = "utf-8"

report_dir = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d"

def load_benchmark():
    path = os.path.join(report_dir, "true_benchmark_full.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def make_actor_aware(rows):
    new_rows = []
    for r in rows:
        actor = r["actor"]
        new_rows.append({
            **r,
            "src_node": f"{actor}::{r['src_technique']}",
            "tgt_node": f"{actor}::{r['tgt_technique']}",
        })
    return new_rows

def train_kge_fixed(all_actor_rows, model_name="TransE", dim=128, epochs=200):
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline as kge_pipeline

    all_triples = np.array(
        [[r["src_node"], "NEXT_TTP", r["tgt_node"]] for r in all_actor_rows],
        dtype=str
    )

    full_tf = TriplesFactory.from_labeled_triples(triples=all_triples)
    train_tf, test_tf = full_tf.split([0.8, 0.2], random_state=42)

    num_train = train_tf.num_triples
    print(f"  {model_name}: {num_train} train triples, {test_tf.num_triples} test triples")

    result = kge_pipeline(
        model=model_name,
        training=train_tf,
        testing=test_tf,
        training_kwargs={"num_epochs": epochs, "batch_size": min(256, num_train)},
        model_kwargs={"embedding_dim": dim},
        random_seed=42,
        device="cpu",
    )

    metrics = result.metric_results.to_flat_dict()

    def get_metric(suffixes):
        for mode in ["both.realistic.", "both.optimistic.", "tail.realistic."]:
            for s in suffixes:
                k = mode + s
                if k in metrics:
                    return metrics[k]
        return 0.0

    h1  = round(get_metric(["hits_at_1"]) * 100, 2)
    h3  = round(get_metric(["hits_at_3"]) * 100, 2)
    mrr = round(get_metric(["inverse_harmonic_mean_rank"]), 4)

    return {"hits@1": f"{h1}%", "hits@3": f"{h3}%", "mrr": mrr, "macro_f1": "N/A (ranking metric)"}

def main():
    print("Fixing TransE / RotatE...")
    rows = load_benchmark()
    actor_rows = make_actor_aware(rows)

    results = {}

    print("\n[TransE]")
    try:
        r = train_kge_fixed(actor_rows, "TransE", dim=128, epochs=200)
        results["TransE"] = r
        print(f"  H@1={r['hits@1']}, H@3={r['hits@3']}, MRR={r['mrr']}")
    except Exception as e:
        results["TransE"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    print("\n[RotatE]")
    try:
        r = train_kge_fixed(actor_rows, "RotatE", dim=128, epochs=200)
        results["RotatE"] = r
        print(f"  H@1={r['hits@1']}, H@3={r['hits@3']}, MRR={r['mrr']}")
    except Exception as e:
        results["RotatE"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    # Print all available metric keys for debugging if both fail
    if all("error" in v for v in results.values()):
        print("\nDEBUG: Dumping available metrics from a test run...")
        try:
            from pykeen.triples import TriplesFactory
            from pykeen.pipeline import pipeline as kge_pipeline
            import numpy as np
            rows_sample = actor_rows[:50]
            all_triples = np.array([[r["src_node"], "NEXT_TTP", r["tgt_node"]] for r in rows_sample], dtype=str)
            full_tf = TriplesFactory.from_labeled_triples(triples=all_triples)
            train_tf, test_tf = full_tf.split([0.8, 0.2], random_state=42)
            result = kge_pipeline(model="TransE", training=train_tf, testing=test_tf,
                                  training_kwargs={"num_epochs": 5}, random_seed=42, device="cpu")
            flat = result.metric_results.to_flat_dict()
            print("Available metric keys:")
            for k in sorted(flat.keys())[:30]:
                print(f"  {k}: {flat[k]}")
        except Exception as e2:
            print(f"  Debug run also failed: {e2}")

    # Merge into existing results table
    existing = {
        "Markov Chain": {"hits@1": "3.85%", "hits@3": "10.77%", "mrr": "0.0780", "macro_f1": "0.0092"},
        "GAT / R-GCN": {"hits@1": "3.85%", "hits@3": "10.77%", "mrr": "0.1012", "macro_f1": "0.2967"},
        "Temporal GNN": {"hits@1": "3.08%", "hits@3": "9.23%", "mrr": "0.0656", "macro_f1": "0.1448"},
        "LLM (Qwen 7B)": {"hits@1": "3.33%", "hits@3": "3.33%", "mrr": "N/A", "macro_f1": "N/A"},
    }
    existing.update(results)

    lines = ["# UPDATED RESULTS TABLE\n"]
    lines.append("*All values from the corrected 648-transition actor-isolated benchmark.*\n")
    lines.append("| Model | Hits@1 | Hits@3 | MRR | F1 |")
    lines.append("| --- | --- | --- | --- | --- |")
    order = ["Markov Chain", "TransE", "RotatE", "GAT / R-GCN", "Temporal GNN", "LLM (Qwen 7B)"]
    for name in order:
        m = existing.get(name, {})
        if "error" in m:
            row = f"| {name} | ERROR | ERROR | ERROR | ERROR |"
        else:
            row = f"| {name} | {m.get('hits@1','N/A')} | {m.get('hits@3','N/A')} | {m.get('mrr','N/A')} | {m.get('macro_f1','N/A')} |"
        lines.append(row)

    with open(os.path.join(report_dir, "UPDATED_RESULTS_TABLE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nFinal table written.")
    print("\n=== UPDATED RESULTS TABLE ===")
    print(f"{'Model':<25} {'H@1':>8} {'H@3':>8} {'MRR':>10} {'F1':>10}")
    print("-" * 65)
    for name in order:
        m = existing.get(name, {})
        if "error" in m:
            print(f"{name:<25} {'ERROR':>8} {'ERROR':>8} {'ERROR':>10} {'ERROR':>10}")
        else:
            print(f"{name:<25} {str(m.get('hits@1','N/A')):>8} {str(m.get('hits@3','N/A')):>8} {str(m.get('mrr','N/A')):>10} {str(m.get('macro_f1','N/A')):>10}")

if __name__ == "__main__":
    main()
