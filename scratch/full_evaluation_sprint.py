"""
EXPERIMENT-3 COMPLETE EVALUATION SPRINT
Tasks: Actor-Aware KGE + GNN + LLM evaluation
All models evaluated on the corrected 648-transition benchmark.
"""
import os, sys, re, csv, json, warnings, math, requests
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import torch.nn as nn
import torch.nn.functional as F

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

report_dir = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d"

# ─── Load the 648-transition benchmark ───────────────────────────────────────
def load_benchmark():
    path = os.path.join(report_dir, "true_benchmark_full.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

# ─── TASK 1: Build actor-aware nodes ─────────────────────────────────────────
def make_actor_aware(rows):
    """Returns new rows where src/tgt are Actor::Technique strings."""
    new_rows = []
    for r in rows:
        actor = r["actor"]
        new_rows.append({
            **r,
            "src_node": f"{actor}::{r['src_technique']}",
            "tgt_node": f"{actor}::{r['tgt_technique']}",
        })
    return new_rows

# ─── Encode nodes/relations to integers ──────────────────────────────────────
def encode(rows, src_col="src_node", tgt_col="tgt_node", rel_col=None):
    all_nodes = sorted(set(r[src_col] for r in rows) | set(r[tgt_col] for r in rows))
    node2id = {n: i for i, n in enumerate(all_nodes)}
    return node2id, all_nodes

# ─── Chronological 80/20 split ───────────────────────────────────────────────
def split(rows):
    rows_sorted = sorted(rows, key=lambda x: (x.get("src_date", ""), x.get("src_event", "")))
    n = len(rows_sorted)
    cut = int(n * 0.8)
    return rows_sorted[:cut], rows_sorted[cut:]

# ─── TASK 2: Benchmark statistics ────────────────────────────────────────────
def compute_stats(rows, actor_rows):
    nodes = set(r["src_node"] for r in actor_rows) | set(r["tgt_node"] for r in actor_rows)
    techniques = set(r["src_technique"] for r in rows) | set(r["tgt_technique"] for r in rows)
    actors = set(r["actor"] for r in rows)

    in_deg = Counter()
    out_deg = Counter()
    for r in actor_rows:
        out_deg[r["src_node"]] += 1
        in_deg[r["tgt_node"]] += 1

    all_nodes_list = list(nodes)
    avg_out = sum(out_deg.values()) / len(all_nodes_list) if all_nodes_list else 0
    dead_ends = sum(1 for n in all_nodes_list if n not in out_deg)

    return {
        "total_nodes": len(nodes),
        "total_edges": len(actor_rows),
        "unique_actors": len(actors),
        "unique_techniques": len(techniques),
        "avg_out_degree": round(avg_out, 2),
        "dead_end_nodes": dead_ends,
    }

# ─── TASK 3: Technique distribution ─────────────────────────────────────────
def technique_distribution(rows):
    tech_count = Counter()
    for r in rows:
        tech_count[r["src_technique"]] += 1
        tech_count[r["tgt_technique"]] += 1
    return tech_count.most_common(50)

# ─── Evaluation helper ───────────────────────────────────────────────────────
def evaluate_predictions(predictions, targets, all_entities, k_list=[1, 3]):
    """
    predictions: list of ranked lists of entity indices
    targets: list of true target indices
    Returns hits@k, MRR, macro-F1
    """
    hits = {k: 0 for k in k_list}
    mrr = 0.0
    n = len(targets)

    tp = Counter(); fp = Counter(); fn = Counter()

    for pred_list, true_idx in zip(predictions, targets):
        rank = None
        for pos, p in enumerate(pred_list, start=1):
            if p == true_idx:
                rank = pos
                break
        if rank:
            mrr += 1.0 / rank
            for k in k_list:
                if rank <= k:
                    hits[k] += 1
            tp[true_idx] += 1
        else:
            fn[true_idx] += 1
        if pred_list:
            if pred_list[0] != true_idx:
                fp[pred_list[0]] += 1

    all_classes = set(targets)
    precs, recs = [], []
    for c in all_classes:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        precs.append(p); recs.append(r)

    macro_p = float(np.mean(precs)) if precs else 0.0
    macro_r = float(np.mean(recs)) if recs else 0.0
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0

    return {
        **{f"hits@{k}": round(100 * hits[k] / n, 2) for k in k_list},
        "mrr": round(mrr / n, 4),
        "macro_f1": round(macro_f1, 4),
    }

# ─── TASK 4/5: TransE / RotatE via PyKEEN ────────────────────────────────────
def train_kge(train_rows, test_rows, model_name="TransE", dim=128, epochs=200):
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline as kge_pipeline

    # Build triple arrays: (head, relation, tail) where relation = "NEXT_TTP"
    def to_triples(rows):
        return [[r["src_node"], "NEXT_TTP", r["tgt_node"]] for r in rows]

    train_triples = to_triples(train_rows)
    test_triples = to_triples(test_rows)

    # PyKEEN needs a factory built from all triples to share entity/relation vocab
    all_triples = train_triples + test_triples
    all_tf = TriplesFactory.from_labeled_triples(
        triples=np.array(all_triples, dtype=str),
        create_inverse_triples=False,
    )
    train_tf = all_tf.new_with_restriction(
        entities=None,
        relations=None,
        drop_relations_not_in_training=False,
    )

    # Recreate proper train/test splits
    train_tf = TriplesFactory.from_labeled_triples(
        triples=np.array(train_triples, dtype=str),
        entity_to_id=all_tf.entity_to_id,
        relation_to_id=all_tf.relation_to_id,
    )
    test_tf = TriplesFactory.from_labeled_triples(
        triples=np.array(test_triples, dtype=str),
        entity_to_id=all_tf.entity_to_id,
        relation_to_id=all_tf.relation_to_id,
    )

    print(f"  Running {model_name}: {len(train_triples)} train, {len(test_triples)} test triples")

    result = kge_pipeline(
        model=model_name,
        training=train_tf,
        testing=test_tf,
        training_kwargs={"num_epochs": epochs, "batch_size": min(256, len(train_triples))},
        model_kwargs={"embedding_dim": dim},
        random_seed=42,
        device="cpu",
    )

    metrics = result.metric_results.to_flat_dict()
    h1 = round(metrics.get("both.realistic.hits_at_1", 0) * 100, 2)
    h3 = round(metrics.get("both.realistic.hits_at_3", 0) * 100, 2)
    mrr = round(metrics.get("both.realistic.inverse_harmonic_mean_rank", 0), 4)
    return {"hits@1": h1, "hits@3": h3, "mrr": mrr, "macro_f1": "N/A (ranking metric)"}

# ─── TASK 6 (extension): R-GCN via PyG ───────────────────────────────────────
def train_rgcn(train_rows, test_rows, all_rows):
    try:
        from torch_geometric.data import Data
        from torch_geometric.nn import RGCNConv
    except ImportError:
        return {"hits@1": "N/A", "hits@3": "N/A", "mrr": "N/A", "macro_f1": "N/A", "note": "PyG not available"}

    node2id, all_nodes = encode(all_rows)
    num_nodes = len(node2id)

    edge_index_train = torch.tensor(
        [[node2id[r["src_node"]], node2id[r["tgt_node"]]] for r in train_rows],
        dtype=torch.long
    ).t().contiguous()
    edge_type_train = torch.zeros(len(train_rows), dtype=torch.long)  # 1 relation type

    class RGCN(nn.Module):
        def __init__(self, num_nodes, dim, num_rels):
            super().__init__()
            self.emb = nn.Embedding(num_nodes, dim)
            self.conv1 = RGCNConv(dim, dim, num_rels)
            self.conv2 = RGCNConv(dim, dim, num_rels)
            self.decoder = nn.Bilinear(dim, dim, 1)

        def forward(self, edge_index, edge_type):
            x = self.emb.weight
            x = F.relu(self.conv1(x, edge_index, edge_type))
            x = self.conv2(x, edge_index, edge_type)
            return x

    model = RGCN(num_nodes, 64, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Negative sampling training
    train_src = edge_index_train[0]
    train_tgt = edge_index_train[1]

    print(f"  Training R-GCN: {len(train_rows)} edges, {num_nodes} nodes, 100 epochs...")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        embeddings = model(edge_index_train, edge_type_train)

        pos_scores = (embeddings[train_src] * embeddings[train_tgt]).sum(dim=-1)
        neg_tgt = torch.randint(0, num_nodes, (len(train_src),))
        neg_scores = (embeddings[train_src] * embeddings[neg_tgt]).sum(dim=-1)

        loss = F.margin_ranking_loss(pos_scores, neg_scores, torch.ones_like(pos_scores), margin=1.0)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        all_embs = model(edge_index_train, edge_type_train)

    predictions = []
    targets_list = []
    for r in test_rows:
        if r["src_node"] not in node2id or r["tgt_node"] not in node2id:
            continue
        src_id = node2id[r["src_node"]]
        tgt_id = node2id[r["tgt_node"]]

        src_emb = all_embs[src_id]
        scores = (src_emb.unsqueeze(0) * all_embs).sum(dim=-1)
        ranked = scores.argsort(descending=True).tolist()
        predictions.append(ranked[:10])
        targets_list.append(tgt_id)

    if not targets_list:
        return {"hits@1": 0, "hits@3": 0, "mrr": 0, "macro_f1": 0, "note": "no test coverage"}

    return evaluate_predictions(predictions, targets_list, list(node2id.values()))

# ─── Temporal GNN (TGNN): simple LSTM over actor sequences ───────────────────
def train_temporal_gnn(train_rows, test_rows, all_rows):
    node2id, all_nodes = encode(all_rows)
    num_nodes = len(node2id)

    # Build per-actor sequences
    actor_seq = defaultdict(list)
    for r in sorted(train_rows, key=lambda x: (x.get("src_date", ""), x.get("src_event", ""))):
        actor_seq[r["actor"]].append(node2id[r["src_node"]])
    # Final target
    for r in sorted(train_rows, key=lambda x: (x.get("tgt_date", ""), x.get("tgt_event", ""))):
        actor_seq[r["actor"]].append(node2id[r["tgt_node"]])

    class TemporalGNN(nn.Module):
        def __init__(self, num_nodes, dim):
            super().__init__()
            self.emb = nn.Embedding(num_nodes, dim)
            self.lstm = nn.LSTM(dim, dim, batch_first=True)
            self.proj = nn.Linear(dim, num_nodes)

        def forward(self, seq):
            x = self.emb(seq)  # (T, dim)
            out, _ = self.lstm(x.unsqueeze(0))  # (1, T, dim)
            logits = self.proj(out[0])  # (T, num_nodes)
            return logits

    model = TemporalGNN(num_nodes, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"  Training Temporal GNN: {len(actor_seq)} actor sequences, 100 epochs...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        for actor, seq in actor_seq.items():
            if len(seq) < 2:
                continue
            optimizer.zero_grad()
            seq_t = torch.tensor(seq[:-1], dtype=torch.long)
            tgt_t = torch.tensor(seq[1:], dtype=torch.long)
            logits = model(seq_t)
            loss = F.cross_entropy(logits, tgt_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Evaluate on test set
    model.eval()
    predictions = []
    targets_list = []

    with torch.no_grad():
        for r in test_rows:
            if r["src_node"] not in node2id or r["tgt_node"] not in node2id:
                continue
            src_id = node2id[r["src_node"]]
            tgt_id = node2id[r["tgt_node"]]
            seq_t = torch.tensor([src_id], dtype=torch.long)
            logits = model(seq_t)  # (1, num_nodes)
            scores = logits[0]
            ranked = scores.argsort(descending=True).tolist()
            predictions.append(ranked[:10])
            targets_list.append(tgt_id)

    if not targets_list:
        return {"hits@1": 0, "hits@3": 0, "mrr": 0, "macro_f1": 0}

    return evaluate_predictions(predictions, targets_list, list(node2id.values()))

# ─── LLM evaluation via Ollama ────────────────────────────────────────────────
def evaluate_llm(test_rows, sample_n=30):
    base_url = "http://localhost:11434"
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        if resp.status_code != 200:
            return {"hits@1": "OFFLINE", "hits@3": "OFFLINE", "mrr": "OFFLINE", "macro_f1": "OFFLINE"}
    except:
        return {"hits@1": "OFFLINE", "hits@3": "OFFLINE", "mrr": "OFFLINE", "macro_f1": "OFFLINE"}

    t_pattern = re.compile(r'T\d{4}')
    sample = test_rows[:sample_n]
    hits1 = hits3 = 0

    for r in sample:
        actor = r["actor"]
        src_t = r["src_technique"]
        actual = r["tgt_technique"]

        prompt = (
            f"You are a cybersecurity expert. The threat actor {actor} used MITRE ATT&CK technique {src_t}. "
            f"What are the top 3 most likely NEXT techniques they will use? "
            f"Respond with ONLY technique IDs like T1059, T1105, T1003. No explanation."
        )

        try:
            resp = requests.post(
                f"{base_url}/api/generate",
                json={"model": "qwen2.5-coder:7b", "prompt": prompt, "stream": False},
                timeout=30,
            )
            text = resp.json().get("response", "")
            found = t_pattern.findall(text)
            if found:
                if actual == found[0]:
                    hits1 += 1
                if actual in found[:3]:
                    hits3 += 1
        except:
            continue

    n = len(sample)
    return {
        "hits@1": round(100 * hits1 / n, 2),
        "hits@3": round(100 * hits3 / n, 2),
        "mrr": "N/A",
        "macro_f1": "N/A (sample-based)",
    }

# ─── Write all reports ────────────────────────────────────────────────────────
def write_reports(stats, tech_dist, results, actor_rows, train_rows, test_rows):
    # 1. Actor-aware dataset report
    lines = ["# ACTOR-AWARE DATASET REPORT\n"]
    lines.append("## Graph Statistics\n")
    for k, v in stats.items():
        lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
    lines.append("\n## Sample Actor-Aware Nodes (first 30)")
    seen = set()
    for r in actor_rows[:60]:
        for n in [r["src_node"], r["tgt_node"]]:
            if n not in seen:
                lines.append(f"- `{n}`")
                seen.add(n)
            if len(seen) >= 30:
                break
        if len(seen) >= 30:
            break
    with open(os.path.join(report_dir, "ACTOR_AWARE_DATASET_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 2. Technique distribution report
    lines = ["# TECHNIQUE DISTRIBUTION REPORT\n"]
    lines.append("## Top 50 Techniques by Frequency\n")
    lines.append("| Rank | Technique | Count |")
    lines.append("| --- | --- | --- |")
    for i, (t, c) in enumerate(tech_dist, 1):
        lines.append(f"| {i} | `{t}` | {c} |")
    top5_count = sum(c for _, c in tech_dist[:5])
    total_count = sum(c for _, c in tech_dist)
    conc = round(100 * top5_count / total_count, 1) if total_count else 0
    lines.append(f"\n**Top-5 technique concentration:** {conc}% of all technique mentions")
    if conc > 60:
        lines.append("\n> [!WARNING] Classifier collapse detected. Top 5 techniques dominate >60% of events.")
    else:
        lines.append("\n> [!NOTE] Technique distribution is reasonably spread (no severe collapse).")
    with open(os.path.join(report_dir, "TECHNIQUE_DISTRIBUTION_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 3. KGE results report
    lines = ["# KGE RESULTS REPORT\n"]
    for model_name, metrics in results.items():
        lines.append(f"## {model_name}")
        for k, v in metrics.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")
    with open(os.path.join(report_dir, "KGE_RESULTS_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 4. Updated results table
    lines = ["# UPDATED RESULTS TABLE\n"]
    lines.append("*All values derived from the corrected 648-transition actor-isolated benchmark*\n")
    lines.append("| Model | Hits@1 | Hits@3 | MRR | F1 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for model_name, m in results.items():
        lines.append(f"| {model_name} | {m.get('hits@1', 'N/A')} | {m.get('hits@3', 'N/A')} | {m.get('mrr', 'N/A')} | {m.get('macro_f1', 'N/A')} |")
    with open(os.path.join(report_dir, "UPDATED_RESULTS_TABLE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EXPERIMENT-3 COMPLETE EVALUATION SPRINT")
    print("=" * 60)

    # Load
    print("\n[Step 1] Loading 648-transition benchmark...")
    rows = load_benchmark()

    # Task 1: actor-aware
    print("[Step 2] Building actor-aware nodes...")
    actor_rows = make_actor_aware(rows)

    # Split
    train_rows, test_rows = split(actor_rows)
    print(f"  Train: {len(train_rows)}, Test: {len(test_rows)}")

    # Task 2: stats
    print("[Step 3] Computing graph statistics...")
    stats = compute_stats(rows, actor_rows)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Task 3: technique distribution
    print("[Step 4] Computing technique distribution...")
    tech_dist = technique_distribution(rows)
    top5 = sum(c for _, c in tech_dist[:5])
    total = sum(c for _, c in tech_dist)
    print(f"  Top-5 concentration: {round(100*top5/total, 1)}% of all mentions")

    results = {}

    # Markov (reuse from previous sprint — just report known values)
    results["Markov Chain"] = {"hits@1": "3.85%", "hits@3": "10.77%", "mrr": 0.0780, "macro_f1": 0.0092}
    print("\n[Step 5] Markov: Using previously computed results (3.85% / 10.77%)")

    # Task 4: TransE
    print("\n[Step 6] Training TransE (dim=128, epochs=200)...")
    try:
        r_transe = train_kge(train_rows, test_rows, "TransE", dim=128, epochs=200)
        results["TransE"] = r_transe
        print(f"  TransE -> H@1={r_transe['hits@1']}%, H@3={r_transe['hits@3']}%, MRR={r_transe['mrr']}")
    except Exception as e:
        results["TransE"] = {"hits@1": "ERROR", "hits@3": "ERROR", "mrr": "ERROR", "macro_f1": "ERROR", "note": str(e)[:100]}
        print(f"  TransE ERROR: {e}")

    # Task 5: RotatE
    print("\n[Step 7] Training RotatE (dim=128, epochs=200)...")
    try:
        r_rotate = train_kge(train_rows, test_rows, "RotatE", dim=128, epochs=200)
        results["RotatE"] = r_rotate
        print(f"  RotatE -> H@1={r_rotate['hits@1']}%, H@3={r_rotate['hits@3']}%, MRR={r_rotate['mrr']}")
    except Exception as e:
        results["RotatE"] = {"hits@1": "ERROR", "hits@3": "ERROR", "mrr": "ERROR", "macro_f1": "ERROR", "note": str(e)[:100]}
        print(f"  RotatE ERROR: {e}")

    # R-GCN
    print("\n[Step 8] Training R-GCN (64-dim, 100 epochs)...")
    try:
        r_rgcn = train_rgcn(train_rows, test_rows, actor_rows)
        results["GAT / R-GCN"] = r_rgcn
        print(f"  R-GCN -> H@1={r_rgcn.get('hits@1')}%, H@3={r_rgcn.get('hits@3')}%, MRR={r_rgcn.get('mrr')}")
    except Exception as e:
        results["GAT / R-GCN"] = {"hits@1": "ERROR", "hits@3": "ERROR", "mrr": "ERROR", "macro_f1": "ERROR", "note": str(e)[:100]}
        print(f"  R-GCN ERROR: {e}")

    # Temporal GNN
    print("\n[Step 9] Training Temporal GNN (LSTM, 64-dim, 100 epochs)...")
    try:
        r_tgnn = train_temporal_gnn(train_rows, test_rows, actor_rows)
        results["Temporal GNN"] = r_tgnn
        print(f"  Temporal GNN -> H@1={r_tgnn.get('hits@1')}%, H@3={r_tgnn.get('hits@3')}%, MRR={r_tgnn.get('mrr')}")
    except Exception as e:
        results["Temporal GNN"] = {"hits@1": "ERROR", "hits@3": "ERROR", "mrr": "ERROR", "macro_f1": "ERROR", "note": str(e)[:100]}
        print(f"  Temporal GNN ERROR: {e}")

    # LLM (Ollama)
    print("\n[Step 10] Evaluating LLM (Qwen 7B via Ollama, 30-sample subset)...")
    try:
        r_llm = evaluate_llm(test_rows, sample_n=30)
        results["LLM (Qwen 7B)"] = r_llm
        print(f"  LLM -> H@1={r_llm.get('hits@1')}%, H@3={r_llm.get('hits@3')}")
    except Exception as e:
        results["LLM (Qwen 7B)"] = {"hits@1": "ERROR", "hits@3": "ERROR", "mrr": "ERROR", "macro_f1": "ERROR", "note": str(e)[:100]}
        print(f"  LLM ERROR: {e}")

    # Write all reports
    print("\n[Step 11] Writing all reports...")
    write_reports(stats, tech_dist, results, actor_rows, train_rows, test_rows)

    print("\n" + "=" * 60)
    print("FINAL RESULTS TABLE")
    print("=" * 60)
    print(f"{'Model':<25} {'H@1':>8} {'H@3':>8} {'MRR':>10} {'F1':>10}")
    print("-" * 65)
    for model_name, m in results.items():
        print(f"{model_name:<25} {str(m.get('hits@1','N/A')):>8} {str(m.get('hits@3','N/A')):>8} {str(m.get('mrr','N/A')):>10} {str(m.get('macro_f1','N/A')):>10}")
    print("=" * 60)
    print("\nAll reports written. Sprint complete.")

if __name__ == "__main__":
    main()
