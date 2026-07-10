"""
demo_predict.py — Experiment-3 Inference API

Usage:
    python demo_predict.py --actor Turla --technique T1213
    python demo_predict.py --actor Lazarus --technique T1291
"""
import os, sys, re, argparse, json
import numpy as np, torch
from collections import defaultdict
from datetime import datetime
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from pipeline.xml_parser import parse_xml_file
from pipeline.deterministic_classifier_v2 import DeterministicClassifierV2

DATASET_DIR = os.path.join(base_dir, "CTI_Report_Dataset")
TOP_N = 2

APT_PATTERN = re.compile(
    r'(APT\s?\d+|Lazarus|Turla|Sofacy|Fancy Bear|Cozy Bear|HAFNIUM|Equation Group|'
    r'FIN\d+|Carbanak|Buhtrap|MuddyWater|OceanLotus|Kimsuky|Sandworm|' 
    r'REvil|LockBit|Ryuk|DarkSide|Conti|BlackMatter|Cl0p|TA\d+|SILENCE|' 
    r'Gorgon|Gallmaker|Patchwork|Tick|Rancor|Andariel|Bluenoroff|' 
    r'ChessMaster|RedEyes|VERMIN|BISMUTH|Operation\s+\w+)', re.IGNORECASE)

def get_date(e):
    if e.date:
        try: return datetime.strptime(e.date, "%Y-%m-%d")
        except: pass
    return datetime.min

def build_benchmark(classifier):
    xml_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith("ReportEvent.xml")])
    all_events = []
    for fname in xml_files:
        all_events.extend(parse_xml_file(os.path.join(DATASET_DIR, fname)))
    actor_groups = defaultdict(dict)
    for evt in all_events:
        for m in APT_PATTERN.findall(evt.info or ""):
            actor_groups[m.strip().title()][evt.event_id] = evt   
    timelines = {a: sorted(e.values(), key=lambda x: (get_date(x), x.event_id))
                 for a, e in actor_groups.items() if len(e) > 1}
    cache = {}
    def get_techs(evt):
        if evt.event_id not in cache:
            res = classifier.classify_event(evt)
            cache[evt.event_id] = [t["id"] for t in res["techniques"][:TOP_N]] if res and res.get("techniques") else []
        return cache[evt.event_id]
    rows = []
    for actor, evts in timelines.items():
        for i in range(len(evts)-1):
            for st in get_techs(evts[i]):
                for tt in get_techs(evts[i+1]):
                    rows.append({"src_node": f"{actor}::{st}", "tgt_node": f"{actor}::{tt}"})
    return rows

def train_rotate_model(rows):
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline as kge_pipeline
    triples = np.array([[r["src_node"], "NEXT_TTP", r["tgt_node"]] for r in rows], dtype=str)
    tf = TriplesFactory.from_labeled_triples(triples=triples)
    train_tf, _ = tf.split([0.8, 0.2], random_state=42)
    result = kge_pipeline(model="RotatE", training=train_tf, testing=_,
        training_kwargs={"num_epochs": 200, "batch_size": min(256, train_tf.num_triples)},
        model_kwargs={"embedding_dim": 128}, random_seed=42, device="cpu")
    return result.model, tf.entity_to_id, tf.relation_to_id["NEXT_TTP"]

def predict(actor, technique, model, entity2id, rel_id, topk=5):
    id2entity = {v: k for k, v in entity2id.items()}
    node = f"{actor.title()}::{technique}"
    if node not in entity2id:
        # Try case-insensitive match
        for k in entity2id:
            if k.lower() == node.lower():
                node = k
                break
        else:
            return {"error": f"Node '{node}' not found in graph. Known actors: {sorted(set(k.split(chr(58)*2)[0] for k in entity2id))[:10]}"}
    src_id = torch.tensor([[entity2id[node], rel_id]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        scores = model.score_t(hr_batch=src_id).squeeze(0)
    ranked_ids = torch.argsort(scores, descending=True).tolist()
    predictions = []
    for idx in ranked_ids[:topk]:
        entity = id2entity[idx]
        if "::" in entity:
            pred_actor, pred_tech = entity.split("::", 1)
            if pred_actor == actor.title():  # Same actor preferred
                predictions.append({"technique": pred_tech, "node": entity, "score": float(scores[idx])})
    if not predictions:  # fallback: return any top-k
        for idx in ranked_ids[:topk]:
            entity = id2entity[idx]
            if "::" in entity:
                _, pred_tech = entity.split("::", 1)
                predictions.append({"technique": pred_tech, "node": entity, "score": float(scores[idx])})
    return {"actor": actor, "current_technique": technique, "predictions": predictions[:topk]}

def main():
    parser = argparse.ArgumentParser(description="Experiment-3 Next-TTP Prediction")
    parser.add_argument("--actor", default="Turla")
    parser.add_argument("--technique", default="T1213")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    print(f"Loading data and training RotatE model (first run takes ~60s)...")
    classifier = DeterministicClassifierV2(base_dir)
    rows = build_benchmark(classifier)
    print(f"  Benchmark: {len(rows)} transitions")
    model, entity2id, rel_id = train_rotate_model(rows)
    result = predict(args.actor, args.technique, model, entity2id, rel_id, topk=args.topk)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
