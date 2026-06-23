"""
Export Data and Model Assets for Experiment-3

Builds the frozen Top-2 Actor-aware benchmark, saves the DataFrames to
data/final_benchmark/ (nodes.csv, edges.csv, train.csv, test.csv),
trains RotatE one last time, and exports the model to models/.
"""
import os, sys, json
import pandas as pd
import numpy as np
import torch

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.deterministic_classifier_v2 import DeterministicClassifierV2
from scratch.final_production_run import build_frozen_benchmark

def main():
    print("Loading official benchmark...")
    classifier = DeterministicClassifierV2(base_dir)
    train, test, all_nodes, all_techniques, timelines = build_frozen_benchmark(classifier)

    # 1. Export Data to CSV
    data_dir = os.path.join(base_dir, "data", "final_benchmark")
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Exporting dataset to {data_dir}...")
    
    # nodes.csv
    nodes_df = pd.DataFrame([{"node_id": n, "type": "Actor::Technique" if "::" in n else "Technique"} for n in all_nodes])
    nodes_df.to_csv(os.path.join(data_dir, "nodes.csv"), index=False)
    
    # edges.csv (all)
    edges_df = pd.DataFrame(train + test)
    edges_df.to_csv(os.path.join(data_dir, "edges.csv"), index=False)
    
    # train.csv & test.csv
    pd.DataFrame(train).to_csv(os.path.join(data_dir, "train.csv"), index=False)
    pd.DataFrame(test).to_csv(os.path.join(data_dir, "test.csv"), index=False)
    
    print("Dataset exported successfully.")

    # 2. Train and Export RotatE Model
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("Training RotatE for final model export (~60s)...")
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline as kge_pipeline

    all_rows = train + test
    all_triples = np.array([[r["src_node"], "NEXT_TTP", r["tgt_node"]] for r in all_rows], dtype=str)
    full_tf = TriplesFactory.from_labeled_triples(triples=all_triples)
    ratio = len(train) / len(all_rows)
    train_tf, test_tf = full_tf.split([ratio, 1 - ratio], random_state=42)

    result = kge_pipeline(
        model="RotatE", training=train_tf, testing=test_tf,
        training_kwargs={"num_epochs": 200, "batch_size": min(256, train_tf.num_triples)},
        model_kwargs={"embedding_dim": 128}, random_seed=42, device="cpu"
    )

    model = result.model
    model.eval()
    
    # Save Model Weights
    torch.save(model.state_dict(), os.path.join(models_dir, "rotate_final.pt"))
    
    # Save Mappings
    with open(os.path.join(models_dir, "entity_to_id.json"), "w") as f:
        json.dump(full_tf.entity_to_id, f, indent=2)
        
    with open(os.path.join(models_dir, "relation_to_id.json"), "w") as f:
        json.dump(full_tf.relation_to_id, f, indent=2)
        
    print(f"Model exported successfully to {models_dir}")

if __name__ == "__main__":
    main()
