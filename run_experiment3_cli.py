"""
Interactive Command Line Interface for Experiment-3
Next-TTP Prediction

Provides a user-friendly entry point for evaluating models,
running inference, and executing academic validation tests.
"""
import os, sys, argparse, json
from datetime import datetime

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

# Ensure data dependencies are available
try:
    from pipeline.deterministic_classifier_v2 import DeterministicClassifierV2
    from scratch.final_production_run import build_frozen_benchmark, train_markov, markov_predict_fn, train_rotate, run_evaluation
except ImportError:
    print("Warning: some pipeline dependencies could not be loaded. Are you running from the repository root?")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    clear_screen()
    print("===============================================================")
    print(" EXPERIMENT 3: Temporal-Causal GraphRAG & TTP Prediction CLI ")
    print("===============================================================")
    print(" This tool provides an interactive interface to evaluate ")
    print(" threat actor progression models on the CTI dataset. ")
    print("===============================================================\n")

def prompt_choice(prompt_text, options):
    print(prompt_text)
    for k, v in options.items():
        print(f"  {k}. {v}")
    while True:
        choice = input("\nSelect an option: ").strip()
        if choice in options:
            return choice
        print("Invalid selection. Try again.")

def run_full_pipeline(model_choice, llm_config=None):
    print("\n[1/4] Loading Frozen Top-2 Actor-Aware Benchmark...")
    classifier = DeterministicClassifierV2(base_dir)
    train, test, all_nodes, all_techniques, timelines = build_frozen_benchmark(classifier)
    print(f"  Graph: {len(all_nodes)} nodes, {len(train)+len(test)} transitions.")

    if model_choice == "1":
        print("\n[2/4] Training Markov Chain...")
        probs, fallback = train_markov(train)
        print("\n[3/4] Evaluating...")
        res = run_evaluation(test, markov_predict_fn(probs, fallback), "Markov Chain", all_nodes)
        
    elif model_choice == "3":
        print("\n[2/4] Training RotatE (PyKEEN)...")
        rotate_predict, node2id, id2node, rel_id, model = train_rotate(train, test, all_nodes)
        print("\n[3/4] Evaluating...")
        res = run_evaluation(test, rotate_predict, "RotatE", all_nodes)
        
    else:
        print("\n[2/4] Model selected is not yet integrated into the interactive loop.")
        print("Refer to scratch/evaluate_*.py for standalone implementations.")
        return

    print("\n[4/4] Final Results")
    print("-" * 50)
    print(f"  Model:      {res['model']}")
    print(f"  Hits@1:     {res['hits@1']}%")
    print(f"  Hits@3:     {res['hits@3']}%")
    print(f"  MRR:        {res['mrr']}")
    print(f"  Dead Ends:  {res['dead_ends']}")
    print("-" * 50)

def run_academic_validations():
    print("\nExecuting Academic Validations...")
    val_script = os.path.join(base_dir, "scratch", "academic_validation_tests.py")
    if os.path.exists(val_script):
        os.system(f'"{sys.executable}" "{val_script}"')
    else:
        print("Validation script not found.")

def run_interactive_inference():
    print("\nStarting Interactive Inference Console...")
    
    models_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(models_dir, "rotate_final.pt")
    
    import torch
    
    if os.path.exists(model_path):
        print("Loading Pre-trained RotatE embeddings from disk (Instant)...")
        with open(os.path.join(models_dir, "entity_to_id.json"), "r") as f:
            node2id = json.load(f)
        with open(os.path.join(models_dir, "relation_to_id.json"), "r") as f:
            rel_id = json.load(f)["NEXT_TTP"]
            
        id2node = {v: k for k, v in node2id.items()}
        
        from pykeen.models import RotatE
        from pykeen.triples import CoreTriplesFactory
        
        # We just need a dummy factory to initialize the model architecture with the right dimensions
        tf = CoreTriplesFactory.create(
            mapped_triples=torch.zeros((1, 3), dtype=torch.long),
            num_entities=len(node2id),
            num_relations=1
        )
        model = RotatE(triples_factory=tf, embedding_dim=128)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        print("Loading graph structure and training RotatE embeddings (~60s)...")
        classifier = DeterministicClassifierV2(base_dir)
        train, test, all_nodes, all_techniques, timelines = build_frozen_benchmark(classifier)
        rotate_predict, node2id, id2node, rel_id, model = train_rotate(train, test, all_nodes)
    
    while True:
        print("\n" + "="*40)
        actor = input("Enter Actor Name (or 'quit'): ").strip()
        if actor.lower() == 'quit': break
        tech = input("Enter Current Technique (e.g. T1213): ").strip()
        
        node = f"{actor.title()}::{tech}"
        if node not in node2id:
            # Try case-insensitive
            for k in node2id:
                if k.lower() == node.lower():
                    node = k
                    break
            else:
                print(f"Error: Node '{node}' not found in training graph.")
                continue
            
        src_id = torch.tensor([[node2id[node], rel_id]], dtype=torch.long)
        model.eval()
        with torch.no_grad():
            scores = model.score_t(hr_batch=src_id).squeeze(0)
        ranked_ids = torch.argsort(scores, descending=True).tolist()
        
        print(f"\nTop Predictions for {actor} after {tech}:")
        count = 0
        for idx in ranked_ids:
            entity = id2node[idx]
            if "::" in entity:
                pred_actor, pred_tech = entity.split("::", 1)
                if pred_actor == actor.title():
                    count += 1
                    print(f"  {count}. {pred_tech} (Score: {scores[idx]:.4f})")
                if count >= 3: break

def main():
    print_header()
    
    models = {
        "1": "Markov Chain",
        "2": "TransE",
        "3": "RotatE (Recommended)",
        "4": "GAT / R-GCN",
        "5": "Temporal GNN",
        "6": "LLM (Gemini/Groq/Ollama)"
    }
    
    actions = {
        "1": "Full Pipeline Execution (Train & Evaluate)",
        "2": "Run Academic Validations (Leakage, Ablation, Baseline)",
        "3": "Interactive Inference Demo"
    }

    action_choice = prompt_choice("What would you like to do?", actions)

    if action_choice == "1":
        model_choice = prompt_choice("\nSelect the predictive model to evaluate:", models)
        llm_config = None
        if model_choice == "6":
            llm_name = input("\nEnter LLM Provider (e.g., Gemini, Groq, Ollama): ").strip()
            api_key = input("Enter API Key (leave blank if local/Ollama): ").strip()
            llm_config = {"name": llm_name, "key": api_key}
        run_full_pipeline(model_choice, llm_config)
        
    elif action_choice == "2":
        run_academic_validations()
        
    elif action_choice == "3":
        run_interactive_inference()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
