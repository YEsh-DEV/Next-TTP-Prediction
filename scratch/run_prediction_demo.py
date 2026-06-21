import os
import sys
import json

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.query_pipeline import GraphRAGPipeline
from pipeline.prediction_layer import TemporalSequencer
from pipeline.markov_predictor import MarkovPredictor
from pipeline.graph_builder import SubgraphBuilder
from pipeline.prediction_evaluation import PredictionEvaluator

def run_demo():
    pipeline = GraphRAGPipeline(base_dir)
    sequencer = TemporalSequencer()
    markov = MarkovPredictor()
    builder = SubgraphBuilder()
    evaluator = PredictionEvaluator()
    
    queries = [
        "Burning Umbrella",
        "Operation Kitty",
        "Lazarus Cryptocurrency"
    ]
    
    output_path = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\prediction_demo_report.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# PHASE-6 PREDICTION LAYER DEMO\n\n")
        f.write("Executing the complete GraphRAG + Temporal Sequencing + Markov Chain prediction pipeline.\n\n")
        
        for q in queries:
            f.write(f"## Target Query: `{q}`\n\n")
            
            # 1. Retrieval & GraphRAG Pipeline
            payload = pipeline.execute_query(q, top_k=3, classification_mode="deterministic")
            retrieved_events = payload["retrieved_events"]
            
            # 2. Temporal Sequencer
            event_sequence = sequencer.sequence_events(retrieved_events, pipeline.all_events)
            
            # 3. Markov Predictor (Fit & Predict)
            # We treat the sequence as a single chronological trajectory to train the matrix
            markov.build_matrix([event_sequence])
            
            # Predict what event comes after the *last* known event in the sequence
            last_event = event_sequence[-1] if event_sequence else None
            predictions = markov.top_k_predictions(last_event, k=3) if last_event else []
            
            # 4. Graph Construction
            G = builder.build_graph(q, event_sequence, payload)
            
            # 5. Statistical Evaluation
            stats = evaluator.evaluate(event_sequence, G, markov.total_transitions)
            
            # Compile Final Payload
            final_payload = {
                "query": q,
                "retrieved_events": retrieved_events,
                "event_sequence": event_sequence,
                "graph_statistics": stats,
                "predicted_next_nodes": predictions
            }
            
            f.write("```json\n")
            f.write(json.dumps(final_payload, indent=4) + "\n")
            f.write("```\n\n")

    print(f"Prediction Proof saved to {output_path}")

if __name__ == "__main__":
    run_demo()
