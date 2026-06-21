import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.query_pipeline import GraphRAGPipeline
from pipeline.prediction_layer import TemporalSequencer
from pipeline.markov_predictor import GlobalMarkovPredictor
from pipeline.deterministic_classifier_v2 import DeterministicClassifierV2

def run_global_demo():
    print("Loading Global Markov Matrix with V2 Deterministic Engine...")
    pipeline = GraphRAGPipeline(base_dir)
    sequencer = TemporalSequencer()
    markov = GlobalMarkovPredictor(base_dir)
    classifier_v2 = DeterministicClassifierV2(base_dir)
    
    queries = ["Burning Umbrella", "Operation Kitty", "Lazarus Cryptocurrency"]
    
    for q in queries:
        print(f"\n=================================================")
        print(f"QUERY: {q}")
        print(f"=================================================")
        
        # 1. Retrieve Events
        payload = pipeline.execute_query(q, top_k=3, classification_mode="deterministic")
        retrieved_events = payload["retrieved_events"]
        
        # 2. Sequence Events
        event_sequence = sequencer.sequence_events(retrieved_events, pipeline.all_events)
        
        # 3. Get latest event & Map to Technique using V2
        if not event_sequence:
            continue
            
        latest_event_id = event_sequence[-1]
        latest_event_obj = pipeline.all_events.get(latest_event_id)
        
        res = classifier_v2.classify_event(latest_event_obj)
        if not res['techniques']:
            continue
            
        latest_technique = res['techniques'][0]['id']
        
        # 4. Predict Next Technique from Global Matrix
        predictions = markov.top_k_predictions(latest_technique, k=5)
        
        print(f"Current Technique (from latest event {latest_event_id}): {latest_technique}")
        print(f"Top 5 Predicted Next Techniques:")
        for i, p in enumerate(predictions):
            print(f"  {i+1}. {p['state']} (Probability: {p['probability']})")

if __name__ == "__main__":
    run_global_demo()
