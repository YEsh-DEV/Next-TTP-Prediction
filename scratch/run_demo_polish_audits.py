import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pipeline.query_pipeline import GraphRAGPipeline
from pipeline.markov_predictor import GlobalMarkovPredictor
from pipeline.prediction_layer import TemporalSequencer

report_dir = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d"

def run_task1_and_3():
    pipeline = GraphRAGPipeline(base_dir)
    markov = GlobalMarkovPredictor(base_dir)
    sequencer = TemporalSequencer()
    
    queries = ["Burning Umbrella", "Operation Kitty", "Lazarus Cryptocurrency"]
    
    with open(os.path.join(report_dir, "terminal_state_selection_proof.md"), "w", encoding="utf-8") as f1, \
         open(os.path.join(report_dir, "prediction_confidence_report.md"), "w", encoding="utf-8") as f3:
        
        f1.write("# TERMINAL STATE SELECTION PROOF\n\n")
        f3.write("# PREDICTION CONFIDENCE REPORT\n\n")
        
        for q in queries:
            f1.write(f"## Query: {q}\n")
            f3.write(f"## Query: {q}\n")
            
            # Retrieval
            q_emb = pipeline.model.encode([q], show_progress_bar=False).tolist()[0]
            retrieval_res = pipeline.cti_collection.query(query_embeddings=[q_emb], n_results=3)
            retrieved_ids = retrieval_res['ids'][0]
            
            f1.write("**Retrieved Events:**\n")
            events_to_sort = []
            for eid in retrieved_ids:
                evt = pipeline.all_events.get(eid)
                if evt:
                    det = pipeline.deterministic_classifier.classify_event(evt)
                    top_tech = det['techniques'][0]['id'] if det['techniques'] else "NONE"
                    f1.write(f"- {eid} (Date: {evt.date}) → Mapped: {top_tech}\n")
                    events_to_sort.append(evt)
            
            # Sequencing
            events_to_sort.sort(key=lambda x: x.date)
            f1.write("\n**Chronological Ordering:**\n")
            for i, evt in enumerate(events_to_sort):
                det = pipeline.deterministic_classifier.classify_event(evt)
                top_tech = det['techniques'][0]['id'] if det['techniques'] else "NONE"
                f1.write(f"{i+1}. {evt.event_id} ({evt.date}) → {top_tech}\n")
                
            terminal_event = events_to_sort[-1] if events_to_sort else None
            terminal_state = "NONE"
            if terminal_event:
                det = pipeline.deterministic_classifier.classify_event(terminal_event)
                if det['techniques']:
                    terminal_state = det['techniques'][0]['id']
            
            f1.write(f"\n**Terminal State = {terminal_state}**\n")
            f1.write("\n**Reason:**\n")
            f1.write(f"It is the primary mapped technique of the chronologically latest retrieved event ({terminal_event.event_id} from {terminal_event.date}). The predictor assumes the attacker's most recent action dictates their next logical step.\n\n")
            
            # Task 3 Math
            f3.write(f"**Current State:** {terminal_state}\n\n")
            preds = markov.top_k_predictions(terminal_state, k=3)
            
            if preds:
                counts = markov.transition_counts.get(terminal_state, {})
                total_outgoing = sum(counts.values())
                
                f3.write(f"**Total Outgoing Transitions:** {total_outgoing}\n\n")
                
                for p in preds:
                    nxt = p['state']
                    cnt = counts.get(nxt, 0)
                    prob = p['probability']
                    
                    f3.write(f"### {terminal_state} -> {nxt}\n")
                    f3.write(f"- Count = {cnt}\n")
                    f3.write(f"- Total Outgoing = {total_outgoing}\n")
                    f3.write(f"- Probability = {prob:.2f}\n\n")
            else:
                f3.write("No predictions available for this state.\n\n")

if __name__ == "__main__":
    run_task1_and_3()
    print("Tasks 1 and 3 generated.")
