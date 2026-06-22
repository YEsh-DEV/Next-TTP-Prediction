import os
import sys
import argparse
import time

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from pipeline.query_pipeline import GraphRAGPipeline
from pipeline.deterministic_classifier_v2 import DeterministicClassifierV2
from pipeline.markov_predictor import GlobalMarkovPredictor

class CLIWrapper:
    def __init__(self, clear_screen=True):
        if clear_screen:
            print("Loading weights and initializing Datastores... (ChromaDB, Neo4j, SentenceTransformers)")
            
        self.pipeline = GraphRAGPipeline(base_dir)
        self.pipeline.deterministic_classifier = DeterministicClassifierV2(base_dir)
        self.markov = GlobalMarkovPredictor(base_dir)
        
        if clear_screen:
            os.system('cls' if os.name == 'nt' else 'clear')

    def get_related_apts(self, techniques):
        from collections import Counter
        apt_counter = Counter()
        if not techniques:
            return []
        try:
            with self.pipeline.neo4j_driver.session() as session:
                for t in techniques:
                    res = session.run("MATCH (a:APTGroup)-[:USES]->(t:Technique {technique_id: $tid}) RETURN a.name AS name", tid=t)
                    for r in res:
                        apt_counter[r["name"]] += 1
                        
                    res2 = session.run("MATCH (a:APTGroup)-[:USES]->(s:Software)-[:USES]->(t:Technique {technique_id: $tid}) RETURN a.name AS name", tid=t)
                    for r in res2:
                        apt_counter[r["name"]] += 1
        except Exception:
            pass
        return apt_counter.most_common(10)

    def execute_and_print(self, query):
        start_time = time.time()
        payload = self.pipeline.execute_query(query, top_k=3, classification_mode="deterministic")
        
        retrieved = [str(x) for x in payload.get("retrieved_events", [])]
        mapped = payload.get("mapped_techniques", [])
        
        preds = []
        if mapped:
            for t in mapped:
                preds = self.markov.top_k_predictions(t, k=3)
                if preds:
                    break
            
        apts = self.get_related_apts(mapped)
        
        runtime = time.time() - start_time
        
        print("=================================================")
        print("QUERY")
        print("=====\n")
        print(f"{query}\n")
        
        print("=================================================")
        print("RETRIEVED EVENTS")
        print("================\n")
        for r in retrieved:
            print(r)
        print()
        
        print("=================================================")
        print("MAPPED TECHNIQUES")
        print("=================\n")
        for m in mapped:
            print(m)
        print()
            
        print("=================================================")
        print("RELATED APTS")
        print("============\n")
        for a, count in apts:
            print(f"{a} (Score: {count})")
        print()
            
        print("=================================================")
        print("PREDICTED NEXT TTPS")
        print("===================\n")
        for i, p in enumerate(preds):
            print(f"{i+1}. {p['state']} (Probability: {p['probability']})")
        print()
            
        print("=================================================")
        print("EXECUTION STATS")
        print("===============\n")
        print(f"Retrieved Events: {len(retrieved)}")
        print(f"Mapped Techniques: {len(mapped)}")
        print(f"Prediction Count: {len(preds)}")
        print(f"Runtime: {runtime:.2f}s\n")

    def demo_mode(self, query):
        import pandas as pd
        df = pd.read_excel(os.path.join(base_dir, "MitreEnterprise.xlsx"))
        mitre_dict = {str(row['Tactic ID']): str(row['Tactic Name']) for _, row in df.iterrows()}

        print("=================================================")
        print("PROFESSOR DEMO WORKFLOW")
        print("=================================================\n")
        print(f"1. QUERY: {query}\n")
        
        payload = self.pipeline.execute_query(query, top_k=3, classification_mode="deterministic")
        retrieved_ids = payload.get("retrieved_events", [])
        
        print("2. RETRIEVAL & CHRONOLOGICAL TIMELINE:")
        events_to_sort = []
        for eid in retrieved_ids:
            evt = self.pipeline.all_events.get(eid)
            if evt: events_to_sort.append(evt)
        events_to_sort.sort(key=lambda x: x.date)
        
        for i, evt in enumerate(events_to_sort):
            det = self.pipeline.deterministic_classifier.classify_event(evt)
            top_tech = det['techniques'][0]['id'] if det['techniques'] else "NONE"
            print(f"   [{i+1}] {evt.event_id} (Date: {evt.date}) -> {top_tech} ({mitre_dict.get(top_tech, 'Unknown')})")
        print()
        
        mapped = payload.get("mapped_techniques", [])
        print(f"3. TOP MAPPED TECHNIQUES: {', '.join(mapped[:5])} ...\n")
        
        apts = self.get_related_apts(mapped)
        print("4. GRAPH CONTEXT (TOP 5 APTS BY RELEVANCE):")
        for a, count in apts[:5]:
            print(f"   - {a} (Score: {count})")
        print()
        
        pred_state = None
        preds = []
        if mapped:
            for t in mapped:
                preds = self.markov.top_k_predictions(t, k=3)
                if preds:
                    pred_state = t
                    break
                    
        print(f"5. MARKOV PREDICTION (TERMINAL STATE: {pred_state}):")
        if preds:
            counts = self.markov.transition_counts.get(pred_state, {})
            total_outgoing = sum(counts.values())
            for i, p in enumerate(preds):
                nxt = p['state']
                cnt = counts.get(nxt, 0)
                prob = p['probability']
                name = mitre_dict.get(nxt, "Unknown")
                print(f"   [{i+1}] {nxt}: {name}")
                print(f"       Probability: {prob:.2f} (Transitions: {cnt}/{total_outgoing})")
        else:
            print("   No predictions available.")
        print("\n6. EXPLANATION:")
        print(f"   The Temporal GraphRAG pipeline mapped the query to the terminal technique {pred_state}.")
        print(f"   Using the global Markov transition matrix built from {len(self.pipeline.all_events)} events, it determined that the highly contextual next step is {preds[0]['state'] if preds else 'Unknown'} based on historical CTI occurrence rates.")
        print("=================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Experiment-3 Next-TTP Predictor CLI")
    parser.add_argument("--query", type=str, help="Run a specific query directly")
    parser.add_argument("--interactive", action="store_true", help="Run in continuous interactive loop")
    parser.add_argument("--demo", action="store_true", help="Run in Professor Demo mode")
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        # MODE 1
        print("=========================")
        print("Experiment-3")
        print("Temporal GraphRAG")
        print("Next-TTP Predictor")
        print("=========================")
        q = input("\nEnter query: ")
        cli = CLIWrapper()
        cli.execute_and_print(q)
        
    elif args.demo:
        # MODE 4
        cli = CLIWrapper()
        query = args.query if args.query else "Operation Kitty"
        cli.demo_mode(query)
        
    elif args.query:
        # MODE 2
        cli = CLIWrapper()
        cli.execute_and_print(args.query)
        
    elif args.interactive:
        # MODE 3
        cli = CLIWrapper()
        while True:
            try:
                q = input("Query: ")
                if q.lower() in ("exit", "quit"):
                    break
                print("\nPrediction:")
                cli.execute_and_print(q)
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
