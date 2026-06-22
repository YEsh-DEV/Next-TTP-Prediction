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
        apts = set()
        if not techniques:
            return list(apts)
        try:
            with self.pipeline.neo4j_driver.session() as session:
                for t in techniques:
                    res = session.run("MATCH (a:APTGroup)-[:USES]->(t:Technique {technique_id: $tid}) RETURN a.name AS name", tid=t)
                    for r in res:
                        apts.add(r["name"])
        except Exception:
            pass
        return list(apts)

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
        for a in apts:
            print(a)
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

def main():
    parser = argparse.ArgumentParser(description="Experiment-3 Next-TTP Predictor CLI")
    parser.add_argument("--query", type=str, help="Run a specific query directly")
    parser.add_argument("--interactive", action="store_true", help="Run in continuous interactive loop")
    
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
