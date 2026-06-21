import networkx as nx
from typing import List

class PredictionEvaluator:
    """
    Computes mathematical topology statistics for the resulting prediction graph.
    """
    def evaluate(self, event_sequence: List[str], G: nx.DiGraph, total_transitions: int) -> dict:
        return {
            "sequence_length": len(event_sequence),
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "graph_density": round(nx.density(G), 4) if G.number_of_nodes() > 0 else 0.0,
            "transition_count": total_transitions
        }
