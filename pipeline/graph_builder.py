import networkx as nx
from typing import List, Dict

class SubgraphBuilder:
    """
    Builds the localized, dynamic NetworkX graph around the retrieved query subgraph.
    """
    def build_graph(self, query: str, event_sequence: List[str], mapping_payload: dict) -> nx.DiGraph:
        G = nx.DiGraph()
        
        # Add Query Root Node
        G.add_node(query, type="Query")
        
        # Add Temporal Events and PRECEDES edges
        for i in range(len(event_sequence)):              
            evt = event_sequence[i]
            G.add_node(evt, type="Event")
            G.add_edge(evt, query, relationship="RETRIEVED_FROM")
            
            if i > 0:
                prev_evt = event_sequence[i-1]
                G.add_edge(prev_evt, evt, relationship="PRECEDES")
                
        # Add Mapped Techniques
        for t in mapping_payload.get("mapped_techniques", []):
            G.add_node(t, type="Technique")
            for evt in event_sequence:
                G.add_edge(evt, t, relationship="OBSERVED_TECHNIQUE")
                
        # Add Neo4j Associated Infrastructure
        for s in mapping_payload.get("related_software", []):
            G.add_node(s, type="Software")
            for t in mapping_payload.get("mapped_techniques", []):
                G.add_edge(s, t, relationship="USES")
                
        for apt in mapping_payload.get("related_apt_groups", []):
            G.add_node(apt, type="APTGroup")
            for t in mapping_payload.get("mapped_techniques", []):
                G.add_edge(apt, t, relationship="USES")
            for s in mapping_payload.get("related_software", []):
                G.add_edge(apt, s, relationship="USES")
                
        return G
