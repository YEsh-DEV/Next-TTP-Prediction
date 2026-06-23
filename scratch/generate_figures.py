"""
Generate Publication-Quality Figures for Experiment-3

1. Pipeline Architecture Diagram (Using graphviz/networkx)
2. Actor-Aware Concept Graph
3. Results Bar Chart
"""
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fig_dir = os.path.join(base_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# Apply a professional style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

def generate_pipeline_figure():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Draw boxes
    boxes = [
        ("CTI Reports\n(XML Format)", (0.1, 0.5)),
        ("Actor Extraction\n(Regex / NLP)", (0.3, 0.5)),
        ("Semantic Mapping\n(Top-2 Techniques)", (0.5, 0.5)),
        ("Graph Construction\n(Actor::Technique Nodes)", (0.7, 0.5)),
        ("RotatE Prediction\n(Complex Space)", (0.9, 0.5))
    ]

    for i, (text, (x, y)) in enumerate(boxes):
        ax.text(x, y, text, ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#ecf0f1', edgecolor='#2c3e50', lw=2))
        
        # Draw arrows
        if i < len(boxes) - 1:
            ax.annotate("", xy=(boxes[i+1][1][0] - 0.08, y), xytext=(x + 0.08, y),
                        arrowprops=dict(arrowstyle="->", lw=2, color='#2c3e50'))

    plt.title("Figure 1: Temporal-Causal GraphRAG Pipeline", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure1_pipeline.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_actor_graph_figure():
    G = nx.DiGraph()
    
    # Generic graph (Ablation)
    G.add_edge("T1213", "T1228")
    G.add_edge("T1228", "T1375")
    G.add_edge("T1213", "T1375")
    
    # Turla graph
    G.add_edge("Turla::T1213", "Turla::T1228")
    G.add_edge("Turla::T1228", "Turla::T1375")
    
    # Lazarus graph
    G.add_edge("Lazarus::T1213", "Lazarus::T1428")
    G.add_edge("Lazarus::T1428", "Lazarus::T1375")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Generic
    generic_nodes = ["T1213", "T1228", "T1375"]
    pos1 = nx.spring_layout(G.subgraph(generic_nodes), seed=42)
    nx.draw(G.subgraph(generic_nodes), pos1, ax=ax1, with_labels=True, 
            node_color='#e74c3c', node_size=3000, font_size=10, font_weight='bold', 
            arrowsize=20, edge_color='#7f8c8d')
    ax1.set_title("A) Standard Technique Graph (Collapses Context)", fontsize=14, pad=10)

    # Plot Actor-Aware
    actor_nodes = [n for n in G.nodes() if "::" in n]
    pos2 = nx.spring_layout(G.subgraph(actor_nodes), seed=42)
    
    color_map = []
    for node in G.subgraph(actor_nodes).nodes():
        if "Turla" in node: color_map.append('#3498db')
        else: color_map.append('#2ecc71')

    nx.draw(G.subgraph(actor_nodes), pos2, ax=ax2, with_labels=True, 
            node_color=color_map, node_size=3000, font_size=10, font_weight='bold', 
            arrowsize=20, edge_color='#7f8c8d')
    ax2.set_title("B) Proposed Actor-Aware Graph (Preserves Context)", fontsize=14, pad=10)

    plt.suptitle("Figure 2: Architectural Novelty — Actor-Aware State Isolation", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure2_actor_graph.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_results_chart():
    models = ['Markov Chain', 'TransE', 'GAT / R-GCN', 'Temporal GNN', 'LLM-only\n(Qwen 7B)', 'RotatE\n(Proposed)']
    hits_at_1 = [1.72, 3.57, 3.85, 3.08, 3.33, 41.38]
    hits_at_3 = [3.45, 6.75, 10.77, 9.23, 3.33, 87.93]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, hits_at_1, width, label='Hits@1 (%)', color=COLORS[0])
    rects2 = ax.bar(x + width/2, hits_at_3, width, label='Hits@3 (%)', color=COLORS[2])

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3: Next-TTP Prediction Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend()

    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure3_results.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Figure 1 (Pipeline)...")
    generate_pipeline_figure()
    
    print("Generating Figure 2 (Actor Graph)...")
    generate_actor_graph_figure()
    
    print("Generating Figure 3 (Results Chart)...")
    generate_results_chart()
    
    print(f"All figures saved to {fig_dir}")
