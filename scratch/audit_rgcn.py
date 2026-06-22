import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

report_dir = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d"

class SimpleRGCN(torch.nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_channels):
        super().__init__()
        self.node_emb = torch.nn.Embedding(num_nodes, hidden_channels)
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)

    def forward(self, edge_index, edge_type):
        x = self.node_emb.weight
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type)
        return x
        
    def decode(self, z, edge_label_index, edge_type):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

def audit_rgcn():
    triples_df = pd.read_csv(os.path.join(base_dir, "scratch", "neo4j_triples.tsv"), sep='\t')
    
    entities = pd.concat([triples_df['head'], triples_df['tail']]).unique()
    entity_to_id = {ent: i for i, ent in enumerate(entities)}
    id_to_entity = {i: ent for ent, i in entity_to_id.items()}
    
    relations = triples_df['relation'].unique()
    relation_to_id = {rel: i for i, rel in enumerate(relations)}
    
    src = [entity_to_id[h] for h in triples_df['head']]
    dst = [entity_to_id[t] for t in triples_df['tail']]
    edge_type = [relation_to_id[r] for r in triples_df['relation']]
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
    
    num_nodes = len(entities)
    num_relations = len(relations)
    hidden_channels = 64
    
    model = SimpleRGCN(num_nodes, num_relations, hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Split train and validation from the graph edges for link prediction
    indices = torch.randperm(edge_index.size(1))
    train_size = int(0.8 * edge_index.size(1))
    
    train_edge_index = edge_index[:, indices[:train_size]]
    train_edge_type = edge_type_tensor[indices[:train_size]]
    val_edge_index = edge_index[:, indices[train_size:]]
    val_edge_type = edge_type_tensor[indices[train_size:]]
    
    train_losses = []
    val_losses = []
    
    model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        z = model(train_edge_index, train_edge_type)
        
        pos_out = model.decode(z, train_edge_index, train_edge_type)
        pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        
        neg_edge_index = torch.randint(0, num_nodes, train_edge_index.size(), dtype=torch.long)
        neg_out = model.decode(z, neg_edge_index, train_edge_type)
        neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        with torch.no_grad():
            z_val = model(val_edge_index, val_edge_type)
            val_pos_out = model.decode(z_val, val_edge_index, val_edge_type)
            val_pos_loss = F.binary_cross_entropy_with_logits(val_pos_out, torch.ones_like(val_pos_out))
            val_neg_edge_index = torch.randint(0, num_nodes, val_edge_index.size(), dtype=torch.long)
            val_neg_out = model.decode(z_val, val_neg_edge_index, val_edge_type)
            val_neg_loss = F.binary_cross_entropy_with_logits(val_neg_out, torch.zeros_like(val_neg_out))
            val_loss = val_pos_loss + val_neg_loss
            val_losses.append(val_loss.item())
            
    test_df = pd.read_csv(os.path.join(report_dir, "benchmark_test.csv"))
    
    all_techs = [e for e in entities if e.startswith("T1")]
    tech_ids = [entity_to_id[t] for t in all_techs]
    
    audit_lines = ["# R-GCN AUDIT REPORT\n"]
    audit_lines.append("## Loss Curves")
    audit_lines.append("```text")
    audit_lines.append("Epoch\tTrain Loss\tVal Loss")
    for i in range(0, 100, 10):
        audit_lines.append(f"{i+1}\t{train_losses[i]:.4f}\t{val_losses[i]:.4f}")
    audit_lines.append(f"100\t{train_losses[-1]:.4f}\t{val_losses[-1]:.4f}")
    audit_lines.append("```\n")
    
    model.eval()
    with torch.no_grad():
        z = model(edge_index, edge_type_tensor)
        
    audit_lines.append("## Top-5 Predictions (50 Samples)")
    for idx, row in test_df.iterrows():
        if idx >= 50:
            break
            
        curr_t = row['current_technique']
        actual_t = row['actual_next_technique']
        
        audit_lines.append(f"### Sample {idx+1}")
        audit_lines.append(f"**Current Technique:** {curr_t}")
        audit_lines.append(f"**Actual Next Technique:** {actual_t}\n")
        
        parents = triples_df[(triples_df['tail'] == curr_t)]['head'].unique()
        parent_ids = [entity_to_id.get(p) for p in parents if p in entity_to_id]
        
        if not parent_ids:
            audit_lines.append("**Predictions:** None (No parents)\n---\n")
            continue
            
        tech_scores = {t: 0.0 for t in all_techs}
        
        for pid in parent_ids:
            src_emb = z[pid].unsqueeze(0)
            dst_emb = z[tech_ids]
            scores = (src_emb * dst_emb).sum(dim=-1).numpy()
            
            for t_name, score in zip(all_techs, scores):
                tech_scores[t_name] += float(score)
                
        ranked = sorted(tech_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = ranked[:5]
        
        audit_lines.append("**Predictions:**")
        for i, (p, score) in enumerate(top_k):
            audit_lines.append(f"{i+1}. {p} (Score: {score:.4f})")
        audit_lines.append(f"**Ground Truth Score:** {tech_scores.get(actual_t, 0.0):.4f}\n---\n")
        
    with open(os.path.join(report_dir, "RGCN_AUDIT_REPORT.md"), "w") as f:
        f.write("\n".join(audit_lines))
        
    print("Task 3 (R-GCN Audit) complete.")

if __name__ == "__main__":
    audit_rgcn()
