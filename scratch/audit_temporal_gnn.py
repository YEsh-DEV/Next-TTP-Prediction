import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

report_dir = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d"

class TemporalGNN(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_channels):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_channels)
        self.rgcn = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)
        self.fc = nn.Linear(hidden_channels, num_nodes)

    def forward(self, edge_index, edge_type, seq_nodes):
        x = self.node_emb.weight
        node_embeddings = self.rgcn(x, edge_index, edge_type).relu()
        seq_embs = node_embeddings[seq_nodes]
        out, _ = self.gru(seq_embs)
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        return logits

def audit_temporal_gnn():
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
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    
    num_nodes = len(entities)
    num_relations = len(relations)
    hidden_channels = 64
    
    model = TemporalGNN(num_nodes, num_relations, hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    train_df = pd.read_csv(os.path.join(report_dir, "benchmark_train.csv"))
    
    train_seqs = []
    train_targets = []
    
    for _, row in train_df.iterrows():
        c = row['current_technique']
        t = row['actual_next_technique']
        if c in entity_to_id and t in entity_to_id:
            train_seqs.append([entity_to_id[c]])
            train_targets.append(entity_to_id[t])
            
    # Split train and validation (80% / 20% of the training split)
    indices = torch.randperm(len(train_seqs))
    split_idx = int(0.8 * len(train_seqs))
    
    tr_seqs = torch.tensor([train_seqs[i] for i in indices[:split_idx]], dtype=torch.long)
    tr_targs = torch.tensor([train_targets[i] for i in indices[:split_idx]], dtype=torch.long)
    
    val_seqs = torch.tensor([train_seqs[i] for i in indices[split_idx:]], dtype=torch.long)
    val_targs = torch.tensor([train_targets[i] for i in indices[split_idx:]], dtype=torch.long)
    
    train_losses = []
    val_losses = []
    
    model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        logits = model(edge_index, edge_type, tr_seqs)
        loss = criterion(logits, tr_targs)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        with torch.no_grad():
            val_logits = model(edge_index, edge_type, val_seqs)
            val_loss = criterion(val_logits, val_targs)
            val_losses.append(val_loss.item())
            
    audit_lines = ["# TEMPORAL GNN AUDIT REPORT\n"]
    audit_lines.append("## Loss Curves")
    audit_lines.append("```text")
    audit_lines.append("Epoch\tTrain Loss\tVal Loss")
    for i in range(0, 100, 10):
        audit_lines.append(f"{i+1}\t{train_losses[i]:.4f}\t{val_losses[i]:.4f}")
    audit_lines.append(f"100\t{train_losses[-1]:.4f}\t{val_losses[-1]:.4f}")
    audit_lines.append("```\n")
    
    model.eval()
    test_df = pd.read_csv(os.path.join(report_dir, "benchmark_test.csv"))
    all_techs = [e for e in entities if e.startswith("T1")]
    
    audit_lines.append("## Test Predictions")
    with torch.no_grad():
        for idx, row in test_df.iterrows():
            curr_t = row['current_technique']
            actual_t = row['actual_next_technique']
            
            audit_lines.append(f"### Sample {idx+1}")
            audit_lines.append(f"**Current Technique:** {curr_t}")
            audit_lines.append(f"**Actual Next Technique:** {actual_t}\n")
            
            if curr_t not in entity_to_id:
                audit_lines.append("**Predictions:** None (Current state unseen in graph)\n---\n")
                continue
                
            seq = torch.tensor([[entity_to_id[curr_t]]], dtype=torch.long)
            logits = model(edge_index, edge_type, seq).squeeze()
            
            tech_scores = {t: logits[entity_to_id[t]].item() for t in all_techs if t in entity_to_id}
            ranked = sorted(tech_scores.items(), key=lambda x: x[1], reverse=True)
            top_k = ranked[:5]
            
            audit_lines.append("**Predictions:**")
            for i, (p, score) in enumerate(top_k):
                audit_lines.append(f"{i+1}. {p} (Logit: {score:.4f})")
            audit_lines.append(f"**Ground Truth Logit:** {tech_scores.get(actual_t, 0.0):.4f}\n---\n")
            
    audit_lines.append("## Verdict")
    audit_lines.append("Did the model genuinely fail or was evaluation buggy?")
    audit_lines.append("If Val Loss stagnates or rises while Train Loss drops to ~0.0, the model genuinely failed due to **Dataset Collapse** (overfitting to train sequence paths but unable to generalize to test paths).")
    
    with open(os.path.join(report_dir, "TEMPORAL_GNN_AUDIT_REPORT.md"), "w") as f:
        f.write("\n".join(audit_lines))
        
    print("Task 4 (Temporal GNN Audit) complete.")

if __name__ == "__main__":
    audit_temporal_gnn()
