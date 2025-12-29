# model_gat.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GATNodeScorer(nn.Module):
    def __init__(self, in_dim: int, code_dim: int = 768, hidden: int = 256, heads: int = 4,
                 num_relations: int = 5, code_weight: float = 3.0):
        super().__init__()
        assert in_dim >= code_dim, "in_dim 必须 >= code_dim"
        self.code_dim = code_dim
        self.code_weight = code_weight

        self.rel_emb = nn.Embedding(num_relations, hidden)
        self.proj_in = nn.Linear(in_dim, hidden)

        self.gat1 = GATConv(hidden, hidden // heads, heads=heads, dropout=0.2)
        self.gat2 = GATConv(hidden, hidden // heads, heads=heads, dropout=0.2)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_type=None, edge_weight=None):
        x_code = x[:, :self.code_dim]
        x_num = x[:, self.code_dim:]
        x = torch.cat([x_num, self.code_weight * x_code], dim=-1)

        h = torch.relu(self.proj_in(x))

        if edge_index.numel() > 0 and edge_type is not None:
            rel = self.rel_emb(edge_type)
            if edge_weight is not None and edge_weight.numel() == rel.size(0):
                rel = rel * edge_weight.unsqueeze(-1)
            msg = h[edge_index[0]] + rel
            h = h.index_add(0, edge_index[1], msg)

        h = self.gat1(h, edge_index)
        h = self.gat2(h, edge_index)
        score = self.out(h).squeeze(-1)
        return score
