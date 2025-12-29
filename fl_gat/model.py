# fl_gat/model.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


def choose_hidden_dim(cfg: Dict, ds) -> int:
    """
    Choose hidden dim purely based on average #nodes of sampled graphs.
    """
    if not cfg.get("auto_hidden", False):
        return int(cfg["hidden"])

    max_samples_cfg = int(cfg.get("auto_hidden_sample", 100) or 0)
    if max_samples_cfg <= 0 or len(ds) == 0:
        return int(cfg["hidden"])

    max_samples = min(len(ds), max_samples_cfg)

    total_nodes = 0
    for i in range(max_samples):
        data, *_ = ds[i]
        total_nodes += int(data.num_nodes)

    avg_nodes = total_nodes / float(max_samples)

    if avg_nodes < 100:
        return 64
    elif avg_nodes < 200:
        return 128
    else:
        return 192


class GATNodeScorer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        code_dim: int = 768,
        hidden: int = 256,
        heads: int = 4,
        num_relations: int = 5,
        code_weight: float = 1.0,
        dropout: float = 0.2,
        code_at_front: bool = True,
    ):
        super().__init__()
        assert in_dim >= code_dim, "in_dim must be >= code_dim"
        self.code_dim = code_dim
        self.code_weight = float(code_weight)
        self.code_at_front = bool(code_at_front)

        self.num_dim = in_dim - code_dim

        self.rel_emb = nn.Embedding(num_relations, hidden)
        self.proj_in = nn.Linear(in_dim, hidden)

        self.gat1 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout)

        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def _split_features(self, x: torch.Tensor):
        if self.code_at_front:
            x_code = x[:, :self.code_dim]
            x_num = x[:, self.code_dim:]
        else:
            x_num = x[:, :self.num_dim]
            x_code = x[:, self.num_dim:]
        return x_num, x_code

    def forward(self, x, edge_index, edge_type=None, edge_weight=None):
        x_num, x_code = self._split_features(x)

        # Light normalization to reduce scale mismatch
        if x_num.numel() > 0:
            m = x_num.mean(dim=0, keepdim=True)
            s = x_num.std(dim=0, keepdim=True).clamp_min(1e-6)
            x_num = (x_num - m) / s

        if x_code.numel() > 0:
            x_code = F.normalize(x_code, p=2, dim=1)

        x = torch.cat([x_num, self.code_weight * x_code], dim=-1)

        h = torch.relu(self.proj_in(x))

        # Relation-aware message bias (dtype aligned for AMP)
        if edge_index.numel() > 0 and edge_type is not None:
            rel = self.rel_emb(edge_type).to(h.dtype)
            if edge_weight is not None and edge_weight.numel() == rel.size(0):
                rel = rel * edge_weight.to(h.dtype).unsqueeze(-1)
            msg = h[edge_index[0]] + rel
            h = h.index_add(0, edge_index[1], msg)

        h1 = self.gat1(h, edge_index)
        h = self.ln1(self.dropout(h1) + h)
        h2 = self.gat2(h, edge_index)
        h = self.ln2(self.dropout(h2) + h)
        score = self.out(h).squeeze(-1)
        return score
