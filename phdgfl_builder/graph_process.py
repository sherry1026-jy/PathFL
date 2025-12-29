# phdgfl_builder/graph_process.py
# -*- coding: utf-8 -*-
import json
import numpy as np
import torch
from pathlib import Path
from torch_geometric.data import Data

from .merge import merge_nodes_by_line
from .edges import build_edges_and_degrees
from .features import build_node_features
from .encoder import CodeBERTEncoder

def process_graph(json_path: Path, cfg: dict, logger=None):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            js = json.load(f)

        nodes = js.get("nodes", {})
        edges = js.get("edges", {})
        if len(nodes) < cfg["min_nodes"]:
            return None

        if cfg.get("merge_nodes_by_line", False):
            nodes, edges = merge_nodes_by_line(nodes, edges)

        mode = cfg.get("edge_build_mode", "full")
        if mode == "none":
            N = len(nodes)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type  = torch.empty((0,), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float32)
            deg_in  = np.zeros(N, dtype=np.float32)
            deg_out = np.zeros(N, dtype=np.float32)
        else:
            edge_index, edge_type, edge_weight, deg_in, deg_out = build_edges_and_degrees(nodes, edges, cfg)

        feature_mode = cfg.get("feature_mode", "full")
        if feature_mode == "no_code":
            N = len(nodes)
            code_embeds = np.zeros((N, cfg["code_embed_dim"]), dtype=np.float32)
        else:
            CodeBERTEncoder.init(cfg, logger=logger)
            code_texts = [(info.get("code_content") or info.get("code") or "") for info in nodes.values()]
            bs = cfg["batch_size"]
            chunks = []
            for i in range(0, len(code_texts), bs):
                chunks.append(CodeBERTEncoder.encode_batch(code_texts[i:i+bs], cfg))
            code_embeds = np.concatenate(chunks, axis=0)

        if feature_mode == "code_only":
            x_np = code_embeds
        else:
            x_np = build_node_features(nodes, code_embeds, deg_in, deg_out)

        x = torch.as_tensor(x_np, dtype=torch.float32)

        radius = int(cfg.get("label_expand_radius", 0))
        node_lines = []
        for _, info in nodes.items():
            ln = info.get("line", -1)
            try:
                ln = int(ln) if ln is not None else -1
            except Exception:
                ln = -1
            node_lines.append(ln)

        is_def_list = [int(info.get("is_defect", 0)) for info in nodes.values()]
        seed_lines = set(int(node_lines[i]) for i, v in enumerate(is_def_list) if v == 1 and node_lines[i] >= 0)

        expanded_lines = set(seed_lines)
        if radius > 0 and seed_lines:
            for ln in list(seed_lines):
                for d in range(-radius, radius + 1):
                    expanded_lines.add(ln + d)

        y_list = []
        for i, ln in enumerate(node_lines):
            if ln >= 0 and expanded_lines:
                y_list.append(1 if ln in expanded_lines else 0)
            else:
                y_list.append(is_def_list[i])
        y = torch.as_tensor(y_list, dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_weight=edge_weight,
            y=y,
            filename=js.get("filename", json_path.name),
            identifier=json_path.stem,
        )
        data.node_line = torch.as_tensor(node_lines, dtype=torch.long)
        fs = [1 if bool(info.get("from_static", False)) else 0 for _, info in nodes.items()]
        data.from_static = torch.as_tensor(fs, dtype=torch.long)

        return data

    except Exception as e:
        if logger:
            logger.error(f"处理失败 {json_path}: {e}", exc_info=True)
        return None
