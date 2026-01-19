# pathfl_builder/worker.py
# -*- coding: utf-8 -*-
import time
import torch
from pathlib import Path
from typing import Tuple, Dict, Any

from .graph_process import process_graph
from .encoder import CodeBERTEncoder

def process_one_project(args: Tuple[str, int, str, dict, str]) -> Dict[str, Any]:
    dataset_name, pid, in_dir_str, cfg, log_level = args
    in_dir = Path(in_dir_str)

    stats = {"dataset": dataset_name, "pid": pid, "graphs": 0, "nodes": 0, "edges": 0, "duration": 0.0, "outputs": []}
    t0 = time.time()

    # 每个进程内初始化一次（幂等）
    CodeBERTEncoder.init(cfg, logger=None)

    json_files = sorted(in_dir.glob("*.json"))
    graphs = []

    for json_path in json_files:
        g = process_graph(json_path, cfg, logger=None)
        if g is None:
            continue
        stats["graphs"] += 1
        stats["nodes"] += int(g.num_nodes)
        stats["edges"] += int(g.num_edges)

        if cfg["save_mode"] == "per_graph":
            out_dir = Path(cfg["output_root_base"]) / dataset_name / f"proj_{pid}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{json_path.stem}.pt"
            torch.save(g, out_path)
            stats["outputs"].append(str(out_path))
        else:
            graphs.append(g)

    if cfg["save_mode"] == "per_project" and graphs:
        out_dir = Path(cfg["output_root_base"]) / dataset_name / f"proj_{pid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"proj_{pid}.pt"
        torch.save(graphs, out_path)
        stats["outputs"].append(str(out_path))

    stats["duration"] = time.time() - t0
    return stats
