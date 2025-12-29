# fl_gat/dataset.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .io_utils import (
    ensure_list,
    extract_pid_from_path,
    project_name_of_root,
    apply_edge_subset,
    make_pt_loader,
)


class GraphListDataset(Dataset):
    """
    Each sample is one .pt graph (one Java file).
    Directory layout:
      <project_root>/proj_<pid>/*.pt
    """
    def __init__(self, cfg: Dict, processed_roots: Union[str, List[str]]):
        super().__init__()
        self.cfg = cfg
        self.items: List[Tuple[str, int, str, Path]] = []
        self.load_pt = make_pt_loader(cfg)

        roots = [Path(r) for r in ensure_list(processed_roots)]
        assert cfg.get("save_mode", "per_graph") == "per_graph", "Only per_graph is supported."

        for root in roots:
            project = project_name_of_root(root)
            for proj_dir in sorted(root.glob("proj_*")):
                if not proj_dir.is_dir():
                    continue
                pid = extract_pid_from_path(proj_dir)
                for pt_path in sorted(proj_dir.glob("*.pt")):
                    identifier = pt_path.stem
                    self.items.append((project, pid, identifier, pt_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Tuple[Data, str, int, str]:
        project, pid, identifier, pt_path = self.items[idx]
        data = self.load_pt(str(pt_path))
        if isinstance(data, list):
            data = data[0]

        if not hasattr(data, "edge_type"):
            data.edge_type = torch.zeros(data.edge_index.size(1), dtype=torch.long)
        if not hasattr(data, "edge_weight"):
            data.edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float)

        data = apply_edge_subset(self.cfg, data)

        data.identifier = getattr(data, "identifier", identifier)
        data.filename = getattr(data, "filename", identifier)
        return data, project, pid, identifier


def index_by_bug(ds: GraphListDataset) -> Dict[str, Dict[int, List[int]]]:
    """
    Return: {project: {pid: [sample_idx, ...]}}
    """
    bug_map: Dict[str, Dict[int, List[int]]] = {}
    for i, (project, pid, identifier, path) in enumerate(ds.items):
        bug_map.setdefault(project, {}).setdefault(pid, []).append(i)
    return bug_map


def bug_has_positive(ds: GraphListDataset, bug_graph_indices: List[int]) -> bool:
    for gidx in bug_graph_indices:
        data, *_ = ds[gidx]
        if int(data.y.sum().item()) > 0:
            return True
    return False
