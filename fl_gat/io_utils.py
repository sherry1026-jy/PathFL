# fl_gat/io_utils.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import logging
import random
import re
import copy
import importlib
import inspect
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
import torch.serialization as ts
from torch_geometric.data import Data

logger = logging.getLogger("fl_gat")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_list(x: Union[str, List[str]]) -> List[str]:
    if isinstance(x, (list, tuple)):
        return [str(p) for p in x]
    return [str(x)]


def extract_pid_from_path(p: Path) -> int:
    m = re.search(r"proj_(\d+)", str(p)) or re.search(r"proj_(\d+)", p.stem)
    return int(m.group(1)) if m else -1


def project_name_of_root(root: Union[str, Path]) -> str:
    return Path(root).name


def append_result_text(cfg: Dict, line: str):
    out_dir = Path(cfg.get("result_dir", "./results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / cfg.get("result_text_file", "results.txt")
    with txt_path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


# -------------------------
# Safe PyG deserialization
# -------------------------
def register_pyg_safe_globals():
    allow = []
    try:
        from torch_geometric.data import Data as _Data
        allow.append(_Data)
    except Exception:
        pass
    try:
        mod = importlib.import_module("torch_geometric.data.data")
        for name, obj in vars(mod).items():
            if inspect.isclass(obj):
                if (name.endswith("Attr") or name.endswith("Storage") or
                    name.endswith("Coll") or "Data" in name):
                    allow.append(obj)
    except Exception:
        pass

    seen, uniq = set(), []
    for c in allow:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    if uniq:
        ts.add_safe_globals(uniq)


register_pyg_safe_globals()


def safe_load(path: Path, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def make_pt_loader(cfg: Dict):
    """
    Create a (possibly cached) pt loader according to cfg.
    NOTE: cache size is fixed at creation time.
    """
    cache_enabled = bool(cfg.get("cache_graphs_in_memory", True))
    max_items = int(cfg.get("cache_max_items", 50000))

    def _load_pt_nocache(path_str: str):
        return safe_load(Path(path_str), map_location="cpu")

    if not cache_enabled:
        def load_pt(path_str: str):
            return _load_pt_nocache(path_str)
        return load_pt

    _cached = lru_cache(maxsize=max_items)(_load_pt_nocache)

    def load_pt(path_str: str):
        d0 = _cached(path_str)
        return copy.deepcopy(d0)  # avoid in-place mutation contaminating cache

    return load_pt


# -------------------------
# Edge types + subset filter
# -------------------------
EDGE_TYPE_ID = {
    "data_flow": 0,
    "call": 1,
    "cfg": 2,
    "return": 3,
    "entry": 4,
}


def apply_edge_subset(cfg: Dict, data: Data) -> Data:
    mode = (cfg.get("edge_subset_mode", "full") or "full").lower()
    if mode == "full":
        return data
    if not hasattr(data, "edge_type") or data.edge_index.numel() == 0:
        return data

    et = data.edge_type
    keep = torch.ones_like(et, dtype=torch.bool)

    if mode == "only_data":
        keep = (et == EDGE_TYPE_ID["data_flow"])
    elif mode == "only_cfg":
        keep = (et == EDGE_TYPE_ID["cfg"])
    elif mode == "only_call":
        keep = (et == EDGE_TYPE_ID["call"]) | (et == EDGE_TYPE_ID["return"]) | (et == EDGE_TYPE_ID["entry"])
    elif mode == "data_cfg":
        keep = (et == EDGE_TYPE_ID["data_flow"]) | (et == EDGE_TYPE_ID["cfg"])
    elif mode == "no_data":
        keep = (et != EDGE_TYPE_ID["data_flow"])
    elif mode == "no_cfg":
        keep = (et != EDGE_TYPE_ID["cfg"])
    elif mode == "no_call":
        keep = (et != EDGE_TYPE_ID["call"]) & (et != EDGE_TYPE_ID["return"])
    else:
        return data

    data.edge_index = data.edge_index[:, keep]
    data.edge_type = data.edge_type[keep]
    if hasattr(data, "edge_weight") and data.edge_weight.numel() == keep.numel():
        data.edge_weight = data.edge_weight[keep]

    return data
