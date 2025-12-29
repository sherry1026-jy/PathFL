# fl_gat/config.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import copy
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_CONFIG: Dict = {
    # Dataset roots (will be filled by CLI discovery)
    "processed_roots": [],       # list of project roots that contain proj_*/*.pt
    "raw_json_roots": [],        # list of project roots that contain <pid>/<identifier>.json
    "save_mode": "per_graph",

    # Evaluation protocol:
    #   - "kfold": within-project K-fold (default)
    #   - "loo"  : within-project leave-one-out
    #   - "xproj": cross-project (leave-one-project-out)
    "eval_protocol": "kfold",
    "k_folds": 4,

    # Training
    "seed": 42,
    "epochs": 60,
    "epoch_sweep": None,         # e.g., [20, 30, 40] -> re-train per value
    "batch_size": 1,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
    "train_only_positive_graphs": True,

    "train_scope": "file",

    # Early stopping (optional)
    "early_stop": False,
    "early_stop_gap": 3,         # compare loss(ep) vs loss(ep-gap)

    # Model / GAT
    "hidden": 64,
    "auto_hidden": True,
    "auto_hidden_sample": 100,   # sample up to N graphs to estimate avg nodes
    "heads": 2,
    "num_relations": 5,
    "code_dim": 768,
    "code_weight": 0.80,
    "dropout": 0.1,

    # Loss weights
    "rank_neg_per_pos": 6,
    "rank_margin": 0.5,
    "w_focal": 0.1,
    "w_pair": 0.6,
    "w_list": 0.1,
    "w_mar": 2.0,
    "w_mfr": 1.0,
    "w_topk": 6.0,
    "topk_K": 1,
    "topk_margin": 0.0,
    "tau_rank": 0.6,
    "tau_mfr": 0.4,

    # Evaluation: merge nodes into line-level items
    "eval_merge_same_line": True,
    "merge_reduce": "max",       # "max" or "mean"

    # Metrics options
    "mar_mode": "all_positives", 
    "topk_mode": "bug",          
    "rank_scope": "global",      

    "train_noise_strategy": "none",

    # Global evaluation enhancements (rank_scope="global")
    "global_use_file_norm": True,
    "global_norm_type": "rank",  # "zscore" / "minmax" / "rank"
    "global_top_m_per_file": 50, # <=0 means no truncation

    # Line-level MIL training
    "use_line_level_train": True,
    "line_pool": "softmax",      # "softmax" / "max" / "mean"
    "line_pool_tau": 1.0,
    "line_loss_weight": 0.8,
    "node_loss_weight": 0.2,

    # AMP / performance
    "use_amp": True,
    "amp_dtype": "bf16",         # "bf16" or "fp16"
    "num_workers": 4,
    "pin_memory": True,
    "prefetch_factor": 2,
    "cache_graphs_in_memory": True,
    "cache_max_items": 50000,
    "use_tf32": True,

    # Feature layout
    "code_at_front": True,       # your dataset uses [code(768) | numeric_features]

    # Result output
    "result_dir": "./results",
    "result_text_file": "results.txt",

    # Edge subset control (ablation)
    "edge_subset_mode": "full",  # full/only_data/only_cfg/only_call/data_cfg/no_data/no_cfg/no_call
}


def make_config(overrides: Optional[Dict] = None) -> Dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


def _looks_like_project_root(p: Path) -> bool:
    # A project root should directly contain proj_* dirs with *.pt inside.
    if not p.exists() or not p.is_dir():
        return False
    for d in p.glob("proj_*"):
        if d.is_dir() and any(d.glob("*.pt")):
            return True
    return False


def discover_processed_roots(processed_root: str) -> List[str]:
    """
    Accept either:
      - a single project root (contains proj_*/.pt),
      - or a dataset root that contains multiple project subdirs (Time/Mockito/...).
    """
    base = Path(processed_root)
    if _looks_like_project_root(base):
        return [str(base)]

    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"processed_root not found: {processed_root}")

    candidates = []
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and _looks_like_project_root(sub):
            candidates.append(str(sub))

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find any project roots under: {processed_root}. "
            f"Expected folders like <ProjectName>/proj_*/xxx.pt"
        )
    return candidates


def discover_raw_json_roots(raw_root: str) -> List[str]:
    """
    Raw JSON root can be:
      - a single project root (contains <pid>/<identifier>.json),
      - or a dataset root containing multiple project subdirs (Time/Mockito/...).
    """
    base = Path(raw_root)
    if base.exists() and base.is_dir():
        # If base contains numeric dirs directly, treat it as a single project root.
        if any(d.is_dir() and d.name.isdigit() for d in base.iterdir()):
            return [str(base)]

        candidates = []
        for sub in sorted(base.iterdir()):
            if sub.is_dir() and any(d.is_dir() and d.name.isdigit() for d in sub.iterdir()):
                candidates.append(str(sub))
        if candidates:
            return candidates

    raise FileNotFoundError(
        f"Cannot discover raw json project roots under: {raw_root}. "
        f"Expected <ProjectName>/<pid>/<identifier>.json"
    )
