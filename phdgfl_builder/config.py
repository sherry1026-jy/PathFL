# phdgfl_builder/config.py
# -*- coding: utf-8 -*-
import os
import torch

DEFAULT_CONFIG = {
    "input_root_base": "/home/vipuser/test/dependency_graphs_all_120",
    "output_root_base": "/home/vipuser/test/processed_dataset_v4_120",

    "datasets": ["Closure", "Time", "Chart", "Mockito", "Lang", "Math"],

    "save_mode": "per_graph",
    "min_nodes": 1,
    "label_expand_radius": 1,

    "codebert_model": "microsoft/codebert-base",
    "max_code_length": 128,
    "code_embed_dim": 768,
    "compile_hf_model": False,
    "use_amp": True,
    "empty_code_fallback": "/* EMPTY */",

    "edge_type_mapping": {"data_flow": 0, "call": 1, "cfg": 2, "return": 3, "entry": 4},
    "edge_type_weight_coef": {"data_flow": 1.20, "call": 0.80, "cfg": 1.20, "return": 0.80, "entry": 0.80},

    "batch_size": 32 if torch.cuda.is_available() else 8,
    "num_workers": 2 if torch.cuda.is_available() else max(1, (os.cpu_count() or 2) - 1),

    "log_level": "INFO",
    "export_index_csv": True,

    "merge_nodes_by_line": True,
    "edge_build_mode": "full",
    "feature_mode": "full",
}


def apply_cli_overrides(cfg: dict, args):
    if args.save_mode is not None:
        cfg["save_mode"] = args.save_mode
    if args.num_workers is not None:
        cfg["num_workers"] = int(args.num_workers)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.merge_nodes_by_line is not None:
        cfg["merge_nodes_by_line"] = bool(int(args.merge_nodes_by_line))
    if args.edge_build_mode is not None:
        cfg["edge_build_mode"] = args.edge_build_mode
    if args.feature_mode is not None:
        cfg["feature_mode"] = args.feature_mode
    if args.label_expand_radius is not None:
        cfg["label_expand_radius"] = int(args.label_expand_radius)
    if args.export_index_csv is not None:
        cfg["export_index_csv"] = bool(int(args.export_index_csv))
    if args.log_level is not None:
        cfg["log_level"] = args.log_level
