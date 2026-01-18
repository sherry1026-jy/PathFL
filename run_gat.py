# run_fl_gat.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import logging
import torch

from fl_gat.config import make_config, discover_processed_roots, discover_raw_json_roots
from fl_gat.runner import run_kfold, run_cross_project
from fl_gat.io_utils import set_seed


def build_argparser():
    p = argparse.ArgumentParser(
        description="K-fold / Cross-project Fault Localization with GAT (pt graphs)"
    )
    p.add_argument(
        "processed_root",
        type=str,
        help=("Path to processed .pt graphs root. "
              "Can be a project root (contains proj_*/.pt) or a dataset root containing multiple projects."),
    )
    p.add_argument(
        "--raw-json-root",
        type=str,
        default=None,
        help=("Optional: raw json root for node->line fallback. "
              "Can be a project root (<pid>/<identifier>.json) or a dataset root containing multiple projects."),
    )
    p.add_argument(
        "--protocol",
        type=str,
        default="kfold",
        choices=["kfold", "xproj"],
        help="Evaluation protocol.",
    )
    p.add_argument("--k-folds", type=int, default=4, help="K for within-project K-fold.")
    p.add_argument("--epochs", type=int, default=60, help="Default max epochs if no epoch-sweep is given.")
    p.add_argument(
        "--epoch-sweep",
        type=int,
        nargs="*",
        default=None,
        help="If provided, re-train for each value (e.g., --epoch-sweep 20 30 40).",
    )
    p.add_argument("--result-dir", type=str, default="./results", help="Directory for result text file.")
    p.add_argument("--result-file", type=str, default="results.txt", help="Result text filename.")
    p.add_argument(
        "--edge-subset-mode",
        type=str,
        default="full",
        choices=["full", "only_data", "only_cfg", "only_call", "data_cfg", "no_data", "no_cfg", "no_call"],
        help="Edge ablation mode.",
    )
    return p


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    args = build_argparser().parse_args()

    processed_roots = discover_processed_roots(args.processed_root)
    raw_roots = []
    if args.raw_json_root:
        raw_roots = discover_raw_json_roots(args.raw_json_root)

    cfg = make_config({
        "processed_roots": processed_roots,
        "raw_json_roots": raw_roots,
        "eval_protocol": args.protocol,
        "k_folds": args.k_folds,
        "epochs": args.epochs,
        "epoch_sweep": args.epoch_sweep,
        "result_dir": args.result_dir,
        "result_text_file": args.result_file,
        "edge_subset_mode": args.edge_subset_mode,
    })

    # TF32 toggles (safe)
    if cfg.get("use_tf32", True):
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    set_seed(int(cfg["seed"]))

    if cfg["eval_protocol"] == "kfold":
        run_kfold(cfg)
    elif cfg["eval_protocol"] in ("xproj", "cross_project"):
        run_cross_project(cfg)
    else:
        run_kfold(cfg)


if __name__ == "__main__":
    main()
