# build_dataset.py
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

from phdgfl_builder.config import DEFAULT_CONFIG, apply_cli_overrides
from phdgfl_builder.builder import run_build


def parse_args():
    p = argparse.ArgumentParser(
        description="Build PyG graphs (.pt) from HPDG JSONs (Defects4J style folders supported)."
    )
    p.add_argument("input_root", type=str, help="Input folder: root/dataset/pid or dataset/pid or pid/*.json or folder with *.json")
    p.add_argument("--output_root", type=str, default=None, help="Output root folder (default: <input_root>_processed)")
    p.add_argument("--datasets", type=str, default=None,
                   help="Comma-separated dataset names, e.g., Closure,Time,Chart,Mockito,Lang,Math. (only works when input_root is a base root)")
    p.add_argument("--save_mode", type=str, default=None, choices=["per_graph", "per_project"])
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--merge_nodes_by_line", type=int, default=None, choices=[0, 1])
    p.add_argument("--edge_build_mode", type=str, default=None, choices=["full", "none"])
    p.add_argument("--feature_mode", type=str, default=None, choices=["full", "no_code", "code_only"])
    p.add_argument("--label_expand_radius", type=int, default=None)
    p.add_argument("--export_index_csv", type=int, default=None, choices=[0, 1])
    p.add_argument("--mp_start_method", type=str, default="spawn", choices=["spawn", "fork", "forkserver"])
    p.add_argument("--log_level", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(DEFAULT_CONFIG)

    in_root = Path(args.input_root).resolve()
    out_root = Path(args.output_root).resolve() if args.output_root else Path(str(in_root) + "_processed").resolve()

    cfg["input_root_base"] = str(in_root)
    cfg["output_root_base"] = str(out_root)

    # datasets 仅当 input_root 是“总根目录”时才有意义；否则 builder 会按输入路径自动判定范围
    if args.datasets is not None:
        ds = [x.strip() for x in args.datasets.split(",") if x.strip()]
        cfg["datasets"] = ds

    apply_cli_overrides(cfg, args)

    run_build(cfg, mp_start_method=args.mp_start_method)


if __name__ == "__main__":
    main()
