# phdgfl_builder/builder.py
# -*- coding: utf-8 -*-
import os
import csv
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

from .log_utils import setup_logger
from .monitor import PerformanceMonitor
from .discover import discover_tasks
from .worker import process_one_project
from .encoder import CodeBERTEncoder


# ✅ 关键：必须是“模块顶层函数”，spawn 才能 pickle
def pool_initializer(cfg: dict):
    CodeBERTEncoder.init(cfg, logger=None)


def run_build(cfg: dict, mp_start_method: str = "spawn"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    logger = setup_logger("build-all", cfg.get("log_level", "INFO"))

    in_root = Path(cfg["input_root_base"])
    out_root = Path(cfg["output_root_base"])
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks(in_root, cfg)
    if not tasks:
        logger.error(f"未发现任何可处理任务：{in_root}")
        return

    worker_args = [(ds, pid, str(pdir), cfg, cfg.get("log_level", "INFO")) for (ds, pid, pdir) in tasks]

    monitor = PerformanceMonitor()

    try:
        mp.set_start_method(mp_start_method, force=True)
    except RuntimeError:
        pass

    ctx = mp.get_context(mp_start_method)

    all_stats: List[Dict] = []
    nw = int(cfg["num_workers"])

    logger.info(f"输入: {in_root}")
    logger.info(f"输出: {out_root}")
    logger.info(
        f"任务数(项目): {len(worker_args)} | workers={nw} | save_mode={cfg['save_mode']} | "
        f"feature_mode={cfg['feature_mode']} | edge_mode={cfg['edge_build_mode']}"
    )

    # ✅ 注意这里：initializer 用顶层函数 + initargs 传 cfg
    with ctx.Pool(processes=nw, initializer=pool_initializer, initargs=(cfg,)) as pool:
        for st in tqdm(pool.imap_unordered(process_one_project, worker_args),
                       total=len(worker_args), desc="🚀 构建图数据", unit="proj"):
            monitor.update(st["dataset"], st["pid"], st["graphs"], st["nodes"], st["edges"], st["duration"])
            all_stats.append(st)

    monitor.report()
    print(f"\n数据已保存到: {cfg['output_root_base']}")
    print(f"保存模式: {cfg['save_mode']}")

    if cfg.get("export_index_csv", True):
        try:
            index_rows = []
            for st in all_stats:
                for p in st.get("outputs", []):
                    index_rows.append({"dataset": st["dataset"], "pid": st["pid"], "pt_path": p})

            idx_csv = out_root / "index.csv"
            with open(idx_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["dataset", "pid", "pt_path"])
                w.writeheader()
                w.writerows(index_rows)

            logger.info(f"索引表已导出：{idx_csv}（{len(index_rows)} 条）")
        except Exception as e:
            logger.warning(f"导出 index.csv 失败：{e}")
