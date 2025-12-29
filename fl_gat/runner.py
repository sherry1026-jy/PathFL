# fl_gat/runner.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import time
import hashlib
import logging
from typing import Dict, List, Tuple

import torch

from .dataset import GraphListDataset, index_by_bug, bug_has_positive
from .io_utils import set_seed, append_result_text
from .train_eval import (
    train_once,
    predict_one_bug,
    eval_one_bug,
    aggregate_counts,
)

logger = logging.getLogger("fl_gat")


def _bug_key(project: str, pid: int) -> str:
    return f"{project}:{pid}"


def _bucket(project: str, pid: int, k: int) -> int:
    h = int(hashlib.md5(_bug_key(project, pid).encode()).hexdigest(), 16)
    return h % k


def run_kfold(cfg: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"Device={device.type} | AMP={cfg.get('use_amp')} | cache={cfg.get('cache_graphs_in_memory')} | workers={cfg.get('num_workers')}"
    )

    ds = GraphListDataset(cfg, cfg["processed_roots"])
    bug_map = index_by_bug(ds)
    ks = (1, 3, 5, 10)
    k = int(cfg.get("k_folds", 4) or 4)

    epoch_sweep = cfg.get("epoch_sweep") or [int(cfg.get("epochs", 60))]
    epoch_sweep = sorted({int(e) for e in epoch_sweep if int(e) > 0})
    total_trials = len(epoch_sweep)

    for trial_id, max_ep in enumerate(epoch_sweep, start=1):
        logger.info("=" * 80)
        logger.info(f"K-fold (per-project) | max_epochs={max_ep}")

        set_seed(int(cfg["seed"]))
        per_project_summary: Dict[str, Tuple[Dict[str, int], Dict[str, float], int]] = {}
        overall_results: List[Dict[str, float]] = []

        for project, pid_map in bug_map.items():
            pids_pos = [pid for pid, idxs in pid_map.items() if bug_has_positive(ds, idxs)]
            if not pids_pos:
                logger.info(f"[Project {project}] No positive bugs, skipped.")
                continue

            logger.info(f"[Project {project}] bugs={len(pids_pos)} | K={k} | max_epochs={max_ep}")
            project_bug_results: List[Dict[str, float]] = []

            for fold in range(k):
                train_bug_indices: List[List[int]] = []
                test_pids: List[int] = []

                for pid in pids_pos:
                    if _bucket(project, pid, k) == fold:
                        test_pids.append(pid)
                    else:
                        train_bug_indices.append(pid_map[pid])

                logger.info(f"[Project {project}] Fold {fold+1}/{k} | train_bugs={len(train_bug_indices)} | test_bugs={len(test_pids)}")

                if not train_bug_indices or not test_pids:
                    logger.info(f"[Project {project}] Fold {fold+1}: empty train/test, skipped.")
                    continue

                t0 = time.time()
                model = train_once(
                    cfg, ds, train_bug_indices, device,
                    max_epochs=max_ep,
                    force_no_early_stop=True,  # fixed epochs for epoch_sweep fairness
                )
                train_time = time.time() - t0
                if model is None:
                    logger.info(f"[Project {project}] Fold {fold+1}: train_once returned None, skipped.")
                    continue

                for pid in test_pids:
                    bug_indices = pid_map[pid]
                    pred_scores, gold_ids = predict_one_bug(
                        cfg, model, ds, bug_indices, device,
                        merge_same_line=bool(cfg["eval_merge_same_line"]),
                        reduce=str(cfg["merge_reduce"]),
                    )
                    bug_res = eval_one_bug(cfg, pred_scores, gold_ids, ks=ks)
                    project_bug_results.append(bug_res)
                    overall_results.append(bug_res)

                logger.info(f"[Project {project}] Fold {fold+1} done | train {train_time:.1f}s | tested {len(test_pids)} bugs")

            hits_p, avg_mm_p, n_p = aggregate_counts(project_bug_results, ks=ks)
            if n_p > 0:
                per_project_summary[project] = (hits_p, avg_mm_p, n_p)

        hits_all, avg_mm_all, n_all = aggregate_counts(overall_results, ks=ks)

        logger.info("-" * 80)
        logger.info(f"Summary | max_epochs={max_ep}")

        candidate = {
            "protocol": "kfold",
            "k_folds": k,
            "max_epochs": max_ep,
            "dropout": cfg["dropout"],
            "w_topk": cfg["w_topk"],
            "lr": cfg["lr"],
            "heads": cfg["heads"],
            "line_pool": cfg["line_pool"],
            "edge_subset_mode": cfg.get("edge_subset_mode", "full"),
        }
        append_result_text(cfg, f"[tune] Trial {trial_id}/{total_trials} :: candidate = {candidate}")

        for project, (hits_p, avg_mm_p, n_p) in per_project_summary.items():
            tops_str = " ".join([f"Top@{k_}={hits_p[f'Top@{k_}']}/{n_p}" for k_ in ks])
            line = f"Project {project} (bugs={n_p}) :: {tops_str} | MAR={avg_mm_p['MAR']:.2f} MFR={avg_mm_p['MFR']:.2f}"
            logger.info("  " + line)
            append_result_text(cfg, line)

        if n_all > 0:
            tops_str = " ".join([f"Top@{k_}={hits_all[f'Top@{k_}']}/{n_all}" for k_ in ks])
            line = f"Overall ({n_all} bugs) :: {tops_str} | MAR={avg_mm_all['MAR']:.2f} MFR={avg_mm_all['MFR']:.2f}"
            logger.info("  " + line)
            append_result_text(cfg, line)
        else:
            logger.info("  Overall: no valid bugs.")
            append_result_text(cfg, "Overall: no valid bugs.")

        append_result_text(cfg, "")


def run_cross_project(cfg: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"[XProj] Device={device.type} | AMP={cfg.get('use_amp')} | cache={cfg.get('cache_graphs_in_memory')} | workers={cfg.get('num_workers')}"
    )

    ds = GraphListDataset(cfg, cfg["processed_roots"])
    bug_map = index_by_bug(ds)
    ks = (1, 3, 5, 10)

    epoch_sweep = cfg.get("epoch_sweep") or [int(cfg.get("epochs", 60))]
    epoch_sweep = sorted({int(e) for e in epoch_sweep if int(e) > 0})
    total_trials = len(epoch_sweep)
    all_projects = sorted(bug_map.keys())

    for trial_id, max_ep in enumerate(epoch_sweep, start=1):
        logger.info("=" * 80)
        logger.info(f"Cross-Project | max_epochs={max_ep}")

        set_seed(int(cfg["seed"]))

        per_project_summary: Dict[str, Tuple[Dict[str, int], Dict[str, float], int]] = {}
        overall_results: List[Dict[str, float]] = []

        for target_project in all_projects:
            pid_map_test = bug_map[target_project]
            test_pids = [pid for pid, idxs in pid_map_test.items() if bug_has_positive(ds, idxs)]

            train_bug_indices: List[List[int]] = []
            for pj, pid_map in bug_map.items():
                if pj == target_project:
                    continue
                for pid, idxs in pid_map.items():
                    if bug_has_positive(ds, idxs):
                        train_bug_indices.append(idxs)

            logger.info(f"[XProj] Target={target_project} | train_bugs={len(train_bug_indices)} | test_bugs={len(test_pids)} | max_epochs={max_ep}")

            if not train_bug_indices or not test_pids:
                logger.info(f"[XProj] Target={target_project}: empty train/test, skipped.")
                continue

            t0 = time.time()
            model = train_once(
                cfg, ds, train_bug_indices, device,
                max_epochs=max_ep,
                force_no_early_stop=True,
            )
            train_time = time.time() - t0
            if model is None:
                logger.info(f"[XProj] Target={target_project}: train_once returned None, skipped.")
                continue

            project_bug_results: List[Dict[str, float]] = []
            for pid in test_pids:
                bug_indices = pid_map_test[pid]
                pred_scores, gold_ids = predict_one_bug(
                    cfg, model, ds, bug_indices, device,
                    merge_same_line=bool(cfg["eval_merge_same_line"]),
                    reduce=str(cfg["merge_reduce"]),
                )
                bug_res = eval_one_bug(cfg, pred_scores, gold_ids, ks=ks)
                project_bug_results.append(bug_res)
                overall_results.append(bug_res)

            hits_p, avg_mm_p, n_p = aggregate_counts(project_bug_results, ks=ks)
            if n_p > 0:
                per_project_summary[target_project] = (hits_p, avg_mm_p, n_p)

            tops_str = " ".join([f"Top@{k_}={hits_p[f'Top@{k_}']}/{n_p}" for k_ in ks]) if n_p > 0 else "no-bugs"
            logger.info(f"[XProj] Target={target_project} done | bugs={n_p} | {tops_str} | MAR={avg_mm_p['MAR']:.2f} MFR={avg_mm_p['MFR']:.2f} | train {train_time:.1f}s")

        hits_all, avg_mm_all, n_all = aggregate_counts(overall_results, ks=ks)

        candidate = {
            "protocol": "cross_project",
            "max_epochs": max_ep,
            "dropout": cfg["dropout"],
            "w_topk": cfg["w_topk"],
            "lr": cfg["lr"],
            "heads": cfg["heads"],
            "line_pool": cfg["line_pool"],
            "edge_subset_mode": cfg.get("edge_subset_mode", "full"),
        }
        append_result_text(cfg, f"[tune] XProj Trial {trial_id}/{total_trials} :: candidate = {candidate}")

        for project, (hits_p, avg_mm_p, n_p) in per_project_summary.items():
            tops_str = " ".join([f"Top@{k_}={hits_p[f'Top@{k_}']}/{n_p}" for k_ in ks])
            line = f"[XProj] Target {project} (bugs={n_p}) :: {tops_str} | MAR={avg_mm_p['MAR']:.2f} MFR={avg_mm_p['MFR']:.2f}"
            append_result_text(cfg, line)

        if n_all > 0:
            tops_str = " ".join([f"Top@{k_}={hits_all[f'Top@{k_}']}/{n_all}" for k_ in ks])
            line = f"[XProj] Overall ({n_all} bugs) :: {tops_str} | MAR={avg_mm_all['MAR']:.2f} MFR={avg_mm_all['MFR']:.2f}"
            append_result_text(cfg, line)
            logger.info(line)
        else:
            append_result_text(cfg, "[XProj] Overall: no valid bugs.")
            logger.info("[XProj] Overall: no valid bugs.")

        append_result_text(cfg, "")
