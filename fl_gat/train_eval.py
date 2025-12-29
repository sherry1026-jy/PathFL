# fl_gat/train_eval.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import time
import copy
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .model import GATNodeScorer, choose_hidden_dim
from .dataset import GraphListDataset, bug_has_positive
from .io_utils import append_result_text

logger = logging.getLogger("fl_gat")


# -------------------------
# Losses
# -------------------------
def focal_loss_with_logits(logits, targets, alpha=0.85, gamma=2.0):
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    w = alpha * targets + (1 - alpha) * (1 - targets)
    loss = -w * (1 - pt).pow(gamma) * (
        targets * torch.log(p + 1e-8) + (1 - targets) * torch.log(1 - p + 1e-8)
    )
    return loss.mean()


def pairwise_rank_loss(scores, y, neg_per_pos=4, margin=0.1):
    pos_idx = (y == 1).nonzero(as_tuple=True)[0]
    neg_idx = (y == 0).nonzero(as_tuple=True)[0]
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return scores.sum() * 0.0

    num_pairs = min(pos_idx.numel() * neg_per_pos, neg_idx.numel())
    sel_neg = neg_idx[torch.randint(low=0, high=neg_idx.numel(), size=(num_pairs,), device=y.device)]
    sel_pos = pos_idx.repeat_interleave(neg_per_pos)[:num_pairs]
    s_pos, s_neg = scores[sel_pos], scores[sel_neg]
    return F.relu(margin - (s_pos - s_neg)).mean()


def graph_listwise_loss(scores, y, temperature=0.7, label_smoothing=0.05):
    P = int(y.sum().item())
    if P == 0:
        return scores.sum() * 0.0
    q = torch.zeros_like(scores)
    q[y == 1] = 1.0 / P
    if label_smoothing > 0:
        eps = float(label_smoothing)
        q = (1 - eps) * q + eps * (1.0 / len(scores))
    log_p = F.log_softmax(scores / float(temperature), dim=0)
    return -(q * log_p).sum()


def mar_surrogate_loss(scores: torch.Tensor, y: torch.Tensor, tau: float = 0.6):
    pos = scores[y == 1]
    neg = scores[y == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.sum() * 0.0
    diff = (neg.unsqueeze(1) - pos.unsqueeze(0)) / float(tau)
    soft_rank = 1.0 + torch.sigmoid(diff).sum(dim=0)
    return soft_rank.mean()


def mfr_surrogate_loss(scores: torch.Tensor, y: torch.Tensor, tau_softmin: float = 0.4, tau_rank: float = 0.6):
    pos = scores[y == 1]
    neg = scores[y == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.sum() * 0.0
    diff = (neg.unsqueeze(1) - pos.unsqueeze(0)) / float(tau_rank)
    soft_rank = 1.0 + torch.sigmoid(diff).sum(dim=0)
    t = float(tau_softmin)
    return -t * torch.logsumexp(-soft_rank / t, dim=0)


def topk_mfr_loss(scores, y, K=10, margin=0.0):
    pos = scores[y == 1]
    neg = scores[y == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.sum() * 0.0
    kth = min(K, neg.numel())
    neg_kth = torch.topk(neg, kth, largest=True).values[-1]
    best_pos = torch.max(pos)
    return F.relu(margin + neg_kth - best_pos)


def combined_rank_loss(cfg: Dict, scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        cfg["w_focal"] * focal_loss_with_logits(scores, y)
        + cfg["w_pair"] * pairwise_rank_loss(
            scores, y,
            neg_per_pos=cfg["rank_neg_per_pos"],
            margin=cfg["rank_margin"],
        )
        + cfg["w_list"] * graph_listwise_loss(scores, y)
        + cfg["w_mar"] * mar_surrogate_loss(scores, y, tau=cfg["tau_rank"])
        + cfg["w_mfr"] * mfr_surrogate_loss(scores, y, tau_softmin=cfg["tau_mfr"], tau_rank=cfg["tau_rank"])
        + cfg["w_topk"] * topk_mfr_loss(scores, y, K=cfg["topk_K"], margin=cfg["topk_margin"])
    )


# -------------------------
# Label noise control
# -------------------------
def reduce_label_noise(
    scores_node: torch.Tensor,
    y_node: torch.Tensor,
    node_line: torch.Tensor,
    from_static: Optional[torch.Tensor] = None,
    strategy: str = "none",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Noise control strategies:
      - "none": do nothing
      - "line_top1": for each positive line, keep only the top-1 node as positive, set others to 0
      - "line_top1_dyn": same, but prefer dynamic nodes (from_static==0) if available
    NOTE:
      - only labels are changed; scores are untouched.
      - nodes with node_line < 0 are ignored.
    """
    strategy = (strategy or "none").lower()
    if strategy == "none":
        return scores_node, y_node

    device = scores_node.device
    y_node = y_node.to(device)
    node_line = node_line.to(device)
    if from_static is not None:
        from_static = from_static.to(device)

    valid_mask = (node_line >= 0)
    if valid_mask.sum() == 0:
        return scores_node, y_node

    y_new = torch.zeros_like(y_node)

    pos_mask = (y_node > 0.5) & valid_mask
    if pos_mask.sum() == 0:
        return scores_node, y_node

    pos_lines = node_line[pos_mask].unique()

    for ln in pos_lines.tolist():
        line_mask = (node_line == ln)

        if strategy == "line_top1_dyn" and from_static is not None:
            dyn_mask = line_mask & (from_static == 0)
            if dyn_mask.any():
                line_mask = dyn_mask

        idx_line = line_mask.nonzero(as_tuple=True)[0]
        if idx_line.numel() == 0:
            idx_line = (node_line == ln).nonzero(as_tuple=True)[0]
            if idx_line.numel() == 0:
                continue

        scores_line = scores_node[idx_line]
        top_rel = torch.argmax(scores_line)
        top_idx = idx_line[top_rel]
        y_new[top_idx] = 1.0

    return scores_node, y_new


# -------------------------
# Line-level MIL view
# -------------------------
def line_level_view(
    scores: torch.Tensor,
    y: torch.Tensor,
    node_line: torch.Tensor,
    pool: str = "mean",
    tau: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate node-level scores/labels into line-level "bags".
      - line label is OR over nodes on the same line.
      - score pooling: max / mean / softmax-weighted-mean
    Only nodes with node_line >= 0 are used.
    """
    device = scores.device
    node_line = node_line.to(device)
    y = y.to(device)

    mask = (node_line >= 0)
    if mask.sum() == 0:
        return scores, y

    lines = node_line[mask]
    s = scores[mask]
    y_node = y[mask]

    uniq_lines, inv = torch.unique(lines, return_inverse=True)
    num_lines = uniq_lines.size(0)

    line_labels = torch.zeros(num_lines, dtype=y_node.dtype, device=device)
    line_labels = line_labels.scatter_add(0, inv, y_node)
    line_labels = (line_labels > 0).to(y.dtype)

    pool = (pool or "mean").lower()

    if pool == "max":
        line_scores = torch.full((num_lines,), -1e9, dtype=s.dtype, device=device)
        for i in range(s.size(0)):
            idx = inv[i]
            line_scores[idx] = torch.maximum(line_scores[idx], s[i])

    elif pool == "softmax":
        t = float(tau) if tau is not None else 0.5
        a = torch.exp(s / t)
        sum_a = torch.zeros(num_lines, dtype=a.dtype, device=device).scatter_add(0, inv, a)
        weighted = torch.zeros(num_lines, dtype=s.dtype, device=device).scatter_add(0, inv, a * s)
        line_scores = weighted / (sum_a + 1e-8)

    else:
        line_sum = torch.zeros(num_lines, dtype=s.dtype, device=device).scatter_add(0, inv, s)
        line_cnt = torch.zeros(num_lines, dtype=s.dtype, device=device).scatter_add(0, inv, torch.ones_like(s))
        line_scores = line_sum / (line_cnt + 1e-8)

    return line_scores, line_labels


# -------------------------
# Training
# -------------------------
def _autocast_context(cfg: Dict, device: torch.device):
    use_amp = bool(cfg.get("use_amp", True)) and (device.type == "cuda")
    if not use_amp:
        return nullcontext(), False, torch.float32

    use_bf16 = (str(cfg.get("amp_dtype", "bf16")).lower() == "bf16")
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True), True, autocast_dtype


def train_one_epoch(cfg: Dict, model, loader, device, optimizer) -> float:
    ctx, use_amp, _ = _autocast_context(cfg, device)
    use_bf16 = (str(cfg.get("amp_dtype", "bf16")).lower() == "bf16")
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and (not use_bf16)))

    use_line_level = bool(cfg.get("use_line_level_train", False))
    w_line = float(cfg.get("line_loss_weight", 0.8))
    w_node = float(cfg.get("node_loss_weight", 0.2))
    s = w_line + w_node
    if s <= 0:
        w_line, w_node = 1.0, 0.0
    else:
        w_line, w_node = w_line / s, w_node / s

    model.train()
    total_loss = 0.0

    for batch in loader:
        data = batch[0] if isinstance(batch, (list, tuple)) else batch
        data = data.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with ctx:
            scores_node = model(
                data.x, data.edge_index,
                getattr(data, "edge_type", None),
                getattr(data, "edge_weight", None),
            )
            y_node = data.y.float()

            if hasattr(data, "node_line"):
                scores_node, y_node = reduce_label_noise(
                    scores_node, y_node,
                    getattr(data, "node_line"),
                    getattr(data, "from_static", None),
                    strategy=cfg.get("train_noise_strategy", "none"),
                )

            node_loss = combined_rank_loss(cfg, scores_node, y_node)

            line_loss = None
            if use_line_level and hasattr(data, "node_line"):
                scores_line, y_line = line_level_view(
                    scores_node, y_node,
                    getattr(data, "node_line"),
                    pool=cfg.get("line_pool", "mean"),
                    tau=float(cfg.get("line_pool_tau", 1.0)),
                )
                line_loss = combined_rank_loss(cfg, scores_line, y_line)

            loss = (w_line * line_loss + w_node * node_loss) if line_loss is not None else node_loss

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if cfg.get("grad_clip_norm", 0) and float(cfg["grad_clip_norm"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.get("grad_clip_norm", 0) and float(cfg["grad_clip_norm"]) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
            optimizer.step()

        total_loss += float(loss.detach().cpu())

    return total_loss / max(1, len(loader))


def build_train_loader(cfg: Dict, ds: GraphListDataset, train_bug_indices: List[List[int]]) -> Optional[DataLoader]:
    train_scope = str(cfg.get("train_scope", "file")).lower()

    if train_scope == "file":
        train_graph_indices: List[int] = []
        for idxs in train_bug_indices:
            for gidx in idxs:
                data, *_ = ds[gidx]
                if cfg.get("train_only_positive_graphs", True) and int(data.y.sum().item()) <= 0:
                    continue
                train_graph_indices.append(gidx)

        if not train_graph_indices:
            return None

        class _Subset(Dataset):
            def __init__(self, base: GraphListDataset, ids: List[int]):
                self.base = base
                self.ids = ids

            def __len__(self):
                return len(self.ids)

            def __getitem__(self, i):
                data, *_ = self.base[self.ids[i]]
                return data

        train_ds = _Subset(ds, train_graph_indices)
        nw = int(cfg.get("num_workers", 0))
        loader = DataLoader(
            train_ds,
            batch_size=int(cfg.get("batch_size", 1)),
            shuffle=True,
            num_workers=nw,
            pin_memory=bool(cfg.get("pin_memory", False)),
            persistent_workers=(nw > 0),
            prefetch_factor=(int(cfg.get("prefetch_factor", 2)) if nw > 0 else None),
        )
        return loader

    # bug-level merged training (kept for compatibility)
    bug_lists: List[List[int]] = []
    for idxs in train_bug_indices:
        if not idxs:
            continue
        if cfg.get("train_only_positive_graphs", True) and (not bug_has_positive(ds, idxs)):
            continue
        bug_lists.append(idxs)

    if not bug_lists:
        return None

    class BugLevelDataset(Dataset):
        def __init__(self, base: GraphListDataset, bug_graph_idx_lists: List[List[int]]):
            self.base = base
            self.bug_graph_idx_lists = bug_graph_idx_lists

        def __len__(self):
            return len(self.bug_graph_idx_lists)

        def __getitem__(self, i):
            idxs = self.bug_graph_idx_lists[i]
            datas = [self.base[gidx][0] for gidx in idxs]

            if len(datas) == 1:
                return datas[0]

            xs, edge_indices, edge_types, edge_weights, ys = [], [], [], [], []
            offset = 0
            for d in datas:
                xs.append(d.x)
                edge_indices.append(d.edge_index + offset)
                if hasattr(d, "edge_type") and d.edge_type is not None:
                    edge_types.append(d.edge_type)
                if hasattr(d, "edge_weight") and d.edge_weight is not None:
                    edge_weights.append(d.edge_weight)
                ys.append(d.y)
                offset += d.num_nodes

            x = torch.cat(xs, dim=0)
            edge_index = torch.cat(edge_indices, dim=1)
            y = torch.cat(ys, dim=0)

            merged = Data(x=x, edge_index=edge_index, y=y)
            merged.edge_type = torch.cat(edge_types, dim=0) if edge_types else torch.zeros(edge_index.size(1), dtype=torch.long)
            merged.edge_weight = torch.cat(edge_weights, dim=0) if edge_weights else torch.ones(edge_index.size(1), dtype=torch.float)
            return merged

    train_ds = BugLevelDataset(ds, bug_lists)
    nw = int(cfg.get("num_workers", 0))
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=nw,
        pin_memory=bool(cfg.get("pin_memory", False)),
        persistent_workers=(nw > 0),
        prefetch_factor=(int(cfg.get("prefetch_factor", 2)) if nw > 0 else None),
    )
    return loader


def init_model(cfg: Dict, ds: GraphListDataset, device: torch.device) -> nn.Module:
    sample_data, *_ = ds[0]
    in_dim = sample_data.x.size(1)

    hidden_dim = choose_hidden_dim(cfg, ds)
    logger.info(f"[auto_hidden] hidden_dim={hidden_dim}")

    model = GATNodeScorer(
        in_dim=in_dim,
        code_dim=int(cfg["code_dim"]),
        hidden=int(hidden_dim),
        heads=int(cfg["heads"]),
        num_relations=int(cfg["num_relations"]),
        code_weight=float(cfg["code_weight"]),
        dropout=float(cfg["dropout"]),
        code_at_front=bool(cfg.get("code_at_front", True)),
    ).to(device)
    return model


def train_once(
    cfg: Dict,
    ds: GraphListDataset,
    train_bug_indices: List[List[int]],
    device: torch.device,
    max_epochs: Optional[int] = None,
    force_no_early_stop: bool = False,
) -> Optional[nn.Module]:
    bug_indices_all = list(train_bug_indices)

    train_loader = build_train_loader(cfg, ds, bug_indices_all)
    if train_loader is None:
        return None

    model = init_model(cfg, ds, device)

    max_epochs = int(cfg.get("epochs", 50) if max_epochs is None else max_epochs)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    use_early = bool(cfg.get("early_stop", False)) and (not force_no_early_stop)
    gap = int(cfg.get("early_stop_gap", 5) or 5)

    epoch_losses: List[float] = []
    best_state = copy.deepcopy(model.state_dict())
    best_train_loss = float("inf")
    tol = 1e-6

    for ep in range(1, max_epochs + 1):
        train_loss = train_one_epoch(cfg, model, train_loader, device, opt)
        sch.step()

        epoch_losses.append(float(train_loss))
        logger.info(f"  ep {ep:02d}/{max_epochs:02d} | train_loss={train_loss:.4f} | batches={len(train_loader)}")

        if train_loss + tol < best_train_loss:
            best_train_loss = train_loss
            best_state = copy.deepcopy(model.state_dict())

        if (not use_early) or (ep <= gap + 3):
            continue

        prev_ep = ep - gap
        prev_loss = epoch_losses[prev_ep - 1]
        logger.info(f"  [early-stop] compare ep {ep} (loss={train_loss:.4f}) vs ep {prev_ep} (loss={prev_loss:.4f})")

        if train_loss > prev_loss + tol:
            logger.info(f"  early-stopping at epoch {ep} (loss increased: {prev_loss:.4f} -> {train_loss:.4f})")
            break

    if use_early:
        model.load_state_dict(best_state)

    return model


# -------------------------
# Evaluation helpers
# -------------------------
def stable_rank(pred_items: List[Tuple[str, float]]) -> List[Tuple[str, float, int]]:
    pred_items = sorted(pred_items, key=lambda x: (-x[1], x[0]))
    ranked = []
    last_s, last_r = None, 0
    seen = 0
    for eid, s in pred_items:
        seen += 1
        if s != last_s:
            last_r = seen
            last_s = s
        ranked.append((eid, s, last_r))
    return ranked


def _elem_file_id(eid: str) -> str:
    if ":" in eid:
        return eid.split(":", 1)[0]
    if "#" in eid:
        return eid.split("#", 1)[0]
    return eid


def _group_by_file(pred_scores: Dict[str, float]) -> Dict[str, List[Tuple[str, float]]]:
    by_file: Dict[str, List[Tuple[str, float]]] = {}
    for eid, s in pred_scores.items():
        by_file.setdefault(_elem_file_id(eid), []).append((eid, s))
    return by_file


# -------------------------
# Raw JSON fallback (only if node_line is missing)
# -------------------------
def _raw_root_map(cfg: Dict) -> Dict[str, Path]:
    return {Path(r).name: Path(r) for r in (cfg.get("raw_json_roots") or [])}


def load_json_for(cfg: Dict, project: str, pid: int, identifier: str) -> Optional[dict]:
    root = _raw_root_map(cfg).get(project)
    if not root:
        return None
    p = root / str(pid) / f"{identifier}.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    cand = list((root / str(pid)).glob(f"{identifier}*.json"))
    if cand:
        with open(cand[0], "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def nodeindex_to_line_list(cfg: Dict, project: str, pid: int, identifier: str, n_nodes: int) -> Optional[List[int]]:
    js = load_json_for(cfg, project, pid, identifier)
    if not js or "nodes" not in js:
        return None
    lines = []
    for _, info in js["nodes"].items():
        lines.append(info.get("line", -1))
    return lines if len(lines) == n_nodes else None


@torch.no_grad()
def predict_one_bug(
    cfg: Dict,
    model,
    ds: GraphListDataset,
    bug_indices: List[int],
    device: torch.device,
    merge_same_line: bool,
    reduce: str,
) -> Tuple[Dict[str, float], List[str]]:
    ctx, use_amp, _ = _autocast_context(cfg, device)

    model.eval()
    pred_scores: Dict[str, float] = {}
    gold_ids: List[str] = []

    for idx in bug_indices:
        data, project, pid, identifier = ds[idx]
        data = data.to(device, non_blocking=True)

        with ctx:
            scores = model(
                data.x, data.edge_index,
                getattr(data, "edge_type", None),
                getattr(data, "edge_weight", None),
            ).detach().cpu()

        y = data.y.detach().cpu()

        if not merge_same_line:
            for i in range(data.num_nodes):
                eid = f"{identifier}#node{i}"
                pred_scores[eid] = float(scores[i])
                if int(y[i].item()) == 1:
                    gold_ids.append(eid)
            continue

        # line-level merge
        if hasattr(data, "node_line"):
            lines = data.node_line.detach().cpu().tolist()
        else:
            lines = nodeindex_to_line_list(cfg, project, pid, identifier, data.num_nodes)

        if lines is None:
            for i in range(data.num_nodes):
                eid = f"{identifier}#node{i}"
                pred_scores[eid] = float(scores[i])
                if int(y[i].item()) == 1:
                    gold_ids.append(eid)
            continue

        groups: Dict[int, List[int]] = {}
        for i, ln in enumerate(lines):
            groups.setdefault(int(ln), []).append(i)

        for ln, idxs in groups.items():
            key = f"{identifier}:{ln}"
            vals = scores[idxs]
            pred_scores[key] = float(vals.mean().item() if reduce == "mean" else vals.max().item())
            if int(y[idxs].sum().item()) > 0:
                gold_ids.append(key)

    return pred_scores, gold_ids


def eval_one_bug(cfg: Dict, pred_scores: Dict[str, float], gold_ids: Iterable[str], ks=(1, 3, 5, 10)) -> Dict[str, float]:
    gold = set(gold_ids)
    rank_scope = str(cfg.get("rank_scope", "global")).lower()
    mar_mode = str(cfg.get("mar_mode", "best_only")).lower()
    topk_mode = str(cfg.get("topk_mode", "bug")).lower()

    # ---- per_file: only evaluate within defect files ----
    if rank_scope == "per_file":
        by_file = _group_by_file(pred_scores)
        if not by_file:
            return {**{f"Top@{k}": 0.0 for k in ks}, "MFR": 0.0, "MAR": 0.0}

        defect_files = {_elem_file_id(g) for g in gold}
        if defect_files:
            by_file = {f: items for f, items in by_file.items() if f in defect_files}

        if not by_file:
            N_total = 0
            res = {f"Top@{k}": 0.0 for k in ks}
            res["MFR"] = float(N_total)
            res["MAR"] = float(N_total)
            return res

        per_file_rankmap: Dict[str, Dict[str, int]] = {}
        per_file_sizes: Dict[str, int] = {}

        for f, items in by_file.items():
            ranked = stable_rank(items)
            per_file_sizes[f] = len(ranked)
            per_file_rankmap[f] = {eid: r for (eid, _s, r) in ranked}

        from collections import defaultdict
        gold_ranks_by_file: Dict[str, List[int]] = defaultdict(list)
        for g in gold:
            f = _elem_file_id(g)
            fmap = per_file_rankmap.get(f)
            if fmap and (g in fmap):
                gold_ranks_by_file[f].append(int(fmap[g]))

        if not gold_ranks_by_file:
            N_total = sum(per_file_sizes.values())
            res = {f"Top@{k}": 0.0 for k in ks}
            res["MFR"] = float(N_total)
            res["MAR"] = float(N_total)
            return res

        per_file_min = {f: min(rs) for f, rs in gold_ranks_by_file.items()}

        res = {}
        if topk_mode == "fileavg":
            for k in ks:
                hits = [1.0 if rmin <= k else 0.0 for rmin in per_file_min.values()]
                res[f"Top@{k}"] = float(np.mean(hits)) if hits else 0.0
        else:
            for k in ks:
                res[f"Top@{k}"] = 1.0 if any(rmin <= k for rmin in per_file_min.values()) else 0.0

        best_rank = min(per_file_min.values())
        res["MFR"] = float(best_rank)

        if mar_mode == "best_only":
            res["MAR"] = float(best_rank)
        elif mar_mode == "per_file_min":
            res["MAR"] = float(np.mean(list(per_file_min.values())))
        else:
            all_ranks = [r for rs in gold_ranks_by_file.values() for r in rs]
            res["MAR"] = float(np.mean(all_ranks)) if all_ranks else float(best_rank)
        return res

    # ---- global: filter non-defect files, normalize per file, then global rank ----
    by_file = _group_by_file(pred_scores)
    defect_files = {_elem_file_id(g) for g in gold}
    if defect_files:
        by_file = {f: items for f, items in by_file.items() if f in defect_files}

    if not by_file:
        return {**{f"Top@{k}": 0.0 for k in ks}, "MFR": 0.0, "MAR": 0.0}

    use_norm = bool(cfg.get("global_use_file_norm", False))
    norm_type = str(cfg.get("global_norm_type", "zscore")).lower()
    top_m = int(cfg.get("global_top_m_per_file", 0) or 0)

    normalized_items: List[Tuple[str, float]] = []
    for f, items in by_file.items():
        scores = np.array([s for (_eid, s) in items], dtype=np.float32)

        if use_norm and len(scores) > 1:
            if norm_type == "zscore":
                m = float(scores.mean())
                std = float(scores.std())
                scores = (scores - m) / std if std >= 1e-8 else (scores * 0.0)
            elif norm_type == "minmax":
                lo = float(scores.min())
                hi = float(scores.max())
                scores = (scores - lo) / (hi - lo) if (hi - lo) >= 1e-8 else (scores * 0.0)
            elif norm_type == "rank":
                order = np.argsort(-scores)
                ranks = np.empty_like(order)
                ranks[order] = np.arange(len(scores))
                scores = 1.0 - ranks / float(len(scores) - 1) if len(scores) > 1 else np.zeros_like(scores)

        file_items = [(eid, float(s)) for (eid, _), s in zip(items, scores)]
        if top_m > 0 and len(file_items) > top_m:
            file_items = sorted(file_items, key=lambda x: -x[1])[:top_m]

        normalized_items.extend(file_items)

    if not normalized_items:
        return {**{f"Top@{k}": 0.0 for k in ks}, "MFR": 0.0, "MAR": 0.0}

    ranked = stable_rank(normalized_items)
    rank_map = {eid: r for (eid, _s, r) in ranked}
    gold_ranks = [(eid, rank_map[eid]) for eid in gold if eid in rank_map]
    N = len(ranked)

    if not gold_ranks:
        res = {f"Top@{k}": 0.0 for k in ks}
        res["MFR"] = float(N)
        res["MAR"] = float(N)
        return res

    best_rank = min(r for (_eid, r) in gold_ranks)
    res = {}
    if topk_mode == "fileavg":
        from collections import defaultdict
        gold_by_file = defaultdict(list)
        for eid, r in gold_ranks:
            gold_by_file[_elem_file_id(eid)].append(r)
        file_hit = [min(rs) for rs in gold_by_file.values()]
        for k in ks:
            res[f"Top@{k}"] = float(np.mean([1.0 if rmin <= k else 0.0 for rmin in file_hit])) if file_hit else 0.0
    else:
        for k in ks:
            res[f"Top@{k}"] = 1.0 if best_rank <= k else 0.0

    res["MFR"] = float(best_rank)

    if mar_mode == "best_only":
        res["MAR"] = float(best_rank)
    elif mar_mode == "per_file_min":
        from collections import defaultdict
        by_file2 = defaultdict(list)
        for eid, r in gold_ranks:
            by_file2[_elem_file_id(eid)].append(int(r))
        per_file_min = [min(rs) for rs in by_file2.values()]
        res["MAR"] = float(np.mean(per_file_min)) if per_file_min else float(N)
    else:
        res["MAR"] = float(np.mean([r for (_eid, r) in gold_ranks]))
    return res


def aggregate_counts(results: List[Dict[str, float]], ks=(1, 3, 5, 10)) -> Tuple[Dict[str, int], Dict[str, float], int]:
    if not results:
        return ({f"Top@{k}": 0 for k in ks}, {"MFR": 0.0, "MAR": 0.0}, 0)

    hits = {f"Top@{k}": 0 for k in ks}
    mfrs, mars = [], []
    for r in results:
        for k in ks:
            hits[f"Top@{k}"] += int(round(float(r.get(f"Top@{k}", 0.0))))
        mfrs.append(float(r["MFR"]))
        mars.append(float(r["MAR"]))
    return hits, {"MFR": float(np.mean(mfrs)), "MAR": float(np.mean(mars))}, len(results)
