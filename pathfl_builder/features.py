# pathfl_builder/features.py
# -*- coding: utf-8 -*-
import numpy as np

def build_node_features(nodes_dict: dict, code_embeds: np.ndarray, deg_in: np.ndarray, deg_out: np.ndarray):
    struct, dataflow, flags, degs = [], [], [], []
    id_list = list(nodes_dict.keys())

    for i, (_, info) in enumerate(nodes_dict.items()):
        is_entry = int(info.get("is_entry", 0))
        is_exit  = int(info.get("is_exit", 0))
        has_mtd  = 1 if info.get("method") else 0
        struct.append([is_entry, is_exit, has_mtd])

        defs_cnt = len(info.get("defs", {}) or {})
        uses_cnt = len(info.get("uses", {}) or {})
        calls_obj = info.get("calls", {}) or {}
        calls_cnt = len(calls_obj) if isinstance(calls_obj, (dict, list)) else 0
        dataflow.append([defs_cnt, uses_cnt, calls_cnt])

        code_text = info.get("code_content") or info.get("code") or ""
        has_code = 1 if (code_text and str(code_text).strip()) else 0
        from_static = 1 if bool(info.get("from_static", False)) else 0
        try:
            ln = int(info.get("line", -1))
        except Exception:
            ln = -1
        has_line = 1 if ln >= 0 else 0
        flags.append([from_static, has_code, has_line])

        di = float(deg_in[i]) if i < len(deg_in) else 0.0
        do = float(deg_out[i]) if i < len(deg_out) else 0.0
        degs.append([di, do])

    struct = np.asarray(struct, dtype=np.float32)
    dataflow = np.asarray(dataflow, dtype=np.float32)
    flags = np.asarray(flags, dtype=np.float32)
    degs = np.asarray(degs, dtype=np.float32)

    if code_embeds.shape[0] != len(id_list):
        raise ValueError("code_embeds 数量与节点数不一致")

    x = np.hstack([code_embeds, struct, dataflow, flags, degs])  # 779
    return x
