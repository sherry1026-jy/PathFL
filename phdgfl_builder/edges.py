# phdgfl_builder/edges.py
# -*- coding: utf-8 -*-
import numpy as np
import torch

def build_edges_and_degrees(nodes_dict: dict, edges_dict: dict, cfg: dict):
    id_list = list(nodes_dict.keys())
    id2idx = {nid: i for i, nid in enumerate(id_list)}
    N = len(id_list)

    srcs, dsts, etypes, eweights = [], [], [], []
    deg_in = np.zeros(N, dtype=np.float32)
    deg_out = np.zeros(N, dtype=np.float32)

    coef_map = cfg.get("edge_type_weight_coef", {})

    for key, edge_list in (edges_dict or {}).items():
        try:
            src, dst = key.split("-")
        except Exception:
            continue
        if src not in id2idx or dst not in id2idx:
            continue
        sidx, didx = id2idx[src], id2idx[dst]

        for einfo in edge_list:
            et = str(einfo.get("type", "")).lower()
            if et in ("control_flow", "cfg"):
                et_name = "cfg"
                et_id = cfg["edge_type_mapping"]["cfg"]
            elif et in ("data_flow", "dataflow", "data"):
                et_name = "data_flow"
                et_id = cfg["edge_type_mapping"]["data_flow"]
            elif et == "call":
                et_name = "call"
                et_id = cfg["edge_type_mapping"]["call"]
            elif et == "return":
                et_name = "return"
                et_id = cfg["edge_type_mapping"]["return"]
            elif et == "entry":
                et_name = "entry"
                et_id = cfg["edge_type_mapping"]["entry"]
            else:
                et_name = "cfg"
                et_id = cfg["edge_type_mapping"]["cfg"]

            cnt = float(einfo.get("count", 1.0))
            base_w = np.log1p(cnt)
            coef = float(coef_map.get(et_name, 1.0))
            w = base_w * coef

            srcs.append(sidx); dsts.append(didx)
            etypes.append(et_id); eweights.append(w)

            deg_out[sidx] += 1.0
            deg_in[didx]  += 1.0

    if len(srcs) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
    else:
        edge_index = torch.as_tensor([srcs, dsts], dtype=torch.long)
        edge_type = torch.as_tensor(etypes, dtype=torch.long)
        edge_weight = torch.as_tensor(eweights, dtype=torch.float32)

    return edge_index, edge_type, edge_weight, deg_in, deg_out
