# phdgfl_builder/merge.py
# -*- coding: utf-8 -*-
from collections import defaultdict
from copy import deepcopy

def merge_nodes_by_line(nodes: dict, edges: dict):
    groups = defaultdict(list)
    for nid, info in nodes.items():
        ln = info.get("line", -1)
        try:
            ln = int(ln) if ln is not None else -1
        except Exception:
            ln = -1
        if ln < 0:
            groups[(ln, nid)].append(nid)
        else:
            groups[ln].append(nid)

    new_nodes = {}
    old2new = {}

    for gkey, nid_list in groups.items():
        rep = nid_list[0]
        rep_info = deepcopy(nodes[rep])

        if not isinstance(gkey, tuple):
            rep_info["line"] = int(gkey)

        for nid in nid_list[1:]:
            info = nodes[nid]

            for flag in ("is_entry", "is_exit", "is_defect"):
                if info.get(flag, 0):
                    rep_info[flag] = 1

            fs = bool(rep_info.get("from_static", False))
            fs_other = bool(info.get("from_static", False))
            if fs and not fs_other:
                rep_info["from_static"] = False
            else:
                rep_info["from_static"] = bool(rep_info.get("from_static", False)) and bool(info.get("from_static", False))

            if not rep_info.get("method") and info.get("method"):
                rep_info["method"] = info.get("method")

            for field in ("defs", "uses", "calls"):
                v_rep = rep_info.get(field)
                v_other = info.get(field)
                if not v_other:
                    continue
                if isinstance(v_rep, dict) or v_rep is None:
                    if v_rep is None:
                        v_rep = {}
                    if isinstance(v_other, dict):
                        for k, v in v_other.items():
                            if k not in v_rep:
                                v_rep[k] = v
                    elif isinstance(v_other, list):
                        for k in v_other:
                            if k not in v_rep:
                                v_rep[k] = True
                    rep_info[field] = v_rep
                elif isinstance(v_rep, list):
                    tmp = list(v_rep)
                    if isinstance(v_other, dict):
                        tmp.extend(list(v_other.keys()))
                    elif isinstance(v_other, list):
                        tmp.extend(v_other)
                    rep_info[field] = list(dict.fromkeys(tmp))

        new_nodes[rep] = rep_info
        for nid in nid_list:
            old2new[nid] = rep

    new_edges_counter = defaultdict(float)
    for key, edge_list in (edges or {}).items():
        try:
            src, dst = key.split("-")
        except Exception:
            continue
        if src not in old2new or dst not in old2new:
            continue
        new_src = old2new[src]
        new_dst = old2new[dst]
        for einfo in edge_list:
            etype = str(einfo.get("type", ""))
            cnt = float(einfo.get("count", 1.0))
            new_edges_counter[(new_src, new_dst, etype)] += cnt

    new_edges = {}
    for (ns, nd, etype), cnt in new_edges_counter.items():
        k = f"{ns}-{nd}"
        new_edges.setdefault(k, []).append({"type": etype, "count": cnt})

    return new_nodes, new_edges
