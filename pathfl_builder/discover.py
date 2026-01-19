# pathfl_builder/discover.py
# -*- coding: utf-8 -*-
from pathlib import Path

def is_pid_dir(p: Path) -> bool:
    return p.is_dir() and p.name.isdigit()

def discover_tasks(input_root: Path, cfg: dict):
    """
    支持 4 种输入：
    A) input_root 直接含 *.json             => 单项目
    B) input_root 是 dataset 目录（含 pid 子目录） => 单数据集多项目
    C) input_root 是总根目录（含 dataset 子目录） => 多数据集多项目
    D) input_root 指向某个 pid 目录          => 单项目
    返回 tasks: List[(dataset_name, pid, in_dir)]
    """
    input_root = input_root.resolve()

    # D/A: 目录里直接有 json
    jsons = list(input_root.glob("*.json"))
    if jsons:
        return [(input_root.name or "custom", 0, input_root)]

    # D: pid 目录
    if is_pid_dir(input_root):
        ds_name = input_root.parent.name
        return [(ds_name, int(input_root.name), input_root)]

    # B: dataset 目录（含数字子目录）
    pid_dirs = [p for p in input_root.iterdir() if is_pid_dir(p)]
    if pid_dirs:
        ds_name = input_root.name
        tasks = [(ds_name, int(p.name), p) for p in sorted(pid_dirs, key=lambda x: int(x.name))]
        return tasks

    # C: base root（含 dataset 子目录）
    datasets = cfg.get("datasets", None)
    ds_dirs = [p for p in input_root.iterdir() if p.is_dir()]
    if datasets is not None:
        ds_dirs = [p for p in ds_dirs if p.name in set(datasets)]

    tasks = []
    for ds in sorted(ds_dirs, key=lambda x: x.name):
        pid_dirs = [p for p in ds.iterdir() if is_pid_dir(p)]
        for p in sorted(pid_dirs, key=lambda x: int(x.name)):
            tasks.append((ds.name, int(p.name), p))

    return tasks
