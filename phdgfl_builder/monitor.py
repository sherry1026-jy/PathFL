# phdgfl_builder/monitor.py
# -*- coding: utf-8 -*-
import time
import numpy as np
from datetime import timedelta
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.stats = defaultdict(list)

    def update(self, dataset, pid, graphs, nodes, edges, duration):
        self.stats["dataset"].append(dataset)
        self.stats["pid"].append(pid)
        self.stats["graphs"].append(graphs)
        self.stats["nodes"].append(nodes)
        self.stats["edges"].append(edges)
        self.stats["duration"].append(duration)

    def report(self):
        T = time.time() - self.start_time
        print("\n=== 性能报告 ===")
        print(f"处理项目数: {len(self.stats['pid'])}")
        print(f"总图数: {sum(self.stats['graphs'])}")
        print(f"总节点数: {sum(self.stats['nodes']):,}")
        print(f"总边数: {sum(self.stats['edges']):,}")
        if self.stats["duration"]:
            print(f"平均项目耗时: {np.mean(self.stats['duration']):.1f}s")
        print(f"总耗时: {timedelta(seconds=int(T))}")
        print(f"节点吞吐量: {sum(self.stats['nodes'])/max(1,T):.1f} nodes/s")
