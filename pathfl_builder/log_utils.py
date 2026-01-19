# pathfl_builder/log_utils.py
# -*- coding: utf-8 -*-
import logging
import torch

class PerformanceFormatter(logging.Formatter):
    def format(self, record):
        mem = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
        record.mem_usage = f"{mem:.2f}GB" if mem > 0 else "N/A"
        return super().format(record)

def setup_logger(name: str, level: str):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    handler = logging.StreamHandler()
    handler.setFormatter(PerformanceFormatter('%(asctime)s [%(levelname)s][MEM:%(mem_usage)s] %(message)s'))
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger
