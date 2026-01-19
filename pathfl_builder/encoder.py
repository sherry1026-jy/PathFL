# pathfl_builder/encoder.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import multiprocessing as mp
from transformers import AutoTokenizer, AutoModel

class CodeBERTEncoder:
    _initialized = False
    _lock = mp.Lock()
    device = None
    tokenizer = None
    model = None

    @classmethod
    def init(cls, cfg: dict, logger=None):
        if cls._initialized:
            return
        with cls._lock:
            if cls._initialized:
                return
            if logger:
                logger.info(f"[PID {os.getpid()}] 初始化 CodeBERT...")

            cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # ✅ 强制本地加载，避免离线环境卡住
            cls.tokenizer = AutoTokenizer.from_pretrained(cfg["codebert_model"], local_files_only=True)
            cls.model = AutoModel.from_pretrained(cfg["codebert_model"], local_files_only=True).to(cls.device)
            cls.model.eval()

            if cfg.get("compile_hf_model", False) and hasattr(torch, "compile"):
                try:
                    cls.model = torch.compile(cls.model)
                    if logger:
                        logger.info("已对 CodeBERT 启用 torch.compile")
                except Exception as e:
                    if logger:
                        logger.warning(f"torch.compile 失败，已忽略：{e}")

            cls._initialized = True

    @classmethod
    def encode_batch(cls, texts, cfg: dict):
        if not texts:
            return np.zeros((0, cfg["code_embed_dim"]), dtype=np.float32)

        texts = [t if (t and str(t).strip()) else cfg["empty_code_fallback"] for t in texts]

        inputs = cls.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=cfg["max_code_length"],
            padding=True
        ).to(cls.device)

        use_amp = cfg.get("use_amp", True) and torch.cuda.is_available()
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type=cls.device.type, enabled=True):
                    out = cls.model(**inputs).last_hidden_state
            else:
                out = cls.model(**inputs).last_hidden_state

        emb = out[:, 0, :].detach().cpu().numpy().astype(np.float32)

        if torch.cuda.is_available():
            del inputs, out
            torch.cuda.empty_cache()
        return emb
