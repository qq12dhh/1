"""第四章复现通用工具函数。"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def db_to_linear(x_db):
    return 10 ** (np.asarray(x_db) / 10.0)


def dbm_to_watt(dbm):
    return 10 ** ((np.asarray(dbm) - 30.0) / 10.0)


def safe_norm(x: np.ndarray, axis=None, keepdims=False, eps: float = 1e-12) -> np.ndarray:
    return np.sqrt(np.maximum(np.sum(x * x, axis=axis, keepdims=keepdims), eps))


def masked_argmax(values: np.ndarray, mask: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.float64).copy()
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return int(np.argmax(values))
    values[~mask] = -np.inf
    return int(np.argmax(values))


def moving_average(x: np.ndarray, window: int = 10) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window, dtype=np.float64) / window, mode="valid")


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
