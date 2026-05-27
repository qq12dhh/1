"""第四章基准切换策略。"""
from __future__ import annotations

import numpy as np

from utils import masked_argmax


def max_capacity_action(cg_bps: np.ndarray, visible: np.ndarray, prev_sat: int | None = None) -> int:
    """最大信道容量/最大接收质量策略。"""
    return masked_argmax(cg_bps, visible)


def max_visible_time_action(remaining_time_s: np.ndarray, visible: np.ndarray, prev_sat: int | None = None) -> int:
    """最大可见时长策略：当前卫星仍可见则保持，否则选剩余可见时长最长者。"""
    if prev_sat is not None and 0 <= prev_sat < len(visible) and visible[prev_sat]:
        return int(prev_sat)
    return masked_argmax(remaining_time_s, visible)


def min_hop_action(hop_count: np.ndarray, visible: np.ndarray, prev_sat: int | None = None) -> int:
    """最小路径跳数策略，作为调试/消融基准。"""
    score = -np.asarray(hop_count, dtype=np.float64)
    return masked_argmax(score, visible)


def min_path_quality_action(path_quality: np.ndarray, visible: np.ndarray, prev_sat: int | None = None) -> int:
    """最小路径质量代价策略，作为调试/消融基准。"""
    score = -np.asarray(path_quality, dtype=np.float64)
    return masked_argmax(score, visible)
