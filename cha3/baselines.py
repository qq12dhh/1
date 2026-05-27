"""第三章对比切换策略。"""
from __future__ import annotations

import numpy as np

from utils import masked_argmax


def max_snr_action(snr_db: np.ndarray, visible: np.ndarray, prev_sat: int | None = None) -> int:
    """最大信噪比策略。"""
    return masked_argmax(snr_db, visible)


def max_visible_time_action(remaining_time_s: np.ndarray, visible: np.ndarray, prev_sat: int | None = None) -> int:
    """最大可见时长策略：当前卫星仍可见就保持，否则选剩余可见时长最大卫星。"""
    if prev_sat is not None and 0 <= prev_sat < len(visible) and visible[prev_sat]:
        return int(prev_sat)
    return masked_argmax(remaining_time_s, visible)

