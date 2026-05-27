""" MDP 环境。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from config_ch3 import SimConfig
from utils import masked_argmax


class HandoverEnv:
    """单用户 episode 环境。

    每个 episode 从多用户数据集中抽一个用户，在 20 分钟、120 个时隙内选择接入卫星。
    状态：
        [sigma, prev_sat_onehot, visible_mask, ET_norm, ER_norm, remaining_norm]
    动作：
        选择 0..M-1 号卫星。
    """

    def __init__(
        self,
        dataset_path: str | Path,
        cfg: SimConfig,
        num_users: Optional[int] = None,
        sensitive_ratio: float = 0.5,
        fixed_sigma: Optional[int] = None,
        seed: int = 42,
    ):
        self.cfg = cfg
        self.dataset_path = Path(dataset_path)
        self.data = np.load(self.dataset_path, allow_pickle=False)
        self.visible_all = self.data["visible"].astype(bool)
        self.et_raw_all = self.data["et_bps"].astype(np.float32)
        self.er_raw_all = self.data["er_bps"].astype(np.float32)
        self.snr_db_all = self.data["snr_db"].astype(np.float32)
        self.remaining_all = self.data["remaining_time_s"].astype(np.float32)

        self.t_num, self.user_total, self.sat_num = self.visible_all.shape
        self.num_users = min(num_users or cfg.num_users, self.user_total)
        self.sensitive_ratio = sensitive_ratio
        self.fixed_sigma = fixed_sigma
        self.rng = np.random.default_rng(seed)

        self.state_dim = 1 + 5 * self.sat_num
        self.action_dim = self.sat_num

        self.et_threshold = cfg.bc_hz * np.log2(1.0 + 10 ** (cfg.snr_threshold_sensitive_db / 10.0))
        self.er_threshold = cfg.bc_hz * np.log2(1.0 + 10 ** (cfg.snr_threshold_tolerant_db / 10.0))
        self._refresh_effective_rates()

        self.t = 0
        self.user_idx = 0
        self.sigma = 0
        self.prev_sat = -1

    def _refresh_effective_rates(self) -> None:
        """按论文约束 C4 加入卫星带宽共享效应。

        原始链路容量只反映单用户物理链路质量；第三章实验中用户数增加会导致速率下降，
        这里用“可见用户数近似同星竞争用户数”计算带宽共享因子：

            share = min(1, B / (N_visible_to_sat * Bc))

        num_users 越大，同一卫星可见用户越多，ET/ER 越低。
        """
        users = slice(0, self.num_users)
        visible_load = self.visible_all[:, users, :].sum(axis=1).astype(np.float32)  # [T, M]
        visible_load = np.maximum(visible_load, 1.0)
        share = np.minimum(1.0, self.cfg.sat_total_bandwidth_hz / (visible_load * self.cfg.bc_hz))
        self.er_all = self.er_raw_all * share[:, None, :]
        self.et_all = self.et_raw_all * share[:, None, :]

    def _sample_sigma(self) -> int:
        if self.fixed_sigma is not None:
            return int(self.fixed_sigma)
        return int(self.rng.random() < self.sensitive_ratio)

    def reset(self, user_idx: Optional[int] = None, sigma: Optional[int] = None) -> np.ndarray:
        self.t = 0
        self.user_idx = int(user_idx if user_idx is not None else self.rng.integers(0, self.num_users))
        self.sigma = int(sigma if sigma is not None else self._sample_sigma())
        vis = self.visible_all[self.t, self.user_idx]
        self.prev_sat = masked_argmax(self.remaining_all[self.t, self.user_idx], vis)
        return self._make_state()

    def _make_state(self) -> np.ndarray:
        vis = self.visible_all[self.t, self.user_idx].astype(np.float32)
        prev = np.zeros(self.sat_num, dtype=np.float32)
        if 0 <= self.prev_sat < self.sat_num:
            prev[self.prev_sat] = 1.0

        et = self.et_all[self.t, self.user_idx] / max(self.cfg.tau_max_bps, 1.0)
        er = self.er_all[self.t, self.user_idx] / max(self.cfg.tau_max_bps, 1.0)
        rem = self.remaining_all[self.t, self.user_idx] / max(self.cfg.sim_minutes * 60.0, 1.0)

        return np.concatenate([
            np.array([float(self.sigma)], dtype=np.float32),
            prev,
            vis,
            np.clip(et, 0.0, 1.5).astype(np.float32),
            np.clip(er, 0.0, 1.5).astype(np.float32),
            np.clip(rem, 0.0, 1.0).astype(np.float32),
        ]).astype(np.float32)

    def action_mask(self) -> np.ndarray:
        return self.visible_all[self.t, self.user_idx].astype(bool)

    def step(self, action: int):
        action = int(action)
        vis = self.visible_all[self.t, self.user_idx]
        invalid = not bool(vis[action])
        if invalid:
            selected = masked_argmax(self.remaining_all[self.t, self.user_idx], vis)
            invalid_penalty = 1.0
        else:
            selected = action
            invalid_penalty = 0.0

        et = float(self.et_all[self.t, self.user_idx, selected])
        er = float(self.er_all[self.t, self.user_idx, selected])
        rem = float(self.remaining_all[self.t, self.user_idx, selected])
        snr_db = float(self.snr_db_all[self.t, self.user_idx, selected])
        handover = 1.0 if selected != self.prev_sat else 0.0

        et_max = float(np.max(self.et_all[self.t, self.user_idx][vis])) if np.any(vis) else 1.0
        er_max = float(np.max(self.er_all[self.t, self.user_idx][vis])) if np.any(vis) else 1.0
        rem_max = float(np.max(self.remaining_all[self.t, self.user_idx][vis])) if np.any(vis) else 1.0
        et_norm = et / max(et_max, 1.0)
        er_norm = er / max(er_max, 1.0)
        rem_norm = rem / max(rem_max, 1.0)

        if self.sigma == 1:
            w1, w2, w3, w4, lam = self.cfg.reward_sensitive
            qos_penalty = max(0.0, (self.et_threshold - et) / max(self.et_threshold, 1.0))
        else:
            w1, w2, w3, w4, lam = self.cfg.reward_tolerant
            qos_penalty = max(0.0, (self.er_threshold - er) / max(self.er_threshold, 1.0))

        reward = w1 * et_norm + w2 * er_norm + w3 * rem_norm - w4 * handover - lam * qos_penalty - invalid_penalty

        self.prev_sat = selected
        self.t += 1
        done = self.t >= min(self.t_num, self.cfg.max_steps_per_episode)
        next_state = np.zeros(self.state_dim, dtype=np.float32) if done else self._make_state()
        info = {
            "selected_sat": selected,
            "invalid": invalid,
            "handover": handover,
            "et_bps": et,
            "er_bps": er,
            "remaining_time_s": rem,
            "snr_db": snr_db,
            "sigma": self.sigma,
            "user_idx": self.user_idx,
        }
        return next_state, float(reward), done, info
