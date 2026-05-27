"""MDP 环境：业务感知 + 星间路径质量/跳数。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from config_ch4 import SimConfig
from utils import masked_argmax


class Ch4HandoverEnv:
    """单终端 episode 环境。

    状态向量论：
        [上一时隙接入 one-hot, 当前 C^g, 当前路径跳数 d, 当前路径质量 Q,
         上一时隙实际速率 tau(t-1), 业务类型 sigma]
    动作：选择一颗可见卫星作为接入卫星。
    奖励：以速率、跳数、切换开销为核心，加入轻量全局负载影响项。
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
        self.cg_raw_all = self.data["cg_bps"].astype(np.float32)
        self.snr_db_all = self.data["snr_db"].astype(np.float32)
        self.remaining_all = self.data["remaining_time_s"].astype(np.float32)
        self.path_quality_all = self.data["path_quality"].astype(np.float32)      # [T,N,G]
        self.hop_count_all = self.data["hop_count"].astype(np.float32)            # [T,N,G]
        self.next_hop_all = self.data["route_next_hop"].astype(np.int16)          # [T,N,G]
        self.user_gateway_idx_all = self.data["user_gateway_idx"].astype(np.int16)

        self.t_num, self.user_total, self.sat_num = self.visible_all.shape
        self.num_users = min(num_users or cfg.num_users, self.user_total)
        self.sensitive_ratio = sensitive_ratio
        self.fixed_sigma = fixed_sigma
        self.rng = np.random.default_rng(seed)

        self.state_dim = 4 * self.sat_num + 2
        self.action_dim = self.sat_num

        self.q_scale = float(np.percentile(self.path_quality_all, 95))
        self.q_scale = max(self.q_scale, 1.0)

        self._refresh_effective_access_capacity()

        self.t = 0
        self.user_idx = 0
        self.sigma = 0
        self.prev_sat = -1
        self.prev_rate_bps = 0.0

    def _refresh_effective_access_capacity(self) -> None:
        """根据卫星接入资源容量 L 计算共享后的星地链路 C^g。"""
        users = slice(0, self.num_users)
        self.visible_load = self.visible_all[:, users, :].sum(axis=1).astype(np.float32)  # [T,N]
        effective_load = np.maximum(self.visible_load, 1.0)
        share = np.minimum(1.0, self.cfg.access_channels_per_sat / effective_load)
        self.cg_all = np.minimum(self.cg_raw_all * share[:, None, :], self.cfg.tau_max_bps).astype(np.float32)
        self.max_visible_load = float(np.max(effective_load))

    def _sample_sigma(self) -> int:
        if self.fixed_sigma is not None:
            return int(self.fixed_sigma)
        return int(self.rng.random() < self.sensitive_ratio)

    def reset(self, user_idx: Optional[int] = None, sigma: Optional[int] = None) -> np.ndarray:
        self.t = 0
        self.user_idx = int(user_idx if user_idx is not None else self.rng.integers(0, self.num_users))
        self.sigma = int(sigma if sigma is not None else self._sample_sigma())
        vis = self.visible_all[self.t, self.user_idx]
        # 初始连接采用最大可见时长，模拟初始化阶段保守接入。
        self.prev_sat = masked_argmax(self.remaining_all[self.t, self.user_idx], vis)
        self.prev_rate_bps = self._end_to_end_rate(self.t, self.user_idx, self.prev_sat)[0]
        return self._make_state()

    def _user_gateway(self, user_idx: int) -> int:
        return int(self.user_gateway_idx_all[user_idx])

    def _path_metrics(self, t: int, user_idx: int, sat: int) -> tuple[float, float, int]:
        g = self._user_gateway(user_idx)
        q = float(self.path_quality_all[t, sat, g])
        hops = float(self.hop_count_all[t, sat, g])
        next_hop = int(self.next_hop_all[t, sat, g])
        return q, hops, next_hop

    def _isl_limit(self, t: int, user_idx: int, sat: int) -> float:
        q, hops, _ = self._path_metrics(t, user_idx, sat)
        if hops <= 0.0:
            return float(self.cfg.feeder_capacity_bps)
        avg_load = np.clip(q / max(hops, 1.0), 0.0, 0.98)
        # 路径负载越大、跳数越多，可用星间瓶颈速率越低。
        return float(self.cfg.isl_capacity_bps * max(0.04, 1.0 - avg_load) / (1.0 + 0.05 * hops))

    def _end_to_end_rate(self, t: int, user_idx: int, sat: int) -> tuple[float, float, float, float, int]:
        cg = float(self.cg_all[t, user_idx, sat])
        isl = self._isl_limit(t, user_idx, sat)
        q, hops, next_hop = self._path_metrics(t, user_idx, sat)
        tau = min(cg, isl, self.cfg.tau_max_bps)
        return tau, cg, q, hops, next_hop

    def _make_state(self) -> np.ndarray:
        g = self._user_gateway(self.user_idx)
        prev = np.zeros(self.sat_num, dtype=np.float32)
        if 0 <= self.prev_sat < self.sat_num:
            prev[self.prev_sat] = 1.0

        cg = self.cg_all[self.t, self.user_idx] / max(self.cfg.tau_max_bps, 1.0)
        hops = self.hop_count_all[self.t, :, g] / max(float(self.cfg.max_hops), 1.0)
        q = self.path_quality_all[self.t, :, g] / self.q_scale
        prev_rate = np.array([self.prev_rate_bps / max(self.cfg.tau_max_bps, 1.0)], dtype=np.float32)
        sigma = np.array([float(self.sigma)], dtype=np.float32)

        return np.concatenate([
            prev,
            np.clip(cg, 0.0, 1.5).astype(np.float32),
            np.clip(hops, 0.0, 2.0).astype(np.float32),
            np.clip(q, 0.0, 2.0).astype(np.float32),
            np.clip(prev_rate, 0.0, 1.5),
            sigma,
        ]).astype(np.float32)

    def action_mask(self) -> np.ndarray:
        return self.visible_all[self.t, self.user_idx].astype(bool)

    def step(self, action: int):
        action = int(action)
        vis = self.visible_all[self.t, self.user_idx]
        invalid = not (0 <= action < self.sat_num and bool(vis[action]))
        if invalid:
            selected = masked_argmax(self.cg_all[self.t, self.user_idx], vis)
            invalid_penalty = 1.0
        else:
            selected = action
            invalid_penalty = 0.0

        tau, cg, q, hops, next_hop = self._end_to_end_rate(self.t, self.user_idx, selected)
        handover = 1.0 if selected != self.prev_sat else 0.0

        r_tau = tau / max(self.cfg.tau_max_bps, 1.0)
        r_d = 1.0 - min(hops, float(self.cfg.max_hops)) / max(float(self.cfg.max_hops), 1.0)
        r_h = 1.0 - handover
        if self.sigma == 1:
            w1, w2, w3 = self.cfg.reward_sensitive
        else:
            w1, w2, w3 = self.cfg.reward_tolerant

        local_reward = w1 * r_tau + w2 * r_d + w3 * r_h
        q_norm = min(q / self.q_scale, 2.0)
        path_penalty = self.cfg.path_quality_reward_weight * q_norm

        # 集中式训练中的全局影响项用卫星负载近似：越少拥塞，越有正向全局收益。
        load_norm = float(self.visible_load[self.t, selected] / max(self.max_visible_load, 1.0))
        global_term = (1.0 - load_norm) - 0.5
        reward = local_reward - path_penalty + self.cfg.global_reward_weight * w1 * global_term - invalid_penalty

        self.prev_sat = selected
        self.prev_rate_bps = tau
        self.t += 1
        done = self.t >= min(self.t_num, self.cfg.max_steps_per_episode)
        next_state = np.zeros(self.state_dim, dtype=np.float32) if done else self._make_state()
        info = {
            "selected_sat": selected,
            "next_hop": next_hop,
            "invalid": invalid,
            "handover": handover,
            "throughput_bps": tau,
            "tau_bps": tau,
            "cg_bps": cg,
            "snr_db": float(self.snr_db_all[self.t - 1, self.user_idx, selected]),
            "hop_count": hops,
            "path_quality": q,
            "r_tau": r_tau,
            "r_d": r_d,
            "r_h": r_h,
            "sigma": self.sigma,
            "user_idx": self.user_idx,
            "gateway_idx": self._user_gateway(self.user_idx),
        }
        return next_state, float(reward), done, info
