"""第四章：基于 Dueling-DDQN 的业务感知低轨卫星链路切换复现配置。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple
import json

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "ch4"
DATASET_DIR = OUTPUT_DIR / "dataset"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
FIGURE_DIR = OUTPUT_DIR / "figures"


@dataclass
class SimConfig:
    # -------- 基础随机与仿真时间 --------
    seed: int = 42
    sim_minutes: int = 20
    slot_seconds: int = 5

    # -------- 星座参数：论文表 4-1 --------
    num_sats: int = 256
    num_planes: int = 16
    sats_per_plane: int = 16
    altitude_km: float = 1000.0
    inclination_deg: float = 54.0
    phase_factor: int = 1
    min_elevation_deg: float = 5.0

    # -------- 用户与地面站 --------
    num_users: int = 200          # 第四章收敛实验默认 200 个终端
    user_min: int = 50
    user_max: int = 300
    user_step: int = 50
    num_ground_stations: int = 8
    center_lat_deg: float = 29.563
    center_lon_deg: float = 106.551
    user_area_radius_km: float = 250.0
    gateway_ring_radius_km: float = 450.0

    # -------- 星地链路参数：论文表 4-1 --------
    carrier_freq_hz: float = 20e9
    bc_hz: float = 10e6
    sat_total_bandwidth_hz: float = 200e6
    tau_max_bps: float = 500e6
    pt_gt_gr_db: float = 80.0
    atmospheric_loss_db: float = 2.9
    noise_density_dbm_per_hz: float = -173.0
    rice_k_db: float = 10.0

    # -------- 星间/馈电链路参数 --------
    isl_capacity_bps: float = 2.5e9        # 表 4-1：星间链路容量 C^s
    feeder_capacity_bps: float = 2.5e9
    path_update_interval_s: int = 5        # 图片中的“路径质量广播更新”秒级周期
    landing_sats_per_gateway: int = 1      # active landing satellites per gateway segment
    link_load_base: float = 0.18
    link_load_variation: float = 0.45
    link_load_noise: float = 0.04
    path_quality_reward_weight: float = 0.03
    global_reward_weight: float = 0.05

    # -------- 第四章差异化奖励权重：论文表 4-1 --------
    # r = w1 * 速率子奖励 + w2 * 跳数子奖励 + w3 * 切换开销子奖励
    reward_sensitive: Tuple[float, float, float] = (0.65, 0.25, 0.10)
    reward_tolerant: Tuple[float, float, float] = (0.20, 0.20, 0.60)

    # -------- 深度强化学习参数 --------
    gamma: float = 0.99
    lr: float = 1e-3
    tau: float = 0.005
    epsilon_init: float = 0.5
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9995
    replay_size: int = 200_000
    batch_size: int = 512
    hidden_dim: int = 256
    train_episodes: int = 300
    warmup_steps: int = 2048

    # -------- 评估参数 --------
    eval_episodes: int = 3
    sensitive_ratios: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    @property
    def num_slots(self) -> int:
        return int(self.sim_minutes * 60 / self.slot_seconds)

    @property
    def max_steps_per_episode(self) -> int:
        return self.num_slots

    @property
    def user_counts(self) -> Tuple[int, ...]:
        return tuple(range(self.user_min, self.user_max + 1, self.user_step))

    @property
    def max_hops(self) -> int:
        # 论文式(4-24)：轨道数量与轨内卫星数量之和的一半
        return int((self.num_planes + self.sats_per_plane) // 2)

    @property
    def access_channels_per_sat(self) -> int:
        return max(1, int(self.sat_total_bandwidth_hz // self.bc_hz))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["num_slots"] = self.num_slots
        d["user_counts"] = self.user_counts
        d["max_hops"] = self.max_hops
        d["access_channels_per_sat"] = self.access_channels_per_sat
        return d


def ensure_dirs() -> None:
    for p in (OUTPUT_DIR, DATASET_DIR, MODEL_DIR, LOG_DIR, FIGURE_DIR):
        p.mkdir(parents=True, exist_ok=True)


def save_config(cfg: SimConfig, path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or (OUTPUT_DIR / "config_ch4.json")
    path.write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
