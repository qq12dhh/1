"""DDQN 用户链路切换统一配置。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple
import json

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "ch3"
DATASET_DIR = OUTPUT_DIR / "dataset"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
FIGURE_DIR = OUTPUT_DIR / "figures"


@dataclass
class SimConfig:
    seed: int = 42

    # 仿真时长：20 分钟，时隙：1 秒
    sim_minutes: int = 20
    slot_seconds: int = 5

    # 星座参数
    num_sats: int = 256
    num_planes: int = 16
    sats_per_plane: int = 16
    altitude_km: float = 1000.0
    inclination_deg: float = 54.0
    phase_factor: int = 1
    min_elevation_deg: float = 5.0

    # 用户与地面站
    num_users: int = 100          # 训练时抽样用户数
    user_min: int = 50
    user_max: int = 300           # 数据集最大用户数
    user_step: int = 50
    num_ground_stations: int = 8
    center_lat_deg: float = 29.563
    center_lon_deg: float = 106.551
    user_area_radius_km: float = 250.0
    gateway_ring_radius_km: float = 450.0

    # 信道参数
    carrier_freq_hz: float = 20e9
    bc_hz: float = 10e6           # 可用命令行切换为 20e6
    # 卫星总带宽 B。论文未给具体数值；这里取 200 MHz 用于体现多用户资源竞争。
    sat_total_bandwidth_hz: float = 200e6
    tau_max_bps: float = 500e6
    pt_gt_gr_db: float = 80.0
    atmospheric_loss_db: float = 2.9
    noise_density_dbm_per_hz: float = -173.0
    rice_k_db: float = 10.0
    feeder_capacity_bps: float = 2e9
    snr_threshold_sensitive_db: float = 10.0
    snr_threshold_tolerant_db: float = 3.0

    # 奖励权重：ET, ER, 剩余可见时长, 切换惩罚, QoS 惩罚系数
    reward_sensitive: Tuple[float, float, float, float, float] = (0.5, 0.2, 0.1, 0.2, 2.0)
    reward_tolerant: Tuple[float, float, float, float, float] = (0.1, 0.2, 0.5, 0.2, 2.0)

    # 深度强化学习参数
    gamma: float = 0.99
    lr: float = 5e-4
    tau: float = 0.005
    epsilon_init: float = 0.5
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.99999
    replay_size: int = 200_000
    batch_size: int = 1024
    hidden_dim: int = 256
    train_episodes: int = 1000
    warmup_steps: int = 5000

    # 评估参数
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

    def to_dict(self) -> dict:
        d = asdict(self)
        d["num_slots"] = self.num_slots
        d["user_counts"] = self.user_counts
        return d


def ensure_dirs() -> None:
    for p in (OUTPUT_DIR, DATASET_DIR, MODEL_DIR, LOG_DIR, FIGURE_DIR):
        p.mkdir(parents=True, exist_ok=True)


def save_config(cfg: SimConfig, path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or (OUTPUT_DIR / "config_ch3.json")
    path.write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path

