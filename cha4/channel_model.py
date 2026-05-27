"""星地信道模型：SNR、星地链路容量 C^g 计算。"""
from __future__ import annotations

import numpy as np

from config_ch4 import SimConfig
from utils import db_to_linear, dbm_to_watt

C_LIGHT = 299_792_458.0


def rician_fading(shape, k_db: float = 10.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """生成单位平均功率的 Rician 小尺度衰落功率增益。"""
    rng = rng or np.random.default_rng()
    k = db_to_linear(k_db)
    s = np.sqrt(k / (k + 1.0))
    sigma = np.sqrt(1.0 / (2.0 * (k + 1.0)))
    real = s + sigma * rng.standard_normal(shape)
    imag = sigma * rng.standard_normal(shape)
    return real * real + imag * imag


def noise_power_watt(bandwidth_hz: float, cfg: SimConfig) -> float:
    return float(dbm_to_watt(cfg.noise_density_dbm_per_hz) * bandwidth_hz)


def free_space_channel_gain(distance_m: np.ndarray, cfg: SimConfig, fading=None) -> np.ndarray:
    d = np.maximum(distance_m.astype(np.float64), 1.0)
    fs_gain = (C_LIGHT / (4.0 * np.pi * cfg.carrier_freq_hz * d)) ** 2
    atmospheric = db_to_linear(-cfg.atmospheric_loss_db)
    if fading is None:
        fading = 1.0
    return fs_gain * atmospheric * fading


def snr_linear(distance_m: np.ndarray, cfg: SimConfig, fading=None) -> np.ndarray:
    gain = free_space_channel_gain(distance_m, cfg, fading=fading)
    link_budget = db_to_linear(cfg.pt_gt_gr_db)
    noise = noise_power_watt(cfg.bc_hz, cfg)
    return gain * link_budget / max(noise, 1e-30)


def compute_link_metrics(distance_m: np.ndarray, visible: np.ndarray, cfg: SimConfig,
                         rng: np.random.Generator | None = None) -> dict:
    """输入 shape 均为 [T, U, N]，输出星地链路容量 C^g 等指标。"""
    rng = rng or np.random.default_rng(cfg.seed)
    fading = rician_fading(distance_m.shape, cfg.rice_k_db, rng)
    snr = snr_linear(distance_m, cfg, fading=fading)
    cg = cfg.bc_hz * np.log2(1.0 + snr)
    cg = np.where(visible, cg, 0.0)
    return {
        "snr_linear": snr.astype(np.float32),
        "snr_db": (10.0 * np.log10(np.maximum(snr, 1e-30))).astype(np.float32),
        "cg_bps": cg.astype(np.float32),
    }
