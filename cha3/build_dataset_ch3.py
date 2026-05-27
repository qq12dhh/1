"""生成数据集。

默认生成：
- 256 颗 LEO 卫星
- 300 个用户
- 20 分钟，1 秒时隙，共 1200 个时隙

为了训练速度和可重复性，本脚本使用内置 Walker-like 圆轨道几何模型生成
可见性、距离、仰角和剩余可见时长，再调用 channel_model 计算 SNR、ER、ET。

"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from channel_model import compute_link_metrics
from config_ch3 import DATASET_DIR, SimConfig, ensure_dirs, save_config
from utils import safe_norm, save_npz, seed_everything

MU = 3.986004418e14
R_EARTH = 6371e3
OMEGA_EARTH = 7.2921159e-5


def lla_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: float | np.ndarray = 0.0) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    r = R_EARTH + alt_m
    return np.stack([
        r * np.cos(lat) * np.cos(lon),
        r * np.cos(lat) * np.sin(lon),
        r * np.sin(lat),
    ], axis=-1)


def random_points_near(center_lat: float, center_lon: float, radius_km: float, n: int,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    r = radius_km * np.sqrt(rng.random(n))
    theta = 2 * np.pi * rng.random(n)
    north_km = r * np.cos(theta)
    east_km = r * np.sin(theta)
    lat = center_lat + north_km / 111.0
    lon = center_lon + east_km / (111.0 * np.cos(np.deg2rad(center_lat)))
    return lat, lon


def gateway_ring(center_lat: float, center_lon: float, radius_km: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    north_km = radius_km * np.cos(theta)
    east_km = radius_km * np.sin(theta)
    lat = center_lat + north_km / 111.0
    lon = center_lon + east_km / (111.0 * np.cos(np.deg2rad(center_lat)))
    return lat, lon


def satellite_positions_eci(cfg: SimConfig, times_s: np.ndarray) -> np.ndarray:
    a = R_EARTH + cfg.altitude_km * 1000.0
    inc = np.deg2rad(cfg.inclination_deg)
    mean_motion = np.sqrt(MU / a ** 3)
    positions = np.zeros((len(times_s), cfg.num_sats, 3), dtype=np.float64)

    sat_idx = 0
    for p in range(cfg.num_planes):
        raan = 2 * np.pi * p / cfg.num_planes
        for s in range(cfg.sats_per_plane):
            if sat_idx >= cfg.num_sats:
                break
            u0 = 2 * np.pi * (s / cfg.sats_per_plane + cfg.phase_factor * p / cfg.num_sats)
            u = u0 + mean_motion * times_s
            cu, su = np.cos(u), np.sin(u)
            co, so = np.cos(raan), np.sin(raan)
            ci, si = np.cos(inc), np.sin(inc)
            positions[:, sat_idx, 0] = a * (co * cu - so * su * ci)
            positions[:, sat_idx, 1] = a * (so * cu + co * su * ci)
            positions[:, sat_idx, 2] = a * su * si
            sat_idx += 1
    return positions


def eci_to_ecef(pos_eci: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    out = np.empty_like(pos_eci)
    for ti, t in enumerate(times_s):
        th = OMEGA_EARTH * t
        c, s = np.cos(th), np.sin(th)
        x = pos_eci[ti, :, 0]
        y = pos_eci[ti, :, 1]
        out[ti, :, 0] = c * x + s * y
        out[ti, :, 1] = -s * x + c * y
        out[ti, :, 2] = pos_eci[ti, :, 2]
    return out


def visibility_and_geometry(sat_ecef: np.ndarray, user_ecef: np.ndarray,
                            min_elev_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_num, sat_num, _ = sat_ecef.shape
    user_num = user_ecef.shape[0]
    visible = np.zeros((t_num, user_num, sat_num), dtype=bool)
    distance = np.zeros((t_num, user_num, sat_num), dtype=np.float32)
    elevation = np.zeros((t_num, user_num, sat_num), dtype=np.float32)
    user_up = user_ecef / safe_norm(user_ecef, axis=1, keepdims=True)

    for ti in range(t_num):
        rel = sat_ecef[ti][None, :, :] - user_ecef[:, None, :]
        dist = safe_norm(rel, axis=2)
        los = rel / dist[:, :, None]
        sin_el = np.sum(los * user_up[:, None, :], axis=2)
        el = np.rad2deg(np.arcsin(np.clip(sin_el, -1.0, 1.0)))
        visible[ti] = el >= min_elev_deg
        distance[ti] = dist.astype(np.float32)
        elevation[ti] = el.astype(np.float32)
    return visible, distance, elevation


def ensure_at_least_one_visible(visible: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    out = visible.copy()
    none = ~np.any(out, axis=2)
    if np.any(none):
        best = np.argmax(elevation, axis=2)
        ts, us = np.where(none)
        out[ts, us, best[ts, us]] = True
    return out


def remaining_visible_time(visible: np.ndarray, slot_seconds: int) -> np.ndarray:
    t_num, user_num, sat_num = visible.shape
    rem = np.zeros((t_num, user_num, sat_num), dtype=np.float32)
    rem[-1] = visible[-1]
    for t in range(t_num - 2, -1, -1):
        rem[t] = np.where(visible[t], rem[t + 1] + 1.0, 0.0)
    return rem * slot_seconds


def build_dataset(cfg: SimConfig, output: Path | None = None) -> Path:
    seed_everything(cfg.seed)
    ensure_dirs()
    save_config(cfg)
    rng = np.random.default_rng(cfg.seed)

    times_s = np.arange(cfg.num_slots, dtype=np.float64) * cfg.slot_seconds
    user_lat, user_lon = random_points_near(
        cfg.center_lat_deg, cfg.center_lon_deg, cfg.user_area_radius_km, cfg.user_max, rng
    )
    gs_lat, gs_lon = gateway_ring(
        cfg.center_lat_deg, cfg.center_lon_deg, cfg.gateway_ring_radius_km, cfg.num_ground_stations
    )
    user_ecef = lla_to_ecef(user_lat, user_lon)
    sat_eci = satellite_positions_eci(cfg, times_s)
    sat_ecef = eci_to_ecef(sat_eci, times_s)

    visible, distance_m, elevation_deg = visibility_and_geometry(sat_ecef, user_ecef, cfg.min_elevation_deg)
    visible = ensure_at_least_one_visible(visible, elevation_deg)
    rem_time_s = remaining_visible_time(visible, cfg.slot_seconds)
    metrics = compute_link_metrics(distance_m, visible, cfg, rng)

    output = output or DATASET_DIR / (
        f"ch3_dataset_{cfg.num_sats}sats_{cfg.user_max}users_"
        f"{cfg.bc_hz/1e6:.0f}MHz_{cfg.slot_seconds}s.npz"
    )
    save_npz(
        output,
        times_s=times_s.astype(np.float32),
        user_lat=user_lat.astype(np.float32),
        user_lon=user_lon.astype(np.float32),
        gs_lat=gs_lat.astype(np.float32),
        gs_lon=gs_lon.astype(np.float32),
        sat_ecef=sat_ecef.astype(np.float32),
        visible=visible,
        distance_m=distance_m,
        elevation_deg=elevation_deg,
        remaining_time_s=rem_time_s.astype(np.float32),
        **metrics,
    )
    print(f"[OK] dataset saved: {output}")
    print(f"     slots={cfg.num_slots}, users={cfg.user_max}, sats={cfg.num_sats}, Bc={cfg.bc_hz/1e6:.1f} MHz")
    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bc-mhz", type=float, default=10.0, choices=[10.0, 20.0])
    p.add_argument("--users", type=int, default=300)
    p.add_argument("--sats", type=int, default=256)
    p.add_argument("--minutes", type=int, default=20)
    p.add_argument("--slot-seconds", type=int, default=SimConfig().slot_seconds)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sats == 256:
        planes, sats_per_plane = 16, 16
    else:
        planes = 1
        sats_per_plane = args.sats
    cfg = SimConfig(
        bc_hz=args.bc_mhz * 1e6,
        user_max=args.users,
        num_sats=args.sats,
        num_planes=planes,
        sats_per_plane=sats_per_plane,
        sim_minutes=args.minutes,
        slot_seconds=args.slot_seconds,
    )
    build_dataset(cfg)
