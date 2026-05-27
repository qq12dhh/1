"""生成Dueling-DDQN 业务感知切换数据集。

数据集中同时包含：
1) 星地接入链路：可见性、距离、仰角、SNR、星地链路容量 C^g；
2) 星间回传链路：每个地面站网段对应的 next hop、路径质量 Q、跳数 d；
3) 用户目的网段：每个用户终端映射到最近的地面信关站网段。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from channel_model import compute_link_metrics
from config_ch4 import DATASET_DIR, SimConfig, ensure_dirs, save_config
from leo_geometry import (
    assign_users_to_gateways,
    eci_to_ecef,
    gateway_ring,
    lla_to_ecef,
    random_points_near,
    remaining_visible_time,
    satellite_positions_eci,
    visibility_and_geometry,
    ensure_at_least_one_visible,
)
from path_table import compute_path_tables
from utils import save_npz, seed_everything


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
    gs_ecef = lla_to_ecef(gs_lat, gs_lon)
    user_gateway_idx = assign_users_to_gateways(user_ecef, gs_ecef)

    print("[1/4] generating Walker-like satellite geometry ...")
    sat_eci = satellite_positions_eci(cfg, times_s)
    sat_ecef = eci_to_ecef(sat_eci, times_s)

    print("[2/4] computing user-satellite visibility and channel metrics ...")
    visible, distance_m, elevation_deg = visibility_and_geometry(sat_ecef, user_ecef, cfg.min_elevation_deg)
    visible = ensure_at_least_one_visible(visible, elevation_deg)
    rem_time_s = remaining_visible_time(visible, cfg.slot_seconds)
    metrics = compute_link_metrics(distance_m, visible, cfg, rng)

    print("[3/4] computing ISL path information table and broadcast values ...")
    path_tables = compute_path_tables(sat_ecef, gs_ecef, cfg, rng)

    output = output or DATASET_DIR / (
        f"ch4_dataset_{cfg.num_sats}sats_{cfg.user_max}users_"
        f"{cfg.bc_hz/1e6:.0f}MHz_{cfg.slot_seconds}s.npz"
    )
    print("[4/4] saving dataset ...")
    save_npz(
        output,
        times_s=times_s.astype(np.float32),
        user_lat=user_lat.astype(np.float32),
        user_lon=user_lon.astype(np.float32),
        gs_lat=gs_lat.astype(np.float32),
        gs_lon=gs_lon.astype(np.float32),
        user_gateway_idx=user_gateway_idx.astype(np.int16),
        sat_ecef=sat_ecef.astype(np.float32),
        visible=visible,
        distance_m=distance_m,
        elevation_deg=elevation_deg,
        remaining_time_s=rem_time_s.astype(np.float32),
        **metrics,
        **path_tables,
    )
    print(f"[OK] dataset saved: {output}")
    print(
        f"     slots={cfg.num_slots}, users={cfg.user_max}, sats={cfg.num_sats}, "
        f"GS={cfg.num_ground_stations}, Bc={cfg.bc_hz/1e6:.1f} MHz, "
        f"path_update={cfg.path_update_interval_s}s"
    )
    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bc-mhz", type=float, default=10.0, choices=[10.0, 20.0])
    p.add_argument("--users", type=int, default=300)
    p.add_argument("--sats", type=int, default=256)
    p.add_argument("--minutes", type=int, default=20)
    p.add_argument("--slot-seconds", type=int, default=SimConfig().slot_seconds)
    p.add_argument("--path-update-seconds", type=int, default=SimConfig().path_update_interval_s)
    p.add_argument("--output", type=str, default="")
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
        path_update_interval_s=args.path_update_seconds,
    )
    build_dataset(cfg, Path(args.output) if args.output else None)
