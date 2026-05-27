"""查看第四章数据集中的星间路径信息表。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config_ch4 import DATASET_DIR, SimConfig
from path_table import format_route_rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="")
    p.add_argument("--bc-mhz", type=float, default=10.0)
    p.add_argument("--slot-seconds", type=int, default=SimConfig().slot_seconds)
    p.add_argument("--time-index", type=int, default=0)
    p.add_argument("--sat", type=int, default=0, help="0-based satellite index")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = SimConfig(bc_hz=args.bc_mhz * 1e6, slot_seconds=args.slot_seconds)
    dataset = Path(args.dataset) if args.dataset else DATASET_DIR / (
        f"ch4_dataset_{cfg.num_sats}sats_{cfg.user_max}users_"
        f"{cfg.bc_hz/1e6:.0f}MHz_{cfg.slot_seconds}s.npz"
    )
    if not dataset.exists():
        raise FileNotFoundError(f"dataset not found: {dataset}. 请先运行 build_dataset_ch4.py")
    data = np.load(dataset, allow_pickle=False)
    t = min(max(args.time_index, 0), data["path_quality"].shape[0] - 1)
    sat = min(max(args.sat, 0), data["path_quality"].shape[1] - 1)
    rows = format_route_rows(data["route_next_hop"][t], data["hop_count"][t], data["path_quality"][t], sat)
    print(f"数据集: {dataset}")
    print(f"时隙 t={t}, 卫星 S{sat:03d} 的路径信息表：")
    print("目的网段\t下一跳卫星\t路径质量Q\t跳数d")
    for r in rows:
        print(f"{r['destination_segment']}\t{r['next_hop']}\t\t{r['path_quality']:.4f}\t\t{r['hop_count']}")
