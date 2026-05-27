"""一键运行第四章复现流程：数据集 -> 训练 -> 评估 -> 绘图。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config_ch4 import DATASET_DIR, SimConfig, ensure_dirs


def run(cmd: list[str]) -> None:
    print("\n>>> " + " ".join(cmd))
    subprocess.check_call(cmd)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bc-mhz", type=float, default=10.0, choices=[10.0, 20.0])
    p.add_argument("--slot-seconds", type=int, default=SimConfig().slot_seconds)
    p.add_argument("--episodes", type=int, default=SimConfig().train_episodes)
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--algo", choices=["dueling", "ddqn", "dqn", "all"], default="all")
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--surface", action="store_true", help="评估并绘制三维曲面数据")
    p.add_argument("--force-build", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_dirs()
    py = sys.executable
    cfg = SimConfig(bc_hz=args.bc_mhz * 1e6, slot_seconds=args.slot_seconds)
    dataset = DATASET_DIR / (
        f"ch4_dataset_{cfg.num_sats}sats_{cfg.user_max}users_"
        f"{cfg.bc_hz/1e6:.0f}MHz_{cfg.slot_seconds}s.npz"
    )
    if args.force_build or not dataset.exists():
        run([py, "build_dataset_ch4.py", "--bc-mhz", str(args.bc_mhz), "--slot-seconds", str(args.slot_seconds)])
    else:
        print(f"[SKIP] dataset exists: {dataset}")

    run([
        py, "train_ch4.py", "--bc-mhz", str(args.bc_mhz), "--slot-seconds", str(args.slot_seconds),
        "--episodes", str(args.episodes), "--users", str(args.users), "--algo", args.algo,
    ])

    eval_cmd = [
        py, "eval_ch4.py", "--bc-mhz", str(args.bc_mhz), "--slot-seconds", str(args.slot_seconds),
        "--eval-episodes", str(args.eval_episodes), "--users", str(args.users),
    ]
    if args.surface:
        eval_cmd.append("--surface")
    run(eval_cmd)

    plot_cmd = [py, "plot_ch4.py", "--bc-mhz", str(args.bc_mhz), "--users", str(args.users)]
    if args.surface:
        plot_cmd.append("--surface")
    run(plot_cmd)
