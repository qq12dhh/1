"""第三章复现一键脚本。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config_ch3 import SimConfig

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd):
    print("\n>>> " + " ".join(str(x) for x in cmd))
    subprocess.check_call([str(x) for x in cmd], cwd=ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bc-mhz", type=float, default=10.0, choices=[10.0, 20.0])
    p.add_argument("--slot-seconds", type=int, default=SimConfig().slot_seconds)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--train-users", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    run([PY, "build_dataset_ch3.py", "--bc-mhz", args.bc_mhz, "--minutes", 20, "--slot-seconds", args.slot_seconds, "--users", 300, "--sats", 256])
    if not args.skip_train:
        run([PY, "train_ch3.py", "--bc-mhz", args.bc_mhz, "--slot-seconds", args.slot_seconds, "--episodes", args.episodes, "--users", args.train_users, "--algo", "both"])
    run([PY, "eval_ch3.py", "--bc-mhz", args.bc_mhz, "--slot-seconds", args.slot_seconds, "--eval-episodes", args.eval_episodes])
    run([PY, "plot_ch3.py", "--bc-mhz", args.bc_mhz])
    print("\n[OK] Chapter 3 reproduction pipeline finished.")


if __name__ == "__main__":
    main()
