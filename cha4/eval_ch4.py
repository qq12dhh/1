"""评估第四章 Dueling-DDQN 与基准算法。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np

from agents import DQNAgent
from baselines import max_capacity_action, max_visible_time_action, min_hop_action, min_path_quality_action
from config_ch4 import DATASET_DIR, LOG_DIR, MODEL_DIR, SimConfig, ensure_dirs
from env_ch4 import Ch4HandoverEnv
from utils import seed_everything


def rollout_env(name: str, env: Ch4HandoverEnv, policy: Callable, cfg: SimConfig) -> dict:
    rewards, handovers, rates, cgs, hops, qualities = [], [], [], [], [], []
    for ep in range(cfg.eval_episodes):
        state = env.reset(user_idx=ep % env.num_users)
        done = False
        while not done:
            mask = env.action_mask()
            action = policy(state, mask)
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            handovers.append(info["handover"])
            rates.append(info["throughput_bps"] / 1e6)
            cgs.append(info["cg_bps"] / 1e6)
            hops.append(info["hop_count"])
            qualities.append(info["path_quality"])
    return {
        "algo": name,
        "reward": float(np.mean(rewards)),
        "handover_rate": float(np.mean(handovers)),
        "throughput_mbps": float(np.mean(rates)),
        "avg_cg_mbps": float(np.mean(cgs)),
        "avg_hops": float(np.mean(hops)),
        "avg_path_quality": float(np.mean(qualities)),
    }


def eval_rl(name: str, model_path: Path, dataset: Path, cfg: SimConfig, num_users: int,
            sensitive_ratio: float, fixed_sigma=None, network: str = "dueling", ddqn: bool = True) -> dict:
    env = Ch4HandoverEnv(dataset, cfg, num_users=num_users, sensitive_ratio=sensitive_ratio,
                         fixed_sigma=fixed_sigma, seed=cfg.seed + 100)
    agent = DQNAgent(env.state_dim, env.action_dim, cfg, ddqn=ddqn, network=network)
    agent.load(model_path)
    agent.epsilon = 0.0
    return rollout_env(name, env, lambda state, mask: agent.select_action(state, mask, explore=False), cfg)


def eval_baseline(name: str, dataset: Path, cfg: SimConfig, num_users: int,
                  sensitive_ratio: float, fixed_sigma=None) -> dict:
    env = Ch4HandoverEnv(dataset, cfg, num_users=num_users, sensitive_ratio=sensitive_ratio,
                         fixed_sigma=fixed_sigma, seed=cfg.seed + 200)

    def policy(state, mask):
        t = env.t
        u = env.user_idx
        g = env._user_gateway(u)
        if name == "max_capacity":
            return max_capacity_action(env.cg_all[t, u], env.visible_all[t, u], env.prev_sat)
        if name == "max_visible_time":
            return max_visible_time_action(env.remaining_all[t, u], env.visible_all[t, u], env.prev_sat)
        if name == "min_hop":
            return min_hop_action(env.hop_count_all[t, :, g], env.visible_all[t, u], env.prev_sat)
        if name == "min_path_quality":
            return min_path_quality_action(env.path_quality_all[t, :, g], env.visible_all[t, u], env.prev_sat)
        raise ValueError(name)

    return rollout_env(name, env, policy, cfg)


def _eval_algo(algo: str, dataset: Path, cfg: SimConfig, n: int, ratio: float, fixed_sigma=None) -> dict | None:
    mhz = int(cfg.bc_hz / 1e6)
    if algo == "dueling_ddqn":
        model = MODEL_DIR / f"dueling_ddqn_{mhz}MHz.pt"
        if not model.exists():
            print(f"[WARN] missing model: {model}; skip {algo}")
            return None
        return eval_rl(algo, model, dataset, cfg, n, ratio, fixed_sigma=fixed_sigma, network="dueling", ddqn=True)
    if algo == "ddqn":
        model = MODEL_DIR / f"ddqn_{mhz}MHz.pt"
        if not model.exists():
            print(f"[WARN] missing model: {model}; skip {algo}")
            return None
        return eval_rl(algo, model, dataset, cfg, n, ratio, fixed_sigma=fixed_sigma, network="plain", ddqn=True)
    if algo == "dqn":
        model = MODEL_DIR / f"dqn_{mhz}MHz.pt"
        if not model.exists():
            print(f"[WARN] missing model: {model}; skip {algo}")
            return None
        return eval_rl(algo, model, dataset, cfg, n, ratio, fixed_sigma=fixed_sigma, network="plain", ddqn=False)
    return eval_baseline(algo, dataset, cfg, n, ratio, fixed_sigma=fixed_sigma)


def run_eval(dataset: Path, cfg: SimConfig, include_surface: bool = False, include_extra_baselines: bool = False) -> Path:
    ensure_dirs()
    mhz = int(cfg.bc_hz / 1e6)
    out = LOG_DIR / f"eval_ch4_{mhz}MHz.csv"
    rows = []
    algos = ["dueling_ddqn", "ddqn", "max_capacity", "max_visible_time"]
    if include_extra_baselines:
        algos += ["min_hop", "min_path_quality"]

    # 图 4-6：固定 200 用户，改变时敏业务占比。
    for ratio in cfg.sensitive_ratios:
        for algo in algos:
            r = _eval_algo(algo, dataset, cfg, cfg.num_users, ratio, fixed_sigma=None)
            if r is None:
                continue
            r.update({"scene": "ratio", "num_users": cfg.num_users, "sensitive_ratio": ratio})
            rows.append(r)
            print("ratio", ratio, algo, r)

    # 图 4-5：终端数量变化后的适应性，用 50% 混合业务比例。
    for n in cfg.user_counts:
        for algo in algos:
            r = _eval_algo(algo, dataset, cfg, n, 0.5, fixed_sigma=None)
            if r is None:
                continue
            r.update({"scene": "users", "num_users": n, "sensitive_ratio": 0.5})
            rows.append(r)
            print("users", n, algo, r)

    # 可选：生成用户数 × 时敏占比曲面数据。
    if include_surface:
        for ratio in cfg.sensitive_ratios:
            for n in cfg.user_counts:
                for algo in algos:
                    r = _eval_algo(algo, dataset, cfg, n, ratio, fixed_sigma=None)
                    if r is None:
                        continue
                    r.update({"scene": "surface", "num_users": n, "sensitive_ratio": ratio})
                    rows.append(r)
                    print("surface", ratio, n, algo, r)

    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "scene", "num_users", "sensitive_ratio", "algo", "reward", "handover_rate",
            "throughput_mbps", "avg_cg_mbps", "avg_hops", "avg_path_quality"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] saved eval: {out}")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="")
    p.add_argument("--bc-mhz", type=float, default=10.0, choices=[10.0, 20.0])
    p.add_argument("--slot-seconds", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--surface", action="store_true", help="同时输出用户数×时敏占比曲面数据")
    p.add_argument("--extra-baselines", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_kwargs = dict(bc_hz=args.bc_mhz * 1e6, eval_episodes=args.eval_episodes, num_users=args.users)
    if args.slot_seconds is not None:
        cfg_kwargs["slot_seconds"] = args.slot_seconds
    cfg = SimConfig(**cfg_kwargs)
    seed_everything(cfg.seed)
    dataset = Path(args.dataset) if args.dataset else DATASET_DIR / (
        f"ch4_dataset_{cfg.num_sats}sats_{cfg.user_max}users_"
        f"{cfg.bc_hz/1e6:.0f}MHz_{cfg.slot_seconds}s.npz"
    )
    if not dataset.exists():
        raise FileNotFoundError(f"dataset not found: {dataset}. 请先运行 build_dataset_ch4.py")
    run_eval(dataset, cfg, include_surface=args.surface, include_extra_baselines=args.extra_baselines)
