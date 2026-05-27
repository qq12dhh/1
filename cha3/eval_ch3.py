"""评估第三章 DDQN/DQN/最大信噪比/最大可见时长策略。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np

from agents import DQNAgent
from baselines import max_snr_action, max_visible_time_action
from config_ch3 import DATASET_DIR, LOG_DIR, MODEL_DIR, SimConfig, ensure_dirs
from env_ch3 import HandoverEnv
from utils import seed_everything


def rollout_env(name: str, env: HandoverEnv, policy: Callable, cfg: SimConfig) -> dict:
    rewards, handovers, ets, ers = [], [], [], []
    for ep in range(cfg.eval_episodes):
        state = env.reset(user_idx=ep % env.num_users)
        done = False
        while not done:
            mask = env.action_mask()
            action = policy(state, mask)
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            handovers.append(info["handover"])
            ets.append(info["et_bps"] / 1e6)
            ers.append(info["er_bps"] / 1e6)
    return {
        "algo": name,
        "reward": float(np.mean(rewards)),
        "handover_rate": float(np.mean(handovers)),
        "et_mbps": float(np.mean(ets)),
        "er_mbps": float(np.mean(ers)),
    }


def eval_rl(name: str, model_path: Path, dataset: Path, cfg: SimConfig, num_users: int,
            sensitive_ratio: float, fixed_sigma=None, ddqn: bool = True) -> dict:
    env = HandoverEnv(dataset, cfg, num_users=num_users, sensitive_ratio=sensitive_ratio,
                      fixed_sigma=fixed_sigma, seed=cfg.seed + 100)
    agent = DQNAgent(env.state_dim, env.action_dim, cfg, ddqn=ddqn)
    agent.load(model_path)
    agent.epsilon = 0.0
    return rollout_env(name, env, lambda state, mask: agent.select_action(state, mask, explore=False), cfg)


def eval_baseline(name: str, dataset: Path, cfg: SimConfig, num_users: int,
                  sensitive_ratio: float, fixed_sigma=None) -> dict:
    env = HandoverEnv(dataset, cfg, num_users=num_users, sensitive_ratio=sensitive_ratio,
                      fixed_sigma=fixed_sigma, seed=cfg.seed + 200)

    def policy(state, mask):
        t = env.t
        u = env.user_idx
        if name == "max_snr":
            return max_snr_action(env.snr_db_all[t, u], env.visible_all[t, u], env.prev_sat)
        if name == "max_visible_time":
            return max_visible_time_action(env.remaining_all[t, u], env.visible_all[t, u], env.prev_sat)
        raise ValueError(name)

    return rollout_env(name, env, policy, cfg)


def run_eval(dataset: Path, cfg: SimConfig) -> Path:
    ensure_dirs()
    ddqn_model = MODEL_DIR / f"ddqn_{int(cfg.bc_hz/1e6)}MHz.pt"
    dqn_model = MODEL_DIR / f"dqn_{int(cfg.bc_hz/1e6)}MHz.pt"
    out = LOG_DIR / f"eval_{int(cfg.bc_hz/1e6)}MHz.csv"
    rows = []

    for scene, fixed_sigma, ratio in [("tolerant", 0, 0.0), ("sensitive", 1, 1.0)]:
        for n in cfg.user_counts:
            for algo in ["ddqn", "dqn", "max_snr", "max_visible_time"]:
                if algo == "ddqn":
                    if not ddqn_model.exists():
                        continue
                    r = eval_rl(algo, ddqn_model, dataset, cfg, n, ratio, fixed_sigma=fixed_sigma, ddqn=True)
                elif algo == "dqn":
                    if not dqn_model.exists():
                        continue
                    r = eval_rl(algo, dqn_model, dataset, cfg, n, ratio, fixed_sigma=fixed_sigma, ddqn=False)
                else:
                    r = eval_baseline(algo, dataset, cfg, n, ratio, fixed_sigma=fixed_sigma)
                r.update({"scene": scene, "num_users": n, "sensitive_ratio": ratio})
                rows.append(r)
                print(scene, n, algo, r)

    for ratio in cfg.sensitive_ratios:
        for n in cfg.user_counts:
            for algo in ["ddqn", "dqn", "max_snr", "max_visible_time"]:
                if algo == "ddqn":
                    if not ddqn_model.exists():
                        continue
                    r = eval_rl(algo, ddqn_model, dataset, cfg, n, ratio, fixed_sigma=None, ddqn=True)
                elif algo == "dqn":
                    if not dqn_model.exists():
                        continue
                    r = eval_rl(algo, dqn_model, dataset, cfg, n, ratio, fixed_sigma=None, ddqn=False)
                else:
                    r = eval_baseline(algo, dataset, cfg, n, ratio, fixed_sigma=None)
                r.update({"scene": "mixed", "num_users": n, "sensitive_ratio": ratio})
                rows.append(r)
                print("mixed", ratio, n, algo, r)

    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["scene", "num_users", "sensitive_ratio", "algo", "reward", "handover_rate", "et_mbps", "er_mbps"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] saved eval: {out}")
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="")
    p.add_argument("--bc-mhz", type=float, default=10.0, choices=[10.0, 20.0])
    p.add_argument("--slot-seconds", type=int, default=None, help="时隙长度；默认使用 config_ch3.py 中的 slot_seconds")
    p.add_argument("--eval-episodes", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_kwargs = dict(bc_hz=args.bc_mhz * 1e6, eval_episodes=args.eval_episodes)
    if args.slot_seconds is not None:
        cfg_kwargs["slot_seconds"] = args.slot_seconds
    cfg = SimConfig(**cfg_kwargs)
    seed_everything(cfg.seed)
    dataset = Path(args.dataset) if args.dataset else DATASET_DIR / (
        f"ch3_dataset_{cfg.num_sats}sats_{cfg.user_max}users_"
        f"{cfg.bc_hz/1e6:.0f}MHz_{cfg.slot_seconds}s.npz"
    )
    if not dataset.exists():
        raise FileNotFoundError(f"dataset not found: {dataset}")
    run_eval(dataset, cfg)
