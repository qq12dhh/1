"""训练第三章 DQN / DDQN。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from agents import DQNAgent
from config_ch3 import DATASET_DIR, LOG_DIR, MODEL_DIR, SimConfig, ensure_dirs
from env_ch3 import HandoverEnv
from utils import seed_everything


def train_one(name: str, dataset: Path, cfg: SimConfig, ddqn: bool, sensitive_ratio: float = 0.5) -> Path:
    env = HandoverEnv(dataset, cfg, num_users=cfg.num_users, sensitive_ratio=sensitive_ratio, seed=cfg.seed)
    agent = DQNAgent(env.state_dim, env.action_dim, cfg, ddqn=ddqn)
    log_path = LOG_DIR / f"{name}_train.csv"
    model_path = MODEL_DIR / f"{name}.pt"

    with log_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "episode", "reward", "avg_reward", "avg50_reward", "avg100_reward",
            "loss", "epsilon", "handover_rate", "avg_et_mbps", "avg_er_mbps"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        recent_rewards = []
        all_rewards = []

        for ep in range(1, cfg.train_episodes + 1):
            state = env.reset()
            done = False
            total_reward = 0.0
            losses, handovers, ets, ers = [], [], [], []

            while not done:
                mask = env.action_mask()
                action = agent.select_action(state, mask, explore=True)
                next_state, reward, done, info = env.step(action)
                next_mask = env.action_mask() if not done else np.zeros(env.action_dim, dtype=bool)
                agent.replay.push(state, action, reward, next_state, done, mask, next_mask)
                stats = agent.update()
                if stats.loss > 0:
                    losses.append(stats.loss)
                total_reward += reward
                handovers.append(info["handover"])
                ets.append(info["et_bps"] / 1e6)
                ers.append(info["er_bps"] / 1e6)
                state = next_state

            ep_reward = total_reward / cfg.max_steps_per_episode
            recent_rewards.append(ep_reward)
            all_rewards.append(ep_reward)
            if len(recent_rewards) > 20:
                recent_rewards.pop(0)

            row = {
                "episode": ep,
                "reward": ep_reward,
                "avg_reward": float(np.mean(recent_rewards)),
                "avg50_reward": float(np.mean(all_rewards[-50:])),
                "avg100_reward": float(np.mean(all_rewards[-100:])),
                "loss": float(np.mean(losses)) if losses else 0.0,
                "epsilon": agent.epsilon,
                "handover_rate": float(np.mean(handovers)) if handovers else 0.0,
                "avg_et_mbps": float(np.mean(ets)) if ets else 0.0,
                "avg_er_mbps": float(np.mean(ers)) if ers else 0.0,
            }
            writer.writerow(row)
            if ep == 1 or ep % 10 == 0:
                print(
                    f"[{name}] ep={ep:04d} reward={row['reward']:.4f} "
                    f"avg20={row['avg_reward']:.4f} avg100={row['avg100_reward']:.4f} "
                    f"eps={agent.epsilon:.3f}"
                )

    agent.save(model_path)
    print(f"[OK] saved model: {model_path}")
    print(f"[OK] saved log: {log_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="")
    p.add_argument("--bc-mhz", type=float, default=10.0, choices=[10.0, 20.0])
    p.add_argument("--slot-seconds", type=int, default=None, help="时隙长度；默认使用 config_ch3.py 中的 slot_seconds")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--users", type=int, default=100)
    p.add_argument("--algo", choices=["ddqn", "dqn", "both"], default="both")
    p.add_argument("--lr", type=float, default=None, help="覆盖 config_ch3.py 中的学习率")
    p.add_argument("--epsilon-decay", type=float, default=None, help="覆盖 config_ch3.py 中的 epsilon_decay")
    p.add_argument("--warmup-steps", type=int, default=None, help="覆盖 config_ch3.py 中的 warmup_steps")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_dirs()
    cfg_kwargs = dict(bc_hz=args.bc_mhz * 1e6, train_episodes=args.episodes, num_users=args.users)
    if args.slot_seconds is not None:
        cfg_kwargs["slot_seconds"] = args.slot_seconds
    if args.lr is not None:
        cfg_kwargs["lr"] = args.lr
    if args.epsilon_decay is not None:
        cfg_kwargs["epsilon_decay"] = args.epsilon_decay
    if args.warmup_steps is not None:
        cfg_kwargs["warmup_steps"] = args.warmup_steps
    cfg = SimConfig(**cfg_kwargs)
    seed_everything(cfg.seed)
    dataset = Path(args.dataset) if args.dataset else DATASET_DIR / (
        f"ch3_dataset_{cfg.num_sats}sats_{cfg.user_max}users_"
        f"{cfg.bc_hz/1e6:.0f}MHz_{cfg.slot_seconds}s.npz"
    )
    if not dataset.exists():
        raise FileNotFoundError(f"dataset not found: {dataset}. 请先运行 build_dataset_ch3.py")
    if args.algo in ("ddqn", "both"):
        train_one(f"ddqn_{int(cfg.bc_hz/1e6)}MHz", dataset, cfg, ddqn=True)
    if args.algo in ("dqn", "both"):
        train_one(f"dqn_{int(cfg.bc_hz/1e6)}MHz", dataset, cfg, ddqn=False)
