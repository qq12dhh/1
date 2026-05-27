"""训练 Dueling-DDQN / DDQN / DQN。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from agents import DQNAgent
from config_ch4 import DATASET_DIR, LOG_DIR, MODEL_DIR, SimConfig, ensure_dirs
from env_ch4 import Ch4HandoverEnv
from utils import seed_everything


def train_one(
    name: str,
    dataset: Path,
    cfg: SimConfig,
    network: str,
    ddqn: bool,
    sensitive_ratio: float = 0.5,
    device: str | None = None,
) -> Path:
    env = Ch4HandoverEnv(dataset, cfg, num_users=cfg.num_users, sensitive_ratio=sensitive_ratio, seed=cfg.seed)
    agent = DQNAgent(env.state_dim, env.action_dim, cfg, ddqn=ddqn, network=network, device=device)
    log_path = LOG_DIR / f"{name}_train.csv"
    model_path = MODEL_DIR / f"{name}.pt"

    print(f"[{name}] device={agent.device}, state_dim={env.state_dim}, action_dim={env.action_dim}")
    with log_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "episode", "reward", "avg20_reward", "avg50_reward", "avg100_reward",
            "loss", "epsilon", "handover_rate", "throughput_mbps", "avg_hops", "avg_path_quality"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        recent_rewards: list[float] = []
        all_rewards: list[float] = []

        for ep in range(1, cfg.train_episodes + 1):
            state = env.reset()
            done = False
            total_reward = 0.0
            losses, handovers, rates, hops_list, q_list = [], [], [], [], []

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
                rates.append(info["throughput_bps"] / 1e6)
                hops_list.append(info["hop_count"])
                q_list.append(info["path_quality"])
                state = next_state

            ep_reward = total_reward / cfg.max_steps_per_episode
            recent_rewards.append(ep_reward)
            all_rewards.append(ep_reward)
            if len(recent_rewards) > 20:
                recent_rewards.pop(0)

            row = {
                "episode": ep,
                "reward": ep_reward,
                "avg20_reward": float(np.mean(recent_rewards)),
                "avg50_reward": float(np.mean(all_rewards[-50:])),
                "avg100_reward": float(np.mean(all_rewards[-100:])),
                "loss": float(np.mean(losses)) if losses else 0.0,
                "epsilon": agent.epsilon,
                "handover_rate": float(np.mean(handovers)) if handovers else 0.0,
                "throughput_mbps": float(np.mean(rates)) if rates else 0.0,
                "avg_hops": float(np.mean(hops_list)) if hops_list else 0.0,
                "avg_path_quality": float(np.mean(q_list)) if q_list else 0.0,
            }
            writer.writerow(row)
            f.flush()
            if ep == 1 or ep % 10 == 0:
                print(
                    f"[{name}] ep={ep:04d} reward={row['reward']:.4f} "
                    f"avg20={row['avg20_reward']:.4f} avg100={row['avg100_reward']:.4f} "
                    f"eps={agent.epsilon:.3f} rate={row['throughput_mbps']:.1f}Mbps "
                    f"hop={row['avg_hops']:.2f}"
                )

    agent.save(model_path)
    print(f"[OK] saved model: {model_path}")
    print(f"[OK] saved log: {log_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="")
    p.add_argument("--bc-mhz", type=float, default=10.0, choices=[10.0, 20.0])
    p.add_argument("--slot-seconds", type=int, default=None)
    p.add_argument("--episodes", type=int, default=SimConfig().train_episodes)
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--algo", choices=["dueling", "ddqn", "dqn", "all"], default="dueling")
    p.add_argument("--sensitive-ratio", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epsilon-decay", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="例如 cuda 或 cpu；默认自动检测")
    p.add_argument("--tag", type=str, default="", help="附加到模型/日志名后，用于学习率对比等实验")
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
    if args.batch_size is not None:
        cfg_kwargs["batch_size"] = args.batch_size
    if args.warmup_steps is not None:
        cfg_kwargs["warmup_steps"] = args.warmup_steps
    cfg = SimConfig(**cfg_kwargs)
    seed_everything(cfg.seed)
    dataset = Path(args.dataset) if args.dataset else DATASET_DIR / (
        f"ch4_dataset_{cfg.num_sats}sats_{cfg.user_max}users_"
        f"{cfg.bc_hz/1e6:.0f}MHz_{cfg.slot_seconds}s.npz"
    )
    if not dataset.exists():
        raise FileNotFoundError(f"dataset not found: {dataset}. 请先运行 build_dataset_ch4.py")

    suffix = f"_{args.tag}" if args.tag else ""
    mhz = int(cfg.bc_hz / 1e6)
    if args.algo in ("dueling", "all"):
        train_one(f"dueling_ddqn_{mhz}MHz{suffix}", dataset, cfg, network="dueling", ddqn=True,
                  sensitive_ratio=args.sensitive_ratio, device=args.device)
    if args.algo in ("ddqn", "all"):
        train_one(f"ddqn_{mhz}MHz{suffix}", dataset, cfg, network="plain", ddqn=True,
                  sensitive_ratio=args.sensitive_ratio, device=args.device)
    if args.algo in ("dqn", "all"):
        train_one(f"dqn_{mhz}MHz{suffix}", dataset, cfg, network="plain", ddqn=False,
                  sensitive_ratio=args.sensitive_ratio, device=args.device)
