"""DQN / DDQN / Dueling-DDQN 智能体。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover
    torch = None
    nn = None
    optim = None

from config_ch4 import SimConfig


class QNetwork(nn.Module):
    """普通 DQN 头，用作 DDQN/DQN 基准。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """Dueling 网络：Q(s,a)=V(s)+A(s,a)-mean_a A(s,a)。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x):
        feat = self.feature(x)
        v = self.value(feat)
        a = self.advantage(feat)
        return v + (a - a.mean(dim=1, keepdim=True))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.append((
            state.astype(np.float32),
            int(action),
            float(reward),
            next_state.astype(np.float32),
            bool(done),
            mask.astype(bool),
            next_mask.astype(bool),
        ))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, masks, next_masks = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            np.asarray(masks, dtype=bool),
            np.asarray(next_masks, dtype=bool),
        )


@dataclass
class AgentStats:
    loss: float = 0.0
    epsilon: float = 0.0


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: SimConfig,
        ddqn: bool = True,
        network: str = "dueling",
        device: str | None = None,
    ):
        if torch is None:
            raise RuntimeError("PyTorch 未安装，无法训练。请在 handover 环境中安装 torch。")
        self.cfg = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ddqn = ddqn
        self.network = network
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        net_cls = DuelingQNetwork if network == "dueling" else QNetwork
        self.online = net_cls(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target = net_cls(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=cfg.lr, amsgrad=False)
        self.replay = ReplayBuffer(cfg.replay_size)
        self.epsilon = cfg.epsilon_init

    def select_action(self, state: np.ndarray, mask: np.ndarray, explore: bool = True) -> int:
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return 0
        if explore and random.random() < self.epsilon:
            return int(random.choice(valid))
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(s).squeeze(0).detach().cpu().numpy()
        q[~mask] = -np.inf
        return int(np.argmax(q))

    def update(self) -> AgentStats:
        if len(self.replay) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return AgentStats(0.0, self.epsilon)

        states, actions, rewards, next_states, dones, masks, next_masks = self.replay.sample(self.cfg.batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_masks_t = torch.as_tensor(next_masks, dtype=torch.bool, device=self.device)

        q = self.online(states_t).gather(1, actions_t)
        with torch.no_grad():
            if self.ddqn:
                next_q_online = self.online(next_states_t).masked_fill(~next_masks_t, -1e9)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q = self.target(next_states_t).gather(1, next_actions)
            else:
                next_q = self.target(next_states_t).masked_fill(~next_masks_t, -1e9).max(dim=1, keepdim=True).values
            target = rewards_t + self.cfg.gamma * (1.0 - dones_t) * next_q

        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.soft_update()
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)
        return AgentStats(float(loss.item()), self.epsilon)

    def soft_update(self):
        tau = self.cfg.tau
        with torch.no_grad():
            for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
                target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "epsilon": self.epsilon,
            "ddqn": self.ddqn,
            "network": self.network,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }, path)

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt.get("target", ckpt["online"]))
        self.epsilon = float(ckpt.get("epsilon", self.cfg.epsilon_min))
