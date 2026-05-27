"""第四章复现实验绘图。

绘图风格按用户给出的模板：中文字体、红/蓝/绿/橙学术配色，3D 曲面采用 cm.jet 并叠加红色采样点。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from config_ch4 import FIGURE_DIR, LOG_DIR, SimConfig, ensure_dirs
from utils import moving_average

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
except Exception as exc:  # pragma: no cover
    plt = None
    cm = None
    _MATPLOTLIB_ERROR = exc
else:
    _MATPLOTLIB_ERROR = None


STYLE = {
    "dueling_ddqn": {"color": "#d62728", "linestyle": "-", "marker": "^", "label": "Dueling-DDQN切换策略"},
    "ddqn": {"color": "#1f77b4", "linestyle": "--", "marker": "o", "label": "DDQN切换策略"},
    "dqn": {"color": "#9467bd", "linestyle": ":", "marker": "v", "label": "DQN切换策略"},
    "max_capacity": {"color": "#2ca02c", "linestyle": "-", "marker": "s", "label": "最大信道容量切换策略"},
    "max_visible_time": {"color": "#ff7f0e", "linestyle": "-", "marker": "d", "label": "最大可见时长切换策略"},
    "min_hop": {"color": "#8c564b", "linestyle": "-.", "marker": "x", "label": "最小跳数切换策略"},
    "min_path_quality": {"color": "#17becf", "linestyle": "-.", "marker": "*", "label": "最小路径质量切换策略"},
}


def setup_matplotlib() -> None:
    if plt is None:
        raise RuntimeError(f"matplotlib 不可用：{_MATPLOTLIB_ERROR}")
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "sans-serif"


def read_csv_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out = dict(r)
            for k, v in list(out.items()):
                try:
                    if v is not None and v != "":
                        out[k] = float(v)
                except ValueError:
                    pass
            rows.append(out)
    return rows


def savefig(name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[OK] saved figure: {path}")
    return path


def plot_training(log_dir: Path = LOG_DIR) -> None:
    logs = sorted(log_dir.glob("*train.csv"))
    if not logs:
        print("[WARN] no train logs found, skip convergence plot")
        return
    plt.figure(figsize=(8, 6))
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    for i, log in enumerate(logs):
        rows = read_csv_rows(log)
        if not rows:
            continue
        x = np.array([r["episode"] for r in rows], dtype=float)
        y = np.array([r.get("avg20_reward", r.get("reward", 0.0)) for r in rows], dtype=float)
        plt.plot(x, y, color=colors[i % len(colors)], linewidth=1.8, label=log.stem.replace("_train", ""))
    plt.xlabel("训练轮次", fontsize=14)
    plt.ylabel("平均奖励", fontsize=14)
    plt.grid(True, linestyle="--", color="gray", alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    savefig("fig4_4_reward_convergence.png")


def _group_metric(rows: list[dict], scene: str, metric: str, fixed_users: int | None = None):
    data: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        if r.get("scene") != scene:
            continue
        if fixed_users is not None and int(r.get("num_users", -1)) != fixed_users:
            continue
        algo = str(r.get("algo"))
        x = float(r.get("sensitive_ratio", 0.0)) * 100.0
        y = float(r.get(metric, 0.0))
        data.setdefault(algo, []).append((x, y))
    for algo in data:
        data[algo] = sorted(data[algo])
    return data


def plot_ratio_performance(rows: list[dict], users: int = 200) -> None:
    metrics = [
        ("avg_hops", "路径跳数", "(a) 不同时敏业务占比下的路径跳数"),
        ("handover_rate", "切换频率", "(b) 不同时敏业务占比下的切换频率"),
        ("throughput_mbps", "数据流速率(Mbps)", "(c) 不同时敏业务占比下的数据流速率"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        data = _group_metric(rows, "ratio", metric, fixed_users=users)
        for algo, pts in data.items():
            st = STYLE.get(algo, {"label": algo})
            x = np.array([p[0] for p in pts], dtype=float)
            y = np.array([p[1] for p in pts], dtype=float)
            ax.plot(
                x, y,
                color=st.get("color"), linestyle=st.get("linestyle", "-"),
                marker=st.get("marker", "o"), markersize=7, linewidth=1.6,
                label=st.get("label", algo),
            )
        ax.set_xlabel("时敏业务占比(%)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=13)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(True, linestyle="--", color="gray", alpha=0.7)
    axes[-1].legend(loc="best", fontsize=10)
    plt.tight_layout()
    savefig("fig4_6_performance_compare.png")


def plot_users_adaptation(rows: list[dict]) -> None:
    plt.figure(figsize=(8, 6))
    for algo in ["dueling_ddqn", "ddqn", "max_capacity", "max_visible_time"]:
        pts = [r for r in rows if r.get("scene") == "users" and r.get("algo") == algo]
        if not pts:
            continue
        pts = sorted(pts, key=lambda r: r["num_users"])
        x = np.array([r["num_users"] for r in pts], dtype=float)
        y = np.array([r["reward"] for r in pts], dtype=float)
        st = STYLE.get(algo, {"label": algo})
        plt.plot(x, y, color=st.get("color"), linestyle=st.get("linestyle", "-"),
                 marker=st.get("marker", "o"), markersize=7, linewidth=1.6,
                 label=st.get("label", algo))
    plt.xlabel("用户数量", fontsize=14)
    plt.ylabel("平均奖励", fontsize=14)
    plt.xticks([50, 100, 150, 200, 250, 300], fontsize=12)
    plt.grid(True, linestyle="--", color="gray", alpha=0.7)
    plt.legend(loc="best", fontsize=11)
    plt.tight_layout()
    savefig("fig4_5_user_number_adaptation.png")


def plot_surface(rows: list[dict], algo: str = "dueling_ddqn", metric: str = "throughput_mbps") -> None:
    pts = [r for r in rows if r.get("scene") == "surface" and r.get("algo") == algo]
    if not pts:
        print("[WARN] no surface rows found, skip 3D surface plot")
        return
    users = sorted({int(r["num_users"]) for r in pts})
    ratios = sorted({float(r["sensitive_ratio"]) * 100.0 for r in pts})
    Z = np.full((len(ratios), len(users)), np.nan, dtype=float)
    for r in pts:
        i = ratios.index(float(r["sensitive_ratio"]) * 100.0)
        j = users.index(int(r["num_users"]))
        Z[i, j] = float(r[metric])
    if np.isnan(Z).any():
        print("[WARN] surface data incomplete, skip")
        return

    X, Y = np.meshgrid(np.array(users), np.array(ratios))
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.jet,
        linewidth=0.5,
        antialiased=True,
        alpha=0.9,
        rstride=1, cstride=1,
        vmin=np.min(Z), vmax=np.max(Z),
    )
    ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), color="darkred", s=45, zorder=5, marker="o")
    ax.set_xlabel("用户数量", fontsize=14, labelpad=16)
    ax.set_ylabel("时敏业务占比(%)", fontsize=14, labelpad=16)
    zlabel = "数据流速率(Mbps)" if metric == "throughput_mbps" else metric
    ax.set_zlabel(zlabel + "     ", fontsize=14, labelpad=14)
    ax.set_xlim(min(users), max(users))
    ax.set_ylim(min(ratios), max(ratios))
    ax.set_xticks(users)
    ax.set_yticks(ratios)
    ax.set_zlim(np.min(Z) - 0.005 * abs(np.min(Z)), np.max(Z) + 0.005 * abs(np.max(Z)))
    ax.view_init(elev=25, azim=-55)
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.8, aspect=12)
    cbar.set_label(zlabel, fontsize=12)
    cbar.set_ticks(np.linspace(np.min(Z), np.max(Z), 8))
    print("=== Z值实际范围 ===")
    print(f"Z最小值: {np.min(Z):.4f}")
    print(f"Z最大值: {np.max(Z):.4f}")
    print(f"色条范围: {np.min(Z):.4f} ~ {np.max(Z):.4f}")
    plt.tight_layout()
    savefig(f"fig4_surface_{algo}_{metric}.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-csv", type=str, default="")
    p.add_argument("--bc-mhz", type=float, default=10.0)
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--surface", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_dirs()
    setup_matplotlib()
    eval_csv = Path(args.eval_csv) if args.eval_csv else LOG_DIR / f"eval_ch4_{int(args.bc_mhz)}MHz.csv"
    rows = read_csv_rows(eval_csv)
    plot_training(LOG_DIR)
    if rows:
        plot_ratio_performance(rows, users=args.users)
        plot_users_adaptation(rows)
        if args.surface:
            plot_surface(rows, algo="dueling_ddqn", metric="throughput_mbps")
            plot_surface(rows, algo="dueling_ddqn", metric="handover_rate")
    else:
        print(f"[WARN] eval csv not found or empty: {eval_csv}")
