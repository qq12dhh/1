"""

风格特征：
- 中文字体：SimHei / Microsoft YaHei
- 折线图：红/蓝/绿/橙四色，marker 与线型区分
- 3D 图：jet 色图、红色采样点、颜色条范围严格匹配 Z 实际范围
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    HAS_MPL = True
except Exception:
    plt = None
    cm = None
    HAS_MPL = False
    from PIL import Image, ImageDraw, ImageFont

from config_ch3 import FIGURE_DIR, LOG_DIR, SimConfig, ensure_dirs
from utils import moving_average


ALGO_ORDER = ["ddqn", "dqn", "max_snr", "max_visible_time"]
ALGO_LABELS = {
    "ddqn": "DDQN切换策略",
    "dqn": "DQN切换策略",
    "max_snr": "最大信噪比切换策略",
    "max_visible_time": "最大可见时长切换策略",
}
ALGO_STYLE = {
    "ddqn": dict(color="#d62728", linestyle="-", marker="^"),
    "dqn": dict(color="#1f77b4", linestyle="--", marker="o"),
    "max_snr": dict(color="#2ca02c", linestyle="-", marker="s"),
    "max_visible_time": dict(color="#ff7f0e", linestyle="-", marker="d"),
}


def configure_mpl_style() -> None:
    if not HAS_MPL:
        return
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "sans-serif"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pil_font(size=16):
    if HAS_MPL:
        return None
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _save_line_pil(series: list[tuple[list[float], list[float], str]], out: Path,
                   title: str, xlabel: str, ylabel: str):
    """matplotlib 不存在时的简易备用折线图。"""
    w, h = 1000, 650
    margin_l, margin_r, margin_t, margin_b = 90, 40, 70, 90
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    font = _pil_font(18)
    small = _pil_font(14)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    valid = [(np.asarray(x, dtype=float), np.asarray(y, dtype=float), label)
             for x, y, label in series if len(x) and len(y)]
    if not valid:
        img.save(out)
        return
    all_x = np.concatenate([x for x, _, _ in valid])
    all_y = np.concatenate([y[np.isfinite(y)] for _, y, _ in valid])
    if len(all_y) == 0:
        img.save(out)
        return
    xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
    if abs(xmax - xmin) < 1e-9:
        xmax += 1
    if abs(ymax - ymin) < 1e-9:
        ymax += 1
    pad = 0.08 * (ymax - ymin)
    ymin, ymax = ymin - pad, ymax + pad

    def xy(xv, yv):
        px = margin_l + (xv - xmin) / (xmax - xmin) * (w - margin_l - margin_r)
        py = h - margin_b - (yv - ymin) / (ymax - ymin) * (h - margin_t - margin_b)
        return px, py

    d.line([(margin_l, margin_t), (margin_l, h - margin_b), (w - margin_r, h - margin_b)], fill="black", width=2)
    d.text((w // 2 - 180, 20), title, fill="black", font=font)
    d.text((w // 2 - 70, h - 45), xlabel, fill="black", font=small)
    d.text((10, h // 2), ylabel, fill="black", font=small)
    for idx, (xs, ys, label) in enumerate(valid):
        pts = [xy(float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(y)]
        if len(pts) >= 2:
            d.line(pts, fill=colors[idx % len(colors)], width=3)
        for p in pts:
            d.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=colors[idx % len(colors)])
        d.text((w - 280, 80 + idx * 28), label, fill=colors[idx % len(colors)], font=small)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def _save_heatmap_pil(panels: list[tuple[np.ndarray, str]], out: Path, title: str):
    """matplotlib 不存在时用 2D 热力图替代 3D 曲面图。"""
    w, h = 1200, 850
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    font = _pil_font(18)
    small = _pil_font(14)
    d.text((w // 2 - 220, 20), title + "（缺少matplotlib，使用热力图备用）", fill="black", font=font)
    vals = np.concatenate([p[np.isfinite(p)] for p, _ in panels if np.any(np.isfinite(p))])
    vmin, vmax = (float(vals.min()), float(vals.max())) if len(vals) else (0.0, 1.0)
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0
    positions = [(60, 90), (650, 90), (60, 470), (650, 470)]
    for (z, label), (x0, y0) in zip(panels, positions):
        d.text((x0, y0 - 30), label, fill="black", font=font)
        rows, cols = z.shape
        cell_w, cell_h = 480 / max(cols, 1), 260 / max(rows, 1)
        for r in range(rows):
            for c in range(cols):
                val = z[r, c]
                frac = 0.0 if not np.isfinite(val) else (float(val) - vmin) / (vmax - vmin)
                frac = max(0.0, min(1.0, frac))
                color = (int(255 * frac), int(80), int(255 * (1 - frac)))
                d.rectangle([x0 + c * cell_w, y0 + r * cell_h, x0 + (c + 1) * cell_w, y0 + (r + 1) * cell_h], fill=color)
        d.rectangle([x0, y0, x0 + 480, y0 + 260], outline="black", width=2)
        d.text((x0, y0 + 270), "x: 用户数量, y: 时敏业务占比", fill="black", font=small)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def _auto_ylim(ys_list: list[np.ndarray], metric: str) -> tuple[float, float, np.ndarray]:
    vals = np.concatenate([np.asarray(y, dtype=float) for y in ys_list])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0, 1.0, np.linspace(0.0, 1.0, 6)
    ymin, ymax = float(vals.min()), float(vals.max())
    if metric == "handover_rate":
        pad = max(0.02, 0.10 * (ymax - ymin if ymax > ymin else 1.0))
        lo = max(0.0, ymin - pad)
        hi = min(1.0, ymax + pad)
    else:
        pad = max(5.0, 0.10 * (ymax - ymin if ymax > ymin else ymax * 0.1 + 1.0))
        lo = max(0.0, ymin - pad)
        hi = ymax + pad
    return lo, hi, np.linspace(lo, hi, 6)


def plot_convergence(cfg: SimConfig):
    if not HAS_MPL:
        series = []
        for name, label in [("ddqn", "DDQN"), ("dqn", "DQN")]:
            p = LOG_DIR / f"{name}_{int(cfg.bc_hz / 1e6)}MHz_train.csv"
            if p.exists():
                rows = read_csv(p)
                ep = [int(r["episode"]) for r in rows]
                rew = np.array([float(r["reward"]) for r in rows])
                ma = moving_average(rew, 10)
                series.append((ep[-len(ma):], ma.tolist(), label))
        out = FIGURE_DIR / f"fig3_3_convergence_{int(cfg.bc_hz / 1e6)}MHz.png"
        _save_line_pil(series, out, "奖励收敛曲线", "训练轮数", "平均奖励")
        print(f"[OK] {out}")
        return

    configure_mpl_style()
    plt.figure(figsize=(8, 7))
    for algo in ["ddqn", "dqn"]:
        p = LOG_DIR / f"{algo}_{int(cfg.bc_hz / 1e6)}MHz_train.csv"
        if not p.exists():
            continue
        rows = read_csv(p)
        ep = np.array([int(r["episode"]) for r in rows])
        rew = np.array([float(r["reward"]) for r in rows])
        ma = moving_average(rew, 10)
        style = ALGO_STYLE[algo]
        plt.plot(
            ep[-len(ma):], ma,
            color=style["color"], linestyle=style["linestyle"], marker=style["marker"],
            markersize=4, markevery=max(1, len(ma) // 12), linewidth=1.5,
            label=ALGO_LABELS[algo],
        )
    plt.xlabel("训练轮数", fontsize=16)
    plt.ylabel("平均奖励", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle="--", color="gray", alpha=0.7)
    plt.legend(loc="best", fontsize=14)
    out = FIGURE_DIR / f"fig3_3_convergence_{int(cfg.bc_hz / 1e6)}MHz.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def group_eval(rows, scene: str, metric: str):
    xs = sorted({int(r["num_users"]) for r in rows if r["scene"] == scene})
    data = {a: [] for a in ALGO_ORDER}
    for algo in ALGO_ORDER:
        for x in xs:
            vals = [
                float(r[metric]) for r in rows
                if r["scene"] == scene and r["algo"] == algo and int(r["num_users"]) == x
            ]
            data[algo].append(np.mean(vals) if vals else np.nan)
    return xs, data


def plot_line_eval(cfg: SimConfig):
    eval_path = LOG_DIR / f"eval_{int(cfg.bc_hz / 1e6)}MHz.csv"
    if not eval_path.exists():
        print(f"[WARN] missing {eval_path}")
        return
    rows = read_csv(eval_path)
    specs = [
        ("tolerant", "er_mbps", "传输速率(Mbps)", "fig3_4_tolerant_rate"),
        ("tolerant", "handover_rate", "切换频率", "fig3_5_tolerant_handover"),
        ("sensitive", "et_mbps", "数据流传输速率(Mbps)", "fig3_6_sensitive_rate"),
        ("sensitive", "handover_rate", "切换频率", "fig3_7_sensitive_handover"),
    ]

    for scene, metric, ylabel, fname in specs:
        xs, data = group_eval(rows, scene, metric)
        if not HAS_MPL:
            series = []
            for algo in ALGO_ORDER:
                ys = data[algo]
                if not np.all(np.isnan(ys)):
                    series.append((xs, ys, ALGO_LABELS[algo]))
            out = FIGURE_DIR / f"{fname}_{int(cfg.bc_hz / 1e6)}MHz.png"
            _save_line_pil(series, out, fname, "用户数量", ylabel)
            print(f"[OK] {out}")
            continue

        configure_mpl_style()
        plt.figure(figsize=(8, 7))
        ys_for_ylim = []
        for algo in ALGO_ORDER:
            ys = np.asarray(data[algo], dtype=float)
            if np.all(np.isnan(ys)):
                continue
            ys_for_ylim.append(ys)
            style = ALGO_STYLE[algo]
            plt.plot(
                xs, ys,
                color=style["color"], linestyle=style["linestyle"], marker=style["marker"],
                markersize=8, linewidth=1.5, label=ALGO_LABELS[algo],
            )

        ymin, ymax, yticks = _auto_ylim(ys_for_ylim, metric)
        plt.xlabel("用户数量", fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.xticks(xs, fontsize=14)
        plt.yticks(yticks, fontsize=14)
        plt.xlim(min(xs) - 10, max(xs) + 10)
        plt.ylim(ymin, ymax)
        plt.grid(True, linestyle="--", color="gray", alpha=0.7)
        plt.legend(loc="best", fontsize=14)
        out = FIGURE_DIR / f"{fname}_{int(cfg.bc_hz / 1e6)}MHz.png"
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] {out}")


def _build_z(rows: list[dict], users: list[int], ratios: list[float], algo: str, metric: str) -> np.ndarray:
    z = np.full((len(ratios), len(users)), np.nan, dtype=float)
    for ri, ratio in enumerate(ratios):
        for ui, user in enumerate(users):
            vals = [
                float(r[metric]) for r in rows
                if r["algo"] == algo
                and int(r["num_users"]) == user
                and abs(float(r["sensitive_ratio"]) - ratio) < 1e-9
            ]
            if vals:
                z[ri, ui] = float(np.mean(vals))
    return z


def _interp_surface(users: list[int], ratios_pct: list[float], z: np.ndarray, n: int = 100):
    """不用 scipy，仅用两次 np.interp 做双线性风格插值。"""
    users_arr = np.asarray(users, dtype=float)
    ratios_arr = np.asarray(ratios_pct, dtype=float)
    x_dense = np.linspace(users_arr.min(), users_arr.max(), n)
    y_dense = np.linspace(ratios_arr.min(), ratios_arr.max(), n)

    z_filled = z.copy()
    # 理论上 eval.csv 是满网格；这里防御性填补缺失值。
    if np.any(~np.isfinite(z_filled)):
        finite_mean = np.nanmean(z_filled)
        z_filled[~np.isfinite(z_filled)] = finite_mean if np.isfinite(finite_mean) else 0.0

    temp = np.vstack([np.interp(x_dense, users_arr, row) for row in z_filled])
    z_dense = np.vstack([np.interp(y_dense, ratios_arr, temp[:, j]) for j in range(temp.shape[1])]).T
    x_grid, y_grid = np.meshgrid(x_dense, y_dense)
    return x_grid, y_grid, z_dense


def _plot_single_surface(users, ratios, z, algo, metric, fname, cfg):
    ratios_pct = [r * 100.0 if r <= 1.0 else r for r in ratios]
    x_grid, y_grid, z_grid = _interp_surface(users, ratios_pct, z, 100)
    scatter_x, scatter_y = np.meshgrid(np.asarray(users, dtype=float), np.asarray(ratios_pct, dtype=float))
    scatter_z = z

    zmin = float(np.nanmin(z_grid))
    zmax = float(np.nanmax(z_grid))
    zlabel = "切换频率" if metric == "handover_rate" else "数据流传输速率(Mbps)"

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        x_grid, y_grid, z_grid,
        cmap=cm.jet,
        linewidth=0.5,
        antialiased=True,
        alpha=0.9,
        rstride=10,
        cstride=10,
        vmin=zmin,
        vmax=zmax,
    )
    ax.scatter(scatter_x.ravel(), scatter_y.ravel(), scatter_z.ravel(),
               color="darkred", s=50, zorder=5, marker="o")

    ax.set_xlabel("用户数量", fontsize=14, labelpad=20)
    ax.set_ylabel("时敏业务占比(%)", fontsize=14, labelpad=20)
    ax.set_zlabel(zlabel + "     ", fontsize=14, labelpad=15)
    ax.set_title(ALGO_LABELS[algo], fontsize=15, pad=12)
    ax.set_xlim(min(users), max(users))
    ax.set_ylim(min(ratios_pct), max(ratios_pct))
    ax.set_zlim(zmin - 0.005 * max(abs(zmin), 1.0), zmax + 0.005 * max(abs(zmax), 1.0))
    ax.set_xticks(users)
    if set(ratios_pct) >= {0.0, 50.0, 100.0}:
        ax.set_yticks([0, 25, 50, 75, 100])
    else:
        ax.set_yticks(ratios_pct)
    ax.view_init(elev=25, azim=-55)
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.8, aspect=12)
    cbar.set_label(zlabel, fontsize=12)
    cbar.set_ticks(np.linspace(zmin, zmax, 8))

    out = FIGURE_DIR / f"{fname}_{algo}_{int(cfg.bc_hz / 1e6)}MHz.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")
    print(f"[{out.name}] Z最小值: {zmin:.4f}, Z最大值: {zmax:.4f}, 色条范围: {zmin:.4f} ~ {zmax:.4f}")


def _plot_combined_surface(users, ratios, z_by_algo, metric, fname, cfg):
    ratios_pct = [r * 100.0 if r <= 1.0 else r for r in ratios]
    zlabel = "切换频率" if metric == "handover_rate" else "数据流传输速率(Mbps)"
    fig = plt.figure(figsize=(14, 10), dpi=150)

    for i, algo in enumerate(ALGO_ORDER, start=1):
        z = z_by_algo.get(algo)
        if z is None or np.all(~np.isfinite(z)):
            continue
        x_grid, y_grid, z_grid = _interp_surface(users, ratios_pct, z, 100)
        scatter_x, scatter_y = np.meshgrid(np.asarray(users, dtype=float), np.asarray(ratios_pct, dtype=float))
        zmin, zmax = float(np.nanmin(z_grid)), float(np.nanmax(z_grid))

        ax = fig.add_subplot(2, 2, i, projection="3d")
        surf = ax.plot_surface(
            x_grid, y_grid, z_grid,
            cmap=cm.jet,
            linewidth=0.5,
            antialiased=True,
            alpha=0.9,
            rstride=10,
            cstride=10,
            vmin=zmin,
            vmax=zmax,
        )
        ax.scatter(scatter_x.ravel(), scatter_y.ravel(), z.ravel(),
                   color="darkred", s=18, zorder=5, marker="o")
        ax.set_title(ALGO_LABELS[algo], fontsize=12)
        ax.set_xlabel("用户数量", fontsize=10, labelpad=8)
        ax.set_ylabel("时敏业务占比(%)", fontsize=10, labelpad=8)
        ax.set_zlabel(zlabel, fontsize=10, labelpad=8)
        ax.set_xlim(min(users), max(users))
        ax.set_ylim(min(ratios_pct), max(ratios_pct))
        ax.set_xticks(users)
        ax.set_yticks([0, 50, 100] if max(ratios_pct) >= 100 else ratios_pct)
        ax.view_init(elev=25, azim=-55)
        fig.colorbar(surf, ax=ax, shrink=0.55, aspect=10)

    out = FIGURE_DIR / f"{fname}_{int(cfg.bc_hz / 1e6)}MHz.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")


def plot_surface(cfg: SimConfig):
    eval_path = LOG_DIR / f"eval_{int(cfg.bc_hz / 1e6)}MHz.csv"
    if not eval_path.exists():
        return
    rows = [r for r in read_csv(eval_path) if r["scene"] == "mixed"]
    if not rows:
        return

    users = sorted({int(r["num_users"]) for r in rows})
    ratios = sorted({float(r["sensitive_ratio"]) for r in rows})

    for metric, fname in [
        ("et_mbps", "fig3_8_mixed_rate"),
        ("handover_rate", "fig3_9_mixed_handover"),
    ]:
        z_by_algo = {algo: _build_z(rows, users, ratios, algo, metric) for algo in ALGO_ORDER}

        if not HAS_MPL:
            panels = [(z_by_algo[algo], ALGO_LABELS[algo]) for algo in ALGO_ORDER]
            out = FIGURE_DIR / f"{fname}_{int(cfg.bc_hz / 1e6)}MHz.png"
            _save_heatmap_pil(panels, out, fname)
            print(f"[OK] {out}")
            continue

        configure_mpl_style()
        # 保留原来的 2x2 总览图
        _plot_combined_surface(users, ratios, z_by_algo, metric, fname, cfg)
        # 额外输出每种策略的单独 3D 曲面图，风格与你给的模板一致
        for algo in ALGO_ORDER:
            z = z_by_algo[algo]
            if not np.all(~np.isfinite(z)):
                _plot_single_surface(users, ratios, z, algo, metric, fname, cfg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bc-mhz", type=float, default=10.0, choices=[10.0, 20.0])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_dirs()
    cfg = SimConfig(bc_hz=args.bc_mhz * 1e6)
    plot_convergence(cfg)
    plot_line_eval(cfg)
    plot_surface(cfg)
