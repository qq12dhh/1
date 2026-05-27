"""星间路径信息表与路径质量广播更新机制。"""
from __future__ import annotations

from collections import deque

import numpy as np

from config_ch4 import SimConfig
from utils import safe_norm


def build_walker_adjacency(cfg: SimConfig) -> np.ndarray:
    """每颗卫星连接轨内前后两星 + 相邻轨道面同序号两星，共 4 条 ISL。"""
    if cfg.num_sats != cfg.num_planes * cfg.sats_per_plane:
        # 非标准规模时退化为单环拓扑，便于快速调试。
        adj = np.zeros((cfg.num_sats, 2), dtype=np.int16)
        for i in range(cfg.num_sats):
            adj[i, 0] = (i - 1) % cfg.num_sats
            adj[i, 1] = (i + 1) % cfg.num_sats
        return adj

    adj = np.zeros((cfg.num_sats, 4), dtype=np.int16)
    for p in range(cfg.num_planes):
        for s in range(cfg.sats_per_plane):
            idx = p * cfg.sats_per_plane + s
            adj[idx, 0] = p * cfg.sats_per_plane + ((s - 1) % cfg.sats_per_plane)
            adj[idx, 1] = p * cfg.sats_per_plane + ((s + 1) % cfg.sats_per_plane)
            adj[idx, 2] = ((p - 1) % cfg.num_planes) * cfg.sats_per_plane + s
            adj[idx, 3] = ((p + 1) % cfg.num_planes) * cfg.sats_per_plane + s
    return adj


def gateway_visibility(sat_ecef: np.ndarray, gs_ecef: np.ndarray,
                       min_elev_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """返回 landing_mask[T,N,G] 与 elevation[T,N,G]。"""
    t_num, sat_num, _ = sat_ecef.shape
    g_num = gs_ecef.shape[0]
    mask = np.zeros((t_num, sat_num, g_num), dtype=bool)
    elevation = np.zeros((t_num, sat_num, g_num), dtype=np.float32)
    gs_up = gs_ecef / safe_norm(gs_ecef, axis=1, keepdims=True)
    for t in range(t_num):
        rel = sat_ecef[t][:, None, :] - gs_ecef[None, :, :]
        dist = safe_norm(rel, axis=2)
        los = rel / dist[:, :, None]
        sin_el = np.sum(los * gs_up[None, :, :], axis=2)
        el = np.rad2deg(np.arcsin(np.clip(sin_el, -1.0, 1.0)))
        elevation[t] = el.astype(np.float32)
        mask[t] = el >= min_elev_deg
    return mask, elevation


def generate_isl_loads(sat_ecef: np.ndarray, adjacency: np.ndarray, cfg: SimConfig,
                       rng: np.random.Generator) -> np.ndarray:
    """生成 l_ij(t)：由星间距离、周期性负载与随机扰动合成的链路负载率。"""
    t_num, sat_num, _ = sat_ecef.shape
    deg = adjacency.shape[1]
    loads = np.zeros((t_num, sat_num, deg), dtype=np.float32)
    edge_phase = rng.uniform(0.0, 2.0 * np.pi, size=(sat_num, deg))
    times = np.arange(t_num, dtype=np.float64)
    period = max(1, t_num)

    # 先计算所有 ISL 距离并做全局归一化。
    dist = np.zeros((t_num, sat_num, deg), dtype=np.float32)
    for k in range(deg):
        nb = adjacency[:, k]
        dist[:, :, k] = safe_norm(sat_ecef - sat_ecef[:, nb, :], axis=2).astype(np.float32)
    d_min = float(np.min(dist))
    d_max = float(np.max(dist))
    dist_norm = (dist - d_min) / max(d_max - d_min, 1.0)

    for t in range(t_num):
        periodic = 0.5 + 0.5 * np.sin(2.0 * np.pi * times[t] / period + edge_phase)
        noise = rng.normal(0.0, cfg.link_load_noise, size=(sat_num, deg))
        loads[t] = np.clip(
            cfg.link_load_base
            + 0.25 * dist_norm[t]
            + cfg.link_load_variation * periodic
            + noise,
            0.02,
            0.95,
        ).astype(np.float32)
    return loads


def _edge_load(edge_load_t: np.ndarray, adjacency: np.ndarray, i: int, j: int) -> float:
    pos = np.where(adjacency[i] == j)[0]
    if len(pos) == 0:
        return 1.0
    return float(edge_load_t[i, pos[0]])


def _bfs_route_for_gateway(roots: np.ndarray, elevation_g: np.ndarray,
                           edge_load_t: np.ndarray, adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sat_num = adjacency.shape[0]
    hops = np.full(sat_num, 32767, dtype=np.int16)
    q = np.full(sat_num, np.inf, dtype=np.float32)
    next_hop = np.full(sat_num, -1, dtype=np.int16)
    parent = np.full(sat_num, -1, dtype=np.int16)

    if roots.size == 0:
        roots = np.array([int(np.argmax(elevation_g))], dtype=np.int64)

    dq: deque[int] = deque()
    # 多个落地星同时作为根；按仰角从高到低入队，平局时更稳定。
    roots = roots[np.argsort(-elevation_g[roots])]
    for r in roots:
        hops[r] = 0
        q[r] = 0.0
        next_hop[r] = r
        parent[r] = r
        dq.append(int(r))

    while dq:
        cur = dq.popleft()
        for nb in adjacency[cur]:
            nb = int(nb)
            if hops[nb] > hops[cur] + 1:
                hops[nb] = np.int16(hops[cur] + 1)
                parent[nb] = cur
                next_hop[nb] = np.int16(cur)
                # 图片/论文式(4-8)：Q_i,k = Q_j,k + l_i,j，其中 j 为下一跳。
                q[nb] = np.float32(q[cur] + _edge_load(edge_load_t, adjacency, nb, cur))
                dq.append(nb)

    # 极端非连通时兜底。
    bad = ~np.isfinite(q)
    if np.any(bad):
        best = int(roots[0])
        q[bad] = 10.0
        hops[bad] = np.int16(adjacency.shape[0])
        next_hop[bad] = np.int16(best)
    return next_hop, hops, q


def compute_path_tables(sat_ecef: np.ndarray, gs_ecef: np.ndarray, cfg: SimConfig,
                        rng: np.random.Generator | None = None) -> dict:
    """计算每个广播更新时刻的星间路径信息表。

    输出字段对应图片表格：
    - destination segment：由 gateway index 表示
    - next hop：route_next_hop[T,N,G]
    - path quality：path_quality[T,N,G]
    - hop count：hop_count[T,N,G]
    """
    rng = rng or np.random.default_rng(cfg.seed)
    t_num, sat_num, _ = sat_ecef.shape
    g_num = gs_ecef.shape[0]
    adjacency = build_walker_adjacency(cfg)
    edge_load = generate_isl_loads(sat_ecef, adjacency, cfg, rng)
    landing_mask, landing_elevation = gateway_visibility(sat_ecef, gs_ecef, cfg.min_elevation_deg)

    route_next_hop = np.full((t_num, sat_num, g_num), -1, dtype=np.int16)
    hop_count = np.zeros((t_num, sat_num, g_num), dtype=np.int16)
    path_quality = np.zeros((t_num, sat_num, g_num), dtype=np.float32)

    update_slots = max(1, int(round(cfg.path_update_interval_s / max(cfg.slot_seconds, 1))))
    last_tables: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for t in range(t_num):
        if t % update_slots == 0 or last_tables is None:
            nh_t = np.zeros((sat_num, g_num), dtype=np.int16)
            hp_t = np.zeros((sat_num, g_num), dtype=np.int16)
            q_t = np.zeros((sat_num, g_num), dtype=np.float32)
            for g in range(g_num):
                roots = np.flatnonzero(landing_mask[t, :, g])
                # ?????????????????????????
                # ???????????????????????????????????0????
                # ???????????????????????????????1?????????
                if roots.size > cfg.landing_sats_per_gateway:
                    order = np.argsort(-landing_elevation[t, roots, g])
                    roots = roots[order[: cfg.landing_sats_per_gateway]]
                nh, hp, qq = _bfs_route_for_gateway(roots, landing_elevation[t, :, g], edge_load[t], adjacency)
                nh_t[:, g] = nh
                hp_t[:, g] = hp
                q_t[:, g] = qq
            last_tables = (nh_t, hp_t, q_t)
        route_next_hop[t], hop_count[t], path_quality[t] = last_tables

    return {
        "adjacency": adjacency.astype(np.int16),
        "edge_load": edge_load.astype(np.float32),
        "landing_mask": landing_mask,
        "route_next_hop": route_next_hop,
        "hop_count": hop_count,
        "path_quality": path_quality,
    }


def format_route_rows(route_next_hop: np.ndarray, hop_count: np.ndarray, path_quality: np.ndarray,
                      sat_idx: int, max_rows: int = 8) -> list[dict]:
    """将某颗卫星的路径表格式化为“目的网段/下一跳/路径质量/跳数”。"""
    rows = []
    g_num = path_quality.shape[1]
    for g in range(min(g_num, max_rows)):
        rows.append({
            "destination_segment": f"G{g + 1:02d}",
            "next_hop": int(route_next_hop[sat_idx, g]),
            "path_quality": float(path_quality[sat_idx, g]),
            "hop_count": int(hop_count[sat_idx, g]),
        })
    return rows
