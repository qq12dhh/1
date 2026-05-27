"""可选：在 STK 中创建第三章可视化场景。

训练和画图默认使用 build_dataset_ch3.py 的内置轨道几何数据；
本脚本用于在 STK 中查看 256 星、用户和地面站布局。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np

from build_dataset_ch3 import gateway_ring, random_points_near
from config_ch3 import SimConfig


def stk_time(dt: datetime) -> str:
    return dt.strftime("%d %b %Y %H:%M:%S.000")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", action="store_true", help="后台运行 STK，不显示界面")
    p.add_argument("--users", type=int, default=50)
    p.add_argument("--sats", type=int, default=256)
    args = p.parse_args()

    from agi.stk12.stkdesktop import STKDesktop
    import agi.stk12.stkobjects.stkobjects as stkobj
    import agi.stk12.stkutil as stkutil

    # STK Python 包中的枚举由 COM 包装层动态注册，部分 IDE 静态检查会误报“未解析引用”。
    # 用 getattr 获取可避免 IDE 报红，同时运行时仍使用 STK 官方枚举对象。
    stk_object_type = getattr(stkobj, "AgESTKObjectType")
    propagator_type = getattr(stkobj, "AgEVePropagatorType")
    coordinate_system = getattr(stkutil, "AgECoordinateSystem")

    cfg = SimConfig(num_users=args.users, user_max=args.users, num_sats=args.sats)
    if args.sats != 256:
        cfg.num_planes = 1
        cfg.sats_per_plane = args.sats
    rng = np.random.default_rng(cfg.seed)

    stk = STKDesktop.StartApplication(visible=not args.hidden)
    root = stk.Root
    try:
        root.CloseScenario()
    except Exception:
        pass
    root.NewScenario("Ch3_DDQN_Handover")
    scenario = root.CurrentScenario
    try:
        root.UnitPreferences.SetCurrentUnit("Distance", "km")
        root.UnitPreferences.SetCurrentUnit("Angle", "deg")
        root.UnitPreferences.SetCurrentUnit("Time", "sec")
    except Exception:
        pass
    start = datetime(2026, 5, 26, 0, 0, 0)
    stop = start + timedelta(minutes=cfg.sim_minutes)
    scenario.SetTimePeriod(stk_time(start), stk_time(stop))
    root.Rewind()

    for pidx in range(cfg.num_planes):
        raan = 360.0 * pidx / cfg.num_planes
        for sidx in range(cfg.sats_per_plane):
            idx = pidx * cfg.sats_per_plane + sidx + 1
            if idx > args.sats:
                break
            sat = f"Sat_{idx:03d}"
            mean_anomaly = 360.0 * (sidx / cfg.sats_per_plane + cfg.phase_factor * pidx / max(cfg.num_sats, 1))
            mean_anomaly = mean_anomaly % 360.0

            # OrbitWizard 对 Circular 轨道的可选参数支持较少，部分 STK 版本会拒绝
            # ArgOfPerigee/TrueAnomaly。这里改用 Object Model 设置 TwoBody + Classical，
            # 可以稳定指定半长轴、倾角、RAAN 和相位。
            satellite = scenario.Children.New(stk_object_type.eSatellite, sat)
            satellite.SetPropagatorType(propagator_type.ePropagatorTwoBody)
            propagator = satellite.Propagator
            sma_km = 6371.0 + cfg.altitude_km
            propagator.InitialState.Representation.AssignClassical(
                coordinate_system.eCoordinateSystemJ2000,
                sma_km,
                0.0,
                cfg.inclination_deg,
                0.0,
                raan,
                mean_anomaly,
            )
            propagator.Propagate()
            if idx % 50 == 0:
                print(f"created {idx} satellites")

    lats, lons = random_points_near(cfg.center_lat_deg, cfg.center_lon_deg, cfg.user_area_radius_km, args.users, rng)
    for i, (lat, lon) in enumerate(zip(lats, lons), start=1):
        name = f"User_{i:03d}"
        root.ExecuteCommand(f"New / */Place {name}")
        root.ExecuteCommand(f"SetPosition */Place/{name} Geodetic {lat:.6f} {lon:.6f} 0")

    glats, glons = gateway_ring(cfg.center_lat_deg, cfg.center_lon_deg, cfg.gateway_ring_radius_km, cfg.num_ground_stations)
    for i, (lat, lon) in enumerate(zip(glats, glons), start=1):
        name = f"GS_{i:02d}"
        root.ExecuteCommand(f"New / */Facility {name}")
        root.ExecuteCommand(f"SetPosition */Facility/{name} Geodetic {lat:.6f} {lon:.6f} 0")

    root.ExecuteCommand("VO * View Home")
    print("[OK] STK scenario Ch3_DDQN_Handover created")


if __name__ == "__main__":
    main()




