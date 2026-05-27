from agi.stk12.stkdesktop import STKDesktop

# =====================================
# 启动 STK
# =====================================
stk = STKDesktop.StartApplication(visible=True)

root = stk.Root

# =====================================
# 关闭旧场景
# =====================================
try:
    root.CloseScenario()
except:
    pass

# =====================================
# 创建场景
# =====================================
root.NewScenario("Satellite_Demo")

scenario = root.CurrentScenario

print("场景创建成功")

# =====================================
# 设置时间
# =====================================
scenario.SetTimePeriod(
    "26 May 2026 00:00:00.000",
    "27 May 2026 00:00:00.000"
)

root.Rewind()

# =====================================
# 创建三颗卫星
# =====================================

for i in range(3):

    sat_name = f"Sat_{i+1}"

    # 创建卫星
    root.ExecuteCommand(
        f'New / */Satellite {sat_name}'
    )

    print(f"{sat_name} 创建成功")

    # 使用 STK 内置轨道
    root.ExecuteCommand(
        f'OrbitWizard */Satellite/{sat_name} '
        f'Circular '
        f'Inclination {30 + i*20} '
        f'Altitude {500 + i*200}'
    )

# =====================================
# 缩放视角
# =====================================
root.ExecuteCommand(
    "VO * View Home"
)

# =====================================
# 开始动画
# =====================================
root.ExecuteCommand(
    "Animate * Start"
)

print("卫星开始运行")