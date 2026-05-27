# 代码说明

路径：`D:\python_code\handover\cha4`

重点加入**星间路径信息表/路径质量广播更新机制**：

- 每颗卫星维护到各地面站网段的路径表：`目的网段、下一跳、路径质量 Q、跳数 d`；
- 与地面站直连的落地卫星：对应网段 `Q=0, d=0`；
- 非落地卫星按递推式更新：`Q_i,k = Q_j,k + l_i,j`，其中 `j` 是最小跳数路由确定的下一跳，`l_i,j` 是星间链路负载率；
- 按 `path_update_interval_s` 秒级周期广播，终端状态中使用路径质量和跳数辅助切换决策。

## 主要文件

| 文件 | 作用 |
|---|---|
| `config_ch4.py` | 统一参数配置，默认随机种子 42、256 星、8 地面站、200 训练用户、Bc=10 MHz |
| `build_dataset_ch4.py` | 生成星地链路 + 星间路径表数据集 |
| `path_table.py` | 星间邻接、链路负载、路径质量广播更新机制 |
| `env_ch4.py` | MDP 环境，状态为上一接入/Cg/跳数/Q/上一速率/业务类型 |
| `agents.py` | DQN、DDQN、Dueling-DDQN 智能体 |
| `train_ch4.py` | 训练 Dueling-DDQN/DDQN/DQN |
| `eval_ch4.py` | 评估 Dueling-DDQN、DDQN、最大信道容量、最大可见时长 |
| `plot_ch4.py` | 绘制收敛曲线、图 4-6 对比图、可选 3D 曲面图 |
| `inspect_path_table_ch4.py` | 打印某颗卫星的路径信息表 |
| `stk_generate_ch4.py` | 可选：调出 STK 创建第四章可视化场景 |
| `run_ch4_all.py` | 一键运行数据集、训练、评估、绘图 |

## 推荐运行顺序

在 PowerShell 中执行：

```powershell
Set-Location 'D:\python_code\handover\cha4'

# 1. 生成数据集（默认 20分钟、5秒时隙、10MHz、256星、300用户）
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe build_dataset_ch4.py

# 2. 查看图片对应的路径信息表字段
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe inspect_path_table_ch4.py --time-index 0 --sat 0

# 3. 训练第四章核心算法 Dueling-DDQN
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe train_ch4.py --algo dueling --episodes 300 --users 200

# 如需论文图 4-6 中 DDQN 对比，再训练 DDQN
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe train_ch4.py --algo ddqn --episodes 300 --users 200

# 4. 评估并保存 CSV
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe eval_ch4.py --users 200 --eval-episodes 3

# 5. 绘图
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe plot_ch4.py
```

一键运行：

```powershell
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe run_ch4_all.py --algo all --episodes 300 --users 200
```

## 常用参数

```powershell
# 20 MHz 复现
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe build_dataset_ch4.py --bc-mhz 20
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe train_ch4.py --bc-mhz 20 --algo all

# 改为 1 秒时隙（数据集和训练必须保持一致）
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe build_dataset_ch4.py --slot-seconds 1
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe train_ch4.py --slot-seconds 1 --algo dueling

# 三维曲面数据与绘图
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe eval_ch4.py --surface
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe plot_ch4.py --surface

# 学习率对比（图 4-4）
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe train_ch4.py --lr 0.001 --tag lr1e-3
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe train_ch4.py --lr 0.0001 --tag lr1e-4
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe train_ch4.py --lr 0.01 --tag lr1e-2
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe plot_ch4.py
```

## 输出目录

- 数据集：`D:\python_code\handover\cha4\outputs\ch4\dataset`
- 模型：`D:\python_code\handover\cha4\outputs\ch4\models`
- 日志：`D:\python_code\handover\cha4\outputs\ch4\logs`
- 图片：`D:\python_code\handover\cha4\outputs\ch4\figures`

## STK 可视化

默认会调出 STK 界面：

```powershell
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe stk_generate_ch4.py --users 50
```

如果想后台运行才加 `--hidden`。
