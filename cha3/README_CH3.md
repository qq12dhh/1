# DDQN 用户链路切换

## 固定参数

- 仿真时长：20 分钟
- 时隙：1 秒
- 默认 `Bc = 10 MHz`，也支持 `20 MHz`
- `epsilon_min = 0.05`
- `tau = 0.005`
- 随机种子：42

## 文件说明

- `config_ch3.py`：统一参数配置
- `build_dataset_ch3.py`：生成 256 星、300 用户、1200 时隙数据集
- `channel_model.py`：信道、SNR、容量、ET/ER 计算
- `env_ch3.py`：MDP 环境
- `agents.py`：DQN/DDQN
- `baselines.py`：最大信噪比、最大可见时长
- `train_ch3.py`：训练 DDQN/DQN
- `eval_ch3.py`：评估四类算法
- `plot_ch3.py`：画图 3-3 到 3-9
- `run_ch3_all.py`：一键运行
- `stk_generate_ch3.py`：可选，在 STK 中创建场景用于可视化

## 一键运行

```powershell
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe D:\python_code\handover\run_ch3_all.py --bc-mhz 10 --episodes 1000 --train-users 100 --eval-episodes 3
```

如果使用 20 MHz：

```powershell
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe D:\python_code\handover\run_ch3_all.py --bc-mhz 20 --episodes 1000 --train-users 100 --eval-episodes 3
```

## 分步运行

```powershell
cd D:\python_code\handover
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe build_dataset_ch3.py --bc-mhz 10
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe train_ch3.py --bc-mhz 10 --episodes 1000 --users 100 --algo both
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe eval_ch3.py --bc-mhz 10 --eval-episodes 3
D:\13-DevelopSoft\30-Anaconda3\21-envs\handover\python.exe plot_ch3.py --bc-mhz 10
```

## 输出

```text
D:\python_code\handover\outputs\ch3\dataset
D:\python_code\handover\outputs\ch3\models
D:\python_code\handover\outputs\ch3\logs
D:\python_code\handover\outputs\ch3\figures
```

## 说明

`build_dataset_ch3.py` 使用内置轨道几何模型生成链路数据，速度比逐条 STK Access 更快，适合算法和画图。  
`stk_generate_ch3.py` 用于在 STK 中创建 256 星场景做可视化检查。
