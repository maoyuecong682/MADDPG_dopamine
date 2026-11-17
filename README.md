# SMARTS 双车驾驶：MADDPG 基线与 Robo-Dopamine 改进

本项目研究一个问题：在变道、汇入和无保护左转中，引入任务进度后，两辆车能否更安全地完成任务？五组实验使用相同的 SMARTS 场景、观测、动作和 MADDPG 网络，通过训练参数切换进度估计、奖励和经验回放。默认运行纯 MADDPG 基线。

**当前进度**：五种配置均已通过 CPU 冒烟训练、模型重载和独立评估。每组短跑 360 个环境步、329 次联合更新，每个场景评估 1 回合，均未出现两车安全到达。正式效果比较需要下文的三种子实验。

## 1. 从项目根目录开始

**本机 WSL** 已有 Ubuntu 24.04、Python 3.11 和可用的 CPU 环境。打开 Ubuntu 终端：

```bash
cd ~/projects/rl_final
source ~/venvs/a4-baseline/bin/activate
```

**新机器**使用 Linux x86_64 或 WSL2，在项目根目录用 Python 3.11 创建环境。以下为 CPU 安装示例：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

A5 的 GPU 机器在最后一步安装与驱动兼容的 CUDA 版 PyTorch。运行 `python -c "import torch; print(torch.cuda.is_available())"` 确认输出为 `True`，再用 `--device cuda` 训练。[PyTorch 安装入口](https://pytorch.org/get-started/locally/)

**建图并运行测试**。场景生成器根据随项目提供的统计文件构建 3 种任务 × 4 档交通条件。跨机器复现时，在新环境中重新生成场景：

```bash
export PYTHONHASHSEED=0
netconvert --version
python scenarios.py
python -m unittest -v test_baseline test_dopamine
```

**跑一次完整版**。已有冻结的进度模型 H，可直接用于训练。将第一条命令的 `--mode dopamine` 改为 `--mode baseline` 即可跑基线。每次训练选一个新的输出目录，再用同一目录中的检查点评估：

```bash
python run.py train --mode dopamine --seed 1 --device cpu --steps 360 --warmup 32 --batch 32 --horizon 60 --out runs/quick-full
python run.py eval --model runs/quick-full/model.pt --episodes 4 --horizon 300 --out runs/quick-full
```

`--steps` 是三个任务合计的环境步数，两辆车同时行动计一步。`--horizon` 是单回合上限，每步 0.1 秒。`eval --episodes 4` 表示**每个场景**测 4 回合，共 12 回合。

## 2. 环境、数据和基础算法

| 任务 | 两辆策略车要完成的动作 |
|---|---|
| 变道 | 从不同车道出发，各自到达另一条车道上的目标。 |
| 汇入 | 匝道车与主路车进入同一下游车道。 |
| 无保护左转 | 一车左转，一车对向直行。另有一名背景行人以 1.2 m/s 沿路侧固定路线行走，仅两辆车参与策略更新。 |

INTERACTION DR v1.1 两个站点的 33 段记录、13,339 条汽车轨迹用于离线标定背景车速、发车率、变道间距和汇入相对位置。日常建图读取已计算的统计文件；训练样本由两辆策略车与 SMARTS 在线交互生成。[统计结果](interaction.json) · [校准方法与原始数据复现步骤](calibration.md)

每车 actor 根据 24 维局部观测输出纵向控制和转向。训练时，每车的 critic 读取两车同步的观测、动作与存活标记。基础奖励取该车行驶距离增量，发生碰撞、驶离道路、偏离路线、逆行或压路肩时扣 10，无安全违规且到达目标时加 10。一辆车退出后，另一辆仍可继续行动。

## 3. 五种实验配置

| `--mode` | 进度估计 | 写入回放的奖励 | 联合样本采样 |
|---|---|---|---|
| `baseline` | 关闭 | 基础奖励 | 全池均匀 |
| `dopamine` | B/I/F 融合 | 归一化＋势函数塑形 | 任务→进度桶→转移 |
| `no-reward` | B/I/F 融合 | 基础奖励 | 任务→进度桶→转移 |
| `f-only` | 仅起点参照 F | 归一化＋势函数塑形 | 任务→进度桶→转移 |
| `no-replay` | B/I/F 融合 | 归一化＋势函数塑形 | 全池均匀 |

**进度模型 H** 分别以目标 B、上一时刻 I 和起点 F 为参照，预测相对进步或退步，再换算到同一尺度取均值。随项目提供的 H 使用独立 SMARTS 轨迹训练，按完整回合分为训练 24、验证 6、测试 6。B/I/F 在留出回合上的进度 MAE 为 0.0040，衡量的是对规则标签的拟合程度。[冻结权重](progress_artifacts/h.pt) · [离线报告](progress_artifacts/report.json)

**奖励与回放**：每车的改进奖励为 `r_base / C_task + β × (γ × φ_next − φ_current)`。真正终止时下一势值置零，时间截断时保留。默认 β=1，三个任务的尺度依次为 100、120、120。分层回放先抽非空任务，再抽非空进度桶，最后抽取一条双车联合转移。默认每任务 5 桶，以转移开始时仍在场车辆的平均进度分桶。[公式、单项开关、消融设计与论文依据](dopamine.md)

## 4. A5 正式训练与评估

五种配置各跑 3 个训练种子，共 15 次训练。先在目标机器上校准耗时，再统一确定环境步预算。下例按每次 50,000 环境步执行。各组共用同一份 H、测试种子和交通档位，评估关闭探索噪声：

```bash
set -e
export PYTHONHASHSEED=0
for mode in baseline dopamine no-reward f-only no-replay; do
  for seed in 1 2 3; do
    d="runs/formal/$mode/seed-$seed"
    python run.py train --mode "$mode" --seed "$seed" --device cuda --steps 50000 --out "$d"
    python run.py eval --device cuda --model "$d/model.pt" --episodes 20 --horizon 300 --out "$d"
  done
done
```

每个模型共测 60 回合。按场景先计算每个训练种子的指标，再汇报三个种子的均值和样本标准差：

| 指标 | 统计口径 |
|---|---|
| 碰撞率 ↓ | 任一受控车发生碰撞，该回合计一次。 |
| 安全成功率 ↑ | 两辆车都在时限内到达各自目标，且全回合无安全违规。 |
| 成功通行时间 ↓ | 仅对成功回合统计，以最后一辆车到达时的仿真时间为准。无成功样本时为 `null`。 |
| 冲突窗口内急减速次数 ↓ | 速度变化率低于 −3 m/s²，同一辆车连续满足条件计一次，两车相加后按回合取均值。 |

## 5. 代码和运行产物

| 文件 | 用途 |
|---|---|
| [scenarios.py](scenarios.py) · [environment.py](environment.py) | 构建校准场景，处理观测、动作、基础奖励与安全事件。 |
| [interaction_stats.py](interaction_stats.py) | 从原始 INTERACTION 轨迹重新计算场景参数。 |
| [maddpg.py](maddpg.py) · [run.py](run.py) | 定义网络、均匀回放与更新规则，提供混合训练和分场景评估入口。 |
| [fit_progress.py](fit_progress.py) · [progress.py](progress.py) | 采集独立轨迹、拟合 H，并在训练时估计进度。 |
| [progress_replay.py](progress_replay.py) | 管理任务与进度桶，保持双车样本按仿真时刻对齐。 |
| [test_baseline.py](test_baseline.py) · [test_dopamine.py](test_dopamine.py) | 测试真实场景、固定行人、网络更新、进度递推、终止处理和分层采样。 |

每次训练的输出目录包含 `config.json`（实际开关与 H 摘要）、`train.jsonl`（逐回合日志）和 `model.pt`（供评估加载的检查点）。日志里的 `returns` 是基础奖励回报，`training_returns` 是用于更新 critic 的回报。评估生成 `evaluation.json`，其中 `per_task` 是场景汇总，`episodes` 保留逐回合记录。

版本管理保留源码、校准统计、冻结 H 及报告。[smoke/](smoke/) 保存最初场景的历史冒烟记录。本机的 `runs/interaction-smoke/` 和 `runs/dopamine-smoke/` 分别存放校准后的基线及五种配置的冒烟结果。`runs/` 和生成的 `scenarios/` 由 Git 忽略，新的检出目录可按第 1 节重建。
