# SMARTS 双车 MADDPG 基线

这个项目把两辆独立策略车放入同一个 SMARTS 仿真回合。两车各自读取局部传感器并输出连续控制，训练时各自的 critic 可以看到两车的信息。变道、汇入和无保护左转轮流训练，共享同一套模型和均匀回放池。附带的 smoke/model.pt 是 360 步的 CPU 冒烟检查点，可用来复现管线和诊断失败行为。

## 1. 本机复现

已验证环境为 Windows 主机上的 WSL2 Ubuntu 24.04、Python 3.11.15、SMARTS 2.0.1、PyTorch CPU。WSL 工作目录为 ~/projects/rl_final。本机已经建好了 a4-baseline 环境，其 PyTorch 来自已有的 snn 环境。打开 Ubuntu 终端执行：

```bash
cd ~/projects/rl_final
source ~/venvs/a4-baseline/bin/activate
export PYTHONHASHSEED=0
python scenarios.py
python -m unittest -v test_baseline
python run.py train --seed 1 --steps 360 --warmup 32 --batch 32 --horizon 60 --out runs/smoke-1
python run.py eval --model runs/smoke-1/model.pt --episodes 4 --horizon 300 --out runs/smoke-1
```

已有检查点可跳过训练，直接运行：

```bash
python run.py eval --model smoke/model.pt --episodes 4 --out runs/recheck
```

在 **A5 的新机器**上使用 Linux 或 WSL2、Python 3.11，进入项目后准备独立环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements.txt
# CPU 机器
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
export PYTHONHASHSEED=0
python scenarios.py
python -m unittest -v test_baseline
```

GPU 机器用与显卡及驱动兼容的 CUDA 版 PyTorch 替换上面的 CPU 安装命令，然后验证 `python -c "import torch; print(torch.cuda.is_available())"` 输出 True。SUMO 的 netconvert 由 eclipse-sumo 提供，激活环境后应能执行 `netconvert --version`。场景生成不需要下载数据集。各次训练和评估使用新的输出目录，避免混合不同运行的日志。

## 2. 从场景到一条训练样本

| 文件 | 主要入口 | 负责的工作 |
|---|---|---|
| scenarios.py | build(task) | 用 netconvert 构建 SUMO 路网，用 SMARTS Scenario Studio 写入两车任务和背景车流。 |
| environment.py | Episode、encode、actuator | 创建 HiWayEnvV1，读取原始观测，完成一次联合动作，计算奖励，累计回合指标。 |
| maddpg.py | MADDPG、Replay | 维护每车的 actor、critic 和目标网络，存储并均匀抽取同步的双车转移，更新参数。 |
| run.py | train、evaluate | 控制种子、探索、任务轮换和步数，保存模型，加载检查点后独立评估。 |
| test_baseline.py | BaselineTests | 检查覆盖写入与同步采样、终止掩码、梯度更新，以及三个真实场景中的双车控制。 |
| smoke/ | train.jsonl、model.pt、evaluation.json | 本次实测的训练记录、检查点、逐回合指标及场景汇总。 |

**场景。** 两车初速 6 m/s，步长 0.1 s。变道场景是一条双车道，两车从不同车道出发，目标在各自起点的另一条车道。汇入场景让匝道车和主路车驶向同一条下游车道。无保护左转场景让一车从西向北左转，另一车从东向西直行，路口按优先权通行。每个场景有四档背景流量，分别为每小时 320、480、640、800 辆的发车设置。背景车由仿真器控制。这里使用程序生成的车辆交通，未使用 INTERACTION 轨迹；行人和非机动车场景尚未覆盖。

**观测。** 每辆车输入 24 个浮点数。前 2 个为速度和转向，随后 2 个为相对目标位置，6 个为最近车道参考路径上三个路点的相对位置。最后 14 个描述距离最近的两辆邻车，每车 7 项，依次为存在标记、相对纵横位置、速度、相对航向的正弦与余弦、纵向相对速度。位置转换到自车前向和右向坐标系，缺失邻车补零。距离和速度按固定物理尺度缩放。目标位置是任务给定的导航输入。

**动作与奖励。** actor 输出两个介于 −1 和 1 的值。第一维为有符号纵向指令，正数映射到油门，负数的绝对值映射到刹车；第二维直接映射到 SMARTS 转向。每车每步奖励为 SMARTS 原始行驶距离增量，发生碰撞、驶离道路、偏离路线、逆行或压路肩任一事件扣 10，无安全事件且到达目标加 10。同一步多项违规只扣一次，并配置这些违规终止对应车辆。±10 是项目选定的基础奖励系数，各实验组应共用。此次使用普通距离、安全和到达奖励及全池均匀采样。

**一条转移。** Replay 的每行是 `(obs, act, reward, next_obs, alive, next_alive)`。其中 obs 是 [2,24]，act 是 [2,2]，奖励和布尔掩码都是 [2]。同一行的两车数据来自同一仿真时刻，每个 batch 对整行均匀无放回抽样。已退出车辆的状态和动作补零，critic 输入另含存活标记。单车退出时停止对其后续占位行计算损失，另一辆车可继续。接口在真正终止时关闭 bootstrap。外部 horizon 和训练步数用尽属于采样截断，保留最后真实下一状态的 bootstrap，下一回合重新 reset。

## 3. MADDPG 怎么更新

每车单独有一个 actor 和一个 critic，均为两层 128 单元 ReLU 的 MLP。actor 的末层用 tanh。每个 critic 接收两车 48 维观测、4 维动作、2 维存活标记，输出本车的标量 Q 值。

1. 训练初期均匀随机探索。之后使用 actor 动作叠加标准差 0.15 的高斯噪声，并裁剪到合法范围。
2. 目标 actor 产生下一步联合动作。第 i 辆车的 TD 目标为 `r_i + gamma * next_alive_i * target_Q_i(next_joint)`。critic 最小化有效样本上的均方误差。
3. 更新第 i 个 actor 时，仅用该 actor 的输出替换回放中本车的动作，保留同期另一车的真实动作，最大化 critic 对本车的估值。
4. 每次更新后，目标参数取 `(1-tau)*旧目标 + tau*在线参数`。评估只调用两车 actor，关闭噪声、采样和梯度更新。

正式默认值为回放容量 100000，batch 128，warmup 1000 环境步，折扣 0.99，软更新系数 0.01。actor 和 critic 的 Adam 学习率分别为 0.0001、0.001。满足 warmup 和 batch 条件后，每个环境步做一次联合更新。两车同时行动算一个环境步。日志 losses 依次记录最近一次更新中各有效车辆的 critic MSE 均值和 actor 损失均值。model.pt 保存在线和目标网络的权重以及训练步数、种子和更新计数。它支持独立评估，未保存优化器和回放池以供断点续训。

## 4. 本次测试和评估

三个自动化测试通过。360 步短跑使用种子 1、batch 32、warmup 32、每回合最多 60 步，覆盖全部三种任务，8 段采样共 329 次联合更新。随后从磁盘加载模型，在单独的评估种子上每个任务测 4 回合，限时均为 300 步（30 s），覆盖四档车流。

| 任务 | 回合数 | 碰撞率 | 安全成功率 | 成功回合平均通行时间 | 窗口内急减速次数/回合 |
|---|---:|---:|---:|---:|---:|
| 变道 | 4 | 100% | 0% | 无成功样本 | 0 |
| 汇入 | 4 | 100% | 0% | 无成功样本 | 0 |
| 无保护左转 | 4 | 100% | 0% | 无成功样本 | 0 |

碰撞率按回合计数，两辆受控车中任意一辆发生碰撞即计一次。成功要求两辆车在时限内均到达自己的目标且全回合无上述安全事件。成功用时取最后一辆车完成时的仿真时间；没有成功样本时 JSON 中为 null。急减速用相邻步车速差除以 0.1 s，严格低于 −3 m/s² 时计入；同一车在窗口内连续满足条件只计一次，两车次数相加。变道窗口为地图 x 坐标 35 至 100 m，汇入和左转窗口分别为距汇合点 (75,50) 和路口中心 (80,80) 25 m 以内。

在变道评估种子 2000000000 上，两车第一步纵向动作约为 −0.989 和 −0.978，约 1.5 s 后车速接近零，其中一辆被背景车追尾。相同起点给固定 0.35 油门、零转向时，两车在 3 s 内均由 6 m/s 加速至约 10.9 m/s。该对照检查了控制链路。停在冲突窗口外也可能得到零次窗口内急减速，不能单独据此判断舒适性。当前权重尚未学会完成驾驶任务；短跑已经验证在线采集、网络更新、存盘、重新加载和独立评估。

## 5. A5 正式训练

在项目根目录、已激活的 Linux 环境中执行。下面按三组种子各训练 50000 环境步，各模型在相同测试清单上每场景测 20 回合。若使用 CPU，可将 cuda 改为 cpu。

```bash
export PYTHONHASHSEED=0
for seed in 1 2 3; do
  python run.py train --device cuda --seed "$seed" --steps 50000 --out "runs/seed-$seed" || break
  python run.py eval --device cuda --model "runs/seed-$seed/model.pt" --episodes 20 --out "runs/seed-$seed" || break
done
```

train.jsonl 每行对应一个完成或截断的采样回合，含累计环境步、更新次数、累计训练墙钟时间、两车各自回报和安全事件。最后一行可能因总预算用尽而缩短。evaluation.json 的 per_task 用于逐场景汇报，episodes 保留回合种子、交通档位、到达车辆数、违规、用时和两车回报。测试种子从 2000000000 起按回合递增，交通档位按回合轮换。训练与评估使用相同的道路拓扑和四档流量，各自使用独立回合；这些结果衡量同类场景表现。正式比较宜先计算每个训练种子的各项场景指标，再报告三个训练种子的均值及样本标准差；成功用时同时列出成功样本数。此轮未执行正式三种子训练。

环境调用依据 [SMARTS 2.0.1 上游实现](https://github.com/huawei-noah/SMARTS/tree/v2.0.1)，算法基于 [Lowe 等人的 MADDPG](https://arxiv.org/abs/1706.02275)。项目中的 PyTorch 训练器、观测适配、场景与指标为本次实现。

