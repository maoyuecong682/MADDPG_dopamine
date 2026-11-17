# SMARTS-MADDPG 的 Robo-Dopamine 改进

所有模式使用同一版 INTERACTION 校准场景、固定路侧行人、双车 MADDPG 和四项驾驶指标。默认模式仍为 baseline。H 权重在 progress_artifacts/h.pt，正常运行无需重新拟合 H 或下载上游视觉模型。

## 训练和评估

在 Ubuntu 终端执行：

```bash
cd ~/projects/rl_final
source ~/venvs/a4-baseline/bin/activate
export PYTHONHASHSEED=0
python -m unittest -v test_baseline test_dopamine
python run.py train --mode dopamine --steps 360 --warmup 32 --batch 32 --horizon 60 --out runs/try-full
python run.py eval --model runs/try-full/model.pt --episodes 4 --out runs/try-full
```

| --mode | 进度 | 写入经验的奖励 | 回放采样 | 用途 |
|---|---|---|---|---|
| baseline | 关闭 | 原基础奖励 | 全池均匀 | Q1 基线 |
| dopamine | B/I/F 融合 | 归一化＋势函数塑形 | 任务→进度桶→联合样本 | Q1 完整版 |
| no-reward | B/I/F 融合 | 原基础奖励 | 分层 | Q2 |
| f-only | 仅 F，共用同一个 H | 归一化＋势函数塑形 | 分层 | Q3 |
| no-replay | B/I/F 融合 | 归一化＋势函数塑形 | 全池均匀 | Q4 |

可用 --progress off/f/b/i/bif、--reward base/shaped 和 --replay uniform/stratified 单独覆盖预设。奖励塑形或进度分桶需要打开一种进度模式。--bins 默认 5。--h-model 指定冻结模型。每次运行使用新的输出目录；eval 直接读取 actor 权重，在相同测试清单上禁用探索噪声，旧版 model.pt 也可评估。

A5 复用同一份 H、相同训练预算和测试清单。下例在已经配置 CUDA 的机器上执行。可先用同一短预算校准耗时，再统一设置正式环境步数。

```bash
set -e
export PYTHONHASHSEED=0
for mode in baseline dopamine no-reward f-only no-replay; do
  for seed in 1 2 3; do
    d="runs/formal/$mode/$seed"
    python run.py train --mode "$mode" --seed "$seed" --device cuda --steps 50000 --out "$d"
    python run.py eval --model "$d/model.pt" --device cuda --episodes 20 --out "$d"
  done
done
```

## 新增的进度模型 H

fit_progress.py 用带速度变化和转向扰动的路点跟踪器在真实 SMARTS 场景采集独立轨迹。每个回合只属于一个划分，两辆车同组。此次为训练 24 回合、验证 6 回合、测试 6 回合，每种任务均有覆盖。采集种子使用 12、13、14 亿三个独立区段，均与 RL 训练/最终测试种子范围分开。道路拓扑沿用同一批场景。报告中的 seed、variant 和 lengths 可以定位每段轨迹。进度规则是低维状态上的监督信号：变道取纵向完成度与目标车道横向完成度的 85/15 加权和；汇入和路口取沿起点、冲突点、目标折线的相对前进距离，扣除横向偏离。起始为 0，未到达的标签最高 0.99；只在 SMARTS 报告到达时给实际状态标签 1。

H 的输入是 before/after 在起终点坐标系中的二维位置、二者位移、路线方向及长度、任务 one-hot，共 12 维。两层 64 单元 SiLU 网络预测 [-1,1] 的 hop。监督目标在前进时取 (q−p)/(1−p)，退步时取 (q−p)/p。同回合内构造起点锚、已知目标锚、正序/逆序邻帧和随机帧对。失败轨迹也有地图给定的目标坐标，其最后实测状态保留真实规则标签。训练根据验证集 hop MAE 选权重，最后一次在测试集评估；在线强化学习仅加载冻结 H。H 不向 actor 提供额外观测。重训命令：

```bash
python fit_progress.py --out runs/h-refit
python run.py train --mode dopamine --h-model runs/h-refit/h.pt --out runs/full-refit
```

progress.py 对每辆车独立维护进度。F 为 H(start,now)，B 为 1+H(goal,now)。I 从上一时刻所选模式的进度 p 出发，若相邻 hop 为正，I=p+(1−p)hop；否则 I=p+p·hop。各估计限制在 [0,1]，完整版取三者均值，reset 时归零。训练时使用前后真实位置与任务既定目标，规则标签只用于 H 的离线阶段。

## 基础奖励的替换位置

run.py 在 env.step 后、回放写入前按车计算：
`r_train = r_base/C_task + beta*(gamma*phi_next - phi_current)`。这里采用作业文档给出的可执行伪代码，将进度自身作为势。gamma 直接读取 MADDPG 的折扣因子，默认 beta=1。C_task 是本实现预设的正尺度，依次为 100、120、120，是按约百米任务尺度选的超参数。使用 --task-scales 100 120 120 和 --beta 1 可显式固定。车辆真实终止时 phi_next 置零；外部时限或环境步预算截断时保留最后进度和 bootstrap；已退出车辆的占位行奖励为零。固定的每任务正尺度会改变混训时各任务的相对权重，递推估计也依赖历史，跨任务驾驶表现仍以独立评估为准。

## 联合回放和运行记录

progress_replay.py 按同一物理时刻写一条双车联合数据，奖励分别使用各自进度。按转移开始时仍在场车辆的平均进度确定该联合样本的桶，phase 即从 0 起的桶编号，进度 1 落入最后一桶。元数据含任务、各车及联合进度、phase、回合号。sample_with_metadata 返回槽位编号、六个联合张量和元数据；critic 沿用原接口。每个分层样本依次均匀选择非空任务、该任务的非空桶、桶内槽位，允许有放回。容量耗尽时用交换尾项法清理旧桶索引。全池均匀模式按原方案无放回采样。

config.json 和 model.pt 记录实际开关、折扣、beta、任务尺度、K 和 H 文件摘要。train.jsonl 的 returns 始终为基础奖励回合和，training_returns 为送入 critic 的奖励和；sampled_task_bins 是累计抽样次数（行依次变道、汇入、左转），sample_s 为本回合采样及张量组装耗时。末尾可能包含预算用尽的部分回合。evaluation.json 按原口径记录碰撞、安全成功、成功用时及急减速。

## 实现依据与验收

- 论文第 3.1 节式 (4)–(8) 给出递推、起点与目标锚及平均融合；第 3.2 节式 (10)–(11) 给出折扣势差。[CVPR 论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Tan_General_Process_Reward_Modeling_for_Robotic_Reinforcement_Learning_CVPR_2026_paper.pdf)
- 官方示例对 incremental、forward、backward 三种推理模式选不同 BEFORE 图像，参考起终点固定。课程实现将图像和任务提示替换为 SMARTS 状态及任务编码。[官方推理实现（固定提交）](https://github.com/FlagOpen/Robo-Dopamine/blob/2c714abca66329f8e2113587226bc9556726f6c2/examples/inference.py)
- 官方 GRM 数据生成脚本按相对分数及帧间隔组织离线训练对。在线 MADDPG 的任务/进度分层回放，以及基础奖励除以 C_task，来自本组 data+meth+expr.md 的设计。[官方数据生成代码](https://github.com/FlagOpen/Robo-Dopamine/blob/2c714abca66329f8e2113587226bc9556726f6c2/dataset/utils/1_generate_data.py)

交付的 H 在留出测试回合的进度 MAE 为 B/I/F 0.0040、F 0.0054、I 0.0471、B 0.0048，按车按时间步统计；方向准确率排除标签变化绝对值不超过 0.0001 的步骤。任务几何和规则标签与训练相同，结果衡量对这些规则的拟合。五组真实 CPU 冒烟结果在 runs/dopamine-smoke 下，每组 360 环境步、329 次联合更新、每场景评估 1 回合。相同设置下默认基线的 48 个参数张量与上一版逐一相同。完整版短跑实际仅采到首个进度桶，因此这轮主要验证管线；另有跨桶采样及环形覆盖的单元测试。正式三种子训练与性能比较由 A5 执行。

