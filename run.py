"""Train and evaluate the two-car SMARTS baseline."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from environment import ACT_DIM, IDS, OBS_DIM, Episode
from maddpg import MADDPG, Replay
from scenarios import TASKS
from progress import HopModel, Progress, positions, references
from progress_replay import ProgressReplay


PRESETS = {
    "baseline": ("off", "base", "uniform"),
    "dopamine": ("bif", "shaped", "stratified"),
    "no-reward": ("bif", "base", "stratified"),
    "f-only": ("f", "shaped", "stratified"),
    "no-replay": ("bif", "shaped", "uniform"),
}


def shaped_reward(base, current, following, active, next_active, scale, beta, gamma):
    potential = np.where(next_active, following, 0.)
    return np.where(active, base/scale + beta*(gamma*potential-current), 0.).astype(np.float32)


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def aggregate(rows):
    result = {}
    for task in TASKS:
        group = [r for r in rows if r["task"] == task]
        times = [r["passage_s"] for r in group if r["success"]]
        result[task] = dict(episodes=len(group), collision_rate=np.mean([r["collision"] for r in group]),
                            success_rate=np.mean([r["success"] for r in group]),
                            mean_success_time_s=np.mean(times) if times else None,
                            mean_hard_brakes=np.mean([r["hard_brakes"] for r in group]))
    return result


def train(args, model, rng):
    progress_mode, reward_mode, replay_mode = PRESETS[args.mode]
    progress_mode = args.progress or progress_mode
    reward_mode = args.reward or reward_mode
    replay_mode = args.replay or replay_mode
    scales = dict(zip(TASKS,args.task_scales))
    if not np.isfinite(args.beta) or not all(np.isfinite(s) and s>0 for s in scales.values()):
        raise ValueError("beta must be finite; task scales must be finite and positive")
    use_progress = progress_mode != "off"
    if not use_progress and (reward_mode=="shaped" or replay_mode=="stratified"):
        raise ValueError("Select --progress bif or f to supply progress for reward/replay.")
    hop_model = HopModel.load_frozen(args.h_model) if use_progress else None
    config = dict(mode=args.mode,progress=progress_mode,reward=reward_mode,replay=replay_mode,
                  scales=scales,beta=args.beta,bins=args.bins,gamma=model.gamma,
                  h_sha256=hashlib.sha256(args.h_model.read_bytes()).hexdigest() if use_progress else None)
    replay = (ProgressReplay(100000,rng,args.bins,replay_mode=="stratified")
              if use_progress else Replay(100000, rng))
    steps = updates = episode = 0
    losses = None
    start = time.monotonic()
    with (args.out / "train.jsonl").open("x") as log:
        (args.out/"config.json").write_text(json.dumps(config,indent=2))
        while steps < args.steps:
            task = TASKS[episode % len(TASKS)]
            # Reserve a disjoint seed range for evaluation.
            seed = (args.seed * 100003 + episode) % 1000000000
            variant = (args.seed + episode // len(TASKS)) % 4
            env = Episode(task, seed, variant)
            try:
                if use_progress:
                    estimator = Progress(hop_model,task,*references(env),progress_mode)
                training_return = np.zeros(len(IDS))
                sample_seconds = 0.
                while env.active.any() and env.steps < args.horizon and steps < args.steps:
                    state, alive = env.state, env.active.copy()
                    if steps < args.warmup:
                        act = rng.uniform(-1, 1, (len(IDS), ACT_DIM))
                    else:
                        act = model.act(state) + rng.normal(0, .15, (len(IDS), ACT_DIM))
                    act = act.clip(-1, 1).astype(np.float32) * alive[:, None]
                    reward = env.step(act)
                    if use_progress:
                        current = estimator.value.copy()
                        following = estimator.update(positions(env.raw,estimator.previous),alive)
                        if reward_mode=="shaped":
                            reward = shaped_reward(reward,current,following,alive,env.active,
                                                   scales[task],args.beta,model.gamma)
                        replay.add_joint(task,current,episode,state,act,reward,env.state,alive,env.active)
                    else:
                        replay.add(state, act, reward, env.state, alive, env.active)
                    training_return += reward
                    steps += 1
                    if steps >= args.warmup and len(replay.rows) >= args.batch:
                        t0 = time.perf_counter()
                        batch = replay.sample(args.batch, args.device)
                        sample_seconds += time.perf_counter()-t0
                        losses = model.update(batch)
                        updates += 1
                record = dict(task=task, seed=seed, variant=variant, env_steps=steps, updates=updates,
                              losses=losses, training_returns=training_return.tolist(), sample_s=sample_seconds,
                              wall_s=round(time.monotonic()-start, 3), **env.metrics())
                if use_progress:
                    record.update(progress=estimator.value.tolist(), sampled_task_bins=replay.sample_counts.tolist())
                print(json.dumps(record), file=log, flush=True)
            finally:
                env.close()
            episode += 1
    if updates == 0:
        raise RuntimeError("Training finished with zero updates. Increase steps or reduce warmup/batch.")
    torch.save(dict(model=model.state_dict(), steps=steps, updates=updates, seed=args.seed, config=config),
               args.out / "model.pt")
    print(f"trained {steps} steps, {updates} joint updates, {episode} episodes")


def evaluate(args, model):
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = []
    for task in TASKS:
        for i in range(args.episodes):
            seed = 2000000000 + i
            variant = i % 4
            env = Episode(task, seed, variant)
            try:
                while env.active.any() and env.steps < args.horizon:
                    env.step(model.act(env.state))
                rows.append(dict(task=task, seed=seed, variant=variant, **env.metrics()))
            finally:
                env.close()
    result = dict(model=str(args.model), training_seed=checkpoint["seed"],
                  config=checkpoint.get("config",{"mode":"baseline"}),
                  training_steps=checkpoint["steps"], updates=checkpoint["updates"],
                  horizon=args.horizon, per_task=aggregate(rows), episodes=rows)
    with (args.out / "evaluation.json").open("x") as report:
        json.dump(result, report, indent=2, allow_nan=False)
    print(json.dumps(result["per_task"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    train_cmd = commands.add_parser("train")
    train_cmd.add_argument("--steps", type=positive, default=50000)
    train_cmd.add_argument("--batch", type=positive, default=128)
    train_cmd.add_argument("--warmup", type=positive, default=1000)
    train_cmd.add_argument("--seed", type=int, default=1)
    train_cmd.add_argument("--mode", choices=PRESETS, default="baseline")
    train_cmd.add_argument("--progress",choices=["off","bif","f","b","i"])
    train_cmd.add_argument("--reward",choices=["base","shaped"])
    train_cmd.add_argument("--replay",choices=["uniform","stratified"])
    train_cmd.add_argument("--bins",type=positive,default=5)
    train_cmd.add_argument("--beta",type=float,default=1.)
    train_cmd.add_argument("--task-scales",nargs=3,type=float,default=[100.,120.,120.],
                           metavar=("LANE_CHANGE","MERGE","LEFT_TURN"))
    train_cmd.add_argument("--h-model",type=Path,default=Path(__file__).resolve().parent/"progress_artifacts/h.pt")
    eval_cmd = commands.add_parser("eval")
    eval_cmd.add_argument("--model", required=True, type=Path)
    eval_cmd.add_argument("--episodes", type=positive, default=20, help="per task")
    for cmd in (train_cmd, eval_cmd):
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        cmd.add_argument("--horizon", type=positive, default=300)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    seed = getattr(args, "seed", 0)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = MADDPG(len(IDS), OBS_DIM, ACT_DIM, args.device)
    if args.command == "train":
        train(args, model, np.random.default_rng(seed))
    else:
        evaluate(args, model)


if __name__ == "__main__":
    main()

