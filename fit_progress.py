"""Fit H once using separate SMARTS episodes; split whole episodes before pairing."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from environment import Episode, IDS
from progress import HopModel, Progress, hop_target, inputs, labels, positions, references
from scenarios import TASKS


def collect(task, seed, variant, horizon):
    rng = np.random.default_rng(seed)
    env = Episode(task, seed, variant)
    try:
        start, goal = references(env)
        xy = start.copy()
        tracks = [[p.copy()] for p in start]
        truth = [[0.] for _ in IDS]
        cruise = rng.uniform(5, 10, len(IDS))
        for step in range(horizon):
            alive = env.active.copy()
            actions = np.zeros((2,2), np.float32)
            for i in np.flatnonzero(alive):
                ob = env.raw[IDS[i]]
                ego = ob.ego_vehicle_state
                paths = [p for p in ob.waypoint_paths if p]
                if not paths:
                    continue
                # Prefer the path that ends closest to our goal, including target lane.
                path = min(paths, key=lambda p: np.linalg.norm(p[-1].pos[:2]-goal[i]))
                target = path[min(10,len(path)-1)].pos[:2]
                if task == "lane_change":
                    target = np.array([xy[i,0]+10,goal[i,1]])
                dx,dy = target-xy[i]
                desired = np.arctan2(-dx,dy)
                error = np.arctan2(np.sin(desired-ego.heading),np.cos(desired-ego.heading))
                actions[i] = [np.clip(.15*(cruise[i]-ego.speed),-.5,.6),
                              np.clip(-1.6*error+rng.normal(0,.035),-.7,.7)]
            env.step(actions)
            xy = positions(env.raw, xy)
            for i in np.flatnonzero(alive):
                tracks[i].append(xy[i].copy())
                truth[i].append(labels(task,xy[i],start[i],goal[i],env.goal[i]))
            if not env.active.any():
                break
        return [(task,start[i],goal[i],np.asarray(tracks[i]),np.array(truth[i],np.float32))
                for i in range(len(IDS))]
    finally:
        env.close()


def pairs(tracks, rng):
    xs, ys = [], []
    for task, start, goal, trajectory, truth in tracks:
        # The mission anchor is known even if the car failed; never relabel a failure as success.
        points = np.vstack([trajectory, goal])
        values = np.append(truth, 1.)
        n = len(trajectory)
        b = rng.integers(1,n,256)
        a = np.concatenate([np.zeros_like(b),np.full_like(b,n),b-1,b,rng.integers(0,n,len(b))])
        b = np.concatenate([b,b,b,b-1,rng.integers(0,n,len(b))])
        xs.append(inputs(task,start,goal,points[a],points[b]))
        ys.append(hop_target(values[a],values[b]))
    return torch.tensor(np.concatenate(xs)), torch.tensor(np.concatenate(ys),dtype=torch.float32)


def score(model, tracks):
    metrics = {}
    for mode in ("bif","f","i","b"):
        errors, hits = [], []
        for task,start,goal,path,target in tracks:
            estimator = Progress(model,task,start[None],goal[None],mode)
            preds = [0.]
            for pos in path[1:]:
                preds.append(estimator.update(pos[None],np.array([True]))[0])
            errors.extend(np.abs(target-preds))
            diff = np.diff(target)
            moving = np.abs(diff)>1e-4
            hits.extend((np.sign(np.diff(preds)[moving])==np.sign(diff[moving])).tolist())
        metrics[mode] = dict(mae=float(np.mean(errors)), direction_accuracy=float(np.mean(hits)) if hits else None,
                             states=len(errors), directional_steps=len(hits))
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--episodes",type=int,default=8,help="training episodes per task")
    p.add_argument("--heldout",type=int,default=2,help="episodes per task in each validation/test split")
    p.add_argument("--epochs",type=int,default=70)
    p.add_argument("--horizon",type=int,default=220)
    args=p.parse_args()
    if min(args.episodes,args.heldout,args.epochs,args.horizon)<1:
        p.error("counts must be positive")
    args.out.mkdir(parents=True,exist_ok=True)
    if any((args.out/name).exists() for name in ("h.pt","report.json")):
        p.error("use a new output directory")
    torch.set_num_threads(1)
    torch.manual_seed(19)
    rng=np.random.default_rng(19)
    splits, manifest = {}, []
    for split,base,count in (("train",1200000000,args.episodes),("validation",1300000000,args.heldout),
                             ("test",1400000000,args.heldout)):
        tracks=[]
        for j in range(count):
            for i,task in enumerate(TASKS):
                seed=base+3*j+i
                variant=(base//100000000+j)%4
                traces=collect(task,seed,variant,args.horizon)
                tracks.extend(traces)
                manifest.append(dict(split=split,seed=seed,task=task,variant=variant,
                                     lengths=[len(t[3]) for t in traces],
                                     max_labels=[float(t[4].max()) for t in traces]))
        splits[split]=tracks
        print(split, len(tracks), "car trajectories",flush=True)
    x,y=pairs(splits["train"],rng)
    vx,vy=pairs(splits["validation"],rng)
    model=HopModel()
    opt=torch.optim.Adam(model.parameters(),lr=.002)
    best=float("inf")
    for epoch in range(args.epochs):
        for idx in torch.randperm(len(x)).split(256):
            loss=(model(x[idx]).squeeze(-1)-y[idx]).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            error=(model(vx).squeeze(-1)-vy).abs().mean().item()
        if error<best:
            best, weights, selected=error,deepcopy(model.state_dict()),epoch+1
    model.load_state_dict(weights)
    result=dict(format=1,selected_epoch=selected,val_hop_mae=best,test=score(model,splits["test"]),
                train_pairs=len(x),split_manifest=manifest)
    (args.out/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    torch.save(dict(format=1,weights=weights),args.out/"h.pt")
    print(json.dumps(result["test"]))


if __name__=="__main__":
    main()

