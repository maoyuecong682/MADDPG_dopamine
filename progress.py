"""Low-dimensional hop model and B/I/F estimation for the three SMARTS tasks."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from environment import IDS
from scenarios import TASKS


def positions(raw, fallback):
    result = fallback.copy()
    for i, key in enumerate(IDS):
        if key in raw:
            result[i] = raw[key].ego_vehicle_state.position[:2]
    return result


def references(env):
    start = positions(env.raw, np.zeros((len(IDS), 2), np.float32))
    goal = np.array([env.raw[key].ego_vehicle_state.mission.goal.position[:2]
                     for key in IDS], dtype=np.float32)
    return start, goal


def labels(task, xy, start, goal, reached):
    """Offline geometric supervision only. Never called by MADDPG training."""
    if task == "lane_change":
        longitudinal = (xy[0]-start[0])/(goal[0]-start[0])
        lateral = 1 - abs(goal[1]-xy[1])/max(abs(goal[1]-start[1]), .5)
        p = .85*np.clip(longitudinal, 0, 1) + .15*np.clip(lateral, 0, 1)
    else:
        junction = np.array((75., 50.) if task == "merge" else (80., 80.))
        points = np.stack([start, junction, goal])
        edges = np.diff(points, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        t = np.clip(((xy-points[:-1])*edges).sum(1)/lengths**2, 0, 1)
        offset = np.linalg.norm(points[:-1]+edges*t[:,None]-xy, axis=1)
        k = offset.argmin()
        p = (lengths[:k].sum()+lengths[k]*t[k]-offset[k])/lengths.sum()
    return 1. if reached else float(np.clip(p, 0, .99))


def hop_target(p, q):
    delta = q-p
    return delta / np.maximum(np.where(delta >= 0, 1-p, p), 1e-7)


def advance(p, hop):
    return np.clip(p + hop*np.where(hop >= 0, 1-p, p), 0, 1)


def inputs(task, start, goal, before, after):
    # Coordinates in a route-aligned frame, normalized by start-goal distance.
    route = goal-start
    length = np.maximum(np.linalg.norm(route, axis=-1, keepdims=True), 1)
    unit = route/length
    normal = np.stack([-unit[...,1],unit[...,0]], axis=-1)

    def project(pos):
        diff = (pos-start)/length
        return np.stack([(diff*unit).sum(-1),(diff*normal).sum(-1)], -1)

    a, b = project(before), project(after)
    return np.concatenate([a, b, b-a, np.broadcast_to(unit,a.shape),
                           np.broadcast_to(length/100,(*a.shape[:-1],1)),
                           np.broadcast_to(np.eye(3)[TASKS.index(task)], (*a.shape[:-1],3))],
                          axis=-1).astype(np.float32)


class HopModel(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(12, 64), nn.SiLU(), nn.Linear(64,64),
                         nn.SiLU(), nn.Linear(64,1), nn.Tanh())

    @classmethod
    def load_frozen(cls, path):
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
        if ckpt["format"] != 1:
            raise ValueError("Unsupported H checkpoint")
        model = cls()
        model.load_state_dict(ckpt["weights"])
        return model.eval().requires_grad_(False)


class Progress:
    def __init__(self, model, task, start, goal, mode):
        self.model, self.task, self.mode = model, task, mode
        self.start, self.goal, self.previous = start, goal, start.copy()
        self.value = np.zeros(len(start), dtype=np.float32)

    @torch.inference_mode()
    def update(self, current, active):
        anchors = {"f":self.start, "b":self.goal, "i":self.previous}
        modes = ("f","b","i") if self.mode == "bif" else (self.mode,)
        features = np.stack([inputs(self.task,self.start,self.goal,anchors[m],current) for m in modes])
        hops = self.model(torch.from_numpy(features)).squeeze(-1).numpy()
        absolute = []
        for m, h in zip(modes, hops):
            if m == "b":
                p = 1+h
            elif m == "i":
                p = advance(self.value, h)
            else:
                p = h
            absolute.append(np.clip(p, 0, 1))
        self.value = np.where(active, np.mean(absolute, axis=0), self.value)
        self.previous = current.copy()
        return self.value

