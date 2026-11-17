"""Independent deterministic actors, centralized critics, uniform joint replay."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn


class Replay:
    def __init__(self, capacity, rng):
        self.rows = []
        self.capacity = capacity
        self.cursor = 0
        self.rng = rng

    def add(self, *fields):
        row = tuple(x.copy() for x in fields)
        if len(self.rows) < self.capacity:
            self.rows.append(row)
        else:
            self.rows[self.cursor] = row
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, count, device):
        indices = self.rng.choice(len(self.rows), count, replace=False)
        return [torch.as_tensor(np.stack(col), device=device, dtype=torch.float32)
                for col in zip(*(self.rows[i] for i in indices))]


def mlp(inputs, outputs, actor=False):
    layers = [nn.Linear(inputs, 128), nn.ReLU(), nn.Linear(128, 128),
              nn.ReLU(), nn.Linear(128, outputs)]
    if actor:
        nn.init.uniform_(layers[-1].weight, -0.003, 0.003)
        nn.init.zeros_(layers[-1].bias)
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class MADDPG(nn.Module):
    def __init__(self, n, obs_dim, act_dim, device="cpu", gamma=.99, tau=.01):
        super().__init__()
        self.gamma, self.tau, self.device = gamma, tau, device
        self.actors = nn.ModuleList(mlp(obs_dim, act_dim, True) for _ in range(n))
        self.critics = nn.ModuleList(mlp(n*(obs_dim+act_dim+1), 1) for _ in range(n))
        self.target_actors = deepcopy(self.actors).requires_grad_(False)
        self.target_critics = deepcopy(self.critics).requires_grad_(False)
        self.to(device)
        self.actor_opts = [torch.optim.Adam(net.parameters(), lr=1e-4) for net in self.actors]
        self.critic_opts = [torch.optim.Adam(net.parameters(), lr=1e-3) for net in self.critics]

    @torch.no_grad()
    def act(self, obs):
        local = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return torch.stack([net(local[i]) for i, net in enumerate(self.actors)]).cpu().numpy()

    @staticmethod
    def joint(obs, actions, alive):
        return torch.cat((obs.flatten(1), actions.flatten(1), alive), dim=1)

    def update(self, batch):
        obs, act, rew, nxt, alive, next_alive = batch
        joint = self.joint(obs, act, alive)
        with torch.no_grad():
            next_act = torch.stack([net(nxt[:,i]) for i,net in enumerate(self.target_actors)], 1)
            next_joint = self.joint(nxt, next_act * next_alive.unsqueeze(-1), next_alive)
        losses = []
        for i, (actor, critic) in enumerate(zip(self.actors, self.critics)):
            valid = alive[:,i].bool()
            if not valid.any():
                continue
            with torch.no_grad():
                y = rew[:,i] + self.gamma * next_alive[:,i] * self.target_critics[i](next_joint).squeeze(1)
            loss = (critic(joint).squeeze(1)[valid] - y[valid]).square().mean()
            self.critic_opts[i].zero_grad()
            loss.backward()
            self.critic_opts[i].step()

            policy_actions = act.clone()
            policy_actions[:,i] = actor(obs[:,i])
            critic.requires_grad_(False)
            objective = -critic(self.joint(obs, policy_actions, alive))[valid].mean()
            self.actor_opts[i].zero_grad()
            objective.backward()
            self.actor_opts[i].step()
            critic.requires_grad_(True)
            losses.append([loss.item(), objective.item()])
        with torch.no_grad():
            for slow, fast in ((self.target_actors, self.actors), (self.target_critics, self.critics)):
                for target, online in zip(slow.parameters(), fast.parameters()):
                    target.lerp_(online, self.tau)
        if not np.isfinite(losses).all():
            raise FloatingPointError(f"Nonfinite update: {losses}")
        return np.mean(losses, axis=0).tolist()

