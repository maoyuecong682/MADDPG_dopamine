"""SMARTS low-dimensional interface and episode-level driving metrics."""
import numpy as np
from smarts.core.agent_interface import AgentInterface, DoneCriteria, NeighborhoodVehicles, Waypoints
from smarts.core.controllers import ActionSpaceType
from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1
from scenarios import ROOT

IDS = ("car_0", "car_1")
DT = 0.1
OBS_DIM = 24
ACT_DIM = 2


def encode(obs):
    ego = obs.ego_vehicle_state
    xy = np.asarray(ego.position[:2])
    angle = float(ego.heading)
    forward = np.array([-np.sin(angle), np.cos(angle)])
    right = np.array([np.cos(angle), np.sin(angle)])
    axes = np.stack((forward, right))

    def relative(position, scale):
        return (axes @ (np.asarray(position[:2]) - xy) / scale).tolist()

    features = [ego.speed / 15, ego.steering / np.pi]
    features.extend(relative(ego.mission.goal.position, 100))
    path = min(obs.waypoint_paths or [[]],
               key=lambda p: np.linalg.norm(p[0].pos[:2] - xy) if p else np.inf)
    for index in (0, 5, 15):
        features.extend(relative(path[min(index, len(path)-1)].pos, 15) if path else [0, 0])
    neighbors = sorted(obs.neighborhood_vehicle_states or [],
                       key=lambda v: (np.linalg.norm(v.position[:2] - xy), v.id))
    for v in neighbors[:2]:
        heading = float(v.heading) - angle
        features.extend([1, *relative(v.position, 50), v.speed / 15,
                         np.sin(heading), np.cos(heading), (v.speed*np.cos(heading)-ego.speed)/15])
    features.extend([0] * (OBS_DIM - len(features)))
    return np.clip(np.asarray(features, dtype=np.float32), -5, 5)


def actuator(action):
    drive, steer = action
    return (float(max(drive, 0)), float(max(-drive, 0)), float(steer))


def in_window(task, position):
    x, y = position[:2]
    if task == "lane_change":
        return 35 <= x <= 100
    center = (75, 50) if task == "merge" else (80, 80)
    return np.hypot(x-center[0], y-center[1]) <= 25


class Episode:
    def __init__(self, task, seed, variant):
        self.task = task
        scenario = ROOT / task / str(variant)
        if not (scenario / "build" / "missions.pkl").exists():
            raise FileNotFoundError(f"Build the maps first with python scenarios.py ({scenario})")
        interface = AgentInterface(action=ActionSpaceType.Continuous,
                    neighborhood_vehicle_states=NeighborhoodVehicles(radius=50),
                    waypoint_paths=Waypoints(lookahead=20),
                    done_criteria=DoneCriteria(on_shoulder=True, wrong_way=True))
        self.env = HiWayEnvV1(scenarios=[str(scenario)], agent_interfaces={k:interface for k in IDS},
                              fixed_timestep_sec=DT, seed=seed, headless=True,
                              observation_options="unformatted", action_options="unformatted")
        try:
            self.raw, _ = self.env.reset(seed=seed)
            if set(self.raw) != set(IDS):
                raise RuntimeError(f"Expected both mission cars at reset, got {list(self.raw)}")
        except Exception:
            self.env.close()
            raise
        self.active = np.ones(2, dtype=bool)
        self.goal = np.zeros(2, dtype=bool)
        self.unsafe = self.collision = False
        self.braking = np.zeros(2, dtype=bool)
        self.brakes = self.steps = 0
        self.total = np.zeros(2)
        self.state = np.stack([encode(self.raw[k]) for k in IDS])

    def step(self, action):
        keys = [k for i,k in enumerate(IDS) if self.active[i]]
        raw, distance, done, _, _ = self.env.step({k: actuator(action[i])
                                                   for i,k in enumerate(IDS) if self.active[i]})
        state = np.zeros_like(self.state)
        reward = np.zeros(2, dtype=np.float32)
        for k in keys:
            i = IDS.index(k)
            obs = raw[k]  # SMARTS 2.0.1 supplies the final observation for a departing car.
            event = obs.events
            unsafe = bool(event.collisions or event.off_road or event.off_route
                          or event.wrong_way or event.on_shoulder)
            self.unsafe |= unsafe
            self.collision |= bool(event.collisions)
            self.goal[i] |= event.reached_goal
            reward[i] = distance[k] - 10 * unsafe + 10 * (event.reached_goal and not unsafe)
            ego = obs.ego_vehicle_state
            accel = (ego.speed - self.raw[k].ego_vehicle_state.speed) / DT
            braking = bool(accel < -3 and in_window(self.task, ego.position))
            self.brakes += int(braking and not self.braking[i])
            self.braking[i] = braking
            if done[k]:
                self.active[i] = False
            else:
                state[i] = encode(obs)
        self.state, self.raw = state, raw
        self.total += reward
        self.steps += 1
        return reward

    def metrics(self):
        success = bool(self.goal.all() and not self.unsafe)
        return dict(steps=self.steps, collision=self.collision, success=success,
                    reached_goals=int(self.goal.sum()), unsafe=self.unsafe,
                    passage_s=round(self.steps*DT, 3) if success else None,
                    hard_brakes=self.brakes, returns=self.total.tolist())

    def close(self):
        self.env.close()

