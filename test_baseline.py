import unittest

import numpy as np
import torch

from environment import ACT_DIM, IDS, OBS_DIM, Episode, actuator
from maddpg import MADDPG, Replay
from scenarios import TASKS


class BaselineTests(unittest.TestCase):
    def test_replay_overwrite_and_alignment(self):
        replay = Replay(3, np.random.default_rng(17))
        for i in range(5):
            pair = np.array([i, -i], np.float32)
            replay.add(pair, pair*2)
            pair[:] = 42
        state, action = replay.sample(3, "cpu")
        self.assertEqual(set(state[:, 0].tolist()), {2, 3, 4})
        torch.testing.assert_close(state*2, action)
        torch.testing.assert_close(state[:,0], -state[:,1])

    def test_update_and_masks(self):
        torch.manual_seed(11)
        model = MADDPG(2, OBS_DIM, ACT_DIM)
        s = torch.randn(16, 2, OBS_DIM)
        s[:,1] = 0
        a = torch.zeros(16, 2, ACT_DIM)
        r = torch.ones(16, 2)
        active = torch.tensor([[1., 0.]]).repeat(16, 1)
        # Terminal rows: arbitrary next states must leave the TD target unchanged.
        import copy
        twin = copy.deepcopy(model)
        before = [net[-2].weight.clone() for net in model.actors]
        args = (s, a, r, s, active, torch.zeros_like(active))
        losses = model.update(args)
        other = twin.update((s, a, r, s+2, active, torch.zeros_like(active)))
        np.testing.assert_allclose(losses, other)
        after = [net[-2].weight for net in model.actors]
        self.assertFalse(torch.equal(before[0], after[0]))
        torch.testing.assert_close(before[1], after[1])
        self.assertFalse(torch.equal(before[0], model.target_actors[0][-2].weight))

    def test_real_maps_and_control(self):
        self.assertEqual(actuator([-.5, .3]), (0, .5, .3))
        self.assertEqual(actuator([.7, -.2]), (.7, 0, -.2))
        for task in TASKS:
            with self.subTest(task=task):
                env = Episode(task, 77, 0)
                try:
                    self.assertEqual(set(env.raw), set(IDS))
                    start = env.state.copy()
                    for _ in range(5):
                        rewards = env.step(np.array([[.3, 0], [.4, 0]]))
                        self.assertTrue(np.isfinite(rewards).all())
                    self.assertEqual(env.state.shape, (2, OBS_DIM))
                    self.assertTrue(np.isfinite(env.state).all())
                    self.assertFalse(np.array_equal(start, env.state))
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()

