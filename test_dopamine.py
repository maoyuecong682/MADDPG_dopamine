import unittest

import numpy as np
import torch

from progress import Progress, advance, hop_target, inputs
from progress_replay import ProgressReplay
from run import shaped_reward


class ExactHop(torch.nn.Module):
    """Straight-line reference path; normalized longitudinal position is progress."""
    def forward(self, x):
        p, q = x[...,0], x[...,2]
        denominator = torch.where(q >= p,1-p,p).clamp(min=1e-7)
        return ((q-p)/denominator).unsqueeze(-1)


class DopamineTests(unittest.TestCase):
    def test_progress_and_regress(self):
        a = np.array([0.,.4,.5,1.,0.,1.])
        b = np.array([.5,.5,.2,0.,0.,1.])
        np.testing.assert_allclose(advance(a,hop_target(a,b)),b)
        start=np.array([[0.,0.],[2.,1.]])
        goal=start+np.array([100.,0.])
        for mode in ("bif","b","i","f"):
            tracker=Progress(ExactHop(),"merge",start,goal,mode)
            for p in (.4,.65,.2,1.):
                np.testing.assert_allclose(tracker.update(start+(goal-start)*p,[True,True]),p,atol=1e-6)
            np.testing.assert_allclose(tracker.update(start,[False,True]),[1.,0.],atol=1e-6)

    def test_potential_terminal_and_cutoff(self):
        base=np.array([1.,1.,0.])
        p=np.array([.4,.4,.4])
        q=np.array([.7,.7,.7])
        shaped=shaped_reward(base,p,q,[True,True,False],[False,True,False],10.,2.,.9)
        np.testing.assert_allclose(shaped,[-.7,.56,0.],atol=1e-6)

    def test_strata_and_overwrite(self):
        replay=ProgressReplay(64,np.random.default_rng(3),5)
        alive=np.array([True,False])
        def add(task,p,e):
            value=np.array([e,-e],dtype=np.float32)
            replay.add_joint(task,np.array([p,1.]),e,value,value,value,value,alive,alive)
        for i in range(61):
            add("merge",.05,i)
        add("merge",.95,61)
        add("lane_change",.1,62)
        add("lane_change",.1,63)
        ids,batch,meta=replay.sample_with_metadata(3000,"cpu")
        task=np.array([m["task_id"] for m in meta])
        self.assertLess(abs(np.mean(task==1)-.5),.04)
        phases=np.array([m["phase"] for m in meta])
        self.assertLess(abs(np.mean(phases[task==1]==4)-.5),.06)
        np.testing.assert_array_equal(batch[0][:,0].numpy(),ids)
        for j in range(64,130):
            add("left_turn",1.,j)
        ids,batch,meta=replay.sample_with_metadata(256,"cpu")
        self.assertTrue(all(m["task_id"]==2 and m["phase"]==4 for m in meta))
        np.testing.assert_array_equal(batch[0][:,0].numpy(),[m["episode_id"] for m in meta])
        self.assertEqual(sum(len(b) for row in replay.buckets for b in row),64)


if __name__=="__main__":
    unittest.main()

