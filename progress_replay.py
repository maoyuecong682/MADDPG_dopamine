"""A single joint ring and an O(1) overwrite index for task/progress sampling."""
import numpy as np

from maddpg import Replay
from scenarios import TASKS


class ProgressReplay(Replay):
    def __init__(self, capacity, rng, bins, stratified=True):
        super().__init__(capacity,rng)
        self.bins, self.stratified = bins, stratified
        self.buckets = [[[] for _ in range(bins)] for _ in TASKS]
        self.metadata = [None]*capacity
        self.offsets = np.empty(capacity,dtype=int)
        self.sample_counts = np.zeros((len(TASKS),bins),dtype=int)
        self.last_ids = None

    def add_joint(self, task, progress, episode_id, *fields):
        alive = fields[4]
        joint_progress = float(np.mean(progress[alive]))
        phase = min(self.bins-1, int(self.bins*joint_progress))
        slot = self.cursor
        old = self.metadata[slot]
        if old is not None:
            bucket = self.buckets[old["task_id"]][old["phase"]]
            offset = self.offsets[slot]
            last = bucket[-1]
            bucket[offset] = last
            self.offsets[last] = offset
            bucket.pop()
        task_id = TASKS.index(task)
        bucket = self.buckets[task_id][phase]
        self.offsets[slot] = len(bucket)
        bucket.append(slot)
        self.metadata[slot] = dict(task_id=task_id, phase=phase, progress=joint_progress,
                                   agent_progress=progress.copy(), episode_id=episode_id)
        super().add(*fields)

    def draw(self,count):
        if not self.stratified:
            return self.rng.choice(len(self.rows),count,replace=False)
        active = [i for i,groups in enumerate(self.buckets) if any(groups)]
        tasks = self.rng.choice(active,count)
        indices = np.empty(count,dtype=int)
        for task_id in np.unique(tasks):
            locations = np.flatnonzero(tasks==task_id)
            buckets = self.buckets[task_id]
            phases = self.rng.choice([k for k,v in enumerate(buckets) if v],len(locations))
            for phase in np.unique(phases):
                destinations = locations[phases==phase]
                pool = buckets[phase]
                choices = self.rng.integers(len(pool),size=len(destinations))
                indices[destinations] = [pool[k] for k in choices]
        return indices

    def sample_with_metadata(self,count,device):
        ids = self.draw(count)
        meta = [self.metadata[i] for i in ids]
        for row in meta:
            self.sample_counts[row["task_id"],row["phase"]] += 1
        self.last_ids = ids
        return ids, self.batch_at(ids,device), meta

    def sample(self,count,device):
        return self.sample_with_metadata(count,device)[1]

