# tasks/random_reach.py
import numpy as np
from tasks.base_task import ReachTask


class Task(ReachTask):
    def __init__(self, n_points=8, radius=0.10, seed=0):
        self.n_points = n_points
        self.radius = radius
        self.rng = np.random.default_rng(seed)
        self.center = None
        self.targets = None

    def build_waypoints(self, env):
        self.center = env.states["fingertip"][0].copy()
        ang = self.rng.uniform(0, 2 * np.pi, size=self.n_points)
        self.targets = np.stack(
            [
                self.center[0] + self.radius * np.cos(ang),
                self.center[1] + self.radius * np.sin(ang),
            ],
            axis=-1,
        )
        waypoints = [self.center.copy()]
        for i in range(self.n_points):
            waypoints.append(self.targets[i].copy())
            waypoints.append(self.center.copy())
        return waypoints
