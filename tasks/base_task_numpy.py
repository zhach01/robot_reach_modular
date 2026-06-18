# tasks/base_task.py
from abc import ABC, abstractmethod
import numpy as np


class ReachTask(ABC):
    @abstractmethod
    def build_waypoints(self, env) -> list: ...


class CenterOutTask(ReachTask):
    def __init__(self, n_targets=4, radius=0.10):
        self.n_targets = n_targets
        self.radius = radius
        self.targets = None
        self.center = None

    def build_waypoints(self, env):
        self.center = env.states["fingertip"][0].copy()
        thetas = np.linspace(0, 2 * np.pi, self.n_targets, endpoint=False)
        self.targets = np.stack(
            [
                self.center[0] + self.radius * np.cos(thetas),
                self.center[1] + self.radius * np.sin(thetas),
            ],
            axis=-1,
        )
        waypoints = [self.center.copy()]
        for i in range(self.n_targets):
            waypoints.append(self.targets[i].copy())
            waypoints.append(self.center.copy())
        return waypoints
