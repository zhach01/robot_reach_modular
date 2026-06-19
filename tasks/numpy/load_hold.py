# tasks/numpy/load_hold.py
# Benchmark task T3: static load holding (paper Fig. 31, "0.5 kg").
# The arm reaches a target and then holds it against a constant external
# end-effector force representing the held load. In the horizontal-plane model
# (g=0) the load is applied as a constant Cartesian force F = mass * g_eff in a
# fixed direction; the controller must reject it to hold the target.
import numpy as np
from .base_task import ReachTask


class LoadHoldTask(ReachTask):
    def __init__(self, mass=0.5, g_eff=9.81, direction=(0.0, -1.0), reach=(0.08, 0.0)):
        self.mass = float(mass)
        self.g_eff = float(g_eff)
        d = np.asarray(direction, dtype=float)[:2]
        self.direction = d / (np.linalg.norm(d) + 1e-12)
        self.reach = np.asarray(reach, dtype=float)
        self.start = None
        self.target = None
        self.targets = None
        self.center = None

    def build_waypoints(self, env):
        self.start = env.states["fingertip"][0, :2].copy()
        self.target = self.start + self.reach
        self.center = self.start.copy()
        self.targets = np.stack([self.target], axis=0)
        return [self.start.copy(), self.target.copy()]

    def endpoint_load(self):
        """Constant external end-effector force (N), (2,). Applied every step by the
        benchmark runner via env.step(endpoint_load=...)."""
        return self.mass * self.g_eff * self.direction
