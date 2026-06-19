# tasks/numpy/obstacle.py
# Benchmark task T2: obstacle avoidance (paper Fig. 31).
# The reference path detours around a circular obstacle via a waypoint offset
# perpendicular to the straight start->goal line. The runner measures tracking
# RMSE plus the minimum fingertip clearance to the obstacle.
import numpy as np
from .base_task import ReachTask


class ObstacleAvoidanceTask(ReachTask):
    def __init__(self, reach=(0.16, 0.12), obstacle_radius=0.03, clearance=0.02):
        self.reach = np.asarray(reach, dtype=float)   # goal offset from start (m)
        self.obstacle_radius = float(obstacle_radius)  # m
        self.clearance = float(clearance)             # detour margin past the obstacle (m)
        self.start = None
        self.goal = None
        self.obstacle_center = None
        self.targets = None
        self.center = None

    def build_waypoints(self, env):
        self.start = env.states["fingertip"][0, :2].copy()
        self.goal = self.start + self.reach
        mid = 0.5 * (self.start + self.goal)
        self.obstacle_center = mid.copy()             # obstacle sits on the straight path

        # unit perpendicular to start->goal; offset the detour waypoint to clear the obstacle
        d = self.goal - self.start
        L = float(np.linalg.norm(d)) + 1e-12
        perp = np.array([-d[1], d[0]]) / L
        offset = self.obstacle_radius + self.clearance
        detour = mid + perp * offset

        # plotting helpers (consistent with CenterOutTask attributes)
        self.center = self.start.copy()
        self.targets = np.stack([self.goal], axis=0)
        return [self.start.copy(), detour, self.goal.copy()]

    def clearance_metric(self, path_xy):
        """Minimum distance from the fingertip path to the obstacle CENTRE (m).
        Compare to obstacle_radius: values <= radius mean a collision."""
        p = np.asarray(path_xy, dtype=float)[:, :2]
        return float(np.min(np.linalg.norm(p - self.obstacle_center[None, :], axis=1)))
