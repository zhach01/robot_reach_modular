# base_task_torch.py
# -*- coding: utf-8 -*-
"""
Torch version of base_task.py

Original (NumPy) API: :contentReference[oaicite:1]{index=1}
    class ReachTask(ABC):
        @abstractmethod
        def build_waypoints(self, env) -> list: ...

    class CenterOutTask(ReachTask):
        def __init__(self, n_targets=4, radius=0.10):
            ...
        def build_waypoints(self, env):
            center = env.states["fingertip"][0]
            targets on a circle, return list of waypoints (center, target_i, center, ...)

This Torch version:
    - Uses Torch tensors instead of NumPy.
    - Keeps the same conceptual API (ReachTask + CenterOutTask).
    - build_waypoints(...) returns a single Tensor of shape (N_waypoints, 2),
      fully differentiable and living on env.device / env.dtype.

Typical usage:
    task = CenterOutTask(n_targets=8, radius=0.10)
    waypoints = task.build_waypoints(env_torch)  # (1 + 2*n_targets, 2)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class ReachTask(ABC):
    @abstractmethod
    def build_waypoints(self, env) -> Tensor:
        """
        Build a sequence of waypoints in task space.

        For 2D reaching tasks:
            returns: (N, 2) Torch tensor of [x, y] points.
        """
        ...


class CenterOutTask(ReachTask):
    """
    Center-out reaching task (Torch version).

    - Picks the current fingertip position (from env.states["fingertip"][0])
      as the center.
    - Builds n_targets equally spaced points on a circle of radius `radius`.
    - Returns waypoints:
          [center,
           target0, center,
           target1, center,
           ...
           target_{n_targets-1}, center]   -> shape (1 + 2*n_targets, 2)
    """

    def __init__(self, n_targets: int = 4, radius: float = 0.10):
        self.n_targets = int(n_targets)
        self.radius = float(radius)

        self.targets: Optional[Tensor] = None  # (n_targets, 2)
        self.center: Optional[Tensor] = None   # (2,)

    def build_waypoints(self, env) -> Tensor:
        """
        Build center-out waypoints based on the current env fingertip.

        env.states["fingertip"] is expected to be:
            - shape (B, 2) or (B, 4) Torch tensor,
              and we use the first batch entry as the current center.

        Returns:
            waypoints: (1 + 2*n_targets, 2) Torch tensor on env.device/env.dtype
        """
        fingertip = env.states["fingertip"]
        ft = torch.as_tensor(fingertip)

        if ft.dim() == 1:
            ft = ft.view(1, -1)

        # Use only x, y of the first batch entry
        center = ft[0, :2]
        device, dtype = center.device, center.dtype

        self.center = center.clone()

        # Evenly spaced angles [0, 2π)
        thetas = torch.linspace(
            0.0,
            2.0 * math.pi,
            steps=self.n_targets + 1,
            device=device,
            dtype=dtype,
        )[:-1]  # drop endpoint

        xs = center[0] + self.radius * torch.cos(thetas)
        ys = center[1] + self.radius * torch.sin(thetas)
        self.targets = torch.stack([xs, ys], dim=-1)  # (n_targets, 2)

        # Build waypoint sequence: center, target_i, center, ...
        wps = [self.center.unsqueeze(0)]
        for i in range(self.n_targets):
            wps.append(self.targets[i].unsqueeze(0))
            wps.append(self.center.unsqueeze(0))

        waypoints = torch.cat(wps, dim=0)  # (1 + 2*n_targets, 2)
        return waypoints


# ---------------------------------------------------------------------
# Simple smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=6, sci_mode=False)

    print("[base_task_torch] Simple smoke test...")

    class DummyEnv:
        def __init__(self):
            # fingertip: (B,2)
            self.states = {
                "fingertip": torch.tensor([[0.08, 0.52]], dtype=torch.get_default_dtype())
            }

    env = DummyEnv()
    task = CenterOutTask(n_targets=4, radius=0.10)
    waypoints = task.build_waypoints(env)

    print("  waypoints shape:", waypoints.shape)
    print("  waypoints:", waypoints)
    print("  center:", task.center)
    print("  targets:", task.targets)

    print("[base_task_torch] Smoke test done.")
