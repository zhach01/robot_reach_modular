# random_reach_torch.py
# -*- coding: utf-8 -*-
"""
Torch version of tasks/random_reach.py

Original NumPy version: :contentReference[oaicite:1]{index=1}

    class Task(ReachTask):
        def __init__(self, n_points=8, radius=0.10, seed=0):
            ...
        def build_waypoints(self, env):
            center = env.states["fingertip"][0]
            ang = rng.uniform(0, 2*pi, size=n_points)
            targets = center + radius * [cos(ang), sin(ang)]
            waypoints = [center, t0, center, t1, center, ...]

Torch version:

    - Uses ReachTask from base_task_torch.
    - Uses a torch.Generator for RNG, seeded and device-aware.
    - Works on the same device/dtype as env.states["fingertip"].
    - Returns a single Tensor of waypoints with shape (1 + 2*n_points, 2):
          [center,
           target0, center,
           target1, center,
           ...
           target_{n_points-1}, center]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from tasks.base_task_torch import ReachTask


class Task(ReachTask):
    """
    Random reach task (Torch).

    - Draws n_points random angles on a circle of radius `radius`.
    - Center is the current fingertip position (env.states["fingertip"][0, :2]).
    - Returns waypoints as a (1 + 2*n_points, 2) Tensor on the same
      device/dtype as the fingertip state.

    Attributes after build_waypoints():
        center  : (2,) Tensor
        targets : (n_points, 2) Tensor
    """

    def __init__(self, n_points: int = 8, radius: float = 0.10, seed: Optional[int] = 0):
        self.n_points = int(n_points)
        self.radius = float(radius)
        self.seed = seed

        # Will be created on first build_waypoints, on the correct device
        self.generator: Optional[torch.Generator] = None

        # Filled by build_waypoints
        self.center: Optional[Tensor] = None   # (2,)
        self.targets: Optional[Tensor] = None  # (n_points, 2)

    def _ensure_generator(self, device: torch.device):
        """Create / move RNG to the given device if needed."""
        if self.generator is None or self.generator.device != device:
            self.generator = torch.Generator(device=device)
            if self.seed is not None:
                self.generator.manual_seed(int(self.seed))

    def build_waypoints(self, env) -> Tensor:
        """
        Build random center-out waypoints for the current env state.

        env.states["fingertip"] is expected to be:
            - Torch tensor of shape (B, 2) or (B, 4) (we take the first row),
              or a compatible array that torch.as_tensor can convert.

        Returns:
            waypoints: (1 + 2*n_points, 2) Tensor
        """
        fingertip = env.states["fingertip"]
        ft = torch.as_tensor(fingertip)

        if ft.dim() == 1:
            ft = ft.view(1, -1)

        # center is x,y of first batch entry
        center = ft[0, :2]
        device, dtype = center.device, center.dtype

        self._ensure_generator(device)

        self.center = center.clone()

        # random angles on [0, 2π)
        ang = 2.0 * math.pi * torch.rand(
            (self.n_points,),
            device=device,
            dtype=dtype,
            generator=self.generator,
        )  # (n_points,)

        xs = self.center[0] + self.radius * torch.cos(ang)
        ys = self.center[1] + self.radius * torch.sin(ang)
        self.targets = torch.stack([xs, ys], dim=-1)  # (n_points, 2)

        # Build waypoint sequence: center, target_i, center, ...
        wps = [self.center.unsqueeze(0)]
        for i in range(self.n_points):
            wps.append(self.targets[i].unsqueeze(0))
            wps.append(self.center.unsqueeze(0))

        waypoints = torch.cat(wps, dim=0)  # (1 + 2*n_points, 2)
        return waypoints


# ---------------------------------------------------------------------
# Simple smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=6, sci_mode=False)

    print("[random_reach_torch] Simple smoke test...")

    class DummyEnv:
        def __init__(self):
            # fingertip: (B,2)
            self.states = {
                "fingertip": torch.tensor([[0.08, 0.52]], dtype=torch.get_default_dtype())
            }

    env = DummyEnv()
    task = Task(n_points=8, radius=0.10, seed=42)

    wps = task.build_waypoints(env)
    print("  waypoints shape:", wps.shape)
    print("  waypoints:", wps)
    print("  center:", task.center)
    print("  targets shape:", task.targets.shape)

    # Call again to check RNG / device handling
    wps2 = task.build_waypoints(env)
    print("  waypoints (2nd call) shape:", wps2.shape)

    print("[random_reach_torch] Smoke test done.")
