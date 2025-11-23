# center_out_torch.py
# -*- coding: utf-8 -*-
"""
Torch alias for the center-out task.

NumPy version:
    from tasks.base_task import CenterOutTask as Task

Torch version:
    uses CenterOutTask from base_task_torch, which builds waypoints as a
    Torch tensor on env.device/env.dtype.
"""

from tasks.base_task_torch import CenterOutTask as Task


# ---------------------------------------------------------------------
# Simple smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=6, sci_mode=False)

    print("[center_out_torch] Simple smoke test...")

    class DummyEnv:
        def __init__(self):
            # fingertip: (B,2)
            self.states = {
                "fingertip": torch.tensor([[0.08, 0.52]], dtype=torch.get_default_dtype())
            }

    from tasks.center_out_torch import Task as CenterOutTaskTorch

    env = DummyEnv()
    task = CenterOutTaskTorch(n_targets=4, radius=0.10)
    waypoints = task.build_waypoints(env)

    print("  waypoints shape:", waypoints.shape)
    print("  waypoints:", waypoints)
    print("  center:", task.center)
    print("  targets:", task.targets)

    print("[center_out_torch] Smoke test done.")
