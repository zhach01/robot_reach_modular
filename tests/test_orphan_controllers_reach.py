"""
Smoke/regression test for the NMPC torch controller on the analytic fast-path
(nmpc_task_torch). It is not wired to any script, so this test is its only
coverage: build it with the standard torch env and confirm a short closed-loop
reach runs, stays finite, and makes clear progress toward the target on the
analytic dynamics path.

Run:  python -m pytest tests/test_orphan_controllers_reach.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import model_lib.numpy.skeleton as sk_np
sk_np.USE_CACHE = False

torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)

from config import (PlantConfig, ControlToggles, ControlGains, Numerics,
                    InternalForceConfig, TrajectoryConfig)

EP = 0.4  # short episode keeps the test fast


def _build_env():
    from model_lib.torch.environment import Environment
    from model_lib.torch.muscles import RigidTendonHillMuscle
    from model_lib.torch.effector import RigidTendonArm26
    from trajectory.torch.minjerk import MinJerkLinearTrajectoryTorch, MinJerkParams
    pc, tc = PlantConfig(), TrajectoryConfig()
    mus = RigidTendonHillMuscle(min_activation=0.02, dtype=torch.float64)
    arm = RigidTendonArm26(muscle=mus, timestep=pc.timestep, damping=pc.damping,
                           n_ministeps=pc.n_ministeps,
                           integration_method=pc.integration_method, dtype=torch.float64)
    env = Environment(effector=arm, max_ep_duration=EP, action_noise=0.0, obs_noise=0.0,
                      action_frame_stacking=1, proprioception_delay=arm.dt, vision_delay=arm.dt)
    q0 = torch.deg2rad(torch.tensor(pc.q0_deg, dtype=torch.float64))
    env.reset(options={"joint_state": torch.cat([q0, torch.zeros(2)]).unsqueeze(0),
                       "deterministic": True})
    ft0 = env.states["fingertip"][0, :2].clone()
    target = ft0 + torch.tensor([0.10, 0.0], dtype=torch.float64)
    traj = MinJerkLinearTrajectoryTorch(
        torch.stack([ft0, target], 0),
        MinJerkParams(Vmax=tc.Vmax, Amax=tc.Amax, Jmax=tc.Jmax, gamma=tc.gamma_time_scale))
    return env, arm, q0, target, traj


def _run(env, arm, ctrl, traj, target):
    from sim.torch.simulator import TargetReachSimulatorTorch
    steps = int(EP / arm.dt)
    logs = TargetReachSimulatorTorch(env, arm, ctrl, traj, steps).run()
    k, _ = logs.time(float(arm.dt))
    xf = logs.x_log[:k][k - 1, :2]
    xf = xf.detach().numpy() if hasattr(xf, "detach") else np.asarray(xf)
    assert np.all(np.isfinite(xf)), "non-finite fingertip"
    start_err = 0.10  # 10 cm target offset
    final_err = float(np.linalg.norm(xf - target.numpy()))
    # in 0.4 s both controllers should cover most of the 10 cm reach
    assert final_err < 0.05, f"insufficient progress: {final_err*1000:.1f} mm"


def test_nmpc_torch_reaches():
    from controller.torch.nmpc_task import NonlinearMPCControllerTorch, NMPCParams
    env, arm, q0, target, traj = _build_env()
    p = NMPCParams(N=12, dt_mpc=arm.dt)
    _run(env, arm, NonlinearMPCControllerTorch(env, arm, p), traj, target)
