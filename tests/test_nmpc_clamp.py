"""
Audit H4: NMPC must (a) still track with adequate force budget and (b) actually
respect the task-force limit |F_k| <= Fmax via the projected clamp-in-loop
(previously the optimizer ignored the limit and only F0 was clipped post-hoc).

Run:  python -m pytest tests/test_nmpc_clamp.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import model_lib.skeleton_numpy as sk_np
sk_np.USE_CACHE = False

from config import PlantConfig, TrajectoryConfig


def _build(Fmax, ep=1.5):
    from model_lib.environment_numpy import Environment
    from model_lib.muscles_numpy import RigidTendonHillMuscle
    from model_lib.effector_numpy import RigidTendonArm26
    from trajectory.minjerk_numpy import MinJerkLinearTrajectory, MinJerkParams
    from controller.nmpc_task_numpy import NonlinearMPCController, NMPCParams
    pc, tc = PlantConfig(), TrajectoryConfig()
    mus = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(muscle=mus, timestep=pc.timestep, damping=pc.damping,
                           n_ministeps=pc.n_ministeps, integration_method=pc.integration_method)
    env = Environment(effector=arm, max_ep_duration=ep, action_noise=0.0, obs_noise=0.0,
                      proprioception_delay=arm.dt, vision_delay=arm.dt)
    q0 = np.deg2rad(np.array(pc.q0_deg))
    env.reset(options={"joint_state": np.concatenate([q0, np.zeros(2)])[None, :],
                       "deterministic": True})
    ft0 = np.asarray(env.states["fingertip"][0][:2], float)
    target = ft0 + np.array([0.10, 0.0])
    traj = MinJerkLinearTrajectory(np.stack([ft0, target], 0),
                                   MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale))
    p = NMPCParams(N=12, dt_mpc=arm.dt, use_muscles=True, Fmax=Fmax)
    ctrl = NonlinearMPCController(env, arm, p)
    return env, arm, ctrl, traj, target


def test_nmpc_tracks_with_adequate_force():
    from sim.simulator_numpy import TargetReachSimulator
    env, arm, ctrl, traj, target = _build(Fmax=600.0, ep=1.5)
    steps = int(1.5 / arm.dt)
    logs = TargetReachSimulator(env, arm, ctrl, traj, steps).run()
    k, _ = logs.time(float(arm.dt))
    xf = np.asarray(logs.x_log[:k][k - 1, :2], float)
    err = float(np.linalg.norm(xf - target))
    assert err < 0.02, f"NMPC failed to track: {err*1000:.1f} mm"


def test_nmpc_respects_force_limit():
    """With a tight Fmax the clamp must hold every commanded F0 within the box."""
    env, arm, ctrl, traj, target = _build(Fmax=8.0, ep=1.0)

    # Capture the task force F0 returned by the horizon solve each step.
    captured = {"maxabs": 0.0, "n": 0}
    orig = ctrl._solve_horizon

    def wrapped(*a, **k):
        F0, Lr, mt, le = orig(*a, **k)
        captured["maxabs"] = max(captured["maxabs"], float(np.max(np.abs(F0))))
        captured["n"] += 1
        return F0, Lr, mt, le

    ctrl._solve_horizon = wrapped

    dt = float(arm.dt)
    for k in range(40):
        x_d, xd_d, xdd_d = traj.sample(k * dt)
        diag = ctrl.compute(x_d, xd_d, xdd_d)
        env.step(np.asarray(diag["act"]).reshape(1, -1), deterministic=True)

    assert captured["n"] > 0, "solver was never exercised"
    assert captured["maxabs"] <= 8.0 + 1e-6, (
        f"force limit violated: max|F0|={captured['maxabs']:.2f} > 8.0"
    )
