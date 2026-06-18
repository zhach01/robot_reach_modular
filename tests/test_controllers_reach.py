"""
Closed-loop tracking regression tests on the (fixed) differentiable torch plant.

A controller that implements its law correctly must drive the fingertip to a
10 cm target to sub-centimetre accuracy. Guards:
  - PD/IF baseline still tracks.
  - Sliding-mode (audit H2/H3): previously diverged (~370 mm) because the torch
    equivalent control omitted the lambda*e_v velocity-feedback term and the
    eta floors (0.50/0.35) defeated the singularity gate. Must now track.

Run:  python -m pytest tests/test_controllers_reach.py -q
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

from config import (PlantConfig, ControlGains, ControlToggles, Numerics,
                    InternalForceConfig, TrajectoryConfig)


def _build_env():
    from model_lib.torch.environment import Environment
    from model_lib.torch.muscles import RigidTendonHillMuscle
    from model_lib.torch.effector import RigidTendonArm26
    pc = PlantConfig()
    mus = RigidTendonHillMuscle(min_activation=0.02, dtype=torch.float64)
    arm = RigidTendonArm26(muscle=mus, timestep=pc.timestep, damping=pc.damping,
                           n_ministeps=pc.n_ministeps,
                           integration_method=pc.integration_method, dtype=torch.float64)
    env = Environment(effector=arm, max_ep_duration=pc.max_ep_duration,
                      action_noise=0.0, obs_noise=0.0,
                      proprioception_delay=arm.dt, vision_delay=arm.dt)
    q0 = torch.deg2rad(torch.tensor(pc.q0_deg, dtype=torch.float64))
    joint0 = torch.cat([q0, torch.zeros(2)]).unsqueeze(0)
    env.reset(options={"joint_state": joint0, "deterministic": True})
    return env, arm, pc


def _make_traj(env):
    from trajectory.torch.minjerk import MinJerkLinearTrajectoryTorch, MinJerkParams
    tc = TrajectoryConfig()
    ft0 = env.states["fingertip"][0, :2].clone()
    target = ft0 + torch.tensor([0.10, 0.0], dtype=torch.float64)
    wp = torch.stack([ft0, target], 0)
    traj = MinJerkLinearTrajectoryTorch(
        wp, MinJerkParams(Vmax=tc.Vmax, Amax=tc.Amax, Jmax=tc.Jmax,
                          gamma=tc.gamma_time_scale))
    return traj, target


def _final_error(env, arm, ctrl, traj, target):
    from sim.torch.simulator import TargetReachSimulatorTorch
    steps = int(PlantConfig().max_ep_duration / arm.dt)
    sim = TargetReachSimulatorTorch(env, arm, ctrl, traj, steps)
    logs = sim.run()
    k, _ = logs.time(float(arm.dt))
    xf = logs.x_log[:k][k - 1, :2]
    xf = xf.detach().numpy() if hasattr(xf, "detach") else np.asarray(xf)
    return float(np.linalg.norm(xf - target.numpy()))


def test_pdif_torch_tracks():
    # canonical (optimized) torch PD/IF controller
    from controller.torch.pd_if_controller import PDIFController, PDIFParams
    env, arm, pc = _build_env()
    traj, target = _make_traj(env)
    num = Numerics()
    p = PDIFParams(
        Kp_task=[1600.0, 1600.0], damping_ratio=1.0, Kff=1.0,
        use_critical_damping=True, enable_nullspace=True,
        eps=num.eps, lam_os_max=float(getattr(num, "lam_os_max", 200.0)),
        sigma_thresh=num.sigma_thresh, gate_pow=num.gate_pow,
        bisect_iters=12, enable_internal_force=False,
    )
    ctrl = PDIFController(env, arm, p)
    err = _final_error(env, arm, ctrl, traj, target)
    assert err < 0.01, f"PD/IF failed to track: {err*1000:.1f} mm"


def test_pdif_torch_legacy_tracks():
    # legacy torch PD/IF controller (original Kp_x gain API), kept for coverage
    from controller.torch.pd_if_legacy import PDIFController, PDIFParams
    env, arm, pc = _build_env()
    traj, target = _make_traj(env)
    gn, num, tog, ifc = ControlGains(), Numerics(), ControlToggles(), InternalForceConfig()
    p = PDIFParams(Kp_x=gn.Kp_x, Kff_x=gn.Kff_x, Kp_q=gn.Kp_q, Kd_q=gn.Kd_q, eps=num.eps,
                   lam_os_smin_target=num.lam_os_smin_target, lam_os_max=num.lam_os_max,
                   sigma_thresh=num.sigma_thresh, gate_pow=num.gate_pow,
                   enable_internal_force=tog.enable_internal_force,
                   enable_inertia_comp=tog.enable_inertia_comp,
                   enable_gravity_comp=tog.enable_gravity_comp,
                   enable_velocity_comp=tog.enable_velocity_comp,
                   enable_joint_damping=tog.enable_joint_damping,
                   cocon_a0=ifc.cocon_a0, bisect_iters=ifc.bisect_iters,
                   linesearch_eps=num.linesearch_eps, linesearch_safety=num.linesearch_safety)
    ctrl = PDIFController(env, arm, p)
    err = _final_error(env, arm, ctrl, traj, target)
    assert err < 0.01, f"PD/IF (legacy) failed to track: {err*1000:.1f} mm"


def test_sliding_mode_torch_tracks():
    from controller.torch.sliding_mode import SlidingModeController, SlidingModeParams
    env, arm, pc = _build_env()
    traj, target = _make_traj(env)
    gn, num, tog, ifc = ControlGains(), Numerics(), ControlToggles(), InternalForceConfig()
    p = SlidingModeParams(
        lambda_surf=torch.tensor([10., 10.], dtype=torch.float64),
        K_switch=torch.tensor([2., 2.], dtype=torch.float64),
        phi=torch.tensor([0.02, 0.02], dtype=torch.float64),
        Kff_x=torch.tensor(gn.Kff_x, dtype=torch.float64),
        Kp_q=torch.tensor(gn.Kp_q, dtype=torch.float64),
        Kd_q=torch.tensor(gn.Kd_q, dtype=torch.float64),
        eps=float(num.eps), lam_os_max=float(num.lam_os_max),
        sigma_thresh=float(num.sigma_thresh), gate_pow=float(num.gate_pow),
        enable_inertia_comp=tog.enable_inertia_comp,
        enable_gravity_comp=tog.enable_gravity_comp,
        enable_velocity_comp=tog.enable_velocity_comp,
        enable_joint_damping=tog.enable_joint_damping,
        enable_internal_force=tog.enable_internal_force,
        cocon_a0=float(ifc.cocon_a0), bisect_iters=int(ifc.bisect_iters),
        linesearch_eps=float(num.linesearch_eps), linesearch_safety=float(num.linesearch_safety))
    ctrl = SlidingModeController(env, arm, p)
    err = _final_error(env, arm, ctrl, traj, target)
    assert err < 0.01, f"sliding-mode failed to track: {err*1000:.1f} mm"
