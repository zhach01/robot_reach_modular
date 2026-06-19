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


def test_osc_torch_tracks():
    from controller.torch.osc_controller import OSCController, OSCParams
    env, arm, pc = _build_env()
    traj, target = _make_traj(env)
    ctrl = OSCController(env, arm, OSCParams(Kp=4000.0, Kv=126.0))
    err = _final_error(env, arm, ctrl, traj, target)
    assert err < 0.01, f"OSC failed to track: {err*1000:.1f} mm"


def test_anfis_torch_tracks():
    # canonical ANFIS, torch port. No preloaded rules (rules file is gitignored):
    # from its model-based PD init + online RLS it must still reach the target.
    from controller.torch.anfis_controller import ANFISController, ANFISParams
    env, arm, pc = _build_env()
    traj, target = _make_traj(env)
    ctrl = ANFISController(env, arm, ANFISParams())
    err = _final_error(env, arm, ctrl, traj, target)
    assert err < 0.01, f"ANFIS failed to track: {err*1000:.1f} mm"


def test_motornet_controller_torch_smoke():
    # canonical torch MotorNet (bake-off winner). It is an EPISODIC learned
    # policy whose accuracy needs a trained (gitignored) checkpoint, so the CI
    # test only guards the controller API contract: compute() must return a
    # finite muscle-excitation vector in [0,1] of the right shape.
    from controller.torch.motornet_controller import MotorNetController, MotorNetParams
    env, arm, pc = _build_env()
    ctrl = MotorNetController(env, arm, MotorNetParams(device="cpu"))
    q0 = torch.deg2rad(torch.tensor(pc.q0_deg, dtype=torch.float64))
    ctrl.reset(q0)
    target = env.states["fingertip"][0, :2] + torch.tensor([0.05, 0.0], dtype=torch.float64)
    ctrl.set_goal(target)
    out = ctrl.compute(target, torch.zeros(2), torch.zeros(2))
    a = out["act"]
    assert a.shape[-1] == arm.n_muscles, f"bad action dim: {tuple(a.shape)}"
    assert torch.isfinite(a).all(), "non-finite activations"
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0001, "activations out of [0,1]"


def test_bc_a_torch_smoke():
    # canonical Hybrid/BC = behavior-cloned activation policy (torch). Accuracy
    # needs a trained (gitignored) checkpoint, so the CI test guards the API
    # contract: compute() returns finite muscle activations in [0,1].
    from controller.torch.hybrid_bc_a import (
        RLPolicy, RLPolicyParams, RLControllerATorch, RLControllerAParams)
    env, arm, pc = _build_env()
    m = int(env.states["geometry"].shape[2])
    pi = RLPolicy(RLPolicyParams(obs_dim=14 + 5 * m, act_dim=m, device="cpu"))
    ctrl = RLControllerATorch(env, arm, pi, RLControllerAParams())
    ctrl.reset(torch.deg2rad(torch.tensor(pc.q0_deg, dtype=torch.float64)))
    target = env.states["fingertip"][0, :2] + torch.tensor([0.05, 0.0], dtype=torch.float64)
    out = ctrl.compute(target, torch.zeros(2), torch.zeros(2))
    a = out["act"]
    assert a.shape[-1] == arm.n_muscles, f"bad action dim: {tuple(a.shape)}"
    assert torch.isfinite(a).all(), "non-finite activations"
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0001, "activations out of [0,1]"


def test_synergy_pure_torch_smoke():
    # pure K-synergy baseline (torch). Tracking needs a trained NMF W (gitignored),
    # so the CI test guards the API contract with the default W.
    from controller.torch.synergy_controller_pure import SynergyPureController, SynergyPureParams
    env, arm, pc = _build_env()
    ctrl = SynergyPureController(env, arm, SynergyPureParams())
    ctrl.reset(torch.deg2rad(torch.tensor(pc.q0_deg, dtype=torch.float64)))
    target = env.states["fingertip"][0, :2] + torch.tensor([0.05, 0.0], dtype=torch.float64)
    out = ctrl.compute(target, torch.zeros(2), torch.zeros(2))
    a = out["act"]
    assert a.shape[-1] == arm.n_muscles, f"bad action dim: {tuple(a.shape)}"
    assert torch.isfinite(a).all() and float(a.min()) >= 0.0 and float(a.max()) <= 1.0001


def test_synergy_torch_tracks():
    # canonical (full) SynergyController, torch port: modules + modulation +
    # residual correction must drive the fingertip to the target with the
    # default heuristic W (no trained synergy model needed).
    from controller.torch.synergy_controller import SynergyController, SynergyParams
    env, arm, pc = _build_env()
    traj, target = _make_traj(env)
    ctrl = SynergyController(env, arm, SynergyParams(
        Kp_task=800.0, Kv_task=60.0, c_max=5.0, synergy_strength=0.8))
    err = _final_error(env, arm, ctrl, traj, target)
    assert err < 0.01, f"synergy failed to track: {err*1000:.1f} mm"


def test_sliding_mode_torch_tracks():
    from controller.torch.sliding_mode import SlidingModeController, SlidingModeParams
    env, arm, pc = _build_env()
    traj, target = _make_traj(env)
    num, tog, ifc = Numerics(), ControlToggles(), InternalForceConfig()
    p = SlidingModeParams(
        lambda_surf=[26.0, 26.0], K_switch=[4.0, 4.0], phi=[0.004, 0.004],
        Kff_x=[1.0, 1.0],
        eps=float(num.eps), lam_os_max=float(num.lam_os_max),
        sigma_thresh=float(num.sigma_thresh), gate_pow=float(num.gate_pow),
        enable_inertia_comp=tog.enable_inertia_comp,
        enable_gravity_comp=tog.enable_gravity_comp,
        enable_velocity_comp=tog.enable_velocity_comp,
        bisect_iters=int(ifc.bisect_iters))
    ctrl = SlidingModeController(env, arm, p)
    err = _final_error(env, arm, ctrl, traj, target)
    assert err < 0.01, f"sliding-mode failed to track: {err*1000:.1f} mm"
