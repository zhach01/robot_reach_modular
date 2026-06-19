#!/usr/bin/env python3
# scripts/SYNERGY/main_random_reach_synergy_torch.py
# Torch counterpart of main_random_reach_synergy.py (canonical full SynergyController).

from __future__ import annotations

import numpy as np
import torch

from model_lib.torch.environment import Environment as EnvironmentTorch
from model_lib.torch.muscles import RigidTendonHillMuscle
from model_lib.torch.effector import RigidTendonArm26

from config import PlantConfig, ControlToggles, ControlGains, Numerics, InternalForceConfig, TrajectoryConfig, RunConfig
from trajectory.torch.minjerk import MinJerkLinearTrajectoryTorch, MinJerkParams
from controller.torch.synergy_controller import SynergyController, SynergyParams
from sim.torch.simulator import TargetReachSimulatorTorch

from plotting.plots import plot_all, make_animations, hold_anims
import matplotlib.pyplot as plt


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _scalar(v, default=1.0):
    v = np.atleast_1d(np.asarray(v, dtype=float))
    return float(v[0]) if v.size else float(default)


def build_env_torch(pc: PlantConfig):
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    muscle = RigidTendonHillMuscle(min_activation=0.02, device=device, dtype=torch.get_default_dtype())
    arm = RigidTendonArm26(
        muscle=muscle, timestep=pc.timestep, damping=pc.damping,
        n_ministeps=pc.n_ministeps, integration_method=pc.integration_method,
        device=device, dtype=torch.get_default_dtype(),
    )
    env = EnvironmentTorch(
        effector=arm, max_ep_duration=pc.max_ep_duration, action_noise=0.0, obs_noise=0.0,
        action_frame_stacking=1, proprioception_delay=arm.dt, vision_delay=arm.dt,
        name="RandomReachSynergyEnvTorch",
    )
    q0 = torch.deg2rad(torch.tensor(pc.q0_deg, dtype=torch.get_default_dtype(), device=device))
    qd0 = torch.tensor(pc.qd0, dtype=torch.get_default_dtype(), device=device)
    env.reset(options={"joint_state": torch.cat([q0, qd0]).unsqueeze(0), "deterministic": True})
    return env, arm, q0


def main():
    print("[RandomReach Synergy Torch] demo starting ...")
    pc = PlantConfig(); toggles = ControlToggles(); gains = ControlGains()
    num = Numerics(); ifc = InternalForceConfig(); tc = TrajectoryConfig(); run = RunConfig()

    env, arm, q0 = build_env_torch(pc)
    device = q0.device

    center = env.states["fingertip"][0, :2].clone()
    target = center + torch.tensor([0.10, 0.0], dtype=center.dtype, device=device)
    waypoints = torch.stack([center, target], dim=0)
    traj = MinJerkLinearTrajectoryTorch(
        waypoints, MinJerkParams(Vmax=tc.Vmax, Amax=tc.Amax, Jmax=tc.Jmax, gamma=tc.gamma_time_scale)
    )

    p = SynergyParams(
        Kp_task=800.0, Kv_task=60.0, Kff_x=_scalar(gains.Kff_x, 1.0),
        enable_inertia_comp=toggles.enable_inertia_comp,
        enable_gravity_comp=toggles.enable_gravity_comp,
        enable_velocity_comp=toggles.enable_velocity_comp,
        eps=num.eps, lam_os_max=getattr(num, "lam_os_max", 1e6),
        sigma_thresh=num.sigma_thresh, gate_pow=num.gate_pow,
        bisect_iters=ifc.bisect_iters,
        c_max=5.0, use_prior=False, prior_weight=0.0, synergy_strength=0.8,
    )
    ctrl = SynergyController(env, arm, p)
    ctrl.reset(q0)

    steps = int(pc.max_ep_duration / arm.dt)
    logs = TargetReachSimulatorTorch(env, arm, ctrl, traj, steps).run()

    k, tvec = logs.time(float(arm.dt))
    print("  target x_d:        ", _to_numpy(target))
    print("  final fingertip:   ", _to_numpy(logs.x_log[:k][k - 1, :2]))

    try:
        plot_all(logs, tvec, center=_to_numpy(center), targets=_to_numpy(target).reshape(1, 2))
        if run.animate:
            anims = make_animations(logs, tvec, env, playback=run.playback,
                                    downsample=run.downsample_anim,
                                    center=_to_numpy(center), targets=_to_numpy(target).reshape(1, 2))
            hold_anims(anims)
    except Exception as e:
        print(f"[RandomReach Synergy Torch] WARNING: plotting failed: {e}")

    print("[RandomReach Synergy Torch] demo complete.")
    plt.show()


if __name__ == "__main__":
    main()
