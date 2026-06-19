#!/usr/bin/env python3
# main_center_out_sliding_torch.py
# Torch center-out sliding-mode demo (torch counterpart of main_center_out_sliding.py).

from __future__ import annotations

import math

import numpy as np
import torch

from model_lib.torch.environment import Environment as EnvironmentTorch
from model_lib.torch.muscles import RigidTendonHillMuscle
from model_lib.torch.effector import RigidTendonArm26

from config import (
    PlantConfig, ControlToggles, Numerics, InternalForceConfig,
    TrajectoryConfig, RunConfig,
)
from trajectory.torch.minjerk import MinJerkLinearTrajectoryTorch, MinJerkParams
from controller.torch.sliding_mode import SlidingModeController, SlidingModeParams
from sim.torch.simulator import TargetReachSimulatorTorch

from plotting.plots import plot_all, make_animations, hold_anims
import matplotlib.pyplot as plt


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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
        name="CenterOutSlidingEnvTorch",
    )
    q0 = torch.deg2rad(torch.tensor(pc.q0_deg, dtype=torch.get_default_dtype(), device=device))
    qd0 = torch.tensor(pc.qd0, dtype=torch.get_default_dtype(), device=device)
    env.reset(options={"joint_state": torch.cat([q0, qd0]).unsqueeze(0), "deterministic": True})
    return env, arm, q0


def main():
    print("[CenterOut Sliding Torch] demo starting ...")
    pc = PlantConfig(); toggles = ControlToggles(); num = Numerics()
    ifc = InternalForceConfig(); tc = TrajectoryConfig(); run = RunConfig()

    env, arm, q0 = build_env_torch(pc)
    device = q0.device

    n_targets, radius = 4, 0.10
    center = env.states["fingertip"][0, :2].clone()
    angles = [2.0 * math.pi * i / n_targets for i in range(n_targets)]
    targets = torch.stack(
        [center + radius * torch.tensor([math.cos(a), math.sin(a)], dtype=center.dtype, device=device)
         for a in angles], dim=0,
    )
    wp = [center]
    for i in range(n_targets):
        wp.append(targets[i]); wp.append(center)
    waypoints = torch.stack(wp, dim=0)
    traj = MinJerkLinearTrajectoryTorch(
        waypoints, MinJerkParams(Vmax=tc.Vmax, Amax=tc.Amax, Jmax=tc.Jmax, gamma=tc.gamma_time_scale)
    )

    # mirrors the numpy center-out sliding gains (gentler surface, thick boundary
    # layer, higher robustness). The mirrored controller's cond(J) blend + elbow
    # floor keep the far reaches stable at the full radius.
    p = SlidingModeParams(
        lambda_surf=[18.0, 18.0],
        K_switch=[18.0, 18.0],
        phi=[0.075, 0.075],
        Kff_x=[1.0, 1.0],
        enable_inertia_comp=bool(getattr(toggles, "enable_inertia_comp", True)),
        enable_gravity_comp=bool(getattr(toggles, "enable_gravity_comp", True)),
        enable_velocity_comp=bool(getattr(toggles, "enable_velocity_comp", True)),
        eps=float(getattr(num, "eps", 1e-6)),
        lam_os_max=float(getattr(num, "lam_os_max", 1e6)),
        sigma_thresh=float(getattr(num, "sigma_thresh", 1e-4)),
        gate_pow=float(getattr(num, "gate_pow", 2.0)),
        bisect_iters=int(getattr(ifc, "bisect_iters", 16)),
    )

    ctrl = SlidingModeController(env, arm, p)
    steps = int(pc.max_ep_duration / arm.dt)
    logs = TargetReachSimulatorTorch(env, arm, ctrl, traj, steps).run()

    k, tvec = logs.time(float(arm.dt))
    print("  center:           ", _to_numpy(center))
    print("  final fingertip:  ", _to_numpy(logs.x_log[:k][k - 1, :2]))

    try:
        plot_all(logs, tvec, center=_to_numpy(center), targets=_to_numpy(targets))
        if run.animate:
            anims = make_animations(logs, tvec, env, playback=run.playback,
                                    downsample=run.downsample_anim,
                                    center=_to_numpy(center), targets=_to_numpy(targets))
            hold_anims(anims)
    except Exception as e:
        print(f"[CenterOut Sliding Torch] WARNING: plotting failed: {e}")

    print("[CenterOut Sliding Torch] demo complete.")
    plt.show()


if __name__ == "__main__":
    main()
