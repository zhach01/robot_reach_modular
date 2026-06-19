#!/usr/bin/env python3
# main_center_out_torch.py
# Torch center-out PD/IF demo (torch counterpart of main_center_out.py).
#
# - EnvironmentTorch + RigidTendonArm26 torch plant
# - Optimized torch PD/IF controller (controller.torch.pd_if_controller)
# - Center-out task: out-and-back reaches to N targets on a circle
# - MinJerkLinearTrajectoryTorch

from __future__ import annotations

import math

import numpy as np
import torch

from model_lib.torch.environment import Environment as EnvironmentTorch
from model_lib.torch.muscles import RigidTendonHillMuscle
from model_lib.torch.effector import RigidTendonArm26

from config import (
    PlantConfig,
    ControlToggles,
    ControlGains,
    Numerics,
    InternalForceConfig,
    TrajectoryConfig,
    RunConfig,
)

from trajectory.torch.minjerk import MinJerkLinearTrajectoryTorch, MinJerkParams
from controller.torch.pd_if_controller import PDIFController, PDIFParams
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

    muscle = RigidTendonHillMuscle(
        min_activation=0.02, device=device, dtype=torch.get_default_dtype()
    )
    arm = RigidTendonArm26(
        muscle=muscle,
        timestep=pc.timestep,
        damping=pc.damping,
        n_ministeps=pc.n_ministeps,
        integration_method=pc.integration_method,
        device=device,
        dtype=torch.get_default_dtype(),
    )
    env = EnvironmentTorch(
        effector=arm,
        max_ep_duration=pc.max_ep_duration,
        action_noise=0.0,
        obs_noise=0.0,
        action_frame_stacking=1,
        proprioception_delay=arm.dt,
        vision_delay=arm.dt,
        name="CenterOutEnvTorch",
    )
    q0 = torch.deg2rad(
        torch.tensor(pc.q0_deg, dtype=torch.get_default_dtype(), device=device)
    )
    qd0 = torch.tensor(pc.qd0, dtype=torch.get_default_dtype(), device=device)
    joint0 = torch.cat([q0, qd0]).unsqueeze(0)  # (1,4)
    env.reset(options={"joint_state": joint0, "deterministic": True})
    return env, arm, q0


def main():
    print("[CenterOut PD/IF Torch] demo starting ...")

    pc = PlantConfig()
    toggles = ControlToggles()
    num = Numerics()
    ifc = InternalForceConfig()
    tc = TrajectoryConfig()
    run = RunConfig()

    env, arm, q0 = build_env_torch(pc)
    device = q0.device

    # --- Center-out task: N targets on a circle, out-and-back ---
    n_targets = 4
    radius = 0.10
    center = env.states["fingertip"][0, :2].clone()  # (2,)
    angles = [2.0 * math.pi * i / n_targets for i in range(n_targets)]
    targets = torch.stack(
        [center + radius * torch.tensor([math.cos(a), math.sin(a)],
                                        dtype=center.dtype, device=device)
         for a in angles],
        dim=0,
    )  # (N,2)

    wp = [center]
    for i in range(n_targets):
        wp.append(targets[i])
        wp.append(center)
    waypoints = torch.stack(wp, dim=0)  # (2N+1, 2)

    mj_params = MinJerkParams(
        Vmax=tc.Vmax, Amax=tc.Amax, Jmax=tc.Jmax, gamma=tc.gamma_time_scale
    )
    traj = MinJerkLinearTrajectoryTorch(waypoints, mj_params)

    # --- Optimized torch PD/IF gains (stiffer for the larger center-out reaches) ---
    p = PDIFParams(
        Kp_task=[2400.0, 2400.0],
        damping_ratio=1.0,
        Kff=1.0,
        use_critical_damping=True,
        enable_inertia_comp=bool(getattr(toggles, "enable_inertia_comp", True)),
        enable_gravity_comp=bool(getattr(toggles, "enable_gravity_comp", True)),
        enable_coriolis_comp=bool(getattr(toggles, "enable_velocity_comp", True)),
        enable_nullspace=True,
        Kp_null=20.0,
        Kd_null=5.0,
        eps=float(getattr(num, "eps", 1e-6)),
        lam_os_max=float(getattr(num, "lam_os_max", 200.0)),
        sigma_thresh=float(getattr(num, "sigma_thresh", 1e-4)),
        gate_pow=float(getattr(num, "gate_pow", 2.0)),
        bisect_iters=int(getattr(ifc, "bisect_iters", 12)),
        enable_internal_force=False,
        cocon_level=0.0,
    )

    ctrl = PDIFController(env, arm, p)
    steps = int(pc.max_ep_duration / arm.dt)
    sim = TargetReachSimulatorTorch(env, arm, ctrl, traj, steps)
    logs = sim.run()

    k, tvec = logs.time(float(arm.dt))
    x_log = logs.x_log[:k]
    x_final = x_log[k - 1, :2]
    print("  center:           ", _to_numpy(center))
    print("  final fingertip:  ", _to_numpy(x_final))

    try:
        plot_all(logs, tvec, center=_to_numpy(center), targets=_to_numpy(targets))
        if run.animate:
            anims = make_animations(
                logs, tvec, env,
                playback=run.playback, downsample=run.downsample_anim,
                center=_to_numpy(center), targets=_to_numpy(targets),
            )
            hold_anims(anims)
    except Exception as e:
        print(f"[CenterOut PD/IF Torch] WARNING: plotting/animation failed: {e}")

    print("[CenterOut PD/IF Torch] demo complete.")
    plt.show()


if __name__ == "__main__":
    main()
