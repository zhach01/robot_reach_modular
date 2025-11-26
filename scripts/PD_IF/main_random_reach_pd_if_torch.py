#!/usr/bin/env python3
# main_random_reach_pd_if_torch.py
# Torch version of the NumPy random-reach PD/IF demo.
#
# - Uses EnvironmentTorch + RigidTendonArm26 Torch plant
# - Uses the same PDIFController / PDIFParams (backend-agnostic)
# - Trajectory: MinJerkLinearTrajectoryTorch
#
# It mirrors your original script as closely as possible:
#   - builds plant/env from PlantConfig
#   - runs a single 10 cm reach
#   - logs and (optionally) plots/animates via plotting.plots

from __future__ import annotations

import numpy as np
import torch

from model_lib.environment_torch import Environment as EnvironmentTorch
from model_lib.muscles_torch import RigidTendonHillMuscle
from model_lib.effector_torch import RigidTendonArm26

from config import (
    PlantConfig,
    ControlToggles,
    ControlGains,
    Numerics,
    InternalForceConfig,
    TrajectoryConfig,
    RunConfig,
)

from trajectory.minjerk_torch import MinJerkLinearTrajectoryTorch, MinJerkParams
from controller.pd_if_controller_torch import PDIFController, PDIFParams
from sim.simulator_torch import TargetReachSimulatorTorch

from plotting.plots import plot_all, make_animations, hold_anims
import matplotlib.pyplot as plt


def _to_numpy(x):
    """Helper: convert torch or numpy to plain np.ndarray."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_env_torch(pc: PlantConfig):
    """
    Build Torch-based plant + env, mirroring the NumPy build_env().
    """
    # Use double precision like other Torch smoke tests
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    muscle = RigidTendonHillMuscle(
        min_activation=0.02,
        device=device,
        dtype=torch.get_default_dtype(),
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
        name="RandomReachEnvTorch",
    )

    # Initial joint state (B=1)
    q0 = torch.deg2rad(
        torch.tensor(pc.q0_deg, dtype=torch.get_default_dtype(), device=device)
    )  # (2,)
    qd0 = torch.tensor(pc.qd0, dtype=torch.get_default_dtype(), device=device)  # (2,)
    joint0 = torch.cat([q0, qd0]).unsqueeze(0)  # (1, 4)

    env.reset(
        options={
            "joint_state": joint0,
            "deterministic": True,
        }
    )

    return env, arm, q0


def main():
    print("[RandomReach PD/IF Torch] demo starting ...")

    pc = PlantConfig()
    toggles = ControlToggles()
    gains = ControlGains()
    num = Numerics()
    ifc = InternalForceConfig()
    tc = TrajectoryConfig()
    run = RunConfig()

    # --- Build Torch env/arm ---
    env, arm, q0 = build_env_torch(pc)
    device = q0.device

    # --- Simple one-target random reach (like RandomReachTask(n_points=1, radius=0.10)) ---
    # Use current fingertip as center, target = center + 0.10 along +x
    fingertip0 = env.states["fingertip"][0, :2]  # (2,)
    center = fingertip0
    radius = 0.10
    target = center + torch.tensor([radius, 0.0], dtype=center.dtype, device=device)

    waypoints = torch.stack([center, target], dim=0)  # (2, 2)

    mj_params = MinJerkParams(
        Vmax=tc.Vmax,
        Amax=tc.Amax,
        Jmax=tc.Jmax,
        gamma=tc.gamma_time_scale,
    )
    traj = MinJerkLinearTrajectoryTorch(waypoints, mj_params)

    # --- PD/IF params (same structure as NumPy version) ---
    p = PDIFParams(
        Kp_x=gains.Kp_x,
        Kff_x=gains.Kff_x,
        Kp_q=gains.Kp_q,
        Kd_q=gains.Kd_q,
        eps=num.eps,
        lam_os_smin_target=num.lam_os_smin_target,
        lam_os_max=num.lam_os_max,
        sigma_thresh=num.sigma_thresh,
        gate_pow=num.gate_pow,
        enable_internal_force=toggles.enable_internal_force,
        enable_inertia_comp=toggles.enable_inertia_comp,
        enable_gravity_comp=toggles.enable_gravity_comp,
        enable_velocity_comp=toggles.enable_velocity_comp,
        enable_joint_damping=toggles.enable_joint_damping,
        cocon_a0=ifc.cocon_a0,
        bisect_iters=ifc.bisect_iters,
        linesearch_eps=num.linesearch_eps,
        linesearch_safety=num.linesearch_safety,
        
    )

    ctrl = PDIFController(env, arm, p)

    steps = int(pc.max_ep_duration / arm.dt)
    sim = TargetReachSimulatorTorch(env, arm, ctrl, traj, steps)

    # --- Run simulation ---
    logs = sim.run()

    # --- Basic tracking printout (like other smoke tests) ---
    k, tvec = logs.time(float(arm.dt))
    x_log = logs.x_log[:k]  # (T, 4) or (T, >=2)

    center_np = _to_numpy(center)
    target_np = _to_numpy(target)

    print("  fingertip (before):", center_np)
    print("  target x_d:        ", target_np)

    for idx in [0, 1, 2, 4, 9, 19, 29]:
        if idx < k:
            x = x_log[idx, :2]
            print(f"    step {idx:3d}: x = {_to_numpy(x)}")

    x_final = x_log[k - 1, :2]
    err_final = torch.linalg.norm(
        x_final - target
        if isinstance(x_final, torch.Tensor)
        else torch.as_tensor(x_final, dtype=torch.get_default_dtype(), device=device)
    )

    print("  final fingertip:   ", _to_numpy(x_final))
    print(f"  final |x - x_d|:    {float(err_final):.6f} m")

    # --- Plotting & animation (best-effort; depends on logs/plotting torch support) ---
    try:
        # center/targets for plotting as NumPy
        center_plot = center_np
        targets_plot = target_np.reshape(1, 2)

        plot_all(logs, tvec, center=center_plot, targets=targets_plot)

        if run.animate:
            anims = make_animations(
                logs,
                tvec,
                env,
                playback=run.playback,
                downsample=run.downsample_anim,
                center=center_plot,
                targets=targets_plot,
            )
            hold_anims(anims)
    except Exception as e:
        print(f"[RandomReach PD/IF Torch] WARNING: plotting/animation failed: {e}")

    print("[RandomReach PD/IF Torch] demo complete.")
    plt.show()


if __name__ == "__main__":
    main()
