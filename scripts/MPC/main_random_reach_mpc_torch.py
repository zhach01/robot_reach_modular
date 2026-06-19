#!/usr/bin/env python3
# scripts/MPC/main_random_reach_mpc_torch.py
# Torch counterpart of main_random_reach_mpc.py (canonical NMPC).

from __future__ import annotations

import numpy as np
import torch

from model_lib.torch.environment import Environment as EnvironmentTorch
from model_lib.torch.muscles import RigidTendonHillMuscle
from model_lib.torch.effector import RigidTendonArm26

from config import PlantConfig, TrajectoryConfig, RunConfig
from trajectory.torch.minjerk import MinJerkLinearTrajectoryTorch, MinJerkParams
from controller.torch.nmpc_task import NonlinearMPCControllerTorch, NMPCParams
from sim.torch.simulator import TargetReachSimulatorTorch

from plotting.plots import plot_all, make_animations, hold_anims
import matplotlib.pyplot as plt


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


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
        name="RandomReachNMPCEnvTorch",
    )
    q0 = torch.deg2rad(torch.tensor(pc.q0_deg, dtype=torch.get_default_dtype(), device=device))
    qd0 = torch.tensor(pc.qd0, dtype=torch.get_default_dtype(), device=device)
    env.reset(options={"joint_state": torch.cat([q0, qd0]).unsqueeze(0), "deterministic": True})
    return env, arm, q0


def main():
    print("[RandomReach NMPC Torch] demo starting ...")
    pc = PlantConfig(); tc = TrajectoryConfig(); run = RunConfig()

    env, arm, q0 = build_env_torch(pc)
    device = q0.device

    center = env.states["fingertip"][0, :2].clone()
    target = center + torch.tensor([0.10, 0.0], dtype=center.dtype, device=device)
    waypoints = torch.stack([center, target], dim=0)
    traj = MinJerkLinearTrajectoryTorch(
        waypoints, MinJerkParams(Vmax=tc.Vmax, Amax=tc.Amax, Jmax=tc.Jmax, gamma=tc.gamma_time_scale)
    )

    p = NMPCParams(N=12, dt_mpc=float(arm.dt))
    ctrl = NonlinearMPCControllerTorch(env, arm, p)
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
        print(f"[RandomReach NMPC Torch] WARNING: plotting failed: {e}")

    print("[RandomReach NMPC Torch] demo complete.")
    plt.show()


if __name__ == "__main__":
    main()
