#!/usr/bin/env python3
# main_random_reach_tank_torch.py
# Torch counterpart of main_random_reach_tank.py (energy-tank passivity).
#
# Uses the same tuning as the numpy tank demo (K0=4000, Kff=2.0, critical zeta,
# strict_passivity=False default) -> matches numpy tracking (~1 mm). The torch
# controller now has the same Kff/zeta/strict_passivity knobs as numpy.

from __future__ import annotations

import numpy as np
import torch

from model_lib.torch.environment import Environment as EnvironmentTorch
from model_lib.torch.muscles import RigidTendonHillMuscle
from model_lib.torch.effector import RigidTendonArm26

from config import (
    PlantConfig, ControlToggles, ControlGains, Numerics,
    InternalForceConfig, TrajectoryConfig, RunConfig,
)
from trajectory.torch.minjerk import MinJerkLinearTrajectoryTorch, MinJerkParams
from controller.torch.energy_tank_controller import EnergyTankController, EnergyTankParams
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
        name="RandomReachTankEnvTorch",
    )
    q0 = torch.deg2rad(torch.tensor(pc.q0_deg, dtype=torch.get_default_dtype(), device=device))
    qd0 = torch.tensor(pc.qd0, dtype=torch.get_default_dtype(), device=device)
    env.reset(options={"joint_state": torch.cat([q0, qd0]).unsqueeze(0), "deterministic": True})
    return env, arm, q0


def _tank_params(num, toggles, ifc):
    dt = torch.get_default_dtype()
    return EnergyTankParams(
        D0=torch.diag(torch.tensor([25.0, 25.0], dtype=dt)),
        K0=torch.diag(torch.tensor([4000.0, 4000.0], dtype=dt)),
        Kff=2.0,
        zeta=1.0,
        KI=torch.tensor([0.0, 0.0], dtype=dt),
        Imax=torch.tensor([0.0, 0.0], dtype=dt),
        eps=float(getattr(num, "eps", 1e-6)),
        lam_os_smin_target=float(getattr(num, "lam_os_smin_target", 0.02)),
        lam_os_max=float(getattr(num, "lam_os_max", 200.0)),
        sigma_thresh=float(getattr(num, "sigma_thresh", 1e-4)),
        gate_pow=float(getattr(num, "gate_pow", 2.0)),
        enable_inertia_comp=bool(getattr(toggles, "enable_inertia_comp", True)),
        enable_gravity_comp=bool(getattr(toggles, "enable_gravity_comp", True)),
        enable_velocity_comp=bool(getattr(toggles, "enable_velocity_comp", True)),
        enable_joint_damping=bool(getattr(toggles, "enable_joint_damping", False)),
        enable_internal_force=False,
        cocon_a0=0.0,
        bisect_iters=int(getattr(ifc, "bisect_iters", 16)),
        linesearch_eps=float(getattr(num, "linesearch_eps", 1e-5)),
        linesearch_safety=float(getattr(num, "linesearch_safety", 0.5)),
        E0=0.15, Emin=1e-4, Emax=1.0,
    )


def main():
    print("[RandomReach Tank Torch] demo starting ...")
    pc = PlantConfig(); toggles = ControlToggles(); num = Numerics()
    ifc = InternalForceConfig(); tc = TrajectoryConfig(); run = RunConfig()

    env, arm, q0 = build_env_torch(pc)
    device = q0.device

    center = env.states["fingertip"][0, :2].clone()
    target = center + torch.tensor([0.10, 0.0], dtype=center.dtype, device=device)
    waypoints = torch.stack([center, target], dim=0)
    traj = MinJerkLinearTrajectoryTorch(
        waypoints, MinJerkParams(Vmax=tc.Vmax, Amax=tc.Amax, Jmax=tc.Jmax, gamma=tc.gamma_time_scale)
    )

    ctrl = EnergyTankController(env, arm, _tank_params(num, toggles, ifc))
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
        print(f"[RandomReach Tank Torch] WARNING: plotting failed: {e}")

    print("[RandomReach Tank Torch] demo complete.")
    plt.show()


if __name__ == "__main__":
    main()
