#!/usr/bin/env python3
# scripts/SYNERGY/main_center_out_synergy.py
# Center-out demo for the canonical (full) SynergyController.

import numpy as np
import matplotlib.pyplot as plt

from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26

from config import (
    PlantConfig, ControlToggles, ControlGains, Numerics,
    InternalForceConfig, TrajectoryConfig, RunConfig,
)

from tasks.numpy.base_task import CenterOutTask
from trajectory.numpy.minjerk import MinJerkLinearTrajectory, MinJerkParams
from controller.numpy.synergy_controller import SynergyController, SynergyParams
from sim.numpy.simulator import TargetReachSimulator
from plotting.plots import plot_all, make_animations, hold_anims


def _scalar(v, default=1.0):
    v = np.atleast_1d(np.asarray(v, dtype=float))
    return float(v[0]) if v.size else float(default)


def build_env(pc: PlantConfig):
    muscle = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(
        muscle=muscle, timestep=pc.timestep, damping=pc.damping,
        n_ministeps=pc.n_ministeps, integration_method=pc.integration_method,
    )
    env = Environment(
        effector=arm, max_ep_duration=pc.max_ep_duration, action_noise=0.0,
        obs_noise=0.0, proprioception_delay=arm.dt, vision_delay=arm.dt,
        name="CenterOutSynergyEnv",
    )
    q0 = np.deg2rad(np.array(pc.q0_deg))
    qd0 = np.array(pc.qd0)
    env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})
    return env, arm, q0


def main():
    print("[CenterOut Synergy] demo starting ...")
    pc = PlantConfig(); toggles = ControlToggles(); gains = ControlGains()
    num = Numerics(); ifc = InternalForceConfig(); tc = TrajectoryConfig(); run = RunConfig()

    env, arm, q0 = build_env(pc)

    task = CenterOutTask(n_targets=4, radius=0.10)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(
        waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
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
    sim = TargetReachSimulator(env, arm, ctrl, traj, steps)
    logs = sim.run()

    k, tvec = logs.time(arm.dt)
    plot_all(logs, tvec, center=task.center, targets=task.targets)
    if run.animate:
        anims = make_animations(logs, tvec, env, playback=run.playback,
                                downsample=run.downsample_anim,
                                center=task.center, targets=task.targets)
        hold_anims(anims)

    print("[CenterOut Synergy] demo complete.")
    plt.show()


if __name__ == "__main__":
    main()
