#!/usr/bin/env python3
"""
random_reach_main_motornet.py

Run RandomReach using a trained MotorNet policy (or fallback if no model is provided).

Example:
  python -m scripts.MOTORNET.random_reach_main_motornet --model motornet/saved_model/motornet_policy.pt

If no --model is given, MotorNet runs in NumPy fallback mode (inverse-dynamics-like).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from model_lib.environment_numpy import Environment
from model_lib.muscles_numpy import RigidTendonHillMuscle
from model_lib.effector_numpy import RigidTendonArm26

from config import PlantConfig, TrajectoryConfig, RunConfig
from tasks.random_reach import Task as RandomReachTask
from trajectory.minjerk import MinJerkLinearTrajectory, MinJerkParams
from sim.simulator import TargetReachSimulator
from plotting.plots import plot_all, make_animations, hold_anims

from controller.motornet_fixed import MotorNetFixed, MotorNetFixedParams


def build_env(pc: PlantConfig):
    muscle = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(
        muscle=muscle,
        timestep=pc.timestep,
        damping=pc.damping,
        n_ministeps=pc.n_ministeps,
        integration_method=pc.integration_method,
    )
    env = Environment(
        effector=arm,
        max_ep_duration=pc.max_ep_duration,
        action_noise=0.0,
        obs_noise=0.0,
        proprioception_delay=arm.dt,
        vision_delay=arm.dt,
        name="RandomReachEnv(MotorNet)",
    )
    q0 = np.deg2rad(np.array(pc.q0_deg))
    qd0 = np.array(pc.qd0)
    env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})
    return env, arm, q0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="", help="path to .pt checkpoint (optional)")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=float, default=0.10)
    ap.add_argument("--n_points", type=int, default=1)
    args = ap.parse_args()

    pc = PlantConfig()
    tc = TrajectoryConfig()
    run = RunConfig()

    env, arm, q0 = build_env(pc)

    task = RandomReachTask(n_points=args.n_points, radius=args.radius, seed=args.seed)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale))

    params = MotorNetFixedParams(device=args.device)
    ctrl = MotorNetFixed(env, arm, params)
    ctrl.reset(q0)

    if args.model:
        try:
            ctrl.load(args.model)
            ctrl.trained = True
            print(f"[motornet] loaded policy: {args.model}")
        except Exception as e:
            print(f"[motornet] WARNING: failed to load policy '{args.model}': {e}")

    steps = int(pc.max_ep_duration / arm.dt)
    sim = TargetReachSimulator(env, arm, ctrl, traj, steps)
    logs = sim.run()

    k, tvec = logs.time(arm.dt)
    plot_all(logs, tvec, center=task.center, targets=task.targets)

    if run.animate:
        anims = make_animations(
            logs,
            tvec,
            env,
            playback=run.playback,
            downsample=run.downsample_anim,
            center=task.center,
            targets=task.targets,
        )
        hold_anims(anims)

    print("Random reach (MotorNet) complete.")
    plt.show()


if __name__ == "__main__":
    main()
