#!/usr/bin/env python3
# scripts/make_teacher_dataset_random_reach.py
from __future__ import annotations
import os
import numpy as np
from typing import Tuple

from model_lib.environment_numpy import Environment
from model_lib.muscles_numpy import RigidTendonHillMuscle
from model_lib.effector_numpy import RigidTendonArm26
from config import PlantConfig, TrajectoryConfig
from tasks.random_reach import Task as RandomReachTask
from trajectory.minjerk import MinJerkLinearTrajectory, MinJerkParams
from controller.nmpc_task import NonlinearMPCController, NMPCParams
from controller.hybrid_mpc_rl import HookedController   # <- will emit 14-D obs
from sim.simulator import TargetReachSimulator

SAVE_PATH = "data/random_reach_mpc_ds.npz"

def build_env(pc: PlantConfig) -> Tuple[Environment, RigidTendonArm26, np.ndarray]:
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
        name="RandomReach(NMPC-teacher)",
    )
    q0 = np.deg2rad(np.array(pc.q0_deg))
    qd0 = np.array(pc.qd0)
    env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})
    return env, arm, q0

def main():
    np.random.seed(42)
    pc, tc = PlantConfig(), TrajectoryConfig()

    n_episodes   = 24   # episodes to record
    goals_per_ep = 6    # target switches per episode

    O_all, A_all = [], []

    for ep in range(n_episodes):
        env, arm, _ = build_env(pc)
        task = RandomReachTask(n_points=goals_per_ep, radius=0.10, seed=ep)
        try:
            waypoints = task.build_waypoints(env, reach_T=1.2, settle_T=0.3)
        except TypeError:
            waypoints = task.build_waypoints(env)

        traj = MinJerkLinearTrajectory(
            waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
        )

        p = NMPCParams(N=12, dt_mpc=arm.dt, use_muscles=True)
        teacher = NonlinearMPCController(env, arm, p)
        ctrl = HookedController(env, teacher)   # collects 14-D obs + τ_des

        steps = int(env.max_ep_duration / arm.dt)
        _ = TargetReachSimulator(env, arm, ctrl, traj, steps).run()

        if ctrl.obs:
            O_all.append(np.asarray(ctrl.obs, dtype=np.float32))  # (T, 14)
            A_all.append(np.asarray(ctrl.act, dtype=np.float32))  # (T, 2)

    if not O_all:
        raise RuntimeError("No data collected; increase episode length or targets.")

    O = np.concatenate(O_all, axis=0)
    A = np.concatenate(A_all, axis=0)
    mean = O.mean(axis=0).astype(np.float32)
    std  = np.clip(O.std(axis=0), 1e-6, None).astype(np.float32)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savez_compressed(SAVE_PATH, O=O, A=A, mean=mean, std=std)
    print(f"Saved dataset: {SAVE_PATH} | O={O.shape} A={A.shape}")

if __name__ == "__main__":
    main()
