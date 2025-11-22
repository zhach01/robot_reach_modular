#!/usr/bin/env python3
# scripts/make_teacher_dataset_random_reach_a.py
from __future__ import annotations
import os, argparse
import numpy as np
from typing import Tuple, List

from model_lib.environment_numpy import Environment
from model_lib.muscles_numpy import RigidTendonHillMuscle
from model_lib.effector_numpy import RigidTendonArm26
from config import PlantConfig, TrajectoryConfig
from trajectory.minjerk import MinJerkLinearTrajectory, MinJerkParams
from controller.nmpc_task import NonlinearMPCController, NMPCParams
from controller.hybrid_bc_a import HookedControllerA
from sim.simulator import TargetReachSimulator

SAVE_PATH_DEFAULT = "data/random_reach_a_ds.npz"

# ---------------------------------------------------------------------

def build_env(pc: PlantConfig) -> Tuple[Environment, RigidTendonArm26, np.ndarray]:
    """Create (env, arm) and reset with deterministic q0/qd0."""
    muscle = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(
        muscle=muscle, timestep=pc.timestep, damping=pc.damping,
        n_ministeps=pc.n_ministeps, integration_method=pc.integration_method,
    )
    env = Environment(
        effector=arm, max_ep_duration=pc.max_ep_duration,
        action_noise=0.0, obs_noise=0.0,
        proprioception_delay=arm.dt, vision_delay=arm.dt,
        name="RandomReach(NMPC-teacher,a-dataset)",
    )
    q0 = np.deg2rad(np.array(pc.q0_deg, dtype=np.float32))
    qd0 = np.array(pc.qd0, dtype=np.float32)
    env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})
    return env, arm, q0

def jitter_q0_deg(q0_deg: np.ndarray, jitter_deg: float) -> np.ndarray:
    if jitter_deg <= 0.0:
        return q0_deg.astype(np.float32)
    j = (np.random.rand(*q0_deg.shape).astype(np.float32) * 2.0 - 1.0) * jitter_deg
    return (q0_deg + j).astype(np.float32)

def sample_annulus(rmin: float, rmax: float, n: int) -> np.ndarray:
    """Uniform over area of an annulus via sqrt-trick."""
    # draw radii^2 uniformly, then sqrt
    u = np.random.uniform(0.0, 1.0, size=n)
    radii = np.sqrt((rmax**2 - rmin**2) * u + rmin**2)
    return radii.astype(np.float32)

def build_center(env, mode: str, fixed_xy: Tuple[float,float]) -> np.ndarray:
    if mode == "fixed":
        return np.array([fixed_xy[0], fixed_xy[1]], dtype=np.float32)
    # "current": center = current end-effector position
    return env.states["cartesian"][0, :2].astype(np.float32)

def build_waypoints(
    center_xy: np.ndarray,
    goals_per_ep: int,
    rmin: float,
    rmax: float,
    return_to_center: bool,
    disk_uniform: bool,
) -> np.ndarray:
    """Create a [K,2] sequence of waypoints (center-out pattern optional)."""
    n = goals_per_ep
    angles = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)

    if disk_uniform:
        # uniform in disk between 0..rmax (or rmin..rmax if rmin>0)
        radii = sample_annulus(rmin, rmax, n)
    else:
        # annulus sampling as well (kept same helper to ensure uniform area)
        radii = sample_annulus(rmin, rmax, n)

    targets: List[np.ndarray] = [
        center_xy + r * np.array([np.cos(a), np.sin(a)], dtype=np.float32)
        for r, a in zip(radii, angles)
    ]

    if return_to_center:
        seq = [center_xy]
        for tgt in targets:
            seq.append(tgt)
            seq.append(center_xy)
        waypoints = np.vstack(seq)
    else:
        waypoints = np.vstack([center_xy] + targets)

    return waypoints.astype(np.float32)

# ---------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=SAVE_PATH_DEFAULT)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--goals_per_ep", type=int, default=6)
    ap.add_argument("--radius_min", type=float, default=0.07)
    ap.add_argument("--radius_max", type=float, default=0.16)
    ap.add_argument("--center_mode", type=str, choices=["current","fixed"], default="current")
    ap.add_argument("--center_x", type=float, default=0.00)   # used if center_mode=fixed
    ap.add_argument("--center_y", type=float, default=0.55)   # used if center_mode=fixed
    ap.add_argument("--return_to_center", action="store_true", default=True)
    ap.add_argument("--no-return_to_center", dest="return_to_center", action="store_false")
    ap.add_argument("--disk_uniform", action="store_true", default=True)   # uniform in area
    ap.add_argument("--no-disk_uniform", dest="disk_uniform", action="store_false")
    ap.add_argument("--q0_jitter_deg", type=float, default=8.0)  # per-episode shoulder/elbow jitter
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)

    pc, tc = PlantConfig(), TrajectoryConfig()

    O_all, A_all = [], []

    for ep in range(args.episodes):
        # -------- build env ----------
        env, arm, _ = build_env(pc)

        # per-episode q0 jitter (optional)
        if args.q0_jitter_deg > 0:
            q0_deg_j = jitter_q0_deg(np.array(pc.q0_deg, dtype=np.float32), args.q0_jitter_deg)
            q0 = np.deg2rad(q0_deg_j)
            qd0 = np.array(pc.qd0, dtype=np.float32)
            env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})

        # -------- waypoints ----------
        center_xy = build_center(
            env, mode=args.center_mode, fixed_xy=(args.center_x, args.center_y)
        )
        waypoints = build_waypoints(
            center_xy=center_xy,
            goals_per_ep=args.goals_per_ep,
            rmin=args.radius_min,
            rmax=args.radius_max,
            return_to_center=args.return_to_center,
            disk_uniform=args.disk_uniform,
        )

        # -------- trajectory ----------
        traj = MinJerkLinearTrajectory(
            waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
        )

        # -------- teacher (NMPC generates final activations) ----------
        p = NMPCParams(N=12, dt_mpc=arm.dt, use_muscles=True)
        teacher = NonlinearMPCController(env, arm, p)
        ctrl = HookedControllerA(env, teacher)

        steps = int(env.max_ep_duration / arm.dt)
        _ = TargetReachSimulator(env, arm, ctrl, traj, steps).run()

        if ctrl.obs:
            O_all.append(np.asarray(ctrl.obs, dtype=np.float32))
            A_all.append(np.asarray(ctrl.act, dtype=np.float32))

    if not O_all:
        raise RuntimeError("No data collected. Try increasing --episodes or --goals_per_ep.")

    O = np.concatenate(O_all, axis=0)
    A = np.concatenate(A_all, axis=0)

    mean = O.mean(axis=0).astype(np.float32)
    std  = np.clip(O.std(axis=0), 1e-6, None).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, O=O, A=A, mean=mean, std=std,
                        meta=dict(
                            episodes=args.episodes,
                            goals_per_ep=args.goals_per_ep,
                            radius_min=args.radius_min,
                            radius_max=args.radius_max,
                            center_mode=args.center_mode,
                            center_xy=(args.center_x, args.center_y),
                            return_to_center=bool(args.return_to_center),
                            disk_uniform=bool(args.disk_uniform),
                            q0_jitter_deg=float(args.q0_jitter_deg),
                            seed=int(args.seed),
                        ))
    print(f"Saved dataset: {args.out} | O={O.shape} A={A.shape}")

if __name__ == "__main__":
    main()
