#!/usr/bin/env python3
# scripts/main_random_reach_bc_a.py
from __future__ import annotations
import argparse, numpy as np, matplotlib.pyplot as plt
from typing import Tuple, List

from model_lib.environment_numpy import Environment
from model_lib.muscles_numpy import RigidTendonHillMuscle
from model_lib.effector_numpy import RigidTendonArm26
from config import PlantConfig, TrajectoryConfig, RunConfig
from trajectory.minjerk import MinJerkLinearTrajectory, MinJerkParams
from controller.hybrid_bc_a import RLPolicyParams, RLPolicy, RLControllerA, RLControllerAParams
from sim.simulator import TargetReachSimulator
from plotting.plots import plot_all, make_animations, hold_anims

# -------------------- helpers --------------------

def build_env(pc: PlantConfig) -> Tuple[Environment, RigidTendonArm26]:
    muscle = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(
        muscle=muscle, timestep=pc.timestep, damping=pc.damping,
        n_ministeps=pc.n_ministeps, integration_method=pc.integration_method,
    )
    env = Environment(
        effector=arm, max_ep_duration=pc.max_ep_duration,
        action_noise=0.0, obs_noise=0.0,
        proprioception_delay=arm.dt, vision_delay=arm.dt,
        name="RandomReach(BC a-policy test)",
    )
    q0 = np.deg2rad(np.array(pc.q0_deg, dtype=np.float32))
    qd0 = np.array(pc.qd0, dtype=np.float32)
    env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})
    return env, arm

def jitter_q0_deg(q0_deg: np.ndarray, jitter_deg: float) -> np.ndarray:
    if jitter_deg <= 0.0:
        return q0_deg.astype(np.float32)
    j = (np.random.rand(*q0_deg.shape).astype(np.float32) * 2.0 - 1.0) * jitter_deg
    return (q0_deg + j).astype(np.float32)

def current_center(env) -> np.ndarray:
    return env.states["cartesian"][0, :2].astype(np.float32)

def pick_center(env, mode: str, fixed_xy: Tuple[float, float]) -> np.ndarray:
    if mode == "fixed":
        return np.array([fixed_xy[0], fixed_xy[1]], dtype=np.float32)
    return current_center(env)

def sample_radii(rmin: float, rmax: float, n: int, uniform_area: bool) -> np.ndarray:
    rmin = float(max(0.0, rmin))
    rmax = float(max(rmin, rmax))
    if uniform_area:
        u = np.random.uniform(0.0, 1.0, size=n).astype(np.float32)
        return np.sqrt((rmax**2 - rmin**2) * u + rmin**2)
    else:
        return np.random.uniform(rmin, rmax, size=n).astype(np.float32)

def build_waypoints(
    env,
    goals_per_ep: int,
    center_xy: np.ndarray,
    rmin: float,
    rmax: float,
    return_to_center: bool,
    disk_uniform: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (waypoints, center, targets)."""
    n = int(goals_per_ep)
    angles = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)
    radii  = sample_radii(rmin, rmax, n, uniform_area=disk_uniform)
    targets = np.vstack([
        center_xy + r * np.array([np.cos(a), np.sin(a)], dtype=np.float32)
        for r, a in zip(radii, angles)
    ]).astype(np.float32)

    if return_to_center:
        seq: List[np.ndarray] = [center_xy]
        for t in targets:
            seq.append(t); seq.append(center_xy)
        waypoints = np.vstack(seq).astype(np.float32)
        plot_targets = targets  # for plotting
    else:
        waypoints = np.vstack([center_xy, *targets]).astype(np.float32)
        plot_targets = targets

    return waypoints, center_xy, plot_targets

# -------------------- CLI --------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",default="models/random_reach_bc_a.pt", required=False, help="Path to models/random_reach_bc_a.pt")
    ap.add_argument("--device", type=str, default="cpu")

    # target generation
    ap.add_argument("--goals", type=int, default=8)
    ap.add_argument("--radius_min", type=float, default=0.07)
    ap.add_argument("--radius_max", type=float, default=0.16)
    ap.add_argument("--center_mode", type=str, choices=["current","fixed"], default="current")
    ap.add_argument("--center_x", type=float, default=0.00)
    ap.add_argument("--center_y", type=float, default=0.55)
    ap.add_argument("--return_to_center", action="store_true", default=True)
    ap.add_argument("--no-return_to_center", dest="return_to_center", action="store_false")
    ap.add_argument("--disk_uniform", action="store_true", default=True)
    ap.add_argument("--no-disk_uniform", dest="disk_uniform", action="store_false")
    ap.add_argument("--q0_jitter_deg", type=float, default=0.0)

    # sim + viz
    ap.add_argument("--steps", type=int, default=0, help="Override episode steps (0=env default)")
    ap.add_argument("--seed", type=int, default=202)
    ap.add_argument("--animate", action="store_true", default=False)
    return ap.parse_args()

# -------------------- main --------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)

    pc, tc, run = PlantConfig(), TrajectoryConfig(), RunConfig()
    env, arm = build_env(pc)

    # optional q0 jitter (reseed start pose)
    if args.q0_jitter_deg > 0.0:
        q0_deg_j = jitter_q0_deg(np.array(pc.q0_deg, dtype=np.float32), args.q0_jitter_deg)
        q0 = np.deg2rad(q0_deg_j); qd0 = np.array(pc.qd0, dtype=np.float32)
        env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})

    # infer m from geometry; build obs/act dims for a-policy
    g = env.states["geometry"]  # (B, 2+DOF, m)
    if g.ndim != 3:
        raise RuntimeError(f"Unexpected geometry tensor shape {g.shape}; expected (B, K, m).")
    m = int(g.shape[2])
    obs_dim = 14 + 5 * m
    act_dim = m

    # build waypoints per CLI flags
    center_xy = pick_center(env, args.center_mode, (args.center_x, args.center_y))
    waypoints, center, targets = build_waypoints(
        env,
        goals_per_ep=args.goals,
        center_xy=center_xy,
        rmin=args.radius_min,
        rmax=args.radius_max,
        return_to_center=args.return_to_center,
        disk_uniform=args.disk_uniform,
    )

    traj = MinJerkLinearTrajectory(
        waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
    )

    # policy (rebuilds on load if ckpt dims differ)
    ppi = RLPolicyParams(obs_dim=obs_dim, act_dim=act_dim, hidden=(128,128), device=args.device)
    pi  = RLPolicy(ppi)
    pi.load(args.ckpt)

    # controller
    ctrl = RLControllerA(env, arm, pi, RLControllerAParams())

    # sim (no 'animate' arg in ctor — keep ctor signature compatible)
    steps = args.steps if args.steps > 0 else int(env.max_ep_duration / arm.dt)
    logs  = TargetReachSimulator(env, arm, ctrl, traj, steps).run()

    # plotting + optional animation
    k, tvec = logs.time(arm.dt)
    plot_all(logs, tvec, center=center, targets=targets)
    if args.animate:
        anims = make_animations(
            logs, tvec, env,
            playback=run.playback, downsample=run.downsample_anim,
            center=center, targets=targets
        )
        hold_anims(anims)
    print("Random reach (BC a-policy) complete.")
    plt.show()

if __name__ == "__main__":
    main()
