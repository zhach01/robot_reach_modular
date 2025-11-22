#!/usr/bin/env python3
# scripts/main_random_reach_MPC_RL_controller_hybrid.py

import argparse, numpy as np, matplotlib.pyplot as plt

from model_lib.environment_numpy import Environment
from model_lib.muscles_numpy import RigidTendonHillMuscle
from model_lib.effector_numpy import RigidTendonArm26
from config import PlantConfig, TrajectoryConfig, RunConfig

# Prefer your existing task if present
try:
    from tasks.random_reach import Task as RandomReachTask
    HAVE_TASK = True
except Exception:
    HAVE_TASK = False

from trajectory.minjerk import MinJerkLinearTrajectory, MinJerkParams
from controller.hybrid_bc_a import RLPolicyParams, RLPolicy
from controller.nmpc_task import NMPCParams
from controller.mpc_rl_hybrid import MPC_RL_ControllerHybrid, HybridParams
from sim.simulator import TargetReachSimulator
from plotting.plots import plot_all, make_animations, hold_anims


# ----------------------- fallback waypoint builder (disk-uniform annulus) -----------------------

def _sample_annulus(center_xy, n, rmin, rmax, seed=0, disk_uniform=True):
    rng = np.random.default_rng(seed)
    if disk_uniform:
        u = rng.random(n)
        radii = np.sqrt(rmin * rmin + u * (rmax * rmax - rmin * rmin))
    else:
        radii = rng.uniform(rmin, rmax, size=n)
    angles = rng.uniform(0, 2 * np.pi, size=n)
    pts = np.vstack([center_xy[0] + radii * np.cos(angles),
                     center_xy[1] + radii * np.sin(angles)]).T.astype(np.float32)
    return pts

def _build_center(env, mode, center_xy):
    if mode == "fixed":
        return np.array(center_xy, dtype=np.float32)
    return env.states["cartesian"][0, :2].astype(np.float32)

def _waypoints_annulus(center, n_goals, rmin, rmax, seed, disk_uniform, return_to_center=True):
    targets = _sample_annulus(center, n_goals, rmin, rmax, seed=seed, disk_uniform=disk_uniform)
    if return_to_center:
        seq = [center]
        for t in targets:
            seq.append(t); seq.append(center)
        return np.vstack(seq).astype(np.float32), targets
    return np.vstack([center, *targets]).astype(np.float32), targets


# ---------------------------------------- env builder ----------------------------------------

def build_env(pc: PlantConfig):
    muscle = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(
        muscle=muscle, timestep=pc.timestep, damping=pc.damping,
        n_ministeps=pc.n_ministeps, integration_method=pc.integration_method
    )
    env = Environment(
        effector=arm, max_ep_duration=pc.max_ep_duration,
        action_noise=0.0, obs_noise=0.0,
        proprioception_delay=arm.dt, vision_delay=arm.dt,
        name="RandomReach(MPC_RL_hybrid)"
    )
    q0 = np.deg2rad(np.array(pc.q0_deg)); qd0 = np.array(pc.qd0)
    env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})
    return env, arm


# ---------------------------------------- helpers ----------------------------------------

def _safe_collect(logs, key):
    """Return logs.collect(key) if available, else None."""
    return logs.collect(key) if hasattr(logs, "collect") else None

def _safe_xy_and_ref(logs, traj, tvec):
    """
    Try to recover measured XY (T,2) and reference XY_ref (T,2)
    without assuming logs.cartesian() exists.
    """
    # 1) Try generic collectors
    X = _safe_collect(logs, "x")   # (T,2) expected
    Xd = _safe_collect(logs, "xd") # (T,2) optional
    if X is not None and X.ndim == 2 and X.shape[1] >= 2:
        XY_meas = X[:, :2]
    else:
        XY_meas = None

    Xref = _safe_collect(logs, "xref")  # some loggers store (T,2)
    if Xref is not None and Xref.ndim == 2 and Xref.shape[1] >= 2:
        XY_ref = Xref[:, :2]
    else:
        XY_ref = None

    # 2) If ref missing, rebuild from trajectory (if provided)
    if XY_ref is None and traj is not None and tvec is not None:
        try:
            XY_ref = np.vstack([np.asarray(traj.at(t)[0], dtype=np.float32)[:2] for t in tvec])
        except Exception:
            XY_ref = None

    return XY_meas, XY_ref


# ---------------------------------------- main ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="BC a-policy checkpoint")
    ap.add_argument("--goals", type=int, default=8)
    ap.add_argument("--radius_min", type=float, default=0.07)
    ap.add_argument("--radius_max", type=float, default=0.16)
    ap.add_argument("--center_mode", choices=["current","fixed"], default="current")
    ap.add_argument("--center_x", type=float, default=0.00)
    ap.add_argument("--center_y", type=float, default=0.55)
    ap.add_argument("--disk_uniform", action="store_true")
    ap.add_argument("--return_to_center", dest="return_to_center", action="store_true")
    ap.add_argument("--no-return_to_center", dest="return_to_center", action="store_false")
    ap.set_defaults(return_to_center=True)
    ap.add_argument("--q0_jitter_deg", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=202)
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")

    # Blending/confidence knobs
    ap.add_argument("--beta_min", type=float, default=0.0)
    ap.add_argument("--beta_max", type=float, default=1.0)
    ap.add_argument("--z_ref", type=float, default=1.6)
    ap.add_argument("--sigma_good", type=float, default=0.15)
    ap.add_argument("--print_every", type=int, default=50)

    # DAgger
    ap.add_argument("--dagger", action="store_true")
    ap.add_argument("--dagger_beta_thresh", type=float, default=0.35)
    ap.add_argument("--dagger_path", type=str, default="data/random_reach_a_ds.npz")
    ap.add_argument("--dagger_flush_every", type=int, default=1000)

    args = ap.parse_args()

    np.random.seed(args.seed)
    pc, tc, run = PlantConfig(), TrajectoryConfig(), RunConfig()
    env, arm = build_env(pc)

    # Optional q0 jitter
    if args.q0_jitter_deg > 0.0:
        joint = env.states["joint"][0].copy()
        q = joint[:2] + np.deg2rad(args.q0_jitter_deg) * (np.random.rand(2) * 2 - 1)
        qd = joint[2:]
        env.reset(options={"joint_state": np.concatenate([q, qd])[None, :], "deterministic": True})

    # Waypoints (task)
    if HAVE_TASK:
        try:
            # Try annulus API if your Task supports (rmin, rmax)
            task = RandomReachTask(n_points=args.goals, radius=(args.radius_min, args.radius_max), seed=args.seed)
            waypoints = task.build_waypoints(env, reach_T=1.2, settle_T=0.3)
            center = task.center; targets = task.targets
        except Exception:
            center = _build_center(env, args.center_mode, (args.center_x, args.center_y))
            waypoints, targets = _waypoints_annulus(center, args.goals, args.radius_min, args.radius_max,
                                                    seed=args.seed, disk_uniform=args.disk_uniform,
                                                    return_to_center=args.return_to_center)
    else:
        center = _build_center(env, args.center_mode, (args.center_x, args.center_y))
        waypoints, targets = _waypoints_annulus(center, args.goals, args.radius_min, args.radius_max,
                                                seed=args.seed, disk_uniform=args.disk_uniform,
                                                return_to_center=args.return_to_center)

    traj = MinJerkLinearTrajectory(
        waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
    )

    # Teacher (NMPC) params — MUSCLES ON; send_excitation handled safely inside the hybrid
    teacher_p = NMPCParams(N=12, dt_mpc=arm.dt, use_muscles=True)

    # RL policy
    ppi = RLPolicyParams(obs_dim=14 + 5 * arm.n_muscles, act_dim=arm.n_muscles, device=args.device)
    pi  = RLPolicy(ppi); pi.load(args.ckpt)

    # Hybrid params
    hp = HybridParams(
        beta_min=args.beta_min, beta_max=args.beta_max,
        z_ref=args.z_ref, sigma_good=args.sigma_good,
        print_every=args.print_every,
        dagger_enabled=bool(args.dagger),
        dagger_beta_thresh=args.dagger_beta_thresh,
        dagger_path=args.dagger_path,
        dagger_flush_every=args.dagger_flush_every,
        name="Hybrid(MPC+RL_a)"
    )

    ctrl = MPC_RL_ControllerHybrid(env, arm, teacher_p, pi, hp)

    steps = int(pc.max_ep_duration / arm.dt)
    logs  = TargetReachSimulator(env, arm, ctrl, traj, steps).run()

    # -------- quick metrics (no hard dependency on logs.cartesian) --------
    k, tvec = logs.time(arm.dt)
    X_meas, X_ref = _safe_xy_and_ref(logs, traj, tvec)

    if X_meas is not None and X_ref is not None and len(X_meas) == len(X_ref):
        e = X_ref - X_meas
        rmse = float(np.sqrt(np.mean(e ** 2)))
        print(f"[Hybrid] Position RMSE = {rmse * 1000:.2f} mm over {len(X_meas)} steps.")
    else:
        print("[Hybrid] Skipping RMSE: could not recover measured/ref XY with current logger.")

    betas = _safe_collect(logs, "beta")
    if betas is not None and betas.size > 0:
        betas = np.clip(betas, 0.0, 1.0)
        rl_frac  = float(np.mean(1.0 - betas))
        mpc_frac = 1.0 - rl_frac
        print(f"[Hybrid] Usage: RL ~ {100 * rl_frac:.1f}%  |  MPC ~ {100 * mpc_frac:.1f}%")

    # -------- plots / animations --------
    plot_all(logs, tvec, center=center, targets=targets)
    if args.animate:
        anims = make_animations(logs, tvec, env, playback=run.playback,
                                downsample=run.downsample_anim, center=center, targets=targets)
        hold_anims(anims)
    print("main_random_reach_MPC_RL_controller_hybrid: complete.")
    plt.show()


if __name__ == "__main__":
    main()
