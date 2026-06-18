#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

from model_lib.environment_numpy import Environment
from model_lib.muscles_numpy import RigidTendonHillMuscle
from model_lib.effector_numpy import RigidTendonArm26

from config import (
    PlantConfig,
    ControlToggles,
    ControlGains,
    Numerics,
    InternalForceConfig,
    TrajectoryConfig,
    RunConfig,
)

from tasks.random_reach_numpy import Task as RandomReachTask
from trajectory.minjerk_numpy import MinJerkLinearTrajectory, MinJerkParams

# ✅ Synergy controller
from controller.synergy_controller import SynergyController, SynergyParams

from sim.simulator_numpy import TargetReachSimulator
from plotting.plots import plot_all, make_animations, hold_anims


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
        name="RandomReachEnv(Synergy)",
    )

    q0 = np.deg2rad(np.array(pc.q0_deg))
    qd0 = np.array(pc.qd0)

    env.reset(
        options={
            "joint_state": np.concatenate([q0, qd0])[None, :],
            "deterministic": True,
        }
    )
    return env, arm, q0


def _scalar(x, default=1.0) -> float:
    """Accept scalar/array-like and return float."""
    try:
        v = np.array(x, dtype=float).ravel()
        if v.size >= 1:
            return float(v[0])
    except Exception:
        pass
    return float(default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius", type=float, default=0.10)
    parser.add_argument("--n_points", type=int, default=1)
    parser.add_argument("--load_synergy", type=str, default="")
    parser.add_argument("--save_synergy", type=str, default="")
    parser.add_argument("--online_B_adapt", action="store_true")
    args = parser.parse_args()

    pc = PlantConfig()
    toggles = ControlToggles()
    gains = ControlGains()
    num = Numerics()
    ifc = InternalForceConfig()
    tc = TrajectoryConfig()
    run = RunConfig()

    env, arm, q0 = build_env(pc)

    # --- Task / trajectory ---
    task = RandomReachTask(n_points=args.n_points, radius=args.radius, seed=args.seed)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(
        waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
    )

    # --- Synergy controller params (TRACKING-SAFE) ---
    p = SynergyParams(
        # task-level OSC/PD gains
        Kp_task=800.0,
        Kv_task=60.0,
        Kff_x=_scalar(gains.Kff_x, default=1.0),

        # toggles
        enable_inertia_comp=toggles.enable_inertia_comp,
        enable_gravity_comp=toggles.enable_gravity_comp,
        enable_velocity_comp=toggles.enable_velocity_comp,

        # guards
        eps=num.eps,
        lam_os_max=getattr(num, "lam_os_max", 1e6),
        sigma_thresh=num.sigma_thresh,
        gate_pow=num.gate_pow,

        # muscle inversion
        bisect_iters=ifc.bisect_iters,

        # -------- CRITICAL FIXES FOR TRACKING --------
        # allow synergies to produce large activations
        c_max=5.0,

        # disable prior at first (avoid “frozen” wrong c)
        use_prior=False,
        prior_weight=0.0,

        # keep synergy structure but ensure residual correction can track
        synergy_strength=0.8,

        # optional online learning of B (only matters if use_prior=True later)
        online_B_adapt=bool(args.online_B_adapt),
        rls_lambda=0.995,
        rls_delta=100.0,
    )

    ctrl = SynergyController(env, arm, p)
    ctrl.reset(q0)

    # Optional: load previously saved synergy model (W, B)
    if args.load_synergy:
        try:
            ctrl.load(args.load_synergy)
            print(f"[synergy] loaded W,B from: {args.load_synergy}")
        except Exception as e:
            print(f"[synergy] WARNING: failed to load '{args.load_synergy}': {e}")

    steps = int(pc.max_ep_duration / arm.dt)
    sim = TargetReachSimulator(env, arm, ctrl, traj, steps)
    logs = sim.run()

    # Optional: save W,B after the run (useful if online_B_adapt=True)
    if args.save_synergy:
        try:
            ctrl.save(args.save_synergy)
            print(f"[synergy] saved W,B to: {args.save_synergy}")
        except Exception as e:
            print(f"[synergy] WARNING: failed to save '{args.save_synergy}': {e}")

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

    print("Random reach (Synergy Controller) complete.")
    plt.show()


if __name__ == "__main__":
    main()

