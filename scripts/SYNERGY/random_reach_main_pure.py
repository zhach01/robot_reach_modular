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

from tasks.random_reach import Task as RandomReachTask
from trajectory.minjerk import MinJerkLinearTrajectory, MinJerkParams

from controller.synergy_controller_pure import SynergyPureController, SynergyPureParams

from sim.simulator import TargetReachSimulator
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
        name="RandomReachEnv(SynergyPure)",
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
    v = np.array(x, dtype=float).ravel()
    return float(v[0]) if v.size else float(default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=float, default=0.10)
    ap.add_argument("--n_points", type=int, default=1)
    ap.add_argument("--W_model", type=str, default="synergy/saved_model/W_model.npz",
                    help="NPZ with key 'W'")
    args = ap.parse_args()

    pc = PlantConfig()
    toggles = ControlToggles()
    gains = ControlGains()
    num = Numerics()
    ifc = InternalForceConfig()
    tc = TrajectoryConfig()
    run = RunConfig()

    env, arm, q0 = build_env(pc)

    task = RandomReachTask(n_points=args.n_points, radius=args.radius, seed=args.seed)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(
        waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
    )

    p = SynergyPureParams(
        Kp_task=800.0,
        Kv_task=60.0,
        Kff_x=_scalar(gains.Kff_x, 1.0),

        enable_gravity_comp=toggles.enable_gravity_comp,

        eps=num.eps,
        lam_os_max=getattr(num, "lam_os_max", 1e6),
        sigma_thresh=num.sigma_thresh,
        gate_pow=num.gate_pow,

        bisect_iters=ifc.bisect_iters,

        # IMPORTANT: pure synergy needs enough scale
        c_max=6.0,
        nnls_iters=80,
    )

    ctrl = SynergyPureController(env, arm, p)
    ctrl.reset(q0)

    # Load learned W
    try:
        z = np.load(args.W_model, allow_pickle=True)
        ctrl.set_W(z["W"])
        print(f"[pure synergy] loaded W from: {args.W_model}  (shape={z['W'].shape})")
    except Exception as e:
        print(f"[pure synergy] WARNING: could not load W model '{args.W_model}': {e}")
        print("[pure synergy] Using default random W (likely poor).")

    steps = int(pc.max_ep_duration / arm.dt)
    sim = TargetReachSimulator(env, arm, ctrl, traj, steps)
    logs = sim.run()

    k, tvec = logs.time(arm.dt)
    plot_all(logs, tvec, center=task.center, targets=task.targets)

    if run.animate:
        anims = make_animations(
            logs, tvec, env,
            playback=run.playback,
            downsample=run.downsample_anim,
            center=task.center,
            targets=task.targets,
        )
        hold_anims(anims)

    print("Random reach (Pure Synergy) complete.")
    plt.show()


if __name__ == "__main__":
    main()

