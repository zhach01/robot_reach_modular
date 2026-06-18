#!/usr/bin/env python3
import numpy as np
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
from controller.sliding_mode import SlidingModeController, SlidingModeParams
from sim.simulator import TargetReachSimulator
from plotting.plots import plot_all, make_animations, hold_anims
import matplotlib.pyplot as plt


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
        name="RandomReachEnv(SMC)",
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
    """Accept scalar or array-like, return float."""
    try:
        v = np.array(x, dtype=float).ravel()
        if v.size >= 1:
            return float(v[0])
    except Exception:
        pass
    return float(default)


def main():
    pc = PlantConfig()
    toggles = ControlToggles()
    gains = ControlGains()
    num = Numerics()
    ifc = InternalForceConfig()
    tc = TrajectoryConfig()
    run = RunConfig()

    env, arm, q0 = build_env(pc)

    task = RandomReachTask(n_points=1, radius=0.10, seed=0)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(
        waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
    )

    # ------------------------------------------------------------------
    # Sliding-Mode gains (task-space = x,y)
    # ------------------------------------------------------------------
    lambda_surf = np.array([26.0, 26.0])     # surface slope (1/s)

    # NOTE: with switch_in_accel=True, K_switch behaves like an accel-gain.
    # If you see spikes, try 1.0–2.0 first.
    K_switch    = np.array([4.0, 4.0])

    phi         = np.array([0.004, 0.004])   # boundary layer thickness

    # Feedforward on desired xdd (scalar in new controller)
    Kff_x = _scalar(gains.Kff_x, default=1.0)

    # ------------------------------------------------------------------
    # Build params (MATCHES new SlidingModeParams in controller)
    # ------------------------------------------------------------------
    p = SlidingModeParams(
        # core SMC
        lambda_surf=lambda_surf,
        K_switch=K_switch,
        phi=phi,

        # feedforward & comp flags
        Kff_x=Kff_x,
        enable_gravity_comp=toggles.enable_gravity_comp,
        enable_velocity_comp=toggles.enable_velocity_comp,
        enable_inertia_comp=toggles.enable_inertia_comp,

        # optional nullspace/bias (keep off by default)
        lambda_ns=0.0,
        k_manip=0.0,

        # guards
        eps=num.eps,
        lam_os_max=num.lam_os_max,
        sigma_thresh=num.sigma_thresh,
        gate_pow=num.gate_pow,

        # muscle inversion
        bisect_iters=ifc.bisect_iters,

        # NEW safety knobs (important for singularity behavior)
        switch_in_accel=True,          # should be True for the new robust version
        use_condJ_for_blend=True,      # blend using cond(J) (earlier + correct)
        cond_low=20.0,
        cond_high=80.0,
        elbow_min_rad=np.deg2rad(2.0),
    )

    ctrl = SlidingModeController(env, arm, p)
    ctrl.reset(q0)

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

    print("Random reach (Sliding-Mode) complete.")
    plt.show()


if __name__ == "__main__":
    main()

