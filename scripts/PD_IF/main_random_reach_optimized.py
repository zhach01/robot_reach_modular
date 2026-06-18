#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from model_lib.environment_numpy import Environment
from model_lib.muscles_numpy import RigidTendonHillMuscle
from model_lib.effector_numpy import RigidTendonArm26

from config import (
    PlantConfig,
    ControlToggles,
    Numerics,
    InternalForceConfig,
    TrajectoryConfig,
    RunConfig,
)

from tasks.random_reach import Task as RandomReachTask
from trajectory.minjerk import MinJerkLinearTrajectory, MinJerkParams
from sim.simulator import TargetReachSimulator
from plotting.plots import plot_all, make_animations, hold_anims

# Optimized PD+IF controller
try:
    from controller.pd_if_optimized import OptimizedPDIFController, OptimizedPDIFParams
except ImportError:
    from pd_if_optimized import OptimizedPDIFController, OptimizedPDIFParams


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
        name="RandomReachEnv(OptimizedPDIF)",
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


def main():
    pc = PlantConfig()
    toggles = ControlToggles()
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

    # ============================================================
    # FORCE the exact "aggressive" gains from your optimization report
    # ============================================================
    p = OptimizedPDIFParams(
        # --- optimized gains (from your table) ---
        Kp_task=np.array([1600.0, 1600.0], dtype=float),
        damping_ratio=0.7,
        Kff=1.0,
        use_critical_damping=True,

        # --- dynamics compensation toggles ---
        enable_inertia_comp=bool(getattr(toggles, "enable_inertia_comp", True)),
        enable_gravity_comp=bool(getattr(toggles, "enable_gravity_comp", True)),
        enable_coriolis_comp=bool(getattr(toggles, "enable_velocity_comp", True)),

        # --- nullspace (keep mild & safe) ---
        enable_nullspace=True,
        Kp_null=20.0,
        Kd_null=5.0,

        # --- guards / numerics (keep from config) ---
        eps=float(getattr(num, "eps", 1e-6)),
        lam_os_max=float(getattr(num, "lam_os_max", 200.0)),
        sigma_thresh=float(getattr(num, "sigma_thresh", 1e-4)),
        gate_pow=float(getattr(num, "gate_pow", 2.0)),

        # --- muscle inversion ---
        bisect_iters=int(getattr(ifc, "bisect_iters", 12)),

        # --- internal force: OFF (as in your optimization conclusion) ---
        enable_internal_force=False,
        cocon_level=0.0,
    )

    ctrl = OptimizedPDIFController(env, arm, p)
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

    print("Random reach demo (Optimized PD+IF, aggressive tuning) complete.")
    plt.show()


if __name__ == "__main__":
    main()

