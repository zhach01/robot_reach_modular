#!/usr/bin/env python3
import numpy as np
from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26
from config import (
    PlantConfig,
    ControlToggles,
    ControlGains,
    Numerics,
    InternalForceConfig,
    TrajectoryConfig,
    RunConfig,
)
from tasks.numpy.center_out import Task as CenterOutTask
from trajectory.numpy.minjerk import MinJerkLinearTrajectory, MinJerkParams
from controller.numpy.pd_if_controller import PDIFController, PDIFParams
from sim.numpy.simulator import TargetReachSimulator
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
        name="CenterOutEnv",
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
    gains = ControlGains()
    num = Numerics()
    ifc = InternalForceConfig()
    tc = TrajectoryConfig()
    run = RunConfig()
    env, arm, q0 = build_env(pc)

    task = CenterOutTask(n_targets=4, radius=0.10)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(
        waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
    )

    # Optimized PD+IF gains (same canonical controller as main_random_reach).
    # The critically-damped task gain removes the large center-out overshoot the
    # legacy controller had on the far reaches.
    p = PDIFParams(
        Kp_task=np.array([2400.0, 2400.0], dtype=float),
        damping_ratio=0.7,
        Kff=1.0,
        use_critical_damping=True,
        enable_inertia_comp=bool(getattr(toggles, "enable_inertia_comp", True)),
        enable_gravity_comp=bool(getattr(toggles, "enable_gravity_comp", True)),
        enable_coriolis_comp=bool(getattr(toggles, "enable_velocity_comp", True)),
        enable_nullspace=True,
        Kp_null=20.0,
        Kd_null=5.0,
        eps=float(getattr(num, "eps", 1e-6)),
        lam_os_max=float(getattr(num, "lam_os_max", 200.0)),
        sigma_thresh=float(getattr(num, "sigma_thresh", 1e-4)),
        gate_pow=float(getattr(num, "gate_pow", 2.0)),
        bisect_iters=int(getattr(ifc, "bisect_iters", 12)),
        enable_internal_force=False,
        cocon_level=0.0,
    )
    ctrl = PDIFController(env, arm, p)
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
        hold_anims(anims)  # keep animations alive until plt.show()

    print("Center-out demo complete.")
    plt.show()


if __name__ == "__main__":
    main()
