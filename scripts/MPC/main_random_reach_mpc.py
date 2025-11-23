#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from model_lib.environment_numpy import Environment
from model_lib.muscles_numpy import RigidTendonHillMuscle
from model_lib.effector_numpy import RigidTendonArm26

from config import (
    PlantConfig,
    ControlToggles,
    ControlGains,     # kept for parity, not used by MPC
    Numerics,
    InternalForceConfig,
    TrajectoryConfig,
    RunConfig,
)

from tasks.random_reach import Task as RandomReachTask
from trajectory.minjerk import MinJerkLinearTrajectory, MinJerkParams

# ⬇️ MPC controller
from controller.nmpc_task import NonlinearMPCController, NMPCParams

# Your plotting/sim infra
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
        name="RandomReachEnv_MPC",
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
    # --- configs (unchanged) ---
    pc   = PlantConfig()
    tgl  = ControlToggles()
    _gn  = ControlGains()         # not used by MPC, just kept for parity
    num  = Numerics()
    ifc  = InternalForceConfig()
    tc   = TrajectoryConfig()
    run  = RunConfig()

    # --- env & task ---
    env, arm, q0 = build_env(pc)
    task = RandomReachTask(n_points=1, radius=0.10, seed=0)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(
        waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
    )

    # --- MPC params (good baseline for tight tracking) ---
    p = NMPCParams(N=12, dt_mpc=arm.dt, use_muscles=True)  # no muscles

    ctrl = NonlinearMPCController(env, arm, p)

    # --- simulate (same simulator, so logs/plots remain compatible) ---
    steps = int(pc.max_ep_duration / arm.dt)
    sim = TargetReachSimulator(env, arm, ctrl, traj, steps)
    logs = sim.run()

    # --- plots/animations (unchanged) ---
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

    print("Random reach (MPC) demo complete.")
    plt.show()


if __name__ == "__main__":
    main()

