#!/usr/bin/env python3
# scripts/ANFIS/main_random_reach.py  (canonical ANFIS = bake-off winner)

import os
import numpy as np
import matplotlib.pyplot as plt

from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26

from config import (
    PlantConfig,
    ControlToggles,
    Numerics,
    InternalForceConfig,
    TrajectoryConfig,
    RunConfig,
)

from tasks.numpy.random_reach import Task as RandomReachTask
from trajectory.numpy.minjerk import MinJerkLinearTrajectory, MinJerkParams

from controller.numpy.anfis_controller import ANFISController, ANFISParams

from sim.numpy.simulator import TargetReachSimulator
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
        name="RandomReachEnv",
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

    # trajectory
    task = RandomReachTask(n_points=1, radius=0.10, seed=0)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(
        waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
    )

    # rules persistence
    rules_path = "ANFIS/saved_model/anfis_rules.npz"
    os.makedirs(os.path.dirname(rules_path), exist_ok=True)

    p = ANFISParams(
        # learning ON for demo (set False for deployment)
        enable_learning=True,
        adapt_every=2,
        warmup_steps=50,

        # teacher
        teacher_Kp=300.0,
        teacher_Kd=60.0,
        anchor_rho=0.3,

        # feedforward toggles from your config
        enable_gravity_comp=toggles.enable_gravity_comp,
        enable_coriolis_comp=toggles.enable_velocity_comp,

        # numerics from your config
        eps=num.eps,
        lam_os_max=num.lam_os_max,
        sigma_thresh=num.sigma_thresh,
        gate_pow=num.gate_pow,

        # muscle inversion
        bisect_iters=ifc.bisect_iters,

        # persistence
        rules_path=rules_path,
        autosave_every=0,
    )

    ctrl = ANFISController(env, arm, p)

    loaded = ctrl.load_rules(rules_path)
    print(f"[ANFIS] load_rules={loaded}  path='{rules_path}'")

    # IMPORTANT: simulator expects ctrl.qref
    ctrl.reset(q0)

    # run
    steps = int(pc.max_ep_duration / arm.dt)
    sim = TargetReachSimulator(env, arm, ctrl, traj, steps)
    logs = sim.run()

    # save at end
    try:
        ctrl.save_rules(rules_path)
        print(f"[ANFIS] saved -> '{rules_path}'")
    except Exception as ex:
        print("[ANFIS] WARNING: save failed:", ex)

    # plots
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

    print("Random reach ANFIS demo complete.")
    plt.show()


if __name__ == "__main__":
    main()

