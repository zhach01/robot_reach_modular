#!/usr/bin/env python3
import os
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
from controller.anfis_controller import ANFISController, ANFISParams
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
        name="RandomReachEnv",
    )
    q0 = np.deg2rad(np.array(pc.q0_deg))
    qd0 = np.array(pc.qd0)
    env.reset(
        options={"joint_state": np.concatenate([q0, qd0])[None, :],
                 "deterministic": True}
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

    task = RandomReachTask(n_points=1, radius=0.10, seed=0)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(
        waypoints,
        MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale),
    )

    rules_path = "ANFIS/saved_model/anfis_rules.npz"

    p = ANFISParams(
        n_mf=5,
        mf_type="gaussian",

        # EVAL mode: no online learning
        online_adapt=False,
        lr_premise=0.0,

        lse_reg=1e-4,
        adapt_every=5,
        min_fit_samples=25,
        buffer_size=300,

        enable_gravity_comp=toggles.enable_gravity_comp,
        enable_coriolis_comp=toggles.enable_velocity_comp,

        eps=num.eps,
        lam_os_max=num.lam_os_max,
        sigma_thresh=num.sigma_thresh,
        gate_pow=num.gate_pow,

        bisect_iters=ifc.bisect_iters,
        rules_path=rules_path,
    )

    ctrl = ANFISController(env, arm, p)
    ctrl.reset(q0)
    ctrl.init_pd(gains.Kp_q, gains.Kd_q, bias=0.0)

    if os.path.exists(rules_path):
        ok = ctrl.load_rules(rules_path)
        print(f"[ANFIS] Loaded rules for eval: {ok}")
    else:
        print(f"[ANFIS] No rules file at {rules_path}, running with PD init only.")

    steps = int(pc.max_ep_duration / arm.dt)
    sim = TargetReachSimulator(env, arm, ctrl, traj, steps)
    logs = sim.run()

    # Optional: save rules again (no change in eval mode)
    ctrl.save_rules(rules_path)

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

    plt.show()


if __name__ == "__main__":
    main()

