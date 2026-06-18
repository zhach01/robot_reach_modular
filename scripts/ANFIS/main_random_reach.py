#!/usr/bin/env python3
import os
import numpy as np

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

from controller.anfis_controller import ANFISController, ANFISParams

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

    task = RandomReachTask(n_points=1, radius=0.10, seed=0)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(
        waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
    )

    # ---------------- ANFIS PARAMS ----------------
    rules_path = "ANFIS/saved_model/anfis_rules.npz"
    os.makedirs(os.path.dirname(rules_path), exist_ok=True)

    p = ANFISParams(
        online_adapt=True,

        # teacher inside controller (stable)
        teacher_mode="pd",
        Kp_teacher=20.0,
        Kd_teacher=4.0,
        Ki_teacher=0.0,

        # online LSE
        lse_reg=1e-2,
        adapt_every=50,
        min_fit_samples=150,
        buffer_size=400,

        # replace teacher gradually
        alpha_final=1.0,
        alpha_warmup_steps=1500,

        # persistence
        rules_path=rules_path,
        autosave_every=200,

        # keep your toggles for comp if you want
        enable_gravity_comp=toggles.enable_gravity_comp,
        enable_coriolis_comp=toggles.enable_velocity_comp,

        # guards
        eps=num.eps,
        lam_os_max=num.lam_os_max,
        sigma_thresh=num.sigma_thresh,
        gate_pow=num.gate_pow,

        bisect_iters=ifc.bisect_iters,
    )

    ctrl = ANFISController(env, arm, p)

    # ---------------- LOAD OR WARM-START ----------------
    loaded = ctrl.load_rules(p.rules_path)
    print(f"[ANFIS] load_rules={loaded}  path='{p.rules_path}'")

    if not loaded:
        # Warm-start: ANFIS behaves like PD immediately
        ctrl.init_pd(Kp_q=p.Kp_teacher, Kd_q=p.Kd_teacher, bias=0.0)
        print("[ANFIS] warm-started ANFIS consequents to PD")

    # IMPORTANT: ensures qref exists + clears buffers (does NOT wipe loaded rules)
    ctrl.reset(q0)

    # ---------------- RUN ----------------
    steps = int(pc.max_ep_duration / arm.dt)
    sim = TargetReachSimulator(env, arm, ctrl, traj, steps)
    logs = sim.run()

    # ---------------- SAVE AT END ----------------
    try:
        ctrl.save_rules(p.rules_path)
        print(f"[ANFIS] saved rules to '{p.rules_path}'")
    except Exception as ex:
        print("[ANFIS] WARNING: could not save rules:", ex)

    # ---------------- PLOTS ----------------
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

    print("Random reach demo complete.")
    plt.show()


if __name__ == "__main__":
    main()

