#!/usr/bin/env python3
import os
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
)
from tasks.random_reach_numpy import Task as RandomReachTask
from trajectory.minjerk_numpy import MinJerkLinearTrajectory, MinJerkParams
from controller.anfis_controller import ANFISController, ANFISParams
from sim.simulator_numpy import TargetReachSimulator


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
    gains = ControlGains()
    num = Numerics()
    ifc = InternalForceConfig()
    tc = TrajectoryConfig()

    env, arm, q0 = build_env(pc)

    rules_path = "ANFIS/saved_model/anfis_rules.npz"
    os.makedirs(os.path.dirname(rules_path), exist_ok=True)

    # ANFIS training configuration
    p = ANFISParams(
        n_mf=5,
        mf_type="gaussian",

        # TRAINING mode
        online_adapt=True,
        teacher_mode="id_residual",

        # Consequent LSE
        lse_reg=1e-4,
        adapt_every=5,
        min_fit_samples=30,
        buffer_size=300,

        # Premise learning
        lr_premise=1e-3,
        premise_batch_size=32,
        premise_update_every=10,

        # Freeze when stable
        freeze_after_steps=0,
        freeze_err_thresh=0.005,
        freeze_patience=80,

        # Autosave inside controller (if implemented)
        autosave_every=200,
        save_on_freeze=True,

        # Dynamics compensation
        enable_gravity_comp=toggles.enable_gravity_comp,
        enable_coriolis_comp=toggles.enable_velocity_comp,

        # Guards / numerics
        eps=num.eps,
        lam_os_max=num.lam_os_max,
        sigma_thresh=num.sigma_thresh,
        gate_pow=num.gate_pow,

        # Muscle inversion
        bisect_iters=ifc.bisect_iters,
        rules_path=rules_path,
    )

    ctrl = ANFISController(env, arm, p)
    ctrl.reset(q0)
    ctrl.init_pd(gains.Kp_q, gains.Kd_q, bias=0.0)

    # Optional warm start from existing rules
    if os.path.exists(rules_path):
        if ctrl.load_rules(rules_path):
            print(f"[ANFIS] warm start from {rules_path}")
        else:
            print(f"[ANFIS] could not load {rules_path}, starting from PD initialisation")

    n_episodes = 50
    steps = int(pc.max_ep_duration / arm.dt)

    # Snapshot of rules to monitor between-episode changes
    rb_prev = ctrl.get_rule_base(0).copy()

    for ep in range(n_episodes):
        # Reset env to nominal initial joint state each episode
        qd0 = np.zeros_like(q0)
        env.reset(
            options={
                "joint_state": np.concatenate([q0, qd0])[None, :],
                "deterministic": True,
            }
        )
        ctrl.reset(q0)  # clear ANFIS buffers / episode stats

        # New random target per episode
        task = RandomReachTask(n_points=1, radius=0.10, seed=ep)
        waypoints = task.build_waypoints(env)
        traj = MinJerkLinearTrajectory(
            waypoints,
            MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale),
        )

        # Simulator will call ctrl.compute(...) each step;
        # ANFISController should handle online adaptation internally
        sim = TargetReachSimulator(env, arm, ctrl, traj, steps)
        _ = sim.run()

        # Check how much the rule base changed vs previous episode
        rb_now = ctrl.get_rule_base(0)
        delta = np.max(np.abs(rb_now - rb_prev))
        print(f"[ANFIS] episode {ep+1}/{n_episodes}  max |Δrule| vs prev = {delta:.6e}")
        rb_prev = rb_now.copy()

        # Save current rules to disk
        ctrl.save_rules(rules_path)
        print(f"[ANFIS] episode {ep+1}/{n_episodes} -> saved rules")

        # Optional: visual inspection of learned surfaces every 10 episodes
        if (ep + 1) % 10 == 0:
            try:
                ctrl.plot_surface(joint_idx=0, title=f"Joint 1 surface after ep {ep+1}")
                ctrl.plot_surface(joint_idx=1, title=f"Joint 2 surface after ep {ep+1}")
            except Exception as e:
                print(f"[ANFIS] plot_surface failed (ignored): {e}")

    print("[ANFIS] training completed.")


if __name__ == "__main__":
    main()

