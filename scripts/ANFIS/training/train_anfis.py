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

from tasks.random_reach import Task as RandomReachTask
from trajectory.minjerk import MinJerkLinearTrajectory, MinJerkParams
from controller.anfis_controller import ANFISController, ANFISParams
from sim.simulator import TargetReachSimulator


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


def _reset_to_random_start(env, pc: PlantConfig, rng: np.random.Generator, q0_nom: np.ndarray):
    """
    Small randomization of initial joint state to improve generalization.
    Keeps it safe: small offsets + zero velocities.
    """
    q0 = q0_nom.copy()
    dq = np.deg2rad(rng.uniform(-5.0, 5.0, size=q0.shape))  # ~ +/- 5 degrees
    q0 = q0 + dq
    qd0 = np.zeros_like(q0)
    env.reset(
        options={
            "joint_state": np.concatenate([q0, qd0])[None, :],
            "deterministic": True,
        }
    )
    return q0


def _run_episode(env, arm, ctrl, pc, tc, seed: int, radius: float):
    task = RandomReachTask(n_points=1, radius=radius, seed=seed)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(
        waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
    )
    steps = int(pc.max_ep_duration / arm.dt)
    sim = TargetReachSimulator(env, arm, ctrl, traj, steps)
    _ = sim.run()


def main():
    pc = PlantConfig()
    toggles = ControlToggles()
    gains = ControlGains()
    num = Numerics()
    ifc = InternalForceConfig()
    tc = TrajectoryConfig()

    env, arm, q0_nom = build_env(pc)

    rules_path = "ANFIS/saved_model/anfis_rules.npz"
    os.makedirs(os.path.dirname(rules_path), exist_ok=True)

    # -------------------------
    # GENERALIZATION TRAINING SETTINGS
    # -------------------------
    n_episodes = 60
    targets_per_episode = 6

    replay_every = 3
    replay_count = 3

    radius_min = 0.06
    radius_max = 0.12

    rng = np.random.default_rng(0)

    # -------------------------
    # ✅ ANFIS params with PD teacher
    # -------------------------
    p = ANFISParams(
        n_mf=5,
        mf_type="gaussian",

        # TRAINING MODE: ON
        online_adapt=True,

        # ✅ teacher = PD
        teacher_mode="pd",
        Kp_teacher=20.0,
        Kd_teacher=4.0,
        Ki_teacher=0.0,

        # stable online learning
        lse_reg=1e-2,
        adapt_every=10,
        min_fit_samples=80,
        buffer_size=400,

        # anti-forgetting (optional)
        freeze_after_steps=0,
        freeze_err_thresh=0.0,     # keep OFF during training
        freeze_patience=80,
        err_ema_beta=0.98,

        autosave_every=0,          # we save explicitly per episode
        save_on_freeze=True,

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

    # Warm start (recommended): stable baseline before learning kicks in
    ctrl.init_pd(gains.Kp_q, gains.Kd_q, bias=0.0)

    # Load existing rules if present (continue training)
    if os.path.exists(rules_path):
        ok = ctrl.load_rules(rules_path)
        print(f"[ANFIS] Loaded existing rules: {ok}")

    # Reset controller state buffers (does not wipe loaded rules)
    ctrl.reset(q0_nom)

    past_seeds = []

    for ep in range(n_episodes):
        # -------------------------
        # schedule: reduce plasticity later (stabilize rules)
        # -------------------------
        frac = (ep + 1) / float(n_episodes)
        if frac < 0.5:
            ctrl.p.adapt_every = 10
            ctrl.p.lse_reg = 1e-2
            ctrl.p.buffer_size = 400
            ctrl.p.min_fit_samples = 80
        else:
            ctrl.p.adapt_every = 30
            ctrl.p.lse_reg = 5e-2
            ctrl.p.buffer_size = 600
            ctrl.p.min_fit_samples = 150

        # -------------------------
        # train on multiple targets this episode
        # -------------------------
        for k in range(targets_per_episode):
            seed = ep * 1000 + k
            radius = float(rng.uniform(radius_min, radius_max))

            q0 = _reset_to_random_start(env, pc, rng, q0_nom)
            ctrl.reset(q0)

            _run_episode(env, arm, ctrl, pc, tc, seed=seed, radius=radius)
            past_seeds.append((seed, radius))
            if len(past_seeds) > 300:
                past_seeds.pop(0)

        # -------------------------
        # replay/consolidation
        # -------------------------
        if (ep + 1) % replay_every == 0 and len(past_seeds) > 10:
            for _ in range(replay_count):
                old_seed, old_radius = past_seeds[rng.integers(0, len(past_seeds))]
                q0 = _reset_to_random_start(env, pc, rng, q0_nom)
                ctrl.reset(q0)
                _run_episode(env, arm, ctrl, pc, tc, seed=int(old_seed), radius=float(old_radius))

        # save after each episode
        ctrl.save_rules(rules_path)
        print(f"[ANFIS] episode {ep+1}/{n_episodes} saved -> {rules_path}")

    # -------------------------
    # FINAL: freeze and save
    # -------------------------
    ctrl.p.online_adapt = False
    ctrl.save_rules(rules_path)
    print("[ANFIS] training done. (frozen + saved)")


if __name__ == "__main__":
    main()

