#!/usr/bin/env python3
# scripts/MOTORNET/main_goal_reach_motornet_torch.py
# Episodic goal-reach demo for the canonical torch MotorNet (MotorNetController).
#
# MotorNet is an EPISODIC goal-reaching policy (not a continuous tracker): it
# reaches a goal over `trial_duration` from a fixed start. This demo runs one
# trial per center-out target and reports the final endpoint error.
#
# Train the policy first (checkpoint is gitignored):
#   python -m scripts.MOTORNET.train_motornet_torch --epochs 150 --device cpu \
#       --out motornet/saved_model/mn_controller.pt

from __future__ import annotations

import os
import math
import numpy as np
import torch

from model_lib.torch.environment import Environment
from model_lib.torch.muscles import RigidTendonHillMuscle
from model_lib.torch.effector import RigidTendonArm26

from config import PlantConfig
from controller.torch.motornet_controller import MotorNetController

import matplotlib.pyplot as plt


CKPT = "motornet/saved_model/mn_controller.pt"


def build_env(pc, device, dtype, trial_duration):
    # MotorNet is brittle to its TRAINING plant: the trainer hardcodes
    # damping=0.0, n_ministeps=1, euler (NOT PlantConfig). The policy is only
    # valid on that exact plant, so the demo must reproduce it.
    muscle = RigidTendonHillMuscle(min_activation=0.02, device=device, dtype=dtype)
    arm = RigidTendonArm26(
        muscle=muscle, timestep=0.01, damping=0.0,
        n_ministeps=1, integration_method="euler",
        device=device, dtype=dtype,
    )
    env = Environment(
        effector=arm, max_ep_duration=trial_duration + 0.1, action_noise=0.0, obs_noise=0.0,
        action_frame_stacking=1, proprioception_delay=arm.dt, vision_delay=arm.dt,
        name="GoalReachMotorNetEnv",
    )
    return env, arm


def reset_to_q0(env, q0, device, dtype):
    # MotorNet is brittle to its TRAINING start: reset to the checkpoint's q0,
    # not PlantConfig.q0_deg (a different start makes the policy fail).
    env.reset(options={"joint_state": torch.cat([q0, torch.zeros(2, dtype=dtype, device=device)]).unsqueeze(0),
                       "deterministic": True})
    return q0


def main():
    print("[GoalReach MotorNet Torch] demo starting ...")
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.get_default_dtype()
    pc = PlantConfig()

    if not os.path.exists(CKPT):
        print(f"[motornet] no checkpoint at '{CKPT}'. Train first:")
        print("  python -m scripts.MOTORNET.train_motornet_torch --epochs 150 --device cpu "
              f"--out {CKPT}")
        return

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    params = ckpt["params"]; params.device = str(device)

    env, arm = build_env(pc, device, dtype, float(params.trial_duration))
    q0 = torch.as_tensor(ckpt["q0"], dtype=dtype, device=device)   # trained start
    reset_to_q0(env, q0, device, dtype)
    ctrl = MotorNetController(env, arm, params)
    ctrl.policy.load_state_dict(ckpt["policy_state_dict"]); ctrl.is_trained = True
    print(f"[motornet] loaded canonical policy: {CKPT}  (trained q0_deg={np.rad2deg(ckpt['q0']).round(1)})")

    center = env.states["fingertip"][0, :2].clone()
    n_targets, radius = 8, 0.08
    T = float(params.trial_duration); dt = float(arm.dt); n_steps = int(T / dt)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(float(center[0]), float(center[1]), "ko", label="start")
    final_errs = []
    for i in range(n_targets):
        ang = 2.0 * math.pi * i / n_targets
        goal = center + radius * torch.tensor([math.cos(ang), math.sin(ang)], dtype=dtype, device=device)
        reset_to_q0(env, q0, device, dtype)
        ctrl.reset(q0); ctrl.set_goal(goal)
        path = []
        for _ in range(n_steps):
            d = ctrl.compute(goal, torch.zeros(2, device=device, dtype=dtype), torch.zeros(2, device=device, dtype=dtype))
            env.step(d["act"], deterministic=True)
            path.append(env.states["fingertip"][0, :2].detach().cpu().numpy())
        path = np.asarray(path)
        ferr = float(np.linalg.norm(path[-1] - goal.detach().cpu().numpy())) * 1000
        final_errs.append(ferr)
        ax.plot(path[:, 0], path[:, 1], "-", lw=1.0)
        ax.plot(float(goal[0]), float(goal[1]), "rx")
        print(f"  target {i}: final endpoint error = {ferr:.2f} mm")

    print(f"[GoalReach MotorNet Torch] mean final error = {np.mean(final_errs):.2f} mm "
          f"(max {np.max(final_errs):.2f} mm) over {n_targets} reaches")
    ax.set_aspect("equal"); ax.set_title("MotorNet episodic goal reaches"); ax.legend()
    plt.show()
    print("[GoalReach MotorNet Torch] demo complete.")


if __name__ == "__main__":
    main()
