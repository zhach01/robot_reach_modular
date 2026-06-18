#!/usr/bin/env python3
import os
import argparse
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
)

from tasks.numpy.random_reach import Task as RandomReachTask
from trajectory.numpy.minjerk import MinJerkLinearTrajectory, MinJerkParams

from sim.numpy.simulator import TargetReachSimulator

# pick a “teacher” controller that TRACKS well (recommended: PD/IF)
from controller.numpy.pd_if_controller import PDIFController, PDIFParams
# or you can swap to your working synergy-hybrid controller if you prefer:
# from controller.numpy.synergy_controller import SynergyController, SynergyParams


# -------------------------------
# A wrapper that records compute()
# -------------------------------
class RecordingController:
    def __init__(self, ctrl, min_activation: float = 0.0, subtract_min_activation: bool = True):
        self.ctrl = ctrl
        self.min_activation = float(min_activation)
        self.subtract_min_activation = bool(subtract_min_activation)

        # expose qref immediately if underlying controller has it
        self.qref = getattr(self.ctrl, "qref", None)

        self.reset_buffers()

    def reset_buffers(self):
        self.act_list = []
        self.x_list = []
        self.q_list = []
        self.tau_list = []

    def reset(self, q0):
        if hasattr(self.ctrl, "reset"):
            self.ctrl.reset(q0)

        # keep qref in sync for simulator logging
        self.qref = getattr(self.ctrl, "qref", None)

        self.reset_buffers()

    def compute(self, x_d, xd_d, xdd_d):
        out = self.ctrl.compute(x_d, xd_d, xdd_d)

        # keep qref in sync each step (simulator reads it)
        self.qref = getattr(self.ctrl, "qref", None)

        a = np.array(out["act"], dtype=float).reshape(-1)
        if self.subtract_min_activation:
            a = a - self.min_activation
        a = np.clip(a, 0.0, 1.0)
        self.act_list.append(a)

        if "x" in out:
            self.x_list.append(np.array(out["x"], dtype=float).reshape(-1))
        if "q" in out:
            self.q_list.append(np.array(out["q"], dtype=float).reshape(-1))
        if "tau_des" in out:
            self.tau_list.append(np.array(out["tau_des"], dtype=float).reshape(-1))

        return out


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
        name="SynergyDatasetEnv",
    )
    q0 = np.deg2rad(np.array(pc.q0_deg))
    qd0 = np.array(pc.qd0)
    env.reset(
        options={
            "joint_state": np.concatenate([q0, qd0])[None, :],
            "deterministic": True,
        }
    )
    return env, arm, q0, muscle


def make_teacher_controller(env, arm, toggles, gains, num, ifc):
    # --- PD/IF params (teacher) ---
    p = PDIFParams(
        Kp_x=gains.Kp_x,
        Kff_x=gains.Kff_x,
        Kp_q=gains.Kp_q,
        Kd_q=gains.Kd_q,
        eps=num.eps,
        lam_os_smin_target=num.lam_os_smin_target,
        lam_os_max=num.lam_os_max,
        sigma_thresh=num.sigma_thresh,
        gate_pow=num.gate_pow,
        enable_internal_force=toggles.enable_internal_force,
        enable_inertia_comp=toggles.enable_inertia_comp,
        enable_gravity_comp=toggles.enable_gravity_comp,
        enable_velocity_comp=toggles.enable_velocity_comp,
        enable_joint_damping=toggles.enable_joint_damping,
        cocon_a0=ifc.cocon_a0,
        bisect_iters=ifc.bisect_iters,
        linesearch_eps=num.linesearch_eps,
        linesearch_safety=num.linesearch_safety,
    )
    return PDIFController(env, arm, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--radius", type=float, default=0.10)
    ap.add_argument("--n_points", type=int, default=1)
    ap.add_argument("--out", type=str, default="synergy/training/act_dataset.npz")
    ap.add_argument("--subtract_min_activation", action="store_true")
    args = ap.parse_args()

    pc = PlantConfig()
    toggles = ControlToggles()
    gains = ControlGains()
    num = Numerics()
    ifc = InternalForceConfig()
    tc = TrajectoryConfig()

    all_act = []
    all_x = []
    all_q = []
    all_tau = []
    ep_slices = []

    for ep in range(args.episodes):
        env, arm, q0, muscle = build_env(pc)

        task = RandomReachTask(n_points=args.n_points, radius=args.radius, seed=args.seed0 + ep)
        waypoints = task.build_waypoints(env)
        traj = MinJerkLinearTrajectory(
            waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
        )

        teacher = make_teacher_controller(env, arm, toggles, gains, num, ifc)

        rec = RecordingController(
            teacher,
            min_activation=float(getattr(muscle, "min_activation", 0.0)),
            subtract_min_activation=bool(args.subtract_min_activation),
        )
        rec.reset(q0)

        steps = int(pc.max_ep_duration / arm.dt)
        sim = TargetReachSimulator(env, arm, rec, traj, steps)
        sim.run()

        A = np.asarray(rec.act_list, dtype=float)  # (T,6)
        start = sum(s[1] - s[0] for s in ep_slices) if ep_slices else 0
        end = start + A.shape[0]
        ep_slices.append((start, end))

        all_act.append(A)
        if rec.x_list:   all_x.append(np.asarray(rec.x_list, dtype=float))
        if rec.q_list:   all_q.append(np.asarray(rec.q_list, dtype=float))
        if rec.tau_list: all_tau.append(np.asarray(rec.tau_list, dtype=float))

        print(f"[ep {ep+1:03d}/{args.episodes}] collected T={A.shape[0]} act samples")

    act = np.concatenate(all_act, axis=0) if all_act else np.zeros((0, 6), dtype=float)
    X = np.concatenate(all_x, axis=0) if all_x else None
    Q = np.concatenate(all_q, axis=0) if all_q else None
    TAU = np.concatenate(all_tau, axis=0) if all_tau else None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    save_dict = {
        "act": act,                 # (T,6) in [0,1]
        "dt": float(pc.timestep),
        "episodes": int(args.episodes),
        "ep_slices": np.asarray(ep_slices, dtype=int),
        "radius": float(args.radius),
        "n_points": int(args.n_points),
        "seed0": int(args.seed0),
    }
    if X is not None: save_dict["x"] = X
    if Q is not None: save_dict["q"] = Q
    if TAU is not None: save_dict["tau_des"] = TAU

    np.savez(args.out, **save_dict)
    print(f"[done] saved dataset: {args.out}")
    print(f"       act shape = {act.shape}  (T,6)")


if __name__ == "__main__":
    main()

