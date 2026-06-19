#!/usr/bin/env python3
# scripts/BENCHMARK/run_benchmark.py
# Paper Section X benchmark harness: the 4 classical model-based controllers
# (impedance=PD/IF, passivity=energy-tank, sliding-mode, OSC) x 4 tasks
# (T1 center-out, T2 obstacle, T3 load-hold, T4 Lissajous), with optional sensor
# noise / motor delay / impulsive disturbance, reporting endpoint RMSE, integrated
# effort E = int sum_i a_i^2 dt, and real-time factor (RTF) -> Tables XIV-style.
#
#   python -m scripts.BENCHMARK.run_benchmark                  # nominal
#   python -m scripts.BENCHMARK.run_benchmark --noise --disturb --motor_delay_ms 20
from __future__ import annotations

import argparse
import time
import numpy as np

from config import (PlantConfig, ControlToggles, ControlGains, Numerics,
                    InternalForceConfig, TrajectoryConfig)
from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26

from trajectory.numpy.minjerk import MinJerkLinearTrajectory, MinJerkParams
from trajectory.numpy.lissajous import LissajousTrajectory, LissajousParams
from tasks.numpy.base_task import CenterOutTask
from tasks.numpy.obstacle import ObstacleAvoidanceTask
from tasks.numpy.load_hold import LoadHoldTask

from controller.numpy.pd_if_controller import PDIFController, PDIFParams
from controller.numpy.energy_tank_controller import EnergyTankController, EnergyTankParams
from controller.numpy.sliding_mode import SlidingModeController, SlidingModeParams
from controller.numpy.osc_controller import OSCController, OSCParams


# ----------------------------- env -----------------------------
def build_env(pc, action_noise=0.0, obs_noise=0.0):
    muscle = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(muscle=muscle, timestep=pc.timestep, damping=pc.damping,
                           n_ministeps=pc.n_ministeps, integration_method=pc.integration_method)
    env = Environment(effector=arm, max_ep_duration=pc.max_ep_duration,
                      action_noise=action_noise, obs_noise=obs_noise,
                      proprioception_delay=arm.dt, vision_delay=arm.dt, name="Benchmark")
    q0 = np.deg2rad(np.array(pc.q0_deg)); qd0 = np.array(pc.qd0)
    env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})
    return env, arm, q0


# ----------------------------- controller factory -----------------------------
def make_controller(name, env, arm, cfg):
    toggles, gains, num, ifc = cfg
    if name == "Impedance (PD/IF)":
        p = PDIFParams(
            Kp_task=np.array([1600.0, 1600.0]), damping_ratio=0.7, Kff=1.0,
            use_critical_damping=True,
            enable_inertia_comp=toggles.enable_inertia_comp,
            enable_gravity_comp=toggles.enable_gravity_comp,
            enable_coriolis_comp=toggles.enable_velocity_comp,
            enable_nullspace=True, Kp_null=20.0, Kd_null=5.0,
            eps=num.eps, lam_os_max=float(getattr(num, "lam_os_max", 200.0)),
            sigma_thresh=num.sigma_thresh, gate_pow=num.gate_pow,
            bisect_iters=ifc.bisect_iters, enable_internal_force=False, cocon_level=0.0)
        return PDIFController(env, arm, p)
    if name == "Passivity":
        p = EnergyTankParams(
            D0=np.diag([25.0, 25.0]), K0=np.diag([4000.0, 4000.0]), Kff=2.0, zeta=1.0,
            strict_passivity=False, KI=np.array([0.0, 0.0]), Imax=np.array([0.0, 0.0]),
            eps=num.eps, lam_os_max=num.lam_os_max, sigma_thresh=num.sigma_thresh,
            gate_pow=num.gate_pow, enable_inertia_comp=toggles.enable_inertia_comp,
            enable_gravity_comp=toggles.enable_gravity_comp,
            enable_joint_damping=toggles.enable_joint_damping,
            enable_internal_force=False, cocon_a0=0.0, bisect_iters=ifc.bisect_iters,
            linesearch_eps=num.linesearch_eps, linesearch_safety=num.linesearch_safety,
            E0=0.15, Emin=1e-4, Emax=1.0)
        return EnergyTankController(env, arm, p)
    if name == "Sliding-mode":
        p = SlidingModeParams(
            lambda_surf=np.array([26.0, 26.0]), K_switch=np.array([4.0, 4.0]),
            phi=np.array([0.004, 0.004]), Kff_x=1.0,
            enable_gravity_comp=toggles.enable_gravity_comp,
            enable_velocity_comp=toggles.enable_velocity_comp,
            enable_inertia_comp=toggles.enable_inertia_comp,
            lambda_ns=0.0, k_manip=0.0, eps=num.eps, lam_os_max=num.lam_os_max,
            sigma_thresh=num.sigma_thresh, gate_pow=num.gate_pow, bisect_iters=ifc.bisect_iters,
            switch_in_accel=True, use_condJ_for_blend=True, cond_low=20.0, cond_high=80.0,
            elbow_min_rad=np.deg2rad(2.0))
        return SlidingModeController(env, arm, p)
    if name == "OSC":
        return OSCController(env, arm, OSCParams(Kp=4000.0, Kv=126.0, Kp_posture=50.0, Kv_posture=10.0))
    raise ValueError(name)


CONTROLLERS = ["Impedance (PD/IF)", "Passivity", "Sliding-mode", "OSC"]


# ----------------------------- tasks -----------------------------
def make_task_traj(name, env, tc):
    mjp = MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
    if name == "T1 center-out":
        t = CenterOutTask(n_targets=8, radius=0.10)
        return t, MinJerkLinearTrajectory(t.build_waypoints(env), mjp), {}
    if name == "T2 obstacle":
        t = ObstacleAvoidanceTask(reach=(0.10, 0.06), obstacle_radius=0.025, clearance=0.015)
        return t, MinJerkLinearTrajectory(t.build_waypoints(env), mjp), {}
    if name == "T3 load-hold":
        t = LoadHoldTask(mass=0.5, g_eff=9.81, direction=(0.0, -1.0), reach=(0.08, 0.0))
        return t, MinJerkLinearTrajectory(t.build_waypoints(env), mjp), {"load": t.endpoint_load()}
    if name == "T4 Lissajous":
        center = env.states["fingertip"][0, :2].copy()
        t = CenterOutTask(); t.center = center; t.targets = center[None, :]  # plotting stub
        return t, LissajousTrajectory(center, LissajousParams()), {}
    raise ValueError(name)


TASKS = ["T1 center-out", "T2 obstacle", "T3 load-hold", "T4 Lissajous"]


# ----------------------------- one run -----------------------------
def run_one(env, arm, ctrl, traj, task, steps, q0, info, args):
    dt = float(arm.dt)
    ctrl.reset(q0.copy() if hasattr(q0, "copy") else q0)
    zero = np.zeros((1, 2))
    load = info.get("load", None)
    # motor-delay ring buffer on the applied action
    delay_steps = int(round((args.motor_delay_ms / 1000.0) / dt)) if args.motor_delay_ms > 0 else 0
    abuf = []

    xs, refs, effort = [], [], 0.0
    t0 = time.perf_counter()
    for k in range(steps):
        t_now = k * dt
        x_d, xd_d, xdd_d = traj.sample(t_now)
        diag = ctrl.compute(x_d, xd_d, xdd_d)
        a = np.asarray(diag["act"]).reshape(-1)

        # motor/transport delay: apply the action issued delay_steps ago
        if delay_steps > 0:
            abuf.append(a.copy())
            a_apply = abuf[0] if len(abuf) > delay_steps else abuf[-1]
            if len(abuf) > delay_steps:
                abuf.pop(0)
        else:
            a_apply = a

        # disturbance: 0.35 N.m impulsive elbow torque for 60ms, every `disturb_period_s`
        jl = zero
        if args.disturb:
            phase = t_now % args.disturb_period_s
            if phase < 0.060:
                jl = np.array([[0.0, 0.35]])
        el = load[None, :] if load is not None else zero

        env.step(a_apply[None, :], deterministic=not args.noise, endpoint_load=el, joint_load=jl)

        x = env.states["cartesian"][0, :2].copy()
        xs.append(x); refs.append(np.asarray(x_d)[:2].copy())
        effort += float(np.sum(a_apply ** 2)) * dt
    wall = time.perf_counter() - t0

    xs = np.asarray(xs); refs = np.asarray(refs)
    err = np.linalg.norm(xs - refs, axis=1)
    rmse_cm = float(np.sqrt(np.mean(err ** 2)) * 100.0)
    sim_time = steps * dt
    rtf = sim_time / max(wall, 1e-9)
    extra = ""
    if isinstance(task, ObstacleAvoidanceTask):
        clr = task.clearance_metric(xs) * 1000.0
        hit = "HIT" if clr < task.obstacle_radius * 1000.0 else "clear"
        extra = f" | min-clearance {clr:.1f}mm ({hit}, obstacle r={task.obstacle_radius*1000:.0f}mm)"
    if isinstance(task, LoadHoldTask):
        hold = float(np.mean(err[-int(0.3 * steps):]) * 1000.0)  # last-30% hold error
        extra = f" | hold-err {hold:.2f}mm under {np.linalg.norm(load):.1f}N load"
    return dict(rmse_cm=rmse_cm, effort=effort, rtf=rtf, extra=extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", action="store_true", help="sensor noise (0.25deg joints, 0.6mm endpoint)")
    ap.add_argument("--disturb", action="store_true", help="0.35 N.m impulsive elbow disturbance (60ms periodic)")
    ap.add_argument("--disturb_period_s", type=float, default=1.0)
    ap.add_argument("--motor_delay_ms", type=float, default=0.0)
    args = ap.parse_args()

    pc = PlantConfig()
    cfg = (ControlToggles(), ControlGains(), Numerics(), InternalForceConfig())
    tc = TrajectoryConfig()
    # sensor noise (paper: sigma_theta=0.25deg on joints, sigma_x=0.6mm on endpoint)
    obs_noise = float(np.deg2rad(0.25)) if args.noise else 0.0

    print("=" * 92)
    print("Paper Section-X benchmark | noise=%s disturb=%s motor_delay=%gms" %
          (args.noise, args.disturb, args.motor_delay_ms))
    print("metrics: RMSE [cm], effort E=int sum a^2 dt [-], RTF [x real-time]")
    print("=" * 92)
    header = f"{'controller':<18}" + "".join(f"{t:<16}" for t in TASKS)
    print(header)

    for cname in CONTROLLERS:
        row = f"{cname:<18}"
        extras = []
        for tname in TASKS:
            env, arm, q0 = build_env(pc, action_noise=0.0, obs_noise=obs_noise)
            ctrl = make_controller(cname, env, arm, cfg)
            task, traj, info = make_task_traj(tname, env, tc)
            steps = int(pc.max_ep_duration / arm.dt)
            try:
                m = run_one(env, arm, ctrl, traj, task, steps, q0, info, args)
                row += f"{m['rmse_cm']:.2f}/{m['effort']:.1f}/{m['rtf']:.1f}x".ljust(16)
                if m["extra"]:
                    extras.append(f"    {cname} {tname}:{m['extra']}")
            except Exception as e:
                row += f"{'ERR':<16}"
                extras.append(f"    {cname} {tname}: ERROR {str(e)[:70]}")
        print(row)
        for ex in extras:
            print(ex)
    print("=" * 92)
    print("cell = RMSE[cm] / effort / RTF")


if __name__ == "__main__":
    main()
