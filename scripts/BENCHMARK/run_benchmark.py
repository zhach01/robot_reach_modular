#!/usr/bin/env python3
# scripts/BENCHMARK/run_benchmark.py
# Paper Section-X benchmark harness, calibrated to the paper's protocol.
#
# Controllers (4 classical): impedance=PD/IF, passivity=energy-tank, sliding-mode, OSC.
# Tasks (4): T1 center-out, T2 obstacle, T3 load-hold, T4 Lissajous.
# Protocol (paper Section X-A "common simulation environment"):
#   - plant integration: RK4 @ 0.1 ms sub-steps (PLANT_INTEGRATION / PLANT_SUBSTEP_S).
#   - motor delay: 20 ms transport delay on the applied excitation (ring buffer here).
#   - sensor noise: sigma_theta = 0.25 deg on joint angles, sigma_x = 0.6 mm on the
#     endpoint. Both the noise and the 20 ms delay compensation live in the control law
#     via PredictiveController (the lead predictor IS part of the controller), not in
#     the simulator; the plant always integrates the true state.
#   - disturbance: 0.35 N.m impulsive elbow torque for 60 ms on every 5th reach segment.
# Metrics: endpoint RMSE [cm], integrated effort E = int sum_i a_i^2 dt, RTF [x real-time].
# The perturbed condition is averaged over --seeds noise realizations (mean +/- SD).
#
#   python -m scripts.BENCHMARK.run_benchmark               # nominal + paper-protocol
#   python -m scripts.BENCHMARK.run_benchmark --latex       # + LaTeX table body
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
from controller.numpy.predictive import PredictiveController

# paper protocol constants
SIGMA_THETA = np.deg2rad(0.25)   # rad, joint-angle sensor noise
SIGMA_X = 0.6e-3                  # m, endpoint sensor noise
DELAY_MS = 20.0                  # ms, motor transport delay (paper Section X-A)
DIST_TORQUE = 0.35               # N.m, elbow disturbance
DIST_DUR = 0.060                 # s, disturbance pulse width
DIST_EVERY = 5                   # every 5th reach segment

# plant integration (paper Section X-A: RK4 @ 0.1 ms)
PLANT_INTEGRATION = "rk4"
PLANT_SUBSTEP_S = 0.1e-3          # 0.1 ms sub-step
# delay compensation: Smith-style lead predictor on the controller feedback.
# The 20 ms transport delay caps achievable bandwidth; we feed each controller a
# model-based estimate of the joint/Cartesian state DELAY_MS ahead (second-order
# lead from a clean velocity derivative + FK), recovering the lost phase margin.
DELAY_COMP = True
DELAY_COMP_FRAC = 1.0            # fraction of the transport delay to predict ahead


def build_env(pc):
    muscle = RigidTendonHillMuscle(min_activation=0.02)
    n_ministeps = max(1, int(round(pc.timestep / PLANT_SUBSTEP_S)))   # -> 0.1 ms sub-steps
    arm = RigidTendonArm26(muscle=muscle, timestep=pc.timestep, damping=pc.damping,
                           n_ministeps=n_ministeps, integration_method=PLANT_INTEGRATION)
    env = Environment(effector=arm, max_ep_duration=pc.max_ep_duration, action_noise=0.0,
                      obs_noise=0.0, proprioception_delay=arm.dt, vision_delay=arm.dt, name="Benchmark")
    q0 = np.deg2rad(np.array(pc.q0_deg)); qd0 = np.array(pc.qd0)
    env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})
    return env, arm, q0


# --- tunable common-operating-point gains (set by the tuning harness) ---
IMP_KP = 1600.0        # impedance task stiffness
PAS_K0 = 1600.0        # passivity task stiffness (common operating point with impedance)
PAS_ZETA = 1.0
SMC_LAMBDA = 22.0      # sliding surface slope
SMC_KSW = 12.0          # sliding switching gain (-> robustness + effort)
SMC_PHI = 0.003        # sliding boundary layer


def make_controller(name, env, arm, cfg):
    toggles, gains, num, ifc = cfg
    if name == "Impedance (PD/IF)":
        return PDIFController(env, arm, PDIFParams(
            Kp_task=np.array([IMP_KP, IMP_KP]), damping_ratio=0.7, Kff=1.0, use_critical_damping=True,
            enable_inertia_comp=toggles.enable_inertia_comp, enable_gravity_comp=toggles.enable_gravity_comp,
            enable_coriolis_comp=toggles.enable_velocity_comp, enable_nullspace=True, Kp_null=20.0, Kd_null=5.0,
            eps=num.eps, lam_os_max=float(getattr(num, "lam_os_max", 200.0)), sigma_thresh=num.sigma_thresh,
            gate_pow=num.gate_pow, bisect_iters=ifc.bisect_iters, enable_internal_force=False, cocon_level=0.0))
    if name == "Passivity":
        return EnergyTankController(env, arm, EnergyTankParams(
            D0=np.diag([25.0, 25.0]), K0=np.diag([PAS_K0, PAS_K0]), Kff=2.0, zeta=PAS_ZETA, strict_passivity=False,
            KI=np.array([0.0, 0.0]), Imax=np.array([0.0, 0.0]), eps=num.eps, lam_os_max=num.lam_os_max,
            sigma_thresh=num.sigma_thresh, gate_pow=num.gate_pow, enable_inertia_comp=toggles.enable_inertia_comp,
            enable_gravity_comp=toggles.enable_gravity_comp, enable_joint_damping=toggles.enable_joint_damping,
            enable_internal_force=False, cocon_a0=0.0, bisect_iters=ifc.bisect_iters,
            linesearch_eps=num.linesearch_eps, linesearch_safety=num.linesearch_safety, E0=0.15, Emin=1e-4, Emax=1.0))
    if name == "Sliding-mode":
        return SlidingModeController(env, arm, SlidingModeParams(
            lambda_surf=np.array([SMC_LAMBDA, SMC_LAMBDA]), K_switch=np.array([SMC_KSW, SMC_KSW]), phi=np.array([SMC_PHI, SMC_PHI]), Kff_x=1.0,
            enable_gravity_comp=toggles.enable_gravity_comp, enable_velocity_comp=toggles.enable_velocity_comp,
            enable_inertia_comp=toggles.enable_inertia_comp, lambda_ns=0.0, k_manip=0.0, eps=num.eps,
            lam_os_max=num.lam_os_max, sigma_thresh=num.sigma_thresh, gate_pow=num.gate_pow, bisect_iters=ifc.bisect_iters,
            switch_in_accel=True, use_condJ_for_blend=True, cond_low=20.0, cond_high=80.0, elbow_min_rad=np.deg2rad(2.0)))
    if name == "OSC":
        return OSCController(env, arm, OSCParams(Kp=4000.0, Kv=126.0, Kp_posture=50.0, Kv_posture=10.0))
    raise ValueError(name)


CONTROLLERS = ["Impedance (PD/IF)", "Passivity", "Sliding-mode", "OSC"]


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
        c = env.states["fingertip"][0, :2].copy()
        t = CenterOutTask(); t.center = c; t.targets = c[None, :]
        return t, LissajousTrajectory(c, LissajousParams()), {}
    raise ValueError(name)


TASKS = ["T1 center-out", "T2 obstacle", "T3 load-hold", "T4 Lissajous"]


def _seg_index(tgrid, t):
    if tgrid is None:
        return int(t // 1.0)            # Lissajous: 1 s pseudo-segments
    if t >= tgrid[-1]:
        return len(tgrid) - 2
    return max(0, int(np.searchsorted(tgrid, t) - 1))


def run_one(env, arm, ctrl, traj, task, steps, q0, info, perturb, rng):
    dt = float(arm.dt)
    zero = np.zeros((1, 2))
    load = info.get("load", None)
    tgrid = getattr(traj, "tgrid", None)
    delay_steps = int(round((DELAY_MS / 1000.0) / dt)) if perturb else 0
    eff = env.effector

    # Delay compensation + sensor noise live in the control law (predictive wrapper),
    # not the harness. Under the perturbed protocol every controller is wrapped: it is
    # given the state predicted DELAY_MS ahead (lead disabled if DELAY_COMP is off) with
    # sensor noise on that measurement. The harness keeps only the plant-side effects
    # (actuator transport-delay buffer, disturbance torque, endpoint load).
    if perturb:
        pred_h = (delay_steps * dt) * DELAY_COMP_FRAC if DELAY_COMP else 0.0
        ctrl = PredictiveController(ctrl, eff, pred_h, dt, rng=rng,
                                    sigma_theta=SIGMA_THETA, sigma_x=SIGMA_X)
    ctrl.reset(q0.copy())

    abuf = []
    xs, refs, effort = [], [], 0.0
    t0 = time.perf_counter()
    for k in range(steps):
        t_now = k * dt
        x_d, xd_d, xdd_d = traj.sample(t_now)
        diag = ctrl.compute(x_d, xd_d, xdd_d)

        a = np.asarray(diag["act"]).reshape(-1)
        if delay_steps > 0:
            abuf.append(a.copy())
            a_apply = abuf.pop(0) if len(abuf) > delay_steps else abuf[-1]
        else:
            a_apply = a

        # disturbance on every 5th reach segment, first DIST_DUR seconds
        jl = zero
        if perturb:
            si = _seg_index(tgrid, t_now)
            t_seg0 = tgrid[si] if tgrid is not None else si * 1.0
            if (si % DIST_EVERY) == (DIST_EVERY - 1) and (t_now - t_seg0) < DIST_DUR:
                jl = np.array([[0.0, DIST_TORQUE]])
        el = load[None, :] if load is not None else zero

        env.step(a_apply[None, :], deterministic=True, endpoint_load=el, joint_load=jl)
        x = env.states["cartesian"][0, :2].copy()
        xs.append(x); refs.append(np.asarray(x_d)[:2].copy())
        effort += float(np.sum(a_apply ** 2)) * dt
    wall = time.perf_counter() - t0

    xs = np.asarray(xs); refs = np.asarray(refs)
    err = np.linalg.norm(xs - refs, axis=1)
    out = dict(rmse_cm=float(np.sqrt(np.mean(err ** 2)) * 100.0), effort=effort,
               rtf=(steps * dt) / max(wall, 1e-9))
    if isinstance(task, ObstacleAvoidanceTask):
        out["clearance_mm"] = task.clearance_metric(xs) * 1000.0
        out["obs_r_mm"] = task.obstacle_radius * 1000.0
    if isinstance(task, LoadHoldTask):
        out["hold_mm"] = float(np.mean(err[-int(0.3 * steps):]) * 1000.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="noise realizations for the perturbed condition")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    pc = PlantConfig()
    cfg = (ControlToggles(), ControlGains(), Numerics(), InternalForceConfig())
    tc = TrajectoryConfig()

    def run(cname, tname, perturb, seed):
        env, arm, q0 = build_env(pc)
        ctrl = make_controller(cname, env, arm, cfg)
        task, traj, info = make_task_traj(tname, env, tc)
        steps = int(pc.max_ep_duration / arm.dt)
        return run_one(env, arm, ctrl, traj, task, steps, q0, info, perturb, np.random.default_rng(seed))

    results = {}  # (cond, cname, tname) -> dict of stats
    for cname in CONTROLLERS:
        for tname in TASKS:
            results[("nominal", cname, tname)] = run(cname, tname, False, 0)
            runs = [run(cname, tname, True, s) for s in range(args.seeds)]
            agg = {}
            for key in ("rmse_cm", "effort", "rtf", "clearance_mm", "hold_mm"):
                vals = [r[key] for r in runs if key in r]
                if vals:
                    agg[key] = (float(np.mean(vals)), float(np.std(vals)))
            agg["obs_r_mm"] = runs[0].get("obs_r_mm")
            results[("paper", cname, tname)] = agg

    def cell_nom(r):
        return f"{r['rmse_cm']:.2f}/{r['effort']:.1f}/{r['rtf']:.1f}x"

    def cell_pap(a):
        m, s = a["rmse_cm"]; e = a["effort"][0]
        return f"{m:.2f}+-{s:.2f}/{e:.1f}"

    for cond, fmt in (("NOMINAL (clean)", cell_nom), ("PAPER PROTOCOL (noise+20ms delay+disturb, mean+-SD over %d seeds)" % args.seeds, cell_pap)):
        key = "nominal" if cond.startswith("NOMINAL") else "paper"
        print("\n" + "=" * 100)
        print(cond + "  | cell = RMSE[cm] / effort" + ("/RTF" if key == "nominal" else " (mean+-SD)"))
        print("=" * 100)
        print(f"{'controller':<18}" + "".join(f"{t:<20}" for t in TASKS))
        for cname in CONTROLLERS:
            row = f"{cname:<18}"
            for tname in TASKS:
                r = results[(key, cname, tname)]
                row += (cell_nom(r) if key == "nominal" else cell_pap(r)).ljust(20)
            print(row)
        # task-specific extras
        for cname in CONTROLLERS:
            r = results[(key, cname, "T2 obstacle")]
            clr = r["clearance_mm"] if key == "nominal" else r["clearance_mm"][0]
            obr = r["obs_r_mm"]
            h = results[(key, cname, "T3 load-hold")]
            hold = h["hold_mm"] if key == "nominal" else h["hold_mm"][0]
            print(f"    {cname:<18} T2 clearance {clr:5.1f}mm ({'HIT' if clr < obr else 'clear'}, r={obr:.0f}mm) | T3 hold-err {hold:.2f}mm")

    if args.latex:
        print("\n" + "%" * 60 + "\n% LaTeX table body (paper protocol, RMSE[cm] mean+-SD)\n" + "%" * 60)
        for cname in CONTROLLERS:
            cells = " & ".join("$%.2f \\pm %.2f$" % results[("paper", cname, t)]["rmse_cm"] for t in TASKS)
            print(f"{cname} & {cells} \\\\")


if __name__ == "__main__":
    main()
