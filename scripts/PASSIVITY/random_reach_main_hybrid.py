#!/usr/bin/env python3
"""
scripts/run_random_reach_hybrid_tank.py

Runs random-reach with HybridEnergyTankController and prints/saves a rich report.

It avoids relying on the internal layout of Logs by also exporting controller.trace.
"""

import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt

from model_lib.environment_numpy import Environment
from model_lib.muscles_numpy import RigidTendonHillMuscle
from model_lib.effector_numpy import RigidTendonArm26

from config import (
    PlantConfig, ControlToggles, ControlGains, Numerics,
    InternalForceConfig, TrajectoryConfig, RunConfig,
)

from tasks.random_reach import Task as RandomReachTask
from trajectory.minjerk import MinJerkLinearTrajectory, MinJerkParams
from sim.simulator import TargetReachSimulator
from plotting.plots import plot_all, make_animations, hold_anims

from controller.energy_tank_hybrid import HybridEnergyTankController, HybridEnergyTankParams


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
        name="RandomReachEnv(HybridTank)",
    )
    q0 = np.deg2rad(np.array(pc.q0_deg))
    qd0 = np.array(pc.qd0)
    env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})
    return env, arm, q0


def _vec2(x, default=(1200.0, 1200.0)):
    try:
        v = np.array(x, dtype=float).ravel()
        if v.size == 1:
            return np.array([float(v[0]), float(v[0])], dtype=float)
        if v.size >= 2:
            return np.array([float(v[0]), float(v[1])], dtype=float)
    except Exception:
        pass
    return np.array(default, dtype=float)


def _scalar(x, default=1.0) -> float:
    try:
        v = np.array(x, dtype=float).ravel()
        if v.size >= 1:
            return float(v[0])
    except Exception:
        pass
    return float(default)


def _get_log(logs, key):
    # robust access for various Logs implementations
    if isinstance(logs, dict) and key in logs:
        return logs[key]
    if hasattr(logs, key):
        return getattr(logs, key)
    for attr in ("data", "_data", "store", "_store"):
        if hasattr(logs, attr):
            d = getattr(logs, attr)
            if isinstance(d, dict) and key in d:
                return d[key]
    if key in getattr(logs, "__dict__", {}):
        return logs.__dict__[key]
    return None


def _compute_rmse_from_logs(logs):
    x = _get_log(logs, "x_log")
    xref = _get_log(logs, "xref_log")
    if x is None or xref is None:
        # fall back to cartesian log if used
        x = _get_log(logs, "cart_log") or _get_log(logs, "cartesian_log")
    if x is None or xref is None:
        return None
    x = np.asarray(x)
    xref = np.asarray(xref)
    if xref.ndim == 2 and xref.shape[1] >= 2:
        xref_xy = xref[:, :2]
    else:
        return None
    if x.ndim == 2 and x.shape[1] >= 2:
        x_xy = x[:, :2]
    else:
        return None
    err = xref_xy - x_xy
    err_norm = np.linalg.norm(err, axis=1)
    rmse = float(np.sqrt(np.mean(err_norm**2)))
    maxerr = float(np.max(err_norm))
    return {"rmse_m": rmse, "rmse_cm": 100.0*rmse, "maxerr_cm": 100.0*maxerr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius", type=float, default=0.10)
    parser.add_argument("--n_points", type=int, default=4)
    parser.add_argument("--Kp", type=float, default=1200.0)
    parser.add_argument("--w_base", type=float, default=0.25)
    parser.add_argument("--rho_mu", type=float, default=1.0)
    parser.add_argument("--out_dir", type=str, default="hybrid_tank_runs")
    args = parser.parse_args()

    pc = PlantConfig()
    toggles = ControlToggles()
    gains = ControlGains()
    num = Numerics()
    ifc = InternalForceConfig()
    tc = TrajectoryConfig()
    run = RunConfig()

    env, arm, q0 = build_env(pc)

    task = RandomReachTask(n_points=args.n_points, radius=args.radius, seed=args.seed)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale))

    # ---- tracking stiffness (tune like your best PD/IF) ----
    K0 = np.diag([args.Kp, args.Kp])
    D0 = np.diag([8.0, 8.0])  # modest passive damping; Kv is inertia-shaped

    # NOTE: if your config has weird lam_os_max (e.g., 0.01), override it here:
    lam_os_max = float(getattr(num, "lam_os_max", 200.0))
    if lam_os_max < 1.0:
        lam_os_max = 200.0

    p = HybridEnergyTankParams(
        K0=K0,
        D0=D0,
        Kff=_scalar(getattr(gains, "Kff_x", 1.0), 1.0),
        zeta=0.85,

        w_passivity_base=float(args.w_base),
        adapt_passivity=True,
        E_blend_lo=0.10,
        E_blend_hi=0.35,
        eta_blend_lo=0.35,
        eta_blend_hi=0.85,
        K_scale_min=0.35,
        rho_mu=float(args.rho_mu),

        eps=float(getattr(num, "eps", 1e-6)),
        lam_os_max=lam_os_max,
        sigma_thresh=float(getattr(num, "sigma_thresh", 1e-4)),
        gate_pow=float(getattr(num, "gate_pow", 2.0)),

        enable_inertia_comp=bool(getattr(toggles, "enable_inertia_comp", True)),
        enable_gravity_comp=bool(getattr(toggles, "enable_gravity_comp", True)),
        enable_coriolis_comp=bool(getattr(toggles, "enable_velocity_comp", True)),
        enable_joint_damping=bool(getattr(toggles, "enable_joint_damping", True)),

        E0=0.25, Emin=1e-4, Emax=1.5,
        harvest_gain=1.0,
        store_returned_energy=True,
        eta_min=0.25,

        bisect_iters=int(getattr(ifc, "bisect_iters", 12)),
        clamp_tau_to_muscles=True,
        tau_cap_margin=0.95,

        enable_internal_force=bool(getattr(toggles, "enable_internal_force", False)),
        cocon_a0=float(getattr(ifc, "cocon_a0", 0.0)),
        linesearch_eps=float(getattr(num, "linesearch_eps", 1e-6)),
        linesearch_safety=float(getattr(num, "linesearch_safety", 1.2)),

        store_trace=True,
    )

    ctrl = HybridEnergyTankController(env, arm, p)
    ctrl.reset(q0)

    steps = int(pc.max_ep_duration / arm.dt)

    # --------- banner ---------
    print("="*80)
    print("[HYBRID TANK] CONFIG")
    print("="*80)
    print(f"seed={args.seed} | radius={args.radius} | n_points={args.n_points}")
    print(f"Kp={args.Kp} | w_passivity_base={args.w_base} | rho_mu={args.rho_mu} | dt={arm.dt} | steps={steps}")
    print(f"toggles: inertia={p.enable_inertia_comp}, gravity={p.enable_gravity_comp}, coriolis={p.enable_coriolis_comp}, joint_damp={p.enable_joint_damping}")
    print(f"tank: E0={p.E0}, Emin={p.Emin}, Emax={p.Emax}, harvest_gain={p.harvest_gain}")
    print(f"guards: eps={p.eps}, lam_os_max={p.lam_os_max}, sigma_thresh={p.sigma_thresh}, gate_pow={p.gate_pow}")
    print("="*80)

    sim = TargetReachSimulator(env, arm, ctrl, traj, steps)
    logs = sim.run()

    # --------- compute metrics ---------
    rmse = _compute_rmse_from_logs(logs)

    # also from trace (always available)
    trace = ctrl.get_trace()
    if trace and rmse is None:
        err = np.array([d["err"] for d in trace], dtype=float)
        rmse = {"rmse_m": float(np.sqrt(np.mean(err**2))), "rmse_cm": float(100*np.sqrt(np.mean(err**2))), "maxerr_cm": float(100*np.max(err))}

    print("="*80)
    print("[HYBRID TANK] RUN SUMMARY")
    print("="*80)
    if rmse is None:
        print("Tracking RMSE: (could not compute — logs missing x_log/xref_log and trace empty)")
    else:
        print(f"Tracking RMSE: {rmse['rmse_cm']:.3f} cm | max err: {rmse['maxerr_cm']:.3f} cm")
    print("="*80)

    # --------- export: report + series ---------
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"seed{args.seed}_r{args.radius:.2f}_Kp{args.Kp:.0f}_w{args.w_base:.2f}_rho{args.rho_mu:.2f}"
    report_path = os.path.join(args.out_dir, f"{tag}_report.json")
    series_path = os.path.join(args.out_dir, f"{tag}_series.npz")

    report = {
        "tag": tag,
        "seed": args.seed,
        "radius": args.radius,
        "n_points": args.n_points,
        "Kp": args.Kp,
        "w_passivity_base": args.w_base,
        "rho_mu": args.rho_mu,
        "dt": float(arm.dt),
        "steps": int(steps),
        "toggles": {
            "inertia": bool(p.enable_inertia_comp),
            "gravity": bool(p.enable_gravity_comp),
            "coriolis": bool(p.enable_coriolis_comp),
            "joint_damping": bool(p.enable_joint_damping),
        },
        "tank": {"E0": float(p.E0), "Emin": float(p.Emin), "Emax": float(p.Emax), "harvest_gain": float(p.harvest_gain)},
        "guards": {"eps": float(p.eps), "lam_os_max": float(p.lam_os_max), "sigma_thresh": float(p.sigma_thresh), "gate_pow": float(p.gate_pow)},
        "rmse": rmse,
        "logs_keys": sorted(list(getattr(logs, "__dict__", {}).keys())),
        "trace_len": len(trace),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[HYBRID TANK] Saved report JSON: {report_path}")

    if trace:
        t = np.arange(len(trace)) * float(arm.dt)
        np.savez(
            series_path,
            t=t,
            err=np.array([d["err"] for d in trace], dtype=float),
            E=np.array([d["E"] for d in trace], dtype=float),
            s=np.array([d["s"] for d in trace], dtype=float),
            w_pass=np.array([d["w_pass"] for d in trace], dtype=float),
            gK=np.array([d["gK"] for d in trace], dtype=float),
            eta=np.array([d["eta"] for d in trace], dtype=float),
            P_diss=np.array([d["P_diss"] for d in trace], dtype=float),
            P_inj=np.array([d["P_inj"] for d in trace], dtype=float),
            P_spend=np.array([d["P_spend"] for d in trace], dtype=float),
        )
        print(f"[HYBRID TANK] Saved series NPZ: {series_path}")
    else:
        print("[HYBRID TANK] NOTE: controller trace empty (store_trace=False?) so NPZ not saved.")

    # --------- normal plotting ---------
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

    print("Random reach (Hybrid Energy Tank) complete.")
    plt.show()


if __name__ == "__main__":
    main()
