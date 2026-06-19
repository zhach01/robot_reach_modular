#!/usr/bin/env python3
# scripts/BENCHMARK/run_robustness.py
# Model-uncertainty robustness sweep for the classical controllers (paper Section X
# "Robustness notes"). The plant's rigid-body dynamic parameters (link masses,
# inertias, centre-of-mass offsets) are perturbed by a Latin-hypercube ensemble while
# each controller keeps its NOMINAL internal model -- a genuine model-mismatch test,
# not a re-calibrated one. We run two arms in lockstep: a perturbed plant that is
# stepped, and a nominal-model environment that the controller computes against, fed
# the perturbed plant's measurements each step. The reported degradation is the rise
# in end-effector RMSE over the nominal (unperturbed) plant.
#
#   python -m scripts.BENCHMARK.run_robustness --samples 24 --spread 0.15 --task "T1 center-out"
from __future__ import annotations

import argparse
import numpy as np

from config import (PlantConfig, ControlToggles, ControlGains, Numerics,
                    InternalForceConfig, TrajectoryConfig)
from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26
from model_lib.numpy.skeleton import TwoDofArm

import scripts.BENCHMARK.run_benchmark as B
from controller.numpy.predictive import PredictiveController

# nominal Arm26 rigid-body parameters (RigidTendonArm26 defaults)
NOMINAL = dict(m1=1.82, m2=1.43, l1g=0.135, l2g=0.165, i1=0.051, i2=0.057)
PERTURB_KEYS = ["m1", "m2", "i1", "i2", "l1g", "l2g"]   # dynamic params; lengths fixed
CLASSICAL = ["Impedance (PD/IF)", "Passivity", "Sliding-mode"]


def latin_hypercube(n, d, rng):
    """n stratified samples in [0,1]^d, one per stratum per dimension (no scipy)."""
    cut = np.linspace(0.0, 1.0, n + 1)
    u = rng.uniform(size=(n, d))
    pts = cut[:n, None] + u * (1.0 / n)
    for j in range(d):
        pts[:, j] = pts[rng.permutation(n), j]
    return pts


def build_plant(pc, params):
    skel = TwoDofArm(m1=params["m1"], m2=params["m2"], l1g=params["l1g"],
                     l2g=params["l2g"], i1=params["i1"], i2=params["i2"],
                     l1=0.309, l2=0.333)
    muscle = RigidTendonHillMuscle(min_activation=0.02)
    n_ministeps = max(1, int(round(pc.timestep / B.PLANT_SUBSTEP_S)))
    arm = RigidTendonArm26(muscle=muscle, skeleton=skel, timestep=pc.timestep,
                           damping=pc.damping, n_ministeps=n_ministeps,
                           integration_method=B.PLANT_INTEGRATION)
    env = Environment(effector=arm, max_ep_duration=pc.max_ep_duration, action_noise=0.0,
                      obs_noise=0.0, proprioception_delay=arm.dt, vision_delay=arm.dt,
                      name="Plant")
    q0 = np.deg2rad(np.array(pc.q0_deg)); qd0 = np.array(pc.qd0)
    env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})
    return env, arm, q0


def run_T1_mismatch(pc, cfg, tc, cname, params, rng):
    """T1 center-out under the Section-X protocol, controller on the nominal model,
    plant on `params`. Returns end-effector RMSE [cm] of the true (plant) endpoint."""
    plant_env, plant_arm, q0 = build_plant(pc, params)
    model_env, model_arm, _ = B.build_env(pc)                 # nominal internal model
    plant_eff, model_eff = plant_env.effector, model_env.effector

    task, traj, info = B.make_task_traj(cname_task(), model_env, tc)
    load = info.get("load", None)
    steps = int(pc.max_ep_duration / plant_arm.dt)
    dt = float(plant_arm.dt)
    delay_steps = int(round((B.DELAY_MS / 1000.0) / dt))
    pred_h = (delay_steps * dt) * B.DELAY_COMP_FRAC if B.DELAY_COMP else 0.0

    ctrl = B.make_controller(cname, model_env, model_arm, cfg)
    ctrl = PredictiveController(ctrl, model_eff, pred_h, dt, rng=rng,
                               sigma_theta=B.SIGMA_THETA, sigma_x=B.SIGMA_X)
    ctrl.reset(q0.copy())

    def sync_model_from_plant():
        for k, v in plant_eff.states.items():
            model_eff.states[k] = v.copy()
        model_env.skeleton._set_state(plant_eff.states["joint"][0, :2],
                                      plant_eff.states["joint"][0, 2:])

    sync_model_from_plant()
    abuf, zero = [], np.zeros((1, 2))
    xs, refs = [], []
    for k in range(steps):
        t_now = k * dt
        x_d, xd_d, xdd_d = traj.sample(t_now)
        diag = ctrl.compute(x_d, xd_d, xdd_d)               # nominal model + perturbed feedback
        a = np.asarray(diag["act"]).reshape(-1)
        abuf.append(a.copy())
        a_apply = abuf.pop(0) if len(abuf) > delay_steps else abuf[-1]

        jl = zero
        si = B._seg_index(getattr(traj, "tgrid", None), t_now)
        t_seg0 = traj.tgrid[si] if getattr(traj, "tgrid", None) is not None else si * 1.0
        if (si % B.DIST_EVERY) == (B.DIST_EVERY - 1) and (t_now - t_seg0) < B.DIST_DUR:
            jl = np.array([[0.0, B.DIST_TORQUE]])
        el = load[None, :] if load is not None else zero

        plant_env.step(a_apply[None, :], deterministic=True, endpoint_load=el, joint_load=jl)
        sync_model_from_plant()
        xs.append(plant_eff.states["cartesian"][0, :2].copy())
        refs.append(np.asarray(x_d)[:2].copy())

    xs, refs = np.asarray(xs), np.asarray(refs)
    err = np.linalg.norm(xs - refs, axis=1)
    return float(np.sqrt(np.mean(err ** 2)) * 100.0)


_TASK = ["T1 center-out"]
def cname_task():
    return _TASK[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--spread", type=float, default=0.15, help="+-fractional parameter spread")
    ap.add_argument("--task", default="T1 center-out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    _TASK[0] = args.task

    pc = PlantConfig()
    cfg = (ControlToggles(), ControlGains(), Numerics(), InternalForceConfig())
    tc = TrajectoryConfig()

    rng = np.random.default_rng(args.seed)
    lhs = latin_hypercube(args.samples, len(PERTURB_KEYS), rng)
    # map [0,1] -> [1-spread, 1+spread] multiplier
    mult = 1.0 + (2.0 * lhs - 1.0) * args.spread
    samples = [{k: NOMINAL[k] * mult[i, j] for j, k in enumerate(PERTURB_KEYS)}
               for i in range(args.samples)]

    print(f"Robustness sweep: {args.task} | {args.samples} LHC samples | +-{args.spread*100:.0f}% on {PERTURB_KEYS}")
    print(f"{'controller':<18}{'nominal':>9}{'pert mean':>11}{'degr.':>9}{'degr.SD':>9}{'worst':>8}")
    # fixed noise realisation across nominal + all perturbed runs, so the measured
    # degradation reflects ONLY the parameter mismatch (noise is not a confound).
    NOISE_SEED = 777
    for cname in CLASSICAL:
        nom = run_T1_mismatch(pc, cfg, tc, cname, NOMINAL, np.random.default_rng(NOISE_SEED))
        perts = [run_T1_mismatch(pc, cfg, tc, cname, s, np.random.default_rng(NOISE_SEED))
                 for i, s in enumerate(samples)]
        perts = np.asarray(perts)
        degr = perts - nom
        print(f"{cname:<18}{nom:>8.2f} {np.mean(perts):>10.2f} {np.mean(degr):>+8.2f} "
              f"{np.std(degr):>8.2f} {np.max(degr):>+7.2f}")
    print("(RMSE [cm]; degr. = perturbed - nominal, mean over samples)")


if __name__ == "__main__":
    main()
