# sensitivity/id_calibration.py
"""
Inverse-dynamics calibration of the lever map (optional refinement), Section VIII.C.

Pipeline:
  1. A short kinematic trial (synthetic reaches here) gives joint states
     theta, theta_dot, theta_ddot, from which inverse-dynamics torques are formed
       tau_ID = M(theta) theta_ddot + C(theta,theta_dot) theta_dot + D theta_dot
     (g = 0 in the planar model).
  2. Inner convex allocation: at each instant resolve muscle forces by a
     non-negative least squares with Tikhonov regularization
       min_{Fm >= 0}  || R(theta, p) Fm - tau_ID ||^2 + lambda ||Fm||^2 .
  3. Outer search: adjust a low-dimensional set of lever-map scalings s_i (and a
     selected Fmax scale) with CMA-ES to minimize the summed residual
       min_{s, Fmax}  sum_t || R(theta_t, s) Fm*_t - tau_ID_t ||^2 .

We do NOT replace the angle-dependent moment arms by fixed lengths; only their
per-muscle scalings are optimized. This script demonstrates recovery: data are
generated with a known perturbed lever map, and the calibration drives the residual
down and recovers the scalings from the nominal start.

Optimizer: CMA-ES (Hansen 2016), via the `cma` package.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
import cma

# base (nominal) lever constants and Fmax, consistent with lhc_analysis.torque_model
A_NOM = np.array([0.035, 0.028, 0.025, 0.024])          # a1, a2, a61, a62
FMAX_NOM = np.array([1142.6, 699.8, 987.3, 780.0, 798.5, 624.3])
LAMBDA = 1e-3


def moment_arm_matrix(scal):
    """R(2x6) with per-lever scalings scal=[s_a1,s_a2,s_a61,s_a62] (Section VI fits)."""
    a1, a2, a61, a62 = A_NOM * scal
    R0 = np.array([a1, a1, 0.0, a1 * 0.85, -a61, 0.0])
    R1 = np.array([0.0, 0.0, a2, a2, -a62, -a62])
    return np.stack([R0, R1])


def mass_matrix(theta, m1=1.82, m2=1.44, L1=0.27, Lg1=0.135, Lg2=0.165, I1=0.012, I2=0.018):
    c2 = np.cos(theta[1])
    M11 = I1 + I2 + m1 * Lg1**2 + m2 * (L1**2 + Lg2**2 + 2 * L1 * Lg2 * c2)
    M12 = I2 + m2 * (Lg2**2 + L1 * Lg2 * c2)
    M22 = I2 + m2 * Lg2**2
    return np.array([[M11, M12], [M12, M22]])


def inverse_dynamics(theta, thd, thdd, D=np.diag([0.5, 0.3])):
    """tau_ID = M thdd + C thd + D thd (planar, g=0)."""
    M = mass_matrix(theta)
    m2, L1, Lg2 = 1.44, 0.27, 0.165
    h = -m2 * L1 * Lg2 * np.sin(theta[1])
    C = np.array([h * thd[1] * (2 * thd[0] + thd[1]), -h * thd[0] ** 2])
    return M @ thdd + C + D @ thd


def make_trial(n=120, seed=0):
    """Synthetic short reach trial: smooth joint trajectories -> (theta, thd, thdd)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    states = []
    for _ in range(5):                                   # ~5 reaches
        a = rng.uniform(-0.4, 0.4, 2); b = rng.uniform(0.3, 0.9, 2)
        ph = rng.uniform(0, np.pi, 2); w = rng.uniform(2, 5, 2)
        for tt in t:
            s = np.sin(w * tt + ph)
            th = np.array([0.6, 0.9]) + a * (1 - np.cos(w * tt))
            thd = a * w * np.sin(w * tt)
            thdd = a * w * w * np.cos(w * tt)
            states.append((th, thd, thdd))
    return states


def nnls_tikhonov(R, tau, lam=LAMBDA):
    """min_{F>=0} ||R F - tau||^2 + lam ||F||^2  via augmented NNLS."""
    A = np.vstack([R, np.sqrt(lam) * np.eye(R.shape[1])])
    b = np.concatenate([tau, np.zeros(R.shape[1])])
    F, _ = nnls(A, b)
    return F


def residual(scal, trial, act_meas, tau_meas, fmax=FMAX_NOM):
    """Summed torque residual for lever scalings `scal` against the measured
    activation pattern (fixed) and the measured inverse-dynamics torques."""
    tot = 0.0
    for (th, thd, thdd), a, tau in zip(trial, act_meas, tau_meas):
        R = moment_arm_matrix(scal)
        tot += float(np.sum((R @ (a * fmax) - tau) ** 2))
    return tot


def main():
    trial = make_trial()
    # ground-truth perturbed lever map used to synthesize the measured data
    s_true = np.array([1.12, 0.91, 1.06, 0.95])
    R_true = moment_arm_matrix(s_true)
    act_meas, tau_meas = [], []
    for (th, thd, thdd) in trial:
        tau_id = inverse_dynamics(th, thd, thdd)
        F = nnls_tikhonov(R_true, tau_id)        # inner allocation -> measured activations
        act_meas.append(F / FMAX_NOM)            # "measured EMG" activation pattern
        tau_meas.append(R_true @ F)              # measured joint torque

    x0 = np.array([1.0, 1.0, 1.0, 1.0])          # nominal start: all lever scalings = 1
    r0 = residual(x0, trial, act_meas, tau_meas)
    print("=== Inverse-dynamics lever-map calibration (CMA-ES) ===")
    print(f"trial: {len(trial)} samples (~5 reaches) | inner: NNLS+Tikhonov (lambda={LAMBDA})")
    print(f"initial residual (scalings=1): {r0:.4e}")

    es = cma.CMAEvolutionStrategy(x0, 0.1, {"bounds": [[0.6] * 4, [1.4] * 4],
                                            "seed": 1, "verbose": -9, "maxiter": 200})
    es.optimize(lambda x: residual(x, trial, act_meas, tau_meas))
    xbest, rbest = es.result.xbest, es.result.fbest
    print(f"final residual:                {rbest:.4e}  ({100*(1-rbest/max(r0,1e-12)):.1f}% reduction)")
    print(f"recovered lever scalings: {np.round(xbest, 3)}")
    print(f"ground-truth scalings:    {s_true}")
    print(f"max |recovered - true|:   {np.max(np.abs(xbest - s_true)):.3f}")
    return xbest, rbest, r0


if __name__ == "__main__":
    main()
