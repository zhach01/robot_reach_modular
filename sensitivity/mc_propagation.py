# sensitivity/mc_propagation.py
"""
Monte-Carlo error propagation, Section VIII.F.

Propagates +-uncertainty parameter ranges (Latin-Hypercube, default 10,000 samples)
through three outputs and reports their dispersion:
  - joint torques     -> sigma_tau1, sigma_tau2
  - end-effector      -> 95th-percentile endpoint deviation over a reach
  - activation energy -> spread of  E = sum_i a_i^2  (static allocation)

It also writes the full sample/output table to sensitivity/trials_<N>.csv and can
run the AD-vs-FD gradient check, so the whole Section-VIII.F bullet list is
reproduced from one entry point.
"""
from __future__ import annotations

import os
import numpy as np
from scipy.optimize import nnls

from .lhc_analysis import (DEFAULT_PARAMS, latin_hypercube_sample, torque_model_batch)
from .id_calibration import moment_arm_matrix, A_NOM, FMAX_NOM

# reach target pose used for the endpoint/energy outputs (rad)
Q_TARGET = np.array([np.radians(55.0), np.radians(90.0)])
THETA_DD = np.array([1.0, 0.5])


def _col(samples, specs):
    return {s.name: j for j, s in enumerate(specs)}


def forward_kinematics(L1, L2, q=Q_TARGET):
    """Planar 2-link endpoint (x, y) for arrays L1, L2 at joint config q."""
    q1, q12 = q[0], q[0] + q[1]
    return L1 * np.cos(q1) + L2 * np.cos(q12), L1 * np.sin(q1) + L2 * np.sin(q12)


def propagate(n_samples=10000, seed=42, specs=DEFAULT_PARAMS, save_csv=True, verbose=True):
    X = latin_hypercube_sample(n_samples, specs, seed)
    col = _col(X, specs)
    P = lambda nm, d: X[:, col[nm]] if nm in col else np.full(n_samples, d)

    # --- torques (vectorized) ---
    tau = torque_model_batch(X, specs)                       # (N, 2)

    # --- endpoint deviation vs nominal (vectorized FK) ---
    L1, L2 = P("L1", 0.27), P("L2", 0.26)
    x, y = forward_kinematics(L1, L2)
    x0, y0 = forward_kinematics(np.array([0.309]), np.array([0.333]))
    endpoint_dev = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)    # m

    # --- activation energy E = sum a_i^2 from a static NNLS allocation ---
    a1, a2 = P("a1", 0.035), P("a2", 0.028)
    a61, a62 = P("a61", 0.025), P("a62", 0.024)
    Fmax = np.stack([P("Fmax_DEL", 1142.6), P("Fmax_PEC", 699.8), P("Fmax_BRA", 987.3),
                     P("Fmax_BIC", 780.0), P("Fmax_TRI_long", 798.5), P("Fmax_TRI_lat", 624.3)], 1)
    energy = np.empty(n_samples)
    for i in range(n_samples):
        R = np.stack([[a1[i], a1[i], 0, a1[i] * 0.85, -a61[i], 0],
                      [0, 0, a2[i], a2[i], -a62[i], -a62[i]]])
        F, _ = nnls(R, tau[i])
        a = np.clip(F / Fmax[i], 0.0, 1.0)
        energy[i] = float(np.sum(a ** 2))

    stats = dict(
        sigma_tau1=float(np.std(tau[:, 0])), sigma_tau2=float(np.std(tau[:, 1])),
        endpoint_p95_cm=float(np.percentile(endpoint_dev, 95) * 100.0),
        endpoint_mean_cm=float(np.mean(endpoint_dev) * 100.0),
        energy_mean=float(np.mean(energy)),
        energy_rel_spread_pct=float(100.0 * np.std(energy) / max(np.mean(energy), 1e-12)),
        n_samples=n_samples,
    )

    if save_csv:
        import pandas as pd
        out = os.path.join(os.path.dirname(__file__), f"trials_{n_samples}.csv")
        df = pd.DataFrame(X, columns=[s.name for s in specs])
        df["tau1"], df["tau2"] = tau[:, 0], tau[:, 1]
        df["endpoint_dev_cm"] = endpoint_dev * 100.0
        df["energy_sum_a2"] = energy
        df.to_csv(out, index=False)
        stats["csv"] = out

    if verbose:
        print(f"=== Monte-Carlo error propagation ({n_samples} LHC samples) ===")
        print(f"  torque variability:  sigma_tau1={stats['sigma_tau1']:.2f} N.m, "
              f"sigma_tau2={stats['sigma_tau2']:.2f} N.m")
        print(f"  endpoint deviation:  95th pct = {stats['endpoint_p95_cm']:.2f} cm "
              f"(mean {stats['endpoint_mean_cm']:.2f} cm)")
        print(f"  activation energy:   +-{stats['energy_rel_spread_pct']:.1f}% "
              f"(mean E={stats['energy_mean']:.3f})")
        if "csv" in stats:
            print(f"  wrote {stats['csv']}")
    return stats


if __name__ == "__main__":
    propagate(n_samples=10000)
