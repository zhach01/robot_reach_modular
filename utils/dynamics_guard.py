import numpy as np
from dataclasses import dataclass


# implements [2a],[2b],[2c] + gate [4]
@dataclass
class DynGuardParams:
    sigma_thresh_S: float = 1e-3
    vol_thresh: float = 1e-6
    c_s: float = 1e-3
    c_v: float = 1e-3
    k_scale_dyn: float = 1e-3
    lam_boost: float = 1e-3
    gate_pow: float = 2.0
    eps: float = 1e-9
    lam_os_max: float = 1e-1  # upper bound safety


def op_space_guard_and_gate(
    S: np.ndarray, xd_d: np.ndarray, xdd_d: np.ndarray, P: DynGuardParams
):
    # eigs for 2x2 SPD
    evals = np.linalg.eigvalsh(S)
    sminS = float(max(evals[0], 0.0))
    detS = float(max(evals[0] * evals[1], 0.0))

    # gate (eta) from smin(S)
    eta = float(np.clip(sminS / max(P.sigma_thresh_S, 1e-12), 0.0, 1.0))
    eta2 = eta**P.gate_pow

    # dynamic task scaling
    alpha_S = min(1.0, sminS / (sminS + P.k_scale_dyn))
    xd_sc = alpha_S * xd_d
    xdd_sc = alpha_S * xdd_d

    # adaptive Λ regularization
    lam_os = P.c_s * (P.sigma_thresh_S / (sminS + 1e-12)) + P.c_v * (
        P.vol_thresh / (detS + 1e-18)
    )
    lam_os = float(np.clip(lam_os, 0.0, P.lam_os_max))
    lam_os = lam_os + P.lam_boost * (1.0 - eta)

    G_OS = S + (lam_os + P.eps) * np.eye(2)
    # you can return Lambda explicitly or keep G_OS (better numerically to keep G_OS and call solve)
    Lambda = np.linalg.solve(G_OS, np.eye(2))
    return (
        Lambda,
        lam_os,
        eta,
        eta2,
        xd_sc,
        xdd_sc,
        {"sminS": sminS, "detS": detS, "alpha_S": alpha_S},
    )
