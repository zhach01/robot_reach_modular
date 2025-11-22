import numpy as np
from dataclasses import dataclass
from utils.linear_utils import nnls_small_active_set


# implements [3a],[3b],[3c]
@dataclass
class MuscleGuardParams:
    lam_mus_min: float = 0.0
    lam_mus_max: float = 1e-2
    k_mus: float = 50.0
    g_thresh: float = 1e-6
    eps: float = 1e-9
    # choose weighting policy: "normalized" (F/Fmax) or "absolute"
    weighting: str = "normalized"


def build_weight(Fmax_vec: np.ndarray, policy: str):
    if policy == "normalized":
        return np.diag(np.maximum(Fmax_vec, 1e-9) ** 2)  # penalize normalized forces
    elif policy == "absolute":
        return np.eye(Fmax_vec.shape[0])
    else:
        raise ValueError("Unknown weighting policy")


def stable_sigmoid(x):
    # avoids overflow/underflow
    # σ(x) = 0.5 * (1 + tanh(x/2)) is numerically safer
    return 0.5 * (1.0 + np.tanh(0.5 * np.clip(x, -80.0, 80.0)))


def solve_muscle_forces(
    tau_des: np.ndarray,
    R: np.ndarray,
    Fmax_vec: np.ndarray,
    eta: float,
    P: MuscleGuardParams,
):
    A = -R
    W = build_weight(Fmax_vec, P.weighting)
    Gp = A @ W @ A.T
    sminGp = float(np.linalg.svd(Gp, compute_uv=False)[-1])
    arg = P.k_mus * (sminGp - P.g_thresh)
    gate = stable_sigmoid(arg)

    lam_mus = P.lam_mus_min + (P.lam_mus_max - P.lam_mus_min) / (1.0 + np.exp(gate))
    lam_mus = lam_mus + (1.0 - eta) * 1e-3

    F_p = (
        W @ A.T @ np.linalg.solve(Gp + (P.eps + lam_mus) * np.eye(Gp.shape[0]), tau_des)
    )

    if (sminGp < 1e-8) or np.any(F_p < 0):
        F_des = nnls_small_active_set(tau_des, R)
        negFp = None
    else:
        F_des = F_p
        negFp = int(np.count_nonzero(F_p < 0))
    return F_des, {"sminGp": sminGp, "lam_mus": lam_mus, "neg_Fp": negFp}
