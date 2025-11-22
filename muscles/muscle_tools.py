import numpy as np
from dataclasses import dataclass
from utils.math_utils import right_pinv_rows_weighted


@dataclass
class MuscleNumerics:
    eps: float = 1e-8
    linesearch_eps: float = 1e-6
    linesearch_safety: float = 0.99
    bisect_iters: int = 22


def get_Fmax_vec(env, M):
    raw = np.asarray(env.muscle.max_iso_force)
    if raw.size == 0:
        return np.ones(M, dtype=float)
    if raw.size == 1:
        return np.full(M, float(raw.reshape(-1)[0]))
    if raw.shape[-1] == M:
        return raw.reshape(-1)[-M:].astype(float)
    return np.full(M, float(np.mean(raw)), dtype=float)


def active_force_from_activation(a, geom_lenvel, muscle):
    a = np.asarray(a, dtype=float).reshape(1, 1, -1)
    d = np.zeros_like(a)
    # before: muscle._integrate(0.0, d, a, geom_lenvel)
    if geom_lenvel.ndim == 2:   # (2, m) -> (1, 2, m)
        geom_lenvel = geom_lenvel[np.newaxis, ...]
    if a.ndim == 1:             # (m,)   -> (1, m)
        a = a[np.newaxis, ...]
    out = muscle._integrate(0.0, d, a, geom_lenvel)
   
    names = muscle.state_name
    ia = names.index("activation")
    ifl = names.index("force-length CE")
    ifv = names.index("force-velocity CE")
    return out[0, ia, :] * out[0, ifl, :] * out[0, ifv, :]


def force_to_activation_bisect(F_des, geom_lenvel, muscle, flpe, Fmax, iters=22):
    F_des = np.asarray(F_des, float)
    if np.ndim(Fmax) == 0:
        Fmax_vec = np.full_like(F_des, float(Fmax))
    else:
        Fmax_vec = np.asarray(Fmax, float)
        if Fmax_vec.shape != F_des.shape:
            Fmax_vec = np.full_like(F_des, float(np.mean(Fmax_vec)))
    target_active = np.maximum(F_des / Fmax_vec - flpe, 0.0)
    lo = np.full_like(target_active, muscle.min_activation)
    hi = np.ones_like(target_active)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        af = active_force_from_activation(mid, geom_lenvel, muscle)
        hi = np.where(af > target_active, mid, hi)
        lo = np.where(af > target_active, lo, mid)
    return np.clip(0.5 * (lo + hi), muscle.min_activation, 1.0)


def saturation_repair_tau(A, F_pred, a, a_min, a_max, Fmax_vec, tau_des=None):
    tau_pred = A @ F_pred
    tau_err = (tau_des - tau_pred) if (tau_des is not None) else -tau_pred
    free = (a > a_min + 1e-4) & (a < a_max - 1e-4)
    if not np.any(free):
        return F_pred
    A_free = A[:, free]
    w = 1.0 / np.maximum(Fmax_vec[free], 1e-9)
    Aw = A_free @ np.diag(w)
    try:
        dw, *_ = np.linalg.lstsq(Aw, -tau_err, rcond=None)
    except np.linalg.LinAlgError:
        return F_pred
    dF_free = np.diag(w) @ dw
    F_new = F_pred.copy()
    F_new[free] = np.maximum(F_pred[free] + dF_free, 0.0)
    return F_new


def apply_internal_force_regulation(
    A,
    F_base,
    F_bias,
    Fmax_vec,
    eps=1e-8,
    linesearch_eps=1e-6,
    linesearch_safety=0.99,
    scale=1.0,
):
    """Minimize ||F - F_bias||_W s.t. AF = A F_base. W = diag(1/Fmax^2)."""
    M = F_base.shape[0]
    W = np.diag(1.0 / np.maximum(Fmax_vec, 1e-9) ** 2)
    Winv = np.diag(np.maximum(Fmax_vec, 1e-9) ** 2)
    A_pinv = right_pinv_rows_weighted(A, Winv, eps)
    N_A = np.eye(M) - A_pinv @ A
    B = N_A.T @ W @ N_A
    rhs = N_A.T @ W @ (scale * (F_bias - F_base))
    try:
        z = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        z, *_ = np.linalg.lstsq(B, rhs, rcond=None)
    dF = N_A @ z
    F_try = F_base + dF
    if np.any(F_try < 0):
        alpha_max = 1.0
        for i in np.where(dF < 0)[0]:
            alpha_i = -F_base[i] / dF[i]
            alpha_max = min(alpha_max, alpha_i)
        alpha = max(0.0, min(1.0, linesearch_safety * alpha_max))
        if alpha <= linesearch_eps:
            return F_base
        F_try = F_base + alpha * dF
    return np.clip(F_try, 0.0, None)
