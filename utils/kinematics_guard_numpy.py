import numpy as np
from dataclasses import dataclass
from model_lib.skeleton_numpy import (
    geometricJacobian_cached,
)  # only for grad; you can inject if you prefer


# implements [1a],[1b],[1c]
@dataclass
class KinGuardParams:
    lam_min: float = 1e-6
    lam_max: float = 1e-1
    k_sig: float = 30.0
    sigma_thresh_J: float = 5e-2
    k_scale_J: float = 5e-2
    k_manip: float = 0.1
    grad_eps: float = 1e-5
    enable_manip_nullspace: bool = True


def adaptive_dls_pinv(J_xy: np.ndarray, n: int, P: KinGuardParams):
    U, s, Vt = np.linalg.svd(J_xy, full_matrices=False)
    sminJ = float(s[-1])
    lamJ = P.lam_min + (P.lam_max - P.lam_min) / (
        1.0 + np.exp(P.k_sig * (sminJ - P.sigma_thresh_J))
    )
    JTJ = J_xy.T @ J_xy
    J_dls = np.linalg.solve(JTJ + (lamJ**2) * np.eye(n), J_xy.T)
    return J_dls, sminJ, lamJ


def scale_task_by_J(
    xd_d: np.ndarray, xdd_d: np.ndarray, sminJ: float, P: KinGuardParams
):
    alpha_J = min(1.0, sminJ / (sminJ + P.k_scale_J))
    return alpha_J * xd_d, alpha_J * xdd_d, alpha_J


def _yoshikawa_w(J_xy: np.ndarray) -> float:
    M = J_xy @ J_xy.T
    return float(np.sqrt(max(np.linalg.det(M), 0.0)))


def manip_grad_numeric(env, q, qd, J_xy, eps: float):
    base = max(_yoshikawa_w(J_xy), 1e-12)
    g = np.zeros_like(q)
    for i in range(q.shape[0]):
        dq = np.zeros_like(q)
        dq[i] = eps
        env.skeleton._set_state(q + dq, qd)
        Jp = geometricJacobian_cached(env.skeleton._robot, symbolic=False)[0:2, :]
        wp = max(_yoshikawa_w(Jp), 1e-12)
        g[i] = (np.log(wp) - np.log(base)) / eps
    env.skeleton._set_state(q, qd)
    return g


def add_nullspace_manip(
    qdd_post: np.ndarray, env, q, qd, J_xy, P: KinGuardParams, eta: float
):
    if not P.enable_manip_nullspace:
        return qdd_post, 0.0
    k = P.k_manip * eta  # fade with gate if you want
    g = manip_grad_numeric(env, q, qd, J_xy, P.grad_eps)
    return qdd_post + k * g, k
