"""
Pure Muscle Synergy Controller (STRICT)

Commanded activations are constrained to the synergy subspace:
    a(t) = clip(W c(t)),  c(t) >= 0

We compute a full-space "reference" activation a_ref (from tau* via the
project's muscle solver), then project it onto span(W) using NNLS:
    c = argmin_{c>=0} ||W c - a_ref||^2

No residual correction, no blending to full-space a_ref.
This is intentionally "pure" (and may fail if W is not good).
"""

from dataclasses import dataclass
import numpy as np

from model_lib.numpy.skeleton import (
    geometricJacobian_cached,
    inertiaMatrixCOM_cached,
    gravityCOM_cached,
)

from utils.numpy.kinematics_guard import KinGuardParams, adaptive_dls_pinv
from utils.numpy.dynamics_guard import DynGuardParams, op_space_guard_and_gate
from utils.numpy.muscle_guard import MuscleGuardParams, solve_muscle_forces
from utils.numpy.telemetry import pack_diag, merge_diag

from muscles.numpy.muscle_tools import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
)


# ----------------------------
# NNLS via projected gradient
# ----------------------------

def nnls_pgd(W: np.ndarray, a: np.ndarray, iters: int = 60) -> np.ndarray:
    """
    Solve min_{c>=0} ||Wc - a||^2 using projected gradient descent.
    Good enough for small K (2-4) and runs fast.
    """
    W = np.asarray(W, dtype=float)
    a = np.asarray(a, dtype=float).ravel()
    K = W.shape[1]

    c = np.zeros(K, dtype=float)

    # step ~ 1/L with L = ||W||_2^2
    try:
        L = (np.linalg.norm(W, 2) ** 2) + 1e-12
    except Exception:
        L = (np.sum(W * W) + 1e-12)
    lr = 1.0 / L

    WT = W.T
    for _ in range(iters):
        grad = 2.0 * (WT @ (W @ c - a))
        c = c - lr * grad
        c = np.maximum(c, 0.0)
    return c


# ----------------------------
# Params
# ----------------------------

@dataclass
class SynergyPureParams:
    # Task PD (in task space, x-y)
    Kp_task: float = 800.0
    Kv_task: float = 60.0
    Kff_x: float = 1.0

    # Guards
    eps: float = 1e-6
    lam_os_max: float = 1e6
    sigma_thresh: float = 1e-4
    gate_pow: float = 2.0

    # Dynamics compensation toggles
    enable_gravity_comp: bool = True
    enable_velocity_comp: bool = False   # not used here unless you add C(q,qd)
    enable_inertia_comp: bool = False    # not used here (kept for symmetry)

    # Synergy config
    W_init: np.ndarray | None = None     # (6,K)
    n_synergies: int = 3

    # Limits
    c_max: float = 5.0
    nnls_iters: int = 60

    # Muscle inversion
    bisect_iters: int = 12


# ----------------------------
# Controller
# ----------------------------

class SynergyPureController:
    def __init__(self, env, arm, params: SynergyPureParams):
        self.env = env
        self.arm = arm
        self.p = params

        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=self.p.eps,
            lam_os_max=self.p.lam_os_max,
            gate_pow=self.p.gate_pow,
            sigma_thresh_S=max(self.p.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()

        self.qref = None

        if self.p.W_init is None:
            # simple default (will usually NOT be good enough)
            K = int(self.p.n_synergies)
            W = np.abs(np.random.randn(6, K))
            W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
            self.W = W
        else:
            self.W = np.asarray(self.p.W_init, dtype=float)

    def reset(self, q0: np.ndarray):
        self.qref = q0.copy()

    # --------- Save / Load ---------

    def save(self, path: str):
        np.savez(path, W=self.W)

    def load(self, path: str):
        z = np.load(path, allow_pickle=True)
        self.W = np.asarray(z["W"], dtype=float)

    def set_W(self, W: np.ndarray):
        self.W = np.asarray(W, dtype=float)

    def get_W(self) -> np.ndarray:
        return self.W.copy()

    # --------- Compute ---------

    def compute(self, x_d: np.ndarray, xd_d: np.ndarray, xdd_d: np.ndarray) -> dict:
        # State
        joint = self.env.states["joint"][0]
        q, qd = joint[:2], joint[2:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:]
        self.env.skeleton._set_state(q, qd)

        # Jacobian
        J = geometricJacobian_cached(self.env.skeleton._robot, symbolic=False)
        J_xy = J[0:2, :]
        n = q.shape[0]

        # Kinematic guard
        _, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)

        # Dynamics guard (for eta gating)
        M = inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False)
        Minv = np.linalg.inv(M)
        S = J_xy @ Minv @ J_xy.T
        _, lam_os, eta, eta2, xd_d_g, xdd_d_g, dyn_diag = op_space_guard_and_gate(
            S, xd_d.copy(), xdd_d.copy(), self.dp
        )

        # Task PD force (simple + robust)
        e_x = x_d - x
        e_v = xd_d_g - xd
        F_task = self.p.Kp_task * e_x + self.p.Kv_task * e_v + (self.p.Kff_x * xdd_d_g)

        # Convert to desired joint torque
        tau_task = J_xy.T @ F_task

        # Optional gravity compensation (recommended True)
        if self.p.enable_gravity_comp:
            g = gravityCOM_cached(
                self.env.skeleton._robot,
                self.env.skeleton._gravity_vec,
                symbolic=False,
            ).reshape(-1)
            tau_task = tau_task + g

        # Muscle geometry
        geom = self.env.states["geometry"]
        lenvel = geom[:, :2, :]                  # (B,2,nm)
        R = geom[:, 2:2 + self.env.skeleton.dof, :][0]  # (dof,nm)
        Fmax_vec = get_Fmax_vec(self.env, R.shape[1])

        # Full-space force solve → reference activation a_ref
        F_ref, mus_diag = solve_muscle_forces(tau_task, R, Fmax_vec, eta, self.mp)

        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]

        a_ref = force_to_activation_bisect(
            F_ref, lenvel, self.env.muscle, flpe, Fmax_vec,
            iters=self.p.bisect_iters,
        )

        # PURE synergy projection: a = W c
        c = nnls_pgd(self.W, a_ref, iters=int(self.p.nnls_iters))
        c = np.clip(c, 0.0, float(self.p.c_max))

        a_cmd = self.W @ c
        a_cmd = np.clip(a_cmd, float(self.env.muscle.min_activation), 1.0)

        # Predicted forces / torques (for logging)
        af = active_force_from_activation(a_cmd, lenvel, self.env.muscle)
        F_cmd = Fmax_vec * (af + flpe)
        F_cmd = np.maximum(F_cmd, 0.0)
        tau_cmd = -R @ F_cmd

        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ)
        diag = merge_diag(
            kin_diag, dyn_diag, mus_diag,
            pack_diag(lam_os=lam_os, eta=eta, eta2=eta2,
                      c_max=float(np.max(c)), a_ref_mean=float(np.mean(a_ref)))
        )

        return {
            "tau_des": tau_cmd,
            "R": R,
            "Fmax": Fmax_vec,
            "F_des": F_cmd,
            "act": a_cmd,
            "synergy_coeff": c,
            "q": q,
            "qd": qd,
            "x": x,
            "xd": xd,
            "xref_tuple": (x_d, xd_d_g, xdd_d_g),
            "eta": eta2,
            "diag": diag,
        }

