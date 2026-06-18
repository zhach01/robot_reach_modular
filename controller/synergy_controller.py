"""
Muscle Synergy Controller (robust + literature-aligned)

Core idea:
  a(t) ≈ W c(t),  W >= 0 fixed "modules", c(t) >= 0 low-dimensional coefficients.

Practical control structure used here (robust + works in your sim):
  1) Compute a task-level torque target τ_cmd (OSC/PD) with the same guards used elsewhere.
  2) Compute a synergy-based activation a_syn = clip(W c, [a_min, 1]).
     - c is found by NNLS projection of the "optimal" activation (from τ_cmd) into synergy space.
  3) Compute residual torque τ_res = τ_cmd - τ(a_syn), and solve a correction force via solve_muscle_forces.
  4) Convert corrected forces to activations via force_to_activation_bisect + saturation repair.

Why this fixes "no tracking":
  - If W is imperfect, the residual correction recovers tracking.
  - If W is good, residual is small and you behave like true synergy control.

This is consistent with a common interpretation in the synergy literature:
  "modules + modulation + additional correction/fine tuning".
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

from model_lib.skeleton_numpy import (
    geometricJacobian_cached,
    geometricJacobianDot_cached,
    inertiaMatrixCOM_cached,
    centrifugalCoriolisCOM_cached,
    gravityCOM_cached,
)

from utils.kinematics_guard import KinGuardParams, adaptive_dls_pinv, scale_task_by_J
from utils.dynamics_guard import DynGuardParams, op_space_guard_and_gate
from utils.muscle_guard import MuscleGuardParams, solve_muscle_forces
from utils.telemetry import pack_diag, merge_diag

from muscles.muscle_tools import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
)

# =============================================================================
# Default W (heuristic) - must match your muscle order to be meaningful
# =============================================================================

def get_default_synergies() -> np.ndarray:
    """
    Heuristic default synergy patterns for a 6-muscle Arm26-like model.

    IMPORTANT:
      If your actual env muscle ordering differs, this W is wrong.
      This controller remains stable because of residual correction, but synergy
      meaning will be weak until W is learned/extracted for your ordering.
    """
    W = np.array([
        # S1: flexor-biased
        [0.91, 0.85, 0.88, 0.05, 0.08, 0.35],
        # S2: extensor/abductor-biased
        [0.08, 0.12, 0.05, 0.85, 0.79, 0.81],
        # S3: co-contraction / stabilization
        [0.45, 0.40, 0.42, 0.38, 0.35, 0.50],
    ]).T  # (6,3)

    W = np.maximum(W, 0.0)

    # Use max-normalization (NOT L2) so coefficients in ~[0,1..3] have sensible scale
    W = W / (np.max(W, axis=0, keepdims=True) + 1e-12)
    return W


# =============================================================================
# Tiny NNLS for small K (active-set enumeration)
# =============================================================================

def _nnls_small(A: np.ndarray, b: np.ndarray, reg: float = 1e-10) -> np.ndarray:
    """
    Solve min ||A x - b||^2 + reg||x||^2  s.t. x >= 0 for small K.
    Brute-force active-set enumeration (2^K subsets). K<=6 is fine.
    """
    m, K = A.shape
    best_x = np.zeros(K)
    best_cost = np.inf

    AtA = A.T @ A
    Atb = A.T @ b

    for mask in range(1 << K):
        active = [i for i in range(K) if (mask >> i) & 1]
        if len(active) == 0:
            cost = float(np.linalg.norm(b) ** 2)
            if cost < best_cost:
                best_cost = cost
                best_x = np.zeros(K)
            continue

        idx = np.array(active, dtype=int)
        M = AtA[np.ix_(idx, idx)] + reg * np.eye(len(active))
        y = Atb[idx]
        try:
            x_act = np.linalg.solve(M, y)
        except np.linalg.LinAlgError:
            continue
        if np.any(x_act < -1e-12):
            continue

        x = np.zeros(K)
        x[idx] = np.maximum(x_act, 0.0)
        r = A @ x - b
        cost = float(r.T @ r + reg * (x.T @ x))
        if cost < best_cost:
            best_cost = cost
            best_x = x

    return best_x


# =============================================================================
# Params
# =============================================================================

@dataclass
class SynergyParams:
    # Synergy structure
    W_preset: Optional[np.ndarray] = None    # (6,K)
    n_synergies: Optional[int] = None
    c_max: float = 3.0                      # allow >1 so Wc can reach full activation

    # Optional prior mapping c0 = clip(B z, 0, c_max), z=[e_x,e_y,e_vx,e_vy,1]
    B_preset: Optional[np.ndarray] = None    # (K,5)
    use_prior: bool = False                 # OFF by default (safer)
    prior_weight: float = 1e-2              # small; 0 disables

    # Task control (OSC/PD)
    Kp_task: float = 800.0
    Kv_task: float = 60.0
    Kff_x: float = 1.0

    # Compensation toggles
    enable_inertia_comp: bool = True
    enable_gravity_comp: bool = True
    enable_velocity_comp: bool = True

    # Guards
    eps: float = 1e-6
    lam_os_max: float = 1e6
    sigma_thresh: float = 1e-4
    gate_pow: float = 2.0

    # Muscle inversion
    bisect_iters: int = 12

    # Synergy strength knob (0 -> purely optimal allocation, 1 -> synergy-first)
    synergy_strength: float = 0.7

    # Optional online learning of B via RLS
    online_B_adapt: bool = False
    rls_lambda: float = 0.995
    rls_delta: float = 100.0


# =============================================================================
# Controller
# =============================================================================

class SynergyController:
    def __init__(self, env, arm, params: SynergyParams):
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

        # W
        if self.p.W_preset is not None:
            W = np.array(self.p.W_preset, dtype=float)
        else:
            W = get_default_synergies()

        W = np.maximum(W, 0.0)
        W = W / (np.max(W, axis=0, keepdims=True) + 1e-12)

        if self.p.n_synergies is not None:
            W = W[:, : int(self.p.n_synergies)]

        self.W = W
        self.K = self.W.shape[1]
        self.n_muscles = self.W.shape[0]

        # B prior (K,5)
        if self.p.B_preset is not None:
            B = np.array(self.p.B_preset, dtype=float)
        else:
            B = np.zeros((self.K, 5), dtype=float)
            for k in range(self.K):
                B[k, 0] = 0.4  # e_x
                B[k, 1] = 0.4  # e_y
                B[k, 2] = 0.1  # e_vx
                B[k, 3] = 0.1  # e_vy
                B[k, 4] = 0.05 # bias
        self.B = B[: self.K, :]

        # RLS state
        self._rls_dim = 5
        self._rls_P = (1.0 / max(self.p.rls_delta, 1e-9)) * np.eye(self._rls_dim)

    def reset(self, q0: np.ndarray):
        self.qref = q0.copy()

    # --------------------------
    # Save / load
    # --------------------------
    def save(self, path: str):
        np.savez(path, W=self.W, B=self.B)

    def load(self, path: str):
        d = np.load(path, allow_pickle=True)
        self.W = np.array(d["W"], dtype=float)
        self.B = np.array(d["B"], dtype=float)
        self.K = self.W.shape[1]
        self.n_muscles = self.W.shape[0]

    # --------------------------
    # RLS update of B (optional)
    # --------------------------
    def _rls_update_B(self, z: np.ndarray, c_target: np.ndarray):
        lam = float(self.p.rls_lambda)
        z = z.reshape(-1, 1)  # (d,1)

        P = self._rls_P
        denom = lam + float(z.T @ P @ z)
        K_gain = (P @ z) / max(denom, 1e-12)

        for k in range(self.K):
            y_hat = float(self.B[k, :].reshape(1, -1) @ z)
            err = float(c_target[k] - y_hat)
            self.B[k, :] = self.B[k, :] + (K_gain[:, 0] * err)

        self._rls_P = (P - (K_gain @ z.T @ P)) / max(lam, 1e-6)

    # --------------------------
    # Main compute
    # --------------------------
    def compute(self, x_d: np.ndarray, xd_d: np.ndarray, xdd_d: np.ndarray) -> dict:
        # State
        joint = self.env.states["joint"][0]
        q, qd = joint[:2], joint[2:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:]
        self.env.skeleton._set_state(q, qd)
        if self.qref is None:
            self.qref = q.copy()

        # Kinematics
        J = geometricJacobian_cached(self.env.skeleton._robot, symbolic=False)
        J_xy = J[0:2, :]
        Jdot = geometricJacobianDot_cached(self.env.skeleton._robot, symbolic=False)
        Jdot_xy = Jdot[0:2, :]

        _, sminJ, lamJ = adaptive_dls_pinv(J_xy, 2, self.kp)
        xd_d_s, xdd_d_s, alpha_J = scale_task_by_J(
            xd_d.copy(), xdd_d.copy(), sminJ=sminJ, P=self.kp
        )

        # Dynamics + op-space guard
        M = inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False)
        Minv = np.linalg.inv(M)
        S = J_xy @ Minv @ J_xy.T

        Lambda_reg, lam_os, eta, eta2, xd_d_g, xdd_d_g, dyn_diag = op_space_guard_and_gate(
            S, xd_d_s.copy(), xdd_d_s.copy(), self.dp
        )

        # Errors (textbook tracking)
        e_x = x_d - x
        e_v = xd_d_g - xd

        # Nonlinear terms h
        h = np.zeros_like(q)
        if self.p.enable_velocity_comp:
            C = centrifugalCoriolisCOM_cached(self.env.skeleton._robot, symbolic=False)
            C = np.asarray(C)
            h = h + (C @ qd if C.ndim == 2 else C.reshape(-1))
        if self.p.enable_gravity_comp:
            g = gravityCOM_cached(
                self.env.skeleton._robot, self.env.skeleton._gravity_vec, symbolic=False
            ).reshape(-1)
            h = h + g

        # OSC/PD torque target
        Kp = float(self.p.Kp_task)
        Kv = float(self.p.Kv_task)
        xdd_cmd = (self.p.Kff_x * xdd_d_g) + (Kp * e_x) + (Kv * e_v)

        if self.p.enable_inertia_comp:
            mu = Lambda_reg @ (J_xy @ (Minv @ h)) - Lambda_reg @ (Jdot_xy @ qd)
            F_task = (Lambda_reg @ xdd_cmd) + mu
            tau_cmd = (J_xy.T @ F_task).reshape(-1)
        else:
            F_task = (Kp * e_x) + (Kv * e_v)
            tau_cmd = (J_xy.T @ F_task).reshape(-1)
            if self.p.enable_gravity_comp:
                tau_cmd = tau_cmd + h.reshape(-1)

        # Gate torque by authority eta
        tau_cmd = tau_cmd * float(np.clip(eta, 0.0, 1.0))

        # Muscle geometry
        geom = self.env.states["geometry"]
        lenvel = geom[:, :2, :]                      # (B,2,nm)
        R = geom[0, 2:2 + self.env.skeleton.dof, :]  # (dof,nm)

        Fmax_vec = get_Fmax_vec(self.env, R.shape[1])
        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]

        a_min = float(getattr(self.env.muscle, "min_activation", 0.0))

        # ------------------------------------------------------------------
        # STEP 1: Get "optimal" feasible muscle forces for tau_cmd (full space)
        # ------------------------------------------------------------------
        F_opt, mus_diag = solve_muscle_forces(tau_cmd, R, Fmax_vec, eta, self.mp)

        # Convert to activation (optimal)
        a_opt = force_to_activation_bisect(
            F_opt, lenvel, self.env.muscle, flpe, Fmax_vec,
            iters=self.p.bisect_iters
        )
        a_opt = np.clip(a_opt, a_min, 1.0)

        # ------------------------------------------------------------------
        # STEP 2: Project optimal activation into synergy space: a_syn ≈ W c
        # ------------------------------------------------------------------
        z = np.array([e_x[0], e_x[1], e_v[0], e_v[1], 1.0], dtype=float)
        c0 = np.clip(self.B @ z, 0.0, self.p.c_max)

        if self.p.use_prior and self.p.prior_weight > 0.0:
            w = float(self.p.prior_weight)
            A_aug = np.vstack([self.W, np.sqrt(w) * np.eye(self.K)])
            b_aug = np.concatenate([a_opt, np.sqrt(w) * c0])
            c = _nnls_small(A_aug, b_aug, reg=1e-10)
        else:
            c = _nnls_small(self.W, a_opt, reg=1e-10)

        c = np.clip(c, 0.0, self.p.c_max)

        if self.p.online_B_adapt:
            self._rls_update_B(z, c)

        a_syn = self.W @ c
        a_syn = np.clip(a_syn, a_min, 1.0)

        # ------------------------------------------------------------------
        # STEP 3: Residual correction in torque space (guarantees tracking)
        # ------------------------------------------------------------------
        # Forces from synergy activation
        af_syn = active_force_from_activation(a_syn, lenvel, self.env.muscle)
        F_syn = Fmax_vec * (af_syn + flpe)
        F_syn = np.maximum(F_syn, 0.0)

        tau_syn = (-R @ F_syn).reshape(-1)
        tau_res = (tau_cmd - tau_syn).reshape(-1)

        # Solve residual forces (full muscles) then combine with synergy force
        F_res, mus_diag2 = solve_muscle_forces(tau_res, R, Fmax_vec, eta, self.mp)
        beta = float(np.clip(self.p.synergy_strength, 0.0, 1.0))
        F_mix = (beta * F_syn) + ((1.0 - beta) * F_opt)

        # Add residual correction on top of mixed baseline
        F_final = F_mix + F_res
        F_final = np.maximum(F_final, 0.0)

        # Convert final force -> activation
        a_final = force_to_activation_bisect(
            F_final, lenvel, self.env.muscle, flpe, Fmax_vec,
            iters=self.p.bisect_iters
        )
        a_final = np.clip(a_final, a_min, 1.0)

        # Saturation repair (project back to torque feasibility)
        af_now = active_force_from_activation(a_final, lenvel, self.env.muscle)
        F_pred = Fmax_vec * (af_now + flpe)
        F_pred = np.maximum(F_pred, 0.0)

        F_corr = saturation_repair_tau(
            -R, F_pred, a_final,
            a_min, 1.0,
            Fmax_vec, tau_des=tau_cmd
        )
        if np.any(np.abs(F_corr - F_pred) > 1e-9):
            a_final = force_to_activation_bisect(
                F_corr, lenvel, self.env.muscle, flpe, Fmax_vec,
                iters=max(4, self.p.bisect_iters - 4)
            )
            a_final = np.clip(a_final, a_min, 1.0)
            af_now = active_force_from_activation(a_final, lenvel, self.env.muscle)
            F_pred = Fmax_vec * (af_now + flpe)
            F_pred = np.maximum(F_pred, 0.0)

        tau_out = (-R @ F_pred).reshape(-1)

        # Diagnostics
        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ, alpha_J=alpha_J)
        syn_diag = pack_diag(
            c_mean=float(np.mean(c)),
            c_max=float(np.max(c)) if c.size else 0.0,
            a_syn_mean=float(np.mean(a_syn)),
            a_opt_mean=float(np.mean(a_opt)),
            a_final_mean=float(np.mean(a_final)),
            a_proj_err=float(np.linalg.norm(a_opt - a_syn)),
            tau_cmd_norm=float(np.linalg.norm(tau_cmd)),
            tau_out_norm=float(np.linalg.norm(tau_out)),
            tau_err_norm=float(np.linalg.norm(tau_cmd - tau_out)),
            beta=float(beta),
        )
        diag = merge_diag(
            kin_diag, dyn_diag, mus_diag, mus_diag2, syn_diag,
            pack_diag(lam_os=lam_os, eta=float(eta), eta2=float(eta2))
        )

        return {
            "tau_des": tau_cmd,      # desired torque (task-level)
            "R": R,
            "Fmax": Fmax_vec,
            "F_des": F_pred,         # produced forces
            "act": a_final,          # ACTION sent to env
            "synergy_coeff": c,      # low-dim coefficients
            "q": q,
            "qd": qd,
            "x": x,
            "xd": xd,
            "xref_tuple": (x_d, xd_d_g, xdd_d_g),
            "eta": eta2,
            "diag": diag,
        }

    def get_synergy_matrix(self) -> np.ndarray:
        return self.W.copy()

    def set_synergy_matrix(self, W: np.ndarray):
        W = np.maximum(np.array(W, dtype=float), 0.0)
        W = W / (np.max(W, axis=0, keepdims=True) + 1e-12)
        self.W = W
        self.K = self.W.shape[1]
        self.n_muscles = self.W.shape[0]
        if self.B.shape[0] != self.K:
            self.B = self.B[: self.K, :]

