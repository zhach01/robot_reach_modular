# controller/sliding_mode.py
"""
Sliding Mode Controller (SMC) for 2-DoF musculoskeletal arm (Arm26).

This is the *singularity-robust* version.

Key design choices:
- Switching term is applied in **task-acceleration space** and mapped through Λ_reg
  (regularized operational-space inertia). This prevents τ = Jᵀ F_sw blow-ups near
  kinematic singularities.
- Early blending to a safe joint-space fallback using **cond(J)** (or σ_min(J)),
  not cond(R) alone.
- Optional feasibility scaling before muscle allocation to avoid “impulse then saturate”.

Fixes applied (A–F):
A) enable_inertia_comp now truly toggles Λ/μ usage (Λ=I, μ=0 when disabled)
B) Removed extra alpha_J scaling in qref integration (avoid double scaling)
C) Added dedicated sminJ_thresh for Jacobian gating (separate from dynamics sigma_thresh)
D) Default eta floors set to 0 (can be reintroduced if desired)
E) Robust cond(J) handling (inf/nan guarded)
F) Feasibility scaling applied only when severe (default threshold 0.7)
"""
from dataclasses import dataclass
import numpy as np

from model_lib.skeleton_numpy import (
    geometricJacobian_cached,
    geometricJacobianDot_cached,
    inertiaMatrixCOM_cached,
    centrifugalCoriolisCOM_cached,
    gravityCOM_cached,
)

from utils.kinematics_guard_numpy import KinGuardParams, adaptive_dls_pinv, scale_task_by_J
from utils.dynamics_guard_numpy import DynGuardParams, op_space_guard_and_gate
from utils.muscle_guard_numpy import MuscleGuardParams, solve_muscle_forces
from utils.telemetry_numpy import pack_diag, merge_diag

from muscles.muscle_tools_numpy import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
)


# ---------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------

@dataclass
class SlidingModeParams:
    # Task-space gains
    lambda_surf: np.ndarray   # (2,) surface slope (1/s)
    K_switch: np.ndarray      # (2,) switching gain (accel gain if switch_in_accel=True)
    phi: np.ndarray           # (2,) boundary layer thickness

    # Feedforward & compensation toggles
    Kff_x: float = 1.0
    enable_gravity_comp: bool = True
    enable_velocity_comp: bool = True
    enable_inertia_comp: bool = True

    # Nullspace / bias (optional)
    lambda_ns: float = 0.0
    k_manip: float = 0.0

    # Guard params (dynamic guard uses sigma_thresh_S inside DynGuardParams)
    eps: float = 1e-6
    lam_os_max: float = 1e6
    sigma_thresh: float = 1e-4
    gate_pow: float = 2.0

    # Jacobian gating threshold (FIX C): used for eta_eq / eta_sw based on sminJ(J)
    sminJ_thresh: float = 0.02

    # Muscle inversion
    bisect_iters: int = 16

    # -----------------------------------------------------------------
    # Safety knobs
    # -----------------------------------------------------------------
    switch_in_accel: bool = True
    use_condJ_for_blend: bool = True
    cond_low: float = 20.0
    cond_high: float = 80.0
    elbow_min_rad: float = np.deg2rad(2.0)

    # Feasibility scaling behavior (FIX F)
    tau_scale_apply_thresh: float = 0.7   # only apply scaling if raw scale < this
    tau_scale_safety: float = 0.95        # fraction of estimated feasibility


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _sat_vec(s: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Elementwise sat(s/phi) with safe divide."""
    z = s / (phi + 1e-12)
    return np.clip(z, -1.0, 1.0)

def _blend_weight_from_cond(cond_val: float, lo: float, hi: float) -> float:
    """0 → 1 smoothstep blend over [lo, hi]."""
    if cond_val <= lo:
        return 0.0
    if cond_val >= hi:
        return 1.0
    a = (cond_val - lo) / (hi - lo)
    return float(a * a * (3.0 - 2.0 * a))  # smoothstep

def _safe_cond(A: np.ndarray, fallback: float = 1e9) -> float:
    """Robust condition number."""
    try:
        c = float(np.linalg.cond(A))
        if not np.isfinite(c):
            return float(fallback)
        return c
    except Exception:
        return float(fallback)


# ---------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------

class SlidingModeController:
    def __init__(self, env, arm, params: SlidingModeParams):
        self.env = env
        self.arm = arm
        self.p = params

        # Guards
        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=params.eps,
            lam_os_max=params.lam_os_max,
            gate_pow=params.gate_pow,
            sigma_thresh_S=max(params.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()

        # Internal reference (joint)
        self.qref = None

        # thresholds for blending
        self._cond_low = float(getattr(params, "cond_low", 20.0))
        self._cond_high = float(getattr(params, "cond_high", 80.0))
        self._elbow_min = float(getattr(params, "elbow_min_rad", np.deg2rad(2.0)))

        # FIX D: floors default to 0 (can be changed here if you want)
        self._eta_eq_floor = 0.0
        self._eta_sw_floor = 0.0
        self._eta_sw_pow = 0.75

        # feasibility scaling settings (FIX F)
        self._tau_scale_apply_thresh = float(getattr(params, "tau_scale_apply_thresh", 0.7))
        self._tau_scale_safety = float(getattr(params, "tau_scale_safety", 0.95))

    def reset(self, q0: np.ndarray):
        self.qref = q0.copy()

    def compute(self, x_d: np.ndarray, xd_d: np.ndarray, xdd_d: np.ndarray):
        # -----------------------------------------------------------------
        # State
        # -----------------------------------------------------------------
        joint = self.env.states["joint"][0]
        q, qd = joint[:2], joint[2:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:]
        self.env.skeleton._set_state(q, qd)

        # -----------------------------------------------------------------
        # Kinematics
        # -----------------------------------------------------------------
        J = geometricJacobian_cached(self.env.skeleton._robot, symbolic=False)
        J_xy = J[0:2, :]
        Jdot = geometricJacobianDot_cached(self.env.skeleton._robot, symbolic=False)
        Jdot_xy = Jdot[0:2, :]

        # DLS pseudoinverse + diagnostics
        J_pinv_dls, sminJ, lamJ = adaptive_dls_pinv(J_xy, 2, self.kp)

        # -----------------------------------------------------------------
        # Scale desired task motion near singularities (kinematic safety)
        # -----------------------------------------------------------------
        xd_d_s, xdd_d_s, alpha_J = scale_task_by_J(
            xd_d.copy(), xdd_d.copy(), sminJ=sminJ, P=self.kp
        )

        # -----------------------------------------------------------------
        # Operational-space guard (dynamic safety): Λ_reg and gated refs
        # -----------------------------------------------------------------
        M = inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False)
        Minv = np.linalg.inv(M)
        S = J_xy @ Minv @ J_xy.T

        Lambda_reg, lam_os, eta, eta2, xd_d_g, xdd_d_g, dyn_diag = op_space_guard_and_gate(
            S, xd_d_s.copy(), xdd_d_s.copy(), self.dp
        )

        # -----------------------------------------------------------------
        # Maintain internal joint reference by integrating gated task velocity
        # FIX B: do NOT multiply by alpha_J again (avoid double scaling)
        # -----------------------------------------------------------------
        if self.qref is None:
            self.qref = q.copy()

        qd_des = (J_pinv_dls @ xd_d_g)
        self.qref = self.qref + self.arm.dt * qd_des

        # -----------------------------------------------------------------
        # Inertia compensation toggle (FIX A)
        # -----------------------------------------------------------------
        if self.p.enable_inertia_comp:
            Lambda_used = Lambda_reg
        else:
            Lambda_used = np.eye(2, dtype=float)

        # -----------------------------------------------------------------
        # Dynamics terms h = C qd + g
        # -----------------------------------------------------------------
        h = np.zeros_like(q)
        if self.p.enable_velocity_comp:
            C = centrifugalCoriolisCOM_cached(self.env.skeleton._robot, symbolic=False)
            C = np.asarray(C)
            h = h + (C @ qd if C.ndim == 2 else C.reshape(-1))
        if self.p.enable_gravity_comp:
            g = gravityCOM_cached(
                self.env.skeleton._robot,
                self.env.skeleton._gravity_vec,
                symbolic=False,
            ).reshape(-1)
            h = h + g

        # μ term only meaningful if inertia comp enabled
        if self.p.enable_inertia_comp:
            mu = Lambda_used @ (J_xy @ (Minv @ h)) - Lambda_used @ (Jdot_xy @ qd)
        else:
            mu = np.zeros(2, dtype=float)

        # -----------------------------------------------------------------
        # Sliding surface in task space (textbook tracking error)
        # -----------------------------------------------------------------
        e_x = x_d - x
        e_v = xd_d_g - xd
        s = e_v + self.p.lambda_surf * e_x
        sw = _sat_vec(s, self.p.phi)

        # -----------------------------------------------------------------
        # Eta gating based on Jacobian σ_min (FIX C + FIX D)
        # -----------------------------------------------------------------
        ratio = float(sminJ / (float(self.p.sminJ_thresh) + 1e-12))
        ratio = float(np.clip(ratio, 0.0, 1.0))

        eta_eq = max(self._eta_eq_floor, ratio)
        eta_eq = float(np.clip(eta_eq, 0.0, 1.0))

        eta_sw = max(self._eta_sw_floor, ratio ** self._eta_sw_pow)
        eta_sw = float(np.clip(eta_sw, 0.0, 1.0))

        # -----------------------------------------------------------------
        # Operational-space control law
        # -----------------------------------------------------------------
        if self.p.switch_in_accel:
            # Acceleration-level SMC (robust):
            # eq_part: xdd_d + λ e_v ; sw_part: K sat(s/phi)
            eq_part = (self.p.Kff_x * xdd_d_g) + (self.p.lambda_surf * e_v)
            sw_part = (self.p.K_switch * sw)

            xdd_cmd = (eta_eq * eq_part) + (eta_sw * sw_part)

            F_task = (Lambda_used @ xdd_cmd) + mu
            tau_os = J_xy.T @ F_task
        else:
            # Force-level SMC (kept for debugging / legacy comparison)
            F_eq = (Lambda_used @ (self.p.Kff_x * xdd_d_g)) + mu
            F_sw = -(self.p.K_switch * sw) * eta_sw
            F_task = (eta_eq * F_eq) + F_sw
            tau_os = J_xy.T @ F_task

        # -----------------------------------------------------------------
        # Nullspace / manipulability torque (kept off by default)
        # -----------------------------------------------------------------
        tau_ns = np.zeros_like(q)
        ns_diag = {}

        # -----------------------------------------------------------------
        # Mild posture bias (optional safety; retained)
        # -----------------------------------------------------------------
        q_center = np.array([np.pi / 3, np.pi / 2])
        tau_bias = -0.15 * (q - q_center) * (1.0 - float(np.clip(eta, 0.0, 1.0)))

        # -----------------------------------------------------------------
        # Blending to joint fallback near singularities / low authority
        # -----------------------------------------------------------------
        geom = self.env.states["geometry"]
        R = geom[0, 2:2 + self.env.skeleton.dof, :]  # (dof, n_muscle)

        condR = _safe_cond(R, fallback=1e9)
        condJ = _safe_cond(J_xy, fallback=1e9) if self.p.use_condJ_for_blend else 0.0  # FIX E
        cond_metric = condJ if self.p.use_condJ_for_blend else condR

        w_cr = _blend_weight_from_cond(cond_metric, self._cond_low, self._cond_high)
        w_eta = float(np.clip(1.0 - eta, 0.0, 1.0))
        w = float(np.clip(max(w_cr, 0.5 * w_eta), 0.0, 1.0))

        # Joint fallback target: hold qref; keep elbow away from extension
        q_star = self.qref.copy()
        q_star[1] = max(float(q_star[1]), self._elbow_min)

        # Simple joint-space PD damping (fallback)
        Kp_js = np.array([12.0, 10.0])
        Kd_js = np.array([2.5, 2.0])
        tau_js = Kp_js * (q_star - q) + Kd_js * (0.0 - qd)

        tau_task = (1.0 - w) * tau_os + w * tau_js

        # Add bias / damping
        tau_des = tau_task + tau_ns + tau_bias - 0.06 * qd

        # -----------------------------------------------------------------
        # Torque feasibility scaling (muscle limits) (FIX F)
        # -----------------------------------------------------------------
        Fmax_vec = getattr(self, "_Fmax_cache", None)
        if Fmax_vec is None:  # fixed by the muscle; cache on first use
            Fmax_vec = get_Fmax_vec(self.env, R.shape[1])
            self._Fmax_cache = Fmax_vec
        tau_feas = np.sum(np.abs(R) * Fmax_vec[None, :], axis=1)  # conservative bound
        tau_clip = self._tau_scale_safety * tau_feas

        raw_scale = float(np.min(tau_clip / (np.abs(tau_des) + 1e-12)))
        raw_scale = float(np.clip(raw_scale, 0.0, 1.0))

        if raw_scale < self._tau_scale_apply_thresh:
            tau_des = tau_des * raw_scale
            tau_scale_applied = raw_scale
        else:
            tau_scale_applied = 1.0

        # -----------------------------------------------------------------
        # Muscle allocation + activation inversion
        # -----------------------------------------------------------------
        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta, self.mp)

        idx_flpe = getattr(self, "_idx_flpe", None)
        if idx_flpe is None:
            idx_flpe = self.env.muscle.state_name.index("force-length PE")
            self._idx_flpe = idx_flpe
        flpe = self.env.states["muscle"][0, idx_flpe, :]

        lenvel = geom[:, :2, :]  # (B, 2, n_muscle) -> [len, vel]
        a = force_to_activation_bisect(
            F_des, lenvel, self.env.muscle, flpe, Fmax_vec,
            iters=self.p.bisect_iters,
        )

        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = Fmax_vec * (af_now + flpe)

        F_corr = saturation_repair_tau(
            -R, F_pred, a,
            self.env.muscle.min_activation, 1.0,
            Fmax_vec, tau_des=tau_des,
        )
        if np.any(np.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(
                F_corr, lenvel, self.env.muscle, flpe, Fmax_vec,
                iters=max(4, self.p.bisect_iters - 4),
            )

        # -----------------------------------------------------------------
        # Diagnostics
        # -----------------------------------------------------------------
        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ, alpha_J=alpha_J, k_manip=0.0)

        diag = merge_diag(
            kin_diag, dyn_diag, ns_diag, mus_diag,
            pack_diag(
                lam_os=lam_os,
                eta=float(eta),
                eta2=float(eta2),
                condR=float(condR),
                condJ=float(condJ),
                w=float(w),
                tau_scale=float(tau_scale_applied),
                tau_scale_raw=float(raw_scale),
                eta_eq=float(eta_eq),
                eta_sw=float(eta_sw),
            ),
        )

        return {
            "tau_des": tau_des,
            "R": R,
            "Fmax": Fmax_vec,
            "F_des": F_des,
            "act": a,
            "q": q,
            "qd": qd,
            "x": x,
            "xd": xd,
            "xref_tuple": (x_d, xd_d_g, xdd_d_g),
            "eta": eta2,
            "diag": diag,
        }

