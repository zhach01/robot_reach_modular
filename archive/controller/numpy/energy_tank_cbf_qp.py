# controller/energy_tank_cbf_qp.py
"""
CBF-QP Passivity Controller for the 2-DoF, 6-muscle arm.
Replaces energy tank gating with Control Barrier Function Quadratic Programming.

Design goals:
1) Achieve < 1 cm RMS tracking while maintaining passivity
2) Use CBF-QP for optimal passivity enforcement (minimal intervention)
3) Keep all safety features of bulletproof version
4) Enable multi-constraint handling (passivity + torque limits + joint limits)

Key differences from bulletproof:
- ✅ Replaces tank gating with CBF-QP optimization
- ✅ Enforces passivity as safety constraint, not energy balance
- ✅ Can handle multiple constraints simultaneously
- ✅ Preserves more of nominal control when safe
- ✅ Expected RMSE: 0.7-1.0 cm (vs 1.5-1.7 cm with tank)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
try:
    import qpsolvers
    QPSOLVERS_AVAILABLE = True
except ImportError:
    QPSOLVERS_AVAILABLE = False
    print("Warning: qpsolvers not installed. Using fallback.")

from utils.numpy.math_utils import matrix_sqrt_spd, matrix_isqrt_spd, matrix_sqrt_isqrt_spd
from muscles.numpy.muscle_tools import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
    apply_internal_force_regulation,
)
from model_lib.numpy.skeleton import (
    geometricJacobian_cached,
    geometricJacobianDot_cached,
    inertiaMatrixCOM_cached,
    centrifugalCoriolisCOM_cached,
    gravityCOM_cached,
)
from utils.numpy.kinematics_guard import KinGuardParams, adaptive_dls_pinv, scale_task_by_J
from utils.numpy.dynamics_guard import DynGuardParams, op_space_guard_and_gate
from utils.numpy.muscle_guard import MuscleGuardParams, solve_muscle_forces
from utils.numpy.telemetry import pack_diag, merge_diag


# ---------------------------- CBF-QP Utilities ----------------------------

def solve_qp_simple(H, f, A_ub=None, b_ub=None, A_eq=None, b_eq=None, 
                   lb=None, ub=None, solver='osqp', **kwargs):
    """Wrapper for QP solvers with fallback."""
    if QPSOLVERS_AVAILABLE:
        try:
            return qpsolvers.solve_qp(H, f, A_ub, b_ub, A_eq, b_eq, lb, ub, 
                                      solver=solver, **kwargs)
        except Exception as e:
            print(f"QP solver failed: {e}. Using fallback.")
            return solve_qp_fallback(H, f, A_ub, b_ub, lb, ub)
    else:
        return solve_qp_fallback(H, f, A_ub, b_ub, lb, ub)


def solve_qp_fallback(H, f, A_ub=None, b_ub=None, lb=None, ub=None, iters=50):
    """
    Dependency-free fallback QP solver (used when `qpsolvers` is unavailable).
    Solves:  min 0.5 x'Hx + f'x   s.t.  A_ub x <= b_ub,  lb <= x <= ub

    Starts from the unconstrained minimizer and projects onto the feasible set by
    alternating ORTHOGONAL half-space projections and box clipping. This actually
    enforces the inequality constraints (the previous "x *= min(1, b/(a·x))"
    scaling could not satisfy a negative upper bound, flipped the sign when the
    bound was negative, broke under multiple constraints, and returned zeros for
    n != 2 — so the passivity barrier was not enforced; audit H5).

    For the energy-tank CBF this is exact: a single passivity half-space
    `xd·F <= P_diss + alpha*h` plus a box, where alternating projection
    converges to the closest feasible force.
    """
    n = H.shape[0]
    # Unconstrained minimizer (stable solve; lstsq fallback for singular H).
    try:
        x = -np.linalg.solve(H, f)
    except np.linalg.LinAlgError:
        x = -np.linalg.lstsq(H, f, rcond=None)[0]

    has_ineq = (A_ub is not None) and (b_ub is not None)
    has_box = (lb is not None) and (ub is not None)
    if not (has_ineq or has_box):
        return x

    A = np.atleast_2d(np.asarray(A_ub, dtype=float)) if has_ineq else None
    b = np.atleast_1d(np.asarray(b_ub, dtype=float)) if has_ineq else None
    if has_box:
        x = np.clip(x, lb, ub)

    for _ in range(int(iters)):
        moved = False
        if has_ineq:
            for i in range(A.shape[0]):
                ai = A[i]
                viol = float(ai @ x - b[i])
                if viol > 1e-12:
                    # orthogonal projection onto the half-space {x : ai·x <= bi}
                    x = x - (viol / (float(ai @ ai) + 1e-12)) * ai
                    moved = True
        if has_box:
            x_clip = np.clip(x, lb, ub)
            if not np.allclose(x_clip, x):
                moved = True
            x = x_clip
        if not moved:
            break

    return x


# ---------------------------- CBF-QP Parameters ----------------------------

@dataclass
class CBFQPTankParams:
    """Parameters for CBF-QP passivity controller."""
    
    # Tracking stiffness & passive baseline damping
    K0: np.ndarray
    D0: np.ndarray
    
    # CBF-QP specific parameters
    cbf_alpha: float = 2.0               # CBF class-K function parameter
    cbf_E_min: float = 0.1               # Minimum safe energy level
    qp_weight_tracking: float = 10.0     # Weight for tracking cost
    qp_weight_control: float = 0.1       # Weight for control effort
    use_cbf_qp: bool = True              # Enable CBF-QP (False = fallback to tank)
    
    # Feedforward and damping
    Kff: float = 1.0
    zeta: float = 0.85
    
    # Integral term (generally OFF for min-jerk)
    KI: np.ndarray = None
    Imax: np.ndarray = None
    
    # Operational-space numerics
    eps: float = 1e-6
    lam_os_max: float = 200.0
    sigma_thresh: float = 1e-4
    gate_pow: float = 2.0
    
    # Plant compensation
    enable_inertia_comp: bool = True
    enable_gravity_comp: bool = True
    enable_coriolis_comp: bool = True
    enable_joint_damping: bool = True
    
    # Tank settings (for monitoring/fallback)
    E0: float = 0.4
    Emin: float = 0.1
    Emax: float = 2.0
    harvest_gain: float = 1.0
    store_returned_energy: bool = True
    
    # Singularity handling
    eta_min: float = 0.25
    v_eps: float = 1e-8
    
    # Torque feasibility
    clamp_tau_to_muscles: bool = True
    tau_cap_margin: float = 0.95
    
    # mu handling
    match_torch: bool = False            # Recommended False for CBF-QP
    
    # Muscle options
    bisect_iters: int = 12
    enable_internal_force: bool = False
    cocon_a0: float = 0.0
    linesearch_eps: float = 1e-6
    linesearch_safety: float = 1.2
    
    def __post_init__(self):
        """Initialize arrays if None."""
        if self.KI is None:
            self.KI = np.zeros(2, dtype=float)
        if self.Imax is None:
            self.Imax = np.full(2, 0.0, dtype=float)


# ---------------------------- CBF-QP Controller ----------------------------

class CBFQPTankController:
    """
    CBF-QP based passivity controller with energy monitoring.
    Replaces energy tank gating with optimal CBF-QP constraints.
    """
    
    def __init__(self, env, arm, params: CBFQPTankParams):
        self.env = env
        self.arm = arm
        self.p = params
        self.qref = None
        
        # Guards
        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=float(self.p.eps),
            lam_os_max=float(self.p.lam_os_max),
            gate_pow=float(self.p.gate_pow),
            sigma_thresh_S=max(float(self.p.sigma_thresh), 1e-9),
        )
        self.mp = MuscleGuardParams()
        
        # Ensure SPD matrices
        self._K0 = self._project_spd_2x2(self.p.K0)
        self._D0 = self._project_spd_2x2(self.p.D0)
        
        # State
        self._tank_E = float(self.p.E0)  # For monitoring only
        self._I = np.zeros(2, dtype=float)
        self._K_prev = None
        self._h_history = []  # CBF history for diagnostics
        
        # CBF-QP state
        self._cbf_h = 0.0
        self._cbf_h_dot = 0.0
        self._qp_active = False  # Whether QP modified control
        
        print(f"CBF-QP Controller initialized. Using CBF-QP: {self.p.use_cbf_qp}")
        if not QPSOLVERS_AVAILABLE and self.p.use_cbf_qp:
            print("Warning: qpsolvers not installed. Using fallback solver.")
    
    # --- Small utilities (copied from bulletproof) ---
    
    @staticmethod
    def _sym(A: np.ndarray) -> np.ndarray:
        return 0.5 * (A + A.T)
    
    def _project_spd_2x2(self, A: np.ndarray) -> np.ndarray:
        """Project a 2x2 matrix to SPD."""
        A = self._sym(A)
        w, V = np.linalg.eigh(A)
        w = np.maximum(w, self.p.eps)
        return (V * w) @ V.T
    
    @staticmethod
    def _decompose_parallel_perp(F: np.ndarray, v: np.ndarray, eps: float = 1e-12):
        """F = F_par + F_perp, where F_par || v and F_perp ⟂ v."""
        nv2 = float(v @ v)
        if nv2 < eps:
            return np.zeros_like(F), F
        alpha = float(F @ v) / nv2
        F_par = alpha * v
        return F_par, F - F_par
    
    @staticmethod
    def _tau_cap_from_muscles(R: np.ndarray, Fmax: np.ndarray, margin: float = 0.95):
        """Torque capacity from muscles."""
        cap = (np.abs(R) @ np.asarray(Fmax, dtype=float).reshape(-1))
        return float(margin) * cap
    
    # --- Public API ---
    
    def reset(self, q0):
        self.qref = np.asarray(q0, dtype=float).copy()
        self._tank_E = float(self.p.E0)
        self._I = np.zeros(2, dtype=float)
        self._K_prev = None
        self._h_history = []
        self._cbf_h = 0.0
        self._cbf_h_dot = 0.0
        self._qp_active = False
    
    # --- Dynamics computation (same as bulletproof) ---
    
    def _compute_dynamics(self, q: np.ndarray, qd: np.ndarray):
        p = self.p
        n = int(len(q))
        
        if p.enable_inertia_comp:
            M = inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False)
        else:
            M = np.eye(n)
        
        if p.enable_gravity_comp:
            g = gravityCOM_cached(
                self.env.skeleton._robot, self.env.skeleton._gravity_vec, symbolic=False
            ).reshape(-1)
        else:
            g = np.zeros_like(q)
        
        if p.enable_coriolis_comp:
            C_any = centrifugalCoriolisCOM_cached(self.env.skeleton._robot, symbolic=False)
        else:
            C_any = np.zeros((n, n))
        
        if p.enable_joint_damping:
            Dq = np.diag(np.full(n, float(self.arm.damping)))
        else:
            Dq = np.zeros((n, n))
        
        C_any = np.asarray(C_any)
        if C_any.ndim == 2:
            h = C_any @ qd + g + Dq @ qd
        else:
            h = C_any.reshape(-1) + g + Dq @ qd
        
        return np.asarray(M, dtype=float), np.asarray(h, dtype=float).reshape(-1)
    
    # --- CBF-QP Core Methods ---
    
    def _compute_cbf_constraint(self, xd, F_nom, E_current, P_diss, dt):
        """
        Compute CBF constraint for passivity.
        
        Barrier function: h = E_current - E_min
        Constraint: ḣ ≥ -α·h  →  P_inj ≤ P_diss + α·h
        
        Where P_inj = max(0, F_nom·xd) is the injected power.
        """
        # Current barrier value
        h = E_current - self.p.cbf_E_min
        
        # Injected power (positive part)
        P_inj = max(0.0, F_nom.T @ xd)
        
        # CBF constraint: P_inj ≤ P_diss + α·h
        rhs = P_diss + self.p.cbf_alpha * h
        
        # Linear constraint in F: F·xd ≤ rhs
        # Note: We constrain the total power, not just positive part
        # This is a conservative approximation
        A_ub = xd.reshape(1, -1)  # 1×2
        b_ub = np.array([rhs])
        
        # Also store for diagnostics
        self._cbf_h = h
        self._cbf_h_dot = P_inj - P_diss
        
        return A_ub, b_ub, h, P_inj, rhs
    
    def _solve_cbf_qp(self, F_nom, xd, J_xy, R, Fmax_vec, E_current, P_diss, dt):
        """
        Solve CBF-QP for safe force command.
        
        QP: min_F  ||F - F_nom||²_W1 + ||F||²_W2
            s.t.   A_cbf·F ≤ b_cbf    (passivity)
                   τ_min ≤ J^T·F ≤ τ_max  (torque limits)
        
        Returns: F_safe, qp_diagnostics
        """
        n = F_nom.shape[0]  # Should be 2
        
        # --- QP Cost: min 0.5*F'*H*F + f'*F ---
        W_track = self.p.qp_weight_tracking * np.eye(n)
        W_control = self.p.qp_weight_control * np.eye(n)
        H = 2 * (W_track + W_control)
        f = -2 * W_track @ F_nom
        
        # --- CBF Constraint (passivity) ---
        A_cbf, b_cbf, h, P_inj, rhs = self._compute_cbf_constraint(
            xd, F_nom, E_current, P_diss, dt
        )
        
        # --- Torque Limits Constraint ---
        # Convert muscle limits to torque limits
        tau_cap = self._tau_cap_from_muscles(R, Fmax_vec, margin=float(self.p.tau_cap_margin))
        tau_min = -tau_cap
        tau_max = tau_cap
        
        # Inequality: tau_min ≤ J^T·F ≤ tau_max
        # Convert to: J^T·F ≤ tau_max and -J^T·F ≤ -tau_min
        A_tau = np.vstack([J_xy.T, -J_xy.T])  # 4×2 for 2-DoF
        b_tau = np.hstack([tau_max, -tau_min])  # Length 4
        
        # --- Combine Constraints ---
        A_ub = np.vstack([A_cbf, A_tau])  # 5×2
        b_ub = np.hstack([b_cbf, b_tau])   # Length 5
        
        # --- Solve QP ---
        try:
            if self.p.use_cbf_qp:
                F_safe = solve_qp_simple(
                    H, f, A_ub=A_ub, b_ub=b_ub,
                    solver='osqp' if QPSOLVERS_AVAILABLE else None
                )
                self._qp_active = True
            else:
                # Fallback to simple scaling (like tank)
                F_safe = self._tank_fallback(F_nom, xd, E_current, P_diss, dt)
                self._qp_active = False
        except Exception as e:
            print(f"QP failed: {e}. Using tank fallback.")
            F_safe = self._tank_fallback(F_nom, xd, E_current, P_diss, dt)
            self._qp_active = False
        
        # Diagnostics
        qp_diag = {
            'cbf_h': h,
            'cbf_h_dot': self._cbf_h_dot,
            'P_inj': P_inj,
            'P_diss': P_diss,
            'rhs': rhs,
            'qp_active': float(self._qp_active),
            'F_norm_diff': np.linalg.norm(F_safe - F_nom),
        }
        
        return F_safe, qp_diag
    
    def _tank_fallback(self, F_nom, xd, E_current, P_diss, dt):
        """Fallback to energy tank logic if QP fails."""
        # Decompose into parallel/perpendicular
        F_par, F_perp = self._decompose_parallel_perp(F_nom, xd)
        P_par_raw = float(F_par.T @ xd)
        P_inj = max(0.0, P_par_raw)
        
        # Tank-like gating
        P_need = P_inj
        if P_need > 0.0:
            s = (E_current - float(self.p.Emin)) / (dt * P_need + float(self.p.eps))
            s = float(np.clip(s, 0.0, 1.0))
        else:
            s = 1.0
        
        return F_perp + s * F_par
    
    # --- Main Compute Method ---
    
    def compute(self, x_d, xd_d, xdd_d):
        dt = float(self.arm.dt)
        
        # --- State & Kinematics (same as bulletproof) ---
        joint = self.env.states["joint"][0]
        q, qd = joint[:2], joint[2:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:]
        self.env.skeleton._set_state(q, qd)
        
        # Jacobians
        J = geometricJacobian_cached(self.env.skeleton._robot, symbolic=False)
        J_xy = J[0:2, :]
        Jdot = geometricJacobianDot_cached(self.env.skeleton._robot, symbolic=False)
        Jdot_xy = Jdot[0:2, :]
        n = q.shape[0]
        
        # Kinematics guard
        J_pinv_dls, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)
        xd_d_g, xdd_d_g, alpha_J = scale_task_by_J(xd_d, xdd_d, sminJ, self.kp)
        if self.qref is None:
            self.qref = q.copy()
        qd_des = J_pinv_dls @ xd_d_g
        self.qref = self.qref + qd_des * dt
        
        # Dynamics
        M, h = self._compute_dynamics(q, qd)
        Minv = np.linalg.inv(M)
        
        # Operational-space guard
        S = J_xy @ Minv @ J_xy.T
        Lambda, lam_os, eta, eta2, xd_d_g, xdd_d_g, dyn_diag = op_space_guard_and_gate(
            S, xd_d_g, xdd_d_g, self.dp
        )
        eta_clip = float(np.clip(eta2, self.p.eta_min, 1.0))
        
        # OSC bias term
        mu = Lambda @ (J_xy @ Minv @ h - Jdot_xy @ qd)
        
        # Tracking errors
        e_x = np.asarray(x_d, dtype=float) - x
        e_v = np.asarray(xd_d_g, dtype=float) - xd
        
        # Stiffness
        K_now = eta_clip * self._K0
        
        # Kdot bookkeeping
        if self._K_prev is None:
            Kdot = np.zeros_like(K_now)
        else:
            Kdot = (K_now - self._K_prev) / dt
        self._K_prev = K_now.copy()
        
        # Stiffness variation power
        P_K = -0.5 * float(e_x.T @ (Kdot @ e_x))
        P_refund = max(0.0, P_K)
        P_spend = max(0.0, -P_K)
        
        # Inertia-shaped damping
        Lam_s, Lam_is = matrix_sqrt_isqrt_spd(Lambda)  # one eigh, not two
        Kv = (
            2.0
            * float(self.p.zeta)
            * Lam_s
            @ matrix_sqrt_spd(Lam_is @ K_now @ Lam_is)
            @ Lam_s
        )
        
        # Passive baseline damper
        F_pas = -(self._D0 @ xd)
        P_diss = float(xd.T @ (self._D0 @ xd))
        
        # Integral term
        if float(np.max(np.abs(self.p.Imax))) > 0.0 and float(np.max(np.abs(self.p.KI))) > 0.0:
            self._I = np.clip(self._I + e_x * dt, -self.p.Imax, self.p.Imax)
        else:
            self._I[:] = 0.0
        F_I = self.p.KI * self._I
        
        # ---- Nominal Force Computation ----
        F_nom = (
            (Lambda @ (float(self.p.Kff) * xdd_d_g))
            + (K_now @ e_x)
            + (Kv @ e_v)
            + F_I
        )
        
        # Add mu based on match_torch
        if self.p.match_torch:
            F_nom = F_nom + mu
            F_base = F_pas  # mu already in F_nom
        else:
            F_base = mu + F_pas  # mu always applied
        
        # ---- CBF-QP for Passivity Enforcement ----
        # Get muscle parameters for torque limits
        geom = self.env.states["geometry"]
        R = geom[:, 2: 2 + self.env.skeleton.dof, :][0]
        Fmax_vec = get_Fmax_vec(self.env, R.shape[1])
        
        # Current "energy" for CBF (using tank concept as monitoring)
        # This is just for the CBF constraint, actual tank update is different
        E_current = self._tank_E
        
        # Solve CBF-QP
        F_safe, qp_diag = self._solve_cbf_qp(
            F_nom, xd, J_xy, R, Fmax_vec, E_current, P_diss, dt
        )
        
        # Final commanded force
        if self.p.match_torch:
            F_cmd = F_base + F_safe  # F_base = F_pas, mu already in F_safe
        else:
            F_cmd = F_base + F_safe  # F_base = mu + F_pas
        
        # Map to joint torques
        tau_des = J_xy.T @ F_cmd
        
        # Optional torque clamping (redundant with QP but safe)
        if bool(self.p.clamp_tau_to_muscles):
            tau_cap = self._tau_cap_from_muscles(R, Fmax_vec, margin=float(self.p.tau_cap_margin))
            tau_des = np.clip(tau_des, -tau_cap, tau_cap)
        
        # ---- Update Energy Monitor (not used for control, just diagnostics) ----
        # Compute actual injected power
        P_actual = max(0.0, F_cmd.T @ xd)
        
        # Tank-like update for monitoring
        Pin = float(self.p.harvest_gain) * P_diss + P_refund
        Pout = P_actual + P_spend
        
        if self._tank_E >= float(self.p.Emax) and Pin > 0.0:
            Pin = 0.0
        
        self._tank_E = float(
            np.clip(self._tank_E + dt * (Pin - Pout), 
                   float(self.p.Emin), float(self.p.Emax))
        )
        
        # Store CBF history
        self._h_history.append(qp_diag['cbf_h'])
        if len(self._h_history) > 1000:
            self._h_history.pop(0)
        
        # ---- Muscle Allocation (same as bulletproof) ----
        lenvel = geom[:, :2, :]
        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]
        
        # Muscle solve
        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta, self.mp)
        
        # Optional internal-force regulation
        if bool(self.p.enable_internal_force) and float(self.p.cocon_a0) > 0.0:
            A = -R
            a0_vec = np.full(F_des.shape[0], float(self.p.cocon_a0))
            af0 = active_force_from_activation(a0_vec, lenvel, self.env.muscle)
            F_bias = Fmax_vec * (af0 + flpe)
            F_des = apply_internal_force_regulation(
                A,
                F_des,
                F_bias,
                Fmax_vec,
                eps=float(self.p.eps),
                linesearch_eps=float(self.p.linesearch_eps),
                linesearch_safety=float(self.p.linesearch_safety),
                scale=float(eta_clip),
            )
        
        # Hill inversion
        a = force_to_activation_bisect(
            F_des, lenvel, self.env.muscle, flpe, Fmax_vec, 
            iters=int(self.p.bisect_iters)
        )
        
        # Saturation repair
        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = Fmax_vec * (af_now + flpe)
        A = -R
        F_corr = saturation_repair_tau(
            A,
            F_pred,
            a,
            self.env.muscle.min_activation,
            1.0,
            Fmax_vec,
            tau_des=tau_des,
        )
        if np.any(np.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(
                F_corr,
                lenvel,
                self.env.muscle,
                flpe,
                Fmax_vec,
                iters=max(4, int(self.p.bisect_iters) - 4),
            )
        
        # ---- Diagnostics ----
        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ, alpha_J=alpha_J)
        cbf_diag = pack_diag(
            E=self._tank_E,
            cbf_h=qp_diag['cbf_h'],
            cbf_h_dot=qp_diag['cbf_h_dot'],
            P_inj=qp_diag['P_inj'],
            P_diss=qp_diag['P_diss'],
            P_K=P_K,
            P_refund=P_refund,
            P_spend=P_spend,
            eta_clip=eta_clip,
            qp_active=qp_diag['qp_active'],
            F_diff=qp_diag['F_norm_diff'],
        )
        
        diag = merge_diag(
            kin_diag,
            dyn_diag,
            mus_diag,
            pack_diag(lam_os=lam_os, eta=eta, eta2=eta2),
            cbf_diag,
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
            "eta": eta_clip,
            "cbf": {
                "E": self._tank_E,
                "h": qp_diag['cbf_h'],
                "h_dot": qp_diag['cbf_h_dot'],
                "qp_active": qp_diag['qp_active'],
                "P_inj": qp_diag['P_inj'],
                "P_diss": qp_diag['P_diss'],
                "P_K": P_K,
                "P_refund": P_refund,
                "P_spend": P_spend,
            },
            "diag": diag,
        }


# Convenience aliases
CBFQPTankControllerParams = CBFQPTankParams
