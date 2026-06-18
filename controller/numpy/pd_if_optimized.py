"""
Optimized PD+IF Controller for Musculoskeletal Arm Control

Key optimizations based on literature:
1. Full operational-space dynamics (Khatib 1987)
2. Critically-damped task-space gains (Λ^(1/2) K_p Λ^(1/2) matching)
3. Optional integral action with anti-windup (back-calculation method)
4. Adaptive feedforward compensation
5. Singularity-robust muscle force allocation

Literature references:
- Khatib (1987): Operational Space Formulation
- Nakanishi et al. (2008): Operational Space Control comparison
- Åström & Murray: PID Control (anti-windup)
- Ghannadi et al. (2018): Optimal impedance control for musculoskeletal

Author: Optimized version
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

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
from utils.numpy.math_utils import matrix_sqrt_spd, matrix_isqrt_spd

from muscles.numpy.muscle_tools import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
)


@dataclass
class OptimizedPDIFParams:
    """Parameters for optimized PD+IF controller."""
    
    # Task-space gains (higher than original for better tracking)
    Kp_task: np.ndarray = field(default_factory=lambda: np.array([800.0, 800.0]))
    Kd_task: np.ndarray = field(default_factory=lambda: np.array([60.0, 60.0]))  # If None, computed for critical damping
    
    # Feedforward gain on desired acceleration
    Kff: float = 1.0
    
    # Integral action (optional, for steady-state error rejection)
    enable_integral: bool = False
    Ki_task: np.ndarray = field(default_factory=lambda: np.array([50.0, 50.0]))
    integral_limit: float = 0.5  # Anti-windup: limit on integral term magnitude
    anti_windup_gain: float = 2.0  # Back-calculation gain (1/Ta)
    
    # Dynamics compensation
    enable_inertia_comp: bool = True
    enable_gravity_comp: bool = True
    enable_coriolis_comp: bool = True
    
    # Critical damping option
    use_critical_damping: bool = True  # Compute Kd from Kp and Λ
    damping_ratio: float = 1.0  # 1.0 = critically damped
    
    # Nullspace posture control (simplified)
    enable_nullspace: bool = True
    Kp_null: float = 20.0
    Kd_null: float = 5.0
    
    # Guards
    eps: float = 1e-6
    lam_os_max: float = 1e6
    sigma_thresh: float = 1e-4
    gate_pow: float = 2.0
    
    # Muscle inversion
    bisect_iters: int = 12
    
    # Internal force (co-contraction) - typically not needed with good tracking
    enable_internal_force: bool = False
    cocon_level: float = 0.03


class OptimizedPDIFController:
    """
    Optimized PD+IF controller with:
    - Full operational-space dynamics
    - Critically-damped gains
    - Optional integral action with anti-windup
    - Robust singularity handling
    """
    
    def __init__(self, env, arm, params: OptimizedPDIFParams):
        self.env = env
        self.arm = arm
        self.p = params
        
        # Guard parameters
        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=self.p.eps,
            lam_os_max=self.p.lam_os_max,
            gate_pow=self.p.gate_pow,
            sigma_thresh_S=max(self.p.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()
        
        # State
        self.qref = None
        
        # Integral state (for anti-windup)
        self.integral_error = np.zeros(2)
        self.prev_tau_cmd = np.zeros(2)
        self.prev_tau_actual = np.zeros(2)
    
    def reset(self, q0: np.ndarray):
        """Reset controller state."""
        self.qref = q0.copy()
        self.integral_error = np.zeros(2)
        self.prev_tau_cmd = np.zeros(2)
        self.prev_tau_actual = np.zeros(2)
    
    def _compute_critical_damping(self, Kp_mat: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
        """
        Compute critically-damped derivative gain matrix.
        
        For critically damped response: Kd = 2 * ζ * sqrt(Λ * Kp)
        Using the inertia-weighted formulation from Nakanishi et al.
        """
        try:
            Lam_sqrt = matrix_sqrt_spd(Lambda)
            Lam_isqrt = matrix_isqrt_spd(Lambda)
            
            # Kd = 2 * ζ * Λ^(1/2) * sqrt(Λ^(-1/2) * Kp * Λ^(-1/2)) * Λ^(1/2)
            inner = Lam_isqrt @ Kp_mat @ Lam_isqrt
            inner_sqrt = matrix_sqrt_spd(inner)
            Kd_mat = 2.0 * self.p.damping_ratio * Lam_sqrt @ inner_sqrt @ Lam_sqrt
            return Kd_mat
        except:
            # Fallback to simple diagonal damping
            return np.diag(2.0 * self.p.damping_ratio * np.sqrt(np.diag(Kp_mat)))
    
    def compute(self, x_d: np.ndarray, xd_d: np.ndarray, xdd_d: np.ndarray) -> dict:
        """Main control computation."""
        
        # === State extraction ===
        joint = self.env.states["joint"][0]
        q, qd = joint[:2], joint[2:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:]
        self.env.skeleton._set_state(q, qd)
        
        if self.qref is None:
            self.qref = q.copy()
        
        n = q.shape[0]
        dt = self.arm.dt
        
        # === Kinematics ===
        J = geometricJacobian_cached(self.env.skeleton._robot, symbolic=False)
        J_xy = J[0:2, :]
        Jdot = geometricJacobianDot_cached(self.env.skeleton._robot, symbolic=False)
        Jdot_xy = Jdot[0:2, :]
        
        # Kinematic guard (adaptive DLS)
        J_pinv, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)
        xd_d_scaled, xdd_d_scaled, alpha_J = scale_task_by_J(
            xd_d.copy(), xdd_d.copy(), sminJ, self.kp
        )
        
        # === Dynamics ===
        M = inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False)
        Minv = np.linalg.inv(M)
        
        # Nonlinear terms
        h = np.zeros(n)
        if self.p.enable_gravity_comp:
            g = gravityCOM_cached(
                self.env.skeleton._robot,
                self.env.skeleton._gravity_vec,
                symbolic=False
            ).reshape(-1)
            h = h + g
        
        if self.p.enable_coriolis_comp:
            C = centrifugalCoriolisCOM_cached(self.env.skeleton._robot, symbolic=False)
            C = np.asarray(C)
            if C.ndim == 2:
                h = h + C @ qd
            else:
                h = h + C.reshape(-1)
        
        # === Operational-space dynamics guard ===
        S = J_xy @ Minv @ J_xy.T
        Lambda_reg, lam_os, eta, eta2, xd_d_g, xdd_d_g, dyn_diag = op_space_guard_and_gate(
            S, xd_d_scaled.copy(), xdd_d_scaled.copy(), self.dp
        )
        
        # === Task-space errors ===
        e_x = x_d - x  # Position error
        e_v = xd_d_g - xd  # Velocity error
        
        # === Integral action with anti-windup ===
        if self.p.enable_integral:
            # Update integral with anti-windup (back-calculation)
            # Anti-windup: I += dt * (e + Ka * (u_actual - u_cmd))
            windup_correction = self.p.anti_windup_gain * (self.prev_tau_actual - self.prev_tau_cmd)
            self.integral_error += dt * (e_x + J_pinv @ windup_correction)
            
            # Clamp integral term
            int_norm = np.linalg.norm(self.integral_error)
            if int_norm > self.p.integral_limit:
                self.integral_error *= self.p.integral_limit / int_norm
        
        # === Gain matrices ===
        Kp_mat = np.diag(self.p.Kp_task)
        
        if self.p.use_critical_damping and self.p.enable_inertia_comp:
            Kd_mat = self._compute_critical_damping(Kp_mat, Lambda_reg)
        else:
            Kd_mat = np.diag(self.p.Kd_task)
        
        Ki_mat = np.diag(self.p.Ki_task) if self.p.enable_integral else np.zeros((2, 2))
        
        # === Task-space control law ===
        # F = Λ * (ẍ_d + Kd * ė + Kp * e + Ki * ∫e) + μ
        
        xdd_cmd = self.p.Kff * xdd_d_g + Kd_mat @ e_v + Kp_mat @ e_x
        if self.p.enable_integral:
            xdd_cmd += Ki_mat @ self.integral_error
        
        if self.p.enable_inertia_comp:
            # Full operational-space: F = Λ * ẍ_cmd + μ
            # μ = Λ * (J * M^(-1) * h - J̇ * q̇)
            mu = Lambda_reg @ (J_xy @ (Minv @ h) - Jdot_xy @ qd)
            F_task = Lambda_reg @ xdd_cmd + mu
        else:
            # Simple PD in task space
            F_task = Kp_mat @ e_x + Kd_mat @ e_v
        
        tau_task = J_xy.T @ F_task
        
        # === Nullspace control (simplified) ===
        tau_null = np.zeros(n)
        if self.p.enable_nullspace:
            # Dynamically-consistent nullspace projector
            J_dyn_pinv = Minv @ J_xy.T @ Lambda_reg
            N = np.eye(n) - J_dyn_pinv @ J_xy
            
            # Simple posture control in nullspace
            qdd_null = self.p.Kp_null * (self.qref - q) - self.p.Kd_null * qd
            tau_null = eta2 * (N.T @ (M @ qdd_null + h))
        
        # === Total desired torque ===
        tau_des = tau_task + tau_null
        
        # Gate by eta for safety near singularities
        tau_des = tau_des * float(np.clip(eta, 0.3, 1.0))
        
        # === Muscle force allocation ===
        geom = self.env.states["geometry"]
        lenvel = geom[:, :2, :]
        R = geom[0, 2:2 + self.env.skeleton.dof, :]
        Fmax_vec = get_Fmax_vec(self.env, R.shape[1])
        
        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]
        
        # Robust muscle solve
        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta, self.mp)
        
        # === Optional co-contraction ===
        if self.p.enable_internal_force:
            a_min = float(getattr(self.env.muscle, "min_activation", 0.02))
            cocon = np.full(F_des.shape[0], self.p.cocon_level)
            af_cocon = active_force_from_activation(cocon, lenvel, self.env.muscle)
            F_cocon = Fmax_vec * af_cocon
            F_des = F_des + eta2 * F_cocon
        
        # === Activation back-projection ===
        a = force_to_activation_bisect(
            F_des, lenvel, self.env.muscle, flpe, Fmax_vec,
            iters=self.p.bisect_iters
        )
        
        a_min = float(getattr(self.env.muscle, "min_activation", 0.02))
        a = np.clip(a, a_min, 1.0)
        
        # === Saturation repair ===
        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = Fmax_vec * (af_now + flpe)
        F_pred = np.maximum(F_pred, 0.0)
        
        F_corr = saturation_repair_tau(
            -R, F_pred, a, a_min, 1.0, Fmax_vec, tau_des=tau_des
        )
        
        if np.any(np.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(
                F_corr, lenvel, self.env.muscle, flpe, Fmax_vec,
                iters=max(4, self.p.bisect_iters - 4)
            )
            a = np.clip(a, a_min, 1.0)
            af_now = active_force_from_activation(a, lenvel, self.env.muscle)
            F_pred = Fmax_vec * (af_now + flpe)
            F_pred = np.maximum(F_pred, 0.0)
        
        # Store for anti-windup
        tau_actual = (-R @ F_pred).reshape(-1)
        self.prev_tau_cmd = tau_des.copy()
        self.prev_tau_actual = tau_actual.copy()
        
        # Update reference trajectory
        qd_des = J_pinv @ xd_d_g
        self.qref = self.qref + qd_des * dt
        
        # === Diagnostics ===
        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ, alpha_J=alpha_J)
        ctrl_diag = pack_diag(
            e_x_norm=float(np.linalg.norm(e_x)),
            e_v_norm=float(np.linalg.norm(e_v)),
            int_norm=float(np.linalg.norm(self.integral_error)) if self.p.enable_integral else 0.0,
            tau_task_norm=float(np.linalg.norm(tau_task)),
            tau_null_norm=float(np.linalg.norm(tau_null)),
        )
        diag = merge_diag(
            kin_diag, dyn_diag, mus_diag, ctrl_diag,
            pack_diag(lam_os=lam_os, eta=eta, eta2=eta2)
        )
        
        return {
            "tau_des": tau_des,
            "R": R,
            "Fmax": Fmax_vec,
            "F_des": F_pred,
            "act": a,
            "q": q,
            "qd": qd,
            "x": x,
            "xd": xd,
            "xref_tuple": (x_d, xd_d_g, xdd_d_g),
            "eta": eta2,
            "diag": diag,
        }
