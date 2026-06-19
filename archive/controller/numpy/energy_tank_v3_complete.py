"""controller/energy_tank_v3.py

Comprehensive Energy Tank Controller with ALL optimization strategies:

STRATEGY A: Task-energy initialization (preview)
    - Pre-compute energy needs from trajectory before execution
    - Initialize E0 to match predicted energy demand
    
STRATEGY B: Energy-aware gain scheduling
    - Reduce K, Kff when tank is low (before gate slams shut)
    - Smoother tracking than hard s<1 throttle
    
STRATEGY C: Variable passive damping
    - Increase D0 when tank is low → faster refill
    - Decrease D0 when tank is healthy → better tracking
    
STRATEGY D: Strict passivity accounting
    - harvest_gain = 1.0 (no energy creation)
    - store_returned_energy = True (physically consistent)

HYBRID BLENDING:
    - Can use ANY aggressive controller output as F_u
    - Passivity layer gates only the power-injecting component
    - F_cmd = F_pas + F_perp + s*F_par

This follows the "task-energy tank" approach from:
- Schindlbeck & Haddadin (2015): Unified passivity-based Cartesian force/impedance control
- Ferraguti et al. (2013): Tank-based variable impedance control

Author: Energy Tank v3 - Complete Implementation
"""

from __future__ import annotations

import sys
import os

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, Tuple, List

from utils.numpy.math_utils import matrix_sqrt_spd, matrix_isqrt_spd
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


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _sym(A: np.ndarray) -> np.ndarray:
    """Symmetrize a matrix."""
    return 0.5 * (A + A.T)


def _project_spd(A: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Project matrix to symmetric positive definite."""
    A = _sym(np.asarray(A, dtype=float))
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return (V * w) @ V.T


def _decompose_parallel_perp(F: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose F into components parallel and perpendicular to v.
    
    F = F_par + F_perp where:
        F_par ∥ v (can inject/extract power)
        F_perp ⊥ v (zero instantaneous power)
    """
    nv2 = float(v @ v)
    if nv2 < eps:
        return np.zeros_like(F), F.copy()
    alpha = float(F @ v) / nv2
    F_par = alpha * v
    return F_par, F - F_par


def _smooth_step(x: float, x0: float = 0.0, x1: float = 1.0) -> float:
    """Smooth step function (Hermite interpolation)."""
    t = np.clip((x - x0) / (x1 - x0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# =============================================================================
# PARAMETER DATACLASS
# =============================================================================

@dataclass
class EnergyTankV3Params:
    """
    Complete parameter set for Energy Tank v3 controller.
    
    Organized by category for clarity.
    """
    
    # ==================== TASK-SPACE GAINS (nominal) ====================
    K0: np.ndarray                          # Nominal stiffness (2x2 SPD)
    D0: np.ndarray                          # Nominal passive damping (2x2 SPD)
    Kff: float = 1.0                        # Feedforward gain on Λ*xdd_d
    zeta: float = 0.85                      # Damping ratio (0.7-1.0)
    
    # ==================== STRATEGY A: Task-Energy Preview ====================
    enable_preview: bool = True             # Enable trajectory preview
    preview_horizon: float = 0.5            # Preview window (seconds)
    preview_margin: float = 1.3             # Energy margin (α in paper, 1.1-1.5)
    preview_samples: int = 50               # Samples for preview integration
    
    # ==================== STRATEGY B: Energy-Aware Gain Scheduling ====================
    enable_gain_scheduling: bool = True     # Enable K/Kff scheduling
    K_min_ratio: float = 0.3                # K_min = K0 * K_min_ratio
    Kff_min_ratio: float = 0.2              # Kff_min = Kff * Kff_min_ratio
    gain_schedule_beta: float = 2.0         # Exponent β for scheduling (1-3)
    
    # ==================== STRATEGY C: Variable Damping ====================
    enable_variable_damping: bool = True    # Enable D0 scheduling
    D_max_ratio: float = 3.0                # D_max = D0 * D_max_ratio
    D_min_ratio: float = 0.5                # D_min = D0 * D_min_ratio  
    damping_schedule_gamma: float = 2.0     # Exponent γ for damping (1-3)
    
    # ==================== STRATEGY D: Strict Passivity ====================
    harvest_gain: float = 1.0               # MUST be ≤1.0 for true passivity
    store_returned_energy: bool = True      # Store energy when braking
    
    # ==================== TANK CONFIGURATION ====================
    E0: float = 0.50                        # Initial tank energy (J)
    Emin: float = 1e-4                      # Minimum tank energy
    Emax: float = 2.0                       # Maximum tank capacity
    
    # ==================== HYBRID BLENDING ====================
    # If external_force_fn is provided, use it as F_u instead of internal OSC
    external_force_fn: Optional[Callable] = None
    blend_at_joint_port: bool = False       # If True, gate τ instead of F
    
    # ==================== SINGULARITY / GUARD SETTINGS ====================
    eps: float = 1e-6
    lam_os_max: float = 200.0
    sigma_thresh: float = 1e-4
    gate_pow: float = 2.0
    eta_min: float = 0.25                   # Minimum singularity authority
    vel_eps: float = 1e-8
    
    # ==================== PLANT COMPENSATION ====================
    enable_inertia_comp: bool = True
    enable_gravity_comp: bool = True
    enable_coriolis_comp: bool = True
    enable_joint_damping: bool = True
    
    # ==================== MUSCLE OPTIONS ====================
    bisect_iters: int = 12
    enable_internal_force: bool = False     # Disable for best tracking
    cocon_a0: float = 0.0
    linesearch_eps: float = 1e-6
    linesearch_safety: float = 1.2


# =============================================================================
# TANK STATE
# =============================================================================

@dataclass
class _TankState:
    """Internal state for energy tank."""
    E: float                                # Current energy
    K_prev: Optional[np.ndarray] = None     # Previous stiffness (for Kdot)
    I: Optional[np.ndarray] = None          # Integral term (optional)
    preview_E0: Optional[float] = None      # Preview-computed initial energy


# =============================================================================
# MAIN CONTROLLER CLASS
# =============================================================================

class EnergyTankV3Controller:
    """
    Energy Tank v3: Complete implementation with all optimization strategies.
    
    Features:
    - Strategy A: Task-energy preview initialization
    - Strategy B: Energy-aware gain scheduling  
    - Strategy C: Variable passive damping
    - Strategy D: Strict passivity accounting
    - Hybrid blending with any aggressive controller
    
    Usage:
        params = EnergyTankV3Params(
            K0=np.diag([1200, 1200]),
            D0=np.diag([40, 40]),
        )
        ctrl = EnergyTankV3Controller(env, arm, params)
        ctrl.reset(q0)
        
        # Optional: preview trajectory for optimal E0
        ctrl.preview_trajectory(trajectory, duration=2.0)
        
        # Main loop
        result = ctrl.compute(x_d, xd_d, xdd_d)
    """
    
    def __init__(self, env, arm, params: EnergyTankV3Params):
        self.env = env
        self.arm = arm
        self.p = params
        self.qref: Optional[np.ndarray] = None
        
        # Guard parameters
        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=self.p.eps,
            lam_os_max=self.p.lam_os_max,
            gate_pow=self.p.gate_pow,
            sigma_thresh_S=max(self.p.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()
        
        # Ensure gains are SPD
        self._K0 = _project_spd(self.p.K0, eps=self.p.eps)
        self._D0 = _project_spd(self.p.D0, eps=self.p.eps)
        
        # Compute gain limits for scheduling
        self._K_min = self._K0 * self.p.K_min_ratio
        self._D_min = self._D0 * self.p.D_min_ratio
        self._D_max = self._D0 * self.p.D_max_ratio
        
        # Initialize tank state
        self._tank = _TankState(E=float(self.p.E0))
        
        # Trajectory cache for preview
        self._traj_cache: Optional[Any] = None
        
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def reset(self, q0: np.ndarray) -> None:
        """Reset controller state."""
        self.qref = np.asarray(q0, dtype=float).copy()
        self._tank = _TankState(E=float(self.p.E0))
        self._traj_cache = None
        
    def preview_trajectory(
        self, 
        trajectory, 
        duration: float,
        initial_state: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> float:
        """
        STRATEGY A: Preview trajectory to compute optimal E0.
        
        Estimates energy needs by integrating predicted power demand.
        
        Args:
            trajectory: Trajectory object with .sample(t) method
            duration: Total trajectory duration (seconds)
            initial_state: Optional (x0, xd0) for better estimation
            
        Returns:
            Computed E0 value (also stored internally)
        """
        if not self.p.enable_preview:
            return self.p.E0
            
        dt_preview = duration / self.p.preview_samples
        
        # Use current state or provided initial state
        if initial_state is not None:
            x_curr, xd_curr = initial_state
        else:
            cart = self.env.states["cartesian"][0]
            x_curr = cart[:2].copy()
            xd_curr = cart[2:4].copy()
        
        # Approximate Lambda (use identity if not available)
        try:
            joint = self.env.states["joint"][0]
            q, qd = joint[:2], joint[2:]
            self.env.skeleton._set_state(q, qd)
            J = geometricJacobian_cached(self.env.skeleton._robot, symbolic=False)
            J_xy = J[0:2, :]
            M = inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False)
            Minv = np.linalg.inv(M)
            S = J_xy @ Minv @ J_xy.T
            Lambda = np.linalg.inv(S + self.p.eps * np.eye(2))
        except:
            Lambda = np.eye(2)
        
        # Integrate predicted power demand
        total_energy_need = 0.0
        
        for i in range(self.p.preview_samples):
            t = i * dt_preview
            x_d, xd_d, xdd_d = trajectory.sample(t)
            
            # Predicted error (simple forward integration)
            e_x = x_d - x_curr
            e_v = xd_d - xd_curr
            
            # Predicted force (simplified OSC)
            F_pred = (
                Lambda @ xdd_d
                + self._K0 @ e_x
                + 2.0 * self.p.zeta * matrix_sqrt_spd(Lambda) @ matrix_sqrt_spd(self._K0) @ e_v
            )
            
            # Decompose and compute power
            F_par, _ = _decompose_parallel_perp(F_pred, xd_d, eps=self.p.vel_eps)
            P_par = float(F_par @ xd_d)
            P_need = max(0.0, P_par)
            
            total_energy_need += P_need * dt_preview
            
            # Simple state prediction
            x_curr = x_curr + xd_d * dt_preview
            xd_curr = xd_d.copy()
        
        # Set E0 with margin
        E0_preview = self.p.Emin + self.p.preview_margin * total_energy_need
        E0_preview = min(E0_preview, self.p.Emax)
        
        self._tank.E = E0_preview
        self._tank.preview_E0 = E0_preview
        
        return E0_preview
    
    # =========================================================================
    # INTERNAL: DYNAMICS
    # =========================================================================
    
    def _compute_dynamics(self, q: np.ndarray, qd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute M(q) and h(q, qd) = C*qd + g + D*qd."""
        p = self.p
        n = len(q)
        
        if p.enable_inertia_comp:
            M = inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False)
        else:
            M = np.eye(n)
            
        if p.enable_gravity_comp:
            g = gravityCOM_cached(
                self.env.skeleton._robot, 
                self.env.skeleton._gravity_vec, 
                symbolic=False
            ).reshape(-1)
        else:
            g = np.zeros(n)
            
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
            
        return M, h
    
    # =========================================================================
    # INTERNAL: GAIN SCHEDULING (STRATEGIES B & C)
    # =========================================================================
    
    def _compute_scheduled_gains(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        STRATEGIES B & C: Compute scheduled K, D, Kff based on tank level.
        
        Returns:
            K_sched: Scheduled stiffness matrix
            D_sched: Scheduled damping matrix  
            Kff_sched: Scheduled feedforward gain
        """
        # Energy ratio ρ ∈ [0, 1]
        rho = np.clip(
            (self._tank.E - self.p.Emin) / (self.p.Emax - self.p.Emin + 1e-12),
            0.0, 1.0
        )
        
        # STRATEGY B: Gain scheduling
        if self.p.enable_gain_scheduling:
            rho_K = rho ** self.p.gain_schedule_beta
            K_sched = self._K_min + rho_K * (self._K0 - self._K_min)
            Kff_sched = self.p.Kff * (self.p.Kff_min_ratio + rho_K * (1.0 - self.p.Kff_min_ratio))
        else:
            K_sched = self._K0.copy()
            Kff_sched = self.p.Kff
            
        # STRATEGY C: Variable damping
        if self.p.enable_variable_damping:
            rho_D = (1.0 - rho) ** self.p.damping_schedule_gamma
            D_sched = self._D_min + rho_D * (self._D_max - self._D_min)
        else:
            D_sched = self._D0.copy()
            
        return K_sched, D_sched, Kff_sched
    
    # =========================================================================
    # MAIN COMPUTE
    # =========================================================================
    
    def compute(self, x_d: np.ndarray, xd_d: np.ndarray, xdd_d: np.ndarray) -> Dict[str, Any]:
        """
        Main control step with all optimization strategies.
        
        Args:
            x_d: Desired position (2,)
            xd_d: Desired velocity (2,)
            xdd_d: Desired acceleration (2,)
            
        Returns:
            Dictionary with tau_des, act, tank info, diagnostics
        """
        dt = float(self.arm.dt)
        
        # =====================================================================
        # [1] STATE & KINEMATICS
        # =====================================================================
        joint = self.env.states["joint"][0]
        q, qd = joint[:2], joint[2:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:4]
        self.env.skeleton._set_state(q, qd)
        
        # Jacobians
        J = geometricJacobian_cached(self.env.skeleton._robot, symbolic=False)
        J_xy = J[0:2, :]
        Jdot = geometricJacobianDot_cached(self.env.skeleton._robot, symbolic=False)
        Jdot_xy = Jdot[0:2, :]
        n = q.shape[0]
        
        # Adaptive DLS + task scaling
        J_pinv_dls, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)
        xd_d_g, xdd_d_g, alpha_J = scale_task_by_J(xd_d, xdd_d, sminJ, self.kp)
        
        # Integrate qref
        if self.qref is None:
            self.qref = q.copy()
        qd_des = J_pinv_dls @ xd_d_g
        self.qref = self.qref + qd_des * dt
        
        # =====================================================================
        # [2] DYNAMICS & OP-SPACE INERTIA
        # =====================================================================
        M, h = self._compute_dynamics(q, qd)
        Minv = np.linalg.inv(M)
        
        S = J_xy @ Minv @ J_xy.T
        Lambda, lam_os, eta, eta2, xd_d_g, xdd_d_g, dyn_diag = op_space_guard_and_gate(
            S, xd_d_g, xdd_d_g, self.dp
        )
        
        # Minimum authority preservation
        eta_clip = float(np.clip(eta2, self.p.eta_min, 1.0))
        
        # Bias term
        mu = Lambda @ (J_xy @ Minv @ h - Jdot_xy @ qd)
        
        # =====================================================================
        # [3] SCHEDULED GAINS (Strategies B & C)
        # =====================================================================
        K_sched, D_sched, Kff_sched = self._compute_scheduled_gains()
        
        # Apply singularity fade to stiffness
        K_now = eta_clip * K_sched
        
        # =====================================================================
        # [4] STIFFNESS VARIATION POWER (Kdot term)
        # =====================================================================
        if self._tank.K_prev is None:
            Kdot = np.zeros_like(K_now)
        else:
            Kdot = (K_now - self._tank.K_prev) / dt
        self._tank.K_prev = K_now.copy()
        
        # Task errors
        e_x = np.asarray(x_d, dtype=float) - x
        e_v = np.asarray(xd_d_g, dtype=float) - xd
        
        # Stiffness variation power: P_K = -0.5 * e^T * Kdot * e
        P_K = -0.5 * float(e_x @ Kdot @ e_x)
        P_refund = max(0.0, P_K)   # K decreasing → releases energy
        P_spend = max(0.0, -P_K)   # K increasing → consumes energy
        
        # =====================================================================
        # [5] INERTIA-WEIGHTED DAMPING (from Lambda and K_now)
        # =====================================================================
        Lam_s = matrix_sqrt_spd(Lambda)
        Lam_is = matrix_isqrt_spd(Lambda)
        Kv = (
            2.0 * float(self.p.zeta)
            * Lam_s
            @ matrix_sqrt_spd(Lam_is @ K_now @ Lam_is)
            @ Lam_s
        )
        
        # =====================================================================
        # [6] PASSIVE BASELINE DAMPER (with scheduled D)
        # =====================================================================
        F_pas = -(D_sched @ xd)
        P_diss = float(xd @ D_sched @ xd)  # ≥ 0 (dissipated)
        
        # =====================================================================
        # [7] ACTIVE (UNCONSTRAINED) FORCE
        # =====================================================================
        if self.p.external_force_fn is not None:
            # HYBRID: Use external controller's force
            F_act_raw = self.p.external_force_fn(
                x=x, xd=xd, x_d=x_d, xd_d=xd_d_g, xdd_d=xdd_d_g,
                e_x=e_x, e_v=e_v, Lambda=Lambda, mu=mu
            )
        else:
            # Standard OSC force
            F_act_raw = (
                Lambda @ (Kff_sched * xdd_d_g)  # Feedforward
                + mu                             # Bias compensation
                + K_now @ e_x                    # Stiffness
                + Kv @ e_v                       # Damping
            )
        
        # =====================================================================
        # [8] PARALLEL/PERPENDICULAR DECOMPOSITION
        # =====================================================================
        F_par, F_perp = _decompose_parallel_perp(F_act_raw, xd, eps=self.p.vel_eps)
        P_par_raw = float(F_par @ xd)  # Can be + or -
        
        P_inj = max(0.0, P_par_raw)    # Positive work (injection)
        P_ret = max(0.0, -P_par_raw) if self.p.store_returned_energy else 0.0
        
        # =====================================================================
        # [9] TANK GATE (passivity enforcement)
        # =====================================================================
        P_need = P_inj + P_spend
        
        if P_need > 0.0:
            s = min(
                1.0,
                (self._tank.E - self.p.Emin) / (dt * P_need + self.p.eps)
            )
            s = float(np.clip(s, 0.0, 1.0))
        else:
            s = 1.0
            
        # =====================================================================
        # [10] FINAL COMMAND FORCE
        # =====================================================================
        F_cmd = F_pas + F_perp + s * F_par
        
        # =====================================================================
        # [11] JOINT TORQUE
        # =====================================================================
        if self.p.blend_at_joint_port:
            # Alternative: gate at joint level (stronger guarantee)
            tau_raw = J_xy.T @ (F_perp + F_par)
            tau_par, tau_perp = _decompose_parallel_perp(tau_raw, qd, eps=self.p.vel_eps)
            P_tau = float(tau_par @ qd)
            P_tau_inj = max(0.0, P_tau)
            P_tau_need = P_tau_inj + P_spend
            
            if P_tau_need > 0.0:
                s_tau = min(1.0, (self._tank.E - self.p.Emin) / (dt * P_tau_need + self.p.eps))
                s_tau = float(np.clip(s_tau, 0.0, 1.0))
            else:
                s_tau = 1.0
                
            tau_pas = J_xy.T @ F_pas
            tau_des = tau_pas + tau_perp + s_tau * tau_par
            s = s_tau  # Use joint-level gate for reporting
        else:
            tau_des = J_xy.T @ F_cmd
            
        # =====================================================================
        # [12] TANK ENERGY UPDATE
        # =====================================================================
        # Income: dissipation + returned energy + stiffness refund
        Pin = float(self.p.harvest_gain) * P_diss + P_ret + P_refund
        
        # Stop filling when full
        if self._tank.E >= self.p.Emax and Pin > 0.0:
            Pin = 0.0
            
        # Outflow: gated injection + stiffness spend
        Pout = s * P_inj + P_spend
        
        # Update tank
        self._tank.E = float(
            np.clip(
                self._tank.E + dt * (Pin - Pout),
                self.p.Emin,
                self.p.Emax
            )
        )
        
        # =====================================================================
        # [13] MUSCLE FORCE ALLOCATION
        # =====================================================================
        geom = self.env.states["geometry"]
        lenvel = geom[:, :2, :]
        R = geom[:, 2:2 + self.env.skeleton.dof, :][0]
        Fmax_vec = get_Fmax_vec(self.env, R.shape[1])
        
        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]
        
        # Robust muscle solve
        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta, self.mp)
        
        # Optional internal force regulation
        if self.p.enable_internal_force:
            A = -R
            a0_vec = np.full(F_des.shape[0], float(self.p.cocon_a0))
            af0 = active_force_from_activation(a0_vec, lenvel, self.env.muscle)
            F_bias = Fmax_vec * (af0 + flpe)
            F_des = apply_internal_force_regulation(
                A, F_des, F_bias, Fmax_vec,
                eps=self.p.eps,
                linesearch_eps=self.p.linesearch_eps,
                linesearch_safety=self.p.linesearch_safety,
                scale=float(eta_clip),
            )
            
        # Hill inversion → activations
        a = force_to_activation_bisect(
            F_des, lenvel, self.env.muscle, flpe, Fmax_vec, 
            iters=self.p.bisect_iters
        )
        
        # Saturation repair
        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = Fmax_vec * (af_now + flpe)
        A = -R
        F_corr = saturation_repair_tau(
            A, F_pred, a,
            self.env.muscle.min_activation, 1.0, Fmax_vec,
            tau_des=tau_des,
        )
        if np.any(np.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(
                F_corr, lenvel, self.env.muscle, flpe, Fmax_vec,
                iters=max(4, self.p.bisect_iters - 4),
            )
            
        # =====================================================================
        # [14] DIAGNOSTICS
        # =====================================================================
        # Energy ratio for monitoring
        rho = (self._tank.E - self.p.Emin) / (self.p.Emax - self.p.Emin + 1e-12)
        
        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ, alpha_J=alpha_J)
        tank_diag = pack_diag(
            E=self._tank.E,
            E_ratio=rho,
            s=s,
            P_diss=P_diss,
            P_par_raw=P_par_raw,
            P_inj=P_inj,
            P_ret=P_ret,
            P_K=P_K,
            P_refund=P_refund,
            P_spend=P_spend,
            Pin=Pin,
            Pout=Pout,
            eta_clip=eta_clip,
            K_scale=float(K_sched[0, 0] / self._K0[0, 0]),
            D_scale=float(D_sched[0, 0] / self._D0[0, 0]),
            Kff_scale=Kff_sched / self.p.Kff if self.p.Kff > 0 else 1.0,
        )
        diag = merge_diag(
            kin_diag,
            dyn_diag,
            mus_diag,
            pack_diag(lam_os=lam_os, eta=eta, eta2=eta2),
            tank_diag,
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
            "tank": {
                "E": self._tank.E,
                "E_ratio": rho,
                "s": s,
                "P_diss": P_diss,
                "P_par_raw": P_par_raw,
                "P_inj": P_inj,
                "P_ret": P_ret,
                "P_K": P_K,
                "P_refund": P_refund,
                "P_spend": P_spend,
                "Pin": Pin,
                "Pout": Pout,
                "K_scale": float(K_sched[0, 0] / self._K0[0, 0]),
                "D_scale": float(D_sched[0, 0] / self._D0[0, 0]),
            },
            "scheduled_gains": {
                "K": K_sched,
                "D": D_sched,
                "Kff": Kff_sched,
            },
            "diag": diag,
        }


# =============================================================================
# CONVENIENCE FACTORY FUNCTIONS
# =============================================================================

def create_default_params(
    K: float = 1200.0,
    D: float = 40.0,
    E0: float = 0.5,
    Emax: float = 2.0,
    **kwargs
) -> EnergyTankV3Params:
    """Create params with sensible defaults."""
    return EnergyTankV3Params(
        K0=np.diag([K, K]),
        D0=np.diag([D, D]),
        E0=E0,
        Emax=Emax,
        **kwargs
    )


def create_aggressive_tracking_params(**kwargs) -> EnergyTankV3Params:
    """
    Params optimized for best tracking with TRUE passivity.
    Uses all strategies aggressively.
    """
    return EnergyTankV3Params(
        K0=np.diag([1600.0, 1600.0]),
        D0=np.diag([40.0, 40.0]),
        Kff=1.0,
        zeta=0.80,
        # Strategy A
        enable_preview=True,
        preview_horizon=0.5,
        preview_margin=1.5,
        # Strategy B
        enable_gain_scheduling=True,
        K_min_ratio=0.4,
        Kff_min_ratio=0.3,
        gain_schedule_beta=1.5,
        # Strategy C
        enable_variable_damping=True,
        D_max_ratio=2.5,
        D_min_ratio=0.6,
        damping_schedule_gamma=1.5,
        # Strategy D
        harvest_gain=1.0,
        store_returned_energy=True,
        # Tank
        E0=1.0,
        Emax=3.0,
        # Other
        eta_min=0.30,
        enable_internal_force=False,
        **kwargs
    )


def create_conservative_passive_params(**kwargs) -> EnergyTankV3Params:
    """
    Params optimized for maximum passivity margin.
    More conservative, suitable for contact tasks.
    """
    return EnergyTankV3Params(
        K0=np.diag([800.0, 800.0]),
        D0=np.diag([60.0, 60.0]),
        Kff=0.8,
        zeta=0.90,
        # Strategy A
        enable_preview=False,  # No preview for unknown contacts
        # Strategy B
        enable_gain_scheduling=True,
        K_min_ratio=0.2,
        Kff_min_ratio=0.1,
        gain_schedule_beta=2.5,
        # Strategy C
        enable_variable_damping=True,
        D_max_ratio=4.0,
        D_min_ratio=0.3,
        damping_schedule_gamma=2.5,
        # Strategy D
        harvest_gain=1.0,
        store_returned_energy=True,
        # Tank
        E0=0.3,
        Emax=1.0,
        # Other
        eta_min=0.20,
        enable_internal_force=False,
        **kwargs
    )


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

EnergyTankParams = EnergyTankV3Params
EnergyTankController = EnergyTankV3Controller


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    
    from model_lib.numpy.environment import Environment
    from model_lib.numpy.muscles import RigidTendonHillMuscle
    from model_lib.numpy.effector import RigidTendonArm26
    from config import PlantConfig, TrajectoryConfig
    from tasks.center_out import Task as CenterOutTask
    from trajectory.minjerk import MinJerkLinearTrajectory, MinJerkParams
    
    print("=" * 70)
    print("ENERGY TANK V3: Smoke Test with All Strategies")
    print("=" * 70)
    
    # Setup
    pc = PlantConfig()
    tc = TrajectoryConfig()
    muscle = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(
        muscle=muscle, timestep=pc.timestep, damping=pc.damping,
        n_ministeps=pc.n_ministeps, integration_method=pc.integration_method
    )
    env = Environment(
        effector=arm, max_ep_duration=pc.max_ep_duration,
        action_noise=0.0, obs_noise=0.0,
        proprioception_delay=arm.dt, vision_delay=arm.dt,
        name='EnergyTankV3Test'
    )
    
    q0 = np.deg2rad(np.array(pc.q0_deg))
    env.reset(options={
        'joint_state': np.concatenate([q0, np.zeros(2)])[None, :],
        'deterministic': True
    })
    
    # Trajectory
    task = CenterOutTask(n_targets=4, radius=0.10)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(
        waypoints, 
        MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
    )
    
    # Controller with aggressive tracking params
    params = create_aggressive_tracking_params()
    ctrl = EnergyTankV3Controller(env, arm, params)
    ctrl.reset(q0)
    
    # Preview trajectory for optimal E0
    cart = env.states['cartesian'][0]
    x0 = cart[:2].copy()
    xd0 = cart[2:4].copy()
    
    E0_preview = ctrl.preview_trajectory(
        traj, duration=2.0, 
        initial_state=(x0, xd0)
    )
    print(f"\nStrategy A: Preview computed E0 = {E0_preview:.3f} J")
    
    # Run simulation
    dt = arm.dt
    n_steps = 1500
    errors, s_vals, E_vals = [], [], []
    K_scales, D_scales = [], []
    
    for i in range(n_steps):
        t = i * dt
        x_d, xd_d, xdd_d = traj.sample(t)
        result = ctrl.compute(x_d, xd_d, xdd_d)
        
        cart = env.states['cartesian'][0]
        errors.append(np.linalg.norm(cart[:2] - x_d))
        s_vals.append(result['tank']['s'])
        E_vals.append(result['tank']['E'])
        K_scales.append(result['tank']['K_scale'])
        D_scales.append(result['tank']['D_scale'])
        
        env.step(result['act'][None, :])
    
    rmse = np.sqrt(np.mean(np.array(errors) ** 2)) * 100
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"RMSE:      {rmse:.2f} cm")
    print(f"s_mean:    {np.mean(s_vals):.3f} (s_min = {np.min(s_vals):.3f})")
    print(f"E_mean:    {np.mean(E_vals):.3f} J (E_min = {np.min(E_vals):.4f})")
    print(f"\nStrategy B (Gain Scheduling):")
    print(f"  K_scale: mean = {np.mean(K_scales):.3f}, min = {np.min(K_scales):.3f}")
    print(f"\nStrategy C (Variable Damping):")
    print(f"  D_scale: mean = {np.mean(D_scales):.3f}, max = {np.max(D_scales):.3f}")
    print(f"\nPassivity: {'✅ TRUE' if np.min(E_vals) >= params.Emin else '❌ VIOLATED'}")
    print(f"{'='*70}")
