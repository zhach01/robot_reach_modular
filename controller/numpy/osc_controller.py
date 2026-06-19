# controller/osc_controller.py
"""
Operational Space Control (OSC) - Self-Contained Version

This is a simplified, self-contained OSC controller for use with MotorNet training.
It includes all necessary utilities inline to avoid import dependencies.

Implements the complete Khatib (1987) framework:
- Operational-space inertia: Λ(x) = (J M⁻¹ J^T)⁻¹
- Dynamically consistent inverse: J̄ = M⁻¹ J^T Λ
- Null-space projector: N = I - J̄ J

Control law:
τ = J^T F + N^T τ₀

where F is the task-space force and τ₀ is the null-space control.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

# Robust muscle force->activation allocation -- the same solver used by the
# canonical PD/IF and sliding controllers (and the torch OSC port). The private
# `_solve_muscle_forces_simple` (damped pseudo-inverse) mis-allocated at far
# reaches, producing a sustained tracking offset that was *not* a singularity.
from utils.numpy.muscle_guard import MuscleGuardParams, solve_muscle_forces
from muscles.numpy.muscle_tools import (
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
)


# =============================================================================
# OSC Parameters
# =============================================================================

@dataclass
class OSCParams:
    """Parameters for Operational Space Control."""
    # Task-space gains
    Kp: float = 1000.0  # Position gain (N/m)
    Kv: float = 80.0    # Velocity gain (N·s/m)
    
    # Null-space posture control gains
    Kp_posture: float = 50.0
    Kv_posture: float = 10.0
    
    # Joint limit avoidance
    enable_joint_limits: bool = True
    joint_limit_gain: float = 20.0
    joint_limit_buffer: float = 0.1  # radians from limit
    
    # Damped least squares regularization
    damping: float = 0.01
    eps: float = 1e-6
    
    # Muscle allocation iterations
    bisect_iters: int = 12
    
    # Joint limits (default for 2-DOF planar arm)
    q_min: np.ndarray = field(default_factory=lambda: np.array([-0.5, 0.1]))
    q_max: np.ndarray = field(default_factory=lambda: np.array([2.5, 2.8]))


# =============================================================================
# Inline Utilities (to avoid external dependencies)
# =============================================================================

def _get_jacobian(skeleton_robot) -> np.ndarray:
    """Get geometric Jacobian from skeleton robot."""
    try:
        # Try cached version first
        from model_lib.numpy.skeleton import geometricJacobian_cached
        return geometricJacobian_cached(skeleton_robot, symbolic=False)
    except ImportError:
        pass
    
    # Fallback: direct computation
    try:
        from lib.kinematics.numpy.DifferentialHTM import geometricJacobian
        return geometricJacobian(skeleton_robot, symbolic=False)
    except ImportError:
        pass
    
    raise ImportError("Could not import Jacobian function")


def _get_inertia_matrix(skeleton_robot) -> np.ndarray:
    """Get inertia matrix from skeleton robot."""
    try:
        from model_lib.numpy.skeleton import inertiaMatrixCOM_cached
        return inertiaMatrixCOM_cached(skeleton_robot, symbolic=False)
    except ImportError:
        pass
    
    try:
        from lib.dynamics.numpy.DynamicsHTM import inertiaMatrixCOM
        return inertiaMatrixCOM(skeleton_robot, symbolic=False)
    except ImportError:
        pass
    
    raise ImportError("Could not import inertia matrix function")


def _get_coriolis(skeleton_robot) -> np.ndarray:
    """Get Coriolis/centrifugal matrix from skeleton robot."""
    try:
        from model_lib.numpy.skeleton import centrifugalCoriolisCOM_cached
        return centrifugalCoriolisCOM_cached(skeleton_robot, symbolic=False)
    except ImportError:
        pass
    
    try:
        from lib.dynamics.numpy.DynamicsHTM import centrifugalCoriolisCOM
        return centrifugalCoriolisCOM(skeleton_robot, symbolic=False)
    except ImportError:
        pass
    
    # Return zeros if not available
    return np.zeros((2, 2))


def _get_gravity(skeleton_robot, gravity_vec) -> np.ndarray:
    """Get gravity vector from skeleton robot."""
    try:
        from model_lib.numpy.skeleton import gravityCOM_cached
        return gravityCOM_cached(skeleton_robot, gravity_vec, symbolic=False).reshape(-1)
    except ImportError:
        pass
    
    try:
        from lib.dynamics.numpy.DynamicsHTM import gravityCOM
        return gravityCOM(skeleton_robot, gravity_vec, symbolic=False).reshape(-1)
    except ImportError:
        pass
    
    # Return zeros if not available (planar arm often has no gravity)
    return np.zeros(2)


def _get_Fmax_vec(env, n_muscles: int) -> np.ndarray:
    """Get maximum isometric force vector for muscles."""
    try:
        from muscles.numpy.muscle_tools import get_Fmax_vec
        return get_Fmax_vec(env, n_muscles)
    except ImportError:
        pass
    
    # Fallback: typical values for arm26 model
    try:
        Fmax = env.muscle.max_isometric_force
        if hasattr(Fmax, '__len__'):
            return np.array(Fmax)
        return np.full(n_muscles, Fmax)
    except:
        # Default values from arm26 model
        return np.array([800., 1000., 1000., 600., 600., 600.])


def _damped_pinv(J: np.ndarray, damping: float = 0.01) -> np.ndarray:
    """Compute damped pseudo-inverse."""
    JJT = J @ J.T
    n = JJT.shape[0]
    return J.T @ np.linalg.inv(JJT + damping * np.eye(n))


# =============================================================================
# OSC Controller
# =============================================================================

class OSCController:
    """
    Operational Space Controller.
    
    Provides complete task-space control with null-space projection
    for secondary objectives.
    """
    
    def __init__(self, env, arm, params: OSCParams = None):
        self.env = env
        self.arm = arm
        self.p = params or OSCParams()
        self.qref = None
        self.q_posture = None
        
        # Number of DOF and task dimensions
        self.n_dof = 2
        self.n_task = 2
        self.mp = MuscleGuardParams()
        
        # Previous Jacobian for Jdot estimation
        self.J_prev = None
        self.dt = getattr(arm, 'dt', 0.01)
        
        # Gravity vector (assume planar, no gravity in xy plane)
        self._gravity_vec = getattr(env.skeleton, '_gravity_vec', np.array([0, 0, 0]))
    
    def reset(self, q0: np.ndarray):
        """Reset controller state."""
        self.qref = np.asarray(q0).copy()
        self.q_posture = self.qref.copy()
        self.J_prev = None
    
    def set_posture(self, q_posture: np.ndarray):
        """Set desired posture for null-space control."""
        self.q_posture = np.asarray(q_posture).copy()
    
    def _compute_op_space_inertia(
        self, J: np.ndarray, M: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute operational-space inertia matrix."""
        Minv = np.linalg.inv(M)
        JMinv = J @ Minv
        Lambda_inv = JMinv @ J.T
        
        # Damped inverse for singularity robustness
        Lambda_inv_damped = Lambda_inv + self.p.damping * np.eye(self.n_task)
        Lambda = np.linalg.inv(Lambda_inv_damped)
        
        return Lambda, Lambda_inv
    
    def _compute_dynamically_consistent_inverse(
        self, J: np.ndarray, M: np.ndarray, Lambda: np.ndarray
    ) -> np.ndarray:
        """Compute dynamically consistent pseudo-inverse."""
        Minv = np.linalg.inv(M)
        J_bar = Minv @ J.T @ Lambda
        return J_bar
    
    def _compute_null_space_projector(
        self, J: np.ndarray, J_bar: np.ndarray
    ) -> np.ndarray:
        """Compute null-space projector."""
        N = np.eye(self.n_dof) - J_bar @ J
        return N
    
    def _compute_Jdot(self, J: np.ndarray) -> np.ndarray:
        """Estimate Jacobian time derivative via finite differences."""
        if self.J_prev is None:
            self.J_prev = J.copy()
            return np.zeros_like(J)
        
        Jdot = (J - self.J_prev) / self.dt
        self.J_prev = J.copy()
        return Jdot
    
    def _compute_joint_limit_torque(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Compute repulsive torque from joint limits."""
        if not self.p.enable_joint_limits:
            return np.zeros(self.n_dof)
        
        tau_lim = np.zeros(self.n_dof)
        
        for i in range(self.n_dof):
            dist_min = q[i] - self.p.q_min[i]
            dist_max = self.p.q_max[i] - q[i]
            
            if dist_min < self.p.joint_limit_buffer:
                tau_lim[i] += self.p.joint_limit_gain * (self.p.joint_limit_buffer - dist_min)
            
            if dist_max < self.p.joint_limit_buffer:
                tau_lim[i] -= self.p.joint_limit_gain * (self.p.joint_limit_buffer - dist_max)
        
        return tau_lim
    
    def _compute_posture_torque(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Compute posture control torque for null-space."""
        if self.q_posture is None:
            return np.zeros(self.n_dof)
        
        e_posture = self.q_posture - q
        tau_posture = self.p.Kp_posture * e_posture - self.p.Kv_posture * qd
        return tau_posture
    
    def compute(self, x_d: np.ndarray, xd_d: np.ndarray, xdd_d: np.ndarray) -> dict:
        """
        Compute control action using Operational Space Control.
        
        Args:
            x_d: Desired position (2,)
            xd_d: Desired velocity (2,)
            xdd_d: Desired acceleration (2,)
        
        Returns:
            dict with 'act' (muscle activations) and diagnostics
        """
        # Get current state
        joint = self.env.states["joint"][0]
        q, qd = joint[:2], joint[2:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:4]
        self.env.skeleton._set_state(q, qd)
        
        if self.qref is None:
            self.qref = q.copy()
        if self.q_posture is None:
            self.q_posture = self.qref.copy()
        
        # Get Jacobian
        J = _get_jacobian(self.env.skeleton._robot)
        J_xy = J[0:2, :]
        
        # Dynamics matrices
        M = _get_inertia_matrix(self.env.skeleton._robot)
        C = _get_coriolis(self.env.skeleton._robot)
        g = _get_gravity(self.env.skeleton._robot, self._gravity_vec)
        
        # Operational-space quantities
        Lambda, Lambda_inv = self._compute_op_space_inertia(J_xy, M)
        J_bar = self._compute_dynamically_consistent_inverse(J_xy, M, Lambda)
        N = self._compute_null_space_projector(J_xy, J_bar)
        
        # Jacobian derivative
        Jdot = self._compute_Jdot(J_xy)
        
        # Task-space error
        e_x = x_d - x
        e_v = xd_d - xd
        
        # Task-space acceleration command
        xdd_cmd = self.p.Kp * e_x + self.p.Kv * e_v + xdd_d
        
        # Operational-space bias force
        C = np.asarray(C)
        if C.ndim == 2:
            h = C @ qd + g
        else:
            h = C.reshape(-1) + g
        Minv = np.linalg.inv(M)
        mu = Lambda @ (J_xy @ Minv @ h - Jdot @ qd)
        
        # Task-space force
        F_task = Lambda @ xdd_cmd + mu
        
        # Null-space control
        tau_posture = self._compute_posture_torque(q, qd)
        tau_limits = self._compute_joint_limit_torque(q, qd)
        tau_null = tau_posture + tau_limits
        
        # Total joint torque
        tau_total = J_xy.T @ F_task + N.T @ tau_null
        
        # Get muscle geometry
        geom = self.env.states["geometry"]
        lenvel = geom[:, :2, :]
        R = geom[:, 2:2 + self.n_dof, :][0]
        Fmax_vec = _get_Fmax_vec(self.env, R.shape[1])
        
        # Cap torque to muscle capacity
        tau_cap = 0.95 * np.abs(R) @ Fmax_vec
        tau_capped = np.clip(tau_total, -tau_cap, tau_cap)
        
        # Muscle allocation (robust solver: weighted active-set, matches PD/IF,
        # sliding, and the torch OSC port).
        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]

        F_des, _ = solve_muscle_forces(tau_capped, R, Fmax_vec, 1.0, self.mp)
        a = force_to_activation_bisect(F_des, lenvel, self.env.muscle, flpe, Fmax_vec, iters=12)

        # Saturation repair: if the realized active force can't meet the demanded
        # torque, re-solve once against the achievable force.
        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = np.clip(Fmax_vec * (af_now + flpe), 0.0, None)
        a_min = float(getattr(self.env.muscle, "min_activation", 0.02))
        F_corr = saturation_repair_tau(-R, F_pred, a, a_min, 1.0, Fmax_vec, tau_des=tau_capped)
        F_corr = np.asarray(F_corr).reshape(-1)
        if np.any(np.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(F_corr, lenvel, self.env.muscle, flpe, Fmax_vec, iters=8)
        
        return {
            "tau_des": tau_total,
            "R": R,
            "Fmax": Fmax_vec,
            "F_des": F_des,
            "act": a,
            "q": q,
            "qd": qd,
            "x": x,
            "xd": xd,
            "xref_tuple": (x_d, xd_d, xdd_d),
            "Lambda": Lambda,
            "F_task": F_task,
            "tau_null": tau_null,
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("OSC Controller module loaded successfully!")
    print("Use: from controller.numpy.osc_controller import OSCController, OSCParams")
