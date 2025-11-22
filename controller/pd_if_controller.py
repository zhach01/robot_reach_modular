# controller/pd_if_controller.py
import numpy as np
from dataclasses import dataclass
from utils.math_utils import matrix_sqrt_spd, matrix_isqrt_spd
from utils.linear_utils import nnls_small_active_set
from muscles.muscle_tools import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
    apply_internal_force_regulation,
)
from model_lib.skeleton_numpy import (
    geometricJacobian_cached,
    geometricJacobianDot_cached,
    inertiaMatrixCOM_cached,
    centrifugalCoriolisCOM_cached,
    gravityCOM_cached,
)
from utils.kinematics_guard import (
    KinGuardParams,
    adaptive_dls_pinv,
    scale_task_by_J,
    add_nullspace_manip,
)
from utils.dynamics_guard import DynGuardParams, op_space_guard_and_gate
from utils.muscle_guard import MuscleGuardParams, solve_muscle_forces
from utils.telemetry import pack_diag, merge_diag


@dataclass
class PDIFParams:
    Kp_x: np.ndarray
    Kff_x: np.ndarray
    Kp_q: np.ndarray
    Kd_q: np.ndarray
    eps: float
    lam_os_smin_target: float
    lam_os_max: float
    sigma_thresh: float
    gate_pow: float
    enable_internal_force: bool
    enable_inertia_comp: bool
    enable_gravity_comp: bool
    enable_velocity_comp: bool
    enable_joint_damping: bool
    cocon_a0: float
    bisect_iters: int
    linesearch_eps: float
    linesearch_safety: float


class PDIFController:
    def __init__(self, env, arm, params: PDIFParams):
        self.env = env
        self.arm = arm
        self.p = params
        self.qref = None

        # guard params (you can expose/tune these)
        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=self.p.eps,
            lam_os_max=self.p.lam_os_max,
            gate_pow=self.p.gate_pow,
            sigma_thresh_S=max(self.p.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()

    def reset(self, q0):
        self.qref = q0.copy()

    def _compute_dynamics(self, q, qd):
        p = self.p
        n = len(q)

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

        if p.enable_velocity_comp:
            C_any = centrifugalCoriolisCOM_cached(
                self.env.skeleton._robot, symbolic=False
            )
        else:
            C_any = np.zeros((n, n))

        if p.enable_joint_damping:
            D = np.diag(np.full(n, self.arm.damping))
        else:
            D = np.zeros((n, n))

        C_any = np.asarray(C_any)
        if C_any.ndim == 2:
            h = C_any @ qd + g + D @ qd
        else:
            h = C_any.reshape(-1) + g + D @ qd
        return M, h

    def compute(self, x_d, xd_d, xdd_d):
        # --- state
        joint = self.env.states["joint"][0]
        q, qd = joint[:2], joint[2:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:]
        self.env.skeleton._set_state(q, qd)

        # --- Jacobians
        J = geometricJacobian_cached(self.env.skeleton._robot, symbolic=False)
        J_xy = J[0:2, :]
        Jdot = geometricJacobianDot_cached(self.env.skeleton._robot, symbolic=False)
        Jdot_xy = Jdot[0:2, :]
        n = q.shape[0]

        # ===== [1a] adaptive DLS on J; [1b] kinematic scaling
        J_pinv_dls, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)
        xd_d, xdd_d, alpha_J = scale_task_by_J(xd_d, xdd_d, sminJ, self.kp)

        qd_des = J_pinv_dls @ xd_d
        self.qref = self.qref + qd_des * self.arm.dt

        # --- dynamics terms
        M, h = self._compute_dynamics(q, qd)
        Minv = np.linalg.inv(M)

        # ===== [2a]/[2b]/[2c] op-space guard + gate (returns Λ, scaled cmds, and gate)
        S = J_xy @ Minv @ J_xy.T
        Lambda, lam_os, eta, eta2, xd_d, xdd_d, dyn_diag = op_space_guard_and_gate(
            S, xd_d, xdd_d, self.dp
        )

        # --- operational-space control law
        mu = Lambda @ (J_xy @ Minv @ h - Jdot_xy @ qd)
        e_x = x_d - x
        ed_x = xd_d - xd

        Kp_mat = np.diag(self.p.Kp_x)
        Lam_sqrt = matrix_sqrt_spd(Lambda)
        Lam_isqrt = matrix_isqrt_spd(Lambda)
        Kv_mat = (
            2.0 * Lam_sqrt @ matrix_sqrt_spd(Lam_isqrt @ Kp_mat @ Lam_isqrt) @ Lam_sqrt
        )

        xdd_r = (np.diag(self.p.Kff_x) @ xdd_d) + (Kv_mat @ ed_x) + (Kp_mat @ e_x)
        F_task = Lambda @ xdd_r + mu
        tau_task = J_xy.T @ F_task

        # --- nullspace policy
        J_dyn_pinv = Minv @ J_xy.T @ Lambda
        N = np.eye(n) - J_dyn_pinv @ J_xy

        qdd_post = self.p.Kp_q * (self.qref - q) - self.p.Kd_q * qd
        # ===== [1c] manipulability gradient in nullspace (faded by eta)
        qdd_post, k_manip_used = add_nullspace_manip(
            qdd_post, self.env, q, qd, J_xy, self.kp, eta
        )

        tau_post = M @ qdd_post + h
        # NOTE: keep the minus (back off nullspace near singularities with eta2)
        tau_des = tau_task - eta2 * (N.T @ tau_post)

        # --- muscle geometry & forces
        geom = self.env.states["geometry"]
        lenvel = geom[:, :2, :]
        R = geom[:, 2 : 2 + self.env.skeleton.dof, :][0]
        Fmax_vec = get_Fmax_vec(self.env, R.shape[1])

        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]

        # ===== [3a]/[3b]/[3c] robust muscle solve (WLS/NNLS) with shared gate
        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta, self.mp)

        # --- optional internal force regulation (scaled by eta2)
        if self.p.enable_internal_force:
            a0_vec = np.full(F_des.shape[0], self.p.cocon_a0)
            af = active_force_from_activation(a0_vec, lenvel, self.env.muscle)
            F_bias = Fmax_vec * (af + flpe)
            F_des = apply_internal_force_regulation(
                -R,
                F_des,
                F_bias,
                Fmax_vec,
                eps=self.p.eps,
                linesearch_eps=self.p.linesearch_eps,
                linesearch_safety=self.p.linesearch_safety,
                scale=eta2,
            )

        # --- activation back-projection & saturation repair
        a = force_to_activation_bisect(
            F_des, lenvel, self.env.muscle, flpe, Fmax_vec, iters=self.p.bisect_iters
        )

        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = Fmax_vec * (af_now + flpe)
        F_corr = saturation_repair_tau(
            -R,
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
                iters=max(4, self.p.bisect_iters - 4),
            )

        # ===== [5] diagnostics
        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ, alpha_J=None, k_manip=k_manip_used)
        diag = merge_diag(
            kin_diag, dyn_diag, mus_diag, pack_diag(lam_os=lam_os, eta=eta, eta2=eta2)
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
            "xref_tuple": (x_d, xd_d, xdd_d),
            "eta": eta2,
            "diag": diag,
        }
