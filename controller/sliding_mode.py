# control/sliding_mode.py
import numpy as np
from dataclasses import dataclass

# parity imports
from utils.math_utils import matrix_sqrt_spd, matrix_isqrt_spd  # noqa: F401
from utils.linear_utils import nnls_small_active_set            # noqa: F401

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
class SlidingModeParams:
    # Task-space SMC (2D)
    lambda_surf: np.ndarray   # (2,)
    K_switch:    np.ndarray   # (2,)
    phi:         np.ndarray   # (2,)
    Kff_x:       np.ndarray   # (2,)

    # kept for API parity (not used directly for fallback)
    Kp_q: np.ndarray
    Kd_q: np.ndarray

    # Guards / gates
    eps: float
    lam_os_max: float
    sigma_thresh: float
    gate_pow: float

    # Plant comp toggles
    enable_inertia_comp: bool
    enable_gravity_comp: bool
    enable_velocity_comp: bool
    enable_joint_damping: bool

    # Muscle / IF options
    enable_internal_force: bool
    cocon_a0: float
    bisect_iters: int
    linesearch_eps: float
    linesearch_safety: float


class SlidingModeController:
    """
    Unified robust controller:
      • OS-SMC + JS fallback blend (auto, based on η & cond(R))
      • Λ_reg regularization, α_J scaling for q_ref update
      • joint-limit barrier + small elbow-bias out of full extension
      • nullspace manipulability push (tiny, only when needed)
      • adaptive co-contraction near singularities
      • feasibility-aware torque clip (never below 30% global)
    """

    def __init__(self, env, arm, params: SlidingModeParams):
        self.env, self.arm, self.p = env, arm, params
        self.qref = None

        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=self.p.eps, lam_os_max=self.p.lam_os_max,
            gate_pow=self.p.gate_pow, sigma_thresh_S=max(self.p.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()

        dof = getattr(self.env.skeleton, "dof", 2)
        self._tau_clip  = 300.0
        self._tau_alpha = 1.0
        self._tau_filt  = np.zeros(dof)

        # ----- blending / gates -----
        self._eta_eq_floor = 0.50
        self._eta_sw_floor = 0.35
        self._cond_low     =  400 #600.0     # start weighting to joint-space here
        self._cond_high    =  2500 #5000.0    # full joint-space weight by here
        self._lam_os_floor = 1e-3

        # ----- joint-space fallback (simple PD on q* from DLS) -----
        self._Kp_js = np.array([30.0, 30.0])   # 25 25 joint fallback stiffness
        self._Kd_js = np.array([ 3.5,  3.5])   # 3 3 joint fallback damping
        self._k_pos_map = 1.4                  # 1.2 map x error to q* via J^+ gain

        # ----- nullspace / joint-limit helpers -----
        self._manip_gain     = 1.0 #0.8
        self._lim_margin_frac= 0.25 #0.20
        self._lim_eps        = 1e-4
        self._lim_power      = 2.0
        self._k_lim          = 14.0 #12.0
        self._kd_lim         = 0.6 #0.4
        self._k_sing_Kp      = 16.0 #12.0
        self._k_sing_Kd      = 0.9 #0.8
        self._elbow_bias_rad = 0.20 #0.17   # ~10°
        self._sing_gate_eta  = 0.9 #0.85

    # ---------- utils ----------
    @staticmethod
    def _sat(z): return np.clip(z, -1.0, 1.0)

    def _compute_dynamics(self, q, qd):
        p = self.p
        n = len(q)
        M = inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False) if p.enable_inertia_comp else np.eye(n)
        g = gravityCOM_cached(self.env.skeleton._robot, self.env.skeleton._gravity_vec, symbolic=False).reshape(-1) if p.enable_gravity_comp else np.zeros_like(q)
        C_any = centrifugalCoriolisCOM_cached(self.env.skeleton._robot, symbolic=False) if p.enable_velocity_comp else np.zeros((n, n))
        D = float(self.arm.damping) if p.enable_joint_damping else 0.0
        C_any = np.asarray(C_any)
        h = (C_any @ qd + g + D*qd) if C_any.ndim==2 else (C_any.reshape(-1) + g + D*qd)
        return M, h

    def _joint_limit_tau(self, q, qd):
        q_min = getattr(self.env.skeleton, "q_min", None)
        q_max = getattr(self.env.skeleton, "q_max", None)
        if q_min is None or q_max is None: return np.zeros_like(q)
        q_min = np.asarray(q_min).reshape(-1); q_max = np.asarray(q_max).reshape(-1)
        rng = q_max - q_min; margin = self._lim_margin_frac * rng
        inner_min, inner_max = q_min + margin, q_max - margin
        dmin = np.maximum(0.0, inner_min - q); dmax = np.maximum(0.0, q - inner_max)
        if not (np.any(dmin>0) or np.any(dmax>0)): return np.zeros_like(q)
        dmin_c = dmin + self._lim_eps; dmax_c = dmax + self._lim_eps
        grad = (1.0/(dmin_c**self._lim_power)) - (1.0/(dmax_c**self._lim_power))
        activity = (1.0/(dmin_c**self._lim_power)) + (1.0/(dmax_c**self._lim_power))
        return self._k_lim*grad - self._kd_lim*qd*np.clip(activity, 0.0, 1e6)

    def _singularity_bias_tau(self, q, qd, eta_val):
        if eta_val >= self._sing_gate_eta: return np.zeros_like(q)
        q_min = getattr(self.env.skeleton, "q_min", None)
        q_max = getattr(self.env.skeleton, "q_max", None)
        if q_min is not None and q_max is not None:
            q_center = 0.5*(np.asarray(q_min).reshape(-1)+np.asarray(q_max).reshape(-1))
        else:
            q_center = q.copy(); q_center[1] = self._elbow_bias_rad if q[1]>=0.0 else -self._elbow_bias_rad
        e = q_center - q
        return ((1.0 - eta_val)**2) * (self._k_sing_Kp*e - self._k_sing_Kd*qd)

    def _blend_weight(self, eta_val, condR):
        # w=0 → pure OS-SMC; w=1 → pure Joint fallback
        w_eta = np.clip(1.0 - float(eta_val), 0.0, 1.0)**2
        # normalize condR into [0,1] between thresholds
        cr = float(condR)
        w_cr = np.clip((cr - self._cond_low)/(self._cond_high - self._cond_low), 0.0, 1.0)
        return max(w_eta, w_cr)

    # ---------- main ----------
    def reset(self, q0):
        self.qref = q0.copy()
        self._tau_filt[...] = 0.0

    def compute(self, x_d, xd_d, xdd_d):
        # state
        joint = self.env.states["joint"][0]
        q, qd = joint[:2], joint[2:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:]
        self.env.skeleton._set_state(q, qd)
        if self.qref is None: self.qref = q.copy()

        # Jacobians
        J = geometricJacobian_cached(self.env.skeleton._robot, symbolic=False); J_xy = J[0:2, :]
        Jdot = geometricJacobianDot_cached(self.env.skeleton._robot, symbolic=False); Jdot_xy = Jdot[0:2, :]
        n = q.shape[0]

        # [1] kinematic guard + scaling
        J_pinv_dls, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)
        xd_d, xdd_d, alpha_J_raw = scale_task_by_J(xd_d, xdd_d, sminJ, self.kp)
        alpha_J = float(np.squeeze(np.asarray(alpha_J_raw))); alpha_J = np.clip(alpha_J, 0.0, 1.0)

        # gentle qref update
        qd_des = (0.6 * alpha_J) * (J_pinv_dls @ xd_d)
        self.qref = self.qref + qd_des * self.arm.dt

        # dynamics
        M, h = self._compute_dynamics(q, qd)

        # [2] op-space metrics + gating
        S = J_xy @ np.linalg.solve(M, J_xy.T)
        Lambda, lam_os, eta_raw, eta2, xd_d, xdd_d, dyn_diag = op_space_guard_and_gate(S, xd_d, xdd_d, self.dp)
        eta_val = float(np.squeeze(np.asarray(eta_raw)))
        lam_os_val = float(np.squeeze(np.asarray(lam_os))) + self._lam_os_floor

        # regularized Λ
        S_reg = S + lam_os_val * np.eye(2)
        Lambda_reg = np.linalg.inv(S_reg)

        # ----- OS-SMC torque -----
        mu   = Lambda_reg @ (J_xy @ np.linalg.solve(M, h) - Jdot_xy @ qd)
        F_eq = Lambda_reg @ (self.p.Kff_x * xdd_d) + mu
        e_x  = x - x_d
        e_v  = xd - xd_d
        s    = e_v + (self.p.lambda_surf * e_x)
        sw   = self._sat(s / (self.p.phi + 1e-12))
        eta_eq = max(eta_val, self._eta_eq_floor)
        eta_sw = max(eta_val, self._eta_sw_floor)
        F_sw   = -(self.p.K_switch * sw) * eta_sw
        F_task = (eta_eq * F_eq) + F_sw
        tau_os = J_xy.T @ F_task

        # ----- Joint-space fallback (when singular) -----
        # desired q* via DLS map of position error
        q_star = q + self._k_pos_map * (J_pinv_dls @ (x_d - x))
        qdd_js = self._Kp_js * (q_star - q) - self._Kd_js * qd
        tau_js = M @ qdd_js + h

        # ----- Blend OS & JS -----
        # condition on moment-arm matrix too (feasibility)
        geom = self.env.states["geometry"]; R = geom[:, 2:2 + self.env.skeleton.dof, :][0]
        condR = np.linalg.cond(R) if R.shape[0] >= 2 else 1.0
        w = self._blend_weight(eta_val, condR)
        tau_task = (1.0 - w) * tau_os + w * tau_js

        # ----- direct joint regulation -----
        tau_lim  = self._joint_limit_tau(q, qd)
        tau_bias = self._singularity_bias_tau(q, qd, eta_val)
        tau_ns   = np.zeros_like(q)
        if eta_val < 0.7:
            J_dyn_pinv = np.linalg.solve(M, J_xy.T @ Lambda_reg)
            N = np.eye(n) - J_dyn_pinv @ J_xy
            qdd_manip, _ = add_nullspace_manip(np.zeros(n), self.env, q, qd, J_xy, self.kp, eta_val)
            tau_ns = ((1.0 - eta_val)**2) * self._manip_gain * (N.T @ (M @ qdd_manip + h))

        tau_des = tau_task + tau_lim + tau_bias + tau_ns

        # ----- muscles / allocation -----
        lenvel = geom[:, :2, :]
        Fmax_vec = get_Fmax_vec(self.env, R.shape[1])

        # feasibility-aware torque clip
        tau_feas = (np.abs(R) @ Fmax_vec)
        tau_clip_dyn = 0.9 * float(np.min(tau_feas))
        tau_clip_use = (max(0.3*self._tau_clip, min(self._tau_clip, tau_clip_dyn))
                        if eta_val < 0.6 else self._tau_clip)

        tau_des = np.clip(tau_des, -tau_clip_use, tau_clip_use)
        self._tau_filt = (1 - self._tau_alpha) * self._tau_filt + self._tau_alpha * tau_des
        tau_des = self._tau_filt

        # allocation & optional adaptive internal force
        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]
        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta_val, self.mp)

        if self.p.enable_internal_force:
            scale_if = 0.0
            if eta_val < 0.6: scale_if = (0.6 - eta_val) * 0.6
            if condR > self._cond_high: scale_if = max(scale_if, 0.5)
            if scale_if > 0.0:
                a0_vec = np.full(F_des.shape[0], self.p.cocon_a0)
                af0    = active_force_from_activation(a0_vec, lenvel, self.env.muscle)
                F_bias = Fmax_vec * (af0 + flpe)
                F_des  = apply_internal_force_regulation(
                    -R, F_des, F_bias, Fmax_vec,
                    eps=self.p.eps, linesearch_eps=self.p.linesearch_eps,
                    linesearch_safety=self.p.linesearch_safety,
                    scale=float(np.clip(scale_if, 0.0, 0.6)),
                )

        a = force_to_activation_bisect(F_des, lenvel, self.env.muscle, flpe, Fmax_vec,
                                       iters=self.p.bisect_iters)
        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = Fmax_vec * (af_now + flpe)
        F_corr = saturation_repair_tau(-R, F_pred, a,
                                       self.env.muscle.min_activation, 1.0,
                                       Fmax_vec, tau_des=tau_des)
        if np.any(np.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(F_corr, lenvel, self.env.muscle, flpe, Fmax_vec,
                                           iters=max(4, self.p.bisect_iters - 4))

        # diag
        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ, alpha_J=alpha_J)
        sm_diag  = pack_diag(
            s1=float(s[0]), s2=float(s[1]),
            phi1=float(self.p.phi[0]), phi2=float(self.p.phi[1]),
            tau_clip=float(tau_clip_use), lam_os=float(lam_os_val),
            eta=float(eta_val), eta2=float(eta2), w_js=float(w), condR=float(condR),
        )
        diag = merge_diag(kin_diag, dyn_diag, mus_diag, sm_diag)

        return {
            "tau_des": tau_des, "R": R, "Fmax": Fmax_vec, "F_des": F_des, "act": a,
            "q": q, "qd": qd, "x": x, "xd": xd, "xref_tuple": (x_d, xd_d, xdd_d),
            "eta": eta2, "diag": diag,
        }

