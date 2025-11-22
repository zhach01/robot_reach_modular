#controllors/energy_tank_controller.py
import numpy as np
from dataclasses import dataclass
from utils.math_utils import matrix_sqrt_spd, matrix_isqrt_spd
from utils.linear_utils import (
    nnls_small_active_set,
)  # (not used now; keeping in case you switch back to weighted NNLS)
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
from utils.kinematics_guard import KinGuardParams, adaptive_dls_pinv, scale_task_by_J
from utils.dynamics_guard import DynGuardParams, op_space_guard_and_gate
from utils.muscle_guard import MuscleGuardParams, solve_muscle_forces
from utils.telemetry import pack_diag, merge_diag


@dataclass
class EnergyTankParams:
    # Gains
    D0: np.ndarray  # passive task-space damping (2x2)
    K0: np.ndarray  # nominal task-space stiffness (2x2)
    KI: np.ndarray  # integral gain (2,)
    Imax: np.ndarray  # int windup limits (2,)

    # Numerics / operational-space
    eps: float
    lam_os_smin_target: (
        float  # (kept for compatibility; not used directly—DynGuardParams handles reg)
    )
    lam_os_max: float
    sigma_thresh: float
    gate_pow: float

    # Plant compensation toggles
    enable_inertia_comp: bool
    enable_gravity_comp: bool
    enable_velocity_comp: bool
    enable_joint_damping: bool

    # Muscle & internal-force opts
    enable_internal_force: bool
    cocon_a0: float
    bisect_iters: int
    linesearch_eps: float
    linesearch_safety: float

    # Tank settings
    E0: float = 0.08
    Emin: float = 1e-4
    Emax: float = 0.5


@dataclass
class _TankState:
    E: float
    I: np.ndarray
    K_prev: np.ndarray


def _decompose_parallel_perp(F: np.ndarray, v: np.ndarray, eps: float = 1e-9):
    nv2 = float(v @ v)
    if nv2 < eps:
        return np.zeros_like(F), F
    alpha = float(F @ v) / nv2
    F_par = alpha * v
    return F_par, F - F_par


class EnergyTankController:
    """Passivity / energy-tank controller with the SAME public API as PDIFController."""

    def __init__(self, env, arm, params: EnergyTankParams):
        self.env = env
        self.arm = arm
        self.p = params
        self.qref = None

        # guard params (mirror PDIF style)
        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=self.p.eps,
            lam_os_max=self.p.lam_os_max,
            gate_pow=self.p.gate_pow,
            sigma_thresh_S=max(self.p.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()

        self._tank = _TankState(
            E=float(params.E0), I=np.zeros(2, dtype=float), K_prev=None
        )

    # --- public API (used by simulator) ---
    def reset(self, q0):
        self.qref = q0.copy()
        self._tank = _TankState(
            E=float(self.p.E0), I=np.zeros(2, dtype=float), K_prev=None
        )

    # --- internal helpers ---
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

    # --- main step ---
    def compute(self, x_d, xd_d, xdd_d):
        # --- state & kinematics ---
        joint = self.env.states["joint"][0]
        q, qd = joint[:2], joint[2:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:]
        self.env.skeleton._set_state(q, qd)

        J = geometricJacobian_cached(self.env.skeleton._robot, symbolic=False)
        J_xy = J[0:2, :]
        Jdot = geometricJacobianDot_cached(self.env.skeleton._robot, symbolic=False)
        Jdot_xy = Jdot[0:2, :]
        n = q.shape[0]

        # === Kinematics guard: [1a] adaptive DLS + [1b] scaling for qref integration
        J_pinv_dls, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)
        xd_d, xdd_d, alpha_J = scale_task_by_J(xd_d, xdd_d, sminJ, self.kp)
        qd_des = J_pinv_dls @ xd_d
        self.qref = self.qref + qd_des * self.arm.dt

        # --- dynamics
        M, h = self._compute_dynamics(q, qd)
        Minv = np.linalg.inv(M)

        # === Dynamics guard & gate: [2a]/[2b]/[2c]
        S = J_xy @ Minv @ J_xy.T
        Lambda, lam_os, eta, eta2, xd_d, xdd_d, dyn_diag = op_space_guard_and_gate(
            S, xd_d, xdd_d, self.dp
        )

        # task-space bias
        mu = Lambda @ (J_xy @ Minv @ h - Jdot_xy @ qd)

        # errors
        e_x = x_d - x
        e_v = xd_d - xd

        # passive baseline: dissipative port
        F_pas = -(self.p.D0 @ xd)
        P_diss = float(xd.T @ (self.p.D0 @ xd))  # ≥ 0

        # Critically-damped active damping using exact symmetric form
        Lam_s = matrix_sqrt_spd(Lambda)
        Lam_is = matrix_isqrt_spd(Lambda)
        Kv = 2.0 * Lam_s @ matrix_sqrt_spd(Lam_is @ self.p.K0 @ Lam_is) @ Lam_s

        # Integral term (anti-windup)
        self._tank.I = np.clip(
            self._tank.I + e_x * self.arm.dt, -self.p.Imax, self.p.Imax
        )
        F_I = self.p.KI * self._tank.I

        # variable stiffness (here: constant K0, but keep metering infra)
        K_now = self.p.K0
        if self._tank.K_prev is None:
            Kdot = np.zeros_like(K_now)
        else:
            Kdot = (K_now - self._tank.K_prev) / self.arm.dt
        self._tank.K_prev = K_now.copy()

        # parameter-variation power
        P_K = -0.5 * float(e_x.T @ (Kdot @ e_x))
        P_refund = max(0.0, P_K)  # K decreasing
        P_spend = max(0.0, -P_K)  # K increasing

        # active raw force (could inject energy)
        F_act_raw = (Lambda @ xdd_d) + mu + (self.p.K0 @ e_x) + (Kv @ e_v) + F_I

        # only parallel-to-velocity part can inject energy
        F_par, F_perp = _decompose_parallel_perp(F_act_raw, xd)
        P_par_raw = float(F_par.T @ xd)  # can be ±

        # --- tank gate ---
        P_need = max(0.0, P_par_raw) + P_spend
        if P_need > 0.0:
            s = min(
                1.0, (self._tank.E - self.p.Emin) / (self.arm.dt * P_need + self.p.eps)
            )
            s = max(0.0, s)
        else:
            s = 1.0

        # final command & torque
        F_cmd = F_pas + F_perp + s * F_par
        tau_des = J_xy.T @ F_cmd

        # tank update (stop depositing when full for exact accounting)
        Pin = P_diss + P_refund if (self._tank.E < self.p.Emax) else 0.0
        Pout = s * max(0.0, P_par_raw) + P_spend
        self._tank.E = float(
            np.clip(self._tank.E + self.arm.dt * (Pin - Pout), self.p.Emin, self.p.Emax)
        )

        # --- muscles / allocation ---
        geom = self.env.states["geometry"]
        lenvel = geom[:, :2, :]
        R = geom[:, 2 : 2 + self.env.skeleton.dof, :][0]
        Fmax_vec = get_Fmax_vec(self.env, R.shape[1])
        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]

        # Robust muscle solve (modular): [3a]/[3b]/[3c] using shared gate (eta)
        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta, self.mp)

        # internal-force regulation (optional; fade with eta2)
        if self.p.enable_internal_force:
            A = -R
            a0_vec = np.full(F_des.shape[0], self.p.cocon_a0)
            af = active_force_from_activation(a0_vec, lenvel, self.env.muscle)
            F_bias = Fmax_vec * (af + flpe)
            F_des = apply_internal_force_regulation(
                A,
                F_des,
                F_bias,
                Fmax_vec,
                eps=self.p.eps,
                linesearch_eps=self.p.linesearch_eps,
                linesearch_safety=self.p.linesearch_safety,
                scale=eta2,
            )

        # Hill inversion -> activations
        a = force_to_activation_bisect(
            F_des, lenvel, self.env.muscle, flpe, Fmax_vec, iters=self.p.bisect_iters
        )

        # one-step saturation repair
        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = Fmax_vec * (af_now + flpe)
        A = -R
        F_corr = saturation_repair_tau(
            A, F_pred, a, self.env.muscle.min_activation, 1.0, Fmax_vec, tau_des=tau_des
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

        # --- diagnostics (like PDIF)
        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ, alpha_J=alpha_J)
        tank_diag = pack_diag(
            E=self._tank.E,
            s=s,
            P_diss=P_diss,
            P_par_raw=P_par_raw,
            P_K=P_K,
            P_refund=P_refund,
            P_spend=P_spend,
        )
        diag = merge_diag(
            kin_diag,
            dyn_diag,
            mus_diag,
            pack_diag(lam_os=lam_os, eta=eta, eta2=eta2),
            tank_diag,
        )

        # return in the SAME shape as PDIF for plug-and-play
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
            "tank": {
                "E": self._tank.E,
                "s": s,
                "P_diss": P_diss,
                "P_par_raw": P_par_raw,
                "P_K": P_K,
                "P_refund": P_refund,
                "P_spend": P_spend,
            },
            "diag": diag,
        }
