# controller/energy_tank_hybrid.py
"""
Hybrid / Adaptive Energy-Tank Controller (tracking + passivity), NumPy version.

Goal:
- Keep the *true* energy-tank passivity mechanism (strict when rho_mu=1).
- Improve tracking by *adapting the demanded impedance* (K_now) based on:
    (i) tank energy E  (low E -> lower demanded stiffness),
    (ii) singularity gate eta2 (near singular -> lower K),
  instead of "cheating" by bypassing the tank gate.

Key ideas (math):
- Operational-space tracking "max authority":
    F_trk_raw = Λ (Kff xdd_d) + μ + K_now e_x + K_v e_v + F_I
    τ = J^T F
- Passivity at port (F, xdot): power P = F^T xdot
- Energy tank state E >= Emin:
    E_{k+1} = clip(E_k + dt*(P_in - P_out), Emin, Emax)
- Gate ONLY the power-injecting component (parallel to velocity):
    F_trk_raw = F_perp + F_par,   with F_par || v,  F_perp ⟂ v
    P_par_raw = F_par^T v
    P_inj = max(0, P_par_raw)
    Tank gate: s_tank = min(1, (E - Emin)/(dt*(P_inj + P_spend) + eps))
    F_cmd = F_pas + F_perp + s_tank F_par   (+ optional ungated part if rho_mu<1)

This file is meant to be drop-in with your simulator:
- reset(q0)
- compute(x_d, xd_d, xdd_d) -> dict with tau_des, act, diag, ...
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from utils.math_utils_numpy import matrix_sqrt_spd, matrix_isqrt_spd
from muscles.muscle_tools_numpy import (
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
from utils.kinematics_guard_numpy import KinGuardParams, adaptive_dls_pinv, scale_task_by_J
from utils.dynamics_guard_numpy import DynGuardParams, op_space_guard_and_gate
from utils.muscle_guard_numpy import MuscleGuardParams, solve_muscle_forces
from utils.telemetry_numpy import pack_diag, merge_diag


# ---------------------------- small utilities ----------------------------

def _sym(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def _project_spd_2x2(A: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    A = _sym(A)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return (V * w) @ V.T


def _smoothstep(a: float, b: float, x: float) -> float:
    if b <= a:
        return 1.0 if x >= b else 0.0
    t = (x - a) / (b - a)
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _decompose_parallel_perp(F: np.ndarray, v: np.ndarray, eps: float = 1e-12):
    """F = F_par + F_perp, where F_par || v and F_perp ⟂ v."""
    nv2 = float(v @ v)
    if nv2 < eps:
        return np.zeros_like(F), F
    alpha = float(F @ v) / nv2
    F_par = alpha * v
    return F_par, F - F_par


def _tau_cap_from_muscles(R: np.ndarray, Fmax: np.ndarray, margin: float = 0.95):
    """
    Conservative componentwise torque capacity from muscles:
      tau = R^T F   with 0<=F<=Fmax
    => |tau_i| <= sum_j |R_{i,j}| Fmax_j
    """
    cap = (np.abs(R) @ np.asarray(Fmax, dtype=float).reshape(-1))
    return float(margin) * cap


def _clip_tau_componentwise(tau: np.ndarray, cap: np.ndarray):
    cap = np.asarray(cap, dtype=float).reshape(-1)
    return np.clip(np.asarray(tau, dtype=float).reshape(-1), -cap, cap)


# ---------------------------- params & state ----------------------------

@dataclass
class HybridEnergyTankParams:
    # Nominal tracking impedance (2x2 SPD recommended)
    K0: np.ndarray
    D0: np.ndarray

    # Feedforward scale on desired accel
    Kff: float = 1.0

    # Inertia-shaped damping ratio (0.7–1.0)
    zeta: float = 0.85

    # Optional integral (usually OFF for min-jerk reach)
    KI: np.ndarray | None = None   # (2,)
    Imax: np.ndarray | None = None # (2,)

    # --- adaptive emphasis knobs (do NOT break tank) ---
    # Base “how conservative” we are: used to reduce demanded K when low energy / near singular.
    w_passivity_base: float = 0.25
    adapt_passivity: bool = True

    # Blend regions for energy and singularity
    E_blend_lo: float = 0.10
    E_blend_hi: float = 0.35
    eta_blend_lo: float = 0.35
    eta_blend_hi: float = 0.85

    # Minimum stiffness scaling when fully conservative
    K_scale_min: float = 0.35

    # How much of μ is included in the tank-gated term:
    # rho_mu=1 => strict passivity at (F,xdot) port (recommended).
    # rho_mu<1 => better tracking but can violate strict passivity (only for experiments).
    rho_mu: float = 1.0

    # Guards (operational-space)
    eps: float = 1e-6
    lam_os_max: float = 200.0
    sigma_thresh: float = 1e-4
    gate_pow: float = 2.0

    # Plant compensation toggles
    enable_inertia_comp: bool = True
    enable_gravity_comp: bool = True
    enable_coriolis_comp: bool = True
    enable_joint_damping: bool = True

    # Tank settings
    E0: float = 0.25
    Emin: float = 1e-4
    Emax: float = 1.50
    harvest_gain: float = 1.0
    store_returned_energy: bool = True
    eta_min: float = 0.25
    v_eps: float = 1e-8

    # Torque feasibility clamp
    clamp_tau_to_muscles: bool = True
    tau_cap_margin: float = 0.95

    # Muscle inversion + optional co-contraction/internal force
    bisect_iters: int = 12
    enable_internal_force: bool = False
    cocon_a0: float = 0.0
    linesearch_eps: float = 1e-6
    linesearch_safety: float = 1.2

    # Debug / trace
    store_trace: bool = True


@dataclass
class _TankState:
    E: float
    I: np.ndarray
    K_prev: np.ndarray | None


# ---------------------------- controller ----------------------------

class HybridEnergyTankController:
    """
    Hybrid tracking + energy tank passivity controller.

    Strict passivity note:
      If rho_mu=1.0 and you always apply s_tank on the parallel component,
      you preserve the classic energy-tank guarantee at the (F_cmd, xdot) port.
    """

    def __init__(self, env, arm, params: HybridEnergyTankParams):
        self.env = env
        self.arm = arm
        self.p = params
        self.qref = None

        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=float(self.p.eps),
            lam_os_max=float(self.p.lam_os_max),
            gate_pow=float(self.p.gate_pow),
            sigma_thresh_S=max(float(self.p.sigma_thresh), 1e-9),
        )
        self.mp = MuscleGuardParams()

        # Stable SPD matrices for sqrt/isqrt
        self._K0 = _project_spd_2x2(self.p.K0, eps=self.p.eps)
        self._D0 = _project_spd_2x2(self.p.D0, eps=self.p.eps)

        # Default integral if not provided
        if self.p.KI is None:
            self.p.KI = np.zeros(2, dtype=float)
        if self.p.Imax is None:
            self.p.Imax = np.zeros(2, dtype=float)

        self._tank = _TankState(
            E=float(self.p.E0),
            I=np.zeros(2, dtype=float),
            K_prev=None,
        )

        # optional per-step trace (lets you export without relying on Logs internals)
        self._trace = []  # list of dicts

    def reset(self, q0):
        self.qref = np.asarray(q0, dtype=float).copy()
        self._tank = _TankState(
            E=float(self.p.E0),
            I=np.zeros(2, dtype=float),
            K_prev=None,
        )
        self._trace = []

    def get_trace(self):
        return list(self._trace)

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

    def _adaptive_weights(self, E: float, eta2: float):
        """
        Returns:
          w_pass in [0,1] (more conservative when high),
          gK in [K_scale_min, 1] (stiffness scaling),
        """
        # high when energy is LOW
        wE = 1.0 - _smoothstep(self.p.E_blend_lo, self.p.E_blend_hi, float(E))
        # high when eta2 is LOW (near singular)
        wS = 1.0 - _smoothstep(self.p.eta_blend_lo, self.p.eta_blend_hi, float(eta2))
        if bool(self.p.adapt_passivity):
            w = max(float(self.p.w_passivity_base), float(wE), float(wS))
        else:
            w = float(self.p.w_passivity_base)
        w = float(np.clip(w, 0.0, 1.0))

        gK = 1.0 - w * (1.0 - float(self.p.K_scale_min))
        gK = float(np.clip(gK, float(self.p.K_scale_min), 1.0))
        return w, gK, wE, wS

    def compute(self, x_d, xd_d, xdd_d):
        dt = float(self.arm.dt)

        # --- state & kinematics ---
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

        # [1] Kinematics guard (adaptive DLS + scaling); integrate qref
        J_pinv_dls, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)
        xd_d_g, xdd_d_g, alpha_J = scale_task_by_J(xd_d, xdd_d, sminJ, self.kp)
        if self.qref is None:
            self.qref = q.copy()
        qd_des = J_pinv_dls @ xd_d_g
        self.qref = self.qref + qd_des * dt

        # --- dynamics
        M, h = self._compute_dynamics(q, qd)
        Minv = np.linalg.inv(M)

        # [2] Operational-space guard + gate
        S = J_xy @ Minv @ J_xy.T
        Lambda, lam_os, eta, eta2, xd_d_g, xdd_d_g, dyn_diag = op_space_guard_and_gate(
            S, xd_d_g, xdd_d_g, self.dp
        )
        eta_clip = float(np.clip(eta2, float(self.p.eta_min), 1.0))

        # OSC bias term (Coriolis/gravity in task-space)
        mu = Lambda @ (J_xy @ Minv @ h - Jdot_xy @ qd)

        # Tracking errors
        e_x = np.asarray(x_d, dtype=float) - x
        e_v = np.asarray(xd_d_g, dtype=float) - xd

        # --- adaptive gain scaling (tracking<->passivity emphasis) ---
        w_pass, gK, wE, wS = self._adaptive_weights(self._tank.E, eta_clip)

        # Fade stiffness near singularities and when tank energy is low
        K_now = eta_clip * gK * self._K0

        # Kdot accounting (important if K varies)
        if self._tank.K_prev is None:
            Kdot = np.zeros_like(K_now)
        else:
            Kdot = (K_now - self._tank.K_prev) / dt
        self._tank.K_prev = K_now.copy()

        # Stiffness power: P_K = -0.5 e^T Kdot e
        P_K = -0.5 * float(e_x.T @ (Kdot @ e_x))
        P_refund = max(0.0, P_K)
        P_spend  = max(0.0, -P_K)

        # Inertia-shaped damping Kv (SPD)
        Lam_s  = matrix_sqrt_spd(Lambda)
        Lam_is = matrix_isqrt_spd(Lambda)
        Kv = (
            2.0
            * float(self.p.zeta)
            * Lam_s
            @ matrix_sqrt_spd(Lam_is @ K_now @ Lam_is)
            @ Lam_s
        )

        # Passive baseline damper (always dissipative)
        F_pas = -(self._D0 @ xd)
        P_diss = float(xd.T @ (self._D0 @ xd))  # >= 0

        # Optional integral (anti-windup)
        if float(np.max(np.abs(self.p.Imax))) > 0.0 and float(np.max(np.abs(self.p.KI))) > 0.0:
            self._tank.I = np.clip(self._tank.I + e_x * dt, -self.p.Imax, self.p.Imax)
        else:
            self._tank.I[:] = 0.0
        F_I = self.p.KI * self._tank.I

        # ---- raw tracking force (potentially injects energy) ----
        # Split mu into gated and ungated parts via rho_mu
        rho_mu = float(np.clip(self.p.rho_mu, 0.0, 1.0))
        mu_g = rho_mu * mu
        mu_u = (1.0 - rho_mu) * mu  # may inject energy if rho_mu<1 (not strictly passive)

        F_trk_raw = (
            (Lambda @ (float(self.p.Kff) * xdd_d_g))
            + (K_now @ e_x)
            + (Kv @ e_v)
            + F_I
            + mu_g
        )

        # Robust direction for decomposition (avoid |xd| ~ 0 issues)
        v = np.asarray(xd, dtype=float)
        if float(v @ v) < float(self.p.v_eps):
            # make the direction meaningful using velocity error
            v = v + np.sqrt(float(self.p.v_eps)) * np.asarray(e_v, dtype=float)

        F_par, F_perp = _decompose_parallel_perp(F_trk_raw, v, eps=1e-12)
        P_par_raw = float(F_par.T @ v)

        P_inj = max(0.0, P_par_raw)
        P_ret = max(0.0, -P_par_raw) if bool(self.p.store_returned_energy) else 0.0

        # ---- tank gate (STRICT) ----
        P_need = P_inj + P_spend
        if P_need > 0.0:
            s_tank = (self._tank.E - float(self.p.Emin)) / (dt * P_need + float(self.p.eps))
            s_tank = float(np.clip(s_tank, 0.0, 1.0))
        else:
            s_tank = 1.0

        # Final force at the port (passive + perp + gated-parallel)
        F_cmd = mu_u + F_pas + F_perp + s_tank * F_par

        # Map to joint torques
        tau_des = J_xy.T @ F_cmd

        # ---- torque feasibility clamp ----
        geom = self.env.states["geometry"]
        R = geom[:, 2 : 2 + self.env.skeleton.dof, :][0]
        Fmax_vec = get_Fmax_vec(self.env, R.shape[1])
        if bool(self.p.clamp_tau_to_muscles):
            tau_cap = _tau_cap_from_muscles(R, Fmax_vec, margin=float(self.p.tau_cap_margin))
            tau_des = _clip_tau_componentwise(tau_des, tau_cap)

        # ---- tank update ----
        Pin = float(self.p.harvest_gain) * P_diss + P_ret + P_refund
        Pout = s_tank * P_inj + P_spend

        # stop filling when full
        if self._tank.E >= float(self.p.Emax) and Pin > 0.0:
            Pin = 0.0

        self._tank.E = float(
            np.clip(self._tank.E + dt * (Pin - Pout), float(self.p.Emin), float(self.p.Emax))
        )

        # ---------------- muscle allocation & inversion ----------------
        lenvel = geom[:, :2, :]
        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]

        # Robust muscle solve with shared gate (eta)
        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta, self.mp)

        # Optional internal-force regulation (co-contraction)
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

        # Hill inversion -> activations
        a = force_to_activation_bisect(
            F_des, lenvel, self.env.muscle, flpe, Fmax_vec, iters=int(self.p.bisect_iters)
        )

        # One-step saturation repair
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

        # ---------------- diagnostics ----------------
        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ, alpha_J=alpha_J)
        tank_diag = pack_diag(
            E=self._tank.E,
            s=s_tank,
            w_pass=w_pass,
            gK=gK,
            wE=wE,
            wS=wS,
            P_diss=P_diss,
            P_par_raw=P_par_raw,
            P_inj=P_inj,
            P_ret=P_ret,
            P_K=P_K,
            P_refund=P_refund,
            P_spend=P_spend,
            eta_clip=eta_clip,
            rho_mu=rho_mu,
        )
        diag = merge_diag(
            kin_diag,
            dyn_diag,
            mus_diag,
            pack_diag(lam_os=lam_os, eta=eta, eta2=eta2),
            tank_diag,
        )

        if bool(self.p.store_trace):
            err = float(np.linalg.norm(e_x))
            self._trace.append({
                "E": float(self._tank.E),
                "s": float(s_tank),
                "w_pass": float(w_pass),
                "gK": float(gK),
                "eta": float(eta_clip),
                "err": err,
                "P_diss": float(P_diss),
                "P_inj": float(P_inj),
                "P_spend": float(P_spend),
            })

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
                "E": float(self._tank.E),
                "s": float(s_tank),
                "w_pass": float(w_pass),
                "gK": float(gK),
                "eta": float(eta_clip),
            },
            "diag": diag,
        }


# import-friendly aliases
HybridEnergyTankController = HybridEnergyTankController
HybridEnergyTankParams = HybridEnergyTankParams

