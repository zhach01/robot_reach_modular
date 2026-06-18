# controller/energy_tank_bulletproof.py
"""
Bulletproof Energy-Tank (Passivity) Controller for the 2-DoF, 6-muscle arm.

Design goals:
1) Track well (target: < 1 cm RMS on random reach) while keeping a meaningful
   passivity guarantee at the task-space interaction port when enabled.
2) Be numerically safe (SPD projections, stable sqrt, robust parallel/perp split).
3) Preserve passivity at the task-space port (F_cmd, xdot) using either:
   - 'scaling': classic energy-tank scaling of the parallel (power-injecting) component
   - 'damping': PO/PC-style minimal correction by injecting extra damping along velocity
4) Be plug-and-play with your simulator:
   - reset(q0) creates qref
   - compute(x_d, xd_d, xdd_d) returns dict with tau_des, act, diag, etc.

This version adds PHYSIOLOGICAL SAFETY LAYERS (optional, default ON):
- ✅ Joint singularity avoidance on q2 (keep q2 at least margin_deg away from 0 and pi).
- ✅ mu clamp (task-space bias term) to prevent huge ||mu|| blow-ups.
- ✅ torque clamp (absolute Nm limit) in addition to muscle-feasibility clamp.

Progressive singularity margin:
- Set p.q2_margin_deg to 1,2,3,4,5 or call controller.set_singularity_margin_deg(deg)
- Use p.q2_margin_max_deg to cap the margin automatically.

Notes:
- "match_torch" controls whether mu is included inside the passivity-gated term.
  * match_torch=True  -> stricter passivity at the port but can be conservative
  * match_torch=False -> better tracking; passivity is weaker (mu is ungated)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

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


# ---------------------------- small utilities ----------------------------

def _sym(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def _project_spd_2x2(A: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Project a 2x2 matrix to SPD (for stable sqrt/isqrt)."""
    A = _sym(A)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return (V * w) @ V.T


def _decompose_parallel_perp(F: np.ndarray, v: np.ndarray, eps: float = 1e-12):
    """F = F_par + F_perp, where F_par || v and F_perp ⟂ v."""
    nv2 = float(v @ v)
    if nv2 < eps:
        return np.zeros_like(F), F
    alpha = float(F @ v) / nv2
    F_par = alpha * v
    return F_par, F - F_par


def _safe_cond(A: np.ndarray, fallback: float = 1e9) -> float:
    try:
        c = float(np.linalg.cond(A))
        return c if np.isfinite(c) else fallback
    except Exception:
        return fallback


def _blend_weight_from_cond(cond_val: float, lo: float, hi: float) -> float:
    return float(np.clip((cond_val - lo) / (hi - lo), 0.0, 1.0))


def _tau_cap_from_muscles(R: np.ndarray, Fmax: np.ndarray, margin: float = 0.95):
    """
    Conservative componentwise torque capacity from muscles:
      tau = (-R) F   with 0<=F<=Fmax
    => |tau_i| <= sum_j |R_{i,j}| Fmax_j
    """
    cap = (np.abs(R) @ np.asarray(Fmax, dtype=float).reshape(-1))
    return float(margin) * cap


def _clip_tau_componentwise(tau: np.ndarray, cap: np.ndarray):
    cap = np.asarray(cap, dtype=float).reshape(-1)
    return np.clip(np.asarray(tau, dtype=float).reshape(-1), -cap, cap)


def _clip_norm(v: np.ndarray, vmax: float, eps: float = 1e-12):
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if n <= vmax:
        return v, n, n
    s = float(vmax) / (n + eps)
    return s * v, n, float(np.linalg.norm(s * v))


def _to_vec2(x, default=(0.0, 0.0)) -> np.ndarray:
    try:
        v = np.array(x, dtype=float).ravel()
        if v.size == 1:
            return np.array([float(v[0]), float(v[0])], dtype=float)
        if v.size >= 2:
            return np.array([float(v[0]), float(v[1])], dtype=float)
    except Exception:
        pass
    return np.array(default, dtype=float)


def _deg2rad(d: float) -> float:
    return float(np.deg2rad(float(d)))


def _smooth_cubic_barrier(d: float, zone: float) -> float:
    """
    Smooth barrier shape: 0 at boundary, increasing with distance.
    Uses cubic ramp (C1 continuous at d=0):
      phi(d) = (d/zone)^2 * d  for d>0, else 0
    """
    if d <= 0.0:
        return 0.0
    z = max(zone, 1e-12)
    r = d / z
    return (r * r) * d


def _q2_singularity_barrier_tau(
    q2: float,
    q2d: float,
    margin: float,
    soft_zone: float,
    k: float,
    d: float,
    eps: float = 1e-12,
):
    """
    Barrier torque on joint2 to keep q2 away from 0 and pi by at least `margin`.
    Activates in a soft zone of width `soft_zone` beyond the margin.

    Low singularity (near 0):
      start at q2 < (margin + soft_zone)
      push +tau to increase q2

    High singularity (near pi):
      start at q2 > (pi - margin - soft_zone)
      push -tau to decrease q2
    """
    q2 = float(q2)
    q2d = float(q2d)

    m = max(float(margin), 0.0)
    z = max(float(soft_zone), 0.0)

    # If margin is too big, cap to something sensible (avoid negative safe region).
    m = min(m, np.pi * 0.49)

    tau = 0.0

    # ---- low side (near 0) ----
    low_start = m + z
    if q2 < low_start:
        dpos = low_start - q2  # >0 inside the soft zone
        # smooth "spring"
        tau += float(k) * _smooth_cubic_barrier(dpos, z + eps)
        # directional damping only if moving INTO the boundary (q2d < 0)
        tau += float(d) * max(0.0, -q2d)

    # ---- high side (near pi) ----
    high_start = (np.pi - m) - z
    if q2 > high_start:
        dpos = q2 - high_start  # >0 inside the soft zone
        # push negative
        tau -= float(k) * _smooth_cubic_barrier(dpos, z + eps)
        # damping only if moving INTO the boundary (q2d > 0)
        tau -= float(d) * max(0.0, q2d)

    return float(tau)


# ---------------------------- params & state ----------------------------

@dataclass
class EnergyTankParams:
    # Tracking stiffness & passive baseline damping (2x2, SPD recommended)
    K0: np.ndarray
    D0: np.ndarray

    # Feedforward scale on desired acceleration (scalar)
    Kff: float = 1.0

    # Inertia-weighted damping ratio (1.0 = critical, 0.7–0.9 often faster)
    zeta: float = 0.85

    # Optional integral term (generally OFF for min-jerk reach)
    KI: np.ndarray = None      # shape (2,)
    Imax: np.ndarray = None    # shape (2,)

    # Operational-space numerics / guards
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

    # Scale how much passive dissipation refills the tank (<=1 conservative)
    harvest_gain: float = 1.0

    # Store energy when gated force does negative work (helps recovery)
    store_returned_energy: bool = True

    # Keep minimum authority near singularities (helps reach targets)
    eta_min: float = 0.25

    # Velocity threshold for stable parallel/perp split
    v_eps: float = 1e-8

    # Torque feasibility clamp using muscle geometry (recommended ON)
    clamp_tau_to_muscles: bool = True
    tau_cap_margin: float = 0.95

    # --- Singularity safety: cond(J) blend to a joint-space fallback (same as the
    # sliding-mode / PD-IF controllers). Only active near the full-extension
    # Jacobian singularity (w>0), where near-singularity safety overrides strict
    # passivity (the q2 barrier already injects safety torque the same way). ---
    enable_singularity_blend: bool = True
    sing_cond_low: float = 20.0
    sing_cond_high: float = 80.0
    sing_elbow_min_rad: float = np.deg2rad(2.0)
    sing_Kp_js: np.ndarray = field(default_factory=lambda: np.array([12.0, 10.0]))
    sing_Kd_js: np.ndarray = field(default_factory=lambda: np.array([2.5, 2.0]))

    # Hard physiological torque clamp (Nm) (applied in addition to muscle clamp)
    clamp_tau_abs: bool = True
    tau_abs_max: np.ndarray = None  # (2,) or scalar; default set in __init__ if None

    # mu clamp to prevent huge task-space bias forces (N)
    clamp_mu: bool = True
    mu_max: float = 800.0

    # Singularity avoidance on q2 (elbow): keep q2 away from 0 and pi
    enforce_q2_margin: bool = True
    q2_index: int = 1
    q2_margin_deg: float = 2.0       # progressive value: set 1..5 in experiments
    q2_margin_max_deg: float = 5.0   # cap
    q2_soft_zone_deg: float = 3.0    # barrier starts at margin+soft_zone
    q2_barrier_k: float = 6000.0     # barrier stiffness
    q2_barrier_d: float = 120.0      # barrier damping

    # Passivity method:
    #  - 'scaling' : classic tank scaling of the parallel component
    #  - 'damping' : PO/PC-style minimal correction by extra damping along velocity
    passivity_method: str = "scaling"

    # If True: match Torch behavior more closely by INCLUDING mu inside the gated force.
    # If False: mu is always applied ungated (better tracking; weaker "strict" passivity).
    match_torch: bool = True

    # Muscle / internal-force options
    bisect_iters: int = 12
    enable_internal_force: bool = False
    cocon_a0: float = 0.0
    linesearch_eps: float = 1e-6
    linesearch_safety: float = 1.2


@dataclass
class _TankState:
    E: float
    I: np.ndarray
    K_prev: np.ndarray | None


# ---------------------------- controller ----------------------------

class EnergyTankController:
    """
    Passivity / energy-tank operational-space controller.
    Same API as PDIFController for plug-and-play with the simulator.
    """

    def __init__(self, env, arm, params: EnergyTankParams):
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

        # Normalize matrices to SPD for stable sqrt/isqrt
        self._K0 = _project_spd_2x2(self.p.K0, eps=self.p.eps)
        self._D0 = _project_spd_2x2(self.p.D0, eps=self.p.eps)

        # Default integral knobs if not provided
        if self.p.KI is None:
            self.p.KI = np.zeros(2, dtype=float)
        if self.p.Imax is None:
            self.p.Imax = np.full(2, 0.0, dtype=float)

        # Default absolute torque limits if not provided
        if self.p.tau_abs_max is None:
            self.p.tau_abs_max = np.array([90.0, 90.0], dtype=float)
        else:
            self.p.tau_abs_max = _to_vec2(self.p.tau_abs_max, default=(90.0, 90.0))

        self._tank = _TankState(
            E=float(self.p.E0),
            I=np.zeros(2, dtype=float),
            K_prev=None,
        )

    # --- convenience: progressive singularity margin ---
    def set_singularity_margin_deg(self, deg: float):
        """Set the q2 singularity margin (deg), clipped to [0, q2_margin_max_deg]."""
        m = float(deg)
        m = max(0.0, m)
        m = min(m, float(self.p.q2_margin_max_deg))
        self.p.q2_margin_deg = m

    # --- public API (used by simulator) ---
    def reset(self, q0):
        self.qref = np.asarray(q0, dtype=float).copy()
        self._tank = _TankState(
            E=float(self.p.E0),
            I=np.zeros(2, dtype=float),
            K_prev=None,
        )

    # --- internal helpers ---
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
            C_any = centrifugalCoriolisCOM_cached(
                self.env.skeleton._robot, symbolic=False
            )
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

    # --- main step ---
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
        eta_clip = float(np.clip(eta2, self.p.eta_min, 1.0))

        # OSC bias term
        mu_raw = Lambda @ (J_xy @ Minv @ h - Jdot_xy @ qd)

        # Clamp mu to avoid huge spikes (physiological sanity)
        if bool(self.p.clamp_mu) and float(self.p.mu_max) > 0.0:
            mu, mu_norm_raw, mu_norm = _clip_norm(mu_raw, float(self.p.mu_max), eps=float(self.p.eps))
        else:
            mu = np.asarray(mu_raw, dtype=float).reshape(-1)
            mu_norm_raw = float(np.linalg.norm(mu_raw))
            mu_norm = float(np.linalg.norm(mu))

        # Tracking errors (textbook)
        e_x = np.asarray(x_d, dtype=float) - x
        e_v = np.asarray(xd_d_g, dtype=float) - xd

        # Fade stiffness near singularities (recommended)
        K_now = eta_clip * self._K0

        # Kdot bookkeeping (tank accounting when K varies)
        if self._tank.K_prev is None:
            Kdot = np.zeros_like(K_now)
        else:
            Kdot = (K_now - self._tank.K_prev) / dt
        self._tank.K_prev = K_now.copy()

        # Power from stiffness variation: P_K = -0.5 e^T Kdot e
        P_K = -0.5 * float(e_x.T @ (Kdot @ e_x))
        P_refund = max(0.0, P_K)    # decreasing stiffness releases energy
        P_spend  = max(0.0, -P_K)   # increasing stiffness costs energy

        # Inertia-shaped damping Kv (SPD)
        Lam_s, Lam_is = matrix_sqrt_isqrt_spd(Lambda)  # one eigh, not two
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

        # ---- raw active force (potentially energy-injecting) ----
        F_act_raw = (
            (Lambda @ (float(self.p.Kff) * xdd_d_g))
            + (K_now @ e_x)
            + (Kv @ e_v)
            + F_I
        )
        if self.p.match_torch:
            F_act_raw = F_act_raw + mu

        # Robust direction for decomposition (avoid |xd| ~ 0 issues)
        v = np.asarray(xd, dtype=float)
        if float(v @ v) < float(self.p.v_eps):
            # add a small component along desired velocity error so "parallel" is meaningful
            v = v + np.sqrt(float(self.p.v_eps)) * np.asarray(e_v, dtype=float)

        F_par, F_perp = _decompose_parallel_perp(F_act_raw, v, eps=1e-12)
        P_par_raw = float(F_par.T @ v)  # power of the parallel component

        P_inj = max(0.0, P_par_raw)   # energy we would inject
        P_ret = max(0.0, -P_par_raw) if self.p.store_returned_energy else 0.0

        # ---- tank gate / passivity correction ----
        # Energy needed this step: injected work + stiffness spending
        P_need = P_inj + P_spend
        if P_need > 0.0:
            s = (self._tank.E - float(self.p.Emin)) / (dt * P_need + float(self.p.eps))
            s = float(np.clip(s, 0.0, 1.0))
        else:
            s = 1.0

        # Build the final commanded task force at the port
        d_pc = 0.0  # extra damping injected by PO/PC (if used)
        if str(self.p.passivity_method).lower() == "damping":
            # PO/PC-style: minimally modify force by adding extra damping along velocity
            # Build an ungated nominal force first.
            if self.p.match_torch:
                # mu is inside F_act_raw already (in F_par/F_perp)
                F_nom = F_pas + F_perp + F_par
            else:
                # mu applied ungated
                F_nom = mu + F_pas + F_perp + F_par

            P_nom = float(F_nom.T @ v)
            E_avail = max(0.0, float(self._tank.E - self.p.Emin))
            if (E_avail <= 0.0) and (P_nom > 0.0):
                # Need to remove power injection by adding damping along v
                vv = float(v @ v) + float(self.p.eps)
                # remove all injection when tank empty (strict)
                d_pc = float(P_nom) / vv
                F_cmd = F_nom - d_pc * v
                # set s=0 for accounting clarity (we're not "using" scaling)
                s = 0.0
                # recompute P_inj after correction (for tank update)
                P_inj_eff = 0.0
            else:
                # If we have energy, allow injection but cap by available energy this step
                if P_nom > 0.0:
                    # allowable injection power
                    P_allow = E_avail / (dt + float(self.p.eps))
                    if P_nom > P_allow:
                        vv = float(v @ v) + float(self.p.eps)
                        d_pc = float(P_nom - P_allow) / vv
                        F_cmd = F_nom - d_pc * v
                        P_inj_eff = P_allow
                    else:
                        F_cmd = F_nom
                        P_inj_eff = P_nom
                else:
                    F_cmd = F_nom
                    P_inj_eff = 0.0
        else:
            # Classic scaling of the power-injecting component (parallel part)
            if self.p.match_torch:
                # mu already included; do NOT add it again.
                F_cmd = F_pas + F_perp + s * F_par
            else:
                # mu is always applied ungated
                F_cmd = mu + F_pas + F_perp + s * F_par
            P_inj_eff = s * P_inj

        # Map to joint torques
        tau_des = J_xy.T @ F_cmd

        # ---------------- physiological joint singularity avoidance (q2) ----------------
        tau_barrier = np.zeros_like(tau_des)
        if bool(self.p.enforce_q2_margin):
            q2i = int(self.p.q2_index)
            if 0 <= q2i < q.shape[0]:
                margin_deg = float(np.clip(float(self.p.q2_margin_deg), 0.0, float(self.p.q2_margin_max_deg)))
                soft_deg = max(0.0, float(self.p.q2_soft_zone_deg))
                tau2 = _q2_singularity_barrier_tau(
                    q2=float(q[q2i]),
                    q2d=float(qd[q2i]),
                    margin=_deg2rad(margin_deg),
                    soft_zone=_deg2rad(soft_deg),
                    k=float(self.p.q2_barrier_k),
                    d=float(self.p.q2_barrier_d),
                    eps=float(self.p.eps),
                )
                # apply only on joint2 (q2 index)
                if q2i == 1 and tau_des.size >= 2:
                    tau_barrier = np.array([0.0, tau2], dtype=float)
                else:
                    tau_barrier = np.zeros_like(tau_des)
                    tau_barrier[q2i] = tau2
                tau_des = tau_des + tau_barrier

        # ---- singularity safety: cond(J) blend to a joint-space fallback ----
        # (same mechanism as the sliding-mode / PD-IF controllers). Active only
        # near the full-extension Jacobian singularity (cond(R) is blind to it).
        if getattr(self.p, "enable_singularity_blend", True):
            condJ = _safe_cond(J_xy)
            w_cr = _blend_weight_from_cond(condJ, self.p.sing_cond_low, self.p.sing_cond_high)
            w_eta = float(np.clip(1.0 - float(eta), 0.0, 1.0))
            w = float(np.clip(max(w_cr, 0.5 * w_eta), 0.0, 1.0))
            q_star = self.qref.copy()
            q_star[1] = max(float(q_star[1]), float(self.p.sing_elbow_min_rad))
            tau_js = self.p.sing_Kp_js * (q_star - q) + self.p.sing_Kd_js * (-qd)
            tau_des = (1.0 - w) * tau_des + w * tau_js

        # ---- torque feasibility clamps (muscle + absolute Nm limits) ----
        # Uses instantaneous muscle moment arms and Fmax.
        geom = self.env.states["geometry"]
        R = geom[:, 2 : 2 + self.env.skeleton.dof, :][0]
        Fmax_vec = getattr(self, "_Fmax_cache", None)
        if Fmax_vec is None:  # fixed by the muscle; cache on first use
            Fmax_vec = get_Fmax_vec(self.env, R.shape[1])
            self._Fmax_cache = Fmax_vec

        # Muscle-based conservative torque cap
        if bool(self.p.clamp_tau_to_muscles):
            tau_cap_mus = _tau_cap_from_muscles(R, Fmax_vec, margin=float(self.p.tau_cap_margin))
            tau_des = _clip_tau_componentwise(tau_des, tau_cap_mus)

        # Absolute physiological torque cap (Nm)
        if bool(self.p.clamp_tau_abs):
            tau_abs = _to_vec2(self.p.tau_abs_max, default=(90.0, 90.0))
            tau_des = _clip_tau_componentwise(tau_des, tau_abs)

        # ---- tank update ----
        # Fill: harvested passive dissipation + returned energy + stiffness refund
        # Spend: injected work (after correction) + stiffness spend
        Pin = float(self.p.harvest_gain) * P_diss + P_ret + P_refund
        Pout = float(P_inj_eff) + P_spend

        # stop filling when full (clean accounting)
        if self._tank.E >= float(self.p.Emax) and Pin > 0.0:
            Pin = 0.0

        self._tank.E = float(
            np.clip(self._tank.E + dt * (Pin - Pout), float(self.p.Emin), float(self.p.Emax))
        )

        # ---------------- muscle allocation & inversion ----------------
        lenvel = geom[:, :2, :]
        idx_flpe = getattr(self, "_idx_flpe", None)
        if idx_flpe is None:
            idx_flpe = self.env.muscle.state_name.index("force-length PE")
            self._idx_flpe = idx_flpe
        flpe = self.env.states["muscle"][0, idx_flpe, :]

        # Robust muscle solve with shared singularity gate (eta)
        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta, self.mp)

        # Optional internal-force regulation (co-contraction); fade with eta_clip
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

        # One-step saturation repair (makes tau consistent if muscle limits hit)
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
            s=s,
            passivity_method=str(self.p.passivity_method),
            d_pc=d_pc,
            P_diss=P_diss,
            P_par_raw=P_par_raw,
            P_inj=P_inj,
            P_inj_eff=P_inj_eff,
            P_ret=P_ret,
            P_K=P_K,
            P_refund=P_refund,
            P_spend=P_spend,
            eta_clip=eta_clip,
            match_torch=float(self.p.match_torch),
            mu_norm_raw=float(mu_norm_raw),
            mu_norm=float(mu_norm),
            tau_barrier_2=float(tau_barrier[1]) if tau_barrier.size >= 2 else 0.0,
            q2=float(q[1]) if q.size >= 2 else 0.0,
            q2_margin_deg=float(self.p.q2_margin_deg),
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
                "s": s,
                "passivity_method": str(self.p.passivity_method),
                "d_pc": d_pc,
                "P_diss": P_diss,
                "P_par_raw": P_par_raw,
                "P_inj": P_inj,
                "P_inj_eff": P_inj_eff,
                "P_ret": P_ret,
                "P_K": P_K,
                "P_refund": P_refund,
                "P_spend": P_spend,
            },
            "diag": diag,
        }


# Convenience alias (import-compatible with existing code)
OptimizedEnergyTankParams = EnergyTankParams
OptimizedEnergyTankController = EnergyTankController

