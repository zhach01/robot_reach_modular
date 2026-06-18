# controller/anfis_controller_v4_1.py
"""
ANFIS Controller v4.1 - Corrected Teacher Signal (DROP-IN, PROJECT-SAFE)

Key fixes vs your draft:
1) Teacher uses GUARDED desired accel: qdd_d computed from xdd_d_scaled (not raw).
2) Learning uses APPLIED ANFIS torque (after warmup + safety scaling), not raw tau_anfis.
3) Robust torque-from-muscles computation: tau = -R @ F with shape guards.
4) qref attribute always present for simulator logging.
5) Save/load rules for reuse: consequents + covariance P + MF params.

Notes:
- We do NOT rely on finite-diff qdd. Teacher uses computed-torque residual based on qdd_d from trajectory.
- We keep project convention: moment arms = -R, muscle torque tau = -R @ F.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from model_lib.numpy.skeleton import (
    geometricJacobian_cached,
    inertiaMatrixCOM_cached,
    centrifugalCoriolisCOM_cached,
    gravityCOM_cached,
)

from utils.numpy.kinematics_guard import KinGuardParams, adaptive_dls_pinv
from utils.numpy.dynamics_guard import DynGuardParams, op_space_guard_and_gate
from utils.numpy.muscle_guard import MuscleGuardParams, solve_muscle_forces
from utils.numpy.telemetry import pack_diag, merge_diag
from muscles.numpy.muscle_tools import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
)


# ----------------------------- Params ---------------------------------

@dataclass
class ANFISv41Params:
    # Rules + MFs
    n_mf: int = 5
    e_range: Tuple[float, float] = (-1.0, 1.0)     # joint error proxy [rad]
    ed_range: Tuple[float, float] = (-5.0, 5.0)    # joint error proxy [rad/s]

    # RLS
    forgetting_factor: float = 0.997
    P0_scale: float = 100.0
    rls_param_clip: float = 15.0

    # Learning schedule
    enable_learning: bool = True
    adapt_every: int = 2
    warmup_steps: int = 50

    # Model-based initialization
    omega_n: float = 15.0
    zeta: float = 0.8

    # Computed-torque teacher gains (inside teacher only, NOT runtime PD)
    teacher_Kp: float = 300.0
    teacher_Kd: float = 60.0

    # Feasibility anchoring
    anchor_rho: float = 0.3
    tau_teacher_clip: float = 80.0

    # Output limiting based on confidence
    tau_anfis_max: float = 50.0
    tau_anfis_min_conf: float = 15.0
    confidence_for_full_output: float = 0.5

    # Feedforward
    enable_gravity_comp: bool = True
    enable_coriolis_comp: bool = True
    tau_ff_max: float = 40.0

    # Safety scaling if task error explodes
    error_safety_thresh: float = 0.8   # task-space error norm (meters)
    error_safety_gain: float = 0.3

    # Guards / numerics
    eps: float = 1e-6
    gate_pow: float = 2.0
    sigma_thresh: float = 1e-4
    lam_os_max: float = 1e6

    # Muscle inversion
    bisect_iters: int = 16

    # Persistence
    rules_path: str = "ANFIS/saved_model/anfis_v41_rules.npz"
    autosave_every: int = 0  # 0 disables


# ------------------------- Membership Functions ------------------------

class GaussianMF:
    __slots__ = ("c", "sigma")

    def __init__(self, center: float, sigma: float):
        self.c = float(center)
        self.sigma = float(max(sigma, 1e-6))

    def __call__(self, x: float) -> float:
        x = float(x)
        z = (x - self.c) / self.sigma
        return float(np.exp(-0.5 * z * z))

    def pack(self):
        return (self.c, self.sigma)

    def unpack(self, t):
        self.c = float(t[0])
        self.sigma = float(t[1])


# ------------------------------ ANFIS Module ---------------------------

class ANFISModuleV41:
    """
    One joint module:
      input: (e, ed)
      output: tau

    Consequent per rule: f_r = p_r*e + q_r*ed + r_r
    """

    def __init__(
        self,
        n_mf: int,
        e_range: Tuple[float, float],
        ed_range: Tuple[float, float],
        forgetting_factor: float,
        P0_scale: float,
        inertia: float,
        omega_n: float,
        zeta: float,
    ):
        self.n_mf = int(n_mf)
        self.n_rules = self.n_mf ** 2
        self.n_params = self.n_rules * 3
        self.lambda_ff = float(forgetting_factor)

        self.e_range = tuple(map(float, e_range))
        self.ed_range = tuple(map(float, ed_range))

        self.mfs_e = self._init_mfs(self.e_range, self.n_mf)
        self.mfs_ed = self._init_mfs(self.ed_range, self.n_mf)

        # model-based init: Kp=I*wn^2, Kv=2*zeta*I*wn
        Kp_init = float(inertia) * float(omega_n) ** 2
        Kv_init = 2.0 * float(zeta) * float(inertia) * float(omega_n)

        self.consequent = np.zeros((self.n_rules, 3), dtype=float)
        self.consequent[:, 0] = Kp_init
        self.consequent[:, 1] = Kv_init
        self.consequent[:, 2] = 0.0

        self.P = float(P0_scale) * np.eye(self.n_params, dtype=float)

        self.n_updates = 0
        self.recent_err = []
        self.recent_mag = []

    def _init_mfs(self, r: Tuple[float, float], n: int) -> List[GaussianMF]:
        lo, hi = r
        centers = np.linspace(lo, hi, n)
        sigma = (hi - lo) / (n - 1 + 1e-6) * 0.7
        return [GaussianMF(float(c), float(sigma)) for c in centers]

    def forward(self, e: float, ed: float) -> Tuple[float, np.ndarray]:
        e = float(np.clip(e, self.e_range[0], self.e_range[1]))
        ed = float(np.clip(ed, self.ed_range[0], self.ed_range[1]))

        mu_e = np.array([mf(e) for mf in self.mfs_e], dtype=float)
        mu_ed = np.array([mf(ed) for mf in self.mfs_ed], dtype=float)

        w = np.outer(mu_e, mu_ed).reshape(-1)
        w_sum = float(np.sum(w) + 1e-10)
        w_bar = w / w_sum

        inp = np.array([e, ed, 1.0], dtype=float)
        f = self.consequent @ inp
        tau = float(w_bar @ f)

        phi = np.outer(w_bar, inp).reshape(-1)
        return tau, phi

    def update_rls(self, phi: np.ndarray, y_target: float, y_pred: float, clip: float):
        phi = np.asarray(phi, dtype=float).reshape(-1)
        if phi.shape[0] != self.n_params:
            return

        y_target = float(y_target)
        # RLS innovation must use the CURRENT MODEL prediction phi·theta (= the
        # raw rule output), not the attenuated/applied torque the caller passes.
        # Regressing against the warmup/safety-scaled applied torque made the
        # consequents compensate for the attenuation and biased the gains high
        # (audit H6). phi·theta exactly reproduces forward()'s raw `tau`.
        # (y_pred is kept in the signature for backward compatibility but unused.)
        err = y_target - float(phi @ self.consequent.reshape(-1))

        self.recent_err.append(abs(err))
        self.recent_mag.append(abs(y_target))
        if len(self.recent_err) > 100:
            self.recent_err.pop(0)
            self.recent_mag.pop(0)

        Pphi = self.P @ phi
        denom = float(self.lambda_ff + phi @ Pphi)
        if denom < 1e-10:
            return

        K = Pphi / denom

        delta = K * err
        dn = float(np.linalg.norm(delta))
        if dn > float(clip):
            delta *= float(clip) / (dn + 1e-12)

        theta = self.consequent.reshape(-1)
        theta = theta + delta
        self.consequent = theta.reshape(self.n_rules, 3)

        # stable covariance update
        phiTP = phi @ self.P
        self.P = (self.P - np.outer(K, phiTP)) / self.lambda_ff
        self.P = 0.5 * (self.P + self.P.T)

        # keep PD. eigvalsh returns ascending eigenvalues; a uniform identity
        # shift moves every eigenvalue by `shift`, so the post-shift max is just
        # eig[-1] + shift -> one eigendecomposition instead of two. Bit-identical.
        eig = np.linalg.eigvalsh(self.P)
        mine = float(eig[0])
        shift = 0.0
        if mine < 1e-6:
            shift = 1e-6 - mine + 1e-8
            self.P += shift * np.eye(self.n_params)
        maxe = float(eig[-1]) + shift
        if maxe > 1e5:
            self.P *= (1e5 / maxe)

        self.n_updates += 1

    def get_confidence(self) -> float:
        traceP = float(np.trace(self.P))
        conf_cov = 1.0 / (1.0 + traceP / self.n_params / 50.0)

        if len(self.recent_err) > 20:
            mean_err = float(np.mean(self.recent_err[-50:]))
            mean_mag = float(np.mean(self.recent_mag[-50:]) + 1e-6)
            rel = mean_err / mean_mag
            conf_err = 1.0 / (1.0 + 2.0 * rel)
        else:
            conf_err = 0.2

        return float(0.5 * conf_cov + 0.5 * conf_err)

    def get_effective_gains(self) -> Tuple[float, float]:
        return float(np.mean(self.consequent[:, 0])), float(np.mean(self.consequent[:, 1]))


# ----------------------------- Controller ------------------------------

class ANFISControllerV41:
    """
    Runtime control:
      tau_des = tau_ff + tau_anfis_applied

    Teacher (for learning):
      tau_teacher = (1-rho) * [ M*(qdd_d + Kd*ed + Kp*e) ] + rho * (tau_actual - tau_ff)

    IMPORTANT: learning is done on APPLIED tau_anfis (after warmup + safety scaling).
    """

    def __init__(self, env, arm, params: ANFISv41Params):
        self.env = env
        self.arm = arm
        self.p = params
        self.n_dof = 2
        self._dt = float(getattr(self.arm, "dt", 0.01))

        inertias = self._estimate_inertias()
        self.anfis = [
            ANFISModuleV41(
                n_mf=params.n_mf,
                e_range=params.e_range,
                ed_range=params.ed_range,
                forgetting_factor=params.forgetting_factor,
                P0_scale=params.P0_scale,
                inertia=inertias[i],
                omega_n=params.omega_n,
                zeta=params.zeta,
            )
            for i in range(self.n_dof)
        ]

        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=params.eps,
            lam_os_max=params.lam_os_max,
            gate_pow=params.gate_pow,
            sigma_thresh_S=max(params.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()

        # simulator expects qref
        self.qref: Optional[np.ndarray] = None
        self.step = 0

        # caches for learning
        self._last_phi = [None, None]
        self._last_tau_anfis_applied = None
        self._last_tau_ff = None
        self._last_e_joint = None
        self._last_ed_joint = None
        self._last_qdd_d = None

    # ------------------- persistence -------------------

    def save_rules(self, path: Optional[str] = None):
        path = path or self.p.rules_path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

        data = {
            "n_mf": int(self.p.n_mf),
            "e_range": np.array(self.p.e_range, dtype=float),
            "ed_range": np.array(self.p.ed_range, dtype=float),
        }

        for j in range(self.n_dof):
            data[f"consequent_{j}"] = self.anfis[j].consequent.copy()
            data[f"P_{j}"] = self.anfis[j].P.copy()
            data[f"mfs_e_{j}"] = np.array([mf.pack() for mf in self.anfis[j].mfs_e], dtype=float)
            data[f"mfs_ed_{j}"] = np.array([mf.pack() for mf in self.anfis[j].mfs_ed], dtype=float)

        np.savez(path, **data)

    def load_rules(self, path: Optional[str] = None) -> bool:
        path = path or self.p.rules_path
        if not os.path.exists(path):
            return False
        z = np.load(path, allow_pickle=True)

        for j in range(self.n_dof):
            kc = f"consequent_{j}"
            kP = f"P_{j}"
            ke = f"mfs_e_{j}"
            kd = f"mfs_ed_{j}"

            if kc in z:
                self.anfis[j].consequent = np.asarray(z[kc], dtype=float).copy()
            if kP in z:
                self.anfis[j].P = np.asarray(z[kP], dtype=float).copy()
            if ke in z:
                blob = np.asarray(z[ke], dtype=float)
                for i, mf in enumerate(self.anfis[j].mfs_e):
                    mf.unpack(tuple(blob[i]))
            if kd in z:
                blob = np.asarray(z[kd], dtype=float)
                for i, mf in enumerate(self.anfis[j].mfs_ed):
                    mf.unpack(tuple(blob[i]))

        return True

    # ------------------- helpers -------------------

    def _estimate_inertias(self) -> List[float]:
        try:
            q_nom = np.array([0.96, 1.13], dtype=float)
            self.env.skeleton._set_state(q_nom, np.zeros(2))
            M = np.asarray(inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False), dtype=float)
            return [max(0.1, float(M[0, 0])), max(0.05, float(M[1, 1]))]
        except Exception:
            return [0.3, 0.1]

    def reset(self, q0: np.ndarray):
        self.qref = np.asarray(q0, dtype=float).copy()
        self.step = 0
        self._last_phi = [None, None]
        self._last_tau_anfis_applied = None
        self._last_tau_ff = None
        self._last_e_joint = None
        self._last_ed_joint = None
        self._last_qdd_d = None

    def _compute_feedforward(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        tau_ff = np.zeros(self.n_dof, dtype=float)

        if self.p.enable_gravity_comp:
            g = gravityCOM_cached(
                self.env.skeleton._robot,
                self.env.skeleton._gravity_vec,
                symbolic=False,
            ).reshape(-1)
            tau_ff += np.asarray(g, dtype=float)[: self.n_dof]

        if self.p.enable_coriolis_comp and float(np.max(np.abs(qd))) > 0.05:
            C = np.asarray(centrifugalCoriolisCOM_cached(self.env.skeleton._robot, symbolic=False), dtype=float)
            if C.ndim == 2:
                tau_ff += (C @ qd).reshape(-1)[: self.n_dof]
            else:
                tau_ff += C.reshape(-1)[: self.n_dof]

        return np.clip(tau_ff, -self.p.tau_ff_max, self.p.tau_ff_max)

    def _get_anfis_output_limit(self, conf: float) -> float:
        if conf >= self.p.confidence_for_full_output:
            return float(self.p.tau_anfis_max)
        t = float(conf) / (float(self.p.confidence_for_full_output) + 1e-12)
        return float(self.p.tau_anfis_min_conf + t * (self.p.tau_anfis_max - self.p.tau_anfis_min_conf))

    def _get_error_safety_scale(self, err_norm: float) -> float:
        if err_norm < self.p.error_safety_thresh:
            return 1.0
        excess = err_norm - self.p.error_safety_thresh
        return float(
            self.p.error_safety_gain
            + (1.0 - self.p.error_safety_gain) * np.exp(-excess / (self.p.error_safety_thresh + 1e-12))
        )

    def _tau_from_R_F(self, R: np.ndarray, F: np.ndarray) -> np.ndarray:
        """
        Project convention:
          R extracted as (dof, n_muscles)
          moment arms = -R
          tau = -R @ F

        This function also handles accidental transposes.
        """
        R = np.asarray(R, dtype=float)
        F = np.asarray(F, dtype=float).reshape(-1)

        # expected: (2, m) @ (m,)
        if R.ndim == 2 and R.shape[0] == self.n_dof and R.shape[1] == F.shape[0]:
            return (-R @ F).reshape(-1)

        # if transposed: (m, 2) and F is (m,)
        if R.ndim == 2 and R.shape[1] == self.n_dof and R.shape[0] == F.shape[0]:
            return (-(R.T @ F)).reshape(-1)

        # fallback safe
        try:
            out = (-R @ F).reshape(-1)
            return out[: self.n_dof]
        except Exception:
            return np.zeros(self.n_dof, dtype=float)

    def _compute_teacher(
        self,
        q: np.ndarray,
        e_joint: np.ndarray,
        ed_joint: np.ndarray,
        qdd_d: np.ndarray,
        tau_ff: np.ndarray,
        tau_actual: np.ndarray,
    ) -> np.ndarray:
        # inertia
        M = np.asarray(inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False), dtype=float)

        # computed-torque residual (no C/g here, since tau_ff already has them):
        # tau_ct_res = M * (qdd_d + Kd*ed + Kp*e)
        fb_acc = qdd_d + self.p.teacher_Kd * ed_joint + self.p.teacher_Kp * e_joint
        tau_ct_res = (M @ fb_acc).reshape(-1)[: self.n_dof]
        tau_ct_res = np.clip(tau_ct_res, -self.p.tau_teacher_clip, self.p.tau_teacher_clip)

        # feasible residual
        tau_feas = (tau_actual - tau_ff).reshape(-1)[: self.n_dof]
        tau_feas = np.clip(tau_feas, -self.p.tau_teacher_clip, self.p.tau_teacher_clip)

        rho = float(np.clip(self.p.anchor_rho, 0.0, 1.0))
        tau_teacher = (1.0 - rho) * tau_ct_res + rho * tau_feas
        return tau_teacher

    # ------------------- main compute -------------------

    def compute(self, x_d: np.ndarray, xd_d: np.ndarray, xdd_d: np.ndarray) -> Dict[str, Any]:
        # state
        joint = self.env.states["joint"][0]
        q, qd = joint[:2], joint[2:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:]
        self.env.skeleton._set_state(q, qd)

        if self.qref is None:
            self.qref = np.asarray(q, dtype=float).copy()

        # task errors
        e_x = np.asarray(x_d - x, dtype=float)
        ed_x = np.asarray(xd_d - xd, dtype=float)
        err_norm = float(np.linalg.norm(e_x))

        # Jacobian and DLS pinv
        J = np.asarray(geometricJacobian_cached(self.env.skeleton._robot, symbolic=False), dtype=float)
        J_xy = J[0:2, :]
        J_pinv, sminJ, lamJ = adaptive_dls_pinv(J_xy, self.n_dof, self.kp)

        # operational-space guard FIRST (so teacher uses guarded accel)
        M = np.asarray(inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False), dtype=float)
        Minv = np.linalg.inv(M)
        S = J_xy @ Minv @ J_xy.T

        Lambda, lam_os, eta, eta2, xd_d_scaled, xdd_d_scaled, dyn_diag = op_space_guard_and_gate(
            S, xd_d.copy(), xdd_d.copy(), self.dp
        )

        # joint-space proxies (teacher and ANFIS inputs)
        e_joint = (J_pinv @ e_x).reshape(-1)
        ed_joint = (J_pinv @ ed_x).reshape(-1)

        # desired joint accel proxy from GUARDED xdd
        qdd_d = (J_pinv @ np.asarray(xdd_d_scaled, dtype=float)).reshape(-1)

        # feedforward
        tau_ff = self._compute_feedforward(q, qd)
        self._last_tau_ff = tau_ff.copy()

        # anfis raw + confidence gating
        tau_anfis_raw = np.zeros(self.n_dof, dtype=float)
        last_phi = [None, None]
        confs = [0.0, 0.0]

        for i in range(self.n_dof):
            tau_i, phi = self.anfis[i].forward(e_joint[i], ed_joint[i])
            conf = self.anfis[i].get_confidence()
            confs[i] = conf

            limit = self._get_anfis_output_limit(conf)
            tau_i = float(np.clip(tau_i, -limit, limit))

            tau_anfis_raw[i] = tau_i
            last_phi[i] = phi

        # safety scaling + warmup attenuation (APPLIED signal!)
        error_scale = self._get_error_safety_scale(err_norm)

        tau_anfis_applied = tau_anfis_raw * error_scale
        if self.step < int(self.p.warmup_steps):
            warm = float(self.step) / float(max(int(self.p.warmup_steps), 1))
            tau_anfis_applied *= warm

        # total torque command (joint torques)
        tau_des = tau_ff + tau_anfis_applied

        # ---------------- muscle allocation ----------------
        geom = self.env.states["geometry"]
        lenvel = geom[:, :2, :]
        R = geom[:, 2 : 2 + self.n_dof, :][0]  # expected (dof, n_muscles)

        Fmax_vec = get_Fmax_vec(self.env, R.shape[1])
        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta, self.mp)

        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]

        a = force_to_activation_bisect(
            F_des, lenvel, self.env.muscle, flpe, Fmax_vec,
            iters=self.p.bisect_iters
        )

        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = Fmax_vec * (af_now + flpe)

        F_corr = saturation_repair_tau(
            -R, F_pred, a,
            self.env.muscle.min_activation, 1.0,
            Fmax_vec, tau_des=tau_des
        )

        if np.any(np.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(
                F_corr, lenvel, self.env.muscle, flpe, Fmax_vec,
                iters=max(4, self.p.bisect_iters - 4)
            )

        # actual achieved torque (for anchoring)
        af_final = active_force_from_activation(a, lenvel, self.env.muscle)
        F_actual = Fmax_vec * (af_final + flpe)
        tau_actual = self._tau_from_R_F(R, F_actual)

        # ---------------- learning update ----------------
        self._last_phi = last_phi
        self._last_tau_anfis_applied = tau_anfis_applied.copy()
        self._last_e_joint = e_joint.copy()
        self._last_ed_joint = ed_joint.copy()
        self._last_qdd_d = qdd_d.copy()

        if (
            self.p.enable_learning
            and self.step >= int(self.p.warmup_steps)
            and (self.step % max(int(self.p.adapt_every), 1) == 0)
        ):
            tau_teacher = self._compute_teacher(
                q=q,
                e_joint=self._last_e_joint,
                ed_joint=self._last_ed_joint,
                qdd_d=self._last_qdd_d,
                tau_ff=self._last_tau_ff,
                tau_actual=tau_actual,
            )

            # clip teacher to safe range (already clipped, but keep)
            tau_teacher = np.clip(tau_teacher, -self.p.tau_teacher_clip, self.p.tau_teacher_clip)

            for i in range(self.n_dof):
                if self._last_phi[i] is None:
                    continue
                y_pred = float(self._last_tau_anfis_applied[i])  # IMPORTANT: applied!
                self.anfis[i].update_rls(
                    phi=self._last_phi[i],
                    y_target=float(tau_teacher[i]),
                    y_pred=y_pred,
                    clip=float(self.p.rls_param_clip),
                )

        if self.p.autosave_every and self.p.autosave_every > 0:
            if (self.step % int(self.p.autosave_every)) == 0:
                try:
                    self.save_rules(self.p.rules_path)
                except Exception:
                    pass

        self.step += 1

        # diagnostics
        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ)
        anfis_diag = pack_diag(
            confidence_0=float(confs[0]),
            confidence_1=float(confs[1]),
            tau_ff_0=float(tau_ff[0]),
            tau_ff_1=float(tau_ff[1]),
            tau_anfis_raw_0=float(tau_anfis_raw[0]),
            tau_anfis_raw_1=float(tau_anfis_raw[1]),
            tau_anfis_applied_0=float(tau_anfis_applied[0]),
            tau_anfis_applied_1=float(tau_anfis_applied[1]),
            error_scale=float(error_scale),
            n_updates=int(self.anfis[0].n_updates),
        )

        diag = merge_diag(
            kin_diag, dyn_diag, mus_diag,
            pack_diag(lam_os=lam_os, eta=eta, eta2=eta2),
            anfis_diag,
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
            "xref_tuple": (x_d, xd_d_scaled, xdd_d_scaled),
            "eta": eta2,
            "diag": diag,
            "tau_ff": tau_ff,
            "tau_anfis_raw": tau_anfis_raw,
            "tau_anfis_applied": tau_anfis_applied,
            "tau_actual": tau_actual,
        }

