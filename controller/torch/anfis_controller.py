# controller/torch/anfis_controller.py
# -*- coding: utf-8 -*-
"""
Pure-Torch ANFIS controller -- a faithful B=1 mirror of the canonical numpy
controller (controller/numpy/anfis_controller.py, the bake-off winner).

Same control law: tau_des = tau_ff + tau_anfis_applied, with a computed-torque
teacher learned online via RLS on each joint's (e, ed) -> tau fuzzy module.
Inherits the numpy controller's op-space guard + DLS-pinv singularity handling.

load_rules() reads the SAME anfis_rules.npz produced by the numpy controller, so
the torch port reuses the numpy-trained consequents (numpy<->torch parity).

Operates on B=1 state from the torch simulator. Uses the torch HTM dynamics
(env.skeleton.*) and the torch guard/muscle helpers.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from utils.torch.kinematics_guard import KinGuardParams, adaptive_dls_pinv
from utils.torch.dynamics_guard import DynGuardParams, op_space_guard_and_gate
from utils.torch.muscle_guard import MuscleGuardParams, solve_muscle_forces
from utils.torch.telemetry import pack_diag, merge_diag
from muscles.torch.muscle_tools import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
)


def _t(x, like: Tensor) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(x, device=like.device, dtype=like.dtype)


def _first(x: Tensor) -> Tensor:
    return x[0] if x.dim() >= 3 else x


@dataclass
class ANFISParams:
    # Rules + MFs
    n_mf: int = 5
    e_range: Tuple[float, float] = (-1.0, 1.0)
    ed_range: Tuple[float, float] = (-5.0, 5.0)

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

    # Computed-torque teacher gains
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
    error_safety_thresh: float = 0.8
    error_safety_gain: float = 0.3

    # Guards / numerics
    eps: float = 1e-6
    gate_pow: float = 2.0
    sigma_thresh: float = 1e-4
    lam_os_max: float = 1e6

    # Muscle inversion
    bisect_iters: int = 16

    # Persistence
    rules_path: str = "ANFIS/saved_model/anfis_rules.npz"
    autosave_every: int = 0


# ------------------------------ ANFIS Module ---------------------------

class ANFISModuleTorch:
    """One joint module: input (e, ed) -> tau. Consequent f_r = p_r*e + q_r*ed + r_r.
    Vectorized Gaussian MFs (numerically identical to the numpy per-MF loop)."""

    def __init__(self, n_mf, e_range, ed_range, forgetting_factor, P0_scale,
                 inertia, omega_n, zeta, device, dtype):
        self.device, self.dtype = device, dtype
        self.n_mf = int(n_mf)
        self.n_rules = self.n_mf ** 2
        self.n_params = self.n_rules * 3
        self.lambda_ff = float(forgetting_factor)
        self.e_range = tuple(map(float, e_range))
        self.ed_range = tuple(map(float, ed_range))

        self.c_e, self.s_e = self._init_mfs(self.e_range, self.n_mf)
        self.c_ed, self.s_ed = self._init_mfs(self.ed_range, self.n_mf)

        Kp_init = float(inertia) * float(omega_n) ** 2
        Kv_init = 2.0 * float(zeta) * float(inertia) * float(omega_n)
        self.consequent = torch.zeros((self.n_rules, 3), device=device, dtype=dtype)
        self.consequent[:, 0] = Kp_init
        self.consequent[:, 1] = Kv_init

        self.P = float(P0_scale) * torch.eye(self.n_params, device=device, dtype=dtype)
        self.n_updates = 0
        self.recent_err: List[float] = []
        self.recent_mag: List[float] = []

    def _init_mfs(self, r, n):
        lo, hi = r
        centers = torch.linspace(lo, hi, n, device=self.device, dtype=self.dtype)
        sigma = (hi - lo) / (n - 1 + 1e-6) * 0.7
        sigmas = torch.full((n,), max(sigma, 1e-6), device=self.device, dtype=self.dtype)
        return centers, sigmas

    def forward(self, e: Tensor, ed: Tensor) -> Tuple[Tensor, Tensor]:
        e = torch.clamp(e, self.e_range[0], self.e_range[1])
        ed = torch.clamp(ed, self.ed_range[0], self.ed_range[1])
        mu_e = torch.exp(-0.5 * ((e - self.c_e) / self.s_e) ** 2)        # (n_mf,)
        mu_ed = torch.exp(-0.5 * ((ed - self.c_ed) / self.s_ed) ** 2)    # (n_mf,)
        w = torch.outer(mu_e, mu_ed).reshape(-1)                         # (n_rules,)
        w_bar = w / (torch.sum(w) + 1e-10)
        inp = torch.stack([e, ed, torch.ones((), device=self.device, dtype=self.dtype)])
        f = self.consequent @ inp                                        # (n_rules,)
        tau = w_bar @ f
        phi = torch.outer(w_bar, inp).reshape(-1)                        # (n_params,)
        return tau, phi

    def update_rls(self, phi: Tensor, y_target: Tensor, clip: float):
        phi = phi.reshape(-1)
        if phi.shape[0] != self.n_params:
            return
        # innovation uses the CURRENT model prediction phi.theta (the raw rule
        # output), not the attenuated applied torque (audit H6 in numpy).
        theta = self.consequent.reshape(-1)
        err = y_target - (phi @ theta)

        self.recent_err.append(float(torch.abs(err)))
        self.recent_mag.append(float(torch.abs(y_target)))
        if len(self.recent_err) > 100:
            self.recent_err.pop(0)
            self.recent_mag.pop(0)

        Pphi = self.P @ phi
        denom = self.lambda_ff + (phi @ Pphi)
        if float(denom) < 1e-10:
            return
        K = Pphi / denom
        delta = K * err
        dn = float(torch.linalg.norm(delta))
        if dn > float(clip):
            delta = delta * (float(clip) / (dn + 1e-12))
        self.consequent = (theta + delta).reshape(self.n_rules, 3)

        phiTP = phi @ self.P
        self.P = (self.P - torch.outer(K, phiTP)) / self.lambda_ff
        self.P = 0.5 * (self.P + self.P.T)

        eig = torch.linalg.eigvalsh(self.P)
        mine = float(eig[0])
        shift = 0.0
        if mine < 1e-6:
            shift = 1e-6 - mine + 1e-8
            self.P = self.P + shift * torch.eye(self.n_params, device=self.device, dtype=self.dtype)
        maxe = float(eig[-1]) + shift
        if maxe > 1e5:
            self.P = self.P * (1e5 / maxe)
        self.n_updates += 1

    def get_confidence(self) -> float:
        traceP = float(torch.trace(self.P))
        conf_cov = 1.0 / (1.0 + traceP / self.n_params / 50.0)
        if len(self.recent_err) > 20:
            mean_err = float(np.mean(self.recent_err[-50:]))
            mean_mag = float(np.mean(self.recent_mag[-50:]) + 1e-6)
            conf_err = 1.0 / (1.0 + 2.0 * (mean_err / mean_mag))
        else:
            conf_err = 0.2
        return float(0.5 * conf_cov + 0.5 * conf_err)


# ----------------------------- Controller ------------------------------

class ANFISController:
    def __init__(self, env, arm, params: ANFISParams = None):
        self.env = env
        self.arm = arm
        self.p = params or ANFISParams()
        self.n_dof = int(env.skeleton.dof)
        # Follow the env/effector device+dtype (CPU or CUDA) -- do NOT assume CUDA,
        # or the ANFIS module tensors land on a different device than the state.
        self.dtype = getattr(env, "dtype", torch.get_default_dtype())
        self.device = getattr(env, "device", torch.device("cpu"))

        inertias = self._estimate_inertias()
        self.anfis = [
            ANFISModuleTorch(
                n_mf=self.p.n_mf, e_range=self.p.e_range, ed_range=self.p.ed_range,
                forgetting_factor=self.p.forgetting_factor, P0_scale=self.p.P0_scale,
                inertia=inertias[i], omega_n=self.p.omega_n, zeta=self.p.zeta,
                device=self.device, dtype=self.dtype,
            )
            for i in range(self.n_dof)
        ]

        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=self.p.eps, lam_os_max=self.p.lam_os_max, gate_pow=self.p.gate_pow,
            sigma_thresh_S=max(self.p.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()

        self.qref: Optional[Tensor] = None
        self.step = 0
        self._Fmax_cache: Optional[Tensor] = None
        self._idx_flpe: Optional[int] = None
        self._last_phi = [None, None]
        self._last_e_joint = None
        self._last_ed_joint = None
        self._last_qdd_d = None
        self._last_tau_ff = None
        self._last_tau_anfis_applied = None

    # ------------------- persistence (numpy-compatible npz) -------------------

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
            m = self.anfis[j]
            data[f"consequent_{j}"] = m.consequent.detach().cpu().numpy()
            data[f"P_{j}"] = m.P.detach().cpu().numpy()
            data[f"mfs_e_{j}"] = torch.stack([m.c_e, m.s_e], dim=1).detach().cpu().numpy()
            data[f"mfs_ed_{j}"] = torch.stack([m.c_ed, m.s_ed], dim=1).detach().cpu().numpy()
        np.savez(path, **data)

    def load_rules(self, path: Optional[str] = None) -> bool:
        path = path or self.p.rules_path
        if not os.path.exists(path):
            return False
        z = np.load(path, allow_pickle=True)

        def to_t(a):
            return torch.as_tensor(np.asarray(a, dtype=float), device=self.device, dtype=self.dtype)

        for j in range(self.n_dof):
            m = self.anfis[j]
            if f"consequent_{j}" in z:
                m.consequent = to_t(z[f"consequent_{j}"]).clone()
            if f"P_{j}" in z:
                m.P = to_t(z[f"P_{j}"]).clone()
            if f"mfs_e_{j}" in z:
                blob = to_t(z[f"mfs_e_{j}"])
                m.c_e, m.s_e = blob[:, 0].clone(), blob[:, 1].clone()
            if f"mfs_ed_{j}" in z:
                blob = to_t(z[f"mfs_ed_{j}"])
                m.c_ed, m.s_ed = blob[:, 0].clone(), blob[:, 1].clone()
        return True

    def get_rule_base(self, joint_idx: int = 0) -> np.ndarray:
        return self.anfis[joint_idx].consequent.detach().cpu().numpy()

    # ------------------- helpers -------------------

    def _estimate_inertias(self) -> List[float]:
        try:
            q_nom = torch.tensor([0.96, 1.13], device=self.device, dtype=self.dtype)
            M = _first(self.env.skeleton.mass_matrix(q_nom))
            return [max(0.1, float(M[0, 0])), max(0.05, float(M[1, 1]))]
        except Exception:
            return [0.3, 0.1]

    def reset(self, q0: Tensor):
        q0 = q0.to(device=self.device, dtype=self.dtype)
        self.qref = (q0[0] if q0.ndim == 2 else q0).clone()
        self.step = 0
        self._last_phi = [None, None]
        self._last_e_joint = self._last_ed_joint = self._last_qdd_d = None
        self._last_tau_ff = self._last_tau_anfis_applied = None

    def _compute_feedforward(self, q, qd, g, C) -> Tensor:
        tau_ff = torch.zeros(self.n_dof, device=q.device, dtype=q.dtype)
        if self.p.enable_gravity_comp:
            tau_ff = tau_ff + g[: self.n_dof]
        if self.p.enable_coriolis_comp and float(torch.max(torch.abs(qd))) > 0.05:
            tau_ff = tau_ff + (C @ qd).reshape(-1)[: self.n_dof]
        return torch.clamp(tau_ff, -self.p.tau_ff_max, self.p.tau_ff_max)

    def _get_anfis_output_limit(self, conf: float) -> float:
        if conf >= self.p.confidence_for_full_output:
            return float(self.p.tau_anfis_max)
        t = float(conf) / (float(self.p.confidence_for_full_output) + 1e-12)
        return float(self.p.tau_anfis_min_conf + t * (self.p.tau_anfis_max - self.p.tau_anfis_min_conf))

    def _get_error_safety_scale(self, err_norm: float) -> float:
        if err_norm < self.p.error_safety_thresh:
            return 1.0
        excess = err_norm - self.p.error_safety_thresh
        return float(self.p.error_safety_gain + (1.0 - self.p.error_safety_gain)
                     * np.exp(-excess / (self.p.error_safety_thresh + 1e-12)))

    def _compute_teacher(self, M, e_joint, ed_joint, qdd_d, tau_ff, tau_actual) -> Tensor:
        fb_acc = qdd_d + self.p.teacher_Kd * ed_joint + self.p.teacher_Kp * e_joint
        tau_ct_res = (M @ fb_acc).reshape(-1)[: self.n_dof]
        tau_ct_res = torch.clamp(tau_ct_res, -self.p.tau_teacher_clip, self.p.tau_teacher_clip)
        tau_feas = (tau_actual - tau_ff).reshape(-1)[: self.n_dof]
        tau_feas = torch.clamp(tau_feas, -self.p.tau_teacher_clip, self.p.tau_teacher_clip)
        rho = float(np.clip(self.p.anchor_rho, 0.0, 1.0))
        return (1.0 - rho) * tau_ct_res + rho * tau_feas

    # ------------------- main compute -------------------

    def compute(self, x_d: Tensor, xd_d: Tensor, xdd_d: Tensor) -> Dict[str, Any]:
        joint = self.env.states["joint"]
        n = self.n_dof
        q, qd = joint[0, :n], joint[0, n:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:4]
        self.env.skeleton._set_state(q, qd)
        if self.qref is None:
            self.qref = q.clone()

        x_d = _t(x_d, x).reshape(-1)[:2]
        xd_d = _t(xd_d, x).reshape(-1)[:2]
        xdd_d = _t(xdd_d, x).reshape(-1)[:2]

        e_x = x_d - x
        ed_x = xd_d - xd
        err_norm = float(torch.linalg.norm(e_x))

        # kinematics + DLS pinv
        J_xy = _first(self.env.skeleton.geometric_jacobian(q))[0:2, :]
        J_pinv, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)

        # dynamics + op-space guard FIRST (teacher uses guarded accel)
        M = _first(self.env.skeleton.mass_matrix(q))
        Minv = torch.linalg.inv(M)
        S = J_xy @ Minv @ J_xy.T
        Lambda, lam_os, eta, eta2, xd_d_s, xdd_d_s, dyn_diag = op_space_guard_and_gate(
            S, xd_d.clone(), xdd_d.clone(), self.dp
        )
        xd_d_s = xd_d_s.reshape(-1)[:2]
        xdd_d_s = xdd_d_s.reshape(-1)[:2]
        eta = torch.as_tensor(eta, device=q.device, dtype=q.dtype).reshape(())
        eta2 = torch.as_tensor(eta2, device=q.device, dtype=q.dtype).reshape(())

        # joint-space proxies
        e_joint = (J_pinv @ e_x).reshape(-1)
        ed_joint = (J_pinv @ ed_x).reshape(-1)
        qdd_d = (J_pinv @ xdd_d_s).reshape(-1)

        # bias terms for feedforward / teacher
        C = _first(self.env.skeleton.coriolis_matrix(q, qd))
        g_any = self.env.skeleton.gravity_vector(q)
        g = (g_any[0] if g_any.dim() >= 3 else g_any).reshape(-1)[:n]
        tau_ff = self._compute_feedforward(q, qd, g, C)
        self._last_tau_ff = tau_ff.clone()

        # ANFIS raw + confidence gating
        tau_anfis_raw = torch.zeros(n, device=q.device, dtype=q.dtype)
        last_phi = [None, None]
        confs = [0.0, 0.0]
        for i in range(n):
            tau_i, phi = self.anfis[i].forward(e_joint[i], ed_joint[i])
            conf = self.anfis[i].get_confidence()
            confs[i] = conf
            limit = self._get_anfis_output_limit(conf)
            tau_anfis_raw[i] = torch.clamp(tau_i, -limit, limit)
            last_phi[i] = phi

        # safety scaling + warmup attenuation (applied signal)
        error_scale = self._get_error_safety_scale(err_norm)
        tau_anfis_applied = tau_anfis_raw * error_scale
        if self.step < int(self.p.warmup_steps):
            warm = float(self.step) / float(max(int(self.p.warmup_steps), 1))
            tau_anfis_applied = tau_anfis_applied * warm

        tau_des = tau_ff + tau_anfis_applied

        # ---------------- muscle allocation (batched helpers) ----------------
        geom = self.env.states["geometry"]
        lenvel = geom[:, :2, :]
        R = geom[0, 2:2 + n, :]
        if self._Fmax_cache is None:
            self._Fmax_cache = get_Fmax_vec(self.env, R.shape[1]).to(device=q.device, dtype=q.dtype)
            names = list(self.env.muscle.state_name)
            self._idx_flpe = names.index("force-length PE") if "force-length PE" in names else -1
        Fmax_vec = self._Fmax_cache
        flpe = self.env.states["muscle"][:, self._idx_flpe, :] if self._idx_flpe >= 0 \
            else torch.zeros((1, R.shape[1]), device=q.device, dtype=q.dtype)

        F_des, mus_diag = solve_muscle_forces(tau_des.unsqueeze(0), R.unsqueeze(0), Fmax_vec, eta.reshape(1), self.mp)
        a = force_to_activation_bisect(F_des, lenvel, self.env.muscle, flpe, Fmax_vec, iters=int(self.p.bisect_iters))
        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = Fmax_vec * (af_now + flpe)
        F_corr = saturation_repair_tau(-R.unsqueeze(0), F_pred, a, self.env.muscle.min_activation, 1.0,
                                       Fmax_vec, tau_des=tau_des.unsqueeze(0))
        if torch.any(torch.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(F_corr, lenvel, self.env.muscle, flpe, Fmax_vec,
                                           iters=max(4, int(self.p.bisect_iters) - 4))

        # actual achieved torque (for teacher anchoring)
        af_final = active_force_from_activation(a, lenvel, self.env.muscle)
        F_actual = (Fmax_vec * (af_final + flpe)).reshape(-1)
        tau_actual = (-R @ F_actual).reshape(-1)[:n]

        # ---------------- learning update ----------------
        self._last_phi = last_phi
        self._last_e_joint = e_joint.clone()
        self._last_ed_joint = ed_joint.clone()
        self._last_qdd_d = qdd_d.clone()
        self._last_tau_anfis_applied = tau_anfis_applied.clone()

        if (self.p.enable_learning and self.step >= int(self.p.warmup_steps)
                and (self.step % max(int(self.p.adapt_every), 1) == 0)):
            tau_teacher = self._compute_teacher(M, self._last_e_joint, self._last_ed_joint,
                                                self._last_qdd_d, self._last_tau_ff, tau_actual)
            tau_teacher = torch.clamp(tau_teacher, -self.p.tau_teacher_clip, self.p.tau_teacher_clip)
            for i in range(n):
                if self._last_phi[i] is None:
                    continue
                self.anfis[i].update_rls(self._last_phi[i], tau_teacher[i], float(self.p.rls_param_clip))

        if self.p.autosave_every and self.p.autosave_every > 0 and (self.step % int(self.p.autosave_every)) == 0:
            try:
                self.save_rules(self.p.rules_path)
            except Exception:
                pass

        self.step += 1

        diag = merge_diag(
            pack_diag(sminJ=sminJ, lamJ=lamJ), dyn_diag, mus_diag,
            pack_diag(lam_os=lam_os, eta=eta, eta2=eta2,
                      confidence_0=float(confs[0]), confidence_1=float(confs[1]),
                      n_updates=int(self.anfis[0].n_updates)),
        )
        return {
            "tau_des": tau_des.unsqueeze(0), "R": R.unsqueeze(0), "Fmax": Fmax_vec,
            "F_des": F_des, "act": a, "q": q.unsqueeze(0), "qd": qd.unsqueeze(0),
            "x": x.unsqueeze(0), "xd": xd.unsqueeze(0),
            "xref_tuple": (x_d, xd_d_s, xdd_d_s), "eta": eta2.reshape(1), "diag": diag,
            "tau_ff": tau_ff, "tau_anfis_raw": tau_anfis_raw, "tau_anfis_applied": tau_anfis_applied,
            "tau_actual": tau_actual,
        }
