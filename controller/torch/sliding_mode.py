# controller/torch/sliding_mode.py
# -*- coding: utf-8 -*-
"""
Pure-Torch sliding-mode PD/IF controller — a faithful mirror of the numpy
controller (controller/numpy/sliding_mode.py).

This is a line-by-line port of the proven numpy control law, so it inherits the
numpy controller's robustness near the full-extension Jacobian singularity:
  - acceleration-level SMC with sigma_min(J) eta-gating (switch_in_accel),
  - blend to a joint-space fallback on cond(J) (NOT cond(R) — cond(R) is blind to
    the kinematic singularity),
  - elbow-floor on the fallback target (keeps the elbow out of full extension),
  - torque-feasibility scaling against the muscle limits.

Operates on B=1 state from the torch simulator (the reaching demos are single-arm).
Uses the torch HTM dynamics (env.skeleton.*) and the torch muscle/guard helpers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import math
import torch
from torch import Tensor

from utils.torch.kinematics_guard import KinGuardParams, adaptive_dls_pinv, scale_task_by_J
from utils.torch.dynamics_guard import DynGuardParams, op_space_guard_and_gate
from utils.torch.muscle_guard import MuscleGuardParams, solve_muscle_forces
from muscles.torch.muscle_tools import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
)
from utils.torch.telemetry import pack_diag, merge_diag


def _t(x, like: Tensor) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(x, device=like.device, dtype=like.dtype)


def _first(x: Tensor) -> Tensor:
    """Take the first batch slice if x is batched (dim 3 -> 2, etc.)."""
    return x[0] if x.dim() >= 3 else x


def _sat_vec(s: Tensor, phi: Tensor) -> Tensor:
    return torch.clamp(s / phi, -1.0, 1.0)


def _blend_weight_from_cond(cond_val: Tensor, lo: float, hi: float) -> Tensor:
    return torch.clamp((cond_val - lo) / (hi - lo), 0.0, 1.0)


def _safe_cond(A: Tensor, fallback: float = 1e9) -> Tensor:
    try:
        c = torch.linalg.cond(A)
        if not torch.isfinite(c):
            return torch.tensor(fallback, device=A.device, dtype=A.dtype)
        return c
    except Exception:
        return torch.tensor(fallback, device=A.device, dtype=A.dtype)


@dataclass
class SlidingModeParams:
    """Mirrors the numpy SlidingModeParams (tensor fields accept list/tensor)."""
    lambda_surf: Any = field(default_factory=lambda: [26.0, 26.0])
    K_switch: Any = field(default_factory=lambda: [4.0, 4.0])
    phi: Any = field(default_factory=lambda: [0.004, 0.004])
    Kff_x: Any = field(default_factory=lambda: [1.0, 1.0])

    enable_gravity_comp: bool = True
    enable_velocity_comp: bool = True
    enable_inertia_comp: bool = True

    lambda_ns: float = 0.0
    k_manip: float = 0.0

    eps: float = 1e-6
    lam_os_max: float = 1e6
    sigma_thresh: float = 1e-4
    gate_pow: float = 2.0
    sminJ_thresh: float = 0.02

    bisect_iters: int = 16
    switch_in_accel: bool = True
    use_condJ_for_blend: bool = True
    cond_low: float = 20.0
    cond_high: float = 80.0
    elbow_min_rad: float = math.radians(2.0)

    tau_scale_apply_thresh: float = 0.7
    tau_scale_safety: float = 0.95


class SlidingModeController:
    def __init__(self, env, arm, params: SlidingModeParams):
        self.env = env
        self.arm = arm
        self.p = params
        self.qref: Tensor | None = None
        self._Fmax_cache: Tensor | None = None
        self._idx_flpe: int | None = None

        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=params.eps, lam_os_max=params.lam_os_max,
            gate_pow=params.gate_pow, sigma_thresh_S=max(params.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()

        self._eta_eq_floor = 0.0
        self._eta_sw_floor = 0.0
        self._eta_sw_pow = 0.75
        self._cond_low = float(params.cond_low)
        self._cond_high = float(params.cond_high)
        self._elbow_min = float(params.elbow_min_rad)
        self._tau_scale_apply_thresh = float(params.tau_scale_apply_thresh)
        self._tau_scale_safety = float(params.tau_scale_safety)

    def reset(self, q0: Tensor) -> None:
        q0 = q0.to(dtype=torch.get_default_dtype())
        self.qref = (q0[0] if q0.ndim == 2 else q0).clone()

    def compute(self, x_d: Tensor, xd_d: Tensor, xdd_d: Tensor) -> Dict[str, Any]:
        # ---- state (B=1 -> unbatched control law) ----
        joint = self.env.states["joint"]               # (1, 2n)
        n = int(self.env.skeleton.dof)
        q, qd = joint[0, :n], joint[0, n:]             # (n,)
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:4]
        self.env.skeleton._set_state(q, qd)

        x_d = _t(x_d, x).reshape(-1)[:2]
        xd_d = _t(xd_d, x).reshape(-1)[:2]
        xdd_d = _t(xdd_d, x).reshape(-1)[:2]

        # ---- kinematics ----
        J_xy = _first(self.env.skeleton.geometric_jacobian(q))[0:2, :]        # (2,n)
        Jdot_xy = _first(self.env.skeleton.geometric_jacobian_dot(q, qd))[0:2, :]
        J_pinv_dls, sminJ, lamJ = adaptive_dls_pinv(J_xy, 2, self.kp)
        sminJ = torch.as_tensor(sminJ, device=q.device, dtype=q.dtype).reshape(())

        xd_d_s, xdd_d_s, alpha_J = scale_task_by_J(xd_d.clone(), xdd_d.clone(), sminJ=sminJ, P=self.kp)

        # ---- dynamics + op-space guard ----
        M = _first(self.env.skeleton.mass_matrix(q))                          # (n,n)
        Minv = torch.linalg.inv(M)
        S = J_xy @ Minv @ J_xy.T
        Lambda_reg, lam_os, eta, eta2, xd_d_g, xdd_d_g, dyn_diag = op_space_guard_and_gate(
            S, xd_d_s.clone(), xdd_d_s.clone(), self.dp
        )
        xd_d_g = xd_d_g.reshape(-1)[:2]
        xdd_d_g = xdd_d_g.reshape(-1)[:2]
        eta = torch.as_tensor(eta, device=q.device, dtype=q.dtype).reshape(())
        eta2 = torch.as_tensor(eta2, device=q.device, dtype=q.dtype).reshape(())

        if self.qref is None:
            self.qref = q.clone()
        self.qref = self.qref + float(self.arm.dt) * (J_pinv_dls @ xd_d_g)

        Lambda_used = Lambda_reg if self.p.enable_inertia_comp else torch.eye(2, device=q.device, dtype=q.dtype)

        # ---- bias term h = C qd + g ----
        h = torch.zeros_like(q)
        if self.p.enable_velocity_comp:
            C = _first(self.env.skeleton.coriolis_matrix(q, qd))
            h = h + C @ qd
        if self.p.enable_gravity_comp:
            g_any = self.env.skeleton.gravity_vector(q)
            g = (g_any[0] if g_any.dim() >= 3 else g_any).reshape(-1)[:n]
            h = h + g
        if self.p.enable_inertia_comp:
            mu = Lambda_used @ (J_xy @ (Minv @ h)) - Lambda_used @ (Jdot_xy @ qd)
        else:
            mu = torch.zeros(2, device=q.device, dtype=q.dtype)

        # ---- sliding surface ----
        lam = _t(self.p.lambda_surf, q)
        phi = _t(self.p.phi, q)
        Ksw = _t(self.p.K_switch, q)
        Kff = _t(self.p.Kff_x, q)
        e_x = x_d - x
        e_v = xd_d_g - xd
        s = e_v + lam * e_x
        sw = _sat_vec(s, phi)

        # ---- sminJ eta-gating ----
        ratio = torch.clamp(sminJ / (self.p.sminJ_thresh + 1e-12), 0.0, 1.0)
        eta_eq = torch.clamp(torch.clamp(ratio, min=self._eta_eq_floor), 0.0, 1.0)
        eta_sw = torch.clamp(torch.clamp(ratio ** self._eta_sw_pow, min=self._eta_sw_floor), 0.0, 1.0)

        # ---- operational-space control law ----
        if self.p.switch_in_accel:
            eq_part = (Kff * xdd_d_g) + (lam * e_v)
            sw_part = (Ksw * sw)
            xdd_cmd = (eta_eq * eq_part) + (eta_sw * sw_part)
            F_task = (Lambda_used @ xdd_cmd) + mu
            tau_os = J_xy.T @ F_task
        else:
            F_eq = (Lambda_used @ (Kff * xdd_d_g)) + mu
            F_sw = -(Ksw * sw) * eta_sw
            F_task = (eta_eq * F_eq) + F_sw
            tau_os = J_xy.T @ F_task

        # ---- posture bias ----
        q_center = torch.tensor([math.pi / 3, math.pi / 2], device=q.device, dtype=q.dtype)
        tau_bias = -0.15 * (q - q_center) * (1.0 - torch.clamp(eta, 0.0, 1.0))

        # ---- blend to joint fallback near singularity (cond(J)) ----
        geom = self.env.states["geometry"]
        R = geom[0, 2:2 + n, :]                              # (n,M)
        condR = _safe_cond(R)
        condJ = _safe_cond(J_xy) if self.p.use_condJ_for_blend else torch.zeros((), device=q.device, dtype=q.dtype)
        cond_metric = condJ if self.p.use_condJ_for_blend else condR
        w_cr = _blend_weight_from_cond(cond_metric, self._cond_low, self._cond_high)
        w_eta = torch.clamp(1.0 - eta, 0.0, 1.0)
        w = torch.clamp(torch.maximum(w_cr, 0.5 * w_eta), 0.0, 1.0)

        q_star = self.qref.clone()
        q_star[1] = torch.clamp(q_star[1], min=self._elbow_min)
        Kp_js = torch.tensor([12.0, 10.0], device=q.device, dtype=q.dtype)
        Kd_js = torch.tensor([2.5, 2.0], device=q.device, dtype=q.dtype)
        tau_js = Kp_js * (q_star - q) + Kd_js * (-qd)

        tau_des = (1.0 - w) * tau_os + w * tau_js + tau_bias - 0.06 * qd

        # ---- torque-feasibility scaling ----
        if self._Fmax_cache is None:
            self._Fmax_cache = get_Fmax_vec(self.env, R.shape[1]).to(device=q.device, dtype=q.dtype)
        Fmax_vec = self._Fmax_cache
        tau_feas = torch.sum(torch.abs(R) * Fmax_vec.unsqueeze(0), dim=1)
        tau_clip = self._tau_scale_safety * tau_feas
        raw_scale = torch.clamp(torch.min(tau_clip / (torch.abs(tau_des) + 1e-12)), 0.0, 1.0)
        if float(raw_scale) < self._tau_scale_apply_thresh:
            tau_des = tau_des * raw_scale
            tau_scale_applied = raw_scale
        else:
            tau_scale_applied = torch.ones((), device=q.device, dtype=q.dtype)

        # ---- muscle allocation (batched helpers: add leading dim) ----
        if self._idx_flpe is None:
            names = list(self.env.muscle.state_name)
            self._idx_flpe = names.index("force-length PE") if "force-length PE" in names else -1
        flpe = self.env.states["muscle"][:, self._idx_flpe, :] if self._idx_flpe >= 0 \
            else torch.zeros((1, R.shape[1]), device=q.device, dtype=q.dtype)

        eta_b = eta.reshape(1)
        F_des, mus_diag = solve_muscle_forces(tau_des.unsqueeze(0), R.unsqueeze(0), Fmax_vec, eta_b, self.mp)
        lenvel = geom[:, :2, :]                              # (1,2,M)
        a = force_to_activation_bisect(F_des, lenvel, self.env.muscle, flpe, Fmax_vec, iters=int(self.p.bisect_iters))
        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = Fmax_vec * (af_now + flpe)
        F_corr = saturation_repair_tau(-R.unsqueeze(0), F_pred, a, self.env.muscle.min_activation, 1.0,
                                       Fmax_vec, tau_des=tau_des.unsqueeze(0))
        if torch.any(torch.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(F_corr, lenvel, self.env.muscle, flpe, Fmax_vec,
                                           iters=max(4, int(self.p.bisect_iters) - 4))

        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ, alpha_J=alpha_J, k_manip=0.0)
        diag = merge_diag(
            kin_diag, dyn_diag, mus_diag,
            pack_diag(lam_os=lam_os, eta=eta, eta2=eta2, condR=condR, condJ=condJ,
                      w=w, tau_scale=tau_scale_applied, tau_scale_raw=raw_scale,
                      eta_eq=eta_eq, eta_sw=eta_sw),
        )
        return {
            "tau_des": tau_des.unsqueeze(0), "R": R.unsqueeze(0), "Fmax": Fmax_vec,
            "F_des": F_des, "act": a, "q": q.unsqueeze(0), "qd": qd.unsqueeze(0),
            "x": x.unsqueeze(0), "xd": xd.unsqueeze(0),
            "xref_tuple": (x_d, xd_d_g, xdd_d_g), "eta": eta2.reshape(1), "diag": diag,
        }
