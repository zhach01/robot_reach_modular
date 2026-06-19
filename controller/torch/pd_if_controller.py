# controller/torch/pd_if_controller.py
# -*- coding: utf-8 -*-
"""
Pure-Torch OPTIMIZED PD + Internal-Force (PD/IF) controller.

Torch port of the canonical optimized numpy controller
(controller/numpy/pd_if_controller.py). Mirrors its control law:
  - Task-space stiffness Kp_task with inertia-weighted critical damping
    (Kd = 2 ζ Λ^½ (Λ^-½ Kp Λ^-½)^½ Λ^½, scaled by damping_ratio).
  - Full operational-space inverse dynamics (Λ ẍ_cmd + μ) when inertia comp is on.
  - Simplified dynamically-consistent nullspace posture control (Kp_null/Kd_null),
    faded by eta2 near singularities, with a final eta safety gate on tau_des.
  - Same torch HTM dynamics/kinematics + robust muscle solve as the legacy
    torch controller.

Batched: compute() runs on the (B, ...) state the torch simulator provides (B>=1).

Note: the numpy version's optional integral action (anti-windup) is not ported —
it defaults off and neither demo uses it. The original pre-optimization gain API
(Kp_x/Kp_q/...) is no longer used anywhere; it is preserved only under
archive/controller/torch/pd_if_legacy.py for historical reference.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from torch import Tensor

from utils.torch.math_utils import matrix_sqrt_spd, matrix_isqrt_spd
from muscles.torch.muscle_tools import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
)
from utils.torch.kinematics_guard import (
    KinGuardParams,
    adaptive_dls_pinv,
    scale_task_by_J,
)
from utils.torch.dynamics_guard import (
    DynGuardParams,
    op_space_guard_and_gate,
)
from utils.torch.muscle_guard import (
    MuscleGuardParams,
    solve_muscle_forces,
)
from utils.torch.telemetry import pack_diag, merge_diag


def _to_tensor_like(x: Any, like: Tensor) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(x, device=like.device, dtype=like.dtype)


def _eye(n: int, like: Tensor) -> Tensor:
    return torch.eye(n, device=like.device, dtype=like.dtype)


def _zeros(shape, like: Tensor) -> Tensor:
    return torch.zeros(*shape, device=like.device, dtype=like.dtype)


@dataclass
class PDIFParams:
    """Optimized PD+IF parameters (matches the numpy canonical controller)."""

    # task-space stiffness and (fallback) damping
    Kp_task: Any = field(default_factory=lambda: [800.0, 800.0])   # synced to numpy (was 1600 -> 2x stiffness divergence)
    Kd_task: Any = field(default_factory=lambda: [60.0, 60.0])
    Kff: float = 1.0

    # dynamics compensation
    enable_inertia_comp: bool = True
    enable_gravity_comp: bool = True
    enable_coriolis_comp: bool = True
    enable_joint_damping: bool = False

    # critical damping
    use_critical_damping: bool = True
    damping_ratio: float = 1.0

    # nullspace posture
    enable_nullspace: bool = True
    Kp_null: float = 20.0
    Kd_null: float = 5.0

    # guards / numerics
    eps: float = 1e-6
    lam_os_max: float = 1e6
    sigma_thresh: float = 1e-4
    gate_pow: float = 2.0

    # muscle inversion
    bisect_iters: int = 12

    # internal force (co-contraction)
    enable_internal_force: bool = False
    cocon_level: float = 0.03

    # singularity safety (same protection as the sliding-mode controller):
    # cond(J) blend to a joint-space fallback with an elbow floor, plus
    # torque-feasibility scaling vs the muscle limits.
    enable_singularity_blend: bool = True
    cond_low: float = 20.0
    cond_high: float = 80.0
    elbow_min_rad: float = math.radians(2.0)
    Kp_js: Any = field(default_factory=lambda: [12.0, 10.0])
    Kd_js: Any = field(default_factory=lambda: [2.5, 2.0])
    tau_scale_safety: float = 0.95
    tau_scale_apply_thresh: float = 0.7


class PDIFController:
    def __init__(self, env, arm, params: PDIFParams):
        self.env = env
        self.arm = arm
        self.p = params
        self.qref: Tensor | None = None
        self._Fmax_cache: Tensor | None = None
        self._idx_flpe: int | None = None

        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=self.p.eps,
            lam_os_max=self.p.lam_os_max,
            gate_pow=self.p.gate_pow,
            sigma_thresh_S=max(self.p.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()

    def reset(self, q0: Tensor) -> None:
        q0 = q0.to(dtype=torch.get_default_dtype())
        if q0.ndim == 1:
            self.qref = q0.unsqueeze(0).clone()
        elif q0.ndim == 2:
            self.qref = q0.clone()
        else:
            raise ValueError(f"reset: q0 must be (n,) or (B,n), got {tuple(q0.shape)}")

    # ------------------------------------------------------------------ dynamics
    def _compute_dynamics(self, q: Tensor, qd: Tensor):
        """Batched M (B,n,n) and bias h = C q̇ + g + D q̇ (B,n)."""
        p = self.p
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if qd.ndim == 1:
            qd = qd.unsqueeze(0)
        B, n = q.shape
        device, dtype = q.device, q.dtype

        def _batch_mat(x):
            x = x.to(device=device, dtype=dtype)
            if x.ndim == 2:
                return x.unsqueeze(0).expand(B, -1, -1)
            if x.shape[0] == 1 and B > 1:
                return x.expand(B, -1, -1)
            if x.shape[0] == B:
                return x
            return x[0:1].expand(B, -1, -1)

        if p.enable_inertia_comp:
            M = _batch_mat(self.env.skeleton.mass_matrix(q))
        else:
            M = _eye(n, q[0]).unsqueeze(0).expand(B, -1, -1)

        if p.enable_gravity_comp:
            g_any = self.env.skeleton.gravity_vector(q).to(device=device, dtype=dtype)
            if g_any.ndim == 2:
                g = g_any.view(-1).unsqueeze(0).expand(B, -1)
            else:
                gb = g_any if g_any.shape[0] == B else g_any[0:1].expand(B, -1, -1)
                g = gb[..., 0]
        else:
            g = torch.zeros_like(q)

        if p.enable_coriolis_comp:
            C = _batch_mat(self.env.skeleton.coriolis_matrix(q, qd))
        else:
            C = _zeros((B, n, n), q)

        if p.enable_joint_damping:
            D = torch.diag(torch.full((n,), float(self.arm.damping), device=device, dtype=dtype))
            D = D.unsqueeze(0).expand(B, -1, -1)
        else:
            D = _zeros((B, n, n), q)

        h = torch.einsum("bij,bj->bi", C, qd) + g + torch.einsum("bij,bj->bi", D, qd)
        return M, h

    def _critical_damping(self, Kp_mat: Tensor, Lambda: Tensor) -> Tensor:
        """Kd = 2 ζ Λ^½ (Λ^-½ Kp Λ^-½)^½ Λ^½, batched (Lambda: (B,2,2))."""
        Lam_sqrt = matrix_sqrt_spd(Lambda)
        Lam_isqrt = matrix_isqrt_spd(Lambda)
        inner = torch.einsum("bij,jk->bik", Lam_isqrt, Kp_mat)
        inner = torch.einsum("bij,bjk->bik", inner, Lam_isqrt)
        inner_sqrt = matrix_sqrt_spd(inner)
        Kd = torch.einsum("bij,bjk->bik", Lam_sqrt, inner_sqrt)
        Kd = torch.einsum("bij,bjk->bik", Kd, Lam_sqrt)
        return (2.0 * float(self.p.damping_ratio)) * Kd

    # ------------------------------------------------------------------- compute
    def compute(self, x_d: Tensor, xd_d: Tensor, xdd_d: Tensor) -> Dict[str, Any]:
        joint = self.env.states["joint"]       # (B, 2n)
        cart = self.env.states["cartesian"]    # (B, >=4)
        B = joint.shape[0]
        n = int(self.env.skeleton.dof)
        q, qd = joint[:, :n], joint[:, n:]
        x, xd = cart[:, :2], cart[:, 2:4]
        self.env.skeleton._set_state(q, qd)

        def _b2(z):
            z = _to_tensor_like(z, x)
            if z.ndim == 1:
                z = z.unsqueeze(0)
            return z.expand(B, -1) if z.shape[0] == 1 and B > 1 else z

        x_d, xd_d, xdd_d = _b2(x_d), _b2(xd_d), _b2(xdd_d)

        if self.qref is None:
            self.qref = q.clone()
        qref = _to_tensor_like(self.qref, q)
        if qref.ndim == 1:
            qref = qref.unsqueeze(0)
        if qref.shape[0] == 1 and B > 1:
            qref = qref.expand(B, -1)

        # --- Jacobians (B,2,n) ---
        def _b3(a):
            if a.ndim == 2:
                return a.unsqueeze(0).expand(B, -1, -1)
            return a.expand(B, -1, -1) if a.shape[0] == 1 and B > 1 else a
        J_xy = _b3(self.env.skeleton.geometric_jacobian(q))[:, 0:2, :]
        Jdot_xy = _b3(self.env.skeleton.geometric_jacobian_dot(q, qd))[:, 0:2, :]

        # --- kinematic guard ---
        J_pinv_dls, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)
        xd_d_s, xdd_d_s, alpha_J = scale_task_by_J(xd_d, xdd_d, sminJ, self.kp)

        # --- dynamics + op-space inertia ---
        M, h = self._compute_dynamics(q, qd)
        Minv = torch.linalg.inv(M)
        Jt = J_xy.transpose(-1, -2)
        S = torch.einsum("bij,bjk->bik", J_xy, torch.einsum("bij,bjk->bik", Minv, Jt))

        Lambda, lam_os, eta, eta2, xd_d_g, xdd_d_g, dyn_diag = op_space_guard_and_gate(
            S, xd_d_s, xdd_d_s, self.dp
        )
        if Lambda.ndim == 2:
            Lambda = Lambda.unsqueeze(0).expand(B, -1, -1)
        if xd_d_g.ndim == 1:
            xd_d_g = xd_d_g.unsqueeze(0).expand(B, -1)
        if xdd_d_g.ndim == 1:
            xdd_d_g = xdd_d_g.unsqueeze(0).expand(B, -1)

        def _b1(s):
            if isinstance(s, Tensor):
                s = s.to(device=x.device, dtype=x.dtype)
                if s.ndim == 0:
                    s = s.view(1).expand(B)
                elif s.shape[0] == 1 and B > 1:
                    s = s.expand(B)
                return s
            return torch.full((B,), float(s), device=x.device, dtype=x.dtype)
        eta, eta2 = _b1(eta), _b1(eta2)

        # --- task-space errors + gains ---
        e_x = x_d - x
        e_v = xd_d_g - xd
        Kp_mat = torch.diag(_to_tensor_like(self.p.Kp_task, x[0]))   # (2,2)
        if self.p.use_critical_damping and self.p.enable_inertia_comp:
            Kd_mat = self._critical_damping(Kp_mat, Lambda)          # (B,2,2)
            term_d = torch.einsum("bij,bj->bi", Kd_mat, e_v)
        else:
            Kd_mat = torch.diag(_to_tensor_like(self.p.Kd_task, x[0]))
            term_d = torch.einsum("ij,bj->bi", Kd_mat, e_v)

        xdd_cmd = float(self.p.Kff) * xdd_d_g + term_d + torch.einsum("ij,bj->bi", Kp_mat, e_x)

        # --- operational-space control law ---
        if self.p.enable_inertia_comp:
            Minv_h = torch.einsum("bij,bj->bi", Minv, h)
            JMinvh = torch.einsum("bij,bj->bi", J_xy, Minv_h)
            Jdot_qd = torch.einsum("bij,bj->bi", Jdot_xy, qd)
            mu = torch.einsum("bij,bj->bi", Lambda, JMinvh - Jdot_qd)
            F_task = torch.einsum("bij,bj->bi", Lambda, xdd_cmd) + mu
        else:
            F_task = torch.einsum("ij,bj->bi", Kp_mat, e_x) + term_d
        tau_task = torch.einsum("bij,bj->bi", Jt, F_task)            # (B,n)

        # --- nullspace posture (dynamically consistent), faded by eta2 ---
        tau_null = torch.zeros_like(tau_task)
        if self.p.enable_nullspace:
            Jt_L = torch.einsum("bij,bjk->bik", Jt, Lambda)
            J_dyn_pinv = torch.einsum("bij,bjk->bik", Minv, Jt_L)   # (B,n,2)
            N = _eye(n, q[0]).unsqueeze(0) - torch.einsum("bij,bjk->bik", J_dyn_pinv, J_xy)
            qdd_null = float(self.p.Kp_null) * (qref - q) - float(self.p.Kd_null) * qd
            tau_post = torch.einsum("bij,bj->bi", M, qdd_null) + h
            tau_null = eta2.view(B, 1) * torch.einsum("bij,bj->bi", N.transpose(-1, -2), tau_post)

        tau_des = tau_task + tau_null
        # final eta safety gate near singularities (clip to [0.3, 1.0])
        tau_des = tau_des * torch.clamp(eta, 0.3, 1.0).view(B, 1)

        # --- update joint reference ---
        qd_des = torch.einsum("bij,bj->bi", J_pinv_dls, xd_d_g)
        self.qref = qref + qd_des * float(self.arm.dt)

        # --- muscle force allocation ---
        geom = self.env.states["geometry"]          # (B, 2+n, M)
        lenvel = geom[:, :2, :]
        R = geom[:, 2:2 + n, :]                      # (B,n,M)
        Mm = int(R.shape[-1])
        if self._Fmax_cache is None:
            self._Fmax_cache = get_Fmax_vec(self.env, Mm).to(device=q.device, dtype=q.dtype)
            names: List[str] = list(getattr(self.env.muscle, "state_name", []))
            self._idx_flpe = names.index("force-length PE") if "force-length PE" in names else -1
        Fmax_vec = self._Fmax_cache
        if self._idx_flpe >= 0:
            flpe = self.env.states["muscle"][:, self._idx_flpe, :]
        else:
            flpe = torch.zeros((B, Mm), device=q.device, dtype=q.dtype)

        # === Singularity safety (mirrors the sliding-mode / numpy controller) ===
        if self.p.enable_singularity_blend:
            condJ = torch.linalg.cond(J_xy)                              # (B,)
            condJ = torch.where(torch.isfinite(condJ), condJ, torch.full_like(condJ, 1e9))
            w_cr = torch.clamp((condJ - self.p.cond_low) / (self.p.cond_high - self.p.cond_low), 0.0, 1.0)
            w_eta = torch.clamp(1.0 - eta, 0.0, 1.0)
            w = torch.clamp(torch.maximum(w_cr, 0.5 * w_eta), 0.0, 1.0)  # (B,)
            q_star = qref.clone()
            q_star[:, 1] = torch.clamp(q_star[:, 1], min=float(self.p.elbow_min_rad))
            Kp_js = _to_tensor_like(self.p.Kp_js, q[0])
            Kd_js = _to_tensor_like(self.p.Kd_js, q[0])
            tau_js = Kp_js * (q_star - q) - Kd_js * qd                   # (B,n)
            tau_des = (1.0 - w).view(B, 1) * tau_des + w.view(B, 1) * tau_js

        # Torque-feasibility scaling vs muscle limits.
        tau_feas = torch.sum(torch.abs(R) * Fmax_vec.view(1, 1, -1), dim=2)  # (B,n)
        tau_clip = self.p.tau_scale_safety * tau_feas
        raw_scale = torch.clamp(torch.min(tau_clip / (torch.abs(tau_des) + 1e-12), dim=1)[0], 0.0, 1.0)  # (B,)
        apply = raw_scale < self.p.tau_scale_apply_thresh
        tau_des = torch.where(apply.view(B, 1), tau_des * raw_scale.view(B, 1), tau_des)

        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta, self.mp)

        if self.p.enable_internal_force:
            cocon = torch.full((B, Mm), float(self.p.cocon_level), device=q.device, dtype=q.dtype)
            af_cocon = active_force_from_activation(cocon, lenvel, self.env.muscle)
            F_des = F_des + eta2.view(B, 1) * (Fmax_vec * af_cocon)

        a = force_to_activation_bisect(
            F_des, lenvel, self.env.muscle, flpe, Fmax_vec, iters=int(self.p.bisect_iters)
        )
        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = torch.clamp(Fmax_vec * (af_now + flpe), min=0.0)

        F_corr = saturation_repair_tau(
            -R, F_pred, a, self.env.muscle.min_activation, 1.0, Fmax_vec, tau_des=tau_des
        )
        if torch.any(torch.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(
                F_corr, lenvel, self.env.muscle, flpe, Fmax_vec,
                iters=max(4, int(self.p.bisect_iters) - 4),
            )

        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ, alpha_J=alpha_J)
        diag = merge_diag(
            kin_diag, dyn_diag, mus_diag, pack_diag(lam_os=lam_os, eta=eta, eta2=eta2)
        )
        return {
            "tau_des": tau_des, "R": R, "Fmax": Fmax_vec, "F_des": F_des,
            "act": a, "q": q, "qd": qd, "x": x, "xd": xd,
            "xref_tuple": (x_d, xd_d_g, xdd_d_g), "eta": eta2, "diag": diag,
        }
