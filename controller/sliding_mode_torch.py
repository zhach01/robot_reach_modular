# control/sliding_mode_torch.py
# -*- coding: utf-8 -*-
"""
Sliding-Mode controller (Torch version, batched).

- Pure Torch, no NumPy.
- Fully batched: all core tensors are (B, ·).
- Same telemetry / muscle allocation stack as pd_if_controller_torch.
- Task-space error convention:
    e_x = x_d - x
    e_v = xd_d - xd
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict

from muscles.muscle_tools_torch import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
    apply_internal_force_regulation,
)

import lib.kinematics.HTM_kinematics_torch as _kin
import lib.dynamics.DynamicsHTM_torch as _dyn

from utils.kinematics_guard_torch import (
    KinGuardParams,
    adaptive_dls_pinv,
    scale_task_by_J,
    add_nullspace_manip,
)
from utils.dynamics_guard_torch import DynGuardParams, op_space_guard_and_gate
from utils.muscle_guard_torch import MuscleGuardParams, solve_muscle_forces
from utils.telemetry_torch import pack_diag, merge_diag


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class SlidingModeParams:
    # Task-space SMC (2D)
    lambda_surf: torch.Tensor   # (2,)
    K_switch:    torch.Tensor   # (2,)
    phi:         torch.Tensor   # (2,)
    Kff_x:       torch.Tensor   # (2,)

    # Kept for API parity (not used directly for fallback)
    Kp_q: torch.Tensor
    Kd_q: torch.Tensor

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


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

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

        # Guards
        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=self.p.eps,
            lam_os_max=self.p.lam_os_max,
            gate_pow=self.p.gate_pow,
            sigma_thresh_S=max(self.p.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()

        dof = getattr(self.env.skeleton, "dof", 2)
        self._tau_clip  = 300.0
        self._tau_alpha = 1.0
        self._tau_filt  = torch.zeros(dof)

        # ----- blending / gates -----
        self._eta_eq_floor = 0.50
        self._eta_sw_floor = 0.35
        self._cond_low     =  400.0
        self._cond_high    = 2500.0
        self._lam_os_floor = 1e-3

        # ----- joint-space fallback (simple PD on q* from DLS) -----
        self._Kp_js = torch.tensor([30.0, 30.0], dtype=torch.get_default_dtype())
        self._Kd_js = torch.tensor([ 3.5,  3.5], dtype=torch.get_default_dtype())
        self._k_pos_map = 1.4

        # ----- nullspace / joint-limit helpers -----
        self._manip_gain      = 1.0
        self._lim_margin_frac = 0.25
        self._lim_eps         = 1e-4
        self._lim_power       = 2.0
        self._k_lim           = 14.0
        self._kd_lim          = 0.6
        self._k_sing_Kp       = 16.0
        self._k_sing_Kd       = 0.9
        self._elbow_bias_rad  = 0.20
        self._sing_gate_eta   = 0.9

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    @staticmethod
    def _sat(z: torch.Tensor) -> torch.Tensor:
        """Elementwise saturation in [-1, 1]."""
        return torch.clamp(z, -1.0, 1.0)

    def _to_tensor_like(self, x: Any, like: torch.Tensor) -> torch.Tensor:
        """Convert x to a tensor on like.device/like.dtype."""
        if isinstance(x, torch.Tensor):
            return x.to(device=like.device, dtype=like.dtype)
        return torch.as_tensor(x, device=like.device, dtype=like.dtype)

    def _eye(self, n: int, like: torch.Tensor) -> torch.Tensor:
        """Identity matrix with same device/dtype as `like`."""
        return torch.eye(n, device=like.device, dtype=like.dtype)

    def _zeros(self, shape, like: torch.Tensor) -> torch.Tensor:
        """Zeros with same device/dtype as `like`."""
        return torch.zeros(*shape, device=like.device, dtype=like.dtype)

    def _zeros_like(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def _compute_dynamics(self, q: torch.Tensor, qd: torch.Tensor):
        """
        Batched dynamics: q, qd can be (n,) or (B,n).

        Returns
        -------
        M : (B,n,n)  inertia matrix
        h : (B,n)    bias term = C(q,q̇) q̇ + g(q) + D q̇
        """
        p = self.p

        # ---- normalize shapes: (B,n) ----
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if qd.ndim == 1:
            qd = qd.unsqueeze(0)
        if q.shape != qd.shape:
            raise ValueError(
                f"_compute_dynamics: q and qd must have same shape, got {q.shape} vs {qd.shape}"
            )

        B, n = q.shape
        device = q.device
        dtype = q.dtype

        # ------------------------------------------------------------------
        # Inertia M(q)
        # ------------------------------------------------------------------
        if p.enable_inertia_comp:
            M_any = _dyn.inertiaMatrixCOM(self.env.skeleton._robot)
            if M_any.ndim == 2:
                M = M_any.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            elif M_any.ndim == 3:
                M_any = M_any.to(device=device, dtype=dtype)
                B0 = M_any.shape[0]
                if B0 == 1 and B > 1:
                    M = M_any.expand(B, -1, -1)
                elif B0 == B:
                    M = M_any
                else:
                    M = M_any[0:1, :, :].expand(B, -1, -1)
            else:
                raise ValueError(
                    f"inertiaMatrixCOM must be (n,n) or (B,n,n), got {tuple(M_any.shape)}"
                )
        else:
            M_single = self._eye(n, q[0])
            M = M_single.unsqueeze(0).expand(B, -1, -1)

        # ------------------------------------------------------------------
        # Gravity g(q)
        # ------------------------------------------------------------------
        if p.enable_gravity_comp:
            g_any = _dyn.gravitationalCOM(
                self.env.skeleton._robot,
                g=self.env.skeleton._gravity_vec,
            )
            if g_any.ndim == 2:
                g_single = g_any.to(device=device, dtype=dtype).view(-1)
                g = g_single.unsqueeze(0).expand(B, -1)
            elif g_any.ndim == 3:
                g_any = g_any.to(device=device, dtype=dtype)
                B0 = g_any.shape[0]
                if B0 == 1 and B > 1:
                    g_b = g_any.expand(B, -1, -1)
                elif B0 == B:
                    g_b = g_any
                else:
                    g_b = g_any[0:1, :, :].expand(B, -1, -1)
                g = g_b[..., 0]
            else:
                raise ValueError(
                    f"gravitationalCOM must be (n,1) or (B,n,1), got {tuple(g_any.shape)}"
                )
        else:
            g = self._zeros_like(q)

        # ------------------------------------------------------------------
        # Centrifugal / Coriolis C(q,q̇)
        # ------------------------------------------------------------------
        if p.enable_velocity_comp:
            C_any = _dyn.centrifugalCoriolisCOM(self.env.skeleton._robot)
            if C_any.ndim == 2:
                C = C_any.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            elif C_any.ndim == 3:
                C_any = C_any.to(device=device, dtype=dtype)
                B0 = C_any.shape[0]
                if B0 == 1 and B > 1:
                    C = C_any.expand(B, -1, -1)
                elif B0 == B:
                    C = C_any
                else:
                    C = C_any[0:1, :, :].expand(B, -1, -1)
            else:
                raise ValueError(
                    f"centrifugalCoriolisCOM must be (n,n) or (B,n,n), got {tuple(C_any.shape)}"
                )
        else:
            C = self._zeros((B, n, n), q)

        # ------------------------------------------------------------------
        # Joint viscous damping D
        # ------------------------------------------------------------------
        if p.enable_joint_damping:
            damping_vec = torch.full(
                (n,),
                float(self.arm.damping),
                device=device,
                dtype=dtype,
            )
            D_single = torch.diag(damping_vec)
            D = D_single.unsqueeze(0).expand(B, -1, -1)
        else:
            D = self._zeros((B, n, n), q)

        # ------------------------------------------------------------------
        # Bias term: h(q,q̇) = C q̇ + g + D q̇   (B,n)
        # ------------------------------------------------------------------
        h_C = torch.einsum("bij,bj->bi", C, qd)
        h_D = torch.einsum("bij,bj->bi", D, qd)
        h = h_C + g + h_D

        return M, h

    # ------------------------------------------------------------------
    # Joint-limit & singularity helpers (batched)
    # ------------------------------------------------------------------

    def _joint_limit_tau(self, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        """Batched joint-limit avoidance torque τ_lim."""
        q_min = getattr(self.env.skeleton, "q_min", None)
        q_max = getattr(self.env.skeleton, "q_max", None)
        if q_min is None or q_max is None:
            return self._zeros_like(q)

        q_min = self._to_tensor_like(q_min, q).reshape(-1)
        q_max = self._to_tensor_like(q_max, q).reshape(-1)

        rng = q_max - q_min
        margin = self._lim_margin_frac * rng
        inner_min, inner_max = q_min + margin, q_max - margin

        # For batched: (B,n)
        zero = torch.tensor(0.0, device=q.device, dtype=q.dtype)
        dmin = torch.maximum(zero, inner_min - q)
        dmax = torch.maximum(zero, q - inner_max)

        # If no active limits, early out
        if not (torch.any(dmin > 0) or torch.any(dmax > 0)):
            return self._zeros_like(q)

        dmin_c = dmin + self._lim_eps
        dmax_c = dmax + self._lim_eps

        grad = (1.0 / (dmin_c**self._lim_power)) - (1.0 / (dmax_c**self._lim_power))
        activity = (1.0 / (dmin_c**self._lim_power)) + (1.0 / (dmax_c**self._lim_power))

        return self._k_lim * grad - self._kd_lim * qd * torch.clamp(activity, 0.0, 1e6)

    def _singularity_bias_tau(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        eta_val: torch.Tensor,
    ) -> torch.Tensor:
        """Batched singularity bias torque τ_bias."""
        B = q.shape[0]

        # Handle batched eta_val
        if eta_val.ndim == 0:
            eta_val = eta_val.unsqueeze(0).expand(B)

        # Mask for active bias
        active_mask = eta_val < self._sing_gate_eta
        if not torch.any(active_mask):
            return self._zeros_like(q)

        q_min = getattr(self.env.skeleton, "q_min", None)
        q_max = getattr(self.env.skeleton, "q_max", None)

        if q_min is not None and q_max is not None:
            q_center = 0.5 * (
                self._to_tensor_like(q_min, q).reshape(-1)
                + self._to_tensor_like(q_max, q).reshape(-1)
            )
            q_center = q_center.unsqueeze(0).expand_as(q)
        else:
            # Use elbow bias to avoid full extension
            q_center = q.clone()
            elbow_bias = torch.where(
                q[:, 1] >= 0.0,
                torch.tensor(self._elbow_bias_rad, device=q.device, dtype=q.dtype),
                torch.tensor(-self._elbow_bias_rad, device=q.device, dtype=q.dtype),
            )
            q_center[:, 1] = elbow_bias

        e = q_center - q
        bias_factor = (1.0 - eta_val.unsqueeze(-1)) ** 2
        return bias_factor * (self._k_sing_Kp * e - self._k_sing_Kd * qd)

    def _blend_weight(self, eta_val: torch.Tensor, condR: torch.Tensor) -> torch.Tensor:
        """
        Batched blending weight:

        w = 0 → pure OS-SMC
        w = 1 → pure joint-space fallback
        """
        if eta_val.ndim == 0:
            eta_val = eta_val.view(1)
        if condR.ndim == 0:
            condR = condR.view(1)

        # η-based weight
        w_eta = torch.clamp(1.0 - eta_val, 0.0, 1.0) ** 2

        # cond(R)-based weight (0 at cond_low → 1 at cond_high)
        w_cr = torch.clamp(
            (condR - self._cond_low) / (self._cond_high - self._cond_low),
            0.0,
            1.0,
        )

        return torch.maximum(w_eta, w_cr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, q0: torch.Tensor) -> None:
        """
        Set initial joint reference qref (batched).

        q0:
            - (n,)   for a single sample
            - (B,n)  for batched joints
        """
        q0 = q0.to(dtype=torch.get_default_dtype())
        if q0.ndim == 1:
            self.qref = q0.unsqueeze(0).clone()
        elif q0.ndim == 2:
            self.qref = q0.clone()
        else:
            raise ValueError(
                f"SlidingModeController.reset: q0 must be (n,) or (B,n), got {tuple(q0.shape)}"
            )
        self._tau_filt = self._zeros_like(self.qref)

    def compute(
        self,
        x_d: torch.Tensor,
        xd_d: torch.Tensor,
        xdd_d: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Batched compute for sliding mode controller.

        Inputs (batched):
          x_d, xd_d, xdd_d : (B,2) or (2,) tensors

        Returns dict containing (batched):
          tau_des : (B,n)
          act     : (B,M)
          etc.
        """
        # ------------------------------------------------------------------
        # Current state (BATCHED)
        # ------------------------------------------------------------------
        joint = self.env.states["joint"]       # (B, 2*dof)
        cart = self.env.states["cartesian"]    # (B, >=4)

        if joint.ndim != 2:
            raise ValueError(
                f"env.states['joint'] must be (B, 2*dof), got {tuple(joint.shape)}"
            )
        if cart.ndim != 2 or cart.shape[1] < 4:
            raise ValueError(
                f"env.states['cartesian'] must be (B, >=4), got {tuple(cart.shape)}"
            )

        B = joint.shape[0]
        n = int(self.env.skeleton.dof)

        q = joint[:, :n]       # (B,n)
        qd = joint[:, n:]      # (B,n)

        x = cart[:, :2]        # (B,2)
        xd = cart[:, 2:4]      # (B,2)

        # Make sure robot is updated with batched joint state
        self.env.skeleton._set_state(q, qd)

        # ------------------------------------------------------------------
        # Ensure batched references: (B,2)
        # ------------------------------------------------------------------
        def _ensure_batch2(z: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
            z = z.to(device=like.device, dtype=like.dtype)
            if z.ndim == 1:
                z = z.unsqueeze(0)
            if z.shape[0] == 1 and B > 1:
                z = z.expand(B, -1)
            elif z.shape[0] != B:
                raise ValueError(f"Ref tensor batch {z.shape[0]} != state batch {B}")
            return z

        x_d   = _ensure_batch2(x_d, x)
        xd_d  = _ensure_batch2(xd_d, x)
        xdd_d = _ensure_batch2(xdd_d, x)

        # ------------------------------------------------------------------
        # Batched qref
        # ------------------------------------------------------------------
        if self.qref is None:
            qref = q.clone()
        else:
            qref = self._to_tensor_like(self.qref, q)
            if qref.ndim == 1:
                qref = qref.unsqueeze(0)
            if qref.shape[0] == 1 and B > 1:
                qref = qref.expand(B, -1)
            elif qref.shape[0] != B:
                qref = qref[0:1, :].expand(B, -1)
        self.qref = qref

        # ------------------------------------------------------------------
        # Jacobians (broadcast to (B,6,n) if needed)
        # ------------------------------------------------------------------
        J_any = _kin.geometricJacobian(self.env.skeleton._robot)
        if J_any.ndim == 2:
            J = J_any.unsqueeze(0).expand(B, -1, -1)
        elif J_any.ndim == 3:
            if J_any.shape[0] == 1 and B > 1:
                J = J_any.expand(B, -1, -1)
            elif J_any.shape[0] == B:
                J = J_any
            else:
                J = J_any[0:1, :, :].expand(B, -1, -1)
        else:
            raise ValueError(
                f"geometricJacobian must be (6,n) or (B,6,n), got {tuple(J_any.shape)}"
            )
        J_xy = J[:, 0:2, :]  # (B,2,n)

        Jdot_any = _kin.geometricJacobianDerivative(self.env.skeleton._robot)
        if Jdot_any.ndim == 2:
            Jdot = Jdot_any.unsqueeze(0).expand(B, -1, -1)
        elif Jdot_any.ndim == 3:
            if Jdot_any.shape[0] == 1 and B > 1:
                Jdot = Jdot_any.expand(B, -1, -1)
            elif Jdot_any.shape[0] == B:
                Jdot = Jdot_any
            else:
                Jdot = Jdot_any[0:1, :, :].expand(B, -1, -1)
        else:
            raise ValueError(
                f"geometricJacobianDerivative must be (6,n) or (B,6,n), got {tuple(Jdot_any.shape)}"
            )
        Jdot_xy = Jdot[:, 0:2, :]  # (B,2,n)

        # ------------------------------------------------------------------
        # [1] kinematic guard + scaling (BATCHED)
        # ------------------------------------------------------------------
        J_pinv_dls, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)
        xd_d, xdd_d, alpha_J = scale_task_by_J(xd_d, xdd_d, sminJ, self.kp)

        # gentle qref update (like PD/IF)
        xd_d_col = xd_d.unsqueeze(-1)              # (B,2,1)
        qd_des = (J_pinv_dls @ xd_d_col).squeeze(-1)  # (B,n)
        self.qref = self.qref + qd_des * float(self.arm.dt)

        # ------------------------------------------------------------------
        # Dynamics
        # ------------------------------------------------------------------
        M, h = self._compute_dynamics(q, qd)
        Minv = torch.linalg.inv(M)

        # [2] op-space metrics + gating
        S = torch.einsum(
            "bij,bjk->bik",
            J_xy,
            torch.einsum("bij,bjk->bik", Minv, J_xy.transpose(-1, -2)),
        )
        Lambda, lam_os, eta_raw, eta2, xd_d, xdd_d, dyn_diag = op_space_guard_and_gate(
            S, xd_d, xdd_d, self.dp
        )

        # Normalize shapes
        def _ensure_batch1(s_any):
            if isinstance(s_any, torch.Tensor):
                s_any = s_any.to(device=x.device, dtype=x.dtype)
                if s_any.ndim == 0:
                    s_any = s_any.view(1).expand(B)
                elif s_any.shape[0] == 1 and B > 1:
                    s_any = s_any.expand(B)
            else:
                s_any = torch.full((B,), float(s_any), device=x.device, dtype=x.dtype)
            return s_any

        eta_val     = _ensure_batch1(eta_raw)
        eta2_val    = _ensure_batch1(eta2)
        lam_os_val  = _ensure_batch1(lam_os) + self._lam_os_floor

        # Ensure Lambda is batched
        if Lambda.ndim == 2:
            Lambda = Lambda.unsqueeze(0).expand(B, -1, -1)

        # regularized Λ
        eye2 = self._eye(2, x[0]).unsqueeze(0).expand(B, -1, -1)
        S_reg = S + lam_os_val.view(B, 1, 1) * eye2
        Lambda_reg = torch.linalg.inv(S_reg)

        # ----- OS-SMC torque -----
        Minv_h = torch.einsum("bij,bj->bi", Minv, h)
        JMinvh = torch.einsum("bij,bj->bi", J_xy, Minv_h)
        Jdot_qd = torch.einsum("bij,bj->bi", Jdot_xy, qd)

        mu = torch.einsum("bij,bj->bi", Lambda_reg, JMinvh - Jdot_qd)

        # Gains to proper tensors
        Kff_x       = self._to_tensor_like(self.p.Kff_x,       x[0])
        lambda_surf = self._to_tensor_like(self.p.lambda_surf, x[0])
        K_switch    = self._to_tensor_like(self.p.K_switch,    x[0])
        phi         = self._to_tensor_like(self.p.phi,         x[0])

        F_eq = torch.einsum("bij,bj->bi", Lambda_reg, Kff_x * xdd_d) + mu

        # Sliding surface errors (textbook tracking convention)
        e_x = x_d - x      # (B,2)
        e_v = xd_d - xd    # (B,2)
        s   = e_v + lambda_surf * e_x

        # Saturation
        sw = self._sat(s / (phi + 1e-12))

        # Eta floors
        eta_eq = torch.maximum(
            eta_val,
            torch.tensor(self._eta_eq_floor, device=eta_val.device, dtype=eta_val.dtype),
        )
        eta_sw = torch.maximum(
            eta_val,
            torch.tensor(self._eta_sw_floor, device=eta_val.device, dtype=eta_val.dtype),
        )

        F_sw   = (K_switch * sw) * eta_sw.unsqueeze(-1)
        F_task = (eta_eq.unsqueeze(-1) * F_eq) + F_sw
        tau_os = torch.einsum("bij,bj->bi", J_xy.transpose(-1, -2), F_task)

        # ----- Joint-space fallback (when singular) -----
        # desired q* via DLS map of position error
        e_x_pos = x_d - x
        q_star  = q + self._k_pos_map * torch.einsum("bij,bj->bi", J_pinv_dls, e_x_pos)

        Kp_js = self._to_tensor_like(self._Kp_js, q[0])
        Kd_js = self._to_tensor_like(self._Kd_js, q[0])

        qdd_js = Kp_js * (q_star - q) - Kd_js * qd
        tau_js = torch.einsum("bij,bj->bi", M, qdd_js) + h

        # ----- Blend OS & JS -----
        # condition on moment-arm matrix too (feasibility)
        geom = self.env.states["geometry"]            # (B, ·, M)
        R    = geom[:, 2:2 + self.env.skeleton.dof, :]  # (B,n,M)

        if R.shape[1] >= 2:
            condR = torch.linalg.cond(R)  # (B,)
        else:
            condR = torch.ones(B, device=R.device, dtype=R.dtype)

        w = self._blend_weight(eta_val, condR)  # (B,)

        tau_task = (1.0 - w.unsqueeze(-1)) * tau_os + w.unsqueeze(-1) * tau_js

        # ----- direct joint regulation -----
        tau_lim  = self._joint_limit_tau(q, qd)
        tau_bias = self._singularity_bias_tau(q, qd, eta_val)

        # Nullspace torque
        tau_ns = self._zeros_like(q)
        active_ns_mask = eta_val < 0.7
        if torch.any(active_ns_mask):
            J_dyn_pinv = torch.einsum(
                "bij,bjk->bik",
                Minv,
                torch.einsum("bij,bjk->bik", J_xy.transpose(-1, -2), Lambda_reg),
            )
            I_n = self._eye(n, q[0]).unsqueeze(0).expand(B, -1, -1)
            N   = I_n - torch.einsum("bij,bjk->bik", J_dyn_pinv, J_xy)

            qdd_manip, _ = add_nullspace_manip(
                self._zeros_like(q), self.env, q, qd, J_xy, self.kp, eta_val
            )
            tau_ns_active = (
                (1.0 - eta_val.unsqueeze(-1)) ** 2
                * self._manip_gain
                * torch.einsum(
                    "bij,bj->bi",
                    N.transpose(-1, -2),
                    torch.einsum("bij,bj->bi", M, qdd_manip) + h,
                )
            )

            # Apply only to active batches
            tau_ns = torch.where(
                active_ns_mask.unsqueeze(-1),
                tau_ns_active,
                tau_ns,
            )

        tau_des = tau_task + tau_lim + tau_bias + tau_ns

        # ----- muscles / allocation -----
        lenvel   = geom[:, :2, :]  # (B,2,M)
        Fmax_vec = get_Fmax_vec(self.env, R.shape[-1]).to(device=q.device, dtype=q.dtype)

        # feasibility-aware torque clip
        tau_feas     = torch.abs(R) @ Fmax_vec          # (B,n)
        tau_clip_dyn = 0.9 * torch.min(tau_feas, dim=1)[0]  # (B,)

        base_clip = self._tau_clip * torch.ones_like(tau_clip_dyn)
        low_clip  = 0.3 * base_clip

        tau_clip_use = torch.where(
            eta_val < 0.6,
            torch.maximum(
                low_clip,
                torch.minimum(base_clip, tau_clip_dyn),
            ),
            base_clip,
        )

        tau_des = torch.clamp(
            tau_des,
            -tau_clip_use.unsqueeze(-1),
            tau_clip_use.unsqueeze(-1),
        )

        # Filter (simple IIR)
        if not hasattr(self, "_tau_filt") or self._tau_filt.shape != tau_des.shape:
            self._tau_filt = self._zeros_like(tau_des)
        self._tau_filt = (1.0 - self._tau_alpha) * self._tau_filt + self._tau_alpha * tau_des
        tau_des = self._tau_filt

        # allocation & optional adaptive internal force
        names    = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe     = self.env.states["muscle"][:, idx_flpe, :]  # (B,M)

        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta_val, self.mp)

        if self.p.enable_internal_force:
            scale_if = torch.zeros(B, device=eta_val.device, dtype=eta_val.dtype)
            scale_if = torch.where(
                eta_val < 0.6,
                (0.6 - eta_val) * 0.6,
                scale_if,
            )
            scale_if = torch.where(
                condR > self._cond_high,
                torch.maximum(
                    scale_if,
                    torch.tensor(0.5, device=scale_if.device, dtype=scale_if.dtype),
                ),
                scale_if,
            )

            active_if_mask = scale_if > 0.0
            if torch.any(active_if_mask):
                a0_vec = torch.full(
                    F_des.shape,
                    self.p.cocon_a0,
                    device=F_des.device,
                    dtype=F_des.dtype,
                )
                af0   = active_force_from_activation(a0_vec, lenvel, self.env.muscle)
                F_bias = Fmax_vec * (af0 + flpe)

                F_des_active = apply_internal_force_regulation(
                    -R,
                    F_des,
                    F_bias,
                    Fmax_vec,
                    eps=self.p.eps,
                    linesearch_eps=self.p.linesearch_eps,
                    linesearch_safety=self.p.linesearch_safety,
                    scale=torch.clamp(scale_if, 0.0, 0.6),
                )

                # Apply only to active batches
                F_des = torch.where(
                    active_if_mask.unsqueeze(-1),
                    F_des_active,
                    F_des,
                )

        a = force_to_activation_bisect(
            F_des,
            lenvel,
            self.env.muscle,
            flpe,
            Fmax_vec,
            iters=self.p.bisect_iters,
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
        if torch.any(torch.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(
                F_corr,
                lenvel,
                self.env.muscle,
                flpe,
                Fmax_vec,
                iters=max(4, self.p.bisect_iters - 4),
            )

        # ------------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------------
        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ, alpha_J=alpha_J)
        sm_diag = pack_diag(
            s1=s[:, 0],
            s2=s[:, 1],
            phi1=phi[0],
            phi2=phi[1],
            tau_clip=tau_clip_use,
            lam_os=lam_os_val,
            eta=eta_val,
            eta2=eta2_val,
            w_js=w,
            condR=condR,
        )
        diag = merge_diag(kin_diag, dyn_diag, mus_diag, sm_diag)

        return {
            "tau_des": tau_des,
            "R":       R,
            "Fmax":    Fmax_vec,
            "F_des":   F_des,
            "act":     a,
            "q":       q,
            "qd":      qd,
            "x":       x,
            "xd":      xd,
            "xref_tuple": (x_d, xd_d, xdd_d),
            "eta":     eta2_val,
            "diag":    diag,
        }
# ---------------------------------------------------------------------------
# Simple Torch smoke test (RigidTendonArm26 + Hill)
# ---------------------------------------------------------------------------

def _smoke_test_arm26_hill_torch():
    """
    Torch-only smoke test for the Sliding-Mode controller:

    - RigidTendonArm26 (effector_torch) + RigidTendonHillMuscle
    - Environment (environment_torch)
    - SlidingModeController (this module)
    - Min-jerk straight-line trajectory (minjerk_torch)
    - TargetReachSimulatorTorch (simulator_torch)
    """

    import torch as _torch
    from model_lib.environment_torch import Environment as EnvTorch
    from model_lib.muscles_torch import RigidTendonHillMuscle
    from model_lib.effector_torch import RigidTendonArm26
    from trajectory.minjerk_torch import (
        MinJerkLinearTrajectoryTorch,
        MinJerkParams,
    )
    from sim.simulator_torch import TargetReachSimulatorTorch
    from config import (
        PlantConfig,
        ControlToggles,
        ControlGains,
        Numerics,
        InternalForceConfig,
        TrajectoryConfig,
    )

    print("\n[SlidingMode Torch] smoke test (RigidTendonArm26 + Hill) starting ...")
    _torch.set_default_dtype(_torch.float64)
    _torch.set_printoptions(precision=6, sci_mode=False)

    device = _torch.device("cpu")  # switch to "cuda" if you want

    # ---------- configs ----------
    pc = PlantConfig()
    toggles = ControlToggles()
    gains = ControlGains()
    num = Numerics()
    ifc = InternalForceConfig()
    tc = TrajectoryConfig()

    # ---------- build effector + env ----------
    muscle = RigidTendonHillMuscle(
        min_activation=0.02,
        device=device,
        dtype=_torch.get_default_dtype(),
    )

    arm = RigidTendonArm26(
        muscle=muscle,
        timestep=pc.timestep,
        damping=pc.damping,
        n_ministeps=pc.n_ministeps,
        integration_method=pc.integration_method,
        device=device,
        dtype=_torch.get_default_dtype(),
    )

    env = EnvTorch(
        effector=arm,
        max_ep_duration=pc.max_ep_duration,
        action_noise=0.0,
        obs_noise=0.0,
        action_frame_stacking=1,
        proprioception_delay=arm.dt,
        vision_delay=arm.dt,
        name="SlidingModeEnvTorch",
    )

    # ---------- initial joint state (single) ----------
    q0 = _torch.deg2rad(
        _torch.tensor(pc.q0_deg, dtype=_torch.get_default_dtype(), device=device)
    )
    qd0 = _torch.tensor(pc.qd0, dtype=_torch.get_default_dtype(), device=device)
    joint0 = _torch.cat([q0, qd0], dim=-1).view(1, -1)  # (1, 2*dof)

    env.reset(
        options={
            "joint_state": joint0,
            "deterministic": True,
        }
    )

    # ---------- simple min-jerk trajectory (center → shifted target) ----------
    # center = current fingertip position
    fingertip0 = env.states["fingertip"][0]
    center = fingertip0[:2]

    # target: move +10 cm in x (same pattern as your PD/IF demo)
    radius = 0.10
    target = center + _torch.tensor([radius, 0.0], dtype=center.dtype, device=device)

    waypoints = _torch.stack([center, target], dim=0)  # (2, 2)

    mj_params = MinJerkParams(
        Vmax=tc.Vmax,
        Amax=tc.Amax,
        Jmax=tc.Jmax,
        gamma=tc.gamma_time_scale,
    )
    traj = MinJerkLinearTrajectoryTorch(waypoints, mj_params)

    # ---------- Sliding-Mode controller parameters ----------
    # Use control gains & numerics to populate SlidingModeParams.
    # lambda_surf, K_switch, phi are heuristic defaults you can tune later.
    lambda_surf = _torch.as_tensor(
        [10.0, 10.0],
        device=device,
        dtype=_torch.get_default_dtype(),
    )
    K_switch = _torch.as_tensor(
        [2.0, 2.0],
        device=device,
        dtype=_torch.get_default_dtype(),
    )
    phi = _torch.as_tensor(
        [0.02, 0.02],
        device=device,
        dtype=_torch.get_default_dtype(),
    )

    p = SlidingModeParams(
        lambda_surf=lambda_surf,
        K_switch=K_switch,
        phi=phi,
        Kff_x=_torch.as_tensor(gains.Kff_x, device=device, dtype=_torch.get_default_dtype()),
        Kp_q=_torch.as_tensor(gains.Kp_q, device=device, dtype=_torch.get_default_dtype()),
        Kd_q=_torch.as_tensor(gains.Kd_q, device=device, dtype=_torch.get_default_dtype()),
        eps=float(num.eps),
        lam_os_max=float(num.lam_os_max),
        sigma_thresh=float(num.sigma_thresh),
        gate_pow=float(num.gate_pow),
        enable_inertia_comp=bool(toggles.enable_inertia_comp),
        enable_gravity_comp=bool(toggles.enable_gravity_comp),
        enable_velocity_comp=bool(toggles.enable_velocity_comp),
        enable_joint_damping=bool(toggles.enable_joint_damping),
        enable_internal_force=bool(toggles.enable_internal_force),
        cocon_a0=float(ifc.cocon_a0),
        bisect_iters=int(ifc.bisect_iters),
        linesearch_eps=float(num.linesearch_eps),
        linesearch_safety=float(num.linesearch_safety),
    )

    ctrl = SlidingModeController(env, arm, p)

    # ---------- simulate ----------
    steps = int(pc.max_ep_duration / arm.dt)
    sim = TargetReachSimulatorTorch(env, arm, ctrl, traj, steps)
    logs = sim.run()

    k, tvec = logs.time(arm.dt)
    x_log = logs.x_log[:k]  # (T, 2) — logger keeps the first batch sample

    # ---------- print a few samples like in your PD/IF Torch script ----------
    print("  fingertip (before):", center.detach().cpu().numpy())
    print("  target x_d:        ", target.detach().cpu().numpy())

    for idx in [0, 1, 2, 4, 9]:
        if idx < k:
            x = x_log[idx, :2]
            print(f"    step {idx:3d}: x = {x.detach().cpu().numpy()}")

    x_final = x_log[k - 1, :2]
    err_final = _torch.linalg.norm(x_final - target)
    print("  final fingertip:   ", x_final.detach().cpu().numpy())
    print(f"  final |x - x_d|:    {float(err_final):.6f} m")

    print("[SlidingMode Torch] smoke test (RigidTendonArm26 + Hill) complete ✓")


if __name__ == "__main__":
    _smoke_test_arm26_hill_torch()
