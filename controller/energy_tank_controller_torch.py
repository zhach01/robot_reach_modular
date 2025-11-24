# -*- coding: utf-8 -*-
"""
Pure-Torch Energy Tank controller (passivity / energy-tank PD).

Torch rewrite of controller/energy_tank_controller.py:
- No NumPy: everything in torch.
- HTM-based kinematics/dynamics:
    * lib.kinematics.HTM_kinematics_torch.{geometricJacobian, geometricJacobianDerivative}
    * lib.dynamics.DynamicsHTM_torch.{inertiaMatrixCOM, centrifugalCoriolisCOM, gravitationalCOM}
- Torch guard/muscle utilities:
    * utils.kinematics_guard_torch
    * utils.dynamics_guard_torch
    * utils.muscle_guard_torch
    * utils.math_utils_torch
    * utils.linear_utils_torch
    * utils.telemetry_torch
    * muscles.muscle_tools_torch

Assumptions
-----------
- env is a Torch environment exposing:
    - env.backend == "torch"
    - env.skeleton with attributes:
        * dof
        * _robot  (HTM-based Serial robot)
        * _gravity_vec (3x1 Tensor)
        * _set_state(q, qd)
    - env.muscle with attributes:
        * state_name
        * min_activation
        * max_iso_force
    - env.states dict with keys:
        * "joint"     : (B, 2*dof)       joint state [q, qd]
        * "cartesian" : (B, >=4)         [x, xd, ...]
        * "geometry"  : (B, 2 + dof, M)  [len, vel, R rows]
        * "muscle"    : (B, n_states, M) muscle states
- arm exposes:
    - dt      : time-step (float)
    - damping : joint viscous damping coefficient (float)

The public API matches PDIFController:
- reset(q0)
- compute(x_d, xd_d, xdd_d)
and returns a dict with the same keys as the NumPy tank controller.
"""


from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from utils.math_utils_torch import matrix_sqrt_spd, matrix_isqrt_spd
from utils.linear_utils_torch import nnls_small_active_set  # noqa: F401 (parity)
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
)
from utils.dynamics_guard_torch import (
    DynGuardParams,
    op_space_guard_and_gate,
)
from utils.muscle_guard_torch import (
    MuscleGuardParams,
    solve_muscle_forces,
)
from utils.telemetry_torch import pack_diag, merge_diag


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _to_tensor_like(x: Any, like: Tensor) -> Tensor:
    """Convert x to a tensor on like.device/like.dtype."""
    if isinstance(x, Tensor):
        return x.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(x, device=like.device, dtype=like.dtype)


def _eye(n: int, like: Tensor) -> Tensor:
    """Identity matrix with same device/dtype as `like`."""
    return torch.eye(n, device=like.device, dtype=like.dtype)


def _zeros(shape, like: Tensor) -> Tensor:
    """Zeros with same device/dtype as `like`."""
    return torch.zeros(*shape, device=like.device, dtype=like.dtype)


def _zeros_like(x: Tensor) -> Tensor:
    return torch.zeros_like(x)


def _decompose_parallel_perp(F: Tensor, v: Tensor, eps: float = 1e-9) -> Tuple[Tensor, Tensor]:
    """
    Batched decomposition of F into components parallel and orthogonal to v.

    F, v : (B,2)
    Returns F_par, F_perp (both (B,2)).
    """
    # nv2 = ||v||^2
    nv2 = (v * v).sum(dim=-1)  # (B,)
    dot = (F * v).sum(dim=-1)  # (B,)

    alpha = torch.zeros_like(dot)
    nonzero = nv2 > eps
    alpha[nonzero] = dot[nonzero] / nv2[nonzero]

    F_par = alpha.unsqueeze(-1) * v
    F_perp = F - F_par
    return F_par, F_perp


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class EnergyTankParams:
    # Gains (for planar task space: x,y)
    D0: Any        # passive task-space damping (2x2)
    K0: Any        # nominal task-space stiffness (2x2)
    KI: Any        # integral gain (2,)
    Imax: Any      # int windup limits (2,)

    # Numerics / operational-space
    eps: float
    lam_os_smin_target: float  # kept for compatibility, DynGuard handles actual reg
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


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class EnergyTankController:
    """Passivity / energy-tank controller with PDIF-compatible public API."""

    def __init__(self, env, arm, params: EnergyTankParams):
        self.env = env
        self.arm = arm
        self.p = params

        # joint reference for nullspace-like PD (mirrors PDIF style)
        self.qref: Tensor | None = None

        # guard params
        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=self.p.eps,
            lam_os_max=self.p.lam_os_max,
            gate_pow=self.p.gate_pow,
            sigma_thresh_S=max(self.p.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()

        # tank state (initialized in reset when batch size is known)
        self.E: Tensor | None = None      # (B,)
        self.I: Tensor | None = None      # (B,2)
        self.K_prev: Tensor | None = None # (B,2,2)

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------
    def reset(self, q0: Tensor) -> None:
        """
        Reset joint reference and tank state.

        q0:
            - (n,)   for a single sample
            - (B,n)  for batched joints
        """
        q0 = q0.to(dtype=torch.get_default_dtype())

        if q0.ndim == 1:
            q0 = q0.unsqueeze(0)  # (1,n)
        elif q0.ndim != 2:
            raise ValueError(
                f"EnergyTankController.reset: q0 must be (n,) or (B,n), got {tuple(q0.shape)}"
            )

        self.qref = q0.clone()      # (B,n)

        B, n = q0.shape
        device = q0.device
        dtype = q0.dtype

        self.E = torch.full(
            (B,), float(self.p.E0), device=device, dtype=dtype
        )  # (B,)
        self.I = torch.zeros((B, 2), device=device, dtype=dtype)  # (B,2)
        self.K_prev = None

    # legacy unbatched API for parity if needed
    def reset_(self, q0: Tensor) -> None:
        if q0.ndim != 1:
            raise ValueError("reset_ expects q0 with shape (n,).")
        self.reset(q0)

    # ----------------------------------------------------------
    # Dynamics (batched)
    # ----------------------------------------------------------
    def _compute_dynamics(self, q: Tensor, qd: Tensor):
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
            # M_any can be (n,n) or (B0,n,n)
            if M_any.ndim == 2:
                # (n,n) -> (1,n,n) -> (B,n,n)
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
            M_single = _eye(n, q[0])               # (n,n)
            M = M_single.unsqueeze(0).expand(B, -1, -1)

        # ------------------------------------------------------------------
        # Gravity g(q)
        # ------------------------------------------------------------------
        if p.enable_gravity_comp:
            g_any = _dyn.gravitationalCOM(
                self.env.skeleton._robot,
                g=self.env.skeleton._gravity_vec,
            )
            # g_any: (n,1) or (B0,n,1)
            if g_any.ndim == 2:
                g_single = g_any.to(device=device, dtype=dtype).view(-1)  # (n,)
                g = g_single.unsqueeze(0).expand(B, -1)                   # (B,n)
            elif g_any.ndim == 3:
                g_any = g_any.to(device=device, dtype=dtype)
                B0 = g_any.shape[0]
                if B0 == 1 and B > 1:
                    g_b = g_any.expand(B, -1, -1)
                elif B0 == B:
                    g_b = g_any
                else:
                    g_b = g_any[0:1, :, :].expand(B, -1, -1)
                g = g_b[..., 0]                                          # (B,n)
            else:
                raise ValueError(
                    f"gravitationalCOM must be (n,1) or (B,n,1), got {tuple(g_any.shape)}"
                )
        else:
            g = _zeros_like(q)   # (B,n)

        # ------------------------------------------------------------------
        # Centrifugal / Coriolis C(q,q̇)
        # ------------------------------------------------------------------
        if p.enable_velocity_comp:
            C_any = _dyn.centrifugalCoriolisCOM(self.env.skeleton._robot)
            # C_any: (n,n) or (B0,n,n)
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
            C = _zeros((B, n, n), q)  # (B,n,n)

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
            D_single = torch.diag(damping_vec)        # (n,n)
            D = D_single.unsqueeze(0).expand(B, -1, -1)
        else:
            D = _zeros((B, n, n), q)                  # (B,n,n)

        # ------------------------------------------------------------------
        # Bias term: h(q,q̇) = C q̇ + g + D q̇   (B,n)
        # ------------------------------------------------------------------
        h_C = torch.einsum("bij,bj->bi", C, qd)  # (B,n)
        h_D = torch.einsum("bij,bj->bi", D, qd)  # (B,n)
        h = h_C + g + h_D                        # (B,n)

        return M, h

    # ----------------------------------------------------------
    # Main compute (batched)
    # ----------------------------------------------------------
    def compute(self, x_d: Tensor, xd_d: Tensor, xdd_d: Tensor) -> Dict[str, Any]:
        """
        Batched energy-tank step.

        Inputs (batched):
          x_d, xd_d, xdd_d :
              - shape (2,)   -> treated as (1,2)
              - shape (B,2)

        Returns
        -------
        dict with keys (batched, B samples):
            - "tau_des": (B,n)
            - "R":       (B,n,M)
            - "Fmax":    (M,)
            - "F_des":   (B,M)
            - "act":     (B,M)
            - "q", "qd": (B,n)
            - "x", "xd": (B,2)
            - "xref_tuple": (x_d, xd_d, xdd_d) each (B,2)
            - "eta":     (B,)
            - "tank":    dict with tank power terms (all (B,))
            - "diag":    telemetry dict (pack_diag/merge_diag)
        """
        # ------------------------------------------------------------------
        # Current state
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
        def _ensure_batch2(z: Tensor, like: Tensor) -> Tensor:
            z = z.to(device=like.device, dtype=like.dtype)
            if z.ndim == 1:
                z = z.unsqueeze(0)  # (1,2)
            if z.shape[0] == 1 and B > 1:
                z = z.expand(B, -1)
            elif z.shape[0] != B:
                raise ValueError(
                    f"Ref tensor batch {z.shape[0]} != state batch {B}"
                )
            return z

        x_d = _ensure_batch2(x_d, x)
        xd_d = _ensure_batch2(xd_d, x)
        xdd_d = _ensure_batch2(xdd_d, x)

        # ------------------------------------------------------------------
        # Batched qref & tank state
        # ------------------------------------------------------------------
        if self.qref is None:
            qref = q.clone()
        else:
            qref = _to_tensor_like(self.qref, q)
            if qref.ndim == 1:
                qref = qref.unsqueeze(0)
            if qref.shape[0] == 1 and B > 1:
                qref = qref.expand(B, -1)
            elif qref.shape[0] != B:
                qref = qref[0:1, :].expand(B, -1)
        self.qref = qref  # (B,n)

        # tank state (init lazily if needed)
        if self.E is None or self.E.shape[0] != B:
            device = q.device
            dtype = q.dtype
            self.E = torch.full((B,), float(self.p.E0), device=device, dtype=dtype)
            self.I = torch.zeros((B, 2), device=device, dtype=dtype)
            self.K_prev = None

        # ------------------------------------------------------------------
        # Jacobians (broadcast to (B,6,n) if needed)
        # ------------------------------------------------------------------
        J_any = _kin.geometricJacobian(self.env.skeleton._robot)  # (6,n) or (B,6,n)
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
        # [1a] adaptive DLS on J; [1b] kinematic scaling (for qref integration)
        # ------------------------------------------------------------------
        J_pinv_dls, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)
        xd_d, xdd_d, alpha_J = scale_task_by_J(xd_d, xdd_d, sminJ, self.kp)

        # q̇_des = J⁺_DLS ẋ_d  -> (B,n)
        xd_d_col = xd_d.unsqueeze(-1)                # (B,2,1)
        qd_des = (J_pinv_dls @ xd_d_col).squeeze(-1)  # (B,n)

        # integrate qref
        qref = qref + qd_des * float(self.arm.dt)
        self.qref = qref

        # ------------------------------------------------------------------
        # Dynamics & op-space inertia
        # ------------------------------------------------------------------
        M, h = self._compute_dynamics(q, qd)    # M:(B,n,n), h:(B,n)
        Minv = torch.linalg.inv(M)             # (B,n,n)

        Jt = J_xy.transpose(-1, -2)                               # (B,n,2)
        MinvJt = torch.einsum("bij,bjk->bik", Minv, Jt)           # (B,n,2)
        S = torch.einsum("bij,bjk->bik", J_xy, MinvJt)            # (B,2,2)

        # ------------------------------------------------------------------
        # [2a]/[2b]/[2c] op-space guard + gate
        # ------------------------------------------------------------------
        Lambda, lam_os, eta, eta2, xd_d, xdd_d, dyn_diag = op_space_guard_and_gate(
            S, xd_d, xdd_d, self.dp
        )

        # Normalize shapes
        if Lambda.ndim == 2:
            Lambda = Lambda.unsqueeze(0).expand(B, -1, -1)
        if xd_d.ndim == 1:
            xd_d = xd_d.unsqueeze(0).expand(B, -1)
        if xdd_d.ndim == 1:
            xdd_d = xdd_d.unsqueeze(0).expand(B, -1)

        def _ensure_batch1(s):
            if isinstance(s, Tensor):
                s = s.to(device=x.device, dtype=x.dtype)
                if s.ndim == 0:
                    s = s.view(1).expand(B)
                elif s.shape[0] == 1 and B > 1:
                    s = s.expand(B)
            else:
                s = torch.full(
                    (B,),
                    float(s),
                    device=x.device,
                    dtype=x.dtype,
                )
            return s

        eta = _ensure_batch1(eta)    # (B,)
        eta2 = _ensure_batch1(eta2)  # (B,)

        # ------------------------------------------------------------------
        # Task-space errors
        # ------------------------------------------------------------------
        e_x = x_d - x          # (B,2)
        e_v = xd_d - xd        # (B,2)

        # ------------------------------------------------------------------
        # Passive damping and dissipation power
        # ------------------------------------------------------------------
        D0_mat = _to_tensor_like(self.p.D0, x[0])  # (2,2)

        F_pas = -torch.einsum("ij,bj->bi", D0_mat, xd)  # (B,2)

        D0_xd = torch.einsum("ij,bj->bi", D0_mat, xd)   # (B,2)
        P_diss = torch.einsum("bi,bi->b", xd, D0_xd)    # (B,) ≥ 0

        # ------------------------------------------------------------------
        # Kv for critical damping using exact symmetric form
        # ------------------------------------------------------------------
        K0_mat = _to_tensor_like(self.p.K0, x[0])   # (2,2)

        Lam_s = matrix_sqrt_spd(Lambda)             # (B,2,2)
        Lam_is = matrix_isqrt_spd(Lambda)           # (B,2,2)

        tmpK = torch.einsum("bij,jk->bik", Lam_is, K0_mat)    # (B,2,2)
        tmpK = torch.einsum("bij,bjk->bik", tmpK, Lam_is)     # (B,2,2)
        sqrt_tmpK = matrix_sqrt_spd(tmpK)                     # (B,2,2)

        Kv = 2.0 * torch.einsum("bij,bjk->bik", Lam_s, sqrt_tmpK)
        Kv = torch.einsum("bij,bjk->bik", Kv, Lam_s)          # (B,2,2)

        # ------------------------------------------------------------------
        # Integral term (anti-windup)
        # ------------------------------------------------------------------
        if self.I is None or self.I.shape[0] != B:
            self.I = torch.zeros((B, 2), device=x.device, dtype=x.dtype)

        KI_vec = _to_tensor_like(self.p.KI, x[0])      # (2,)
        Imax_vec = _to_tensor_like(self.p.Imax, x[0])  # (2,)

        Imax_B = Imax_vec.unsqueeze(0).expand(B, -1)   # (B,2)

        self.I = torch.clamp(
            self.I + e_x * float(self.arm.dt),
            min=-Imax_B,
            max=Imax_B,
        )  # (B,2)

        F_I = KI_vec.unsqueeze(0) * self.I             # (B,2)

        # ------------------------------------------------------------------
        # Variable stiffness bookkeeping (Kdot power)
        # ------------------------------------------------------------------
        K_now = K0_mat.unsqueeze(0).expand(B, -1, -1)  # (B,2,2)
        if self.K_prev is None or self.K_prev.shape != K_now.shape:
            Kdot = torch.zeros_like(K_now)
        else:
            Kdot = (K_now - self.K_prev) / float(self.arm.dt)
        self.K_prev = K_now.clone()

        # parameter-variation power P_K = -0.5 * e_xᵀ Kdot e_x
        tmp_eK = torch.einsum("bij,bj->bi", Kdot, e_x)   # (B,2)
        eK = torch.einsum("bi,bi->b", e_x, tmp_eK)       # (B,)
        P_K = -0.5 * eK                                  # (B,)

        P_refund = torch.clamp(P_K, min=0.0)             # (B,)  (K decreasing)
        P_spend = torch.clamp(-P_K, min=0.0)             # (B,)  (K increasing)

        # ------------------------------------------------------------------
        # Task-space bias from dynamics
        # ------------------------------------------------------------------
        Minv_h = torch.einsum("bij,bj->bi", Minv, h)          # (B,n)
        JMinvh = torch.einsum("bij,bj->bi", J_xy, Minv_h)     # (B,2)
        Jdot_qd = torch.einsum("bij,bj->bi", Jdot_xy, qd)     # (B,2)
        mu = torch.einsum("bij,bj->bi", Lambda, JMinvh - Jdot_qd)  # (B,2)

        # ------------------------------------------------------------------
        # Active raw force (may inject energy)
        # F_act_raw = Lambda xdd_d + mu + K0 e_x + Kv e_v + F_I
        # ------------------------------------------------------------------
        Lam_xdd = torch.einsum("bij,bj->bi", Lambda, xdd_d)       # (B,2)
        K0_ex = torch.einsum("ij,bj->bi", K0_mat, e_x)            # (B,2)
        Kv_ev = torch.einsum("bij,bj->bi", Kv, e_v)               # (B,2)

        F_act_raw = Lam_xdd + mu + K0_ex + Kv_ev + F_I           # (B,2)

        # Only the component parallel to true velocity xd can inject/extract energy
        F_par, F_perp = _decompose_parallel_perp(F_act_raw, xd)  # (B,2),(B,2)
        P_par_raw = torch.einsum("bi,bi->b", F_par, xd)          # (B,)

        # ------------------------------------------------------------------
        # Energy-tank gate
        # ------------------------------------------------------------------
        # Required power: for parallel injection + K-increase
        P_need = torch.clamp(P_par_raw, min=0.0) + P_spend       # (B,)

        E = self.E                                               # (B,)
        if E is None:
            E = torch.full((B,), float(self.p.E0),
                           device=x.device, dtype=x.dtype)

        denom = self.arm.dt * P_need + float(self.p.eps)         # (B,)
        ratio = (E - float(self.p.Emin)) / denom                 # (B,)

        s = torch.ones_like(ratio)                               # (B,)
        mask = P_need > 0.0
        s[mask] = torch.clamp(ratio[mask], min=0.0, max=1.0)
        s[~mask] = 1.0

        # Final command force
        F_cmd = F_pas + F_perp + s.unsqueeze(-1) * F_par         # (B,2)

        # ------------------------------------------------------------------
        # Joint torques from tank force
        # ------------------------------------------------------------------
        tau_des = torch.einsum(
            "bij,bj->bi", J_xy.transpose(-1, -2), F_cmd
        )  # (B,n)

        # ------------------------------------------------------------------
        # Tank energy update
        # ------------------------------------------------------------------
        # Pin: dissipated + refund (only if E < Emax)
        P_diss_clamped = P_diss
        Pin = torch.where(
            E < float(self.p.Emax),
            P_diss_clamped + P_refund,
            torch.zeros_like(P_diss_clamped),
        )  # (B,)

        # Pout: gate-controlled parallel injection + K-increase
        Pout = s * torch.clamp(P_par_raw, min=0.0) + P_spend      # (B,)

        E_next = E + float(self.arm.dt) * (Pin - Pout)
        E_next = torch.clamp(E_next, float(self.p.Emin), float(self.p.Emax))
        self.E = E_next                                           # (B,)

        # ------------------------------------------------------------------
        # Muscle geometry & forces (batched)
        # ------------------------------------------------------------------
        geom = self.env.states["geometry"]  # (B, 2 + n, M)
        if geom.ndim != 3:
            raise ValueError(
                f"env.states['geometry'] must be (B, 2 + dof, M), got {tuple(geom.shape)}"
            )

        lenvel = geom[:, :2, :]             # (B,2,M)
        R = geom[:, 2:2 + n, :]             # (B,n,M)
        M_muscles = int(R.shape[-1])

        Fmax_vec = get_Fmax_vec(
            self.env,
            M_muscles,
        ).to(device=q.device, dtype=q.dtype)  # (M,)

        names = list(getattr(self.env.muscle, "state_name", []))
        if "force-length PE" in names:
            idx_flpe = names.index("force-length PE")
            flpe = self.env.states["muscle"][:, idx_flpe, :]      # (B,M)
        else:
            flpe = torch.zeros((B, M_muscles), device=q.device, dtype=q.dtype)

        # Allocation using shared gate (eta)
        F_des, mus_diag = solve_muscle_forces(
            tau_des, R, Fmax_vec, eta, self.mp
        )  # (B,M)

        # ------------------------------------------------------------------
        # Internal-force regulation (optional, scaled by eta2)
        # ------------------------------------------------------------------
        if self.p.enable_internal_force:
            a0_vec = torch.full(
                (B, M_muscles),
                float(self.p.cocon_a0),
                device=F_des.device,
                dtype=F_des.dtype,
            )
            af = active_force_from_activation(a0_vec, lenvel, self.env.muscle)  # (B,M)
            F_bias = Fmax_vec * (af + flpe)                                     # (B,M)

            F_des = apply_internal_force_regulation(
                -R,
                F_des,
                F_bias,
                Fmax_vec,
                eps=float(self.p.eps),
                linesearch_eps=float(self.p.linesearch_eps),
                linesearch_safety=float(self.p.linesearch_safety),
                scale=eta2,
            )

        # ------------------------------------------------------------------
        # Activation back-projection & saturation repair
        # ------------------------------------------------------------------
        a = force_to_activation_bisect(
            F_des,
            lenvel,
            self.env.muscle,
            flpe,
            Fmax_vec,
            iters=int(self.p.bisect_iters),
        )  # (B,M)

        af_now = active_force_from_activation(a, lenvel, self.env.muscle)  # (B,M)
        F_pred = Fmax_vec * (af_now + flpe)                                # (B,M)

        F_corr = saturation_repair_tau(
            -R,
            F_pred,
            a,
            self.env.muscle.min_activation,
            1.0,
            Fmax_vec,
            tau_des=tau_des,
        )  # (B,M)

        if torch.any(torch.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(
                F_corr,
                lenvel,
                self.env.muscle,
                flpe,
                Fmax_vec,
                iters=max(4, int(self.p.bisect_iters) - 4),
            )

        # ------------------------------------------------------------------
        # Diagnostics (kinematics + dynamics + muscles + tank)
        # ------------------------------------------------------------------
        kin_diag = pack_diag(
            sminJ=sminJ,
            lamJ=lamJ,
            alpha_J=alpha_J,
        )
        tank_diag = pack_diag(
            E=self.E,
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

        return {
            "tau_des": tau_des,                           # (B,n)
            "R": R,                                       # (B,n,M)
            "Fmax": Fmax_vec,                             # (M,)
            "F_des": F_des,                               # (B,M)
            "act": a,                                     # (B,M)
            "q": q,                                       # (B,n)
            "qd": qd,                                     # (B,n)
            "x": x,                                       # (B,2)
            "xd": xd,                                     # (B,2)
            "xref_tuple": (x_d, xd_d, xdd_d),             # each (B,2)
            "eta": eta2,                                  # (B,)
            "tank": {
                "E": self.E,            # (B,)
                "s": s,                 # (B,)
                "P_diss": P_diss,       # (B,)
                "P_par_raw": P_par_raw, # (B,)
                "P_K": P_K,             # (B,)
                "P_refund": P_refund,   # (B,)
                "P_spend": P_spend,     # (B,)
            },
            "diag": diag,
        }

    # Optional legacy-style unbatched wrapper (if you ever need it)
    def compute_(self, x_d: Tensor, xd_d: Tensor, xdd_d: Tensor) -> Dict[str, Any]:
        """
        Unbatched wrapper: x_d, xd_d, xdd_d are (2,). Returns first batch slice.
        """
        out = self.compute(
            x_d.view(1, -1),
            xd_d.view(1, -1),
            xdd_d.view(1, -1),
        )
        # unwrap batch dimension for main fields (others left batched in diag/tank)
        return {
            "tau_des": out["tau_des"][0],
            "R": out["R"][0],
            "Fmax": out["Fmax"],
            "F_des": out["F_des"][0],
            "act": out["act"][0],
            "q": out["q"][0],
            "qd": out["qd"][0],
            "x": out["x"][0],
            "xd": out["xd"][0],
            "xref_tuple": tuple(t[0] for t in out["xref_tuple"]),
            "eta": out["eta"][0],
            "tank": {k: v[0] for k, v in out["tank"].items()},
            "diag": out["diag"],
        }


# ---------------------------------------------------------------------------
# Torch smoke test: RigidTendonArm26 + Hill muscle + Energy Tank + min-jerk
# ---------------------------------------------------------------------------
def _smoke_test_arm26_hill_torch():
    """
    Torch-only smoke test (batched):

    - RigidTendonArm26 (effector_torch) + RigidTendonHillMuscle
    - Environment (environment_torch)
    - EnergyTankController (this module, batched compute())
    - Min-jerk straight-line trajectory (minjerk_torch)
    - TargetReachSimulatorTorch (simulator_torch)

    Patterned on the NumPy random_reach demo.
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

    print("\n[EnergyTank Torch] smoke test (RigidTendonArm26 + Hill) starting ...")
    _torch.set_default_dtype(_torch.float64)
    _torch.set_printoptions(precision=6, sci_mode=False)

    # choose device; you can force "cpu" if you prefer
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

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
        name="RandomReachEnvTorch_EnergyTank",
    )

    # ---------- initial joint state (batched) ----------
    B = 2  # batch size

    q0 = _torch.deg2rad(
        _torch.tensor(pc.q0_deg, dtype=_torch.get_default_dtype(), device=device)
    )  # (2,)
    qd0 = _torch.tensor(pc.qd0, dtype=_torch.get_default_dtype(), device=device)  # (2,)

    joint_single = _torch.cat([q0, qd0], dim=-1)              # (2*dof,) = (4,)
    joint0 = joint_single.unsqueeze(0).expand(B, -1).clone()  # (B, 2*dof)

    env.reset(
        options={
            "joint_state": joint0,   # batched initial state
            "deterministic": True,
        }
    )

    # ---------- simple min-jerk trajectory (center → shifted target) ----------
    # center = current fingertip position (take batch 0 as reference)
    fingertip0 = env.states["fingertip"][0]  # (>=2,)
    center = fingertip0[:2]                  # (2,)

    # target: move +10 cm in x (same pattern as NumPy demo)
    radius = 0.10
    target = center + _torch.tensor([radius, 0.0], dtype=center.dtype, device=device)

    waypoints = _torch.stack([center, target], dim=0)  # (2,2)

    mj_params = MinJerkParams(
        Vmax=tc.Vmax,
        Amax=tc.Amax,
        Jmax=tc.Jmax,
        gamma=tc.gamma_time_scale,
    )
    traj = MinJerkLinearTrajectoryTorch(waypoints, mj_params)

    # ---------- Energy Tank controller parameters ----------
    p = EnergyTankParams(
        D0=gains.D0,                     # 2x2
        K0=gains.K0,                     # 2x2
        KI=gains.KI,                     # len=2
        Imax=gains.Imax,                 # len=2
        eps=num.eps,
        lam_os_smin_target=num.lam_os_smin_target,
        lam_os_max=num.lam_os_max,
        sigma_thresh=num.sigma_thresh,
        gate_pow=num.gate_pow,
        enable_inertia_comp=toggles.enable_inertia_comp,
        enable_gravity_comp=toggles.enable_gravity_comp,
        enable_velocity_comp=toggles.enable_velocity_comp,
        enable_joint_damping=toggles.enable_joint_damping,
        enable_internal_force=toggles.enable_internal_force,
        cocon_a0=ifc.cocon_a0,
        bisect_iters=ifc.bisect_iters,
        linesearch_eps=num.linesearch_eps,
        linesearch_safety=num.linesearch_safety,
        # E0, Emin, Emax use dataclass defaults unless you want to override
        # E0=0.5, Emin=1e-4, Emax=0.5,
    )

    ctrl = EnergyTankController(env, arm, p)

    # Explicitly set qref with batched reset (optional; compute() would init it)
    ctrl.reset(q0.view(1, -1))  # (1,2) -> stored as (1,2) then expanded to B

    # ---------- simulate ----------
    steps = int(pc.max_ep_duration / arm.dt)
    sim = TargetReachSimulatorTorch(env, arm, ctrl, traj, steps)
    logs = sim.run()

    k, tvec = logs.time(arm.dt)
    x_log = logs.x_log[:k]  # (T, 4) or (T, >=2)

    # ---------- print a few samples like in your NumPy script ----------
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

    print("[EnergyTank Torch] smoke test (RigidTendonArm26 + Hill) complete ✓")


if __name__ == "__main__":
    _smoke_test_arm26_hill_torch()
