# controller/pd_if_controller_torch.py
# -*- coding: utf-8 -*-
"""
Pure-Torch PD + Internal Force (PD/IF) controller.

Torch rewrite of controller/pd_if_controller.py:
- No NumPy: everything is in torch.
- Uses HTM-based kinematics/dynamics:
    * lib.kinematics.HTM_kinematics_torch.{geometricJacobian, geometricJacobianDerivative}
    * lib.dynamics.DynamicsHTM_torch.{inertiaMatrixCOM, centrifugalCoriolisCOM, gravitationalCOM}
- Uses Torch guard/muscle utilities:
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
        * _robot (HTM-based Serial robot)
        * _gravity_vec (3x1 Tensor)
        * _set_state(q, qd)
    - env.muscle with attributes:
        * state_name
        * min_activation
        * max_iso_force
    - env.states dict with keys:
        * "joint"     : (1, 2*dof) joint state [q, qd]
        * "cartesian" : (1, 2*space_dim) [x, xd]
        * "geometry"  : (1, 2 + dof, M)  [len, vel, R rows]
        * "muscle"    : (1, n_states, M) muscle states
- arm exposes:
    - dt      : time-step (float)
    - damping : joint viscous damping coefficient (float)
"""

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import Tensor

from utils.math_utils_torch import matrix_sqrt_spd, matrix_isqrt_spd
from utils.linear_utils_torch import nnls_small_active_set  # parity import (unused)
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


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class PDIFParams:
    # task-space PD + feedforward gains (length 2 for planar x,y)
    Kp_x: Any
    Kff_x: Any

    # joint-space nullspace PD gains (length = dof)
    Kp_q: Any
    Kd_q: Any

    # dynamic guard / numerical safety
    eps: float
    lam_os_smin_target: float    # kept for parity, not used directly in Torch guard
    lam_os_max: float
    sigma_thresh: float
    gate_pow: float

    # feature toggles
    enable_internal_force: bool
    enable_inertia_comp: bool
    enable_gravity_comp: bool
    enable_velocity_comp: bool
    enable_joint_damping: bool

    # internal force regulation
    cocon_a0: float
    bisect_iters: int
    linesearch_eps: float
    linesearch_safety: float


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class PDIFController:
    def __init__(self, env, arm, params: PDIFParams):
        self.env = env
        self.arm = arm
        self.p = params

        # joint reference used by nullspace joint PD (initialized on first call)
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

    # ----------------------------------------------------------
    # Legacy API: unbatched reset / dynamics / compute
    # ----------------------------------------------------------
    def reset_(self, q0: Tensor):
        """
        Legacy: reset the joint reference to q0 (unbatched).

        q0 : (dof,) Torch tensor.
        """
        self.qref = q0.clone()

    def _compute_dynamics_(self, q: Tensor, qd: Tensor):
        """
        Legacy: unbatched M(q) and h(q, q̇) with Torch HTM dynamics.

        q, qd: (n,)

        Returns
        -------
        M : (n,n) inertia matrix
        h : (n,)   bias term = C(q,q̇) q̇ + g(q) + D q̇
        """
        p = self.p
        n = q.shape[0]

        # --- inertia
        if p.enable_inertia_comp:
            M_any = _dyn.inertiaMatrixCOM(self.env.skeleton._robot)
            # (n,n) or (B,n,n) -> take first slice if batched
            if M_any.dim() == 3:
                M = M_any[0]
            else:
                M = M_any
        else:
            M = _eye(n, q)

        # --- gravity
        if p.enable_gravity_comp:
            g_any = _dyn.gravitationalCOM(
                self.env.skeleton._robot,
                g=self.env.skeleton._gravity_vec,
            )  # (n,1) or (B,n,1)
            if g_any.dim() == 3:
                g = g_any[0].view(-1)
            else:
                g = g_any.view(-1)
        else:
            g = _zeros_like(q)

        # --- Coriolis / centrifugal
        if p.enable_velocity_comp:
            C_any = _dyn.centrifugalCoriolisCOM(self.env.skeleton._robot)
        else:
            C_any = _zeros((n, n), q)

        if C_any.dim() == 3:
            C = C_any[0]
        else:
            C = C_any

        # --- joint viscous damping
        if p.enable_joint_damping:
            damping_vec = torch.full(
                (n,),
                float(self.arm.damping),
                device=q.device,
                dtype=q.dtype,
            )
            D = torch.diag(damping_vec)
        else:
            D = _zeros((n, n), q)

        # bias term: C q̇ + g + D q̇
        h = C @ qd + g + D @ qd
        return M, h

    def compute_(self, x_d: Tensor, xd_d: Tensor, xdd_d: Tensor) -> Dict[str, Any]:
        """
        Legacy: unbatched compute.

        Inputs
        ------
        x_d, xd_d, xdd_d : (2,) Tensors (x, y for planar arm).

        Returns
        -------
        dict with keys (unbatched):
            - "tau_des" : (n,)
            - "R"       : (n,M)
            - "Fmax"    : (M,)
            - "F_des"   : (M,)
            - "act"     : (M,)
            - ...
        """
        # ------------------------------------------------------------------
        # State (unbatched)
        # ------------------------------------------------------------------
        joint = self.env.states["joint"][0]  # (2*dof,)
        q, qd = joint[:2], joint[2:]

        cart = self.env.states["cartesian"][0]  # (2*space_dim,)
        x, xd = cart[:2], cart[2:]

        # Make sure robot is updated
        self.env.skeleton._set_state(q, qd)

        # Initialize qref on first call if needed
        if self.qref is None:
            self.qref = q.clone()

        # Ensure references are tensors on same device/dtype
        x_d = x_d.to(device=x.device, dtype=x.dtype)
        xd_d = xd_d.to(device=x.device, dtype=x.dtype)
        xdd_d = xdd_d.to(device=x.device, dtype=x.dtype)

        # ------------------------------------------------------------------
        # Jacobians (HTM kinematics, Torch)
        # ------------------------------------------------------------------
        J_any = _kin.geometricJacobian(self.env.skeleton._robot)  # (6,n) or (B,6,n)
        if J_any.dim() == 3:
            J = J_any[0]
        else:
            J = J_any
        J_xy = J[0:2, :]  # (2,n)

        Jdot_any = _kin.geometricJacobianDerivative(self.env.skeleton._robot)
        if Jdot_any.dim() == 3:
            Jdot = Jdot_any[0]
        else:
            Jdot = Jdot_any
        Jdot_xy = Jdot[0:2, :]

        n = q.shape[0]

        # ------------------------------------------------------------------
        # [1a] adaptive DLS on J; [1b] kinematic scaling
        # ------------------------------------------------------------------
        J_pinv_dls, sminJ, lamJ = adaptive_dls_pinv(J_xy, int(n), self.kp)
        xd_d, xdd_d, alpha_J = scale_task_by_J(xd_d, xdd_d, sminJ, self.kp)

        qd_des = J_pinv_dls @ xd_d
        self.qref = self.qref + qd_des * float(self.arm.dt)

        # ------------------------------------------------------------------
        # Dynamics: M, h, and operational-space inertia S
        # ------------------------------------------------------------------
        M, h = self._compute_dynamics_(q, qd)
        Minv = torch.linalg.inv(M)

        S = J_xy @ Minv @ J_xy.T  # (2,2)

        # [2a]/[2b]/[2c] op-space guard + gate
        (
            Lambda,
            lam_os,
            eta,
            eta2,
            xd_d,
            xdd_d,
            dyn_diag,
        ) = op_space_guard_and_gate(S, xd_d, xdd_d, self.dp)

        # ------------------------------------------------------------------
        # Operational-space control law
        # ------------------------------------------------------------------
        mu = Lambda @ (J_xy @ (Minv @ h) - Jdot_xy @ qd)

        # task-space tracking errors (textbook convention)
        e_x = x_d - x
        ed_x = xd_d - xd

        # Gains as matrices
        Kp_x = _to_tensor_like(self.p.Kp_x, x)
        Kff_x = _to_tensor_like(self.p.Kff_x, x)

        Kp_mat = torch.diag(Kp_x)
        Kff_mat = torch.diag(Kff_x)

        Lam_sqrt = matrix_sqrt_spd(Lambda)
        Lam_isqrt = matrix_isqrt_spd(Lambda)
        Kv_mat = (
            2.0
            * Lam_sqrt
            @ matrix_sqrt_spd(Lam_isqrt @ Kp_mat @ Lam_isqrt)
            @ Lam_sqrt
        )

        xdd_r = Kff_mat @ xdd_d + Kv_mat @ ed_x + Kp_mat @ e_x
        F_task = Lambda @ xdd_r + mu
        tau_task = J_xy.T @ F_task  # (n,)

        # ------------------------------------------------------------------
        # Nullspace policy (joint PD + manipulability gradient)
        # ------------------------------------------------------------------
        J_dyn_pinv = Minv @ J_xy.T @ Lambda
        N = torch.eye(n, device=q.device, dtype=q.dtype) - J_dyn_pinv @ J_xy

        Kp_q = _to_tensor_like(self.p.Kp_q, q)
        Kd_q = _to_tensor_like(self.p.Kd_q, q)

        qdd_post = Kp_q * (self.qref - q) - Kd_q * qd

        # manipulability gradient in nullspace (faded by eta)
        qdd_post, k_manip_used = add_nullspace_manip(
            qdd_post, self.env, q, qd, J_xy, self.kp, eta
        )

        tau_post = M @ qdd_post + h
        # NOTE: keep the minus (back off nullspace when eta2 → 0 near singularities)
        tau_des = tau_task - eta2 * (N.T @ tau_post)

        # ------------------------------------------------------------------
        # Muscle geometry & forces
        # ------------------------------------------------------------------
        geom = self.env.states["geometry"]  # (1, 2 + dof, M)
        lenvel = geom[:, :2, :]            # (1,2,M)
        R = geom[:, 2 : 2 + self.env.skeleton.dof, :][0]  # (n,M)

        Fmax_vec = get_Fmax_vec(
            self.env,
            R.shape[1],
            device=R.device,
            dtype=R.dtype,
        )  # (M,)

        names = list(self.env.muscle.state_name)
        idx_flpe = names.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]  # (M,)

        # [3a]/[3b]/[3c] robust muscle solve (WLS/NNLS) with shared gate
        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta, self.mp)

        # ------------------------------------------------------------------
        # Optional internal force regulation (scaled by eta2)
        # ------------------------------------------------------------------
        if self.p.enable_internal_force:
            a0_vec = torch.full(
                (F_des.shape[0],),
                float(self.p.cocon_a0),
                device=F_des.device,
                dtype=F_des.dtype,
            )
            af = active_force_from_activation(a0_vec, lenvel, self.env.muscle)
            F_bias = Fmax_vec * (af + flpe)

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
                iters=max(4, int(self.p.bisect_iters) - 4),
            )

        # ------------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------------
        kin_diag = pack_diag(
            sminJ=sminJ,
            lamJ=lamJ,
            alpha_J=None,
            k_manip=k_manip_used,
        )
        diag = merge_diag(
            kin_diag,
            dyn_diag,
            mus_diag,
            pack_diag(lam_os=lam_os, eta=eta, eta2=eta2),
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
            "xref_tuple": (x_d, xd_d, xdd_d),
            "eta": eta2,
            "diag": diag,
        }

    # ----------------------------------------------------------
    # New API: batched reset / compute / dynamics
    # ----------------------------------------------------------
    def reset(self, q0: Tensor) -> None:
        """
        Set initial joint reference qref (batched).

        q0:
            - (n,)   for a single sample
            - (B,n)  for batched joints

        Stored as (B,n) internally.
        """
        q0 = q0.to(dtype=torch.get_default_dtype())
        if q0.ndim == 1:
            self.qref = q0.unsqueeze(0).clone()  # (1,n)
        elif q0.ndim == 2:
            self.qref = q0.clone()               # (B,n)
        else:
            raise ValueError(
                f"PDIFController.reset: q0 must be (n,) or (B,n), got {tuple(q0.shape)}"
            )

    def compute(self, x_d: Tensor, xd_d: Tensor, xdd_d: Tensor) -> Dict[str, Any]:
        """
        Batched compute.

        Inputs (batched):
          x_d, xd_d, xdd_d :
              - shape (2,)   -> treated as (1,2)
              - shape (B,2)

        Returns batched tensors:
          tau_des : (B,n)
          act     : (B,M)
          etc.
        """
        # ------------------------------------------------------------------
        # Current state (BATCHED)
        # ------------------------------------------------------------------
        joint = self.env.states["joint"]       # (B, 2*dof)
        cart = self.env.states["cartesian"]    # (B, 4) for planar

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
        # Batched qref
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
        # [1a] adaptive DLS on J; [1b] kinematic scaling (BATCHED)
        # ------------------------------------------------------------------
        J_pinv_dls, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)
        xd_d, xdd_d, alpha_J = scale_task_by_J(xd_d, xdd_d, sminJ, self.kp)

        # q̇_des = J⁺_DLS ẋ_d  -> (B,n)
        xd_d_col = xd_d.unsqueeze(-1)              # (B,2,1)
        qd_des = (J_pinv_dls @ xd_d_col).squeeze(-1)  # (B,n)

        # Update qref (B,n)
        qref = qref + qd_des * float(self.arm.dt)
        self.qref = qref

        # ------------------------------------------------------------------
        # Dynamics: batched M, h
        # ------------------------------------------------------------------
        M, h = self._compute_dynamics(q, qd)      # M: (B,n,n), h: (B,n)
        Minv = torch.linalg.inv(M)                # (B,n,n)

        # S = J M⁻¹ Jᵀ  (B,2,2)
        Jt = J_xy.transpose(-1, -2)                               # (B,n,2)
        MinvJt = torch.einsum("bij,bjk->bik", Minv, Jt)           # (B,n,2)
        S = torch.einsum("bij,bjk->bik", J_xy, MinvJt)            # (B,2,2)

        # ------------------------------------------------------------------
        # [2a]/[2b]/[2c] op-space guard + gate (BATCHED)
        # ------------------------------------------------------------------
        Lambda, lam_os, eta, eta2, xd_d, xdd_d, dyn_diag = op_space_guard_and_gate(
            S, xd_d, xdd_d, self.dp
        )

        # Normalize shapes: everything batched
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
        # Operational-space control law  (BATCHED)
        # ------------------------------------------------------------------
        Minv_h = torch.einsum("bij,bj->bi", Minv, h)             # (B,n)
        JMinvh = torch.einsum("bij,bj->bi", J_xy, Minv_h)        # (B,2)
        Jdot_qd = torch.einsum("bij,bj->bi", Jdot_xy, qd)       # (B,2)
        tmp = JMinvh - Jdot_qd                                  # (B,2)

        mu = torch.einsum("bij,bj->bi", Lambda, tmp)            # (B,2)

        # e_x, ė_x (B,2)
        e_x = x_d - x
        ed_x = xd_d - xd

        # gains Kp_x, Kff_x (2,) -> broadcasted
        Kp_vec = _to_tensor_like(self.p.Kp_x, x[0])
        Kff_vec = _to_tensor_like(self.p.Kff_x, x[0])
        Kp_mat = torch.diag(Kp_vec)   # (2,2)
        Kff_mat = torch.diag(Kff_vec) # (2,2)

        Lam_sqrt = matrix_sqrt_spd(Lambda)      # (B,2,2)
        Lam_isqrt = matrix_isqrt_spd(Lambda)    # (B,2,2)

        tmpK = torch.einsum("bij,jk->bik", Lam_isqrt, Kp_mat)    # (B,2,2)
        tmpK = torch.einsum("bij,bjk->bik", tmpK, Lam_isqrt)     # (B,2,2)
        sqrt_tmpK = matrix_sqrt_spd(tmpK)                        # (B,2,2)

        Kv_mat = 2.0 * torch.einsum("bij,bjk->bik", Lam_sqrt, sqrt_tmpK)
        Kv_mat = torch.einsum("bij,bjk->bik", Kv_mat, Lam_sqrt)  # (B,2,2)

        term_ff = torch.einsum("ij,bj->bi", Kff_mat, xdd_d)      # (B,2)
        term_d  = torch.einsum("bij,bj->bi", Kv_mat, ed_x)       # (B,2)
        term_p  = torch.einsum("ij,bj->bi", Kp_mat, e_x)         # (B,2)

        xdd_r = term_ff + term_d + term_p     # (B,2)
        F_task = torch.einsum("bij,bj->bi", Lambda, xdd_r) + mu  # (B,2)
        tau_task = torch.einsum(
            "bij,bj->bi", J_xy.transpose(-1, -2), F_task
        )  # (B,n)

        # ------------------------------------------------------------------
        # Nullspace policy (joint PD + manipulability gradient)  (BATCHED)
        # ------------------------------------------------------------------
        Jt = J_xy.transpose(-1, -2)                            # (B,n,2)
        Jt_L = torch.einsum("bij,bjk->bik", Jt, Lambda)        # (B,n,2)
        J_dyn_pinv = torch.einsum("bij,bjk->bik", Minv, Jt_L)  # (B,n,2)

        I_n = torch.eye(n, device=q.device, dtype=q.dtype)
        N = I_n.unsqueeze(0) - torch.einsum("bij,bjk->bik", J_dyn_pinv, J_xy)  # (B,n,n)

        Kp_q = _to_tensor_like(self.p.Kp_q, q[0])  # (n,)
        Kd_q = _to_tensor_like(self.p.Kd_q, q[0])  # (n,)

        qdd_post = Kp_q * (qref - q) - Kd_q * qd   # (B,n)

        qdd_post, k_manip_used = add_nullspace_manip(
            qdd_post, self.env, q, qd, J_xy, self.kp, eta
        )

        tau_post = torch.einsum("bij,bj->bi", M, qdd_post) + h   # (B,n)

        eta2_col = eta2.view(B, 1)  # (B,1)
        tau_des = tau_task - eta2_col * torch.einsum(
            "bij,bj->bi", N.transpose(-1, -2), tau_post
        )  # (B,n)

        # ------------------------------------------------------------------
        # Muscle geometry & forces  (BATCHED)
        # ------------------------------------------------------------------
        geom = self.env.states["geometry"]  # (B, 2 + n, M)
        if geom.ndim != 3:
            raise ValueError(
                f"env.states['geometry'] must be (B, 2 + dof, M), got {tuple(geom.shape)}"
            )

        lenvel = geom[:, :2, :]                          # (B,2,M)
        R = geom[:, 2 : 2 + n, :]                        # (B,n,M)
        M_muscles = int(R.shape[-1])

        Fmax_vec = get_Fmax_vec(
            self.env,
            M_muscles,
        ).to(device=q.device, dtype=q.dtype)             # (M,)

        names = list(getattr(self.env.muscle, "state_name", []))
        if "force-length PE" in names:
            idx_flpe = names.index("force-length PE")
            flpe = self.env.states["muscle"][:, idx_flpe, :]    # (B,M)
        else:
            flpe = torch.zeros(
                (B, M_muscles), device=q.device, dtype=q.dtype
            )

        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta, self.mp)  # (B,M)

        # ------------------------------------------------------------------
        # Optional internal force regulation (scaled by eta2)  (BATCHED)
        # ------------------------------------------------------------------
        if self.p.enable_internal_force:
            a0_vec = torch.full(
                (B, M_muscles),
                float(self.p.cocon_a0),
                device=F_des.device,
                dtype=F_des.dtype,
            )
            af = active_force_from_activation(a0_vec, lenvel, self.env.muscle)  # (B,M)
            F_bias = Fmax_vec * (af + flpe)  # (B,M)

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
        # Activation back-projection & saturation repair  (BATCHED)
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
        # Diagnostics (BATCHED)
        # ------------------------------------------------------------------
        kin_diag = pack_diag(
            sminJ=sminJ,
            lamJ=lamJ,
            alpha_J=alpha_J,
            k_manip=k_manip_used,
        )
        diag = merge_diag(
            kin_diag,
            dyn_diag,
            mus_diag,
            pack_diag(lam_os=lam_os, eta=eta, eta2=eta2),
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
            "diag": diag,
        }

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
            # identity inertia per batch
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



# ---------------------------------------------------------------------------
# Torch smoke test: RigidTendonArm26 + Hill muscle + PD/IF + min-jerk reach
# ---------------------------------------------------------------------------
def _smoke_test_arm26_hill_torch():
    """
    Torch-only smoke test (B=1):

    - RigidTendonArm26 (effector_torch) + RigidTendonHillMuscle
    - Environment (environment_torch)
    - PD/IF controller (this module, batched compute())
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

    print("\n[PD/IF Torch] smoke test (RigidTendonArm26 + Hill) starting ...")
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
        name="RandomReachEnvTorch",
    )

    # ---------- initial joint state (B=1) ----------
    B = 2  # batch size
    
    q0 = _torch.deg2rad(
        _torch.tensor(pc.q0_deg, dtype=_torch.get_default_dtype(), device=device)
    )  # (2,)
    qd0 = _torch.tensor(pc.qd0, dtype=_torch.get_default_dtype(), device=device)  # (2,)
    #joint0 = _torch.cat([q0, qd0], dim=-1).view(1, -1)  # (1, 4)  -> B=1
    joint_single = _torch.cat([q0, qd0], dim=-1)        # (2*dof,) = (4,)
    joint0 = joint_single.unsqueeze(0).expand(B, -1).clone()  # (B, 2*dof) = (4,4)

    env.reset(
        options={
            "joint_state": joint0,   # now batched
            "deterministic": True,
        }
    )

    # ---------- simple min-jerk trajectory (center → shifted target) ----------
    # center = current fingertip position
    fingertip0 = env.states["fingertip"][0]  # (4,) or similar
    center = fingertip0[:2]                  # (2,)

    # target: move +10 cm in x (same pattern as NumPy demo)
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

    # ---------- PD/IF controller parameters ----------
    p = PDIFParams(
        Kp_x=gains.Kp_x,
        Kff_x=gains.Kff_x,
        Kp_q=gains.Kp_q,
        Kd_q=gains.Kd_q,
        eps=num.eps,
        lam_os_smin_target=num.lam_os_smin_target,
        lam_os_max=num.lam_os_max,
        sigma_thresh=num.sigma_thresh,
        gate_pow=num.gate_pow,
        enable_internal_force=toggles.enable_internal_force,
        enable_inertia_comp=toggles.enable_inertia_comp,
        enable_gravity_comp=toggles.enable_gravity_comp,
        enable_velocity_comp=toggles.enable_velocity_comp,
        enable_joint_damping=toggles.enable_joint_damping,
        cocon_a0=ifc.cocon_a0,
        bisect_iters=ifc.bisect_iters,
        linesearch_eps=num.linesearch_eps,
        linesearch_safety=num.linesearch_safety,
    )

    ctrl = PDIFController(env, arm, p)

    # Explicitly set qref with batched reset (optional, compute() would init it anyway)
    ctrl.reset(q0.view(1, -1))  # (1,2) -> B=1, n=2

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

    print("[PD/IF Torch] smoke test (RigidTendonArm26 + Hill) complete ✓")

# Optional: hook into main
if __name__ == "__main__":
    _smoke_test_arm26_hill_torch()
