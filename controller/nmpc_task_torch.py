# controller/nmpc_task_torch.py
# -*- coding: utf-8 -*-
"""
Pure-Torch Nonlinear MPC (task-space, full preview on the discrete op-space model).

Torch rewrite of controller/nmpc_task.py:
- No NumPy in the hot path.
- HTM kinematics/dynamics:
    * lib.kinematics.HTM_kinematics_torch.{geometricJacobian, geometricJacobianDerivative}
    * lib.dynamics.DynamicsHTM_torch.{inertiaMatrixCOM, centrifugalCoriolisCOM, gravitationalCOM}
- Torch guards:
    * utils.kinematics_guard_torch.{KinGuardParams, adaptive_dls_pinv, scale_task_by_J}
    * utils.dynamics_guard_torch.{DynGuardParams, op_space_guard_and_gate}
- Torch muscle stack:
    * utils.muscle_guard_torch.MuscleGuardParams, solve_muscle_forces
    * muscles.muscle_tools_torch.{get_Fmax_vec, force_to_activation_bisect,
                                  active_force_from_activation,
                                  saturation_repair_tau,
                                  apply_internal_force_regulation}

Conventions kept identical to your NumPy version:
- Tracking error convention in task space is textbook:
    e_x = x_d - x,  e_v = xdot_d - xdot
- Regularised op-space metric S = J M^{-1} J^T, guard via op_space_guard_and_gate
- Singularity regularisation lam_sing near small σ_min(J)
- Force gating with alpha_J (same alpha that scales (xd_d, xdd_d))
- Nullspace posture term N = I - J^T (JJ^T)^{-1} J
- Muscle geometry: R = ∂ℓ/∂q,   moment_arms r = -R,   τ = R^T F
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

# --- HTM kinematics/dynamics (Torch) ---
import lib.kinematics.HTM_kinematics_torch as _kin
import lib.dynamics.DynamicsHTM_torch   as _dyn

# --- Guards (Torch) ---
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

# --- Muscle tools (Torch) ---
from muscles.muscle_tools_torch import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
    apply_internal_force_regulation,
)

# ---------------------------------------------------------------------------
# Small helpers (device/dtype-safe)
# ---------------------------------------------------------------------------

def _to_tensor_like(x: Any, like: Tensor) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(x, device=like.device, dtype=like.dtype)

def _eye(n: int, like: Tensor) -> Tensor:
    return torch.eye(n, device=like.device, dtype=like.dtype)

def _zeros(shape, like: Tensor) -> Tensor:
    return torch.zeros(*shape, device=like.device, dtype=like.dtype)

def _zeros_like(x: Tensor) -> Tensor:
    return torch.zeros_like(x)

def _clip(x: Tensor, lo: float, hi: float) -> Tensor:
    return torch.clamp(x, min=lo, max=hi)

def _vec2(x: Any, like: Tensor, default=(0.0, 0.0)) -> Tensor:
    """
    Return (2,) tensor. Accepts None, scalar, (2,), (N,2) -> last row, etc.
    """
    if x is None:
        return torch.tensor(default, device=like.device, dtype=like.dtype)
    x = _to_tensor_like(x, like).reshape(-1)
    if x.numel() == 1:
        return torch.stack([x[0], torch.zeros((), device=like.device, dtype=like.dtype)])
    return x[:2]

def _preview_2(N: int, maybe: Any, like: Tensor) -> Tensor:
    """
    Ensure an (N,2) preview (Torch):
      - if maybe is (2,), tile it to (N,2)
      - if maybe is (N,2), keep it
      - if maybe is (B,2) or (B,N,2): take the last batch row
    """
    t = _to_tensor_like(maybe, like)
    if t.ndim == 1:                  # (2,) or (>=2,)
        t2 = t[:2]
        return t2.unsqueeze(0).expand(N, -1)
    if t.ndim == 2:
        if t.shape[1] == 2:          # (N,2)
            if t.shape[0] == N:
                return t
            # if length mismatch: pad/trim
            if t.shape[0] > N:
                return t[-N:]
            pad = t[-1:].expand(N - t.shape[0], -1)
            return torch.cat([t, pad], dim=0)
        # (B,2) -> use last batch row
        row = t[-1, :2]
        return row.unsqueeze(0).expand(N, -1)
    if t.ndim == 3 and t.shape[-1] == 2:   # (B,N,2)
        return t[-1, :, :] if t.shape[1] == N else t[-1, -N:, :]
    # fallback: vectorize to (2,)
    v = t.reshape(-1)[:2]
    return v.unsqueeze(0).expand(N, -1)

def _activation_to_excitation(a_star: Tensor,
                              a_now: Tensor,
                              dt: float,
                              tau_up: float,
                              tau_down: float) -> Tensor:
    """
    Invert first-order activation dynamics a' = (u - a)/tau
    Choose u so a_{k+1} ≈ a_star in one Euler step (separate up/down taus).
    """
    # decide tau based on direction per muscle
    tau_up_t   = torch.as_tensor(tau_up,   device=a_star.device, dtype=a_star.dtype)
    tau_down_t = torch.as_tensor(tau_down, device=a_star.device, dtype=a_star.dtype)
    tau = torch.where(a_star >= a_now, tau_up_t, tau_down_t)
    tau = torch.clip(tau, 1e-6, 10.0)
    dt  = float(max(dt, 1e-6))
    u = a_star + tau * (a_star - a_now) / dt
    return torch.clip(u, 0.0, 1.0)

# ---------------------------------------------------------------------------
# Parameters (Torch) — keep field names identical to NumPy version
# ---------------------------------------------------------------------------

@dataclass
class NMPCParams:
    # horizon & timing
    N: int = 30
    dt_mpc: float | None = None  # if None, use arm.dt

    # stage / terminal weights
    Wx: Any = field(default_factory=lambda: torch.diag(torch.tensor([1500.0, 1500.0])))
    Wv: Any = field(default_factory=lambda: torch.diag(torch.tensor([20.0, 20.0])))
    Wu: Any = field(default_factory=lambda: torch.diag(torch.tensor([2e-3, 2e-3])))
    WN: Any = field(default_factory=lambda: torch.diag(torch.tensor([40e4, 40e4, 80e2, 80e2])))

    # regularization / limits
    lam_reg: float = 5e-4
    lam_du: float  = 2e-3
    Fmax: float = 600.0
    tau_clip: float = 600.0

    # dynamic compensation toggles
    enable_inertia_comp: bool = True
    enable_gravity_comp: bool = True
    enable_velocity_comp: bool = True
    enable_joint_damping: bool = True

    # guard (op-space)
    eps: float = 1e-8
    lam_os_max: float = 40.0
    sigma_thresh: float = 1e-4
    gate_pow: float = 2.0

    # IK posture (optional; 2-link planar)
    ik_use: bool = True
    elbow_pref: str | None = None  # "up" | "down" | None
    L1: float | None = None
    L2: float | None = None

    # muscle / plant output
    send_excitation: bool = False
    tau_up: float = 0.030
    tau_down: float = 0.080
    bisect_iters: int = 18
    min_activation_override: float | None = None

    # passive compensation
    compensate_passive: bool = True

    # Internal force regulation
    internal_force_scale: float = 0.0  # 0..1  (0 disables)
    linesearch_eps: float = 1e-7
    linesearch_safety: float = 0.2

    # Singularity regularisation & avoidance
    sigma_target: float = 0.10
    lam_sing_gain: float = 60.0
    lam_sing_max: float = 80.0
    force_gate_with_alpha: bool = True

    # Nullspace posture
    ns_avoid_enable: bool = True
    ns_q_rest: Any = field(default_factory=lambda: torch.tensor([0.8, 0.8]))
    ns_kp_base: float = 2.0
    ns_kp_boost: float = 40.0
    ns_kp_max: float = 60.0
    ns_sigma_target: float = 0.10


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class NonlinearMPCControllerTorch:
    """
    Task-space MPC with full preview on the discrete op-space double integrator:
        xdd = Λ_reg (F - μ)
        y = [x; xdot]
        y_{k+1} = A y_k + B (F_k - F_ff,k) + c_k
        τ = Jᵀ F
      where F_ff,k = μ + Λ_reg^{-1} xdd_ref,k  and  c_k = [½dt² xdd_ref,k; dt xdd_ref,k]
    """

    def __init__(self, env, arm, params: NMPCParams):
        self.env = env
        self.arm = arm
        self.p: NMPCParams = params

        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=self.p.eps,
            lam_os_max=self.p.lam_os_max,
            gate_pow=self.p.gate_pow,
            sigma_thresh_S=max(self.p.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()
        self.qref: Tensor | None = None  # last IK posture

    def reset(self, q0: Tensor) -> None:
        """
        Reset controller state.
        q0: (n,) or (B,n) tensor
        """
        if q0.ndim == 1:
            self.qref = q0.clone()
        else:
            self.qref = q0[0].clone()  # Use first batch element

    # ---------- dynamics helpers (Torch, batched safe) ----------
    def _dyn_terms(self, q: Tensor, qd: Tensor) -> Tuple[Tensor, Tensor]:
        """
        q, qd: (B,n) or (n,)
        Returns:
          M: (B,n,n)
          h: (B,n)   with h = C qd + g + D qd
        """
        if q.ndim == 1:
            q = q.unsqueeze(0)
            qd = qd.unsqueeze(0)
        B, n = q.shape

        # inertia
        if self.p.enable_inertia_comp:
            M = _dyn.inertiaMatrixCOM(self.env.skeleton._robot)
            if M.ndim == 2:
                M = M.unsqueeze(0).expand(B, -1, -1)
        else:
            like = q
            M = _eye(n, like).unsqueeze(0).expand(B, -1, -1)

        # gravity
        if self.p.enable_gravity_comp:
            g = _dyn.gravitationalCOM(self.env.skeleton._robot, self.env.skeleton._gravity_vec)
            g = g.reshape(-1)
            g = g.unsqueeze(0).expand(B, -1)
        else:
            g = _zeros((B, n), q)

        # coriolis/centrifugal
        if self.p.enable_velocity_comp:
            C_any = _dyn.centrifugalCoriolisCOM(self.env.skeleton._robot)
            if C_any.ndim == 2:
                C_any = C_any.unsqueeze(0).expand(B, -1, -1)
            h_C = torch.einsum("bij,bj->bi", C_any, qd)
        else:
            h_C = _zeros((B, n), q)

        # viscous joint damping
        if self.p.enable_joint_damping:
            D = float(self.arm.damping)
            h_D = D * qd
        else:
            h_D = _zeros_like(qd)

        h = h_C + g + h_D
        return M, h

    # ---- Analytic IK (2-link planar), Torch version (for posture seeding) ----
    def _get_links(self) -> Tuple[float, float]:
        if self.p.L1 is not None and self.p.L2 is not None:
            return float(self.p.L1), float(self.p.L2)
        sk = self.env.skeleton
        for name in ("link_lengths", "L", "links"):
            if hasattr(sk, name):
                arr = torch.as_tensor(getattr(sk, name)).reshape(-1).float()
                if arr.numel() >= 2:
                    return float(arr[0]), float(arr[1])
        for n1, n2 in (("L1", "L2"), ("l1", "l2"), ("upper_len", "fore_len")):
            if hasattr(sk, n1) and hasattr(sk, n2):
                return float(getattr(sk, n1)), float(getattr(sk, n2))
        return 0.30, 0.25

    @staticmethod
    def _angle_wrap(a: Tensor) -> Tensor:
        return (a + torch.pi) % (2 * torch.pi) - torch.pi

    def _ik_analytic(self, x_d: Tensor, q_curr: Tensor) -> Tensor:
        """
        Closed-form planar 2-link IK, returns (n,) Torch tensor (n>=2 used).
        """
        L1, L2 = self._get_links()
        px, py = float(x_d[0]), float(x_d[1])
        r = (px**2 + py**2) ** 0.5
        r = max(min(r, (L1 + L2) - 1e-9), abs(L1 - L2) + 1e-9)
        if r > 0.0:
            s = r / ( (px**2 + py**2) ** 0.5 + 1e-12)
            px, py = px * s, py * s
        c2 = (px*px + py*py - L1*L1 - L2*L2) / (2.0 * L1 * L2)
        c2 = float(max(min(c2, 1.0), -1.0))
        s2 = (max(0.0, 1.0 - c2*c2)) ** 0.5
        th2_up, th2_dn = torch.atan2(torch.tensor(+s2), torch.tensor(c2)), \
                         torch.atan2(torch.tensor(-s2), torch.tensor(c2))
        k1, k2 = L1 + L2 * c2, L2 * s2
        th1_up = torch.atan2(torch.tensor(py), torch.tensor(px)) - torch.atan2(torch.tensor(+k2), torch.tensor(k1))
        th1_dn = torch.atan2(torch.tensor(py), torch.tensor(px)) - torch.atan2(torch.tensor(-k2), torch.tensor(k1))
        up = torch.stack([th1_up, th2_up])
        dn = torch.stack([th1_dn, th2_dn])

        if   self.p.elbow_pref == "up":   qref = up
        elif self.p.elbow_pref == "down": qref = dn
        else:
            choose_up = torch.linalg.norm(self._angle_wrap(up - q_curr)) <= torch.linalg.norm(self._angle_wrap(dn - q_curr))
            qref = up if choose_up else dn

        if hasattr(self.env.skeleton, "joint_limits"):
            qmin, qmax = self.env.skeleton.joint_limits
            qmin = _to_tensor_like(qmin, qref)
            qmax = _to_tensor_like(qmax, qref)
            qref = torch.minimum(torch.maximum(qref, qmin), qmax)
        return qref

    # ---------- MPC (full preview for ONE batch sample) ----------
    def _solve_horizon_one_(self,
                           y0: Tensor,              # (4,)
                           Xref: Tensor,            # (N,2)
                           Xdref: Tensor,           # (N,2)
                           Xddref: Tensor,          # (N,2)
                           J_xy: Tensor,            # (2,n)
                           Jdot_xy: Tensor,         # (2,n)
                           M: Tensor,               # (n,n)
                           h: Tensor,               # (n,)
                           dt: float,
                           lam_extra: float,
                           like: Tensor) -> Tuple[Tensor, Tensor, Tensor, float]:
        """
        Build time-invariant preview LTI and solve the stacked QP by normal equations.
        Returns F0 (2,), Lambda_reg (2,2), mu_task (2,), lam_eff (float)
        """
        # S = J M^{-1} J^T ------------ op-space metric
        Minv = torch.linalg.inv(M)
        
        # FIX: Use proper transpose for matrix multiplication
        # J_xy: (2, n), J_xy.T: (n, 2) - use transpose() instead of .T
        J_xy_T = J_xy.transpose(0, 1)  # (n, 2)
        S_os = J_xy @ (Minv @ J_xy_T)  # (2,2)

        # Guard & Λ_reg
        # op_space_guard_and_gate returns:
        #   Lambda, lam_os, eta, eta2, xd_d, xdd_d, dyn_diag
        Lambda, lam_os, *_rest = op_space_guard_and_gate(S_os, Xdref[0], Xdref[0], self.dp)
        lam_os = float(lam_os)  # scalar

        # Add singularity regularisation
        lam_eff = lam_os + float(lam_extra)
        S_reg = S_os + lam_eff * _eye(2, like)      # (2,2)
        Lambda_reg = torch.linalg.inv(S_reg)        # (2,2)

        # affine term μ = Jdot qd + J M^{-1} h
        mu_task = Jdot_xy @ y0[2:] + J_xy @ (Minv @ h)   # (2,)

        # Discrete double integrator in task space: y=[x;xd]
        # y_{k+1} = A y_k + B (F_k - F_ff,k) + c_k
        A = torch.block_diag(_eye(2, like), _zeros((2, 2), like))
        A[:2, 2:] = dt * _eye(2, like)     # position integrate velocity
        B = torch.zeros((4, 2), device=like.device, dtype=like.dtype)
        B[:2, :] = 0.5 * dt * dt * Lambda_reg
        B[2:, :] = dt * Lambda_reg

        # Stacks
        N = Xref.shape[0]
        nY = 4 * N
        nU = 2 * N
        T = _zeros((nY, nU), like)

        # F_ff,k = μ + Λ_reg^{-1} xdd_ref,k  and  c_k = [½dt² xdd_ref,k; dt xdd_ref,k]
        Fff_stack = _zeros((nU,), like)
        c_list = []
        As = []
        A_pow = A.clone()
        for k in range(N):
            # columns for u_k
            T[4*k:4*k+4, 2*k:2*k+2] = B
            As.append(A_pow.clone())
            # F_ff and c_k
            Fff_k = mu_task + torch.linalg.solve(Lambda_reg, Xddref[k])
            Fff_stack[2*k:2*k+2] = Fff_k
            c_k = torch.cat([
                0.5 * dt * dt * Xddref[k],
                dt * Xddref[k],
            ], dim=0)
            c_list.append(c_k)
            A_pow = A @ A_pow

        # d_stack = Σ_i A^{k-1-i} c_i ; Sy0_stack = A^k y0
        d_stack   = _zeros((nY,), like)
        Sy0_stack = _zeros((nY,), like)
        for k in range(N):
            d_k = _zeros((4,), like)
            for i in range(k):
                d_k = d_k + As[k-1-i] @ c_list[i]
            d_stack[4*k:4*k+4]   = d_k
            Sy0_stack[4*k:4*k+4] = As[k] @ y0

        # Reference stack Yref = [x_ref; xd_ref]_{k=0..N-1}
        Qblk = torch.block_diag(_to_tensor_like(self.p.Wx, like),
                                _to_tensor_like(self.p.Wv, like))
        Q = torch.kron(_eye(N, like), Qblk)
        # fill reference
        Yref = _zeros((nY,), like)
        for k in range(N):
            Yref[4*k:4*k+2]   = Xref[k]
            Yref[4*k+2:4*k+4] = Xdref[k]
        # terminal
        Q[-4:, -4:] = Q[-4:, -4:] + _to_tensor_like(self.p.WN, like)

        R = torch.kron(_eye(N, like), _to_tensor_like(self.p.Wu, like))

        # Δu penalty
        H = T.T @ Q @ T + R + self.p.lam_reg * _eye(nU, like)
        if self.p.lam_du > 0.0:
            Dm = _eye(2*N, like) - torch.roll(_eye(2*N, like), shifts=2, dims=1)
            Dm[:2, :] = 0.0
            H = H + self.p.lam_du * (Dm.T @ Dm)

        b = T.T @ Q @ (Yref - Sy0_stack - d_stack)

        # Solve H u = b (H is SPD-ish by design)
        ustack = torch.linalg.solve(H, b)

        # Back to forces and clip first control
        Fstack = ustack + Fff_stack
        F0 = _clip(Fstack[:2], -self.p.Fmax, self.p.Fmax)
        return F0, Lambda_reg, mu_task, lam_eff


    # ---------- MPC (full preview for ONE batch sample) ----------
    def _solve_horizon_one(
        self,
        y0: Tensor,              # (4,)
        Xref: Tensor,            # (N,2)
        Xdref: Tensor,           # (N,2)
        Xddref: Tensor,          # (N,2)
        J_xy: Tensor,            # (2,n)
        Jdot_xy: Tensor,         # (2,n)
        M: Tensor,               # (n,n)
        h: Tensor,               # (n,)
        dt: float,
        lam_extra: float,
        like: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, float]:
        """
        Torch port of NumPy _solve_horizon (single sample, no batch dim).
        Returns:
          F0         : (2,)   first task-space force
          Lambda_reg : (2,2)
          mu_task    : (2,)
          lam_eff    : float
        """
        N = self.p.N
        n = M.shape[0]

        # --- Op-space metric S_os = J M^{-1} J^T (Λ^{-1}) ---
        Minv = torch.linalg.inv(M)                         # (n,n)
        S_os = J_xy @ (Minv @ J_xy.transpose(0, 1))        # (2,2)

        # op-space guard: only lam_os is used here (same as NumPy)
        _, lam_os, _, _, _, _, _ = op_space_guard_and_gate(
            S_os, Xdref[0], Xdref[0], self.dp
        )
        lam_eff = float(lam_os) + float(lam_extra)

        # regularised metric S_reg = Λ_reg^{-1}
        I2 = _eye(2, like)
        S_reg      = S_os + lam_eff * I2                   # (2,2)
        Lambda_reg = torch.linalg.inv(S_reg)               # (2,2)

        # mu_task = Λ_reg [ J M^{-1} h - Jdot qdot ]
        term = J_xy @ (Minv @ h) - Jdot_xy @ y0[2:]        # (2,)
        mu_task = Lambda_reg @ term                        # (2,)

        # --- Discrete double integrator model ---
        # y = [x; xd], y_{k+1} = A y_k + B (F_k - F_ff,k) + c_k

        # A = [[I, dt I],
        #      [0,    I ]]
        A = torch.block_diag(I2, I2)                       # (4,4)
        A[:2, 2:] = dt * I2

        # B = [[0.5 dt^2 Λ_reg],
        #      [      dt Λ_reg]]
        B = _zeros((4, 2), like)
        B[:2, :] = 0.5 * (dt ** 2) * Lambda_reg
        B[2:, :] = dt * Lambda_reg

        # stage-wise c_k and F_ff,k
        c_list   = []
        Fff_list = []
        for k in range(N):
            xdd_k = Xddref[k]                              # (2,)
            c_k = torch.cat(
                [0.5 * (dt ** 2) * xdd_k, dt * xdd_k], dim=0
            )                                              # (4,)
            c_list.append(c_k)

            # F_ff,k = μ + S_reg xdd_ref,k  (S_reg = Λ_reg^{-1})
            Fff_k = mu_task + S_reg @ xdd_k                # (2,)
            Fff_list.append(Fff_k)

        Fff_stack = torch.cat(Fff_list, dim=0)             # (2N,)

        # --- Reachability matrix T (lower-triangular Toeplitz) ---
        nY = 4 * N
        nU = 2 * N

        I4 = _eye(4, like)
        As = [I4]
        for _ in range(1, N + 1):
            As.append(A @ As[-1])                          # As[k] = A^k

        T = _zeros((nY, nU), like)
        for k in range(N):
            for j in range(k):
                # effect of u_j on y_k: A^{k-1-j} B
                T[4 * k : 4 * k + 4, 2 * j : 2 * j + 2] = As[k - 1 - j] @ B

        # --- Affine stack: d_stack, Sy0_stack ---
        d_stack   = _zeros((nY,), like)
        Sy0_stack = _zeros((nY,), like)
        for k in range(N):
            d_k = _zeros((4,), like)
            for i in range(k):
                d_k = d_k + As[k - 1 - i] @ c_list[i]
            idx = 4 * k
            d_stack[idx : idx + 4]   = d_k
            Sy0_stack[idx : idx + 4] = As[k] @ y0

        # --- Reference stack Yref = [x_ref; xd_ref]_{k=0..N-1} ---
        Yref = _zeros((nY,), like)
        for k in range(N):
            idx = 4 * k
            Yref[idx : idx + 2]     = Xref[k]
            Yref[idx + 2 : idx + 4] = Xdref[k]

        # --- Quadratic cost: H u = b (same as NumPy) ---
        Wx_t = _to_tensor_like(self.p.Wx, like)
        Wv_t = _to_tensor_like(self.p.Wv, like)
        WN_t = _to_tensor_like(self.p.WN, like)
        Wu_t = _to_tensor_like(self.p.Wu, like)

        Qblk = torch.block_diag(Wx_t, Wv_t)                # (4,4)
        Q = torch.kron(_eye(N, like), Qblk)                # (4N,4N)
        Q[-4:, -4:] = Q[-4:, -4:] + WN_t                   # terminal

        R = torch.kron(_eye(N, like), Wu_t)                # (2N,2N)

        H = T.T @ Q @ T + R + self.p.lam_reg * _eye(nU, like)
        if self.p.lam_du > 0.0:
            # Δu penalty: Dm u ≈ [0; u_1 - u_0; ...]
            Dm = _eye(nU, like) - torch.roll(_eye(nU, like), shifts=2, dims=1)
            Dm[:2, :] = 0.0
            H = H + self.p.lam_du * (Dm.T @ Dm)

        b = T.T @ Q @ (Yref - Sy0_stack - d_stack)

        # Solve normal equations (H is SPD-ish by construction)
        ustack = torch.linalg.solve(H, b)                  # (2N,)

        # back to forces, clip first control
        Fstack = ustack + Fff_stack                        # (2N,)
        F0 = _clip(Fstack[:2], -self.p.Fmax, self.p.Fmax)  # (2,)
        return F0, Lambda_reg, mu_task, lam_eff




    # ---------- public step ----------
    def compute(self, x_d: Any, xd_d: Any, xdd_d: Any) -> Dict[str, Any]:
        """
        Inputs can be (2,), (N,2) previews, or (B,2)/(B,N,2) — we use the last batch row.
        Returns a dict (same keys as your NumPy version).
        """
        # --- Read current state (Torch env uses batched states) ---
        joint = self.env.states["joint"]     # (B, 2*n)
        cart  = self.env.states["cartesian"] # (B, >=4)
        if joint.ndim == 1:
            joint = joint.unsqueeze(0)
        if cart.ndim == 1:
            cart = cart.unsqueeze(0)

        B = joint.shape[0]
        n = self.env.skeleton.dof
        q  = joint[:, :n]
        qd = joint[:, n:]
        x  = cart[:, :2]
        xd = cart[:, 2:4]

        # keep robot up-to-date for kinematics/dynamics calls
        # (set only from first batch row; typical env is B=1)
        self.env.skeleton._set_state(q[0], qd[0])

        # --- Jacobian, scaling by J near singularities ---
        J = _kin.geometricJacobian(self.env.skeleton._robot)  # (6,n) or (B,6,n)
        
        # FIX: Handle different Jacobian shapes properly
        if J.ndim == 3:
            # Batched case: (B, 6, n) - take first batch and first 2 rows
            J_xy = J[0, 0:2, :]  # (2, n)
        elif J.ndim == 2:
            # Non-batched case: (6, n) - take first 2 rows
            J_xy = J[0:2, :]     # (2, n)
        else:
            raise ValueError(f"Unexpected Jacobian shape: {J.shape}")
        
        Jdot = _kin.geometricJacobianDerivative(self.env.skeleton._robot)
        
        # FIX: Handle Jdot shapes similarly
        if Jdot.ndim == 3:
            Jdot_xy = Jdot[0, 0:2, :]  # (2, n)
        elif Jdot.ndim == 2:
            Jdot_xy = Jdot[0:2, :]     # (2, n)
        else:
            raise ValueError(f"Unexpected Jdot shape: {Jdot.shape}")

        # adaptive DLS on J for diagnostics (sminJ, lamJ)
        _Jlike = q[0]
        inv_dls, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)

        # scale xd_d, xdd_d and keep alpha_J for optional force gating
        xd_d_1 = _vec2(xd_d, _Jlike)
        xdd_d_1 = _vec2(xdd_d, _Jlike)
        xd_d_1, xdd_d_1, alpha_J = scale_task_by_J(xd_d_1, xdd_d_1, sminJ, self.kp)

        # IK posture (optional)
        if self.p.ik_use:
            q_curr = q[0]
            self.qref = self._ik_analytic(_vec2(x_d, _Jlike), q_curr)
        else:
            if self.qref is None:
                self.qref = q[0]

        # --- dynamics (Torch) ---
        M, h = self._dyn_terms(q[0], qd[0])       # (1,n,n), (1,n) or (n,n),(n,)
        if M.ndim == 3:
            M = M[0]
        if h.ndim == 2:
            h = h[0]
        y0 = torch.cat([x[0], xd[0]], dim=0)      # (4,)

        # --- lam_sing (extra damping near singularities) ---
        #sminJ_val = float(sminJ)
        #if sminJ_val <= self.p.sigma_target:
         #   ratio = (self.p.ns_sigma_target / max(sminJ_val, 1e-9)) - 1.0  # >=0 near singular
         #   lam_sing = min(max(self.p.lam_sing_gain * (ratio ** 2), 0.0), self.p.lam_sing_max)
        #else:
         #   lam_sing = 0.0

        # --- lam_sing (extra damping near singularities) ---
        sminJ_val = float(sminJ)
        if sminJ_val <= self.p.sigma_target:
            # use sigma_target (same as NumPy) for both threshold and ratio
            ratio = (self.p.sigma_target / max(sminJ_val, 1e-9)) - 1.0  # >= 0 near singular
            lam_sing = min(max(self.p.lam_sing_gain * (ratio ** 2), 0.0), self.p.lam_sing_max,
            )
        else:
            lam_sing = 0.0

        # --- Build preview (N,2) from inputs (last batch row) ---
        N  = self.p.N
        dt = float(self.p.dt_mpc if (self.p.dt_mpc is not None) else self.arm.dt)
        like = q[0]
        Xref   = _preview_2(N, x_d,   like)
        Xdref  = _preview_2(N, xd_d,  like)
        Xddref = _preview_2(N, xdd_d, like)

        # --- Solve MPC (one-shot for last batch row) ---
        F0, _Lambda_reg, _mu_task, lam_eff = self._solve_horizon_one(
            y0, Xref, Xdref, Xddref, J_xy, Jdot_xy, M, h, dt, lam_sing, like
        )

        # optional conservative gating with alpha_J
        if self.p.force_gate_with_alpha:
            F0 = alpha_J * F0

        # base task torque
        tau_task = J_xy.transpose(0, 1) @ F0  # (n,) - use transpose() instead of .T

        # -------- Nullspace singularity avoidance --------
        tau_ns = torch.zeros_like(tau_task)
        if self.p.ns_avoid_enable:
            # manipulability-aware gain
            if sminJ_val <= self.p.ns_sigma_target:
                ratio = (self.p.ns_sigma_target / max(sminJ_val, 1e-9)) - 1.0
                kp_eff = min(self.p.ns_kp_base + self.p.ns_kp_boost * (ratio ** 2), self.p.ns_kp_max)
            else:
                kp_eff = self.p.ns_kp_base

            q_rest = _to_tensor_like(self.p.ns_q_rest, like).reshape(-1)[:n]
            e_post = q_rest - q[0]
            JJt = J_xy @ J_xy.transpose(0, 1)  # Use transpose()
            JJt_inv = torch.linalg.inv(JJt + 1e-9 * _eye(2, like))
            N_null = _eye(n, like) - J_xy.transpose(0, 1) @ JJt_inv @ J_xy  # Use transpose()
            tau_ns = float(kp_eff) * (N_null @ e_post)
        else:
            kp_eff = 0.0

        # total torque and clip
        tau_des = _clip(tau_task + tau_ns, -self.p.tau_clip, self.p.tau_clip)

        # ===================== τ → activation/excitation path =====================
        # Geometry/state
        geom   = self.env.states["geometry"]                 # (B, 2 + dof, M)
        lenvel = geom[:, :2, :]                              # (B, 2, M)
        Rm     = geom[:, 2:2 + n, :][0]                      # (n, M)
        M_musc = Rm.shape[1]
        Fmax_v = get_Fmax_vec(self.env, M_musc,
                              device=Rm.device, dtype=Rm.dtype)  # (M,)

        # Passive term
        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe  = self.env.states["muscle"][0, idx_flpe, :]    # (M,)
        F_pass = Fmax_v * flpe
        tau_passive = -(Rm @ F_pass) if self.p.compensate_passive else torch.zeros(n, device=Rm.device, dtype=Rm.dtype)

        # Active torque to allocate
        tau_need = tau_des - tau_passive

        # --- Branch A: INTERNAL FORCE REGULATION ---
        if float(self.p.internal_force_scale) > 0.0:
            # allocate with guard
            F_des, mus_diag = solve_muscle_forces(
                tau_need, Rm, Fmax_v, 1.0,self.mp)
            # internal force regulation around present bias
            a0_vec = torch.clamp(self.env.states["muscle"][0, names.index("activation"), :], 0.0, 1.0) \
                     if "activation" in names else torch.zeros_like(flpe)
            af_now = active_force_from_activation(a0_vec, lenvel, self.env.muscle)
            F_bias = Fmax_v * (af_now + flpe)
            F_des = apply_internal_force_regulation(
                -Rm, F_des, F_bias,
                eps=self.p.eps,
                linesearch_eps=self.p.linesearch_eps,
                linesearch_safety=self.p.linesearch_safety,
                scale=float(torch.clamp(torch.tensor(self.p.internal_force_scale), 0.0, 1.0)),
            )

            a_des = force_to_activation_bisect(
                F_des, lenvel, self.env.muscle, flpe, Fmax_v, iters=self.p.bisect_iters
            )

            # min activation floor
            a_min = float(self.p.min_activation_override) \
                if self.p.min_activation_override is not None \
                else float(getattr(self.env.muscle, "min_activation", 0.0))
            a_des = torch.clamp(a_des, a_min, 1.0)

            # optionally output excitation instead of activation
            if self.p.send_excitation:
                if "activation" in names:
                    a_now = self.env.states["muscle"][0, names.index("activation"), :]
                else:
                    a_now = a_des
                payload_act = _activation_to_excitation(a_des, a_now, dt, self.p.tau_up, self.p.tau_down)
            else:
                payload_act = a_des

            return {
                "tau_des": tau_des, "R": Rm, "Fmax": Fmax_v, "F_des": F_des, "act": payload_act,
                "q": q[0], "qd": qd[0], "x": x[0], "xd": xd[0],
                "xref_tuple": (Xref[-1], Xdref[-1], Xddref[-1]),
                "eta": 1.0,
                "diag": {
                    "sminJ": float(sminJ), "lamJ": float(lamJ),
                    "lam_sing": float(lam_sing), "lam_eff": float(lam_eff),
                    "alpha_J": float(alpha_J),
                    "kp_ns": float(kp_eff),
                    "comp_passive": bool(self.p.compensate_passive),
                    "send_excitation": bool(self.p.send_excitation),
                },
            }

        # --- Branch B: NO internal-force regulation (classic allocation) ---
        F_des, mus_diag = solve_muscle_forces(
            tau_need, Rm, Fmax_v, 1.0, self.mp)
        
        a_des = force_to_activation_bisect(
            F_des, lenvel, self.env.muscle, flpe, Fmax_v, iters=self.p.bisect_iters
        )

        # min activation floor
        a_min = float(self.p.min_activation_override) \
            if self.p.min_activation_override is not None \
            else float(getattr(self.env.muscle, "min_activation", 0.0))
        a_des = torch.clamp(a_des, a_min, 1.0)

        # optionally excitation
        if self.p.send_excitation:
            if "activation" in names:
                a_now = self.env.states["muscle"][0, names.index("activation"), :]
            else:
                a_now = a_des
            act_field = _activation_to_excitation(a_des, a_now, dt, self.p.tau_up, self.p.tau_down)
        else:
            act_field = a_des

        # predict & repair for safety (optional)
        af_now = active_force_from_activation(a_des, lenvel, self.env.muscle)
        F_pred = Fmax_v * af_now
        F_corr = saturation_repair_tau(-Rm, F_pred, a_des, a_min, 1.0, Fmax_v, tau_des=tau_des)
        if torch.any(torch.abs(F_corr - F_pred) > 1e-9):
            a_des = force_to_activation_bisect(F_corr, lenvel, self.env.muscle, flpe, Fmax_v,
                                               iters=max(4, self.p.bisect_iters - 4))
            if self.p.send_excitation:
                if "activation" in names:
                    a_now = self.env.states["muscle"][0, names.index("activation"), :]
                else:
                    a_now = a_des
                act_field = _activation_to_excitation(a_des, a_now, dt, self.p.tau_up, self.p.tau_down)
            else:
                act_field = a_des

        return {
            "tau_des": tau_des, "R": Rm, "Fmax": Fmax_v, "F_des": F_des, "act": act_field,
            "q": q[0], "qd": qd[0], "x": x[0], "xd": xd[0],
            "xref_tuple": (Xref[-1], Xdref[-1], Xddref[-1]),
            "eta": 1.0,
            "diag": {
                "sminJ": float(sminJ), "lamJ": float(lamJ),
                "lam_sing": float(lam_sing), "lam_eff": float(lam_eff),
                "alpha_J": float(alpha_J),
                "kp_ns": float(kp_eff),
                "comp_passive": bool(self.p.compensate_passive),
                "send_excitation": bool(self.p.send_excitation),
            },
        }


# ---------------------------------------------------------------------------
# Torch smoke test: Nonlinear MPC
# ---------------------------------------------------------------------------

def _smoke_test_nmpc():
    """
    Smoke test for the Nonlinear MPC controller.
    Uses the same setup as your PD/IF test but with NMPC.
    """
    import torch as _torch
    from model_lib.environment_torch import Environment as EnvTorch
    from model_lib.muscles_torch import RigidTendonHillMuscle
    from model_lib.effector_torch import RigidTendonArm26
    from trajectory.minjerk_torch import MinJerkLinearTrajectoryTorch, MinJerkParams
    from sim.simulator_torch import TargetReachSimulatorTorch
    from config import (
        PlantConfig,
        ControlToggles,
        ControlGains,
        Numerics,
        InternalForceConfig,
        TrajectoryConfig,
    )

    print("\n[NMPC Torch] smoke test (RigidTendonArm26 + Hill) starting ...")
    _torch.set_default_dtype(_torch.float64)
    _torch.set_printoptions(precision=6, sci_mode=False)

    # choose device
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
        name="NMPC_ReachEnvTorch",
    )

    # ---------- initial joint state (B=1) ----------
    B = 1  # NMPC typically works with single batch for now
    
    q0 = _torch.deg2rad(
        _torch.tensor(pc.q0_deg, dtype=_torch.get_default_dtype(), device=device)
    )  # (2,)
    qd0 = _torch.tensor(pc.qd0, dtype=_torch.get_default_dtype(), device=device)  # (2,)
    joint0 = _torch.cat([q0, qd0]).unsqueeze(0)  # (1, 4)

    env.reset(
        options={
            "joint_state": joint0,
            "deterministic": True,
        }
    )

    # ---------- simple min-jerk trajectory ----------
    fingertip0 = env.states["fingertip"][0, :2]  # (2,)
    center = fingertip0
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

    # ---------- NMPC controller parameters ----------
    nmpc_params = NMPCParams(
        N=20,  # Horizon
        dt_mpc=arm.dt,
        
        # Weights
        Wx=torch.diag(torch.tensor([1500.0, 1500.0])),
        Wv=torch.diag(torch.tensor([20.0, 20.0])),
        Wu=torch.diag(torch.tensor([2e-3, 2e-3])),
        WN=torch.diag(torch.tensor([40e4, 40e4, 80e2, 80e2])),
        
        # Regularization
        lam_reg=5e-4,
        lam_du=2e-3,
        Fmax=600.0,
        tau_clip=600.0,
        
        # Dynamics compensation
        enable_inertia_comp=toggles.enable_inertia_comp,
        enable_gravity_comp=toggles.enable_gravity_comp,
        enable_velocity_comp=toggles.enable_velocity_comp,
        enable_joint_damping=toggles.enable_joint_damping,
        
        # Guards
        eps=num.eps,
        lam_os_max=num.lam_os_max,
        sigma_thresh=num.sigma_thresh,
        gate_pow=num.gate_pow,
        
        # IK
        ik_use=True,
        elbow_pref="up",
        
        # Muscle output
        send_excitation=False,
        tau_up=0.030,
        tau_down=0.080,
        bisect_iters=ifc.bisect_iters,
        
        # Passive compensation
        compensate_passive=True,
        
        # Internal force
        internal_force_scale=0.0,  # Disable for smoke test
        linesearch_eps=num.linesearch_eps,
        linesearch_safety=num.linesearch_safety,
        
        # Singularity handling
        sigma_target=0.10,
        lam_sing_gain=60.0,
        lam_sing_max=80.0,
        force_gate_with_alpha=True,
        
        # Nullspace
        ns_avoid_enable=True,
        ns_q_rest=torch.tensor([0.8, 0.8]),
        ns_kp_base=2.0,
        ns_kp_boost=40.0,
        ns_kp_max=60.0,
        ns_sigma_target=0.10,
    )

    ctrl = NonlinearMPCControllerTorch(env, arm, nmpc_params)

    # ---------- simulate ----------
    steps = int(pc.max_ep_duration / arm.dt)
    
    # Use the simulator if available, otherwise manual loop
    try:
        sim = TargetReachSimulatorTorch(env, arm, ctrl, traj, steps)
        logs = sim.run()

        k, tvec = logs.time(arm.dt)
        x_log = logs.x_log[:k]  # (T, 4) or (T, >=2)

        # ---------- print results ----------
        print("  fingertip (before):", center.detach().cpu().numpy())
        print("  target x_d:        ", target.detach().cpu().numpy())

        for idx in [0, 1, 2, 4, 9, 19, 29]:
            if idx < k:
                x = x_log[idx, :2]
                print(f"    step {idx:3d}: x = {x.detach().cpu().numpy()}")

        x_final = x_log[k - 1, :2]
        err_final = _torch.linalg.norm(x_final - target)
        print("  final fingertip:   ", x_final.detach().cpu().numpy())
        print(f"  final |x - x_d|:    {float(err_final):.6f} m")

    except Exception as e:
        print(f"Simulator failed: {e}")
        print("Running manual simulation loop...")
        
        # Manual simulation loop
        for step in range(min(steps, 50)):  # Just run first 50 steps
            t = step * arm.dt
            
            # Get desired state from trajectory
            t_tensor = _torch.tensor([t], device=traj.device, dtype=traj.dtype)
            x_d, xd_d, xdd_d = traj.sample(t_tensor)
            
            # Compute control
            result = ctrl.compute(x_d, xd_d, xdd_d)
            
            if step % 10 == 0:
                print(f"Step {step}: x = {result['x'].detach().cpu().numpy()}, "
                      f"target = {x_d[0].detach().cpu().numpy()}, "
                      f"avg_act = {result['act'].mean().item():.3f}")

    print("[NMPC Torch] smoke test complete ✓")


# Optional: hook into main
if __name__ == "__main__":
    _smoke_test_nmpc()