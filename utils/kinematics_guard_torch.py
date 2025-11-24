# kinematics_guard_torch.py
# -*- coding: utf-8 -*-
"""
Torch-only kinematic guards: adaptive DLS pinv, task scaling,
and Yoshikawa manipulability gradient nullspace term.

Main pieces
-----------
- KinGuardParams:
    configuration for damping, scaling and manipulability nullspace.

- adaptive_dls_pinv(J_xy, n, P):
    Damped least-squares pseudo-inverse with adaptive damping λ_J based on
    the min singular value of J_xy.

    Torch-only, supports:
        J_xy: (2, n) or (B, 2, n)
        returns:
            J_dls: (n, 2) or (B, n, 2)
            smin:  scalar tensor for unbatched, or (B,) for batched
            lamJ:  scalar tensor for unbatched, or (B,) for batched

- scale_task_by_J(xd_d, xdd_d, sminJ, P):
    Scales desired task-space velocity/acceleration when J becomes ill
    conditioned:
        alpha = min(1, s / (s + k_scale_J))

    Torch-only, handles both scalar and batched sminJ and broadcasts over
    xd_d / xdd_d.

- manip_grad(env, q, qd, eps):
    Gradient of log Yoshikawa manipulability w.r.t. q (Torch-only).

    Torch path:
        1) NEW graph-preserving path:
            * Uses the underlying Robot and
              lib.kinematics.HTM_kinematics_torch.geometricJacobian to compute
              J(q), then w(q) = sqrt(det(J_xy J_xyᵀ)), logw = log w(q).
            * Computes d logw / d q via autograd w.r.t. a q_used that is
              directly connected to the caller's q (no detach/clone).
        2) If that fails (exception or gradient is None), falls back to an
           internal-leaf implementation that still uses Torch and can also
           do finite differences in Torch as a last resort.

    Requirements:
        - env must expose env.effector.skeleton or env.skeleton
        - that skeleton must have a ._robot with attributes:
            * q  : joint configuration (n×1)
            * denavitHartenberg()
          and be compatible with geometricJacobian(...).

- add_nullspace_manip(qdd_post, env, q, qd, J_xy, P, eta):
    Adds a manipulability-gradient-based nullspace term to qdd_post.

    qdd_post: (n,)
    eta     : scalar or 0D tensor (e.g., from energy tank)
    returns: (qdd_out, k_eff) where k_eff is the effective scalar gain
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch
from torch import Tensor

# Geometric Jacobian from Torch HTM stack
try:
    from lib.kinematics.HTM_kinematics_torch import geometricJacobian as _geomJ_HTM
except ImportError:
    from lib.kinematics.HTM import geometricJacobian as _geomJ_HTM  # noqa: F401

_DEBUG_MANIP = False


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

@dataclass
class KinGuardParams:
    # Damped least-squares parameters
    lam_min: float = 1e-6
    lam_max: float = 1e-1
    k_sig: float = 30.0
    sigma_thresh_J: float = 5e-2

    # Task scaling (near singularity)
    k_scale_J: float = 5e-2

    # Manipulability nullspace term
    k_manip: float = 0.1
    grad_eps: float = 1e-5
    enable_manip_nullspace: bool = True


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _is_torch(x: Any) -> bool:
    return isinstance(x, Tensor)


def _get_skeleton_from_env(env: Any):
    """
    Retrieve a Skeleton-like object from env (Torch env).

    Priority:
        1) env.effector.skeleton
        2) env.skeleton
    """
    skel = getattr(env, "effector", None)
    if skel is not None and hasattr(skel, "skeleton"):
        skel = skel.skeleton
        if _DEBUG_MANIP:
            print("[kinematics_guard_torch] Using env.effector.skeleton")
        return skel

    skel = getattr(env, "skeleton", None)
    if skel is not None:
        if _DEBUG_MANIP:
            print("[kinematics_guard_torch] Using env.skeleton")
        return skel

    raise RuntimeError(
        "manip_grad: could not find a 'skeleton' inside env. "
        "Expected env.effector.skeleton or env.skeleton."
    )


# --------------------------------------------------------------------------- #
# Adaptive DLS pseudo-inverse (Torch-only)
# --------------------------------------------------------------------------- #

def adaptive_dls_pinv(J_xy: Tensor, n: int, P: KinGuardParams):
    """
    Adaptive damped least-squares pseudo-inverse of J_xy (Torch-only).

    Parameters
    ----------
    J_xy : Tensor
        Shape (2, n) or (B, 2, n)
    n : int
        Number of joints
    P : KinGuardParams

    Returns
    -------
    J_dls : Tensor
        Shape (n, 2) for unbatched, or (B, n, 2) for batched
    smins : Tensor
        Scalar 0D tensor (unbatched) or (B,) tensor
    lamJs : Tensor
        Scalar 0D tensor (unbatched) or (B,) tensor
    """
    J = torch.as_tensor(J_xy)

    # Unbatched: (2, n)
    if J.ndim == 2:
        s = torch.linalg.svdvals(J)       # (2,)
        smin = s.min()                    # scalar tensor

        arg = P.k_sig * (smin - P.sigma_thresh_J)
        lamJ = P.lam_min + (P.lam_max - P.lam_min) / (1.0 + torch.exp(arg))

        JTJ = J.transpose(-1, -2) @ J     # (n, n)
        eye_n = torch.eye(n, dtype=J.dtype, device=J.device)
        J_dls = torch.linalg.solve(
            JTJ + (lamJ ** 2) * eye_n,
            J.transpose(-1, -2),
        )  # (n, 2)

        return J_dls, smin, lamJ

    # Batched: (B, 2, n)
    if J.ndim != 3 or J.shape[1] != 2:
        raise ValueError(
            f"adaptive_dls_pinv expects J_xy with shape (2,n) or (B,2,n), got {tuple(J.shape)}"
        )

    B = J.shape[0]
    J_dls_list = []
    smins_list = []
    lamJs_list = []

    eye_n = None

    for b in range(B):
        Jb = J[b]  # (2, n)
        s = torch.linalg.svdvals(Jb)
        smin_b = s.min()

        arg_b = P.k_sig * (smin_b - P.sigma_thresh_J)
        lamJ_b = P.lam_min + (P.lam_max - P.lam_min) / (1.0 + torch.exp(arg_b))

        JTJ_b = Jb.transpose(-1, -2) @ Jb  # (n, n)
        if eye_n is None:
            eye_n = torch.eye(n, dtype=Jb.dtype, device=Jb.device)

        J_dls_b = torch.linalg.solve(
            JTJ_b + (lamJ_b ** 2) * eye_n,
            Jb.transpose(-1, -2),
        )  # (n, 2)

        J_dls_list.append(J_dls_b)
        smins_list.append(smin_b)
        lamJs_list.append(lamJ_b)

    J_dls_stack = torch.stack(J_dls_list, dim=0)       # (B, n, 2)
    smins = torch.stack(smins_list, dim=0)             # (B,)
    lamJs = torch.stack(lamJs_list, dim=0)             # (B,)

    return J_dls_stack, smins, lamJs


# --------------------------------------------------------------------------- #
# Task scaling near singularity (Torch-only)
# --------------------------------------------------------------------------- #

def scale_task_by_J(
    xd_d: Tensor,
    xdd_d: Tensor,
    sminJ,
    P: KinGuardParams,
):
    """
    Scale desired task-space velocity/acceleration based on min singular
    value(s) of J_xy (Torch-only).

    alpha = min(1, s / (s + k_scale_J))

    Parameters
    ----------
    xd_d : Tensor
        Desired task-space velocity, shape (..., 2) typically
    xdd_d : Tensor
        Desired task-space acceleration, same shape as xd_d
    sminJ : float or Tensor
        Min singular value(s) of J_xy; scalar or (B,) Tensor.
    P : KinGuardParams

    Returns
    -------
    xd_d_scaled : Tensor
    xdd_d_scaled : Tensor
    alpha : Tensor
        Scalar (0D) or (B,) tensor of scaling factors
    """
    xd_d = torch.as_tensor(xd_d)
    xdd_d = torch.as_tensor(xdd_d)

    # Build s as tensor
    if isinstance(sminJ, Tensor):
        s = sminJ.to(dtype=torch.float64, device=xd_d.device)
    else:
        s = torch.tensor(sminJ, dtype=torch.float64, device=xd_d.device)

    alpha = torch.minimum(
        torch.ones_like(s),
        s / (s + P.k_scale_J),
    )  # scalar or (B,)

    # Broadcast alpha over the last dimension of xd_d/xdd_d
    alpha_exp = alpha
    while alpha_exp.dim() < xd_d.dim():
        alpha_exp = alpha_exp.unsqueeze(-1)

    alpha_exp = alpha_exp.to(dtype=xd_d.dtype, device=xd_d.device)

    xd_d_scaled = alpha_exp * xd_d
    xdd_d_scaled = alpha_exp.to(dtype=xdd_d.dtype, device=xdd_d.device) * xdd_d
    return xd_d_scaled, xdd_d_scaled, alpha


# --------------------------------------------------------------------------- #
# Yoshikawa manipulability measure (Torch-only)
# --------------------------------------------------------------------------- #

def _yoshikawa_w(J_xy: Tensor) -> Tensor:
    """
    Yoshikawa manipulability w(q) = sqrt(det(J_xy J_xy^T)) (Torch-only).

    Accepts either (2, n) or (B, 2, n) Tensor.
    """
    J = torch.as_tensor(J_xy)
    JJt = J @ J.transpose(-1, -2)        # (..., 2, 2)
    det = torch.det(JJt).clamp(min=0.0)  # (...,)

    return torch.sqrt(det)


# --------------------------------------------------------------------------- #
# Torch helper: log w(q) from the underlying Robot via HTM (graph-preserving)
# --------------------------------------------------------------------------- #

def _logw_from_robot_torch(robot, q_vec: Tensor) -> Tensor:
    """
    Core helper for Torch:
    - robot: underlying Robot object (skel._robot)
    - q_vec: 1D tensor (n,) that we want gradients w.r.t.

    This function:
        * Temporarily rebinds robot.q to a view of q_vec (no copy).
        * Calls robot.denavitHartenberg() to refresh DH.
        * Uses lib.kinematics.HTM_*_torch.geometricJacobian (autograd-safe).
        * Restores robot.q afterwards.

    Returns
    -------
    logw : 0D Torch tensor (scalar)
    """
    assert isinstance(q_vec, Tensor), "_logw_from_robot_torch expects a Torch tensor q_vec."

    q_vec = q_vec.view(-1)
    device = q_vec.device
    dtype = q_vec.dtype

    # Save old q and overwrite with a view of q_vec (no copy, no detach)
    old_q = getattr(robot, "q", None)
    robot.q = q_vec.view(-1, 1)

    # Refresh DH and compute geometric Jacobian via HTM (Torch path)
    robot.denavitHartenberg()  # Torch Serial API: no 'symbolic' kwarg
    J = _geomJ_HTM(robot)  # (6, n); Torch

    if not isinstance(J, Tensor):
        J = torch.as_tensor(J, dtype=dtype, device=device)

    J_xy = J[0:2, :]  # (2, n)
    w = _yoshikawa_w(J_xy)  # scalar
    logw = torch.log(torch.clamp(w, min=1e-12))

    # Restore original q to avoid side effects
    robot.q = old_q

    return logw


# --------------------------------------------------------------------------- #
# NEW: compute log Yoshikawa w(q) with a graph-preserving Torch path
# --------------------------------------------------------------------------- #

def _log_yoshikawa_torch(env, skel, q, qd) -> Tuple[Tensor, Tensor]:
    """
    Helper for Torch: compute log w(q) with *no* internal detach/clone
    so that autograd can see a continuous path from the caller's q.

    Returns
    -------
    logw : 0D Torch tensor
    q_used : Torch tensor actually used inside the computation (for grad)
    """
    robot = skel._robot

    # q_used: preserve external graph if q is already a Tensor
    if isinstance(q, Tensor):
        q_used = q
    else:
        base_q = getattr(robot, "q", None)
        base_device = getattr(base_q, "device", torch.device("cpu")) if isinstance(base_q, Tensor) else torch.device("cpu")
        base_dtype = getattr(base_q, "dtype", torch.get_default_dtype()) if isinstance(base_q, Tensor) else torch.get_default_dtype()
        q_used = torch.as_tensor(q, dtype=base_dtype, device=base_device)

    q_used = q_used.view(-1)
    if not q_used.requires_grad:
        q_used.requires_grad_(True)

    # qd is unused here, kept for API symmetry
    logw = _logw_from_robot_torch(robot, q_used)

    if _DEBUG_MANIP:
        print(
            "[manip_grad] [NEW] logw computed. "
            f"q_used.requires_grad={q_used.requires_grad}, "
            f"logw.grad_fn={logw.grad_fn}"
        )

    return logw, q_used


def _manip_grad_torch_internal(skel, q, qd, eps: float) -> Tensor:
    """
    Internal Torch implementation:

    - Wraps q into a fresh leaf q_leaf (detach().clone().requires_grad_()).
    - Uses autograd on logw(q_leaf) via _logw_from_robot_torch.
    - If autograd returns None, falls back to finite differences in Torch.
    """
    robot = skel._robot

    # Make a fresh leaf from q (no external graph)
    if isinstance(q, Tensor):
        q0 = q.detach()
    else:
        base_q = getattr(robot, "q", None)
        base_device = getattr(base_q, "device", torch.device("cpu")) if isinstance(base_q, Tensor) else torch.device("cpu")
        base_dtype = getattr(base_q, "dtype", torch.get_default_dtype()) if isinstance(base_q, Tensor) else torch.get_default_dtype()
        q0 = torch.as_tensor(q, dtype=base_dtype, device=base_device)

    q_leaf = q0.clone().view(-1).requires_grad_(True)
    logw = _logw_from_robot_torch(robot, q_leaf)

    if _DEBUG_MANIP:
        print(
            "[manip_grad] [OLD] logw computed with internal leaf; "
            f"logw.grad_fn={logw.grad_fn}"
        )

    g = torch.autograd.grad(logw, q_leaf, retain_graph=False, allow_unused=True)[0]

    if g is not None:
        if _DEBUG_MANIP:
            print("[manip_grad] [OLD] internal-leaf autograd produced gradient.")
        return g

    # If autograd says q_leaf is unused (rare), use FD in Torch
    if _DEBUG_MANIP:
        print("[manip_grad] [FD] internal autograd returned None; using finite differences.")

    eps_t = torch.as_tensor(eps, dtype=q0.dtype, device=q0.device)

    def _logw_at(q_vec: Tensor) -> Tensor:
        with torch.no_grad():
            return _logw_from_robot_torch(robot, q_vec)

    base = _logw_at(q0.view(-1))
    g_fd = torch.zeros_like(q0.view(-1))
    for i in range(q0.numel()):
        dq = torch.zeros_like(q0.view(-1))
        dq[i] = eps_t
        logw_plus = _logw_at(q0.view(-1) + dq)
        g_fd[i] = (logw_plus - base) / eps_t

    return g_fd


# --------------------------------------------------------------------------- #
# New manipulability gradient: geometric Jacobian based (Torch-only)
# --------------------------------------------------------------------------- #

def manip_grad(env, q, qd, eps: float) -> Tensor:
    """
    Gradient of log Yoshikawa manipulability w.r.t. q (Torch-only).

    Torch path:
        1) NEW graph-preserving autograd path:
            - compute log w(q) using _log_yoshikawa_torch without detach().
            - g_new = d logw / dq_used via autograd.grad(logw, q_used).
            - This is the best candidate if you want external gradients
              w.r.t. the q you passed in.

        2) If that fails (exception or g_new is None), fall back to OLD
           internal-leaf implementation (_manip_grad_torch_internal),
           which still uses the HTM-based robot path and can fall back
           to finite differences in Torch.

    Arguments
    ---------
    env : Environment-like object
        Must have env.effector.skeleton or env.skeleton.
    q, qd : array-like or Tensor, shape (n,)
        Joint positions and velocities.
    eps : float
        Step for finite-difference backup.

    Returns
    -------
    g : Tensor, shape (n,)
        Gradient d logw(q) / dq.
    """
    skel = _get_skeleton_from_env(env)

    if _DEBUG_MANIP:
        print("[manip_grad] Torch backend.")

    g = None

    # -------- Path 1: NEW graph-preserving autograd --------
    try:
        if _DEBUG_MANIP:
            print("[manip_grad] [NEW] graph-preserving autograd path...")
        logw, q_used = _log_yoshikawa_torch(env, skel, q, qd)

        g_new = torch.autograd.grad(
            logw, q_used, retain_graph=False, allow_unused=True
        )[0]

        if g_new is not None:
            if _DEBUG_MANIP:
                print(
                    "[manip_grad] [NEW] autograd ok. "
                    f"q_used.shape={q_used.shape}, g_new.shape={g_new.shape}"
                )
            g = g_new
        else:
            if _DEBUG_MANIP:
                print("[manip_grad] [NEW] autograd returned None.")
    except Exception as e:
        if _DEBUG_MANIP:
            print("[manip_grad] [NEW] path raised exception:", repr(e))
        g = None

    # -------- Path 2: OLD internal-leaf + FD backup --------
    if g is None:
        if _DEBUG_MANIP:
            print("[manip_grad] [OLD] Falling back to internal-leaf implementation...")
        g = _manip_grad_torch_internal(skel, q, qd, eps)

    return g  # (n,)


# --------------------------------------------------------------------------- #
# Nullspace injection (Torch-only)
# --------------------------------------------------------------------------- #

def add_nullspace_manip_(
    qdd_post: Tensor,
    env,
    q,
    qd,
    J_xy,  # unused, kept for API compatibility
    P: KinGuardParams,
    eta,
):
    """
    Add manipulability-gradient-based nullspace term to qdd_post (Torch-only).

    Parameters
    ----------
    qdd_post : Tensor, shape (n,)
        Current joint accelerations after main task-space control.
    env : Environment-like
        Passed through to manip_grad (must expose skeleton as documented).
    q, qd : Tensor or array-like, shape (n,)
        Joint positions and velocities at the current step.
    J_xy : unused
        Kept only for API compatibility with older code.
    P : KinGuardParams
        Contains k_manip, grad_eps, and enable_manip_nullspace.
    eta : float or Tensor
        Scaling factor (e.g. from an energy tank); can be scalar or 0D tensor.

    Returns
    -------
    qdd_out : Tensor, shape (n,)
        Joint accelerations with added nullspace term.
    k_eff : float
        Effective gain k_manip * eta used for the nullspace term (for logging).
    """
    qdd_post = torch.as_tensor(qdd_post)

    if not P.enable_manip_nullspace:
        if _DEBUG_MANIP:
            print("[add_nullspace_manip] Manipulability nullspace disabled.")
        return qdd_post, 0.0

    # Effective gain (keep as plain float for logging)
    if isinstance(eta, Tensor):
        k_eff = P.k_manip * float(eta.detach().cpu().item())
    else:
        k_eff = P.k_manip * float(eta)

    if _DEBUG_MANIP:
        print(f"[add_nullspace_manip] k_eff = {k_eff}")

    g = manip_grad(env, q, qd, P.grad_eps)

    g_t = torch.as_tensor(g, dtype=qdd_post.dtype, device=qdd_post.device)
    qdd_out = qdd_post + k_eff * g_t

    return qdd_out, k_eff

def add_nullspace_manip(
    qdd_post: Tensor,
    env,
    q,
    qd,
    J_xy,  # unused, kept for API compatibility
    P: KinGuardParams,
    eta,
):
    """
    Add manipulability-gradient-based nullspace term to qdd_post (Torch-only).

    Parameters
    ----------
    qdd_post : Tensor
        - (n,)   for unbatched
        - (B,n)  for batched
    env : Environment-like
        Passed through to manip_grad (must expose skeleton as documented).
    q, qd : Tensor or array-like
        - (n,)   for unbatched
        - (B,n)  for batched
    J_xy : unused
        Kept only for API compatibility with older code.
    P : KinGuardParams
        Contains k_manip, grad_eps, and enable_manip_nullspace.
    eta : float or Tensor
        - scalar / 0D tensor
        - or (B,) tensor for batched case

    Returns
    -------
    qdd_out : Tensor
        - (n,)   for unbatched
        - (B,n)  for batched
    k_eff :
        - float for unbatched (as before)
        - (B,) Tensor for batched (per-sample effective gain)
    """
    qdd_post = torch.as_tensor(qdd_post)
    device, dtype = qdd_post.device, qdd_post.dtype

    # ------------------------------------------
    # Early exit if manipulability disabled
    # ------------------------------------------
    if not P.enable_manip_nullspace:
        if qdd_post.ndim == 1:
            # unbatched: keep exact old behavior
            if _DEBUG_MANIP:
                print("[add_nullspace_manip] Manipulability nullspace disabled (unbatched).")
            return qdd_post, 0.0
        elif qdd_post.ndim == 2:
            # batched: return zeros per batch
            if _DEBUG_MANIP:
                print("[add_nullspace_manip] Manipulability nullspace disabled (batched).")
            B = qdd_post.shape[0]
            k_eff_b = torch.zeros(B, device=device, dtype=dtype)
            return qdd_post, k_eff_b
        else:
            raise ValueError(
                f"add_nullspace_manip: qdd_post must be (n,) or (B,n), got {tuple(qdd_post.shape)}"
            )

    # ------------------------------------------
    # Helper: normalize eta to a (B,) tensor
    # ------------------------------------------
    def _eta_to_batch_vec(eta_in, B_local: int) -> Tensor:
        if isinstance(eta_in, Tensor):
            e = eta_in.to(device=device, dtype=dtype)
            if e.ndim == 0:
                # scalar -> broadcast to (B,)
                e = e.view(1).expand(B_local)
            elif e.ndim == 1:
                if e.shape[0] == 1 and B_local > 1:
                    e = e.expand(B_local)
                elif e.shape[0] != B_local:
                    raise ValueError(
                        f"add_nullspace_manip: eta batch {e.shape[0]} "
                        f"!= B={B_local}"
                    )
            else:
                # Collapse any extra dims to 1D
                e = e.view(-1)
                if e.shape[0] == 1 and B_local > 1:
                    e = e.expand(B_local)
                elif e.shape[0] != B_local:
                    raise ValueError(
                        f"add_nullspace_manip: eta batch {e.shape[0]} "
                        f"!= B={B_local}"
                    )
        else:
            # python scalar / float
            e = torch.full(
                (B_local,),
                float(eta_in),
                device=device,
                dtype=dtype,
            )
        return e

    # ------------------------------------------
    # UNBATCHED CASE: qdd_post: (n,)
    # ------------------------------------------
    if qdd_post.ndim == 1:
        # Effective gain (keep as plain float for logging, as before)
        if isinstance(eta, Tensor):
            e = eta.detach()
            if e.numel() > 1:
                e = e.view(-1)[0]
            eta_val = float(e.cpu().item())
        else:
            eta_val = float(eta)

        k_eff = P.k_manip * eta_val
        if _DEBUG_MANIP:
            print(f"[add_nullspace_manip] (unbatched) k_eff = {k_eff}")

        g = manip_grad(env, q, qd, P.grad_eps)  # (n,)
        g_t = torch.as_tensor(g, dtype=dtype, device=device)
        qdd_out = qdd_post + k_eff * g_t
        return qdd_out, k_eff

    # ------------------------------------------
    # BATCHED CASE: qdd_post: (B,n)
    # ------------------------------------------
    if qdd_post.ndim == 2:
        B, n = qdd_post.shape

        q_t = torch.as_tensor(q, dtype=dtype, device=device)
        qd_t = torch.as_tensor(qd, dtype=dtype, device=device)

        # Normalize q, qd to (B,n)
        if q_t.ndim == 1:
            q_t = q_t.unsqueeze(0).expand(B, -1)
        if qd_t.ndim == 1:
            qd_t = qd_t.unsqueeze(0).expand(B, -1)

        if q_t.shape != (B, n):
            raise ValueError(
                f"add_nullspace_manip: q must be (B,n)={B,n}, got {tuple(q_t.shape)}"
            )
        if qd_t.shape != (B, n):
            raise ValueError(
                f"add_nullspace_manip: qd must be (B,n)={B,n}, got {tuple(qd_t.shape)}"
            )

        # Broadcast eta to (B,)
        eta_b = _eta_to_batch_vec(eta, B)  # (B,)

        qdd_list = []
        k_eff_list = []

        for b in range(B):
            eta_val = float(eta_b[b].detach().cpu().item())
            k_eff_b = P.k_manip * eta_val

            if _DEBUG_MANIP:
                print(f"[add_nullspace_manip] (batch {b}) k_eff = {k_eff_b}")

            g_b = manip_grad(env, q_t[b], qd_t[b], P.grad_eps)  # (n,)
            g_b = torch.as_tensor(g_b, dtype=dtype, device=device)

            qdd_b = qdd_post[b] + k_eff_b * g_b  # (n,)
            qdd_list.append(qdd_b)
            k_eff_list.append(k_eff_b)

        qdd_out = torch.stack(qdd_list, dim=0)  # (B,n)
        k_eff_vec = torch.tensor(
            k_eff_list,
            device=device,
            dtype=dtype,
        )  # (B,)

        return qdd_out, k_eff_vec

    # ------------------------------------------
    # Unsupported shape
    # ------------------------------------------
    raise ValueError(
        f"add_nullspace_manip: qdd_post must be (n,) or (B,n), got {tuple(qdd_post.shape)}"
    )

# --------------------------------------------------------------------------- #
# Tiny Torch-only smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=6, sci_mode=False)

    print("[kinematics_guard_torch] Simple smoke test...")

    # ---- Test DLS pinv + scaling (unbatched) ----
    J = torch.tensor([[1.0, 0.2],
                      [0.1, 0.8]], dtype=torch.get_default_dtype())
    P = KinGuardParams()

    J_dls, smin, lamJ = adaptive_dls_pinv(J, n=2, P=P)
    xd_d = torch.tensor([0.1, 0.0])
    xdd_d = torch.tensor([0.0, 0.1])
    xd_sc, xdd_sc, a = scale_task_by_J(xd_d, xdd_d, smin, P)

    print("  [unbatched] smin:", float(smin), "| lamJ:", float(lamJ), "| alpha:", float(a))
    print("  J_dls:", J_dls)

    # ---- Test DLS pinv + scaling (batched) ----
    J_b = torch.stack([J, 0.5 * J], dim=0)  # (2,2) -> (2,2,2)
    J_dls_b, smin_b, lamJ_b = adaptive_dls_pinv(J_b, n=2, P=P)
    xd_d_b = torch.stack([xd_d, xd_d], dim=0)
    xdd_d_b = torch.stack([xdd_d, xdd_d], dim=0)
    xd_sc_b, xdd_sc_b, a_b = scale_task_by_J(xd_d_b, xdd_d_b, smin_b, P)

    print("  [batched] smin_b:", smin_b)
    print("  [batched] lamJ_b:", lamJ_b)
    print("  [batched] alpha_b:", a_b)
    print("  [batched] J_dls_b shape:", J_dls_b.shape)

    # ---- Manipulability gradient smoke (requires your Torch skeleton stack) ----
    try:
        from model_lib.skeleton_torch import TwoDofArm

        arm = TwoDofArm(
            m1=1.82,
            m2=1.43,
            l1g=0.135,
            l2g=0.165,
            i1=0.051,
            i2=0.057,
            l1=0.309,
            l2=0.333,
            device=torch.device("cpu"),
            dtype=torch.get_default_dtype(),
        )

        class Env:
            def __init__(self, skeleton):
                self.skeleton = skeleton

        env = Env(arm)

        q = torch.tensor([0.4, 0.7], requires_grad=True)
        qd = torch.tensor([0.0, 0.0])

        g = manip_grad(env, q, qd, P.grad_eps)
        print("  manip_grad(q):", g)

        # Test nullspace injection
        qdd0 = torch.zeros_like(q)
        qdd_out, k_eff = add_nullspace_manip(qdd0, env, q, qd, None, P, eta=1.0)
        print("  qdd_out:", qdd_out, "| k_eff:", k_eff)

        # Check differentiability wrt q
        cost = g.dot(q)  # simple scalar
        cost.backward()
        print("  d(cost)/dq:", q.grad)

    except Exception as e:
        print("  [manip_grad smoke] skipped (TwoDofArm / HTM_torch not available):", repr(e))

    print("[kinematics_guard_torch] smoke ✓")
