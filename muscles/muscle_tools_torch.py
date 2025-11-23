# muscle_tools_torch.py
# -*- coding: utf-8 -*-
"""
Torch-only muscle utilities (batchable).

Port of muscles/muscle_tools.py (NumPy) to pure Torch:

- MuscleNumerics
- get_Fmax_vec(env, M, device=None, dtype=None)

- active_force_from_activation(a, geom_lenvel, muscle)
    Unbatched:
        a: (M,)
        geom_lenvel: (2,M) or (1,2,M)
        -> returns (M,)
    Batched:
        a: (B,M)
        geom_lenvel: (B,2,M) or (1,2,M) or (2,M)
        -> returns (B,M)

- force_to_activation_bisect(F_des, geom_lenvel, muscle, flpe, Fmax, iters=22)
    Unbatched:
        F_des: (M,)
        -> returns (M,)
    Batched:
        F_des: (B,M)
        -> returns (B,M)

- saturation_repair_tau(A, F_pred, a, a_min, a_max, Fmax_vec, tau_des=None)
    Unbatched:
        A: (D,M), F_pred/a/Fmax_vec: (M,)
        -> returns (M,)
    Batched:
        A: (B,D,M), F_pred/a/Fmax_vec: (B,M)
        -> returns (B,M)

- apply_internal_force_regulation(A, F_base, F_bias, Fmax_vec, ...)
    Unbatched:
        A: (D,M), F_base/F_bias/Fmax_vec: (M,)
        -> returns (M,)
    Batched:
        A: (B,D,M), F_base/F_bias/Fmax_vec: (B,M)
        -> returns (B,M)

All functions are Torch-only, GPU-safe, and differentiable where it matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, List

import torch
from torch import Tensor

from utils.math_utils_torch import right_pinv_rows_weighted


# --------------------------------------------------------------------------- #
# Numerics parameters (kept for API parity)
# --------------------------------------------------------------------------- #

@dataclass
class MuscleNumerics:
    eps: float = 1e-8
    linesearch_eps: float = 1e-6
    linesearch_safety: float = 0.99
    bisect_iters: int = 22


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _ref_tensor(*args: Any) -> Optional[Tensor]:
    """Return the first Tensor among args, or None if none are Tensors."""
    for v in args:
        if isinstance(v, Tensor):
            return v
    return None


def _as_tensor_like(x: Any, like: Tensor) -> Tensor:
    """Convert x to a tensor with same dtype/device as 'like'."""
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


# --------------------------------------------------------------------------- #
# Fmax helpers
# --------------------------------------------------------------------------- #

def get_Fmax_vec(
    env,
    M: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    Torch version of get_Fmax_vec(env, M).

    env.muscle.max_iso_force can be scalar, list, NumPy array, or Tensor.

    Returns
    -------
    Fmax_vec : Tensor, shape (M,)
    """
    raw = getattr(env.muscle, "max_iso_force", None)

    if raw is None:
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.get_default_dtype()
        return torch.ones(M, dtype=dtype, device=device)

    raw_t = torch.as_tensor(raw)
    if device is None:
        device = raw_t.device if raw_t.numel() > 0 else torch.device("cpu")
    if dtype is None:
        dtype = raw_t.dtype if raw_t.numel() > 0 else torch.get_default_dtype()

    raw_t = raw_t.to(device=device, dtype=dtype)

    if raw_t.numel() == 0:
        return torch.ones(M, dtype=dtype, device=device)

    if raw_t.numel() == 1:
        val = raw_t.reshape(-1)[0]
        return torch.full((M,), val, dtype=dtype, device=device)

    if raw_t.shape[-1] == M:
        return raw_t.reshape(-1)[-M:].clone()

    mean_val = raw_t.to(dtype=torch.float64).mean()
    mean_val = mean_val.to(dtype=dtype)
    return torch.full((M,), mean_val, dtype=dtype, device=device)


# --------------------------------------------------------------------------- #
# Active force from activation (batchable)
# --------------------------------------------------------------------------- #

def active_force_from_activation(a: Any, geom_lenvel: Any, muscle) -> Tensor:
    """
    Torch version of active_force_from_activation.

    Unbatched:
        a : (M,)
        geom_lenvel : (2,M) or (1,2,M)
        -> returns (M,)

    Batched:
        a : (B,M)
        geom_lenvel : (B,2,M) or (1,2,M) or (2,M)
        -> returns (B,M)

    muscle : Torch muscle object
        Must expose:
            - state_name: list of state channel names
            - min_activation: float
            - _integrate(dt, state, activation, geom_lenvel)
              with:
                state: (B, C, M)
                activation: (B,M)  (or compatible)
                geom_lenvel: (B, 2, M)
    """
    ref = _ref_tensor(a, geom_lenvel)
    if ref is None:
        device = torch.device("cpu")
        dtype = torch.get_default_dtype()
    else:
        device, dtype = ref.device, ref.dtype

    # a: (M,) or (B,M)
    a_t = torch.as_tensor(a, dtype=dtype, device=device)
    if a_t.ndim == 1:
        B = 1
        M = a_t.shape[0]
        a_b = a_t.view(1, M)   # (1,M)
        unbatched = True
    elif a_t.ndim == 2:
        B, M = a_t.shape       # (B,M)
        a_b = a_t
        unbatched = False
    else:
        raise ValueError(
            f"active_force_from_activation: a must be (M,) or (B,M), got {tuple(a_t.shape)}"
        )

    # geom_lenvel -> (B,2,M)
    gl = torch.as_tensor(geom_lenvel, dtype=dtype, device=device)
    if gl.ndim == 2:
        # (2,M) -> (1,2,M) -> (B,2,M)
        if gl.shape[0] != 2:
            raise ValueError(
                f"geom_lenvel 2D must be (2,M), got {tuple(gl.shape)}"
            )
        gl_b = gl.unsqueeze(0).expand(B, -1, -1)
    elif gl.ndim == 3:
        # (B0,2,M)
        if gl.shape[1] != 2:
            raise ValueError(
                f"geom_lenvel 3D must be (B0,2,M), got {tuple(gl.shape)}"
            )
        if gl.shape[0] == 1 and B > 1:
            gl_b = gl.expand(B, -1, -1)
        elif gl.shape[0] == B:
            gl_b = gl
        else:
            raise ValueError(
                f"geom_lenvel batch dim {gl.shape[0]} != a batch dim {B}"
            )
    else:
        raise ValueError(
            f"geom_lenvel must be (2,M) or (B,2,M), got {tuple(gl.shape)}"
        )

    # State: (B, C, M) with C = len(state_name)
    C = len(muscle.state_name)
    state0 = torch.zeros((B, C, M), dtype=dtype, device=device)

    # Activation: (B,M) – let the muscle implementation decide how to use it
    act_b = a_b  # (B,M)

    # dt = 0.0 to evaluate instantaneous force contributions
    out = muscle._integrate(0.0, state0, act_b, gl_b)  # expected (B, C, M)

    names = muscle.state_name
    ia = names.index("activation")
    ifl = names.index("force-length CE")
    ifv = names.index("force-velocity CE")

    act_ch = out[:, ia, :]   # (B,M)
    flce   = out[:, ifl, :]  # (B,M)
    fvce   = out[:, ifv, :]  # (B,M)

    F_act = act_ch * flce * fvce  # (B,M)

    if unbatched:
        return F_act[0]  # (M,)
    return F_act        # (B,M)


# --------------------------------------------------------------------------- #
# Force -> activation mapping (bisection, batchable)
# --------------------------------------------------------------------------- #

def force_to_activation_bisect(
    F_des: Any,
    geom_lenvel: Any,
    muscle,
    flpe: Any,
    Fmax: Any,
    iters: int = 22,
) -> Tensor:
    """
    Torch version of force_to_activation_bisect (batchable).

    Unbatched:
        F_des : (M,)
        -> returns (M,)

    Batched:
        F_des : (B,M)
        -> returns (B,M)

    geom_lenvel : as in active_force_from_activation (2,M) or (B,2,M) ...
    muscle : Torch muscle object (same assumptions as above).
    flpe : passive force-length contribution; scalar, (M,) or (B,M)
    Fmax : scalar, (M,) or (B,M)
    """
    ref = _ref_tensor(F_des, geom_lenvel, flpe, Fmax)
    if ref is None:
        device = torch.device("cpu")
        dtype = torch.get_default_dtype()
    else:
        device, dtype = ref.device, ref.dtype

    # F_des: (M,) or (B,M)
    F_des_t = torch.as_tensor(F_des, dtype=dtype, device=device)
    if F_des_t.ndim == 1:
        B = 1
        M = F_des_t.shape[0]
        F_b = F_des_t.view(1, M)
        unbatched = True
    elif F_des_t.ndim == 2:
        B, M = F_des_t.shape
        F_b = F_des_t
        unbatched = False
    else:
        raise ValueError(
            f"force_to_activation_bisect: F_des must be (M,) or (B,M), got {tuple(F_des_t.shape)}"
        )

    # Fmax: scalar, (M,) or (B,M)
    Fmax_t = torch.as_tensor(Fmax, dtype=dtype, device=device)
    if Fmax_t.ndim == 0:
        Fmax_b = torch.full_like(F_b, Fmax_t.view(()))  # (B,M)
    elif Fmax_t.ndim == 1:
        if Fmax_t.shape[0] != M:
            mean_val = Fmax_t.to(dtype=torch.float64).mean().to(dtype=dtype)
            Fmax_b = torch.full_like(F_b, mean_val)
        else:
            Fmax_b = Fmax_t.view(1, M).expand(B, -1)
    elif Fmax_t.ndim == 2:
        if Fmax_t.shape == (B, M):
            Fmax_b = Fmax_t
        elif Fmax_t.shape == (1, M):
            Fmax_b = Fmax_t.expand(B, -1)
        else:
            raise ValueError(
                f"force_to_activation_bisect: Fmax shape {Fmax_t.shape} incompatible "
                f"with (B,M)={(B,M)}"
            )
    else:
        raise ValueError(
            f"force_to_activation_bisect: Fmax must be scalar, (M,) or (B,M), got {tuple(Fmax_t.shape)}"
        )

    # flpe: scalar, (M,) or (B,M)
    flpe_t = torch.as_tensor(flpe, dtype=dtype, device=device)
    if flpe_t.ndim == 0:
        flpe_b = torch.full_like(F_b, flpe_t.view(()))
    elif flpe_t.ndim == 1:
        if flpe_t.shape[0] != M:
            flpe_vec = flpe_t.reshape(-1)[-M:]
            flpe_b = flpe_vec.view(1, M).expand(B, -1)
        else:
            flpe_b = flpe_t.view(1, M).expand(B, -1)
    elif flpe_t.ndim == 2:
        if flpe_t.shape == (B, M):
            flpe_b = flpe_t
        elif flpe_t.shape == (1, M):
            flpe_b = flpe_t.expand(B, -1)
        else:
            raise ValueError(
                f"force_to_activation_bisect: flpe shape {flpe_t.shape} incompatible "
                f"with (B,M)={(B,M)}"
            )
    else:
        raise ValueError(
            f"force_to_activation_bisect: flpe must be scalar, (M,) or (B,M), got {tuple(flpe_t.shape)}"
        )

    # geom_lenvel -> (B,2,M)
    gl = torch.as_tensor(geom_lenvel, dtype=dtype, device=device)
    if gl.ndim == 2:
        if gl.shape[0] != 2:
            raise ValueError(
                f"geom_lenvel 2D must be (2,M), got {tuple(gl.shape)}"
            )
        gl_b = gl.unsqueeze(0).expand(B, -1, -1)
    elif gl.ndim == 3:
        if gl.shape[1] != 2:
            raise ValueError(
                f"geom_lenvel 3D must be (B0,2,M), got {tuple(gl.shape)}"
            )
        if gl.shape[0] == 1 and B > 1:
            gl_b = gl.expand(B, -1, -1)
        elif gl.shape[0] == B:
            gl_b = gl
        else:
            raise ValueError(
                f"geom_lenvel batch dim {gl.shape[0]} != F_des batch dim {B}"
            )
    else:
        raise ValueError(
            f"force_to_activation_bisect: geom_lenvel must be (2,M) or (B,2,M), got {tuple(gl.shape)}"
        )

    # Target active part (B,M)
    target_active = torch.clamp(F_b / Fmax_b - flpe_b, min=0.0)

    # Bisection bounds (B,M)
    lo = torch.full_like(target_active, float(muscle.min_activation))
    hi = torch.ones_like(target_active)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)  # (B,M)
        af = active_force_from_activation(mid, gl_b, muscle)  # (B,M)
        gt = af > target_active
        hi = torch.where(gt, mid, hi)
        lo = torch.where(gt, lo, mid)

    a_out = 0.5 * (lo + hi)
    a_out = torch.clamp(a_out, min=float(muscle.min_activation), max=1.0)

    if unbatched:
        return a_out[0]
    return a_out


# --------------------------------------------------------------------------- #
# Saturation repair in torque space (batchable)
# --------------------------------------------------------------------------- #

def _saturation_repair_tau_single(
    A: Tensor,
    F_pred: Tensor,
    a: Tensor,
    a_min_vec: Tensor,
    a_max_vec: Tensor,
    Fmax_vec: Tensor,
    tau_des: Optional[Tensor],
) -> Tensor:
    """Unbatched core for saturation_repair_tau."""
    device, dtype = A.device, A.dtype

    F_pred_t = F_pred.view(-1)
    a_t = a.view(-1)
    Fmax_t = Fmax_vec.view(-1)

    tau_pred = A @ F_pred_t  # (D,)
    if tau_des is not None:
        tau_err = tau_des.view(-1) - tau_pred
    else:
        tau_err = -tau_pred

    free = (a_t > a_min_vec + 1e-4) & (a_t < a_max_vec - 1e-4)
    if not torch.any(free):
        return F_pred_t

    A_free = A[:, free]  # (D, M_free)
    w = 1.0 / torch.clamp(Fmax_t[free], min=1e-9)  # (M_free,)

    Aw = A_free @ torch.diag(w)  # (D, M_free)

    G = Aw.transpose(-1, -2) @ Aw            # (M_free, M_free)
    rhs = Aw.transpose(-1, -2) @ (-tau_err)  # (M_free,)

    if G.numel() == 0:
        return F_pred_t

    I = torch.eye(G.shape[0], dtype=dtype, device=device)
    G_reg = G + 1e-12 * I

    try:
        dw = torch.linalg.solve(G_reg, rhs)  # (M_free,)
    except RuntimeError:
        dw = torch.linalg.pinv(G_reg) @ rhs

    dF_free = torch.diag(w) @ dw  # (M_free,)

    F_new = F_pred_t.clone()
    F_new[free] = torch.clamp(F_pred_t[free] + dF_free, min=0.0)
    return F_new


def saturation_repair_tau(
    A: Any,
    F_pred: Any,
    a: Any,
    a_min: Any,
    a_max: Any,
    Fmax_vec: Any,
    tau_des: Any = None,
) -> Tensor:
    """
    Torch version of saturation_repair_tau (batchable).

    Unbatched:
        A : (D,M)
        F_pred, a, Fmax_vec : (M,)
        tau_des : (D,) or None
        -> returns (M,)

    Batched:
        A : (B,D,M)
        F_pred, a, Fmax_vec : (B,M)
        tau_des : (B,D) or None
        -> returns (B,M)
    """
    A_t = torch.as_tensor(A)
    device, dtype = A_t.device, A_t.dtype

    # Unbatched
    if A_t.ndim == 2:
        D, M = A_t.shape
        F_pred_t = torch.as_tensor(F_pred, dtype=dtype, device=device).view(M)
        a_t = torch.as_tensor(a, dtype=dtype, device=device).view(M)
        Fmax_t = torch.as_tensor(Fmax_vec, dtype=dtype, device=device).view(M)

        a_min_t = torch.as_tensor(a_min, dtype=dtype, device=device)
        a_max_t = torch.as_tensor(a_max, dtype=dtype, device=device)
        if a_min_t.ndim == 0:
            a_min_vec = torch.full_like(a_t, a_min_t)
        else:
            a_min_vec = a_min_t.view(M)
        if a_max_t.ndim == 0:
            a_max_vec = torch.full_like(a_t, a_max_t)
        else:
            a_max_vec = a_max_t.view(M)

        if tau_des is not None:
            tau_des_t = torch.as_tensor(tau_des, dtype=dtype, device=device)
        else:
            tau_des_t = None

        F_new = _saturation_repair_tau_single(
            A_t, F_pred_t, a_t, a_min_vec, a_max_vec, Fmax_t, tau_des_t
        )
        return F_new

    # Batched: A: (B,D,M)
    if A_t.ndim == 3:
        B, D, M = A_t.shape
        F_pred_t = torch.as_tensor(F_pred, dtype=dtype, device=device)
        a_t = torch.as_tensor(a, dtype=dtype, device=device)
        Fmax_t = torch.as_tensor(Fmax_vec, dtype=dtype, device=device)

        if F_pred_t.shape != (B, M):
            raise ValueError(
                f"saturation_repair_tau: batched F_pred must be (B,M)={B,M}, "
                f"got {tuple(F_pred_t.shape)}"
            )
        if a_t.shape != (B, M):
            raise ValueError(
                f"saturation_repair_tau: batched a must be (B,M)={B,M}, "
                f"got {tuple(a_t.shape)}"
            )
        if Fmax_t.shape == (M,):
            Fmax_b = Fmax_t.view(1, M).expand(B, -1)
        elif Fmax_t.shape == (B, M):
            Fmax_b = Fmax_t
        else:
            raise ValueError(
                f"saturation_repair_tau: Fmax_vec must be (M,) or (B,M), "
                f"got {tuple(Fmax_t.shape)}"
            )

        a_min_t = torch.as_tensor(a_min, dtype=dtype, device=device)
        a_max_t = torch.as_tensor(a_max, dtype=dtype, device=device)

        if a_min_t.ndim == 0:
            a_min_b = torch.full_like(a_t, a_min_t)
        elif a_min_t.shape == (M,):
            a_min_b = a_min_t.view(1, M).expand(B, -1)
        elif a_min_t.shape == (B, M):
            a_min_b = a_min_t
        else:
            raise ValueError(
                f"saturation_repair_tau: a_min must be scalar, (M,) or (B,M), got {tuple(a_min_t.shape)}"
            )

        if a_max_t.ndim == 0:
            a_max_b = torch.full_like(a_t, a_max_t)
        elif a_max_t.shape == (M,):
            a_max_b = a_max_t.view(1, M).expand(B, -1)
        elif a_max_t.shape == (B, M):
            a_max_b = a_max_t
        else:
            raise ValueError(
                f"saturation_repair_tau: a_max must be scalar, (M,) or (B,M), got {tuple(a_max_t.shape)}"
            )

        if tau_des is not None:
            tau_des_t = torch.as_tensor(tau_des, dtype=dtype, device=device)
            if tau_des_t.ndim == 1:
                tau_des_b = tau_des_t.view(1, D).expand(B, -1)
            elif tau_des_t.shape == (B, D):
                tau_des_b = tau_des_t
            else:
                raise ValueError(
                    f"saturation_repair_tau: tau_des batched must be (B,D), got {tuple(tau_des_t.shape)}"
                )
        else:
            tau_des_b = None

        F_list: List[Tensor] = []
        for b in range(B):
            tb = tau_des_b[b] if tau_des_b is not None else None
            F_new_b = _saturation_repair_tau_single(
                A_t[b],
                F_pred_t[b],
                a_t[b],
                a_min_b[b],
                a_max_b[b],
                Fmax_b[b],
                tb,
            )
            F_list.append(F_new_b)

        return torch.stack(F_list, dim=0)  # (B,M)

    raise ValueError(
        f"saturation_repair_tau: A must be (D,M) or (B,D,M), got {tuple(A_t.shape)}"
    )


# --------------------------------------------------------------------------- #
# Internal-force regulation (batchable)
# --------------------------------------------------------------------------- #

def _apply_internal_force_regulation_single(
    A: Tensor,
    F_base: Tensor,
    F_bias: Tensor,
    Fmax_vec: Tensor,
    eps: float,
    linesearch_eps: float,
    linesearch_safety: float,
    scale: float,
) -> Tensor:
    """Unbatched core for apply_internal_force_regulation."""
    device, dtype = A.device, A.dtype

    F_base_t = F_base.view(-1)
    F_bias_t = F_bias.view(-1)
    Fmax_t = Fmax_vec.view(-1)

    M = F_base_t.shape[0]

    w = 1.0 / torch.clamp(Fmax_t, min=1e-9) ** 2
    W = torch.diag(w)  # (M,M)

    winv_diag = torch.clamp(Fmax_t, min=1e-9) ** 2
    Winv = torch.diag(winv_diag)  # (M,M)

    # A_pinv: (M,D) since right_pinv_rows_weighted returns shape (n,m)
    A_pinv = right_pinv_rows_weighted(A, Winv, eps=eps)  # (M,D)

    N_A = torch.eye(M, dtype=dtype, device=device) - A_pinv @ A  # (M,M)

    B = N_A.transpose(-1, -2) @ W @ N_A
    rhs = N_A.transpose(-1, -2) @ W @ (scale * (F_bias_t - F_base_t))

    if B.numel() == 0:
        return F_base_t

    try:
        z = torch.linalg.solve(B, rhs)
    except RuntimeError:
        z = torch.linalg.pinv(B) @ rhs

    dF = N_A @ z
    F_try = F_base_t + dF

    if torch.any(F_try < 0.0):
        alpha_max = 1.0
        mask = dF < 0.0
        idx_neg = torch.where(mask)[0]
        for i in idx_neg:
            if dF[i].abs() < 1e-12:
                continue
            alpha_i = -F_base_t[i] / dF[i]
            alpha_max = min(alpha_max, float(alpha_i))

        alpha = max(0.0, min(1.0, linesearch_safety * alpha_max))
        if alpha <= linesearch_eps:
            return F_base_t

        F_try = F_base_t + alpha * dF

    return torch.clamp(F_try, min=0.0)


def apply_internal_force_regulation(
    A: Any,
    F_base: Any,
    F_bias: Any,
    Fmax_vec: Any,
    eps: float = 1e-8,
    linesearch_eps: float = 1e-6,
    linesearch_safety: float = 0.99,
    scale: float = 1.0,
) -> Tensor:
    """
    Torch version of apply_internal_force_regulation (batchable).

    Minimize ||F - F_bias||_W subject to AF = A F_base, with W = diag(1/Fmax^2),
    and F >= 0 enforced via a simple line-search if needed.

    Unbatched:
        A : (D,M)
        F_base, F_bias, Fmax_vec : (M,)
        -> returns (M,)

    Batched:
        A : (B,D,M)
        F_base, F_bias, Fmax_vec : (B,M)
        -> returns (B,M)
    """
    A_t = torch.as_tensor(A)
    device, dtype = A_t.device, A_t.dtype

    # Unbatched
    if A_t.ndim == 2:
        D, M = A_t.shape
        F_base_t = torch.as_tensor(F_base, dtype=dtype, device=device).view(M)
        F_bias_t = torch.as_tensor(F_bias, dtype=dtype, device=device).view(M)
        Fmax_t = torch.as_tensor(Fmax_vec, dtype=dtype, device=device).view(M)

        F_opt = _apply_internal_force_regulation_single(
            A_t,
            F_base_t,
            F_bias_t,
            Fmax_t,
            eps,
            linesearch_eps,
            linesearch_safety,
            scale,
        )
        return F_opt

    # Batched
    if A_t.ndim == 3:
        B, D, M = A_t.shape
        F_base_t = torch.as_tensor(F_base, dtype=dtype, device=device)
        F_bias_t = torch.as_tensor(F_bias, dtype=dtype, device=device)
        Fmax_t = torch.as_tensor(Fmax_vec, dtype=dtype, device=device)

        if F_base_t.shape != (B, M):
            raise ValueError(
                f"apply_internal_force_regulation: F_base must be (B,M)={B,M}, got {tuple(F_base_t.shape)}"
            )
        if F_bias_t.shape != (B, M):
            raise ValueError(
                f"apply_internal_force_regulation: F_bias must be (B,M)={B,M}, got {tuple(F_bias_t.shape)}"
            )
        if Fmax_t.shape == (M,):
            Fmax_b = Fmax_t.view(1, M).expand(B, -1)
        elif Fmax_t.shape == (B, M):
            Fmax_b = Fmax_t
        else:
            raise ValueError(
                f"apply_internal_force_regulation: Fmax_vec must be (M,) or (B,M), got {tuple(Fmax_t.shape)}"
            )

        F_list: List[Tensor] = []
        for b in range(B):
            F_opt_b = _apply_internal_force_regulation_single(
                A_t[b],
                F_base_t[b],
                F_bias_t[b],
                Fmax_b[b],
                eps,
                linesearch_eps,
                linesearch_safety,
                scale,
            )
            F_list.append(F_opt_b)

        return torch.stack(F_list, dim=0)  # (B,M)

    raise ValueError(
        f"apply_internal_force_regulation: A must be (D,M) or (B,D,M), got {tuple(A_t.shape)}"
    )


# --------------------------------------------------------------------------- #
# Tiny Torch-only smoke test (no real muscles required)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=6, sci_mode=False)

    print("[muscle_tools_torch] Simple smoke test...")

    class DummyMuscle:
        def __init__(self, min_activation=0.01):
            self.state_name = [
                "activation",
                "muscle length",
                "muscle velocity",
                "force-length PE",
                "force-length CE",
                "force-velocity CE",
                "force",
            ]
            self.min_activation = min_activation

        def _integrate(self, dt, state, a, geom_lenvel):
            # Very crude toy model:
            #   activation channel = activation
            #   force-length CE = 1
            #   force-velocity CE = 1
            B, C, M = state.shape
            out = torch.zeros(
                (B, len(self.state_name), M),
                dtype=state.dtype,
                device=state.device,
            )

            ia = self.state_name.index("activation")
            ifl = self.state_name.index("force-length CE")
            ifv = self.state_name.index("force-velocity CE")

            a_t = torch.as_tensor(a, dtype=state.dtype, device=state.device)
            # Accept (B,M) or (B,1,M) or (M,)
            if a_t.ndim == 1:
                # (M,) -> (1,M) -> broadcast to B
                if a_t.shape[0] != M:
                    raise ValueError(
                        f"DummyMuscle: activation 1D must be (M,) with M={M}, "
                        f"got {tuple(a_t.shape)}"
                    )
                a_t = a_t.view(1, M).expand(B, -1)
            elif a_t.ndim == 2:
                # (B,M) expected
                if a_t.shape != (B, M):
                    raise ValueError(
                        f"DummyMuscle: activation 2D must be (B,M)={B,M}, "
                        f"got {tuple(a_t.shape)}"
                    )
            elif a_t.ndim == 3:
                # (B,1,M) -> (B,M)
                if a_t.shape[0] != B or a_t.shape[2] != M:
                    raise ValueError(
                        f"DummyMuscle: activation 3D must be (B,1,M) with "
                        f"B={B}, M={M}, got {tuple(a_t.shape)}"
                    )
                a_t = a_t[:, 0, :]
            else:
                raise ValueError(
                    f"DummyMuscle: activation must be (M,), (B,M) or (B,1,M), "
                    f"got {tuple(a_t.shape)}"
                )

            out[:, ia, :] = a_t
            out[:, ifl, :] = 1.0
            out[:, ifv, :] = 1.0
            return out

    # Dummy env for get_Fmax_vec
    class DummyEnv:
        class M:
            max_iso_force = [10.0, 12.0, 8.0]

        muscle = M()

    env = DummyEnv()
    Fmax_vec = get_Fmax_vec(env, M=3)
    print("  Fmax_vec:", Fmax_vec)

    mus = DummyMuscle()
    geom = torch.zeros(2, 3)  # dummy

    # --- active_force_from_activation unbatched ---
    a0 = torch.tensor([0.1, 0.2, 0.3])
    af = active_force_from_activation(a0, geom, mus)
    print("  [unbatched] active_force_from_activation:", af)

    # --- active_force_from_activation batched ---
    a0_b = torch.stack([a0, 0.5 * a0], dim=0)  # (2,3)
    geom_b = geom.unsqueeze(0)                 # (1,2,3) -> broadcast
    af_b = active_force_from_activation(a0_b, geom_b, mus)
    print("  [batched] active_force_from_activation:", af_b)

    # --- force_to_activation_bisect unbatched ---
    F_des = torch.tensor([1.0, 2.0, 3.0])
    flpe = torch.zeros_like(F_des)
    a_bis = force_to_activation_bisect(F_des, geom, mus, flpe, Fmax_vec)
    print("  [unbatched] force_to_activation_bisect:", a_bis)

    # --- force_to_activation_bisect batched ---
    F_des_b = torch.stack([F_des, 0.5 * F_des], dim=0)
    a_bis_b = force_to_activation_bisect(F_des_b, geom_b, mus, flpe, Fmax_vec)
    print("  [batched] force_to_activation_bisect:", a_bis_b)

    # --- saturation_repair_tau unbatched ---
    A = torch.tensor([[1.0, 0.5, -0.3],
                      [0.0, 0.8,  0.4]])
    F_pred = torch.tensor([1.0, 1.0, 1.0])
    a = torch.tensor([0.5, 0.5, 0.5])
    a_min, a_max = 0.0, 1.0
    F_rep = saturation_repair_tau(A, F_pred, a, a_min, a_max, Fmax_vec)
    print("  [unbatched] saturation_repair_tau F_rep:", F_rep)

    # --- saturation_repair_tau batched ---
    A_b = A.unsqueeze(0).expand(2, -1, -1).clone()
    F_pred_b = torch.stack([F_pred, 0.5 * F_pred], dim=0)
    a_b2 = torch.stack([a, 0.8 * a], dim=0)
    Fmax_b = Fmax_vec  # shared
    F_rep_b = saturation_repair_tau(A_b, F_pred_b, a_b2, a_min, a_max, Fmax_b)
    print("  [batched] saturation_repair_tau F_rep_b:", F_rep_b)

    # --- apply_internal_force_regulation unbatched ---
    F_base = torch.tensor([1.0, 1.0, 1.0])
    F_bias = torch.tensor([2.0, 0.5, 0.5])
    F_opt = apply_internal_force_regulation(A, F_base, F_bias, Fmax_vec)
    print("  [unbatched] apply_internal_force_regulation F_opt:", F_opt)

    # --- apply_internal_force_regulation batched ---
    F_base_b = torch.stack([F_base, 0.5 * F_base], dim=0)
    F_bias_b = torch.stack([F_bias, 0.5 * F_bias], dim=0)
    F_opt_b = apply_internal_force_regulation(A_b, F_base_b, F_bias_b, Fmax_b)
    print("  [batched] apply_internal_force_regulation F_opt_b:", F_opt_b)

    print("[muscle_tools_torch] smoke ✓")
