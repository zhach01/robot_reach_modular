# muscle_guard_torch.py
# -*- coding: utf-8 -*-
"""
Torch-only muscle guard and allocator.

Port of utils/muscle_guard.py to pure Torch:

Implements:
- MuscleGuardParams   (same fields as NumPy version)
- build_weight(Fmax_vec, policy)
- stable_sigmoid(x)
- solve_muscle_forces(tau_des, R, Fmax_vec, eta, P)

Core idea (same as NumPy):
--------------------------------
Given desired joint torques tau_des and moment-arm matrix R:

    R : (D, M)    (D joints, M muscles)

We form:
    A  = -R
    W  = build_weight(Fmax_vec, policy)       (M x M, SPD)
    Gp = A W A^T                             (D x D, SPD-ish)

Compute:
    sminGp = smallest singular value of Gp
    gate   = stable_sigmoid( k_mus * (sminGp - g_thresh) )
    lam_mus = lam_mus_min + (lam_mus_max - lam_mus_min) / (1 + exp(gate))
    lam_mus += (1 - eta) * 1e-3

Then:
    F_p = W A^T (Gp + (eps + lam_mus) I)^(-1) tau_des

If sminGp < 1e-8 or any component of F_p < 0:
    use NNLS fallback:
        F_des = nnls_small_active_set(tau_des, R)
Else:
    F_des = F_p

Torch / batching:
-----------------
- Unbatched:
    tau_des : (D,) or (D,1)
    R       : (D, M)
    Fmax_vec: (M,)
    eta     : float or 0D tensor

- Batched:
    tau_des : (B, D) or (B, D, 1)
    R       : (B, D, M)
    Fmax_vec: (M,) or (B, M)
    eta     : float, 0D tensor, or (B,) tensor

Returns:
    F_des : (M,) or (B, M)
    meta  : dict with keys
        "sminGp": 0D tensor or (B,)
        "lam_mus": 0D tensor or (B,)
        "neg_Fp": int or list[Optional[int]] (count of negative entries
                   in F_p before clipping / fallback, or None for NNLS)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List

import torch
from torch import Tensor

# Prefer Torch NNLS if available; otherwise fall back to NumPy-based version,
# converting tensors <-> numpy only in the (rare) fallback branch.
try:
    from utils.linear_utils_torch import nnls_small_active_set as _nnls_fn
    _NNLS_IS_TORCH = True
except Exception:  # pragma: no cover
    from utils.linear_utils import nnls_small_active_set as _nnls_fn  # type: ignore
    _NNLS_IS_TORCH = False


@dataclass
class MuscleGuardParams:
    lam_mus_min: float = 0.0
    lam_mus_max: float = 1e-2
    k_mus: float = 50.0
    g_thresh: float = 1e-6
    eps: float = 1e-9
    # choose weighting policy: "normalized" (F/Fmax) or "absolute"
    weighting: str = "normalized"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _to_tensor_like(x: Any, like: Tensor) -> Tensor:
    """Convert x to a torch tensor with same dtype/device as 'like'."""
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


def _nnls_wrapper(tau_des: Tensor, R: Tensor) -> Tensor:
    """
    Wrap nnls_small_active_set so it works whether we imported a pure Torch
    or a NumPy version.

    tau_des: (D,) tensor
    R:       (D,M) tensor

    Returns:
        f: (M,) tensor, f >= 0
    """
    if _NNLS_IS_TORCH:
        # Torch implementation: stays on same device/dtype
        return _nnls_fn(tau_des, R)

    # NumPy implementation: convert to CPU/NumPy then back.
    tau_np = tau_des.detach().cpu().numpy()
    R_np = R.detach().cpu().numpy()
    f_np = _nnls_fn(tau_np, R_np)
    return torch.as_tensor(f_np, dtype=tau_des.dtype, device=tau_des.device)


# --------------------------------------------------------------------------- #
# Weight matrix and sigmoid (Torch)
# --------------------------------------------------------------------------- #

def build_weight(Fmax_vec: Tensor, policy: str) -> Tensor:
    """
    Torch version of build_weight(Fmax_vec, policy):

    - "normalized": W = diag(max(Fmax_vec, 1e-9)^2)   (penalize normalized forces)
    - "absolute" : W = I
    """
    F = torch.as_tensor(Fmax_vec)
    device, dtype = F.device, F.dtype

    if F.ndim == 1:
        # (M,)
        M = F.shape[0]
        if policy == "normalized":
            base = torch.clamp(F, min=1e-9) ** 2
            return torch.diag(base)
        elif policy == "absolute":
            return torch.eye(M, dtype=dtype, device=device)
        else:
            raise ValueError(f"Unknown weighting policy '{policy}'")

    elif F.ndim == 2:
        # (B, M) -> (B, M, M)
        B, M = F.shape
        if policy == "normalized":
            base = torch.clamp(F, min=1e-9) ** 2  # (B,M)
            return torch.diag_embed(base)        # (B,M,M)
        elif policy == "absolute":
            I = torch.eye(M, dtype=dtype, device=device)
            return I.unsqueeze(0).expand(B, M, M).clone()
        else:
            raise ValueError(f"Unknown weighting policy '{policy}'")

    else:
        raise ValueError(
            f"build_weight expects Fmax_vec with shape (M,) or (B,M), got {tuple(F.shape)}"
        )


def stable_sigmoid(x: Tensor) -> Tensor:
    """
    Numerically-stable sigmoid-style function:

        σ(x) = 0.5 * (1 + tanh(0.5 * clip(x, -80, 80)))

    Works for scalar or batched tensors.
    """
    x = torch.as_tensor(x)
    x_clamped = torch.clamp(x, min=-80.0, max=80.0)
    return 0.5 * (1.0 + torch.tanh(0.5 * x_clamped))


# --------------------------------------------------------------------------- #
# Core unbatched muscle-force solver (Torch)
# --------------------------------------------------------------------------- #

def _solve_single_muscle_forces(
    tau_des: Tensor,
    R: Tensor,
    Fmax_vec: Tensor,
    eta: Tensor,
    P: MuscleGuardParams,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Unbatched Torch implementation of solve_muscle_forces.

    tau_des : (D,)
    R       : (D, M)
    Fmax_vec: (M,)
    eta     : scalar tensor (0D)
    """
    device, dtype = R.device, R.dtype

    tau_vec = tau_des.view(-1)  # (D,)
    D, M = R.shape

    Fmax_vec = Fmax_vec.view(-1)  # (M,)

    # A = -R
    A = -R  # (D,M)

    # W : (M,M)
    W = build_weight(Fmax_vec, P.weighting)  # (M,M)
    if W.ndim == 3:
        # in case caller passed batched Fmax_vec by mistake, take first batch
        W = W[0]

    # Gp = A W A^T   (D x D)
    Gp = A @ W @ A.transpose(-1, -2)  # (D,D)

    # sminGp from singular values of Gp
    svals = torch.linalg.svdvals(Gp)       # (D,)
    smin_tensor = svals[-1]               # smallest
    sminGp = smin_tensor

    # gate = stable_sigmoid(k_mus * (sminGp - g_thresh))
    arg = P.k_mus * (sminGp - P.g_thresh)
    gate = stable_sigmoid(arg)

    # lam_mus = lam_mus_min + (lam_mus_max - lam_mus_min) / (1 + exp(gate))
    lam_mus = P.lam_mus_min + (P.lam_mus_max - P.lam_mus_min) / (
        1.0 + torch.exp(gate)
    )

    # lam_mus += (1 - eta) * 1e-3
    lam_mus = lam_mus + (1.0 - eta) * 1e-3

    # Regularized solve:
    #   F_p = W A^T (Gp + (eps + lam_mus) I)^(-1) tau_des
    I_D = torch.eye(D, dtype=dtype, device=device)
    lam_tot = P.eps + lam_mus
    G_reg = Gp + lam_tot * I_D                # (D,D)

    # Solve G_reg x = tau_des
    x = torch.linalg.solve(G_reg, tau_vec.unsqueeze(-1)).squeeze(-1)  # (D,)

    F_p = W @ A.transpose(-1, -2) @ x        # (M,)

    # Fallback if ill-conditioned or negative forces
    smin_val = float(smin_tensor.detach().cpu())
    neg_mask = F_p < 0.0
    cond_neg = bool(torch.any(neg_mask))
    cond_singular = smin_val < 1e-8
    cond_fallback = cond_singular or cond_neg

    if cond_fallback:
        # NNLS fallback: solve min ||R f + tau_des|| s.t. f >= 0
        F_des = _nnls_wrapper(tau_vec, R)
        negFp: Optional[int] = None
    else:
        F_des = F_p
        negFp = int(torch.count_nonzero(neg_mask).item())

    meta = {
        "sminGp": sminGp,
        "lam_mus": torch.as_tensor(lam_mus, dtype=dtype, device=device),
        "neg_Fp": negFp,
    }

    return F_des, meta


# --------------------------------------------------------------------------- #
# Public API: batched + unbatched
# --------------------------------------------------------------------------- #

def solve_muscle_forces(
    tau_des: Tensor,
    R: Tensor,
    Fmax_vec: Tensor,
    eta: Any,
    P: MuscleGuardParams,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Torch version of solve_muscle_forces.

    Unbatched:
    ----------
    tau_des : (D,) or (D,1)
    R       : (D, M)
    Fmax_vec: (M,)
    eta     : float or 0D tensor

    Batched:
    --------
    tau_des : (B, D) or (B, D, 1)
    R       : (B, D, M)
    Fmax_vec: (M,) or (B, M)
    eta     : float, 0D tensor, or (B,) tensor

    Returns
    -------
    F_des : Tensor
        (M,) for unbatched, (B, M) for batched
    meta : dict
        Unbatched:
            {
                "sminGp": 0D tensor,
                "lam_mus": 0D tensor,
                "neg_Fp": int or None
            }
        Batched:
            {
                "sminGp": (B,) tensor,
                "lam_mus": (B,) tensor,
                "neg_Fp": list[Optional[int]]
            }
    """
    tau_des = torch.as_tensor(tau_des)
    R = torch.as_tensor(R, dtype=tau_des.dtype, device=tau_des.device)
    Fmax_vec = torch.as_tensor(Fmax_vec, dtype=tau_des.dtype, device=tau_des.device)

    device, dtype = R.device, R.dtype

    # Unbatched: R: (D,M)
    if R.ndim == 2:
        D, M = R.shape

        if tau_des.ndim == 2 and tau_des.shape[1] == 1:
            tau_vec = tau_des.view(D)
        elif tau_des.ndim == 1:
            tau_vec = tau_des.view(D)
        else:
            raise ValueError(
                f"Unbatched case expects tau_des shape (D,) or (D,1), got {tuple(tau_des.shape)}"
            )

        if Fmax_vec.ndim != 1 or Fmax_vec.shape[0] != M:
            raise ValueError(
                f"Unbatched case expects Fmax_vec shape (M,) with M={M}, got {tuple(Fmax_vec.shape)}"
            )

        eta_t = torch.as_tensor(eta, dtype=dtype, device=device).view(())

        F_des, meta = _solve_single_muscle_forces(tau_vec, R, Fmax_vec, eta_t, P)
        return F_des, meta

    # Batched: R: (B,D,M)
    if R.ndim == 3:
        B, D, M = R.shape

        # tau_des: (B,D) or (B,D,1)
        if tau_des.ndim == 3 and tau_des.shape[2] == 1:
            tau_b = tau_des.view(B, D)
        elif tau_des.ndim == 2 and tau_des.shape[0] == B:
            tau_b = tau_des
        else:
            raise ValueError(
                f"Batched case expects tau_des shape (B,D) or (B,D,1) with B={B}, got {tuple(tau_des.shape)}"
            )

        # Fmax_vec: (M,) or (B,M)
        if Fmax_vec.ndim == 1:
            Fmax_b = Fmax_vec.unsqueeze(0).expand(B, M)  # (B,M)
        elif Fmax_vec.ndim == 2 and Fmax_vec.shape == (B, M):
            Fmax_b = Fmax_vec
        else:
            raise ValueError(
                f"Batched case expects Fmax_vec shape (M,) or (B,M) with B={B}, M={M}, got {tuple(Fmax_vec.shape)}"
            )

        # eta: scalar, 0D tensor, or (B,)
        eta_t = torch.as_tensor(eta, dtype=dtype, device=device)
        if eta_t.ndim == 0:
            eta_b = eta_t.view(1).expand(B)   # (B,)
        elif eta_t.ndim == 1 and eta_t.shape[0] == B:
            eta_b = eta_t
        else:
            raise ValueError(
                f"Batched case expects eta scalar or (B,) with B={B}, got {tuple(eta_t.shape)}"
            )

        F_list: List[Tensor] = []
        smin_list: List[Tensor] = []
        lam_list: List[Tensor] = []
        neg_list: List[Optional[int]] = []

        for b in range(B):
            tau_vec = tau_b[b]          # (D,)
            Rb = R[b]                  # (D,M)
            Fmax_vec_b = Fmax_b[b]     # (M,)
            eta_b_i = eta_b[b].view(())  # scalar

            F_des_b, meta_b = _solve_single_muscle_forces(
                tau_vec, Rb, Fmax_vec_b, eta_b_i, P
            )

            F_list.append(F_des_b)
            smin_list.append(meta_b["sminGp"])
            lam_list.append(meta_b["lam_mus"])
            neg_list.append(meta_b["neg_Fp"])

        F_out = torch.stack(F_list, dim=0)       # (B,M)
        sminGp_b = torch.stack(smin_list, dim=0)  # (B,)
        lam_mus_b = torch.stack(lam_list, dim=0)  # (B,)

        meta = {
            "sminGp": sminGp_b,
            "lam_mus": lam_mus_b,
            "neg_Fp": neg_list,
        }
        return F_out, meta

    # Any other R shape is invalid
    raise ValueError(
        f"R must have shape (D,M) or (B,D,M), got {tuple(R.shape)}"
    )


# --------------------------------------------------------------------------- #
# Tiny Torch-only smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=6, sci_mode=False)

    print("[muscle_guard_torch] Simple smoke test...")

    P = MuscleGuardParams()

    # --- Unbatched test ---
    D, M = 2, 4
    R = torch.tensor(
        [
            [1.0, 0.5, -0.3, 0.2],
            [0.0, 0.8,  0.4, 0.1],
        ]
    )
    tau_des = torch.tensor([0.2, -0.1])
    Fmax = torch.tensor([10.0, 12.0, 8.0, 9.0])
    eta = 1.0

    F_des, meta = solve_muscle_forces(tau_des, R, Fmax, eta, P)
    print("  [unbatched] F_des:", F_des)
    print("  [unbatched] meta.sminGp:", float(meta["sminGp"]))
    print("  [unbatched] meta.lam_mus:", float(meta["lam_mus"]))
    print("  [unbatched] meta.neg_Fp:", meta["neg_Fp"])

    # --- Batched test ---
    B = 3
    R_b = R.unsqueeze(0).expand(B, -1, -1).clone()
    Fmax_b = Fmax  # shared across batch
    tau_b = torch.stack(
        [
            torch.tensor([0.2, -0.1]),
            torch.tensor([-0.1, 0.3]),
            torch.tensor([0.0, 0.0]),
        ],
        dim=0,
    )
    eta_b = torch.tensor([1.0, 0.8, 0.5])

    F_des_b, meta_b = solve_muscle_forces(tau_b, R_b, Fmax_b, eta_b, P)
    print("  [batched] F_des_b:\n", F_des_b)
    print("  [batched] meta_b.sminGp:", meta_b["sminGp"])
    print("  [batched] meta_b.lam_mus:", meta_b["lam_mus"])
    print("  [batched] meta_b.neg_Fp:", meta_b["neg_Fp"])

    print("[muscle_guard_torch] smoke ✓")
