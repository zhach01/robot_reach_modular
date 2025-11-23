# linear_utils_torch.py
# -*- coding: utf-8 -*-
"""
Torch-only small linear utilities.

Currently implements:

- nnls_small_active_set(tau_des, R, iters=12)

  Solve the non-negative least squares problem:

      min_f || R f + tau_des ||_2   subject to f >= 0

  using a tiny active-set method suitable for small M (~6–12 muscles).

  Torch-only, GPU-safe, with support for:
    - Unbatched:
        tau_des: (D,)          R: (D, M)      -> f: (M,)
    - Batched:
        tau_des: (B, D)        R: (B, D, M)   -> f: (B, M)

  All math stays in Torch; clipping and active-set logic make it
  piecewise-smooth (not globally smooth), but it is still autograd-friendly
  along the chosen active set.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _to_tensor_like(x: Any, like: Tensor) -> Tensor:
    """Convert x to a tensor with same dtype/device as 'like'."""
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


# --------------------------------------------------------------------------- #
# Core unbatched NNLS active-set solver (Torch)
# --------------------------------------------------------------------------- #

def _nnls_single_active_set(
    tau_des: Tensor,
    R: Tensor,
    iters: int = 12,
    eps_reg: float = 1e-12,
) -> Tensor:
    """
    Unbatched active-set NNLS:

        min_f || R f + tau_des ||  s.t. f >= 0

    Parameters
    ----------
    tau_des : Tensor, shape (D,)
    R       : Tensor, shape (D, M)
    iters   : int
        Max number of active-set iterations.
    eps_reg : float
        Small Tikhonov damping on normal equations.

    Returns
    -------
    f : Tensor, shape (M,)
    """
    tau_des = tau_des.view(-1)      # (D,)
    D, M = R.shape
    device, dtype = R.device, R.dtype

    # active mask: all muscles initially active
    active = torch.ones(M, dtype=torch.bool, device=device)

    # b = -tau_des so we solve R f ≈ -tau_des
    b = -tau_des  # (D,)

    for _ in range(iters):
        if not torch.any(active):
            return torch.zeros(M, dtype=dtype, device=device)

        idx = active.nonzero(as_tuple=False).view(-1)      # (Ma,)
        Ra = R[:, idx]                                     # (D, Ma)

        # Normal equations: (Ra^T Ra) fa = Ra^T (-tau_des)
        AtA = Ra.transpose(-1, -2) @ Ra                   # (Ma, Ma)
        rhs = Ra.transpose(-1, -2) @ b                    # (Ma,)

        # Regularize for numerical safety
        if AtA.numel() > 0:
            eye = torch.eye(AtA.shape[0], dtype=dtype, device=device)
            AtA = AtA + eps_reg * eye

        # Solve for fa
        if AtA.numel() == 0:
            fa = torch.zeros(0, dtype=dtype, device=device)
        else:
            # solve (AtA) fa = rhs
            fa = torch.linalg.solve(AtA, rhs.unsqueeze(-1)).squeeze(-1)  # (Ma,)

        # Reassemble full f
        f = torch.zeros(M, dtype=dtype, device=device)
        f[idx] = fa

        # Negative components violate f >= 0
        neg = f < 0
        if not torch.any(neg):
            # Already feasible
            return f

        # Remove negative components from active set
        active = active & (~neg)

    # Final recomputation with last active set, then clipping
    if not torch.any(active):
        return torch.zeros(M, dtype=dtype, device=device)

    idx = active.nonzero(as_tuple=False).view(-1)
    Ra = R[:, idx]
    AtA = Ra.transpose(-1, -2) @ Ra
    rhs = Ra.transpose(-1, -2) @ b

    if AtA.numel() > 0:
        eye = torch.eye(AtA.shape[0], dtype=dtype, device=device)
        AtA = AtA + eps_reg * eye

    if AtA.numel() == 0:
        fa = torch.zeros(0, dtype=dtype, device=device)
    else:
        fa = torch.linalg.solve(AtA, rhs.unsqueeze(-1)).squeeze(-1)

    f = torch.zeros(M, dtype=dtype, device=device)
    f[idx] = fa
    return torch.clamp(f, min=0.0)


# --------------------------------------------------------------------------- #
# Public API: nnls_small_active_set (Torch, batched + unbatched)
# --------------------------------------------------------------------------- #

def nnls_small_active_set(
    tau_des: Tensor,
    R: Tensor,
    iters: int = 12,
) -> Tensor:
    """
    Torch version of nnls_small_active_set:

        min_f || R f + tau_des ||_2  subject to f >= 0

    Supports:
        - Unbatched:
            tau_des: (D,)     R: (D, M)       -> f: (M,)
        - Batched:
            tau_des: (B, D)   R: (B, D, M)    -> f: (B, M)

    Parameters
    ----------
    tau_des : Tensor
        Desired joint torques with sign convention matching R.
    R : Tensor
        Mapping from muscle forces to joint torques.
    iters : int
        Max number of active-set iterations.

    Returns
    -------
    f : Tensor
        Nonnegative muscle forces, shape:
            (M,) for unbatched
            (B, M) for batched
    """
    tau_des = torch.as_tensor(tau_des)
    R = torch.as_tensor(R, dtype=tau_des.dtype, device=tau_des.device)

    # Unbatched: R: (D, M), tau_des: (D,) or (D,1)
    if R.ndim == 2:
        if tau_des.ndim == 2 and tau_des.shape[1] == 1:
            tau_vec = tau_des.view(-1)
        elif tau_des.ndim == 1:
            tau_vec = tau_des
        else:
            raise ValueError(
                f"Unbatched case expects tau_des shape (D,) or (D,1), got {tuple(tau_des.shape)}"
            )
        return _nnls_single_active_set(tau_vec, R, iters=iters)

    # Batched: R: (B, D, M), tau_des: (B, D) or (B, D,1)
    if R.ndim == 3:
        B, D, M = R.shape
        if tau_des.ndim == 3 and tau_des.shape[2] == 1:
            tau_b = tau_des.view(B, D)
        elif tau_des.ndim == 2 and tau_des.shape[0] == B:
            tau_b = tau_des
        else:
            raise ValueError(
                f"Batched case expects tau_des shape (B,D) or (B,D,1) with B={B}, got {tuple(tau_des.shape)}"
            )

        f_list = []
        for b in range(B):
            tb = tau_b[b]      # (D,)
            Rb = R[b]          # (D,M)
            fb = _nnls_single_active_set(tb, Rb, iters=iters)
            f_list.append(fb)

        return torch.stack(f_list, dim=0)  # (B, M)

    raise ValueError(
        f"R must have shape (D,M) or (B,D,M), got {tuple(R.shape)}"
    )


# --------------------------------------------------------------------------- #
# Tiny Torch-only smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=6, sci_mode=False)

    print("[linear_utils_torch] Simple smoke test...")

    # Unbatched small test
    D, M = 2, 4
    R = torch.tensor([[1.0, 0.5, -0.3, 0.2],
                      [0.0, 0.8,  0.4, 0.1]])
    tau_des = torch.tensor([0.5, -0.2])  # arbitrary

    f = nnls_small_active_set(tau_des, R, iters=12)
    print("  [unbatched] f:", f)
    print("  [unbatched] f >= 0?", bool(torch.all(f >= 0)))
    print("  [unbatched] R f + tau:", R @ f + tau_des)

    # Batched small test
    B = 3
    R_b = R.unsqueeze(0).expand(B, -1, -1).clone()
    tau_b = torch.stack([
        torch.tensor([0.5, -0.2]),
        torch.tensor([-0.1, 0.3]),
        torch.tensor([0.0, 0.0]),
    ], dim=0)

    f_b = nnls_small_active_set(tau_b, R_b, iters=12)
    print("  [batched] f_b:\n", f_b)
    print("  [batched] all f_b >= 0?", bool(torch.all(f_b >= 0)))
    print("  [batched] R_b f_b + tau_b:\n", torch.einsum("bdm,bm->bd", R_b, f_b) + tau_b)

    print("[linear_utils_torch] smoke ✓")
