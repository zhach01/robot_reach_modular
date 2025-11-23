# dynamics_guard_torch.py
# -*- coding: utf-8 -*-
"""
Torch-only operational-space guard + gate.

Port of utils/dynamics_guard.py (NumPy) to pure Torch:
- DynGuardParams: hyperparameters for operational-space guard.
- op_space_guard_and_gate(S, xd_d, xdd_d, P):
    * Takes SPD 2x2 matrix S (or batch of them).
    * Computes:
        - sminS = smallest eigenvalue of S
        - detS  = determinant of S
        - eta, eta^gate_pow   (gate based on sminS)
        - alpha_S             (dynamic task scaling factor)
        - lam_os              (adaptive regularization)
        - G_OS = S + (lam_os + eps) I
        - Lambda = G_OS^{-1}
      and returns (Lambda, lam_os, eta, eta2, xd_sc, xdd_sc, meta_dict)

Supports:
- S: (2, 2) or (B, 2, 2)
- xd_d, xdd_d: (2,) or (B, 2)
All computations are Torch-only and differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import Tensor


@dataclass
class DynGuardParams:
    sigma_thresh_S: float = 1e-3
    vol_thresh: float = 1e-6
    c_s: float = 1e-3
    c_v: float = 1e-3
    k_scale_dyn: float = 1e-3
    lam_boost: float = 1e-3
    gate_pow: float = 2.0
    eps: float = 1e-9
    lam_os_max: float = 1e-1  # upper bound safety


def _to_tensor(x: Any, like: Tensor) -> Tensor:
    """Convert x to a tensor on the same device/dtype as 'like'."""
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


def op_space_guard_and_gate(
    S: Tensor,
    xd_d: Tensor,
    xdd_d: Tensor,
    P: DynGuardParams,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    """
    Torch version of op_space_guard_and_gate.

    Parameters
    ----------
    S : Tensor
        Operational-space inertia matrix (SPD 2x2), shape:
          - (2, 2)       unbatched
          - (B, 2, 2)    batched
    xd_d : Tensor
        Desired task-space velocity, shape:
          - (2,) or (B, 2)
    xdd_d : Tensor
        Desired task-space acceleration, shape:
          - (2,) or (B, 2)
    P : DynGuardParams
        Guard parameters.

    Returns
    -------
    Lambda : Tensor
        Inverse of regularized G_OS: shape (2,2) or (B,2,2)
    lam_os : Tensor
        Regularization scalar(s), shape () or (B,)
    eta : Tensor
        Gate factor(s), in [0,1].
    eta2 : Tensor
        eta**gate_pow, shape () or (B,)
    xd_sc : Tensor
        Scaled desired velocity, shape (2,) or (B,2)
    xdd_sc : Tensor
        Scaled desired acceleration, shape (2,) or (B,2)
    meta : dict
        {
            "sminS":  Tensor scalar or (B,),
            "detS":   Tensor scalar or (B,),
            "alpha_S": Tensor scalar or (B,)
        }
    """
    S = torch.as_tensor(S)
    device, dtype = S.device, S.dtype

    xd_d = torch.as_tensor(xd_d, dtype=dtype, device=device)
    xdd_d = torch.as_tensor(xdd_d, dtype=dtype, device=device)

    # Handle unbatched vs batched
    if S.ndim == 2:
        # (2,2) -> (1,2,2)
        S_b = S.unsqueeze(0)
        batch = False
    elif S.ndim == 3 and S.shape[1:] == (2, 2):
        S_b = S
        batch = True
    else:
        raise ValueError(
            f"S must have shape (2,2) or (B,2,2), got {tuple(S.shape)}"
        )

    B = S_b.shape[0]

    # Ensure xd_d, xdd_d are (B,2)
    if xd_d.ndim == 1:
        xd_b = xd_d.unsqueeze(0).expand(B, -1)  # (B,2)
    elif xd_d.ndim == 2:
        if xd_d.shape[0] != B:
            raise ValueError(
                f"xd_d batch dim {xd_d.shape[0]} != S batch dim {B}"
            )
        xd_b = xd_d
    else:
        raise ValueError(f"xd_d must have shape (2,) or (B,2), got {tuple(xd_d.shape)}")

    if xdd_d.ndim == 1:
        xdd_b = xdd_d.unsqueeze(0).expand(B, -1)  # (B,2)
    elif xdd_d.ndim == 2:
        if xdd_d.shape[0] != B:
            raise ValueError(
                f"xdd_d batch dim {xdd_d.shape[0]} != S batch dim {B}"
            )
        xdd_b = xdd_d
    else:
        raise ValueError(f"xdd_d must have shape (2,) or (B,2), got {tuple(xdd_d.shape)}")

    # Convert scalar params to tensors
    sigma_thresh_S = _to_tensor(P.sigma_thresh_S, S)
    vol_thresh = _to_tensor(P.vol_thresh, S)
    c_s = _to_tensor(P.c_s, S)
    c_v = _to_tensor(P.c_v, S)
    k_scale_dyn = _to_tensor(P.k_scale_dyn, S)
    lam_boost = _to_tensor(P.lam_boost, S)
    eps = _to_tensor(P.eps, S)
    lam_os_max = _to_tensor(P.lam_os_max, S)

    # --- Eigenvalues: S is 2x2 SPD ---
    evals = torch.linalg.eigvalsh(S_b)       # (B,2)
    evals_clamped = torch.clamp(evals, min=0.0)
    # smallest eigenvalue (per batch)
    sminS = evals_clamped[:, 0]              # (B,)
    # determinant from eigenvalues
    detS = evals_clamped[:, 0] * evals_clamped[:, 1]  # (B,)

    # --- gate (eta) from smin(S) ---
    # eta = clip(sminS / sigma_thresh_S, 0, 1)
    sigma_thresh_safe = torch.clamp(sigma_thresh_S, min=_to_tensor(1e-12, S))
    eta = torch.clamp(sminS / sigma_thresh_safe, min=0.0, max=1.0)  # (B,)
    eta2 = eta**P.gate_pow

    # --- dynamic task scaling ---
    # alpha_S = min(1, sminS / (sminS + k_scale_dyn))
    alpha_S = torch.minimum(
        torch.ones_like(sminS),
        sminS / (sminS + k_scale_dyn),
    )  # (B,)

    # Broadcast alpha_S over the (B,2) velocities/accelerations
    alpha_exp = alpha_S.unsqueeze(-1)  # (B,1)
    xd_sc_b = alpha_exp * xd_b        # (B,2)
    xdd_sc_b = alpha_exp * xdd_b      # (B,2)

    # --- adaptive Λ regularization ---
    # lam_os = c_s * (sigma_thresh_S / (sminS + 1e-12)) +
    #          c_v * (vol_thresh / (detS + 1e-18))
    smin_safe = sminS + _to_tensor(1e-12, S)
    det_safe = detS + _to_tensor(1e-18, S)

    lam_os = c_s * (sigma_thresh_S / smin_safe) + c_v * (vol_thresh / det_safe)  # (B,)
    lam_os = torch.clamp(lam_os, min=0.0, max=lam_os_max)
    lam_os = lam_os + lam_boost * (1.0 - eta)                                    # (B,)

    # --- regularized G_OS and Lambda ---
    # G_OS = S + (lam_os + eps) * I
    I2 = torch.eye(2, dtype=dtype, device=device)
    I2_b = I2.unsqueeze(0).expand(B, 2, 2)            # (B,2,2)

    lam_tot = lam_os + eps                            # (B,)
    lam_tot_exp = lam_tot.view(B, 1, 1)               # (B,1,1)

    G_OS = S_b + lam_tot_exp * I2_b                   # (B,2,2)

    # Lambda = G_OS^{-1}
    Lambda_b = torch.linalg.solve(G_OS, I2_b)         # (B,2,2)

    # --- Pack outputs (unbatched vs batched) ---
    if not batch:
        Lambda = Lambda_b[0]
        lam_os_out = lam_os[0]
        eta_out = eta[0]
        eta2_out = eta2[0]
        xd_sc = xd_sc_b[0]
        xdd_sc = xdd_sc_b[0]
        sminS_out = sminS[0]
        detS_out = detS[0]
        alpha_S_out = alpha_S[0]
    else:
        Lambda = Lambda_b
        lam_os_out = lam_os
        eta_out = eta
        eta2_out = eta2
        xd_sc = xd_sc_b
        xdd_sc = xdd_sc_b
        sminS_out = sminS
        detS_out = detS
        alpha_S_out = alpha_S

    meta = {
        "sminS": sminS_out,
        "detS": detS_out,
        "alpha_S": alpha_S_out,
    }

    return Lambda, lam_os_out, eta_out, eta2_out, xd_sc, xdd_sc, meta


# --------------------------------------------------------------------------- #
# Tiny Torch-only smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=6, sci_mode=False)

    print("[dynamics_guard_torch] Simple smoke test...")

    P = DynGuardParams()

    # --- Unbatched test ---
    S = torch.tensor([[2.0, 0.3],
                      [0.3, 1.0]])
    xd_d = torch.tensor([0.1, -0.2])
    xdd_d = torch.tensor([0.0, 0.05])

    Lambda, lam_os, eta, eta2, xd_sc, xdd_sc, meta = op_space_guard_and_gate(
        S, xd_d, xdd_d, P
    )

    print("  [unbatched] Lambda:\n", Lambda)
    print("  [unbatched] lam_os:", lam_os.item())
    print("  [unbatched] eta:", eta.item(), "eta2:", eta2.item())
    print("  [unbatched] xd_sc:", xd_sc)
    print("  [unbatched] xdd_sc:", xdd_sc)
    print("  [unbatched] meta:", {k: float(v) for k, v in meta.items()})

    # --- Batched test ---
    S_b = torch.stack([S, 0.5 * S], dim=0)        # (2,2,2)
    xd_b = torch.stack([xd_d, -xd_d], dim=0)      # (2,2)
    xdd_b = torch.stack([xdd_d, -xdd_d], dim=0)   # (2,2)

    Lambda_b, lam_os_b, eta_b, eta2_b, xd_sc_b, xdd_sc_b, meta_b = op_space_guard_and_gate(
        S_b, xd_b, xdd_b, P
    )

    print("  [batched] Lambda_b shape:", Lambda_b.shape)
    print("  [batched] lam_os_b:", lam_os_b)
    print("  [batched] eta_b:", eta_b)
    print("  [batched] xd_sc_b:", xd_sc_b)
    print("  [batched] xdd_sc_b:", xdd_sc_b)
    print("  [batched] meta_b.sminS:", meta_b["sminS"])
    print("  [batched] meta_b.detS:", meta_b["detS"])
    print("  [batched] meta_b.alpha_S:", meta_b["alpha_S"])

    print("[dynamics_guard_torch] smoke ✓")
