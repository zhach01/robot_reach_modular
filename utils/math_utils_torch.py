# math_utils_torch.py
# -*- coding: utf-8 -*-
"""
Torch-only linear algebra utilities.

Pure Torch, GPU-safe, differentiable versions of:

- matrix_sqrt_spd(S, eps=1e-12)
- matrix_isqrt_spd(S, eps=1e-12)
- right_pinv_rows(A, eps=1e-8)
- right_pinv_rows_weighted(A, Rw, eps=1e-8)

Original NumPy versions: utils/math_utils.py
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _to_tensor_like(x: Any, like: Tensor) -> Tensor:
    """Convert x to a tensor with same dtype/device as 'like'."""
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


# --------------------------------------------------------------------------- #
# SPD matrix square root and inverse square root (Torch, batched)
# --------------------------------------------------------------------------- #

def matrix_sqrt_spd(S: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Torch version of matrix_sqrt_spd(S, eps).

    Computes S^{1/2} for SPD matrix S via eigen-decomposition:

        S_sym = 0.5 (S + S^T)
        S_sym = V diag(w) V^T
        w_clamped = max(w, eps)
        S_sqrt = V diag(sqrt(w_clamped)) V^T

    Supports:
        - Unbatched: S: (n, n)
        - Batched:   S: (B, n, n)

    Returns:
        S_sqrt with same shape as S.
    """
    S = torch.as_tensor(S)
    Ssym = 0.5 * (S + S.transpose(-1, -2))

    w, V = torch.linalg.eigh(Ssym)          # (..., n), (..., n, n)
    w_clamped = torch.clamp(w, min=eps)
    sqrt_w = torch.sqrt(w_clamped)          # (..., n)

    D = torch.diag_embed(sqrt_w)            # (..., n, n)
    S_sqrt = V @ D @ V.transpose(-1, -2)    # (..., n, n)
    return S_sqrt


def matrix_isqrt_spd(S: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Torch version of matrix_isqrt_spd(S, eps).

    Computes S^{-1/2} for SPD matrix S via eigen-decomposition:

        S_sym = 0.5 (S + S^T)
        S_sym = V diag(w) V^T
        w_clamped = max(w, eps)
        S_isqrt = V diag(1/sqrt(w_clamped)) V^T

    Supports:
        - Unbatched: S: (n, n)
        - Batched:   S: (B, n, n)

    Returns:
        S_isqrt with same shape as S.
    """
    S = torch.as_tensor(S)
    Ssym = 0.5 * (S + S.transpose(-1, -2))

    w, V = torch.linalg.eigh(Ssym)          # (..., n), (..., n, n)
    w_clamped = torch.clamp(w, min=eps)
    inv_sqrt_w = 1.0 / torch.sqrt(w_clamped)

    D = torch.diag_embed(inv_sqrt_w)        # (..., n, n)
    S_isqrt = V @ D @ V.transpose(-1, -2)   # (..., n, n)
    return S_isqrt


# --------------------------------------------------------------------------- #
# Right pseudo-inverse of rows (Torch, batched)
# --------------------------------------------------------------------------- #

def right_pinv_rows(A: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Torch version of right_pinv_rows(A, eps):

        G = A A^T
        pinv_rows(A) = A^T (G + eps I)^{-1}

    Typically used when A has more columns than rows (row-wise map).

    Supports:
        - Unbatched: A: (m, n)   ->  pinv: (n, m)
        - Batched:   A: (B, m, n) -> pinv: (B, n, m)

    All operations are done in Torch and are differentiable.
    """
    A = torch.as_tensor(A)
    device, dtype = A.device, A.dtype

    if A.ndim == 2:
        # Unbatched
        A_b = A.unsqueeze(0)    # (1, m, n)
        batch = False
    elif A.ndim == 3:
        A_b = A                 # (B, m, n)
        batch = True
    else:
        raise ValueError(
            f"right_pinv_rows expects A with shape (m,n) or (B,m,n), got {tuple(A.shape)}"
        )

    B, m, n = A_b.shape

    # G = A A^T
    G = A_b @ A_b.transpose(-1, -2)              # (B, m, m)

    # Regularized: G + eps I
    I = torch.eye(m, dtype=dtype, device=device)
    I_b = I.unsqueeze(0).expand(B, m, m)         # (B, m, m)
    eps_t = _to_tensor_like(eps, A_b)
    G_reg = G + eps_t.view(-1, 1, 1) * I_b       # (B, m, m)

    # Solve for inv(G_reg) via G_reg X = I
    invG = torch.linalg.solve(G_reg, I_b)        # (B, m, m)

    # pinv_rows(A) = A^T inv(G_reg)
    pinv_b = A_b.transpose(-1, -2) @ invG        # (B, n, m)

    if not batch:
        return pinv_b[0]
    return pinv_b


def right_pinv_rows_weighted(A: Tensor, Rw: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Torch version of right_pinv_rows_weighted(A, Rw, eps):

        G = A Rw A^T
        pinv_rows_w(A) = Rw A^T (G + eps I)^{-1}

    Rw is a weight matrix (typically SPD).

    Supports:
        - Unbatched:
            A: (m, n),  Rw: (n, n)  ->  pinv: (n, m)
        - Batched:
            A: (B, m, n),
            Rw: (n, n) or (B, n, n) -> pinv: (B, n, m)

    All operations are in Torch and differentiable (given a fixed active set
    if you later use it inside NNLS-type problems).
    """
    A = torch.as_tensor(A)
    device, dtype = A.device, A.dtype
    Rw = torch.as_tensor(Rw, dtype=dtype, device=device)

    if A.ndim == 2:
        # Unbatched
        m, n = A.shape
        if Rw.shape != (n, n):
            raise ValueError(
                f"Unbatched case expects Rw shape (n,n) with n={n}, got {tuple(Rw.shape)}"
            )
        A_b = A.unsqueeze(0)          # (1, m, n)
        Rw_b = Rw.unsqueeze(0)        # (1, n, n)
        batch = False

    elif A.ndim == 3:
        # Batched
        B, m, n = A.shape
        A_b = A

        if Rw.ndim == 2:
            if Rw.shape != (n, n):
                raise ValueError(
                    f"Batched case with shared Rw expects shape (n,n) with n={n}, got {tuple(Rw.shape)}"
                )
            Rw_b = Rw.unsqueeze(0).expand(B, n, n)    # (B, n, n)
        elif Rw.ndim == 3:
            if Rw.shape[0] != B or Rw.shape[1:] != (n, n):
                raise ValueError(
                    f"Batched case expects Rw shape (B,n,n) with B={B}, n={n}, got {tuple(Rw.shape)}"
                )
            Rw_b = Rw
        else:
            raise ValueError(
                f"Rw must have shape (n,n) or (B,n,n), got {tuple(Rw.shape)}"
            )
        batch = True

    else:
        raise ValueError(
            f"right_pinv_rows_weighted expects A shape (m,n) or (B,m,n), got {tuple(A.shape)}"
        )

    B = A_b.shape[0]
    m = A_b.shape[1]
    n = A_b.shape[2]

    # G = A Rw A^T
    # (B,m,n) @ (B,n,n) -> (B,m,n); @ (B,n,m) -> (B,m,m)
    G = A_b @ Rw_b @ A_b.transpose(-1, -2)       # (B, m, m)

    # Regularized: G + eps I
    I = torch.eye(m, dtype=dtype, device=device)
    I_b = I.unsqueeze(0).expand(B, m, m)         # (B, m, m)
    eps_t = _to_tensor_like(eps, A_b)
    G_reg = G + eps_t.view(-1, 1, 1) * I_b       # (B, m, m)

    # inv(G_reg) via solve
    invG = torch.linalg.solve(G_reg, I_b)        # (B, m, m)

    # pinv_rows_w(A) = Rw A^T inv(G_reg)
    pinv_b = Rw_b @ A_b.transpose(-1, -2) @ invG  # (B, n, m)

    if not batch:
        return pinv_b[0]
    return pinv_b


# --------------------------------------------------------------------------- #
# Tiny Torch-only smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=6, sci_mode=False)

    print("[math_utils_torch] Simple smoke test...")

    # --- SPD sqrt / isqrt test (unbatched) ---
    S = torch.tensor([[2.0, 0.3],
                      [0.3, 1.0]])
    S_sqrt = matrix_sqrt_spd(S)
    S_isqrt = matrix_isqrt_spd(S)

    recon = S_sqrt @ S_sqrt.transpose(-1, -2)
    recon_inv = S_isqrt @ S_isqrt.transpose(-1, -2)

    print("  [unbatched] S:\n", S)
    print("  [unbatched] S_sqrt S_sqrt^T:\n", recon)
    print("  [unbatched] ||S - recon||:", torch.norm(S - recon).item())
    print("  [unbatched] S_isqrt S_isqrt^T:\n", recon_inv)

    # --- SPD sqrt / isqrt test (batched) ---
    S_b = torch.stack([S, 0.5 * S], dim=0)  # (2,2,2)
    S_sqrt_b = matrix_sqrt_spd(S_b)
    S_isqrt_b = matrix_isqrt_spd(S_b)
    recon_b = S_sqrt_b @ S_sqrt_b.transpose(-1, -2)

    print("  [batched] S_sqrt_b shape:", S_sqrt_b.shape)
    print("  [batched] ||S_b - recon_b|| per batch:",
          torch.norm(S_b - recon_b, dim=(1, 2)))

    # --- right_pinv_rows test ---
    A = torch.tensor([[1.0, 0.5, -0.3],
                      [0.0, 0.8,  0.4]])  # (m=2, n=3)
    pinv = right_pinv_rows(A)
    print("  [unbatched] right_pinv_rows(A) shape:", pinv.shape)
    # Check approximate left identity on rows: A pinv ≈ I_2
    I2_approx = A @ pinv
    print("  [unbatched] A @ pinv ≈ I:\n", I2_approx)

    # --- right_pinv_rows_weighted test ---
    Rw = torch.tensor([[1.0, 0.2, 0.0],
                       [0.2, 1.5, 0.1],
                       [0.0, 0.1, 2.0]])  # (n=3,n=3) SPD-ish
    pinv_w = right_pinv_rows_weighted(A, Rw)
    print("  [unbatched] right_pinv_rows_weighted(A,Rw) shape:", pinv_w.shape)

    # Batched version
    A_b = A.unsqueeze(0).expand(4, -1, -1).clone()  # (4,2,3)
    Rw_b = Rw  # shared weight
    pinv_b = right_pinv_rows(A_b)
    pinv_w_b = right_pinv_rows_weighted(A_b, Rw_b)

    print("  [batched] right_pinv_rows(A_b) shape:", pinv_b.shape)
    print("  [batched] right_pinv_rows_weighted(A_b,Rw_b) shape:", pinv_w_b.shape)

    print("[math_utils_torch] smoke ✓")
