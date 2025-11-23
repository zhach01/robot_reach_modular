# gating_torch.py
# -*- coding: utf-8 -*-
"""
Torch-friendly gating helpers.

Port of utils/gating.py to pure Torch, with the same semantics:

Original:
---------
- blend(a, b, t):  (1 - t) * a + t * b
- clip01(x):       clip x into [0, 1]

Torch version:
--------------
- Accepts scalars or Tensors.
- If any argument is a Tensor, everything is computed as Torch tensors
  on that tensor's device/dtype and the result is a Tensor.
- If all arguments are Python scalars, returns Python floats (for full
  backward compatibility with the original tiny helpers).

Both functions are differentiable when used with Torch tensors.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _get_ref_tensor(*args: Any) -> Tensor | None:
    """
    Return the first Tensor among args, or None if none are Tensors.
    """
    for v in args:
        if isinstance(v, Tensor):
            return v
    return None


def blend(a: Any, b: Any, t: Any):
    """
    Blend between a and b with weight t in [0, 1]:

        blend(a, b, t) = (1 - t) * a + t * b

    Torch behavior:
    ---------------
    - If any of (a, b, t) is a Tensor:
        * All are converted to tensors on that Tensor's device/dtype.
        * Result is a Tensor with broadcasting.
    - If all are scalars:
        * Uses plain Python float arithmetic and returns a float.

    This keeps it convenient for both Torch controllers and small Python
    utilities.
    """
    ref = _get_ref_tensor(a, b, t)

    if ref is not None:
        # Tensor path: keep device/dtype and allow broadcasting
        a_t = torch.as_tensor(a, dtype=ref.dtype, device=ref.device)
        b_t = torch.as_tensor(b, dtype=ref.dtype, device=ref.device)
        t_t = torch.as_tensor(t, dtype=ref.dtype, device=ref.device)
        return (1.0 - t_t) * a_t + t_t * b_t

    # Scalar path: plain floats
    a_f = float(a)
    b_f = float(b)
    t_f = float(t)
    return (1.0 - t_f) * a_f + t_f * b_f


def clip01(x: Any):
    """
    Clip x into [0, 1].

    Torch behavior:
    ---------------
    - If x is a Tensor:
        -> returns a Tensor: clamp(x, 0, 1) (differentiable, GPU-safe)
    - If x is a scalar:
        -> returns a float, using Torch under the hood.
    """
    if isinstance(x, Tensor):
        return torch.clamp(x, 0.0, 1.0)

    # Scalar path
    x_t = torch.tensor(x, dtype=torch.get_default_dtype())
    return float(torch.clamp(x_t, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Tiny smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=6, sci_mode=False)

    print("[gating_torch] Simple smoke test...")

    # Scalar tests
    a, b, t = 0.0, 10.0, 0.3
    print("  [scalar] blend(0,10,0.3) =", blend(a, b, t))
    print("  [scalar] clip01(-0.2) =", clip01(-0.2))
    print("  [scalar] clip01(0.5)  =", clip01(0.5))
    print("  [scalar] clip01(1.5)  =", clip01(1.5))

    # Tensor tests (CPU)
    a_t = torch.tensor([0.0, 1.0])
    b_t = torch.tensor([10.0, -5.0])
    t_t = torch.tensor([0.25, 0.75])

    out_t = blend(a_t, b_t, t_t)
    print("  [tensor] blend([0,1], [10,-5], [0.25,0.75]) =", out_t)

    x_t = torch.tensor([-0.2, 0.3, 1.2])
    print("  [tensor] clip01([-0.2,0.3,1.2]) =", clip01(x_t))

    print("[gating_torch] smoke ✓")
