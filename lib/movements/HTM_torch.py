# HTM_torch.py
# Pure-PyTorch homogeneous transforms and cross-product matrix utilities.

import sys
import torch
from torch import Tensor
from typing import Union

# keep parent access identical to original intent
sys.path.append(sys.path[0].replace(r"/lib/movements", r""))

# Small numeric identity cached (CPU) – use .to(device) when needed
_ID4 = torch.eye(4, dtype=torch.get_default_dtype())


def _as_tensor(x: Union[Tensor, float, int]) -> Tensor:
    """
    Convert scalar or Tensor to Tensor, preserving dtype when possible.

    - If x is a Tensor: returned as-is.
    - If x is a number: converted with torch.get_default_dtype().
    """
    if isinstance(x, Tensor):
        return x
    return torch.as_tensor(x, dtype=torch.get_default_dtype())


def tx(x: Union[Tensor, float, int] = 0.0, symbolic: bool = False) -> Tensor:
    """
    Translation on x axis (PyTorch version, numeric only).

    Args:
        x: translation amount (scalar or Tensor, shape [...]).
        symbolic: kept for API compatibility; must remain False.

    Returns:
        H: homogeneous transform of shape [..., 4, 4]
    """
    if symbolic:
        raise NotImplementedError("Symbolic mode is not supported in the PyTorch version.")

    x = _as_tensor(x)
    device, dtype = x.device, x.dtype
    eye = _ID4.to(device=device, dtype=dtype)
    H = eye.expand(x.shape + (4, 4)).clone()
    H[..., 0, 3] = x
    return H


def ty(y: Union[Tensor, float, int] = 0.0, symbolic: bool = False) -> Tensor:
    """
    Translation on y axis (PyTorch version, numeric only).

    Args:
        y: translation amount (scalar or Tensor, shape [...]).
        symbolic: kept for API compatibility; must remain False.

    Returns:
        H: homogeneous transform of shape [..., 4, 4]
    """
    if symbolic:
        raise NotImplementedError("Symbolic mode is not supported in the PyTorch version.")

    y = _as_tensor(y)
    device, dtype = y.device, y.dtype
    eye = _ID4.to(device=device, dtype=dtype)
    H = eye.expand(y.shape + (4, 4)).clone()
    H[..., 1, 3] = y
    return H


def tz(z: Union[Tensor, float, int] = 0.0, symbolic: bool = False) -> Tensor:
    """
    Translation on z axis (PyTorch version, numeric only).

    Args:
        z: translation amount (scalar or Tensor, shape [...]).
        symbolic: kept for API compatibility; must remain False.

    Returns:
        H: homogeneous transform of shape [..., 4, 4]
    """
    if symbolic:
        raise NotImplementedError("Symbolic mode is not supported in the PyTorch version.")

    z = _as_tensor(z)
    device, dtype = z.device, z.dtype
    eye = _ID4.to(device=device, dtype=dtype)
    H = eye.expand(z.shape + (4, 4)).clone()
    H[..., 2, 3] = z
    return H


def rx(x: Union[Tensor, float, int] = 0.0, symbolic: bool = False) -> Tensor:
    """
    Rotation about x axis (PyTorch version, numeric only).

    Args:
        x: angle in radians (scalar or Tensor, shape [...]).
        symbolic: kept for API compatibility; must remain False.

    Returns:
        H: homogeneous transform of shape [..., 4, 4]
    """
    if symbolic:
        raise NotImplementedError("Symbolic mode is not supported in the PyTorch version.")

    x = _as_tensor(x)
    device, dtype = x.device, x.dtype
    c = torch.cos(x)
    s = torch.sin(x)

    shape = x.shape + (4, 4)
    H = torch.zeros(shape, device=device, dtype=dtype)
    H[..., 0, 0] = 1.0
    H[..., 1, 1] = c
    H[..., 1, 2] = -s
    H[..., 2, 1] = s
    H[..., 2, 2] = c
    H[..., 3, 3] = 1.0
    return H


def ry(y: Union[Tensor, float, int] = 0.0, symbolic: bool = False) -> Tensor:
    """
    Rotation about y axis (PyTorch version, numeric only).

    Args:
        y: angle in radians (scalar or Tensor, shape [...]).
        symbolic: kept for API compatibility; must remain False.

    Returns:
        H: homogeneous transform of shape [..., 4, 4]
    """
    if symbolic:
        raise NotImplementedError("Symbolic mode is not supported in the PyTorch version.")

    y = _as_tensor(y)
    device, dtype = y.device, y.dtype
    c = torch.cos(y)
    s = torch.sin(y)

    shape = y.shape + (4, 4)
    H = torch.zeros(shape, device=device, dtype=dtype)
    H[..., 0, 0] = c
    H[..., 0, 2] = s
    H[..., 1, 1] = 1.0
    H[..., 2, 0] = -s
    H[..., 2, 2] = c
    H[..., 3, 3] = 1.0
    return H


def rz(z: Union[Tensor, float, int] = 0.0, symbolic: bool = False) -> Tensor:
    """
    Rotation about z axis (PyTorch version, numeric only).

    Args:
        z: angle in radians (scalar or Tensor, shape [...]).
        symbolic: kept for API compatibility; must remain False.

    Returns:
        H: homogeneous transform of shape [..., 4, 4]
    """
    if symbolic:
        raise NotImplementedError("Symbolic mode is not supported in the PyTorch version.")

    z = _as_tensor(z)
    device, dtype = z.device, z.dtype
    c = torch.cos(z)
    s = torch.sin(z)

    shape = z.shape + (4, 4)
    H = torch.zeros(shape, device=device, dtype=dtype)
    H[..., 0, 0] = c
    H[..., 0, 1] = -s
    H[..., 1, 0] = s
    H[..., 1, 1] = c
    H[..., 2, 2] = 1.0
    H[..., 3, 3] = 1.0
    return H


def crossMatrix(r: Union[Tensor, float, int], symbolic: bool = False) -> Tensor:
    """
    Cross-product operator for 3D vectors.

    Args:
        r: 3D vector(s), Tensor of shape [..., 3] or something convertible.
        symbolic: kept for API compatibility; must remain False.

    Returns:
        C: cross-product matrix of shape [..., 3, 3] such that
           (C @ v) == torch.cross(r, v, dim=-1)
    """
    if symbolic:
        raise NotImplementedError("Symbolic mode is not supported in the PyTorch version.")

    r = _as_tensor(r)
    if r.shape[-1] != 3:
        raise ValueError(f"crossMatrix expects r[..., 3], got shape {r.shape}")

    device, dtype = r.device, r.dtype
    rx = r[..., 0]
    ry = r[..., 1]
    rz = r[..., 2]

    shape = r.shape[:-1] + (3, 3)
    C = torch.zeros(shape, device=device, dtype=dtype)
    C[..., 0, 1] = -rz
    C[..., 0, 2] = ry
    C[..., 1, 0] = rz
    C[..., 1, 2] = -rx
    C[..., 2, 0] = -ry
    C[..., 2, 1] = rx
    return C


if __name__ == "__main__":
    # Simple numeric tests (CPU)
    Hx = tx(0.5)
    print("Tx(0.5) =\n", Hx)

    angle = torch.tensor(0.1)
    Rz = rz(angle)
    print("Rz(0.1) =\n", Rz)

    v = torch.tensor([1.0, 2.0, 3.0])
    C = crossMatrix(v)
    w = torch.tensor([0.5, -1.0, 0.0])
    print("C @ w = ", C @ w)
    print("torch.cross(v, w) = ", torch.cross(v, w, dim=-1))
