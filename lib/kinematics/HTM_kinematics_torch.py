# HTM_kinematics_torch.py
"""
Pure-PyTorch kinematics utilities based on DH parameters.

This module is a Torch-only rewrite of the previous mixed SymPy/NumPy/Torch
`lib/kinematics/HTM.py`. It provides:

- forwardHTM(robot):          list of homogeneous transforms for each frame
- forwardCOMHTM(robot):       list of COM transforms
- geometricJacobian(robot):   6×n geometric Jacobian at EE
- geometricJacobianDerivative(robot): 6×n J̇ at EE (batchable)
- geometricJacobianCOM(robot, COM):   6×n geometric J at a COM
- geometricJacobianDerivativeCOM(robot, COM): 6×n J̇ at COM (batchable)
- analyticJacobian(robot):    6×n analytic Jacobian using axis-angle
- analyticJacobianCOM(robot, COM): 6×n analytic Jacobian at COM
- inverseHTM(robot, q0, Hd, K, jacobian="geometric"): iterative IK (Torch)

Requirements on `robot`:
- Attributes:
    - q:  (n, 1) or (B, n, 1) joint positions (torch.Tensor)
    - qd: (n, 1) or (B, n, 1) joint velocities (torch.Tensor)
    - dh:  (n, 4) or (B, n, 4) DH parameters [θ, d, a, α] (torch.Tensor)
    - dhCOM (optional): same shape as dh for COM frames
    - dh_convention: "standard" or "modified" (str)
- Methods:
    - denavitHartenberg()
    - denavitHartenbergCOM()
    - where_is_joint(j) -> (row, col)
    - where_is_com(k)   -> (row, col)

All computations are done in Torch and are autograd-friendly.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional

import torch
from torch import Tensor
from torch.autograd.functional import jacobian as torch_jacobian

_DEBUG_HTM = os.environ.get("HTM_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    if _DEBUG_HTM:
        print("[HTM_TORCH]", *args, **kwargs)


# --- import primitive homogeneous transforms (pure Torch) ---
# We assume you have the canonical HTM_torch.py generated previously
# providing: tx, ty, tz, rx, ry, rz, crossMatrix
try:
    # preferred project layout
    from lib.movements import HTM_torch as HTM
except Exception:
    try:
        import lib.movements.HTM_torch as HTM  # fallback: same folder / PYTHONPATH
    except Exception as e:
        raise ImportError(
            "Cannot import HTM_torch. Make sure the pure-Torch primitive HTM "
            "module (with tx, ty, tz, rx, ry, rz, crossMatrix) is available."
        ) from e


# ---------------- helpers ----------------
def _q(robot) -> Tensor:
    """Return robot joint positions as Tensor."""
    if hasattr(robot, "q"):
        return robot.q
    raise AttributeError("robot must expose attribute 'q' as a torch.Tensor")


def _qd(robot) -> Tensor:
    """Return robot joint velocities as Tensor."""
    if hasattr(robot, "qd"):
        return robot.qd
    raise AttributeError("robot must expose attribute 'qd' as a torch.Tensor")


def _dh_table(robot, use_com: bool = False) -> Tensor:
    """Return DH or DH_COM table as Tensor."""
    name = "dhCOM" if use_com else "dh"
    tbl = getattr(robot, name, None)
    if tbl is None:
        # call robot to build DH
        if use_com:
            if not hasattr(robot, "denavitHartenbergCOM"):
                raise AttributeError(
                    "robot must implement denavitHartenbergCOM() to populate dhCOM"
                )
            robot.denavitHartenbergCOM()
        else:
            if not hasattr(robot, "denavitHartenberg"):
                raise AttributeError(
                    "robot must implement denavitHartenberg() to populate dh"
                )
            robot.denavitHartenberg()
        tbl = getattr(robot, name, None)
    if not isinstance(tbl, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tbl)}")
    return tbl


def _eye4_like(like: Optional[Tensor] = None, batch_shape: Optional[Tuple[int, ...]] = None) -> Tensor:
    """Return a 4×4 identity matrix, optionally batched and matching dtype/device."""
    if like is None:
        eye = torch.eye(4)
    else:
        eye = torch.eye(4, dtype=like.dtype, device=like.device)
    if batch_shape is None or len(batch_shape) == 0:
        return eye
    return eye.expand(batch_shape + (4, 4))


def _step_T(theta: Tensor, d: Tensor, a: Tensor, alpha: Tensor, conv: str) -> Tensor:
    """
    Single DH step transform for Torch backend.

    Inputs may be scalar tensors or batched with same leading shape.
    Returns transform with corresponding batch shape.
    """
    if conv not in ("standard", "modified"):
        raise ValueError(f"Unknown DH convention '{conv}' (expected 'standard' or 'modified').")

    if conv == "standard":
        # ^i-1 T_i = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
        return (
            HTM.rz(theta)
            @ HTM.tz(d)
            @ HTM.tx(a)
            @ HTM.rx(alpha)
        )
    else:
        # modified DH
        # ^i-1 T_i = Rx(alpha) * Tx(a) * Rz(theta) * Tz(d)
        return (
            HTM.rx(alpha)
            @ HTM.tx(a)
            @ HTM.rz(theta)
            @ HTM.tz(d)
        )


# ----- SO(3) utilities (pure Torch, batched) -----
def _rotvec_from_R(R: Tensor) -> Tensor:
    """
    Stable rotation vector phi (3×1) from rotation matrix R.

    Args:
        R: rotation matrix of shape (..., 3, 3)

    Returns:
        phi: rotation vector of shape (..., 3, 1)
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"_rotvec_from_R expects R[...,3,3], got {R.shape}")

    v = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )  # (..., 3)
    vnorm = torch.linalg.norm(v, dim=-1)  # (...)
    s = 0.5 * vnorm
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    c = 0.5 * (trace - 1.0)

    theta = torch.atan2(s, c)  # (...)
    eps = 1e-6 if R.dtype == torch.float32 else 1e-12
    small = theta.abs() < eps

    k_small = 0.5 + (theta * theta) / 12.0

    sin_theta = torch.sin(theta)
    safe_sin = torch.where(small, torch.ones_like(sin_theta), sin_theta)
    safe_theta = torch.where(small, torch.ones_like(theta), theta)
    k_big = safe_theta / (2.0 * safe_sin)

    k = torch.where(small, k_small, k_big)  # (...)
    phi = k.unsqueeze(-1) * v  # (..., 3)
    return phi.unsqueeze(-1)   # (..., 3, 1)


def _skew(a3: Tensor) -> Tensor:
    """
    Skew-symmetric matrix [a]_x for a3 (shape (...,3)).

    Returns:
        A: skew-symmetric matrix of shape (..., 3, 3)
    """
    if a3.shape[-1] != 3:
        raise ValueError(f"_skew expects a3[...,3], got {a3.shape}")
    x = a3[..., 0]
    y = a3[..., 1]
    z = a3[..., 2]
    zero = torch.zeros_like(x)
    row0 = torch.stack([zero, -z, y], dim=-1)
    row1 = torch.stack([z, zero, -x], dim=-1)
    row2 = torch.stack([-y, x, zero], dim=-1)
    A = torch.stack([row0, row1, row2], dim=-2)
    return A


def _so3_left_jacobian_inv(phi3x1: Tensor) -> Tensor:
    """
    SO(3) left Jacobian inverse JL(phi)^{-1} for Torch, batched.

    Args:
        phi3x1: rotation vector of shape (..., 3, 1) or (..., 3)

    Returns:
        J: tensor of shape (..., 3, 3)
    """
    if phi3x1.shape[-2:] == (3, 1):
        phi = phi3x1.squeeze(-1)
    else:
        phi = phi3x1
    if phi.shape[-1] != 3:
        raise ValueError(f"_so3_left_jacobian_inv expects phi[...,3] or phi[...,3,1], got {phi3x1.shape}")

    theta = torch.linalg.norm(phi, dim=-1)  # (...)
    A = _skew(phi)  # (...,3,3)
    A2 = A @ A      # (...,3,3)

    eps = 1e-8
    small = theta < eps

    eye3 = torch.eye(3, dtype=phi.dtype, device=phi.device)
    I = eye3.expand(phi.shape[:-1] + (3, 3))

    J_small = I - 0.5 * A + (1.0 / 12.0) * A2

    safe_theta = torch.where(small, torch.ones_like(theta), theta)
    theta2 = safe_theta * safe_theta
    sin_theta = torch.sin(safe_theta)
    cos_theta = torch.cos(safe_theta)
    denom = 2.0 * safe_theta * sin_theta
    safe_denom = torch.where(
        torch.abs(denom) < 1e-8,
        torch.ones_like(denom),
        denom,
    )
    b = (1.0 / theta2) - (1.0 + cos_theta) / safe_denom
    b = torch.where(small, torch.zeros_like(b), b)

    bA2 = b.unsqueeze(-1).unsqueeze(-1) * A2
    J_big = I - 0.5 * A + bA2

    mask = small.unsqueeze(-1).unsqueeze(-1)
    J = torch.where(mask, J_small, J_big)
    return J


# ---------------- API-compatible functions (Torch only) ----------------
def forwardHTM(robot: object) -> List[Tensor]:
    """
    Forward kinematics (joint frames) for Torch backend.

    Returns:
        frames: list of transforms [T0, T1, ..., T_{n}], where
                T0 is identity and T_k is the product of DH rows up to k-1.
                Each Tk has shape (4,4) or (B,4,4) if batched.
    """
    conv = getattr(robot, "dh_convention", "standard")
    DH = _dh_table(robot, use_com=False)  # (n,4) or (B,n,4)

    if DH.dim() == 2:
        T = _eye4_like(like=DH)
        frames: List[Tensor] = [T]
        for i in range(DH.shape[0]):
            th, d, a, al = DH[i, 0], DH[i, 1], DH[i, 2], DH[i, 3]
            T = T @ _step_T(th, d, a, al, conv)
            frames.append(T)
        return frames

    if DH.dim() == 3:
        B, n, _ = DH.shape
        T = _eye4_like(like=DH, batch_shape=(B,))
        frames = [T]
        for i in range(n):
            th = DH[:, i, 0]
            d = DH[:, i, 1]
            a = DH[:, i, 2]
            al = DH[:, i, 3]
            T = T @ _step_T(th, d, a, al, conv)  # (B,4,4) @ (B,4,4)
            frames.append(T)
        return frames

    raise ValueError(f"Unsupported DH shape {DH.shape} (expected (n,4) or (B,n,4)).")


def forwardCOMHTM(robot: object) -> List[Tensor]:
    """
    Forward kinematics for COM frames (Torch backend).

    Returns:
        framesCOM: list [T0, T_com0, T_com1, ...] where index k+1 is COM k.
    """
    conv = getattr(robot, "dh_convention", "standard")
    frames_joint = forwardHTM(robot)
    DHc = _dh_table(robot, use_com=True)  # (m,4) or (B,m,4)

    framesCOM: List[Tensor] = []
    if DHc.dim() == 2:
        T0 = _eye4_like(like=DHc)
        framesCOM = [T0]
        n_links = int(_q(robot).shape[0])
        for j in range(n_links):
            row, _ = robot.where_is_com(j)
            th, d, a, al = DHc[row, 0], DHc[row, 1], DHc[row, 2], DHc[row, 3]
            T_com = frames_joint[row] @ _step_T(th, d, a, al, conv)
            framesCOM.append(T_com)
        return framesCOM

    if DHc.dim() == 3:
        B, m, _ = DHc.shape
        T0 = _eye4_like(like=DHc, batch_shape=(B,))
        framesCOM = [T0]
        n_links = int(_q(robot).shape[-2])
        for j in range(n_links):
            row, _ = robot.where_is_com(j)
            th = DHc[:, row, 0]
            d = DHc[:, row, 1]
            a = DHc[:, row, 2]
            al = DHc[:, row, 3]
            T_com = frames_joint[row] @ _step_T(th, d, a, al, conv)
            framesCOM.append(T_com)
        return framesCOM

    raise ValueError(f"Unsupported DH_COM shape {DHc.shape} (expected (m,4) or (B,m,4)).")


def axisAngle(H: Tensor) -> Tensor:
    """
    Axis-angle pose x = [p; phi] from homogeneous transform H.

    Args:
        H: homogeneous transform of shape (..., 4, 4)

    Returns:
        x: tensor of shape (..., 6, 1) where the last 3 entries are the
           rotation vector phi, and the first 3 are position p.
    """
    if H.shape[-2:] != (4, 4):
        raise ValueError(f"axisAngle expects H[...,4,4], got {H.shape}")
    R = H[..., :3, :3]
    p = H[..., :3, 3].unsqueeze(-1)  # (...,3,1)
    phi = _rotvec_from_R(R)         # (...,3,1)
    return torch.cat([p, phi], dim=-2)  # (...,6,1)


def geometricJacobian(robot: object) -> Tensor:
    """
    6×n geometric Jacobian at the end-effector (Torch backend).

    Returns:
        J: tensor of shape (6,n) for unbatched, or (B,6,n) for batched state.
    """
    Ts = forwardHTM(robot)
    Te = Ts[-1]
    q = _q(robot)
    n = q.shape[-2]  # number of joints

    if Te.dim() == 2:
        pe = Te[0:3, 3].view(3)
        cols_w = []
        cols_v = []
        for j in range(n):
            row, _ = robot.where_is_joint(j)
            Tj = Ts[row]
            z = Tj[0:3, 2].view(3)
            p = Tj[0:3, 3].view(3)
            cols_w.append(z.view(3, 1))
            cols_v.append(torch.cross(z, pe - p, dim=0).view(3, 1))
        Jw = torch.cat(cols_w, dim=1)  # (3,n)
        Jv = torch.cat(cols_v, dim=1)  # (3,n)
        J = torch.cat([Jv, Jw], dim=0) # (6,n)
        _dbg("geometricJacobian (unbatched): J shape=", tuple(J.shape))
        return J

    if Te.dim() == 3:
        B = Te.shape[0]
        pe = Te[:, 0:3, 3]  # (B,3)
        Jv = torch.zeros((B, 3, n), dtype=Te.dtype, device=Te.device)
        Jw = torch.zeros((B, 3, n), dtype=Te.dtype, device=Te.device)
        for j in range(n):
            row, _ = robot.where_is_joint(j)
            Tj = Ts[row]  # (B,4,4)
            z = Tj[:, 0:3, 2]  # (B,3)
            p = Tj[:, 0:3, 3]  # (B,3)
            Jw[:, :, j] = z
            Jv[:, :, j] = torch.cross(z, pe - p, dim=1)
        J = torch.cat([Jv, Jw], dim=1)  # (B,6,n)
        _dbg("geometricJacobian (batched): J shape=", tuple(J.shape))
        return J

    raise ValueError(f"Unsupported Te shape {Te.shape} (expected (4,4) or (B,4,4)).")


def geometricJacobianDerivative(robot: object) -> Tensor:
    """
    6×n time derivative of geometric Jacobian at end-effector (Torch backend).

    Supports both unbatched and batched states.
    """
    Ts = forwardHTM(robot)
    Te = Ts[-1]
    qd = _qd(robot)
    q = _q(robot)
    n = q.shape[-2]

    # reshape qd to (B,n) or (n,)
    if Te.dim() == 2:
        # unbatched
        if qd.dim() == 2 and qd.shape[1] == 1:
            qd_vec = qd.view(-1)
        else:
            qd_vec = qd.view(-1)
        pe = Te[0:3, 3]
        dJ = torch.zeros((6, n), dtype=Te.dtype, device=Te.device)

        for j in range(n):
            row_j, _ = robot.where_is_joint(j)
            Tj = Ts[row_j]
            z = Tj[0:3, 2]
            p = Tj[0:3, 3]

            acc_lin = torch.zeros(3, dtype=Tj.dtype, device=Tj.device)

            # sum_{i=0}^{j-1} [(zi×z)×(pe-p) + zi×(z×(pe-p))] * qd[i]
            for i in range(j):
                row_i, _ = robot.where_is_joint(i)
                Ti = Ts[row_i]
                zi = Ti[0:3, 2]
                term = (
                    torch.cross(torch.cross(zi, z, dim=0), pe - p, dim=0)
                    + torch.cross(zi, torch.cross(z, pe - p, dim=0), dim=0)
                )
                acc_lin = acc_lin + term * qd_vec[i]

            # sum_{i=j}^{n-1} zi × [ z × (pe - pi) ] * qd[i]
            for i in range(j, n):
                row_i, _ = robot.where_is_joint(i)
                Ti = Ts[row_i]
                zi = Ti[0:3, 2]
                pi = Ti[0:3, 3]
                term = torch.cross(zi, torch.cross(z, pe - pi, dim=0), dim=0)
                acc_lin = acc_lin + term * qd_vec[i]

            acc_ang = torch.zeros(3, dtype=Tj.dtype, device=Tj.device)
            # sum_{i=j}^{n-1} (z × zi) * qd[i]
            for i in range(j, n):
                row_i, _ = robot.where_is_joint(i)
                Ti = Ts[row_i]
                zi = Ti[0:3, 2]
                acc_ang = acc_ang + torch.cross(z, zi, dim=0) * qd_vec[i]

            dJ[0:3, j] = acc_lin
            dJ[3:6, j] = acc_ang

        _dbg("geometricJacobianDerivative (unbatched): dJ shape=", tuple(dJ.shape))
        return dJ

    if Te.dim() == 3:
        # batched
        B = Te.shape[0]
        if qd.dim() == 3 and qd.shape[-1] == 1:
            qd_vec = qd.view(B, n)  # (B,n)
        else:
            qd_vec = qd.view(B, n)

        pe = Te[:, 0:3, 3]  # (B,3)
        dJ = torch.zeros((B, 6, n), dtype=Te.dtype, device=Te.device)

        for j in range(n):
            row_j, _ = robot.where_is_joint(j)
            Tj = Ts[row_j]          # (B,4,4)
            z = Tj[:, 0:3, 2]       # (B,3)
            p = Tj[:, 0:3, 3]       # (B,3)

            acc_lin = torch.zeros((B, 3), dtype=Tj.dtype, device=Tj.device)
            acc_ang = torch.zeros((B, 3), dtype=Tj.dtype, device=Tj.device)

            # sum_{i=0}^{j-1} [(zi×z)×(pe-p) + zi×(z×(pe-p))] * qd[:,i]
            pe_minus_p = pe - p  # (B,3)
            for i in range(j):
                row_i, _ = robot.where_is_joint(i)
                Ti = Ts[row_i]               # (B,4,4)
                zi = Ti[:, 0:3, 2]           # (B,3)
                term = (
                    torch.cross(torch.cross(zi, z, dim=1), pe_minus_p, dim=1)
                    + torch.cross(zi, torch.cross(z, pe_minus_p, dim=1), dim=1)
                )  # (B,3)
                acc_lin = acc_lin + term * qd_vec[:, i].unsqueeze(-1)

            # sum_{i=j}^{n-1} zi × [ z × (pe - pi) ] * qd[:,i]
            for i in range(j, n):
                row_i, _ = robot.where_is_joint(i)
                Ti = Ts[row_i]
                zi = Ti[:, 0:3, 2]           # (B,3)
                pi = Ti[:, 0:3, 3]           # (B,3)
                term = torch.cross(zi, torch.cross(z, pe - pi, dim=1), dim=1)  # (B,3)
                acc_lin = acc_lin + term * qd_vec[:, i].unsqueeze(-1)

            # sum_{i=j}^{n-1} (z × zi) * qd[:,i]
            for i in range(j, n):
                row_i, _ = robot.where_is_joint(i)
                Ti = Ts[row_i]
                zi = Ti[:, 0:3, 2]           # (B,3)
                term = torch.cross(z, zi, dim=1)  # (B,3)
                acc_ang = acc_ang + term * qd_vec[:, i].unsqueeze(-1)

            dJ[:, 0:3, j] = acc_lin
            dJ[:, 3:6, j] = acc_ang

        _dbg("geometricJacobianDerivative (batched): dJ shape=", tuple(dJ.shape))
        return dJ

    raise ValueError(f"Unsupported Te shape {Te.shape} (expected (4,4) or (B,4,4)).")


def geometricJacobianCOM(robot: object, COM: int) -> Tensor:
    """
    6×n geometric Jacobian to a specific COM frame (Torch backend).

    Args:
        COM: 0-based COM index.

    Returns:
        J: tensor of shape (6,n) (unbatched) or (B,6,n) (batched).
    """
    Ts = forwardHTM(robot)
    Tcs = forwardCOMHTM(robot)
    Tc = Tcs[COM + 1]
    q = _q(robot)
    n = q.shape[-2]
    rowCOM, _ = robot.where_is_com(COM)

    if Tc.dim() == 2:
        pcom = Tc[0:3, 3]
        J = torch.zeros((6, n), dtype=Tc.dtype, device=Tc.device)
        for j in range(n):
            row_j, _ = robot.where_is_joint(j)
            if row_j > rowCOM:
                break
            Tj = Ts[row_j]
            z = Tj[0:3, 2]
            pj = Tj[0:3, 3]
            J[0:3, j] = torch.cross(z, pcom - pj, dim=0)
            J[3:6, j] = z
        _dbg("geometricJacobianCOM (unbatched): J shape=", tuple(J.shape))
        return J

    if Tc.dim() == 3:
        B = Tc.shape[0]
        pcom = Tc[:, 0:3, 3]  # (B,3)
        J = torch.zeros((B, 6, n), dtype=Tc.dtype, device=Tc.device)
        for j in range(n):
            row_j, _ = robot.where_is_joint(j)
            if row_j > rowCOM:
                break
            Tj = Ts[row_j]         # (B,4,4)
            z = Tj[:, 0:3, 2]      # (B,3)
            pj = Tj[:, 0:3, 3]     # (B,3)
            J[:, 0:3, j] = torch.cross(z, pcom - pj, dim=1)
            J[:, 3:6, j] = z
        _dbg("geometricJacobianCOM (batched): J shape=", tuple(J.shape))
        return J

    raise ValueError(f"Unsupported Tc shape {Tc.shape} (expected (4,4) or (B,4,4)).")


def geometricJacobianDerivativeCOM(robot: object, COM: int) -> Tensor:
    """
    6×n time derivative of geometric Jacobian to a specific COM (Torch backend).

    Supports both unbatched and batched states.
    """
    Ts = forwardHTM(robot)
    Tcs = forwardCOMHTM(robot)
    Tc = Tcs[COM + 1]
    qd = _qd(robot)
    q = _q(robot)
    n = q.shape[-2]

    rowCOM, _ = robot.where_is_com(COM)

    if Tc.dim() == 2:
        # unbatched
        if qd.dim() == 2 and qd.shape[1] == 1:
            qd_vec = qd.view(-1)
        else:
            qd_vec = qd.view(-1)

        pcom = Tc[0:3, 3]
        dJ = torch.zeros((6, n), dtype=Tc.dtype, device=Tc.device)

        for j in range(n):
            row_j, _ = robot.where_is_joint(j)
            if row_j > rowCOM:
                break

            Tj = Ts[row_j]
            z = Tj[0:3, 2]
            pj = Tj[0:3, 3]

            acc_lin = torch.zeros(3, dtype=Tj.dtype, device=Tj.device)

            # sum_{i=0}^{j-1} [(zi×z)×(pcom-pj) + zi×(z×(pcom-pj))] * qd[i]
            for i in range(j):
                row_i, _ = robot.where_is_joint(i)
                Ti = Ts[row_i]
                zi = Ti[0:3, 2]
                term = (
                    torch.cross(torch.cross(zi, z, dim=0), pcom - pj, dim=0)
                    + torch.cross(zi, torch.cross(z, pcom - pj, dim=0), dim=0)
                )
                acc_lin = acc_lin + term * qd_vec[i]

            # sum_{i=j}^{n-1} zi × [ z × (pcom - pi) ] * qd[i]
            for i in range(j, n):
                row_i, _ = robot.where_is_joint(i)
                Ti = Ts[row_i]
                zi = Ti[0:3, 2]
                pi = Ti[0:3, 3]
                term = torch.cross(zi, torch.cross(z, pcom - pi, dim=0), dim=0)
                acc_lin = acc_lin + term * qd_vec[i]

            acc_ang = torch.zeros(3, dtype=Tj.dtype, device=Tj.device)
            # sum_{i=j}^{n-1} (z × zi) * qd[i]
            for i in range(j, n):
                row_i, _ = robot.where_is_joint(i)
                Ti = Ts[row_i]
                zi = Ti[0:3, 2]
                acc_ang = acc_ang + torch.cross(z, zi, dim=0) * qd_vec[i]

            dJ[0:3, j] = acc_lin
            dJ[3:6, j] = acc_ang

        _dbg("geometricJacobianDerivativeCOM (unbatched): dJ shape=", tuple(dJ.shape))
        return dJ

    if Tc.dim() == 3:
        # batched
        B = Tc.shape[0]
        if qd.dim() == 3 and qd.shape[-1] == 1:
            qd_vec = qd.view(B, n)  # (B,n)
        else:
            qd_vec = qd.view(B, n)

        pcom = Tc[:, 0:3, 3]  # (B,3)
        dJ = torch.zeros((B, 6, n), dtype=Tc.dtype, device=Tc.device)

        for j in range(n):
            row_j, _ = robot.where_is_joint(j)
            if row_j > rowCOM:
                break

            Tj = Ts[row_j]         # (B,4,4)
            z = Tj[:, 0:3, 2]      # (B,3)
            pj = Tj[:, 0:3, 3]     # (B,3)

            acc_lin = torch.zeros((B, 3), dtype=Tj.dtype, device=Tj.device)
            acc_ang = torch.zeros((B, 3), dtype=Tj.dtype, device=Tj.device)

            # sum_{i=0}^{j-1} [(zi×z)×(pcom-pj) + zi×(z×(pcom-pj))] * qd[:,i]
            pcom_minus_pj = pcom - pj  # (B,3)
            for i in range(j):
                row_i, _ = robot.where_is_joint(i)
                Ti = Ts[row_i]
                zi = Ti[:, 0:3, 2]  # (B,3)
                term = (
                    torch.cross(torch.cross(zi, z, dim=1), pcom_minus_pj, dim=1)
                    + torch.cross(zi, torch.cross(z, pcom_minus_pj, dim=1), dim=1)
                )  # (B,3)
                acc_lin = acc_lin + term * qd_vec[:, i].unsqueeze(-1)

            # sum_{i=j}^{n-1} zi × [ z × (pcom - pi) ] * qd[:,i]
            for i in range(j, n):
                row_i, _ = robot.where_is_joint(i)
                Ti = Ts[row_i]
                zi = Ti[:, 0:3, 2]  # (B,3)
                pi = Ti[:, 0:3, 3]  # (B,3)
                term = torch.cross(zi, torch.cross(z, pcom - pi, dim=1), dim=1)  # (B,3)
                acc_lin = acc_lin + term * qd_vec[:, i].unsqueeze(-1)

            # sum_{i=j}^{n-1} (z × zi) * qd[:,i]
            for i in range(j, n):
                row_i, _ = robot.where_is_joint(i)
                Ti = Ts[row_i]
                zi = Ti[:, 0:3, 2]
                term = torch.cross(z, zi, dim=1)
                acc_ang = acc_ang + term * qd_vec[:, i].unsqueeze(-1)

            dJ[:, 0:3, j] = acc_lin
            dJ[:, 3:6, j] = acc_ang

        _dbg("geometricJacobianDerivativeCOM (batched): dJ shape=", tuple(dJ.shape))
        return dJ

    raise ValueError(f"Unsupported Tc shape {Tc.shape} (expected (4,4) or (B,4,4)).")


# ---------- Analytic Jacobians (Torch only) ----------
def analyticJacobian(robot: object) -> Tensor:
    """
    Analytic Jacobian dx/dq of EE axis-angle pose x=[p; phi] (Torch backend).

    Uses torch.autograd.functional.jacobian with create_graph=True.

    Returns:
        J: tensor of shape (6,n)
    """
    q = _q(robot)
    q_flat = q.view(-1).clone().requires_grad_(True)
    _dbg("analyticJacobian: q_flat shape=", tuple(q_flat.shape))

    def f(qvec: Tensor) -> Tensor:
        old_q = robot.q
        robot.q = qvec.view_as(old_q)
        if hasattr(robot, "denavitHartenberg"):
            robot.denavitHartenberg()
        T = forwardHTM(robot)[-1]
        robot.q = old_q
        x = axisAngle(T).view(-1)  # (6,)
        return x

    J = torch_jacobian(f, q_flat, create_graph=True, vectorize=True)  # (6,n)
    _dbg("analyticJacobian: J shape=", tuple(J.shape))
    return J


def analyticJacobianCOM(robot: object, COM: int) -> Tensor:
    """
    Analytic Jacobian of COM axis-angle pose x_com = [p_com; phi_com] (Torch backend).

    Uses torch.autograd.functional.jacobian with create_graph=True.

    Returns:
        J: tensor of shape (6,n)
    """
    q = _q(robot)
    q_flat = q.view(-1).clone().requires_grad_(True)
    _dbg("analyticJacobianCOM: q_flat shape=", tuple(q_flat.shape), "COM=", COM)

    def f(qvec: Tensor) -> Tensor:
        old_q = robot.q
        robot.q = qvec.view_as(old_q)
        if hasattr(robot, "denavitHartenberg"):
            robot.denavitHartenberg()
        if hasattr(robot, "denavitHartenbergCOM"):
            robot.denavitHartenbergCOM()
        Tc = forwardCOMHTM(robot)[COM + 1]
        robot.q = old_q
        x = axisAngle(Tc).view(-1)  # (6,)
        return x

    J = torch_jacobian(f, q_flat, create_graph=True, vectorize=True)
    _dbg("analyticJacobianCOM: J shape=", tuple(J.shape))
    return J


# ---------- Iterative IK (Torch) ----------
def inverseHTM(
    robot: object,
    q0: Tensor,
    Hd: Tensor,
    K: Tensor,
    jacobian: str = "geometric",
    max_iters: int = 15000,
    tol: float = 1e-3,
) -> Tensor:
    """
    Iterative inverse kinematics in Torch, adapted from the NumPy implementation.

    Args:
        q0: initial joint configuration, shape (n,1) or (n,)
        Hd: desired homogeneous transform 4×4 (Torch tensor)
        K:  6×6 gain matrix (Torch tensor)
        jacobian: "geometric" or "analytic"
        max_iters: maximum iterations
        tol: stop if max(|e|) <= tol

    Returns:
        q_hist: tensor of shape (n, T) with the trajectory of joint configs.
    """
    if q0.dim() == 1:
        q0 = q0.view(-1, 1)
    q = q0.clone()  # (n,1)
    n = q0.shape[0]

    Hd = Hd.clone()
    K = K.clone()

    Xd = axisAngle(Hd)  # (6,1)

    zero_col = torch.zeros((3, 1), dtype=Hd.dtype, device=Hd.device)
    bottom_row = torch.tensor([[0, 0, 0, 1]], dtype=Hd.dtype, device=Hd.device)
    Rd = torch.cat(
        [
            torch.cat([Hd[0:3, 0:3], zero_col], dim=1),
            bottom_row,
        ],
        dim=0,
    )  # 4×4

    q_hist = q.clone().reshape(n, 1)

    if not hasattr(robot, "q"):
        raise AttributeError("robot must expose attribute 'q' as torch.Tensor for inverseHTM.")

    z = robot.q.clone()
    j = 0
    while j < max_iters:
        robot.q = q
        if hasattr(robot, "denavitHartenberg"):
            robot.denavitHartenberg()
        fkH = forwardHTM(robot)
        T = fkH[-1]
        X = axisAngle(T)  # (6,1)

        R = torch.cat(
            [
                torch.cat([T[0:3, 0:3], zero_col], dim=1),
                bottom_row,
            ],
            dim=0,
        )
        R_err = Rd @ R.T
        X_err_rot = axisAngle(R_err)[3:6, :]  # (3,1)

        e = torch.cat(
            [
                Xd[0:3, :] - X[0:3, :],
                X_err_rot,
            ],
            dim=0,
        )  # (6,1)

        if torch.max(torch.abs(e)) <= tol:
            break

        if jacobian == "geometric":
            J = geometricJacobian(robot)  # (6,n)
        elif jacobian == "analytic":
            J = analyticJacobian(robot)   # (6,n)
        else:
            J = torch.zeros((6, n), dtype=q.dtype, device=q.device)

        Ke = K @ e        # (6,1)
        J_pinv = torch.linalg.pinv(J)  # (n,6)
        dq = J_pinv @ Ke  # (n,1)

        q = q + dq
        q_hist = torch.cat([q_hist, q.clone().reshape(n, 1)], dim=1)
        j += 1

    robot.q = z
    return q_hist


# ---------- SMOKE TESTS ----------
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)


    class DummyRobot:
        """
        Minimal 2-DoF serial robot with standard DH for smoke tests.

        - q:   (n,1) or (B,n,1)
        - qd:  same shape as q
        - dh:  (n,4) or (B,n,4) with [theta, d, a, alpha]
        - dhCOM: same shape, COM at mid-link distance
        - where_is_joint(j): maps joint j -> (j+1, 0) frame index
        - where_is_com(k): maps COM k -> (k, 0) DH_COM row index (for this dummy)
        """

        def __init__(self, dh: torch.Tensor, dhCOM: torch.Tensor):
            self.dh = dh
            self.dhCOM = dhCOM
            self.dh_convention = "standard"
            n = dh.shape[-2]
            if dh.dim() == 2:
                # unbatched
                self.q = torch.zeros((n, 1), dtype=dh.dtype, device=dh.device)
                self.qd = torch.zeros_like(self.q)
            elif dh.dim() == 3:
                B = dh.shape[0]
                self.q = torch.zeros((B, n, 1), dtype=dh.dtype, device=dh.device)
                self.qd = torch.zeros_like(self.q)
            else:
                raise ValueError("dh must be (n,4) or (B,n,4)")

        def denavitHartenberg(self):
            return

        def denavitHartenbergCOM(self):
            return

        def where_is_joint(self, j: int) -> tuple[int, int]:
            # frame index in forwardHTM list (T0 base, T1 after first joint, etc.)
            return j + 1, 0

        def where_is_com(self, k: int) -> tuple[int, int]:
            # For this dummy: COM k uses DH_COM row k.
            # (rows 0..n-1 exist because dhCOM has shape (n,4))
            return k, 0

    def _make_dummy_robot_unbatched() -> DummyRobot:
        # 2-DoF planar arm example, links length 0.3, 0.2
        n = 2
        dh = torch.zeros((n, 4), dtype=torch.float64)
        # For standard DH: [theta, d, a, alpha]
        dh[0, 2] = 0.3  # a1
        dh[1, 2] = 0.2  # a2
        dhCOM = dh.clone()
        dhCOM[0, 2] = 0.3 * 0.5
        dhCOM[1, 2] = 0.2 * 0.5
        return DummyRobot(dh, dhCOM)

    def _make_dummy_robot_batched(B: int = 5) -> DummyRobot:
        base = _make_dummy_robot_unbatched()
        dh = base.dh.unsqueeze(0).expand(B, -1, -1).clone()      # (B,2,4)
        dhCOM = base.dhCOM.unsqueeze(0).expand(B, -1, -1).clone()
        return DummyRobot(dh, dhCOM)

    def _fd_Jdot(robot: DummyRobot, COM: Optional[int] = None, h: float = 1e-6) -> Tensor:
        """
        Finite-difference approximation of J̇ along q̇:
        J̇ ≈ (J(q + h q̇) - J(q - h q̇)) / (2 h)
        """
        q = robot.q.clone()
        qd = robot.qd.clone()

        # plus
        robot.q = q + h * qd
        if COM is None:
            Jp = geometricJacobian(robot)
        else:
            Jp = geometricJacobianCOM(robot, COM)

        # minus
        robot.q = q - h * qd
        if COM is None:
            Jm = geometricJacobian(robot)
        else:
            Jm = geometricJacobianCOM(robot, COM)

        # restore
        robot.q = q

        return (Jp - Jm) / (2.0 * h)

    # ---- Unbatched smoke test ----
    print("\n[SMOKE] Unbatched 2-DoF robot:")
    robot_u = _make_dummy_robot_unbatched()
    n = robot_u.dh.shape[0]
    # random q, qd
    robot_u.q = torch.randn((n, 1), dtype=torch.float64) * 0.5
    robot_u.qd = torch.randn((n, 1), dtype=torch.float64) * 0.2

    J_u = geometricJacobian(robot_u)
    Jdot_u = geometricJacobianDerivative(robot_u)
    Jdot_fd_u = _fd_Jdot(robot_u)
    err_u = (Jdot_u - Jdot_fd_u).abs().max().item()
    print("  J shape:", tuple(J_u.shape))
    print("  Jdot shape:", tuple(Jdot_u.shape))
    print("  max|Jdot - Jdot_fd| =", err_u)

    # COM version (COM index 0)
    Jc_u = geometricJacobianCOM(robot_u, COM=0)
    Jcdot_u = geometricJacobianDerivativeCOM(robot_u, COM=0)
    Jcdot_fd_u = _fd_Jdot(robot_u, COM=0)
    errc_u = (Jcdot_u - Jcdot_fd_u).abs().max().item()
    print("  J_COM shape:", tuple(Jc_u.shape))
    print("  Jdot_COM shape:", tuple(Jcdot_u.shape))
    print("  max|Jdot_COM - Jdot_COM_fd| =", errc_u)

    # ---- Batched smoke test ----
    print("\n[SMOKE] Batched 2-DoF robot (B=4):")
    B = 4
    robot_b = _make_dummy_robot_batched(B=B)
    n = robot_b.dh.shape[-2]
    robot_b.q = torch.randn((B, n, 1), dtype=torch.float64) * 0.5
    robot_b.qd = torch.randn((B, n, 1), dtype=torch.float64) * 0.2

    J_b = geometricJacobian(robot_b)
    Jdot_b = geometricJacobianDerivative(robot_b)
    Jdot_fd_b = _fd_Jdot(robot_b)
    err_b = (Jdot_b - Jdot_fd_b).abs().max().item()
    print("  J shape:", tuple(J_b.shape))
    print("  Jdot shape:", tuple(Jdot_b.shape))
    print("  max|Jdot - Jdot_fd| =", err_b)

    # COM version batched
    Jc_b = geometricJacobianCOM(robot_b, COM=0)
    Jcdot_b = geometricJacobianDerivativeCOM(robot_b, COM=0)
    Jcdot_fd_b = _fd_Jdot(robot_b, COM=0)
    errc_b = (Jcdot_b - Jcdot_fd_b).abs().max().item()
    print("  J_COM shape:", tuple(Jc_b.shape))
    print("  Jdot_COM shape:", tuple(Jcdot_b.shape))
    print("  max|Jdot_COM - Jdot_COM_fd| =", errc_b)

    print("\n[SMOKE] Done.")
