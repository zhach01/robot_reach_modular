# DifferentialHTM_torch.py
"""
Pure-PyTorch differential kinematics utilities based on HTM_kinematics_torch.

This is a Torch-only rewrite of the original DifferentialHTM.py:
- No NumPy, no SymPy, no `symbolic` flag.
- Uses the pure-Torch HTM_kinematics_torch and HTM_torch primitives.
- Fully differentiable and GPU-ready.

Provided functions:

1. State-space based on Jacobians (batchable)
   ----------------------------------------
   - geometricStateSpace(robot)
   - geometricDerivativeStateSpace(robot)
   - geometricCOMStateSpace(robot, com_index)
   - geometricCOMDerivativeStateSpace(robot, com_index)
   - analyticStateSpace(robot)           (unbatched)
   - analyticCOMStateSpace(robot, com_index)  (unbatched)

   These use:
   - J  = geometricJacobian(...)
   - Jd = geometricJacobianDerivative(...)
   - Jc = geometricJacobianCOM(...)
   - Jcd = geometricJacobianDerivativeCOM(...)
   - Ja / Ja_com for analytic versions.

   Expected robot attributes:
   - q   : (n,1) or (B,n,1)   joint positions
   - qd  : (n,1) or (B,n,1)   joint velocities
   - qdd : (n,1) or (B,n,1)   joint accelerations (for *DerivativeStateSpace)

2. Velocity propagation along frames (batchable)
   ---------------------------------------------
   - velocityPropagation(robot, v0, w0)
   - accelerationPropagation(robot, dv0, dw0, V)

   Computes 6D twists (v; w) per link frame using standard HTM recursion.
   Supports:
   - unbatched: q, qd, qdd of shape (n,1)
   - batched :  q, qd, qdd of shape (B,n,1)
   Returns:
   - list V of length n+1, with each element:
       (6,1)   for unbatched
       (B,6,1) for batched

3. Velocity/acceleration propagation to COMs (batchable)
   -----------------------------------------------------
   - velocityPropagationCOM(robot, vCOM0, wCOM0, V)
   - accelerationPropagationCOM(robot, dvCOM0, dwCOM0, Vcom, dV)

   Uses:
   - forwardHTM(robot)
   - forwardCOMHTM(robot)
   - robot.where_is_joint(j) and robot.where_is_com(j)
   to map COMs to their associated joint frames.

All functions are pure Torch and autograd-safe.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor

# --- import our canonical Torch kinematics and HTM primitives ---
try:
    from lib.kinematics import HTM_kinematics_torch as KIN
except Exception:
    import HTM_kinematics_torch as KIN  # fallback on PYTHONPATH

try:
    from lib.movements import HTM_torch as HTM
except Exception:
    import lib.movements.HTM_torch as HTM  # fallback on PYTHONPATH


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _q(robot) -> Tensor:
    if not hasattr(robot, "q"):
        raise AttributeError("robot must expose attribute 'q' as a torch.Tensor")
    return robot.q


def _qd(robot) -> Tensor:
    if not hasattr(robot, "qd"):
        raise AttributeError("robot must expose attribute 'qd' as a torch.Tensor")
    return robot.qd


def _qdd(robot) -> Tensor:
    if not hasattr(robot, "qdd"):
        raise AttributeError("robot must expose attribute 'qdd' as a torch.Tensor")
    return robot.qdd


def _ensure_col(vec: Tensor) -> Tensor:
    """
    Ensure a (n,) or (n,1) vector is shaped as column (n,1).
    For batched input (B,n) or (B,n,1) -> (B,n,1).
    """
    if vec.dim() == 1:
        return vec.view(-1, 1)
    if vec.dim() == 2:
        # either (n,1) or (B,n)
        if vec.shape[1] == 1:
            return vec
        else:
            # treat as (B,n)
            return vec.unsqueeze(-1)
    if vec.dim() == 3:
        # assume already (B,n,1)
        return vec
    raise ValueError(f"Unsupported vector shape {vec.shape} in _ensure_col")


def _normalize_base_vec(v: Tensor, B: int | None, device, dtype, name: str) -> Tensor:
    """
    Normalize a base vector (v0 or w0, or dv0/dw0) to:
      - (3,1)   if B is None (unbatched)
      - (B,3,1) if B is not None (batched)

    Accepts common shapes:
      - unbatched: (3,), (3,1), (1,3)
      - batched:   (3,), (3,1) [broadcast to all B], (B,3), (B,3,1)
    """
    v = v.to(device=device, dtype=dtype)

    if B is None:
        # unbatched → (3,1)
        if v.dim() == 1:
            if v.numel() != 3:
                raise ValueError(f"{name} must have 3 elements, got {v.shape}")
            return v.view(3, 1)
        elif v.dim() == 2:
            if v.shape == (3, 1):
                return v
            if v.shape == (1, 3):
                return v.view(3, 1)
            raise ValueError(f"{name} must be (3,1) or (1,3), got {v.shape}")
        else:
            raise ValueError(f"{name} must be 1D or 2D in unbatched mode, got {v.shape}")

    # batched → (B,3,1)
    if v.dim() == 1:
        if v.numel() != 3:
            raise ValueError(f"{name} must have 3 elements, got {v.shape}")
        v = v.view(1, 3).expand(B, 3)  # (B,3)
    elif v.dim() == 2:
        if v.shape == (B, 3):
            pass
        elif v.shape == (3, 1):
            v = v.view(1, 3).expand(B, 3)
        elif v.shape == (1, 3):
            v = v.expand(B, 3)
        else:
            raise ValueError(
                f"{name} in batched mode must be (3,), (3,1), (1,3) or (B,3), got {v.shape}"
            )
    elif v.dim() == 3:
        if v.shape == (B, 3, 1):
            return v
        if v.shape[0] == B and v.shape[1] == 3 and v.shape[2] == 1:
            return v
        raise ValueError(f"{name} 3D shape not supported: {v.shape}")
    else:
        raise ValueError(f"{name} must be 1D, 2D or 3D, got {v.shape}")

    # now (B,3)
    return v.unsqueeze(-1)  # (B,3,1)


def _cross_matrix(v: Tensor) -> Tensor:
    """
    Batched skew-symmetric matrix [v]_x for v of shape (...,3).

    If v is 1D with shape (3,), we delegate to HTM.crossMatrix(v) to
    preserve existing semantics. Otherwise we build the matrix ourselves.

    Returns:
        M: skew-symmetric matrix of shape (...,3,3)
    """
    if v.dim() == 1 and v.shape[0] == 3:
        return HTM.crossMatrix(v)

    if v.shape[-1] != 3:
        raise ValueError(f"_cross_matrix expects v[...,3], got {v.shape}")

    x = v[..., 0]
    y = v[..., 1]
    z = v[..., 2]
    zero = torch.zeros_like(x)

    row0 = torch.stack([zero, -z, y], dim=-1)
    row1 = torch.stack([z, zero, -x], dim=-1)
    row2 = torch.stack([-y, x, zero], dim=-1)
    M = torch.stack([row0, row1, row2], dim=-2)
    return M


def _coriolis_term(w: Tensor, r: Tensor) -> Tensor:
    """
    Compute coriolis-like linear term c_lin = w × (w × r)
    in a batch-safe way.

    Args:
        w: angular velocity, shape (3,) or (B,3)
        r: position vector, shape (3,) or (B,3)

    Returns:
        c_lin: shape (3,1) or (B,3,1)
    """
    if w.dim() == 1:
        # unbatched
        inner = torch.linalg.cross(w, r, dim=0)
        c = torch.linalg.cross(w, inner, dim=0)  # (3,)
        return c.view(3, 1)
    elif w.dim() == 2:
        # batched: (B,3)
        inner = torch.linalg.cross(w, r, dim=1)  # (B,3)
        c = torch.linalg.cross(w, inner, dim=1)  # (B,3)
        return c.unsqueeze(-1)                   # (B,3,1)
    else:
        raise ValueError(f"_coriolis_term expects w with dim 1 or 2, got {w.shape}")


# --------------------------------------------------------------------
# 1. State-space equations using Jacobians (Torch, batchable)
# --------------------------------------------------------------------
def geometricStateSpace(robot: object) -> Tensor:
    """
    Xd = J(q) * qd

    Geometric state-space based on the geometric Jacobian.

    Returns:
        Xd:
            - shape (6,1) for unbatched robot
            - shape (B,6,1) for batched robot
    """
    J = KIN.geometricJacobian(robot)  # (6,n) or (B,6,n)
    qd = _ensure_col(_qd(robot))      # (n,1) or (B,n,1)

    if J.dim() == 2:
        # (6,n) @ (n,1) -> (6,1)
        return J @ qd
    elif J.dim() == 3:
        # (B,6,n) @ (B,n,1) -> (B,6,1)
        return torch.matmul(J, qd)
    else:
        raise ValueError(f"Unsupported Jacobian shape {J.shape} in geometricStateSpace")


def geometricDerivativeStateSpace(robot: object) -> Tensor:
    """
    Xdd = J̇(q, qd) * qd + J(q) * qdd

    Geometric derivative of the state-space based on J and J̇.

    Returns:
        Xdd:
            - shape (6,1) for unbatched robot
            - shape (B,6,1) for batched robot
    """
    J = KIN.geometricJacobian(robot)
    Jd = KIN.geometricJacobianDerivative(robot)

    qd = _ensure_col(_qd(robot))
    qdd = _ensure_col(_qdd(robot))

    if J.dim() == 2:
        term1 = Jd @ qd        # (6,1)
        term2 = J @ qdd        # (6,1)
        return term1 + term2
    elif J.dim() == 3:
        term1 = torch.matmul(Jd, qd)   # (B,6,1)
        term2 = torch.matmul(J, qdd)   # (B,6,1)
        return term1 + term2
    else:
        raise ValueError(
            f"Unsupported Jacobian shape {J.shape} in geometricDerivativeStateSpace"
        )


def geometricCOMStateSpace(robot: object, COM: int) -> Tensor:
    """
    Xd_com = J_com(q) * qd

    Geometric state-space at a given COM index.

    Args:
        COM: 0-based index of the center of mass.

    Returns:
        Xd_com:
            - shape (6,1) for unbatched robot
            - shape (B,6,1) for batched robot
    """
    Jc = KIN.geometricJacobianCOM(robot, COM)
    qd = _ensure_col(_qd(robot))

    if Jc.dim() == 2:
        return Jc @ qd
    elif Jc.dim() == 3:
        return torch.matmul(Jc, qd)
    else:
        raise ValueError(
            f"Unsupported COM Jacobian shape {Jc.shape} in geometricCOMStateSpace"
        )


def geometricCOMDerivativeStateSpace(robot: object, COM: int) -> Tensor:
    """
    Xdd_com = J̇_com(q, qd) * qd + J_com(q) * qdd

    Geometric state-space derivative at a given COM index.

    Args:
        COM: 0-based COM index.

    Returns:
        Xdd_com:
            - shape (6,1) for unbatched robot
            - shape (B,6,1) for batched robot
    """
    Jc = KIN.geometricJacobianCOM(robot, COM)
    Jcd = KIN.geometricJacobianDerivativeCOM(robot, COM)
    qd = _ensure_col(_qd(robot))
    qdd = _ensure_col(_qdd(robot))

    if Jc.dim() == 2:
        term1 = Jcd @ qd
        term2 = Jc @ qdd
        return term1 + term2
    elif Jc.dim() == 3:
        term1 = torch.matmul(Jcd, qd)
        term2 = torch.matmul(Jc, qdd)
        return term1 + term2
    else:
        raise ValueError(
            f"Unsupported COM Jacobian shape {Jc.shape} in geometricCOMDerivativeStateSpace"
        )


def analyticStateSpace(robot: object) -> Tensor:
    """
    Xd = J_a(q) * qd

    Analytic state-space using the analytic Jacobian Ja.
    Currently supports unbatched robots only.

    Returns:
        Xd: shape (6,1)
    """
    J = KIN.analyticJacobian(robot)  # (6,n), unbatched
    if J.dim() != 2:
        raise NotImplementedError("analyticStateSpace currently supports only unbatched robots.")
    qd = _ensure_col(_qd(robot))     # (n,1)
    return J @ qd                    # (6,1)


def analyticCOMStateSpace(robot: object, COM: int) -> Tensor:
    """
    Xd_com = J_a_com(q) * qd

    Analytic state-space at COM using analytic Jacobian.
    Currently supports unbatched robots only.

    Args:
        COM: 0-based COM index.

    Returns:
        Xd_com: shape (6,1)
    """
    Jc = KIN.analyticJacobianCOM(robot, COM)
    if Jc.dim() != 2:
        raise NotImplementedError(
            "analyticCOMStateSpace currently supports only unbatched robots."
        )
    qd = _ensure_col(_qd(robot))
    return Jc @ qd


# --------------------------------------------------------------------
# 2. Velocity propagation along frames (batchable)
# --------------------------------------------------------------------
def velocityPropagation(robot: object, v0: Tensor, w0: Tensor) -> List[Tensor]:
    """
    Propagate linear and angular velocities along the kinematic chain.

    Uses standard HTM recursion:
        V_i = M_i * V_{i-1} + M_i * [0; z_{i-1}] * q̇_{i-1}

    Assumptions:
    - q, qd shapes:
        - unbatched: (n,1)
        - batched :  (B,n,1)
    - Standard serial chain: one joint per link, joints ordered along q.
    - Joints are *revolute* about z_{i-1} axis.

    Args:
        v0: base linear velocity, shape (3,), (3,1),
            or in batched mode (B,3) / (B,3,1)
        w0: base angular velocity, same conventions as v0

    Returns:
        V: list of twists V_i for each frame i:
            - unbatched: each V_i is (6,1)
            - batched  : each V_i is (B,6,1)
          with length n+1 (V[0] is base).
    """
    q = _q(robot)
    qd = _qd(robot)

    if q.dim() == 2 and q.shape[1] == 1:
        batched = False
        B = None
        n = q.shape[0]
    elif q.dim() == 3 and q.shape[2] == 1:
        batched = True
        B = q.shape[0]
        n = q.shape[1]
    else:
        raise NotImplementedError(
            f"velocityPropagation expects q of shape (n,1) or (B,n,1), got {q.shape}"
        )

    Ts = KIN.forwardHTM(robot)  # list of transforms, length n+1

    device = Ts[0].device
    dtype = Ts[0].dtype

    v0 = _normalize_base_vec(v0, B, device, dtype, "v0")
    w0 = _normalize_base_vec(w0, B, device, dtype, "w0")

    if batched:
        V: List[Tensor] = [torch.cat([v0, w0], dim=1)]  # (B,6,1)
        I3 = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)  # (1,3,3)
        Z3 = torch.zeros((1, 3, 3), dtype=dtype, device=device)
    else:
        V = [torch.cat([v0, w0], dim=0)]  # (6,1)
        I3 = torch.eye(3, dtype=dtype, device=device)
        Z3 = torch.zeros((3, 3), dtype=dtype, device=device)

    for i in range(1, n + 1):
        Ti = Ts[i]
        Tim1 = Ts[i - 1]

        if not batched:
            p_i = Ti[0:3, 3].view(3, 1)
            p_im1 = Tim1[0:3, 3].view(3, 1)
            r = p_i - p_im1       # (3,1)
            r3 = r.view(3)
            ri = _cross_matrix(r3)  # (3,3)

            Mi_top = torch.cat([I3, -ri], dim=1)   # (3,6)
            Mi_bottom = torch.cat([Z3, I3], dim=1) # (3,6)
            Mi = torch.cat([Mi_top, Mi_bottom], dim=0)  # (6,6)

            z = Tim1[0:3, 2].view(3, 1)
            qd_j = qd[i - 1, 0]

            vec6 = torch.cat([torch.zeros_like(z), z], dim=0)  # (6,1)
            vJoint = Mi @ (vec6 * qd_j)                        # (6,1)

            V_i = Mi @ V[-1] + vJoint
            V.append(V_i)
        else:
            # batched: (B,4,4)
            p_i = Ti[:, 0:3, 3].unsqueeze(-1)     # (B,3,1)
            p_im1 = Tim1[:, 0:3, 3].unsqueeze(-1) # (B,3,1)
            r = p_i - p_im1                       # (B,3,1)
            r3 = r.squeeze(-1)                    # (B,3)
            ri = _cross_matrix(r3)                # (B,3,3)

            I3b = I3.expand(B, 3, 3)
            Z3b = Z3.expand(B, 3, 3)

            Mi_top = torch.cat([I3b, -ri], dim=2)     # (B,3,6)
            Mi_bottom = torch.cat([Z3b, I3b], dim=2)  # (B,3,6)
            Mi = torch.cat([Mi_top, Mi_bottom], dim=1)  # (B,6,6)

            z = Tim1[:, 0:3, 2].unsqueeze(-1)  # (B,3,1)
            qd_j = qd[:, i - 1, 0].view(B, 1, 1)

            zeros_z = torch.zeros_like(z)
            vec6 = torch.cat([zeros_z, z], dim=1)  # (B,6,1)
            vJoint = torch.matmul(Mi, vec6 * qd_j)  # (B,6,1)

            V_prev = V[-1]                         # (B,6,1)
            V_i = torch.matmul(Mi, V_prev) + vJoint
            V.append(V_i)

    return V


def accelerationPropagation(
    robot: object,
    dv0: Tensor,
    dw0: Tensor,
    V: List[Tensor],
) -> List[Tensor]:
    """
    Propagate linear and angular accelerations along the chain.

    Uses recursion:
        dV_i = M_i dV_{i-1} + dV_joint + coriolis

    with:
        dV_joint = (M_nl * V_i) * q̇_{i-1} + M_i * [0; z_{i-1}] * q̈_{i-1}
        coriolis = [ w_i × (w_i × r_i ); 0 ]

    Assumptions:
    - q, qd, qdd shapes:
        - unbatched: (n,1)
        - batched :  (B,n,1)
    - Standard serial chain: one revolute joint per link.
    - V list comes from velocityPropagation(robot, v0, w0).

    Args:
        dv0: base linear acceleration, shape (3,), (3,1),
             or in batched mode (B,3)/(B,3,1)
        dw0: base angular acceleration, same conventions as dv0
        V:   list of twists from velocityPropagation (length n+1)

    Returns:
        dV: list of accelerations per frame (6×1 or B×6×1), length n+1.
    """
    q = _q(robot)
    qd = _qd(robot)
    qdd = _qdd(robot)

    if q.dim() == 2 and q.shape[1] == 1:
        batched = False
        B = None
        n = q.shape[0]
    elif q.dim() == 3 and q.shape[2] == 1:
        batched = True
        B = q.shape[0]
        n = q.shape[1]
    else:
        raise NotImplementedError(
            f"accelerationPropagation expects q of shape (n,1) or (B,n,1), got {q.shape}"
        )

    if len(V) != n + 1:
        raise ValueError(f"V must have length n+1={n+1}, got {len(V)}.")

    Ts = KIN.forwardHTM(robot)
    device = Ts[0].device
    dtype = Ts[0].dtype

    dv0 = _normalize_base_vec(dv0, B, device, dtype, "dv0")
    dw0 = _normalize_base_vec(dw0, B, device, dtype, "dw0")

    if batched:
        dV: List[Tensor] = [torch.cat([dv0, dw0], dim=1)]  # (B,6,1)
        I3 = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)  # (1,3,3)
        Z3 = torch.zeros((1, 3, 3), dtype=dtype, device=device)
    else:
        dV = [torch.cat([dv0, dw0], dim=0)]  # (6,1)
        I3 = torch.eye(3, dtype=dtype, device=device)
        Z3 = torch.zeros((3, 3), dtype=dtype, device=device)

    for i in range(1, n + 1):
        Ti = Ts[i]
        Tim1 = Ts[i - 1]

        if not batched:
            p_i = Ti[0:3, 3].view(3, 1)
            p_im1 = Tim1[0:3, 3].view(3, 1)
            r = p_i - p_im1
            r3 = r.view(3)
            ri = _cross_matrix(r3)

            Mi_top = torch.cat([I3, -ri], dim=1)
            Mi_bottom = torch.cat([Z3, I3], dim=1)
            Mi = torch.cat([Mi_top, Mi_bottom], dim=0)   # (6,6)

            z = Tim1[0:3, 2].view(3, 1)
            z3 = z.view(3)
            z_cross = _cross_matrix(z3)                  # (3,3)

            M_top = torch.cat([Z3, ri @ z_cross], dim=1)
            M_bottom = torch.cat([Z3, -z_cross], dim=1)
            M = torch.cat([M_top, M_bottom], dim=0)      # (6,6)

            qd_j = qd[i - 1, 0]
            qdd_j = qdd[i - 1, 0]

            V_i = V[i]                                   # (6,1)
            dV_joint = (M @ V_i) * qd_j

            vec6 = torch.cat([torch.zeros_like(z), z], dim=0)
            dV_joint = dV_joint + (Mi @ vec6) * qdd_j

            w_i = V_i[3:6, 0]  # (3,)
            c_lin = _coriolis_term(w_i, r3)  # (3,1)
            c = torch.cat([c_lin, torch.zeros_like(c_lin)], dim=0)  # (6,1)

            dV_i = Mi @ dV[-1] + dV_joint + c
            dV.append(dV_i)
        else:
            # batched
            p_i = Ti[:, 0:3, 3].unsqueeze(-1)     # (B,3,1)
            p_im1 = Tim1[:, 0:3, 3].unsqueeze(-1) # (B,3,1)
            r = p_i - p_im1                       # (B,3,1)
            r3 = r.squeeze(-1)                    # (B,3)
            ri = _cross_matrix(r3)                # (B,3,3)

            I3b = I3.expand(B, 3, 3)
            Z3b = Z3.expand(B, 3, 3)

            Mi_top = torch.cat([I3b, -ri], dim=2)      # (B,3,6)
            Mi_bottom = torch.cat([Z3b, I3b], dim=2)   # (B,3,6)
            Mi = torch.cat([Mi_top, Mi_bottom], dim=1) # (B,6,6)

            z = Tim1[:, 0:3, 2].unsqueeze(-1)          # (B,3,1)
            z3 = z.squeeze(-1)                         # (B,3)
            z_cross = _cross_matrix(z3)                # (B,3,3)

            M_top = torch.cat([Z3b, torch.matmul(ri, z_cross)], dim=2)  # (B,3,6)
            M_bottom = torch.cat([Z3b, -z_cross], dim=2)                # (B,3,6)
            M = torch.cat([M_top, M_bottom], dim=1)                     # (B,6,6)

            qd_j = qd[:, i - 1, 0].view(B, 1, 1)
            qdd_j = qdd[:, i - 1, 0].view(B, 1, 1)

            V_i = V[i]                             # (B,6,1)
            dV_joint = torch.matmul(M, V_i) * qd_j

            zeros_z = torch.zeros_like(z)
            vec6 = torch.cat([zeros_z, z], dim=1)  # (B,6,1)
            dV_joint = dV_joint + torch.matmul(Mi, vec6) * qdd_j

            w_i = V_i[:, 3:6, 0]                   # (B,3)
            c_lin = _coriolis_term(w_i, r3)        # (B,3,1)
            zeros_lin = torch.zeros_like(c_lin)
            c = torch.cat([c_lin, zeros_lin], dim=1)  # (B,6,1)

            dV_prev = dV[-1]                       # (B,6,1)
            dV_i = torch.matmul(Mi, dV_prev) + dV_joint + c
            dV.append(dV_i)

    return dV


# --------------------------------------------------------------------
# 3. Velocity / acceleration propagation to COMs (batchable)
# --------------------------------------------------------------------
def velocityPropagationCOM(
    robot: object,
    vCOM0: Tensor,
    wCOM0: Tensor,
    V: List[Tensor],
) -> List[Tensor]:
    """
    Propagate velocities to each COM.

    For each COM index j (0..n-1):
        - Find associated joint frame row via robot.where_is_joint(j)
        - Find COM frame via forwardCOMHTM(robot)[j+1]
        - Use standard relation:
              V_COM_j = M_j * V_joint + M_j * [0; z_j] * q̇_j
          with M_j built from the vector r = p_com - p_joint.

    Assumptions:
    - q, qd shapes:
        - unbatched: (n,1)
        - batched :  (B,n,1)
    - robot implements where_is_joint(j) and where_is_com(j) consistently
      with HTM_kinematics_torch.forwardCOMHTM.

    Args:
        vCOM0: base COM linear velocity (3,), (3,1), or (B,3)/(B,3,1)
        wCOM0: base COM angular velocity (3,), (3,1), or (B,3)/(B,3,1)
        V:     list of frame twists from velocityPropagation (length n+1)

    Returns:
        Vcom: list of COM twists:
              - unbatched: each Vcom[i] is (6,1)
              - batched  : each Vcom[i] is (B,6,1)
              Vcom[0] is for "base COM", Vcom[j+1] is COM j.
    """
    q = _q(robot)
    qd = _qd(robot)

    if q.dim() == 2 and q.shape[1] == 1:
        batched = False
        B = None
        n = q.shape[0]
    elif q.dim() == 3 and q.shape[2] == 1:
        batched = True
        B = q.shape[0]
        n = q.shape[1]
    else:
        raise NotImplementedError(
            f"velocityPropagationCOM expects q of shape (n,1) or (B,n,1), got {q.shape}"
        )

    if len(V) < 1:
        raise ValueError("V must contain at least the base twist.")

    Ts = KIN.forwardHTM(robot)     # frames
    Tcs = KIN.forwardCOMHTM(robot) # COM frames

    device = Ts[0].device
    dtype = Ts[0].dtype

    vCOM0 = _normalize_base_vec(vCOM0, B, device, dtype, "vCOM0")
    wCOM0 = _normalize_base_vec(wCOM0, B, device, dtype, "wCOM0")

    if batched:
        Vcom: List[Tensor] = [torch.cat([vCOM0, wCOM0], dim=1)]  # (B,6,1)
        I3 = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)  # (1,3,3)
        Z3 = torch.zeros((1, 3, 3), dtype=dtype, device=device)
    else:
        Vcom = [torch.cat([vCOM0, wCOM0], dim=0)]  # (6,1)
        I3 = torch.eye(3, dtype=dtype, device=device)
        Z3 = torch.zeros((3, 3), dtype=dtype, device=device)

    for j in range(n):
        row_joint, _ = robot.where_is_joint(j)
        row_com, _ = robot.where_is_com(j)  # not strictly needed but kept for API symmetry

        Tj = Ts[row_joint]
        Tc = Tcs[j + 1]

        if not batched:
            p_joint = Tj[0:3, 3].view(3, 1)
            p_com = Tc[0:3, 3].view(3, 1)
            r = p_com - p_joint
            r3 = r.view(3)
            ri = _cross_matrix(r3)

            Mi_top = torch.cat([I3, -ri], dim=1)
            Mi_bottom = torch.cat([Z3, I3], dim=1)
            Mi = torch.cat([Mi_top, Mi_bottom], dim=0)  # (6,6)

            z = Tj[0:3, 2].view(3, 1)
            qd_j = qd[j, 0]

            vec6 = torch.cat([torch.zeros_like(z), z], dim=0)
            vJoint = Mi @ (vec6 * qd_j)

            V_joint = V[row_joint]
            V_com = Mi @ V_joint + vJoint
            Vcom.append(V_com)
        else:
            p_joint = Tj[:, 0:3, 3].unsqueeze(-1)  # (B,3,1)
            p_com = Tc[:, 0:3, 3].unsqueeze(-1)    # (B,3,1)
            r = p_com - p_joint                    # (B,3,1)
            r3 = r.squeeze(-1)                     # (B,3)
            ri = _cross_matrix(r3)                 # (B,3,3)

            I3b = I3.expand(B, 3, 3)
            Z3b = Z3.expand(B, 3, 3)

            Mi_top = torch.cat([I3b, -ri], dim=2)      # (B,3,6)
            Mi_bottom = torch.cat([Z3b, I3b], dim=2)   # (B,3,6)
            Mi = torch.cat([Mi_top, Mi_bottom], dim=1) # (B,6,6)

            z = Tj[:, 0:3, 2].unsqueeze(-1)            # (B,3,1)
            qd_j = qd[:, j, 0].view(B, 1, 1)

            zeros_z = torch.zeros_like(z)
            vec6 = torch.cat([zeros_z, z], dim=1)      # (B,6,1)
            vJoint = torch.matmul(Mi, vec6 * qd_j)

            V_joint = V[row_joint]                     # (B,6,1)
            V_com = torch.matmul(Mi, V_joint) + vJoint
            Vcom.append(V_com)

    return Vcom


def accelerationPropagationCOM(
    robot: object,
    dvCOM0: Tensor,
    dwCOM0: Tensor,
    Vcom: List[Tensor],
    dV: List[Tensor],
) -> List[Tensor]:
    """
    Propagate accelerations to each COM.

    For each COM j:
        dV_COM_j = M_j dV_joint + (M_nl_j V_COM_j) * q̇_j
                   + M_j [0; z_j] q̈_j + coriolis_j

    with:
        coriolis_j = [ w_COM_j × (w_COM_j × r_j ); 0 ].

    Assumptions:
    - q, qd, qdd shapes:
        - unbatched: (n,1)
        - batched :  (B,n,1)
    - robot implements where_is_joint(j), where_is_com(j).
    - Vcom and dV come from velocityPropagationCOM and accelerationPropagation.

    Args:
        dvCOM0: base COM linear acceleration (3,), (3,1) or (B,3)/(B,3,1)
        dwCOM0: base COM angular acceleration (3,), (3,1) or (B,3)/(B,3,1)
        Vcom:   list of COM twists (length n+1)
        dV:     list of frame accelerations (length n+1)

    Returns:
        dVcom: list of COM accelerations (length n+1).
    """
    q = _q(robot)
    qd = _qd(robot)
    qdd = _qdd(robot)

    if q.dim() == 2 and q.shape[1] == 1:
        batched = False
        B = None
        n = q.shape[0]
    elif q.dim() == 3 and q.shape[2] == 1:
        batched = True
        B = q.shape[0]
        n = q.shape[1]
    else:
        raise NotImplementedError(
            f"accelerationPropagationCOM expects q of shape (n,1) or (B,n,1), got {q.shape}"
        )

    if len(Vcom) != n + 1:
        raise ValueError(f"Vcom must have length n+1={n+1}, got {len(Vcom)}.")
    if len(dV) != n + 1:
        raise ValueError(f"dV must have length n+1={n+1}, got {len(dV)}.")

    Ts = KIN.forwardHTM(robot)
    Tcs = KIN.forwardCOMHTM(robot)

    device = Ts[0].device
    dtype = Ts[0].dtype

    dvCOM0 = _normalize_base_vec(dvCOM0, B, device, dtype, "dvCOM0")
    dwCOM0 = _normalize_base_vec(dwCOM0, B, device, dtype, "dwCOM0")

    if batched:
        dVcom: List[Tensor] = [torch.cat([dvCOM0, dwCOM0], dim=1)]  # (B,6,1)
        I3 = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)  # (1,3,3)
        Z3 = torch.zeros((1, 3, 3), dtype=dtype, device=device)
    else:
        dVcom = [torch.cat([dvCOM0, dwCOM0], dim=0)]  # (6,1)
        I3 = torch.eye(3, dtype=dtype, device=device)
        Z3 = torch.zeros((3, 3), dtype=dtype, device=device)

    for j in range(n):
        row_joint, _ = robot.where_is_joint(j)
        row_com, _ = robot.where_is_com(j)

        Tj = Ts[row_joint]
        Tc = Tcs[j + 1]

        if not batched:
            p_joint = Tj[0:3, 3].view(3, 1)
            p_com = Tc[0:3, 3].view(3, 1)
            r = p_com - p_joint
            r3 = r.view(3)
            ri = _cross_matrix(r3)

            Mi_top = torch.cat([I3, -ri], dim=1)
            Mi_bottom = torch.cat([Z3, I3], dim=1)
            Mi = torch.cat([Mi_top, Mi_bottom], dim=0)   # (6,6)

            z = Tj[0:3, 2].view(3, 1)
            z3 = z.view(3)
            z_cross = _cross_matrix(z3)

            M_top = torch.cat([Z3, ri @ z_cross], dim=1)
            M_bottom = torch.cat([Z3, -z_cross], dim=1)
            M = torch.cat([M_top, M_bottom], dim=0)      # (6,6)

            qd_j = qd[j, 0]
            qdd_j = qdd[j, 0]

            V_com_j = Vcom[j + 1]                        # (6,1)
            dV_joint = (M @ V_com_j) * qd_j

            vec6 = torch.cat([torch.zeros_like(z), z], dim=0)
            dV_joint = dV_joint + (Mi @ vec6) * qdd_j

            w_com = V_com_j[3:6, 0]                     # (3,)
            c_lin = _coriolis_term(w_com, r3)           # (3,1)
            c = torch.cat([c_lin, torch.zeros_like(c_lin)], dim=0)

            dV_com_j = Mi @ dV[row_joint] + dV_joint + c
            dVcom.append(dV_com_j)
        else:
            p_joint = Tj[:, 0:3, 3].unsqueeze(-1)        # (B,3,1)
            p_com = Tc[:, 0:3, 3].unsqueeze(-1)         # (B,3,1)
            r = p_com - p_joint                         # (B,3,1)
            r3 = r.squeeze(-1)                          # (B,3)
            ri = _cross_matrix(r3)                      # (B,3,3)

            I3b = I3.expand(B, 3, 3)
            Z3b = Z3.expand(B, 3, 3)

            Mi_top = torch.cat([I3b, -ri], dim=2)       # (B,3,6)
            Mi_bottom = torch.cat([Z3b, I3b], dim=2)    # (B,3,6)
            Mi = torch.cat([Mi_top, Mi_bottom], dim=1)  # (B,6,6)

            z = Tj[:, 0:3, 2].unsqueeze(-1)             # (B,3,1)
            z3 = z.squeeze(-1)                          # (B,3)
            z_cross = _cross_matrix(z3)                 # (B,3,3)

            M_top = torch.cat([Z3b, torch.matmul(ri, z_cross)], dim=2)  # (B,3,6)
            M_bottom = torch.cat([Z3b, -z_cross], dim=2)                # (B,3,6)
            M = torch.cat([M_top, M_bottom], dim=1)                     # (B,6,6)

            qd_j = qd[:, j, 0].view(B, 1, 1)
            qdd_j = qdd[:, j, 0].view(B, 1, 1)

            V_com_j = Vcom[j + 1]                     # (B,6,1)
            dV_joint = torch.matmul(M, V_com_j) * qd_j

            zeros_z = torch.zeros_like(z)
            vec6 = torch.cat([zeros_z, z], dim=1)      # (B,6,1)
            dV_joint = dV_joint + torch.matmul(Mi, vec6) * qdd_j

            w_com = V_com_j[:, 3:6, 0]                 # (B,3)
            c_lin = _coriolis_term(w_com, r3)          # (B,3,1)
            zeros_lin = torch.zeros_like(c_lin)
            c = torch.cat([c_lin, zeros_lin], dim=1)   # (B,6,1)

            dV_joint_frame = dV[row_joint]             # (B,6,1)
            dV_com_j = torch.matmul(Mi, dV_joint_frame) + dV_joint + c
            dVcom.append(dV_com_j)

    return dVcom


# --------------------------------------------------------------------
# Simple smoke test (unbatched + batched)
# --------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    print("[DifferentialHTM_torch] Simple smoke test...")

    class DummyRobot:
        def __init__(self, n: int = 2, device="cpu"):
            self.device = torch.device(device)
            self.dtype = torch.float64
            self.dh = torch.zeros((n, 4), dtype=self.dtype, device=self.device)
            # simple planar: non-zero a
            self.dh[0, 2] = 0.3
            self.dh[1, 2] = 0.2
            self.dhCOM = self.dh.clone()
            self.dhCOM[:, 2] *= 0.5
            self.dh_convention = "standard"
            self.q = torch.zeros((n, 1), dtype=self.dtype, device=self.device)
            self.qd = torch.zeros_like(self.q)
            self.qdd = torch.zeros_like(self.q)

        def denavitHartenberg(self):
            return

        def denavitHartenbergCOM(self):
            return

        def where_is_joint(self, j: int) -> Tuple[int, int]:
            # frame index = joint index + 1 (since forwardHTM returns [T0, T1, ...])
            return j + 1, 0

        def where_is_com(self, j: int) -> Tuple[int, int]:
            # COM j shares DH row j in this dummy
            return j, 0

    # ---- Unbatched test ----
    rob = DummyRobot(n=2)
    rob.q = torch.tensor([[0.1], [0.2]], dtype=rob.dtype)
    rob.qd = torch.tensor([[0.3], [0.4]], dtype=rob.dtype)
    rob.qdd = torch.tensor([[0.5], [0.6]], dtype=rob.dtype)

    Xd_g = geometricStateSpace(rob)
    Xdd_g = geometricDerivativeStateSpace(rob)
    Xd_gc = geometricCOMStateSpace(rob, COM=0)
    Xdd_gc = geometricCOMDerivativeStateSpace(rob, COM=0)

    print("  [unbatched] geometricStateSpace Xd_g shape:", tuple(Xd_g.shape))
    print("  [unbatched] geometricDerivativeStateSpace Xdd_g shape:", tuple(Xdd_g.shape))
    print("  [unbatched] geometricCOMStateSpace Xd_gc shape:", tuple(Xd_gc.shape))
    print("  [unbatched] geometricCOMDerivativeStateSpace Xdd_gc shape:", tuple(Xdd_gc.shape))

    v0 = torch.zeros(3)
    w0 = torch.zeros(3)
    V = velocityPropagation(rob, v0, w0)
    dv0 = torch.zeros(3)
    dw0 = torch.zeros(3)
    dV = accelerationPropagation(rob, dv0, dw0, V)

    print("  [unbatched] #frames in V:", len(V), "| each twist shape:", tuple(V[0].shape))
    print("  [unbatched] #frames in dV:", len(dV), "| each acc shape:", tuple(dV[0].shape))

    vCOM0 = torch.zeros(3)
    wCOM0 = torch.zeros(3)
    dvCOM0 = torch.zeros(3)
    dwCOM0 = torch.zeros(3)

    Vcom = velocityPropagationCOM(rob, vCOM0, wCOM0, V)
    dVcom = accelerationPropagationCOM(rob, dvCOM0, dwCOM0, Vcom, dV)

    print("  [unbatched] #COM twists:", len(Vcom), "| shape:", tuple(Vcom[0].shape))
    print("  [unbatched] #COM accs:", len(dVcom), "| shape:", tuple(dVcom[0].shape))

    # ---- Batched test ----
    print("\n  [batched] test B=4")
    B = 4
    rob_b = DummyRobot(n=2)
    # upgrade dh / dhCOM to batched
    rob_b.dh = rob_b.dh.unsqueeze(0).expand(B, -1, -1).contiguous()
    rob_b.dhCOM = rob_b.dhCOM.unsqueeze(0).expand(B, -1, -1).contiguous()
    rob_b.q = torch.randn((B, 2, 1), dtype=rob_b.dtype)
    rob_b.qd = torch.randn((B, 2, 1), dtype=rob_b.dtype)
    rob_b.qdd = torch.randn((B, 2, 1), dtype=rob_b.dtype)

    Xd_g_b = geometricStateSpace(rob_b)
    Xdd_g_b = geometricDerivativeStateSpace(rob_b)
    Xd_gc_b = geometricCOMStateSpace(rob_b, COM=0)
    Xdd_gc_b = geometricCOMDerivativeStateSpace(rob_b, COM=0)

    print("  [batched] geometricStateSpace Xd_g_b shape:", tuple(Xd_g_b.shape))
    print("  [batched] geometricDerivativeStateSpace Xdd_g_b shape:", tuple(Xdd_g_b.shape))
    print("  [batched] geometricCOMStateSpace Xd_gc_b shape:", tuple(Xd_gc_b.shape))
    print("  [batched] geometricCOMDerivativeStateSpace Xdd_gc_b shape:", tuple(Xdd_gc_b.shape))

    v0_b = torch.zeros(3)  # will be broadcast to all B
    w0_b = torch.zeros(3)
    V_b = velocityPropagation(rob_b, v0_b, w0_b)
    dv0_b = torch.zeros(3)
    dw0_b = torch.zeros(3)
    dV_b = accelerationPropagation(rob_b, dv0_b, dw0_b, V_b)

    print("  [batched] #frames in V_b:", len(V_b), "| twist shape:", tuple(V_b[0].shape))
    print("  [batched] #frames in dV_b:", len(dV_b), "| acc shape:", tuple(dV_b[0].shape))

    vCOM0_b = torch.zeros(3)
    wCOM0_b = torch.zeros(3)
    dvCOM0_b = torch.zeros(3)
    dwCOM0_b = torch.zeros(3)

    Vcom_b = velocityPropagationCOM(rob_b, vCOM0_b, wCOM0_b, V_b)
    dVcom_b = accelerationPropagationCOM(rob_b, dvCOM0_b, dwCOM0_b, Vcom_b, dV_b)

    print("  [batched] #COM twists:", len(Vcom_b), "| shape:", tuple(Vcom_b[0].shape))
    print("  [batched] #COM accs:", len(dVcom_b), "| shape:", tuple(dVcom_b[0].shape))

    print("\n[DifferentialHTM_torch] Smoke test done.")
