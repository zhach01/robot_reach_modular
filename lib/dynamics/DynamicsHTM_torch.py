# DynamicsHTM_torch.py
"""
Pure-PyTorch dynamics utilities based on HTM_kinematics_torch.

This is the Torch-only rewrite of DynamicsHTM.py:
- No NumPy, no SymPy, no `symbolic` flag.
- Uses HTM_kinematics_torch + HTM_torch primitives.
- Fully differentiable, GPU/CPU-safe, and batchable.

Additions vs first Torch version:
- Damped Least-Squares pseudoinverse _pinv_dls_torch (used in M, G_cart, N).
- Two Coriolis/Centrifugal implementations:
    * method="finite_diff"  (batchable, numeric derivative of D(q))
    * method="analytic"     (Torch autograd Christoffel from D(q), batchable via loop)
- Smoke test compares both implementations for consistency.

Robot interface (Torch version)
-------------------------------
We assume a Serial-like robot object exposing:

- Kinematics:
    robot.dh            : (n,4) or (B,n,4) DH table
    robot.dhCOM         : (n,4) or (B,n,4) COM DH table
    robot.dh_convention : "standard" or "modified"
    robot.q             : (n,1) or (B,n,1) joint positions
    robot.qd            : (n,1) or (B,n,1) joint velocities
    robot.qdd           : (n,1) or (B,n,1) joint accelerations

- Inertial params:
    robot.mass          : list[float] or 1D torch.Tensor, length = n_links
    robot.inertia       : list[3x3-like] or torch.Tensor (n_links,3,3)
    robot.COMs          : list[...] (only used for link count here)

- Optional:
    robot.where_is_joint(j) -> (frame_index, offset)
    robot.where_is_com(j)   -> (row_index, offset)

All functions here are Torch-only.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

try:
    from lib.kinematics import HTM_kinematics_torch as KIN
except Exception:  # fallback if run standalone
    import lib.kinematics.HTM_kinematics_torch as KIN

try:
    from lib.movements import HTM_torch as HTM
except Exception:
    import  lib.movements.HTM_torch as HTM


# --------------------------------------------------------------------
# Basic helpers
# --------------------------------------------------------------------
def _q(robot) -> Tensor:
    if not hasattr(robot, "q"):
        raise AttributeError("robot must expose attribute 'q' as torch.Tensor")
    return robot.q


def _qd(robot) -> Tensor:
    if not hasattr(robot, "qd"):
        raise AttributeError("robot must expose attribute 'qd' as torch.Tensor")
    return robot.qd


def _qdd(robot) -> Tensor:
    if not hasattr(robot, "qdd"):
        raise AttributeError("robot must expose attribute 'qdd' as torch.Tensor")
    return robot.qdd


def _ensure_col(vec: Tensor) -> Tensor:
    """
    Ensure a (n,) or (n,1) vector is shaped as column (n,1).
    For batched input (B,n) or (B,n,1) -> (B,n,1).
    """
    if vec.dim() == 1:
        return vec.view(-1, 1)
    if vec.dim() == 2:
        # (n,1) or (B,n)
        if vec.shape[1] == 1:
            return vec
        return vec.unsqueeze(-1)  # (B,n,1)
    if vec.dim() == 3:
        return vec
    raise ValueError(f"Unsupported vector shape {vec.shape} in _ensure_col")


def _get_batch_info(q: Tensor) -> Tuple[bool, int | None, int]:
    """
    Return (batched, B, n) from q shape.
    q is (n,1) or (B,n,1).
    """
    if q.dim() == 2 and q.shape[1] == 1:
        return False, None, q.shape[0]
    if q.dim() == 3 and q.shape[2] == 1:
        return True, q.shape[0], q.shape[1]
    raise NotImplementedError(
        f"Expected q of shape (n,1) or (B,n,1), got {q.shape}"
    )


def _get_link_count(robot) -> int:
    if hasattr(robot, "mass"):
        return len(robot.mass)
    if hasattr(robot, "inertia"):
        return len(robot.inertia)
    if hasattr(robot, "COMs"):
        return len(robot.COMs)
    raise AttributeError("robot must define mass, inertia or COMs to infer link count")


def _get_mass_tensor(robot, device, dtype) -> Tensor:
    """
    Return mass as 1D tensor of length n_links.
    """
    if not hasattr(robot, "mass"):
        raise AttributeError("robot must expose 'mass' (list or torch.Tensor)")

    m = robot.mass
    if isinstance(m, Tensor):
        return m.to(device=device, dtype=dtype).view(-1)
    return torch.as_tensor(m, dtype=dtype, device=device).view(-1)


def _get_inertia_tensor(robot, device, dtype) -> Tensor:
    """
    Return inertia as tensor of shape (n_links,3,3).
    """
    if not hasattr(robot, "inertia"):
        raise AttributeError("robot must expose 'inertia' (list or torch.Tensor)")

    I = robot.inertia
    if isinstance(I, Tensor):
        if I.dim() == 3 and I.shape[1:] == (3, 3):
            return I.to(device=device, dtype=dtype)
        raise ValueError(
            f"robot.inertia tensor must have shape (n_links,3,3), got {I.shape}"
        )

    mats = []
    for M in I:
        mats.append(torch.as_tensor(M, dtype=dtype, device=device).view(3, 3))
    return torch.stack(mats, dim=0)


def _get_gravity(
    g, device, dtype, batched: bool, B: int | None
) -> Tensor:
    """
    Normalize gravity vector.

    Returns:
        - (3,1)   for unbatched
        - (B,3,1) for batched
    """
    if g is None:
        g_vec = torch.tensor([0.0, 0.0, -9.80665], dtype=dtype, device=device)
    else:
        g_vec = torch.as_tensor(g, dtype=dtype, device=device).view(-1)
    if g_vec.numel() != 3:
        raise ValueError(f"gravity must have 3 elements, got {g_vec.shape}")

    if not batched:
        return g_vec.view(3, 1)
    return g_vec.view(1, 3, 1).expand(B, 3, 1)


# --------------------------------------------------------------------
# Damped Least-Squares pseudoinverse (Torch)
# --------------------------------------------------------------------
def _pinv_dls_torch(J: Tensor, mu: float = 1e-8) -> Tensor:
    """
    Damped Least-Squares pseudoinverse.

    For a matrix J (m x n):

        if m >= n (tall):
            J⁺ = (Jᵀ J + μ² I_n)^(-1) Jᵀ

        if m < n (wide):
            J⁺ = Jᵀ (J Jᵀ + μ² I_m)^(-1)

    Supports:
        - J: (m,n)
        - J: (B,m,n)
    """
    if J.dim() == 2:
        m, n = J.shape
        JT = J.transpose(0, 1)
        if m >= n:
            A = JT @ J
            A = A + (mu**2) * torch.eye(n, dtype=J.dtype, device=J.device)
            # Solve A X = JT -> X = A^{-1} JT
            X = torch.linalg.solve(A, JT)  # (n,m)
            return X
        else:
            A = J @ JT
            A = A + (mu**2) * torch.eye(m, dtype=J.dtype, device=J.device)
            I_m = torch.eye(m, dtype=J.dtype, device=J.device)
            X = torch.linalg.solve(A, I_m)  # (m,m) = (J Jᵀ + μ²I)^(-1)
            return JT @ X  # (n,m)

    if J.dim() == 3:
        B, m, n = J.shape
        JT = J.transpose(1, 2)  # (B,n,m)
        if m >= n:
            # A = Jᵀ J + μ² I_n
            A = JT @ J  # (B,n,n)
            I = torch.eye(n, dtype=J.dtype, device=J.device).expand(B, n, n)
            A = A + (mu**2) * I
            # Solve A X = JT -> X: (B,n,m)
            X = torch.linalg.solve(A, JT)
            return X
        else:
            # A = J Jᵀ + μ² I_m
            A = J @ JT  # (B,m,m)
            I = torch.eye(m, dtype=J.dtype, device=J.device).expand(B, m, m)
            A = A + (mu**2) * I
            # X = A^{-1}
            I_B = torch.eye(m, dtype=J.dtype, device=J.device).expand(B, m, m)
            X = torch.linalg.solve(A, I_B)  # (B,m,m)
            return JT @ X  # (B,n,m)

    raise ValueError(f"_pinv_dls_torch: unsupported shape {J.shape}")


# --------------------------------------------------------------------
# 1. Joint-space inertia and energies
# --------------------------------------------------------------------
def inertiaMatrixCOM(robot: object) -> Tensor:
    """
    D(q): Joint-space inertia matrix about each COM.

    Returns:
        D:
          - (n,n)   for unbatched robot
          - (B,n,n) for batched robot
    """
    q = _ensure_col(_q(robot))
    batched, B, n = _get_batch_info(q)

    device = q.device
    dtype = q.dtype

    # COM forward kinematics
    Tcs = KIN.forwardCOMHTM(robot)  # list; elements (4,4) or (B,4,4)

    n_links = _get_link_count(robot)
    mass = _get_mass_tensor(robot, device, dtype)          # (n_links,)
    inertia = _get_inertia_tensor(robot, device, dtype)    # (n_links,3,3)

    if batched:
        D = torch.zeros((B, n, n), dtype=dtype, device=device)
    else:
        D = torch.zeros((n, n), dtype=dtype, device=device)

    for j in range(n_links):
        JgCOM = KIN.geometricJacobianCOM(robot, COM=j)  # (6,n) or (B,6,n)

        if not batched:
            Jv = JgCOM[0:3, :]      # (3,n)
            Jw = JgCOM[3:6, :]      # (3,n)

            Tc = Tcs[j + 1]         # (4,4)
            R = Tc[0:3, 0:3]        # (3,3)
            I_link = inertia[j]     # (3,3)

            Icom = R.T @ I_link @ R  # (3,3)

            D += mass[j] * (Jv.T @ Jv) + (Jw.T @ Icom @ Jw)
        else:
            Jv = JgCOM[:, 0:3, :]   # (B,3,n)
            Jw = JgCOM[:, 3:6, :]   # (B,3,n)

            Tc = Tcs[j + 1]         # (B,4,4)
            R = Tc[:, 0:3, 0:3]     # (B,3,3)

            I_link = inertia[j]     # (3,3)
            I_b = I_link.unsqueeze(0).expand(B, 3, 3)

            Icom = torch.matmul(R.transpose(1, 2), torch.matmul(I_b, R))  # (B,3,3)

            JvT = Jv.transpose(1, 2)  # (B,n,3)
            JwT = Jw.transpose(1, 2)  # (B,n,3)

            term_lin = torch.matmul(JvT, Jv)                   # (B,n,n)
            term_ang = torch.matmul(JwT, torch.matmul(Icom, Jw))  # (B,n,n)

            D += mass[j] * term_lin + term_ang

    return D


def inertiaMatrixCartesian(robot: object, dls_mu: float = 1e-8) -> Tensor:
    """
    M(q): Cartesian-space inertia matrix.

    M = (J⁺)ᵀ D J⁺, where J⁺ is a DLS pseudoinverse of J.

    Returns:
        M:
          - (6,6)   for unbatched
          - (B,6,6) for batched
    """
    q = _ensure_col(_q(robot))
    batched, B, n = _get_batch_info(q)

    D = inertiaMatrixCOM(robot)           # (n,n) or (B,n,n)
    Jg = KIN.geometricJacobian(robot)     # (6,n) or (B,6,n)

    if not batched:
        Jinv = _pinv_dls_torch(Jg, mu=dls_mu)  # (n,6)
        return Jinv.T @ D @ Jinv               # (6,6)

    # batched
    Jinv = _pinv_dls_torch(Jg, mu=dls_mu)      # (B,n,6)
    DJ = torch.matmul(D, Jinv)                 # (B,n,6)
    M = torch.matmul(Jinv.transpose(1, 2), DJ)  # (B,6,6)
    return M


def kineticEnergyCOM(robot: object) -> Tensor:
    """
    K(q, q̇) = 1/2 q̇ᵀ D(q) q̇.

    Returns:
        - scalar tensor () for unbatched
        - (B,) for batched
    """
    q = _ensure_col(_q(robot))
    batched, B, n = _get_batch_info(q)
    D = inertiaMatrixCOM(robot)
    qd = _ensure_col(_qd(robot))   # (n,1) or (B,n,1)

    if not batched:
        Dqd = D @ qd               # (n,1)
        K = 0.5 * (qd.view(1, n) @ Dqd).squeeze()
        return K
    else:
        Dqd = torch.matmul(D, qd)                # (B,n,1)
        qdT = qd.transpose(1, 2)                 # (B,1,n)
        K = 0.5 * torch.matmul(qdT, Dqd).squeeze(-1).squeeze(-1)  # (B,)
        return K


def potentialEnergyCOM(
    robot: object,
    g=None,
) -> Tensor:
    """
    P(q) = Σ_j m_j gᵀ r_j  (sum over COMs j).

    Args:
        g: gravity vector; if None, defaults to [0, 0, -9.80665].

    Returns:
        - scalar () for unbatched
        - (B,) for batched
    """
    q = _ensure_col(_q(robot))
    batched, B, n = _get_batch_info(q)

    device = q.device
    dtype = q.dtype

    Tcs = KIN.forwardCOMHTM(robot)  # list; element 0 is base, 1.. are COMs
    n_links = _get_link_count(robot)

    mass = _get_mass_tensor(robot, device, dtype)  # (n_links,)
    g_t = _get_gravity(g, device, dtype, batched, B)

    if not batched:
        P = torch.zeros((), dtype=dtype, device=device)
        for j in range(n_links):
            Tc = Tcs[j + 1]               # (4,4)
            r = Tc[0:3, 3].view(3, 1)     # (3,1)
            P += mass[j] * (g_t.view(1, 3) @ r).squeeze()
        return P
    else:
        P = torch.zeros((B,), dtype=dtype, device=device)
        for j in range(n_links):
            Tc = Tcs[j + 1]                   # (B,4,4)
            r = Tc[:, 0:3, 3].unsqueeze(-1)   # (B,3,1)
            gt = g_t.transpose(1, 2)          # (B,1,3)
            Pg = torch.matmul(gt, r).squeeze(-1).squeeze(-1)  # (B,)
            P += mass[j] * Pg
        return P


# --------------------------------------------------------------------
# 2. Gravity vectors
# --------------------------------------------------------------------
def gravitationalCOM(
    robot: object,
    g=None,
) -> Tensor:
    """
    G(q): Joint-space gravity vector.

         G = Σ_j m_j (Jv_com_j)ᵀ g

    where Jv_com_j is the linear-velocity block of the COM Jacobian.

    Args:
        g: gravity vector (defaults to [0,0,-9.80665]).

    Returns:
        - (n,1)   for unbatched
        - (B,n,1) for batched
    """
    q = _ensure_col(_q(robot))
    batched, B, n = _get_batch_info(q)

    device = q.device
    dtype = q.dtype

    n_links = _get_link_count(robot)
    mass = _get_mass_tensor(robot, device, dtype)
    g_t = _get_gravity(g, device, dtype, batched, B)

    if not batched:
        G = torch.zeros((n, 1), dtype=dtype, device=device)
        for j in range(n_links):
            JgCOM = KIN.geometricJacobianCOM(robot, COM=j)  # (6,n)
            Jv = JgCOM[0:3, :]                              # (3,n)
            term = Jv.T @ g_t                               # (n,1)
            G += mass[j] * term
        return G
    else:
        G = torch.zeros((B, n, 1), dtype=dtype, device=device)
        for j in range(n_links):
            JgCOM = KIN.geometricJacobianCOM(robot, COM=j)  # (B,6,n)
            Jv = JgCOM[:, 0:3, :]                           # (B,3,n)
            JvT = Jv.transpose(1, 2)                        # (B,n,3)
            term = torch.matmul(JvT, g_t)                   # (B,n,1)
            G += mass[j] * term
        return G


def gravitationalCartesian(
    robot: object,
    g=None,
    dls_mu: float = 1e-8,
) -> Tensor:
    """
    G_cart(q): Cartesian-space gravity vector at the end-effector.

    Approximated as:
        G_cart = (J⁺)ᵀ G_joint

    where J⁺ is a DLS pseudoinverse of J.

    Returns:
        - (6,1)   for unbatched
        - (B,6,1) for batched
    """
    q = _ensure_col(_q(robot))
    batched, B, n = _get_batch_info(q)

    G = gravitationalCOM(robot, g)      # (n,1) or (B,n,1)
    Jg = KIN.geometricJacobian(robot)   # (6,n) or (B,6,n)

    if not batched:
        Jinv = _pinv_dls_torch(Jg, mu=dls_mu)    # (n,6)
        return Jinv.T @ G                        # (6,1)

    Jinv = _pinv_dls_torch(Jg, mu=dls_mu)        # (B,n,6)
    Gc = torch.matmul(Jinv.transpose(1, 2), G)   # (B,6,1)
    return Gc


# --------------------------------------------------------------------
# 3. Coriolis / Centrifugal matrices
# --------------------------------------------------------------------
def _coriolis_christoffel_from_D_jac(
    vecD: Tensor, jac: Tensor, qd: Tensor
) -> Tensor:
    """
    Build C(q,q̇) from D(q) and ∂vec(D)/∂q using Christoffel formula.

    Args:
        vecD: (n*n,)   flattened D matrix
        jac : (n*n,n) jacobian of vecD wrt q
        qd  : (n,1)   joint velocities

    Returns:
        C: (n,n)
    """
    n2, n = jac.shape
    n_sqrt = int(n2 ** 0.5)
    if n_sqrt * n_sqrt != n2:
        raise ValueError(f"vecD length {n2} is not a perfect square")

    n = n_sqrt
    qd_flat = qd.view(-1)
    C = vecD.new_zeros((n, n))

    for i in range(n):
        for j in range(n):
            cij = 0.0
            idx_ij = i * n + j
            for k in range(n):
                idx_ik = i * n + k
                idx_jk = j * n + k
                dDij_dqk = jac[idx_ij, k]
                dDik_dqj = jac[idx_ik, j]
                dDjk_dqi = jac[idx_jk, i]
                cij = cij + 0.5 * (
                    dDij_dqk + dDik_dqj - dDjk_dqi
                ) * qd_flat[k]
            C[i, j] = cij
    return C


def _coriolis_christoffel_unbatched(robot: object) -> Tensor:
    """
    Analytic Coriolis/Centrifugal C(q,q̇) from D(q) using Torch autograd
    (Christoffel symbols). Unbatched version (q: (n,1)).
    """
    q = _ensure_col(_q(robot))   # (n,1)
    qd = _ensure_col(_qd(robot))  # (n,1)
    _, _, n = _get_batch_info(q)

    # Ensure q participates in autograd graph
    q_flat = q.view(-1)
    if not q_flat.requires_grad:
        q_flat.requires_grad_()

    def f(q_flat_local: Tensor) -> Tensor:
        # q_flat_local: (n,)
        q_local = q_flat_local.view_as(q)   # (n,1)
        # Override robot.q for this call
        robot.q = q_local
        D = inertiaMatrixCOM(robot)         # (n,n)
        return D.reshape(-1)                # (n*n,)

    # vecD at current q
    vecD = f(q_flat)  # (n*n,)

    # jacobian of vecD wrt q_flat: shape (n*n, n)
    J = torch.autograd.functional.jacobian(
        f, q_flat, create_graph=True, vectorize=False
    )

    # Restore robot.q
    robot.q = q

    C = _coriolis_christoffel_from_D_jac(vecD, J, qd)
    return C


def _coriolis_christoffel_torch(robot: object) -> Tensor:
    """
    Analytic Coriolis/Centrifugal C(q,q̇) from D(q) using Torch autograd
    (Christoffel symbols). Handles both unbatched and batched robots.

    Batched case is handled by looping over batch dimension and treating
    each sample as an unbatched robot (q: (n,1), dh sliced if needed).
    """
    q = _ensure_col(_q(robot))
    batched, B, n = _get_batch_info(q)

    if not batched:
        return _coriolis_christoffel_unbatched(robot)

    # batched: q is (B,n,1)
    q_all = q
    qd_all = _ensure_col(_qd(robot))  # (B,n,1)

    # dh, dhCOM may be (n,4) or (B,n,4)
    dh_all = getattr(robot, "dh", None)
    dhCOM_all = getattr(robot, "dhCOM", None)

    device = q_all.device
    dtype = q_all.dtype
    C_all = torch.zeros((B, n, n), dtype=dtype, device=device)

    for b in range(B):
        # slice the state for batch b
        robot.q = q_all[b]   # (n,1)
        robot.qd = qd_all[b]  # (n,1)

        if isinstance(dh_all, Tensor) and dh_all.dim() == 3:
            robot.dh = dh_all[b]        # (n,4)
        if isinstance(dhCOM_all, Tensor) and dhCOM_all.dim() == 3:
            robot.dhCOM = dhCOM_all[b]  # (n,4)

        C_b = _coriolis_christoffel_unbatched(robot)  # (n,n)
        C_all[b] = C_b

    # restore original
    robot.q = q_all
    robot.qd = qd_all
    if dh_all is not None:
        robot.dh = dh_all
    if dhCOM_all is not None:
        robot.dhCOM = dhCOM_all

    return C_all


def centrifugalCoriolisCOM(
    robot: object,
    dq: float = 1e-3,
    method: str = "finite_diff",
) -> Tensor:
    """
    C(q, q̇): Coriolis / centrifugal matrix (joint space).

    method:
        - "finite_diff" (default):
            Original numeric derivative path on D(q) (batchable).
        - "analytic":
            Christoffel-based from D(q) using Torch autograd (batchable via loop).

    Returns:
        C:
          - (n,n)   for unbatched
          - (B,n,n) for batched
    """
    if method == "analytic":
        return _coriolis_christoffel_torch(robot)

    if method != "finite_diff":
        raise NotImplementedError(
            f"centrifugalCoriolisCOM: unknown method '{method}' "
            "(use 'finite_diff' or 'analytic')"
        )

    # ---- finite-difference implementation ----
    q = _ensure_col(_q(robot))
    batched, B, n = _get_batch_info(q)
    device = q.device
    dtype = q.dtype

    D_base = inertiaMatrixCOM(robot)  # (n,n) or (B,n,n)
    qd = _ensure_col(_qd(robot))      # (n,1) or (B,n,1)

    if not batched:
        C = torch.zeros((n, n), dtype=dtype, device=device)
        q_orig = q.clone()

        for j in range(n):
            V = torch.zeros((n, n), dtype=dtype, device=device)
            for k in range(n):
                q_plus = q_orig.clone()
                q_plus[k, 0] = q_plus[k, 0] + dq
                robot.q = q_plus
                D_plus = inertiaMatrixCOM(robot)           # (n,n)
                V[:, k] = (D_plus[:, j] - D_base[:, j]) / dq
            # restore
            robot.q = q_orig

            C += (V - 0.5 * V.T) * qd[j, 0]

        return C

    # batched
    C = torch.zeros((B, n, n), dtype=dtype, device=device)
    q_orig = q.clone()

    for j in range(n):
        V = torch.zeros((B, n, n), dtype=dtype, device=device)
        for k in range(n):
            q_plus = q_orig.clone()
            q_plus[:, k, 0] = q_plus[:, k, 0] + dq
            robot.q = q_plus
            D_plus = inertiaMatrixCOM(robot)               # (B,n,n)
            V[:, :, k] = (D_plus[:, :, j] - D_base[:, :, j]) / dq

        robot.q = q_orig

        skew = V - 0.5 * V.transpose(1, 2)
        qd_j = qd[:, j, 0].view(B, 1, 1)
        C += skew * qd_j

    return C


def centrifugalCoriolisCartesian(
    robot: object,
    dq: float = 1e-3,
    method: str = "finite_diff",
    dls_mu: float = 1e-8,
) -> Tensor:
    """
    N(q, q̇): Coriolis / centrifugal matrix in Cartesian space.

    N = (J⁺)ᵀ C J⁺ - M dJ

    where:
        - C is joint-space Coriolis/Centrifugal (finite_diff or analytic)
        - J is geometric Jacobian
        - dJ is its time derivative (geometricJacobianDerivative)
        - M is Cartesian inertia matrix
        - J⁺ is DLS pseudoinverse of J.

    Args:
        dq: finite-difference step for C(q, q̇) if method='finite_diff'
        method: 'finite_diff' or 'analytic'

    Returns:
        N:
          - (6,6)   for unbatched
          - (B,6,6) for batched
    """
    q = _ensure_col(_q(robot))
    batched, B, n = _get_batch_info(q)

    M = inertiaMatrixCartesian(robot, dls_mu=dls_mu)              # (6,6) or (B,6,6)
    C = centrifugalCoriolisCOM(robot, dq=dq, method=method)       # (n,n) or (B,n,n)
    Jg = KIN.geometricJacobian(robot)                             # (6,n) or (B,6,n)
    dJg = KIN.geometricJacobianDerivative(robot)                  # (6,n) or (B,6,n)

    if not batched:
        Jinv = _pinv_dls_torch(Jg, mu=dls_mu)                      # (n,6)
        term1 = Jinv.T @ C                                         # (6,n)
        term2 = M @ dJg                                            # (6,n)
        N = (term1 - term2) @ Jinv                                 # (6,6)
        return N

    Jinv = _pinv_dls_torch(Jg, mu=dls_mu)                          # (B,n,6)
    term1 = torch.matmul(Jinv.transpose(1, 2), C)                  # (B,6,n)
    term2 = torch.matmul(M, dJg)                                   # (B,6,n)
    N = torch.matmul(term1 - term2, Jinv)                          # (B,6,6)
    return N


# Backward-compat alias name
centrifugalCoriolisCOMOLD = centrifugalCoriolisCOM


# --------------------------------------------------------------------
# Simple smoke test (unbatched + batched, FD vs analytic)
# --------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    print("[DynamicsHTM_torch] Simple smoke test...")

    class DummyRobot:
        def __init__(self, n: int = 2, device="cpu"):
            self.device = torch.device(device)
            self.dtype = torch.float64

            # Basic DH for planar 2-dof
            self.dh = torch.zeros((n, 4), dtype=self.dtype, device=self.device)
            self.dh[0, 2] = 0.3
            self.dh[1, 2] = 0.2
            self.dhCOM = self.dh.clone()
            self.dhCOM[:, 2] *= 0.5
            self.dh_convention = "standard"

            # Joints
            self.q = torch.zeros((n, 1), dtype=self.dtype, device=self.device)
            self.qd = torch.zeros_like(self.q)
            self.qdd = torch.zeros_like(self.q)

            # Inertial parameters
            self.mass = [1.0, 1.0]
            I1 = torch.diag(torch.tensor([0.01, 0.01, 0.01], dtype=self.dtype))
            I2 = torch.diag(torch.tensor([0.01, 0.01, 0.01], dtype=self.dtype))
            self.inertia = [I1, I2]
            self.COMs = [0.15, 0.10]

        # dummy stubs to satisfy kinematics helpers if needed
        def denavitHartenberg(self, symbolic: bool = False):
            return

        def denavitHartenbergCOM(self, symbolic: bool = False):
            return

        def where_is_joint(self, j: int):
            return j + 1, 0

        def where_is_com(self, j: int):
            return j, 0

    # ----- Unbatched -----
    rob = DummyRobot(n=2)
    rob.q = torch.tensor([[0.1], [0.2]], dtype=rob.dtype)
    rob.qd = torch.tensor([[0.3], [0.4]], dtype=rob.dtype)
    rob.qdd = torch.tensor([[0.5], [0.6]], dtype=rob.dtype)

    D = inertiaMatrixCOM(rob)
    M = inertiaMatrixCartesian(rob)
    K = kineticEnergyCOM(rob)
    P = potentialEnergyCOM(rob)
    G = gravitationalCOM(rob)
    Gc = gravitationalCartesian(rob)
    C_fd = centrifugalCoriolisCOM(rob, dq=1e-4, method="finite_diff")
    C_an = centrifugalCoriolisCOM(rob, method="analytic")
    N_fd = centrifugalCoriolisCartesian(rob, dq=1e-4, method="finite_diff")
    N_an = centrifugalCoriolisCartesian(rob, method="analytic")

    print("  [unbatched] D shape:", tuple(D.shape))
    print("  [unbatched] M shape:", tuple(M.shape))
    print("  [unbatched] K shape:", K.shape)
    print("  [unbatched] P shape:", P.shape)
    print("  [unbatched] G shape:", tuple(G.shape))
    print("  [unbatched] G_cart shape:", tuple(Gc.shape))
    print("  [unbatched] C_fd shape:", tuple(C_fd.shape))
    print("  [unbatched] C_an shape:", tuple(C_an.shape))
    print("  [unbatched] N_fd shape:", tuple(N_fd.shape))
    print("  [unbatched] N_an shape:", tuple(N_an.shape))
    print(
        "  [unbatched] max|C_fd - C_an| =",
        float((C_fd - C_an).abs().max()),
    )
    print(
        "  [unbatched] max|N_fd - N_an| =",
        float((N_fd - N_an).abs().max()),
    )

    # ----- Batched -----
    print("\n  [batched] test B=4")
    B = 4
    rob_b = DummyRobot(n=2)
    rob_b.dh = rob_b.dh.unsqueeze(0).expand(B, -1, -1).contiguous()
    rob_b.dhCOM = rob_b.dhCOM.unsqueeze(0).expand(B, -1, -1).contiguous()
    rob_b.q = torch.randn((B, 2, 1), dtype=rob_b.dtype)
    rob_b.qd = torch.randn((B, 2, 1), dtype=rob_b.dtype)
    rob_b.qdd = torch.randn((B, 2, 1), dtype=rob_b.dtype)

    D_b = inertiaMatrixCOM(rob_b)
    M_b = inertiaMatrixCartesian(rob_b)
    K_b = kineticEnergyCOM(rob_b)
    P_b = potentialEnergyCOM(rob_b)
    G_b = gravitationalCOM(rob_b)
    Gc_b = gravitationalCartesian(rob_b)
    C_fd_b = centrifugalCoriolisCOM(rob_b, dq=1e-4, method="finite_diff")
    C_an_b = centrifugalCoriolisCOM(rob_b, method="analytic")
    N_fd_b = centrifugalCoriolisCartesian(rob_b, dq=1e-4, method="finite_diff")
    N_an_b = centrifugalCoriolisCartesian(rob_b, method="analytic")

    print("  [batched] D_b shape:", tuple(D_b.shape))
    print("  [batched] M_b shape:", tuple(M_b.shape))
    print("  [batched] K_b shape:", tuple(K_b.shape))
    print("  [batched] P_b shape:", tuple(P_b.shape))
    print("  [batched] G_b shape:", tuple(G_b.shape))
    print("  [batched] Gc_b shape:", tuple(Gc_b.shape))
    print("  [batched] C_fd_b shape:", tuple(C_fd_b.shape))
    print("  [batched] C_an_b shape:", tuple(C_an_b.shape))
    print("  [batched] N_fd_b shape:", tuple(N_fd_b.shape))
    print("  [batched] N_an_b shape:", tuple(N_an_b.shape))
    print(
        "  [batched] max|C_fd_b - C_an_b| =",
        float((C_fd_b - C_an_b).abs().max()),
    )
    print(
        "  [batched] max|N_fd_b - N_an_b| =",
        float((N_fd_b - N_an_b).abs().max()),
    )

    print("\n[DynamicsHTM_torch] Smoke test done.")
