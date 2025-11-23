# Robot_torch.py
# -*- coding: utf-8 -*-
"""
Pure-Torch Serial robot container with DH tables and COM inertias.

- 0-based conventions (same as lib/Robot.py):
    * DH rows: base row at index 0; joint j (0-based) => row j+1
    * Columns: [θ, d, a, α]
    * dh_convention: "standard" or "modified" (stored but not used here)

- Torch-only:
    * No NumPy
    * No SymPy
    * Fully differentiable and GPU-safe

- Batchable:
    * q, qd, qdd can be (n,1) or (B,n,1)

- Inertia per link i:
    Provide either:
      * 3×3 inertia about COM in link-i frame (torch.Tensor-like)
      * OR adapter dict:
            {
              "I": I_given,
              "m": m_i,
              "R_given_to_link": R,   # (optional, defaults to eye)
              "r_P_to_COM": r        # (optional, defaults to 0; vector from P->COM)
            }
    Which is canonicalized to self.I_com[i] (3×3) = inertia about COM in link frame,
    symmetric and PSD-checked.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Dict, Union

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers (Torch-only)
# ---------------------------------------------------------------------------

def _skew_torch(v: Tensor) -> Tensor:
    """
    Return the 3×3 skew-symmetric matrix [v]_x for v in R^3.
    """
    v = v.reshape(3)
    x, y, z = v[0], v[1], v[2]
    z0 = torch.zeros((), dtype=v.dtype, device=v.device)
    return torch.stack(
        [
            torch.stack([z0, -z,  y], dim=-1),
            torch.stack([z,  z0, -x], dim=-1),
            torch.stack([-y,  x,  z0], dim=-1),
        ],
        dim=-2,
    )


def _check_sym_torch(I: Tensor, name: str = "I", eps: float = 1e-10) -> None:
    """
    Check symmetry of inertia matrix I (Torch).
    """
    if not torch.is_tensor(I):
        I = torch.as_tensor(I, dtype=torch.float64)
    diff = I - I.transpose(-1, -2)
    atol = torch.as_tensor(eps, dtype=I.dtype, device=I.device)
    if not torch.allclose(I, I.transpose(-1, -2), rtol=0.0, atol=atol):
        raise ValueError(
            f"{name} not symmetric; max|I-I^T|={float(diff.abs().max()):.3e}"
        )


def _check_psd_torch(I: Tensor, name: str = "I", eps: float = 1e-10) -> None:
    """
    Check positive semi-definiteness of inertia matrix I (Torch).
    """
    if not torch.is_tensor(I):
        I = torch.as_tensor(I, dtype=torch.float64)
    Is = 0.5 * (I + I.transpose(-1, -2))
    w = torch.linalg.eigvalsh(Is)
    if torch.min(w) < -1e-8:
        raise ValueError(
            f"{name} not PSD; min eigenvalue={float(torch.min(w)):.3e}"
        )


# ---------------------------------------------------------------------------
# Base Robot
# ---------------------------------------------------------------------------

class Robot:
    def __init__(self, name: str = ""):
        self.name = name


# ---------------------------------------------------------------------------
# Serial robot (Torch-only)
# ---------------------------------------------------------------------------

class Serial(Robot):
    """
    Serial robot (Torch-only, batchable, with COM inertia adapter).

    Parameters
    ----------
    q : array-like or torch.Tensor
        Joint positions, shape:
          - (n,), (n,1)            -> stored as (n,1)
          - (B,n), (B,n,1)         -> stored as (B,n,1)

    qd, qdd : same shape conventions as q (optional)
        If None => zeros_like(q).

    linksLengths : array-like
        Link lengths (n,).

    COMs : array-like
        COM distances along link x-axis (n,).

    mass : array-like
        Link masses (n,).

    inertia : List[Tensor | dict] or Tensor
        Inertia data per link; see module docstring.

    dh : optional
        Custom DH table. If provided, we keep it and do NOT auto-refresh it.
        Shape:
          - (n+1,4) or (B,n+1,4)

    dhCOM : optional
        Custom COM-DH table. If provided, we keep it and do NOT auto-refresh it.

    dh_convention : str
        "standard" or "modified" (stored, not used here).

    gravity : array-like or Tensor, optional
        3-vector gravity. Defaults to [0,0,-9.81].

    name : str
        Robot name.

    device : torch.device or str, optional
        Device for tensors. If None, inferred from q if q is a Tensor, else "cpu".

    dtype : torch.dtype, optional
        Dtype. If None, uses torch.get_default_dtype().
    """

    # -------------------------- helpers --------------------------
    @staticmethod
    def _to_tensor(x: Any, device: torch.device, dtype: torch.dtype) -> Tensor:
        if isinstance(x, Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)

    @staticmethod
    def _normalize_state_shape(
        x: Tensor,
    ) -> Tuple[Tensor, bool, Optional[int], int]:
        """
        Normalize state tensor to shape (n,1) or (B,n,1).

        Returns
        -------
        x_norm : Tensor
        batched : bool
        B : Optional[int]
            Batch size if batched, else None.
        n : int
            Number of joints.
        """
        if x.dim() == 1:
            # (n,) -> (n,1)
            x = x.view(-1, 1)
            return x, False, None, x.shape[0]

        if x.dim() == 2:
            # (n,1) or (B,n)
            if x.shape[1] == 1:
                # (n,1)
                return x, False, None, x.shape[0]
            else:
                # (B,n) -> (B,n,1)
                x = x.unsqueeze(-1)
                return x, True, x.shape[0], x.shape[1]

        if x.dim() == 3:
            # (B,n,1)
            return x, True, x.shape[0], x.shape[1]

        raise ValueError(
            f"State tensor must be 1D, 2D, or 3D, got shape {tuple(x.shape)}"
        )

    # --------------------------- init ----------------------------
    def __init__(
        self,
        q: Any,
        qd: Any = None,
        qdd: Any = None,
        linksLengths: Any = None,
        COMs: Any = None,
        mass: Any = None,
        inertia: Any = None,
        *,
        name: str = "",
        dh: Any = None,
        dhCOM: Any = None,
        dh_convention: str = "standard",
        gravity: Any = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(name=name)

        # --- dtype & device ---
        if dtype is None:
            dtype = torch.get_default_dtype()

        if isinstance(q, Tensor):
            if device is None:
                device = q.device
        if device is None:
            device = torch.device("cpu")
        else:
            device = torch.device(device)

        self.device = device
        self.dtype = dtype
        self.backend = "torch"  # for compatibility with other modules

        # --- joint states ---
        q_t = self._to_tensor(q, device, dtype)
        q_t, batched, B, n = self._normalize_state_shape(q_t)

        if qd is None:
            qd_t = torch.zeros_like(q_t)
        else:
            qd_t = self._to_tensor(qd, device, dtype)
            qd_t, batched_qd, B_qd, n_qd = self._normalize_state_shape(qd_t)
            if n_qd != n or batched_qd != batched or (batched and B_qd != B):
                raise ValueError("qd shape incompatible with q")

        if qdd is None:
            qdd_t = torch.zeros_like(q_t)
        else:
            qdd_t = self._to_tensor(qdd, device, dtype)
            qdd_t, batched_qdd, B_qdd, n_qdd = self._normalize_state_shape(qdd_t)
            if n_qdd != n or batched_qdd != batched or (batched and B_qdd != B):
                raise ValueError("qdd shape incompatible with q")

        self._q = q_t
        self._qd = qd_t
        self._qdd = qdd_t

        self._batched = batched
        self._B = B
        self.n = n

        # --- gravity ---
        if gravity is None:
            self.g = torch.tensor([0.0, 0.0, -9.81], dtype=dtype, device=device)
        else:
            self.g = self._to_tensor(gravity, device, dtype).reshape(3)

        # --- structural parameters: links, COMs, mass ---
        if linksLengths is None:
            raise ValueError("linksLengths must be provided (length n).")
        links_vec = self._to_tensor(linksLengths, device, dtype).view(-1)
        if links_vec.numel() != n:
            raise ValueError(
                f"linksLengths must have length n={n}, got {links_vec.numel()}"
            )
        self.links = links_vec
        self.linksLengths = self.links  # alias

        if COMs is None:
            coms_vec = 0.5 * self.links
        else:
            coms_vec = self._to_tensor(COMs, device, dtype).view(-1)
            if coms_vec.numel() != n:
                raise ValueError(
                    f"COMs must have length n={n}, got {coms_vec.numel()}"
                )
        self.coms = coms_vec
        self.COMs = self.coms  # alias

        if mass is None:
            mass_vec = torch.ones(n, dtype=dtype, device=device)
        else:
            mass_vec = self._to_tensor(mass, device, dtype).view(-1)
            if mass_vec.numel() != n:
                raise ValueError(
                    f"mass must have length n={n}, got {mass_vec.numel()}"
                )
        self.mass = mass_vec

        # --- inertia: raw + canonicalization (COM in link frame) ---
        self.inertia_raw: List[Any]
        if inertia is None:
            # default: identity inertia for each link
            self.inertia_raw = [
                torch.eye(3, dtype=dtype, device=device) for _ in range(n)
            ]
        elif isinstance(inertia, Tensor):
            # Expect shape (n,3,3); accept (3,3) => broadcast
            if inertia.dim() == 2 and inertia.shape == (3, 3):
                self.inertia_raw = [inertia.to(device=device, dtype=dtype) for _ in range(n)]
            elif inertia.dim() == 3 and inertia.shape[1:] == (3, 3):
                if inertia.shape[0] != n:
                    raise ValueError(
                        f"inertia tensor must have shape (n,3,3), got {tuple(inertia.shape)}"
                    )
                self.inertia_raw = [inertia[i].to(device=device, dtype=dtype) for i in range(n)]
            else:
                raise ValueError(
                    "inertia tensor must be (n,3,3) or (3,3); "
                    f"got shape {tuple(inertia.shape)}"
                )
        else:
            # assume list-like of length n
            inertia_list = list(inertia)
            if len(inertia_list) != n:
                raise ValueError(
                    f"inertia list must have length n={n}, got {len(inertia_list)}"
                )
            self.inertia_raw = inertia_list

        # We'll fill self.I_com and self.inertia in _canonicalize_inertia()
        self.I_com: Optional[Tensor] = None
        self.inertia: Optional[Tensor] = None
        self._canonicalize_inertia()

        # --- DH convention & user-supplied tables ---
        if dh_convention not in ("standard", "modified"):
            raise ValueError("dh_convention must be 'standard' or 'modified'")
        self.dh_convention: str = dh_convention

        self.user_DH = dh
        self.user_DH_COM = dhCOM
        self._auto_refresh_dh = True

        # Build DH / COM-DH tables
        self.denavitHartenberg()
        self.denavitHartenbergCOM()

    # ---------------------- inertia canonicalization ----------------------
    def _canonicalize_inertia(self) -> None:
        """
        Canonicalize inertia to COM in link frame.

        For each link i:

          If inertia_raw[i] is a dict:
            I_P   = It["I"]
            m     = It["m"]
            R     = It.get("R_given_to_link", eye)
            r     = It.get("r_P_to_COM", zero)

            S     = [r]_x
            I_com = I_P - m * S S^T          # about COM, in "given" frame
            I_l   = R I_com R^T              # expressed in link-i frame

          Else:
            I_l   = given 3×3 (assumed already COM in link frame).

        We then check symmetry and PSD, and store:

            self.I_com[i] = I_l
            self.inertia  = self.I_com
        """
        mats: List[Tensor] = []
        for i, It in enumerate(self.inertia_raw):
            if isinstance(It, dict):
                dtype = self.dtype
                device = self.device

                # I: given inertia (any type -> Tensor (3,3))
                I_raw = It["I"]
                I = (
                    I_raw
                    if isinstance(I_raw, Tensor)
                    else torch.as_tensor(I_raw, dtype=dtype, device=device)
                ).reshape(3, 3)

                # m: scalar
                m_raw = It["m"]
                m = (
                    m_raw
                    if isinstance(m_raw, Tensor)
                    else torch.as_tensor(m_raw, dtype=dtype, device=device)
                )

                # R_given_to_link: orientation of "given frame" wrt link frame
                R_raw = It.get(
                    "R_given_to_link",
                    torch.eye(3, dtype=dtype, device=device),
                )
                R = (
                    R_raw
                    if isinstance(R_raw, Tensor)
                    else torch.as_tensor(R_raw, dtype=dtype, device=device)
                ).reshape(3, 3)

                # r_P_to_COM: vector from given origin P to COM, in given frame
                r_raw = It.get(
                    "r_P_to_COM",
                    torch.zeros(3, dtype=dtype, device=device),
                )
                r = (
                    r_raw
                    if isinstance(r_raw, Tensor)
                    else torch.as_tensor(r_raw, dtype=dtype, device=device)
                ).reshape(3)

                S = _skew_torch(r)
                I_com_given = I - m * (S @ S.transpose(-1, -2))  # about COM, given frame
                I_link = R @ I_com_given @ R.transpose(-1, -2)   # COM inertia expressed in link frame
            else:
                # Direct inertia, assume already COM in link frame
                if isinstance(It, Tensor):
                    I_link = It.to(device=self.device, dtype=self.dtype).reshape(3, 3)
                else:
                    I_link = torch.as_tensor(
                        It, dtype=self.dtype, device=self.device
                    ).reshape(3, 3)

            _check_sym_torch(I_link, f"I[{i}]")
            _check_psd_torch(I_link, f"I[{i}]")
            mats.append(I_link)

        self.I_com = torch.stack(mats, dim=0)  # (n,3,3)
        self.inertia = self.I_com  # alias used by dynamics code

    # ----------------------- synced properties -----------------------------
    @property
    def q(self) -> Tensor:
        return self._q

    @q.setter
    def q(self, value: Any):
        t = self._to_tensor(value, self.device, self.dtype)
        t, batched, B, n = self._normalize_state_shape(t)
        if n != self.n:
            raise ValueError(f"q must have length n={self.n}, got {n}")
        self._q = t
        self._batched = batched
        self._B = B
        self._jointsPositions = t
        # Auto-refresh DH tables if we own them
        if self._auto_refresh_dh:
            if self.user_DH is None:
                self.denavitHartenberg()
            if self.user_DH_COM is None:
                self.denavitHartenbergCOM()

    @property
    def jointsPositions(self) -> Tensor:
        return self.q

    @jointsPositions.setter
    def jointsPositions(self, val: Any):
        self.q = val

    @property
    def qd(self) -> Tensor:
        return self._qd

    @qd.setter
    def qd(self, value: Any):
        t = self._to_tensor(value, self.device, self.dtype)
        t, batched, B, n = self._normalize_state_shape(t)
        if n != self.n:
            raise ValueError(f"qd must have length n={self.n}, got {n}")
        if batched != self._batched or (batched and B != self._B):
            raise ValueError("qd batch shape incompatible with q")
        self._qd = t
        self._jointsVelocities = t

    @property
    def jointsVelocities(self) -> Tensor:
        return getattr(self, "_jointsVelocities", self._qd)

    @jointsVelocities.setter
    def jointsVelocities(self, val: Any):
        self.qd = val

    @property
    def qdd(self) -> Tensor:
        return self._qdd

    @qdd.setter
    def qdd(self, value: Any):
        t = self._to_tensor(value, self.device, self.dtype)
        t, batched, B, n = self._normalize_state_shape(t)
        if n != self.n:
            raise ValueError(f"qdd must have length n={self.n}, got {n}")
        if batched != self._batched or (batched and B != self._B):
            raise ValueError("qdd batch shape incompatible with q")
        self._qdd = t
        self._jointsAccelerations = t

    @property
    def jointsAccelerations(self) -> Tensor:
        return getattr(self, "_jointsAccelerations", self._qdd)

    @jointsAccelerations.setter
    def jointsAccelerations(self, val: Any):
        self.qdd = val

    # ------------------------ DH building ----------------------------------
    def denavitHartenberg(self) -> None:
        """
        Build numeric DH table (Torch-only).

        If self.user_DH is provided, it is used as-is (converted to Tensor).
        Otherwise we build:

            row 0 : [0, 0, 0, 0]
            row j+1 : [q_j, 0, links_j, 0]
        """
        if self.user_DH is not None:
            self.dh = self._to_tensor(self.user_DH, self.device, self.dtype)
            return

        # Build from current q and links
        q = self._q
        if not self._batched:
            # q: (n,1)
            zero = torch.zeros((), dtype=self.dtype, device=self.device)
            th = torch.cat([zero[None], q[:, 0]], dim=0)         # (n+1,)
            d = torch.zeros(self.n + 1, dtype=self.dtype, device=self.device)
            a = torch.cat([zero[None], self.links], dim=0)       # (n+1,)
            al = torch.zeros(self.n + 1, dtype=self.dtype, device=self.device)
            self.dh = torch.stack([th, d, a, al], dim=1)         # (n+1,4)
        else:
            # q: (B,n,1)
            B = self._B
            zero = torch.zeros((B, 1), dtype=self.dtype, device=self.device)  # (B,1)
            th = torch.cat([zero, q[:, :, 0]], dim=1)                         # (B,n+1)
            d = torch.zeros((B, self.n + 1), dtype=self.dtype, device=self.device)
            links = self.links.view(1, self.n).expand(B, self.n)             # (B,n)
            a = torch.cat([zero, links], dim=1)                               # (B,n+1)
            al = torch.zeros((B, self.n + 1), dtype=self.dtype, device=self.device)
            self.dh = torch.stack([th, d, a, al], dim=-1)                     # (B,n+1,4)

    def denavitHartenbergCOM(self) -> None:
        """
        Build numeric COM-DH table (Torch-only).

        If self.user_DH_COM is provided, it is used as-is.
        Otherwise:

            row 0 : [0, 0, 0, 0]
            row j+1 : [q_j, 0, coms_j, 0]
        """
        if self.user_DH_COM is not None:
            self.dhCOM = self._to_tensor(self.user_DH_COM, self.device, self.dtype)
            return

        q = self._q
        if not self._batched:
            zero = torch.zeros((), dtype=self.dtype, device=self.device)
            th = torch.cat([zero[None], q[:, 0]], dim=0)
            d = torch.zeros(self.n + 1, dtype=self.dtype, device=self.device)
            a = torch.cat([zero[None], self.coms], dim=0)
            al = torch.zeros(self.n + 1, dtype=self.dtype, device=self.device)
            self.dhCOM = torch.stack([th, d, a, al], dim=1)
        else:
            B = self._B
            zero = torch.zeros((B, 1), dtype=self.dtype, device=self.device)
            th = torch.cat([zero, q[:, :, 0]], dim=1)
            d = torch.zeros((B, self.n + 1), dtype=self.dtype, device=self.device)
            coms = self.coms.view(1, self.n).expand(B, self.n)
            a = torch.cat([zero, coms], dim=1)
            al = torch.zeros((B, self.n + 1), dtype=self.dtype, device=self.device)
            self.dhCOM = torch.stack([th, d, a, al], dim=-1)

    # ------------------------ lookups & info -------------------------------
    def where_is_joint(self, j: int) -> Tuple[int, int]:
        """
        Locate joint index j (0-based) in DH table.

        Returns (row, col) with:
            row = j+1, col = 0 (theta column).
        """
        if not (0 <= j < self.n):
            raise IndexError(f"joint {j} out of range (0..{self.n-1})")
        return j + 1, 0

    def where_is_com(self, j: int) -> Tuple[int, int]:
        """
        Locate COM index j (0-based) in COM-DH table.

        Returns (row, col) with:
            row = j+1, col = 2 ('a' column).
        """
        if not (0 <= j < self.n):
            raise IndexError(f"com {j} out of range (0..{self.n-1})")
        return j + 1, 2

    def dh_info(self) -> str:
        """
        Simple description string of DH layout and convention.
        """
        return (
            f"DH convention: {self.dh_convention}\n"
            "Rows: base=0, joint j (0-based) -> row (j+1)\n"
            "Cols: 0:theta, 1:d, 2:a, 3:alpha\n"
        )

    # --------------------------- utils ------------------------------------
    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Move all internal tensors to a new device / dtype.

        Example
        -------
        rob.to('cuda')
        rob.to('cuda', torch.float32)
        """
        if device is None:
            device = self.device
        device = torch.device(device)
        if dtype is None:
            dtype = self.dtype

        def _move(x):
            if isinstance(x, Tensor):
                return x.to(device=device, dtype=dtype)
            return x

        self._q = _move(self._q)
        self._qd = _move(self._qd)
        self._qdd = _move(self._qdd)

        if hasattr(self, "_jointsPositions"):
            self._jointsPositions = _move(self._jointsPositions)
        if hasattr(self, "_jointsVelocities"):
            self._jointsVelocities = _move(self._jointsVelocities)
        if hasattr(self, "_jointsAccelerations"):
            self._jointsAccelerations = _move(self._jointsAccelerations)

        self.links = _move(self.links)
        self.linksLengths = self.links
        self.coms = _move(self.coms)
        self.COMs = self.coms
        self.mass = _move(self.mass)

        if self.inertia_raw is not None:
            new_raw = []
            for It in self.inertia_raw:
                if isinstance(It, Tensor):
                    new_raw.append(It.to(device=device, dtype=dtype))
                else:
                    new_raw.append(It)
            self.inertia_raw = new_raw

        self.g = _move(self.g)

        if hasattr(self, "I_com") and self.I_com is not None:
            self.I_com = _move(self.I_com)
        if hasattr(self, "inertia") and self.inertia is not None:
            self.inertia = _move(self.inertia)

        if hasattr(self, "dh"):
            self.dh = _move(self.dh)
        if hasattr(self, "dhCOM"):
            self.dhCOM = _move(self.dhCOM)

        self.device = device
        self.dtype = dtype
        return self


# ---------------------------------------------------------------------------
# Simple smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    print("[Robot_torch] Simple smoke test...")

    # Unbatched 2-DoF
    q = torch.tensor([0.2, -0.3])
    qd = torch.tensor([0.01, 0.02])
    qdd = torch.tensor([0.001, 0.002])

    links = [0.3, 0.2]
    coms = [0.15, 0.10]
    mass = [1.0, 1.5]

    # Inertia as dicts (adapter path)
    I1 = torch.diag(torch.tensor([0.01, 0.01, 0.02]))
    I2 = torch.diag(torch.tensor([0.02, 0.02, 0.03]))
    inertia_dicts = [
        {"I": I1, "m": mass[0], "R_given_to_link": torch.eye(3), "r_P_to_COM": torch.zeros(3)},
        {"I": I2, "m": mass[1], "R_given_to_link": torch.eye(3), "r_P_to_COM": torch.zeros(3)},
    ]

    rob = Serial(
        q=q,
        qd=qd,
        qdd=qdd,
        linksLengths=links,
        COMs=coms,
        mass=mass,
        inertia=inertia_dicts,
        name="TorchSerial2",
    )

    print("  n =", rob.n)
    print("  q shape:", tuple(rob.q.shape))
    print("  qd shape:", tuple(rob.qd.shape))
    print("  qdd shape:", tuple(rob.qdd.shape))
    print("  dh shape:", tuple(rob.dh.shape))
    print("  dhCOM shape:", tuple(rob.dhCOM.shape))
    print("  I_com shape:", tuple(rob.I_com.shape))
    print("  where_is_joint(0):", rob.where_is_joint(0))
    print("  where_is_com(1):", rob.where_is_com(1))
    print("  g:", rob.g)

    # Batched 2-DoF
    print("\n  [batched] B=4")
    q_b = torch.randn(4, 2)        # (B,n)
    qd_b = torch.randn(4, 2, 1)    # (B,n,1)
    qdd_b = torch.zeros(4, 2)      # (B,n)

    rob_b = Serial(
        q=q_b,
        qd=qd_b,
        qdd=qdd_b,
        linksLengths=links,
        COMs=coms,
        mass=mass,
        inertia=inertia_dicts,
        name="TorchSerial2Batched",
    )

    print("  q_b shape:", tuple(rob_b.q.shape))
    print("  qd_b shape:", tuple(rob_b.qd.shape))
    print("  qdd_b shape:", tuple(rob_b.qdd.shape))
    print("  dh_b shape:", tuple(rob_b.dh.shape))
    print("  dhCOM_b shape:", tuple(rob_b.dhCOM.shape))
    print("  I_com_b shape:", tuple(rob_b.I_com.shape))

    print("[Robot_torch] Smoke test done.")
