# minjerk_torch.py
# -*- coding: utf-8 -*-
"""
Torch version of minjerk.py

- MinJerkParams: same dataclass API as NumPy version.
- MinJerkLinearTrajectoryTorch:
    * plans a piecewise min-jerk trajectory between waypoints
    * fully Torch-based, batchable and GPU-safe
    * differentiable w.r.t. waypoints and time (except at segment boundaries)

Waypoints:
    waypoints: array-like (N, D) or Tensor, N >= 2

Sampling:
    x, xd, xdd = traj.sample(t)

    * t can be a scalar float / 0D Tensor  -> x, xd, xdd have shape (D,)
    * t can be a 1D Tensor of times       -> x, xd, xdd have shape (T, D)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Sequence, Tuple, Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------
# Parameters (same API as NumPy version)
# ---------------------------------------------------------------------


@dataclass
class MinJerkParams:
    Vmax: float
    Amax: float
    Jmax: float
    gamma: float = 1.10


# ---------------------------------------------------------------------
# Internal helpers (Torch)
# ---------------------------------------------------------------------


def _minjerk_profile_torch(T: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Torch version of _minjerk_profile.

    T : (...,) segment durations (must be > 0)
    t : (...,) local times within each segment (same shape as T)

    Returns:
        s, sd, sdd, sddd   (all same shape as T & t)
    """
    # Ensure Torch tensors on same device/dtype
    T = torch.as_tensor(T)
    t = torch.as_tensor(t, device=T.device, dtype=T.dtype)

    # Safety clamp on T
    T = torch.clamp(T, min=1e-9)

    tau = torch.clamp(t / T, 0.0, 1.0)

    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    sd = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / T
    sdd = (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / (T**2)
    sddd = (60.0 - 360.0 * tau + 360.0 * tau**2) / (T**3)
    return s, sd, sdd, sddd


def _segment_time_torch(L: Tensor, p: MinJerkParams) -> Tensor:
    """
    Torch version of _segment_time.

    L : (...,) segment lengths (>= 0)
    p : MinJerkParams

    Returns:
        T : (...,) segment durations
    """
    L = torch.as_tensor(L)
    device, dtype = L.device, L.dtype

    Vfac, Afac, Jfac = 1.875, 5.7735026919, 60.0

    L_safe = torch.clamp(L, min=1e-9)

    Vsafe = float(max(p.Vmax, 1e-9))
    Asafe = float(max(p.Amax, 1e-9))
    Jsafe = float(max(p.Jmax, 1e-9))

    T_v = Vfac * L_safe / Vsafe
    T_a = torch.sqrt(Afac * L_safe / Asafe)
    T_j = torch.pow(Jfac * L_safe / Jsafe, 1.0 / 3.0)
    Tmin = torch.full_like(L_safe, 1e-3, device=device, dtype=dtype)

    T = torch.max(torch.max(torch.max(T_v, T_a), T_j), Tmin)
    return float(p.gamma) * T


# ---------------------------------------------------------------------
# Min-jerk trajectory (Torch)
# ---------------------------------------------------------------------


class MinJerkLinearTrajectoryTorch:
    """
    Piecewise linear path with min-jerk time law in Torch.

    Init:
        traj = MinJerkLinearTrajectoryTorch(waypoints, params, device=..., dtype=...)

    Sample:
        x, xd, xdd = traj.sample(t)

    where:
        waypoints: (N, D) array-like or Tensor, N >= 2
        t: float or Tensor of shape (T,)
    """

    def __init__(
        self,
        waypoints: Union[Sequence[Sequence[float]], Tensor],
        params: MinJerkParams,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.params = params

        # set device/dtype
        if isinstance(waypoints, Tensor):
            w = waypoints
            if device is None:
                device = w.device
            if dtype is None:
                dtype = w.dtype
        else:
            if device is None:
                device = torch.device("cpu")
            if dtype is None:
                dtype = torch.get_default_dtype()
            w = torch.as_tensor(waypoints, device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype

        if w.dim() == 1:
            w = w.view(-1, 1)
        if w.dim() != 2:
            raise ValueError(f"waypoints must be 2D (N,D), got {tuple(w.shape)}")
        if w.shape[0] < 2:
            raise ValueError("At least 2 waypoints are required")

        self._plan(w, params)

    # ---- planning ----

    def _plan(self, waypoints: Tensor, params: MinJerkParams):
        """
        Precompute per-segment data in Torch:

            P0: (S, D)   segment start points
            d : (S, D)   P1 - P0
            T : (S,)     segment durations
            tgrid: (S+1,) cumulative times, tgrid[0] = 0
        """
        waypoints = waypoints.to(device=self.device, dtype=self.dtype)

        P0 = waypoints[:-1, :]  # (S, D)
        P1 = waypoints[1:, :]   # (S, D)
        d = P1 - P0             # (S, D)

        L = torch.linalg.norm(d, dim=1)  # (S,)
        T = _segment_time_torch(L, params).to(device=self.device, dtype=self.dtype)  # (S,)

        # cumulative time grid
        t0 = torch.zeros(1, device=self.device, dtype=self.dtype)
        tgrid = torch.cat([t0, torch.cumsum(T, dim=0)], dim=0)  # (S+1,)

        self.P0 = P0
        self.d = d
        self.T = T
        self.tgrid = tgrid
        self.n_segs = T.shape[0]
        self.dim = P0.shape[1]

    # ---- sampling ----

    def sample(self, t: Union[float, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample position, velocity, acceleration at time t.

        t:
          - scalar float or 0D Tensor: returns x, xd, xdd with shape (D,)
          - 1D Tensor of shape (T,): returns x, xd, xdd with shape (T, D)
        """
        if self.n_segs == 0:
            raise ValueError("Empty trajectory (no segments)")

        # Convert t to Tensor
        t_in = torch.as_tensor(t, device=self.device, dtype=self.dtype)
        was_scalar = (t_in.dim() == 0)
        t_flat = t_in.view(-1)  # (Nt,)

        # clamp t to [t_start, t_end]
        t_start = float(self.tgrid[0].item())
        t_end = float(self.tgrid[-1].item())
        t_clamped = torch.clamp(t_flat, t_start, t_end)  # (Nt,)

        # For each t, find segment index k such that tgrid[k] <= t < tgrid[k+1]
        # torch.searchsorted(tgrid, t, right=True) -> index in [0, S]
        idx = torch.searchsorted(self.tgrid, t_clamped, right=True) - 1  # (Nt,)
        idx = torch.clamp(idx, 0, self.n_segs - 1)

        # local time tau = t - tgrid[k]
        t0 = self.tgrid[idx]         # (Nt,)
        tau = t_clamped - t0         # (Nt,)
        Tseg = self.T[idx]           # (Nt,)

        # min-jerk profile per sample
        s, sd, sdd, _ = _minjerk_profile_torch(Tseg, tau)  # (Nt,)

        # segment geometry
        P0 = self.P0[idx, :]  # (Nt, D)
        d = self.d[idx, :]    # (Nt, D)

        s_ = s.unsqueeze(-1)    # (Nt, 1)
        sd_ = sd.unsqueeze(-1)
        sdd_ = sdd.unsqueeze(-1)

        x = P0 + d * s_        # (Nt, D)
        xd = d * sd_           # (Nt, D)
        xdd = d * sdd_         # (Nt, D)

        if was_scalar:
            return x[0], xd[0], xdd[0]
        else:
            return x, xd, xdd


# ---------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    torch.set_printoptions(precision=6, sci_mode=False)
    torch.set_default_dtype(torch.float64)

    print("[minjerk_torch] Simple smoke test...")

    # 2D straight-line trajectory between three waypoints
    waypoints = [[0.0, 0.0], [0.1, 0.2], [0.3, 0.25]]
    params = MinJerkParams(Vmax=1.0, Amax=10.0, Jmax=100.0, gamma=1.1)

    traj = MinJerkLinearTrajectoryTorch(waypoints, params)

    # Scalar time
    t0 = 0.0
    x0, xd0, xdd0 = traj.sample(t0)
    print("  t0:", t0)
    print("  x0:", x0)
    print("  xd0:", xd0)
    print("  xdd0:", xdd0)

    # Vector of times
    t_vec = torch.linspace(0.0, float(traj.tgrid[-1]), steps=5)
    x, xd, xdd = traj.sample(t_vec)
    print("  t_vec:", t_vec)
    print("  x shape:", x.shape, "xd shape:", xd.shape, "xdd shape:", xdd.shape)

    # Check differentiability wrt time
    t_scalar = torch.tensor(0.1, requires_grad=True)
    x1, xd1, xdd1 = traj.sample(t_scalar)
    y = x1.sum()
    y.backward()
    print("  dy/dt at t=0.1:", t_scalar.grad)

    print("[minjerk_torch] Smoke test done.")
