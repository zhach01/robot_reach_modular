# muscles_torch.py
# -*- coding: utf-8 -*-
"""
Pure-PyTorch muscle models (batchable, GPU-safe, differentiable).

This file is a Torch counterpart of `muscles_numpy.py`:
- Same public API and class names:
    * Muscle (base)
    * ReluMuscle
    * MujocoHillMuscle
    * RigidTendonHillMuscle
    * RigidTendonHillMuscleThelen
    * CompliantTendonHillMuscle
- Same geometry_state convention:
    geometry_state: (batch, 2, n_muscles)
        [:,0,:] = musculotendon path length (m)
        [:,1,:] = musculotendon path velocity (m/s)
- muscle_state: (batch, state_dim, n_muscles)
- action / excitation: (batch, 1, n_muscles) or broadcastable.

Differences vs NumPy version:
- Implemented entirely in Torch: no NumPy, no Python loops over batch.
- All parameters live on `self.device` / `self.dtype`.
- Fully differentiable (no .detach(), no .numpy()).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Optional

import torch
import math
from torch import Tensor


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _as_3d(
    arr: Any,
    n_muscles: int | None = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Ensure arr is 3D (batch, channels, n_muscles).

    This mirrors the semantics of the NumPy helper but returns a Tensor.
    """
    t = torch.as_tensor(arr, dtype=dtype, device=device)
    if t.dim() == 1:
        # (n_muscles,)
        t = t.reshape(1, 1, -1)
    elif t.dim() == 2:
        # (batch, channels)
        if n_muscles is None:
            raise ValueError(
                "Cannot promote a (batch,channels) tensor to 3D without n_muscles."
            )
        t = t.reshape(t.shape[0], t.shape[1], n_muscles)
    elif t.dim() == 3:
        pass
    else:
        raise ValueError(f"Expected 1–3D tensor, got {t.dim()}D")
    return t


def _to_param(
    x: Any,
    n_muscles: int,
    *,
    dtype: Optional[torch.dtype],
    device: torch.device,
) -> Tensor:
    """
    Make a (1,1,n_muscles) tensor out of scalar/list-like x.

    If x has size 1 -> broadcast.
    If x has size n_muscles -> keep as is.
    """
    a = torch.as_tensor(x, dtype=dtype, device=device).reshape(1, 1, -1)
    if a.numel() not in (1, n_muscles):
        raise ValueError(
            f"Parameter size must be 1 or n_muscles={n_muscles}, got {a.numel()}"
        )
    if a.numel() == 1:
        a = torch.ones((1, 1, n_muscles), dtype=dtype, device=device) * a.reshape(())
    return a


def _clip(x: Tensor, lo: float | Tensor | None = None, hi: float | Tensor | None = None) -> Tensor:
    """
    Torch equivalent of the NumPy helper:
    - If lo is None: only upper clip.
    - If hi is None: only lower clip.
    - Else: clip to [lo, hi].
    """
    if lo is None and hi is None:
        return x
    if lo is None:
        if not torch.is_tensor(hi):
            hi = torch.as_tensor(hi, dtype=x.dtype, device=x.device)
        return torch.minimum(x, hi)
    if hi is None:
        if not torch.is_tensor(lo):
            lo = torch.as_tensor(lo, dtype=x.dtype, device=x.device)
        return torch.maximum(x, lo)
    if not torch.is_tensor(lo):
        lo = torch.as_tensor(lo, dtype=x.dtype, device=x.device)
    if not torch.is_tensor(hi):
        hi = torch.as_tensor(hi, dtype=x.dtype, device=x.device)
    return torch.clamp(x, lo, hi)


# ---------------------------------------------------------------------
# Base Muscle
# ---------------------------------------------------------------------


@dataclass
class Muscle:
    input_dim: int = 1
    output_dim: int = 1
    min_activation: float = 0.0
    tau_activation: float = 0.015
    tau_deactivation: float = 0.05

    # runtime-populated
    state_name: List[str] | None = None
    dt: float | None = None
    n_muscles: int | None = None
    max_iso_force: Tensor | None = None  # (1,1,nm)
    vmax: Tensor | None = None  # (1,1,nm)
    l0_se: Tensor | None = None  # (1,1,nm)
    l0_ce: Tensor | None = None  # (1,1,nm)
    l0_pe: Tensor | None = None  # (1,1,nm)
    built: bool = False

    # optional knobs recorded like the NumPy version
    to_build_dict: Dict[str, list] | None = None
    to_build_dict_default: Dict[str, Any] | None = None

    # torch specifics
    device: Union[str, torch.device] = "cpu"
    dtype: Optional[torch.dtype] = None

    def __post_init__(self):
        self.device = torch.device(self.device)
        if self.dtype is None:
            self.dtype = torch.get_default_dtype()

        if self.state_name is None:
            self.state_name = []
        if self.to_build_dict is None:
            self.to_build_dict = {"max_isometric_force": []}
        if self.to_build_dict_default is None:
            self.to_build_dict_default = {}

    # --- helpers

    def clip_excitation(self, u: Any) -> Tensor:
        """
        Clamp *excitations* u to [min_activation, 1.0].
        Note: activation a is the internal state updated by activation dynamics.
        """
        u_t = torch.as_tensor(u, dtype=self.dtype, device=self.device)
        return _clip(u_t, self.min_activation, 1.0)

    def clip_activation(self, a: Any) -> Tensor:
        # Back-compat alias: same behavior as clip_excitation
        return self.clip_excitation(a)

    # --- build

    def build(self, timestep: float, max_isometric_force: Any, **kwargs):
        """
        Base build: sets common parameters to (1,1,n_muscles) tensors.

        Subclasses are free to override; this is used directly by ReluMuscle.
        """
        mif = torch.as_tensor(
            max_isometric_force, dtype=self.dtype, device=self.device
        ).reshape(1, 1, -1)
        self.n_muscles = int(mif.numel())
        self.max_iso_force = mif.clone()
        self.dt = float(timestep)

        ones = torch.ones((1, 1, self.n_muscles), dtype=self.dtype, device=self.device)
        self.vmax = ones.clone()
        self.l0_se = ones.clone()
        self.l0_ce = ones.clone()
        self.l0_pe = ones.clone()
        self.built = True

    # --- API

    def get_initial_muscle_state(
        self, batch_size: int, geometry_state: Tensor
    ) -> Tensor:
        return self._get_initial_muscle_state(batch_size, geometry_state)

    def _get_initial_muscle_state(
        self, batch_size: int, geometry_state: Tensor
    ) -> Tensor:
        raise NotImplementedError

    def integrate(
        self,
        dt: float,
        state_derivative: Tensor,
        muscle_state: Tensor,
        geometry_state: Tensor,
    ) -> Tensor:
        return self._integrate(dt, state_derivative, muscle_state, geometry_state)

    def _integrate(
        self,
        dt: float,
        state_derivative: Tensor,
        muscle_state: Tensor,
        geometry_state: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    def ode(self, action: Tensor, muscle_state: Tensor) -> Tensor:
        return self._ode(action, muscle_state)

    def _ode(self, action: Tensor, muscle_state: Tensor) -> Tensor:
        activation = muscle_state[:, :1, :]
        return self.activation_ode(action, activation)

    def activation_ode(self, action: Tensor, activation: Tensor) -> Tensor:
        """
        Thelen-style activation dynamics (same as original NumPy/Torch logic).
        """
        if self.n_muscles is None:
            raise RuntimeError("Muscle.build(...) must be called before activation_ode.")
        # reshape action to (batch,1,n_muscles)
        action = self.clip_excitation(action).reshape(-1, 1, self.n_muscles)
        activation = self.clip_activation(activation)
        tmp = 0.5 + 1.5 * activation
        tau = torch.where(
            action > activation,
            self.tau_activation * tmp,
            self.tau_deactivation / tmp,
        )
        return (action - activation) / tau

    # utility setter (kept for API compatibility)
    def setattr(self, name: str, value):
        setattr(self, name, value)

    def get_save_config(self):
        return {
            "name": str(getattr(self, "__name__", self.__class__.__name__)),
            "state names": self.state_name,
        }

    # device / dtype propagation
    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if device is None:
            device = self.device
        device = torch.device(device)
        if dtype is None:
            dtype = self.dtype

        def _move(x):
            return x.to(device=device, dtype=dtype) if isinstance(x, Tensor) else x

        self.max_iso_force = _move(self.max_iso_force)
        self.vmax = _move(self.vmax)
        self.l0_se = _move(self.l0_se)
        self.l0_ce = _move(self.l0_ce)
        self.l0_pe = _move(self.l0_pe)

        self.device = device
        self.dtype = dtype
        return self


# ---------------------------------------------------------------------
# ReluMuscle
# ---------------------------------------------------------------------


class ReluMuscle(Muscle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__name__ = "ReluMuscle"
        self.state_name = ["activation", "muscle length", "muscle velocity", "force"]
        self.state_dim = len(self.state_name)

    def _integrate(
        self,
        dt: float,
        state_derivative: Tensor,
        muscle_state: Tensor,
        geometry_state: Tensor,
    ) -> Tensor:
        dt_t = torch.as_tensor(dt, dtype=self.dtype, device=self.device)
        activation = muscle_state[:, :1, :] + state_derivative * dt_t
        activation = self.clip_activation(activation)
        forces = activation * self.max_iso_force
        len_vel = geometry_state[:, :2, :]
        return torch.cat([activation, len_vel, forces], dim=1)

    def _get_initial_muscle_state(
        self, batch_size: int, geometry_state: Tensor
    ) -> Tensor:
        shape = geometry_state[:, :1, :].shape
        activation0 = torch.ones(
            shape, dtype=self.dtype, device=self.device
        ) * self.min_activation
        force0 = torch.zeros(shape, dtype=self.dtype, device=self.device)
        len_vel = geometry_state[:, 0:2, :]
        return torch.cat([activation0, len_vel, force0], dim=1)


# ---------------------------------------------------------------------
# MujocoHillMuscle
# ---------------------------------------------------------------------


class MujocoHillMuscle(Muscle):
    def __init__(
        self,
        min_activation: float = 0.0,
        passive_forces: float = 1.0,
        tau_activation: float = 0.01,
        tau_deactivation: float = 0.04,
        **kwargs,
    ):
        super().__init__(
            min_activation=min_activation,
            tau_activation=tau_activation,
            tau_deactivation=tau_deactivation,
            **kwargs,
        )
        self.__name__ = "MujocoHillMuscle"
        self.state_name = [
            "activation",
            "muscle length",
            "muscle velocity",
            "force-length PE",
            "force-length CE",
            "force-velocity CE",
            "force",
        ]
        self.state_dim = len(self.state_name)

        self.to_build_dict = {
            "max_isometric_force": [],
            "optimal_muscle_length": [],
            "tendon_length": [],
            "normalized_slack_muscle_length": [],
            "lmin": [],
            "lmax": [],
            "vmax": [],
            "fvmax": [],
        }
        self.to_build_dict_default = {
            "normalized_slack_muscle_length": 1.3,
            "lmin": 0.5,
            "lmax": 1.6,
            "vmax": 1.5,
            "fvmax": 1.2,
        }
        self.passive_forces = float(passive_forces)

        # derived params set in build()
        self.b: Tensor | None = None
        self.c: Tensor | None = None
        self.p1: Tensor | None = None
        self.p2: Tensor | None = None
        self.mid: Tensor | None = None

    def build(
        self,
        timestep: float,
        max_isometric_force,
        tendon_length,
        optimal_muscle_length,
        normalized_slack_muscle_length,
        lmin,
        lmax,
        vmax,
        fvmax,
    ):
        self.n_muscles = int(torch.as_tensor(tendon_length).numel())
        self.dt = float(timestep)

        self.max_iso_force = _to_param(
            max_isometric_force, self.n_muscles, dtype=self.dtype, device=self.device
        )
        self.l0_pe = _to_param(
            normalized_slack_muscle_length,
            self.n_muscles,
            dtype=self.dtype,
            device=self.device,
        )
        self.l0_ce = _to_param(
            optimal_muscle_length, self.n_muscles, dtype=self.dtype, device=self.device
        )
        self.l0_se = _to_param(
            tendon_length, self.n_muscles, dtype=self.dtype, device=self.device
        )
        self.lmin = _to_param(lmin, self.n_muscles, dtype=self.dtype, device=self.device)
        self.lmax = _to_param(lmax, self.n_muscles, dtype=self.dtype, device=self.device)
        self.vmax = _to_param(vmax, self.n_muscles, dtype=self.dtype, device=self.device)
        self.fvmax = _to_param(
            fvmax, self.n_muscles, dtype=self.dtype, device=self.device
        )

        # derived (all broadcastable to (1,1,n_muscles))
        self.b = 0.5 * (1 + self.lmax)
        self.c = self.fvmax - 1.0
        self.p1 = self.b - 1.0
        self.p2 = 0.25 * self.l0_pe
        self.mid = 0.5 * (self.lmin + 0.95)
        self.built = True

    def _bump(self, L: Tensor, mid: Tensor, lmax: Tensor) -> Tensor:
        """Quadratic spline 'bump' as per original logic."""
        left = 0.5 * (self.lmin + mid)
        right = 0.5 * (mid + lmax)

        out_of_range = (L <= self.lmin) | (L >= lmax)
        less_than_left = L < left
        less_than_mid = L < mid
        less_than_right = L < right

        # x piecewise
        x = torch.where(
            out_of_range,
            torch.zeros_like(L),
            torch.where(
                less_than_left,
                (L - self.lmin) / (left - self.lmin),
                torch.where(
                    less_than_mid,
                    (mid - L) / (mid - left),
                    torch.where(
                        less_than_right,
                        (L - mid) / (right - mid),
                        (lmax - L) / (lmax - right),
                    ),
                ),
            ),
        )
        pfivexx = 0.5 * x * x
        y = torch.where(
            out_of_range,
            torch.zeros_like(L),
            torch.where(
                less_than_left,
                pfivexx,
                torch.where(
                    less_than_mid,
                    1 - pfivexx,
                    torch.where(less_than_right, 1 - pfivexx, pfivexx),
                ),
            ),
        )
        return y

    def _get_initial_muscle_state(
        self, batch_size: int, geometry_state: Tensor
    ) -> Tensor:
        shape = geometry_state[:, :1, :].shape
        muscle_state = torch.ones(
            shape, dtype=self.dtype, device=self.device
        ) * self.min_activation
        state_derivatives = torch.zeros(shape, dtype=self.dtype, device=self.device)
        return self.integrate(self.dt, state_derivatives, muscle_state, geometry_state)

    def _integrate(
        self,
        dt: float,
        state_derivative: Tensor,
        muscle_state: Tensor,
        geometry_state: Tensor,
    ) -> Tensor:
        dt_t = torch.as_tensor(dt, dtype=self.dtype, device=self.device)
        activation = muscle_state[:, :1, :] + state_derivative * dt_t
        activation = self.clip_activation(activation)

        # geometry
        musculotendon_len = geometry_state[:, :1, :]
        muscle_len = _clip(
            (musculotendon_len - self.l0_se) / self.l0_ce, lo=0.001
        )  # normalized
        muscle_vel = geometry_state[:, 1:2, :] / self.vmax

        # passive length element (flpe)
        x = torch.where(
            muscle_len <= 1.0,
            torch.zeros_like(muscle_len),
            torch.where(
                muscle_len <= self.b,
                (muscle_len - 1.0) / self.p1,
                (muscle_len - self.b) / self.p1,
            ),
        )
        flpe = torch.where(
            muscle_len <= 1.0,
            torch.zeros_like(muscle_len),
            torch.where(
                muscle_len <= self.b,
                self.p2 * x**3,
                self.p2 * (1.0 + 3.0 * x),
            ),
        )

        # active length (flce)
        flce = self._bump(
            muscle_len, mid=torch.ones_like(self.mid), lmax=self.lmax
        ) + 0.15 * self._bump(
            muscle_len,
            mid=self.mid,
            lmax=torch.full_like(self.lmax, 0.95),
        )

        # force-velocity CE (fvce)
        fvce = torch.where(
            muscle_vel <= -1.0,
            torch.zeros_like(muscle_vel),
            torch.where(
                muscle_vel <= 0.0,
                (muscle_vel + 1.0) ** 2,
                torch.where(
                    muscle_vel <= self.c,
                    self.fvmax
                    - (self.c - muscle_vel) * (self.c - muscle_vel) / self.c,
                    self.fvmax,
                ),
            ),
        )

        force = (
            activation * flce * fvce + self.passive_forces * flpe
        ) * self.max_iso_force
        # return activation, (denormalized length, velocity), flpe, flce, fvce, force
        return torch.cat(
            [
                activation,
                muscle_len * self.l0_ce,
                muscle_vel * self.vmax,
                flpe,
                flce,
                fvce,
                force,
            ],
            dim=1,
        )


# ---------------------------------------------------------------------
# RigidTendonHillMuscle
# ---------------------------------------------------------------------


class RigidTendonHillMuscle(Muscle):
    def __init__(self, min_activation: float = 0.001, **kwargs):
        super().__init__(min_activation=min_activation, **kwargs)
        self.__name__ = "RigidTendonHillMuscle"
        self.state_name = [
            "activation",
            "muscle length",
            "muscle velocity",
            "force-length PE",
            "force-length CE",
            "force-velocity CE",
            "force",
        ]
        self.state_dim = len(self.state_name)

        # constants
        self.pe_k = 5.0
        self.pe_1 = self.pe_k / 0.66
        self.pe_den = math.exp(self.pe_k) - 1.0
        self.ce_gamma = 0.45
        self.ce_Af = 0.25
        self.ce_fmlen = 1.4

        # derived params set in build()
        self.musculotendon_slack_len: Tensor | None = None
        self.k_pe: Tensor | None = None
        self.s_as = 0.001
        self.f_iso_n_den = 0.66**2
        self.k_se = 1.0 / (0.04**2)
        self.q_crit = 0.3
        self.b_rel_st_den = 5e-3 - self.q_crit  # not used directly but kept
        self.min_flce = 0.01

        self.to_build_dict = {
            "max_isometric_force": [],
            "tendon_length": [],
            "optimal_muscle_length": [],
            "normalized_slack_muscle_length": [],
        }
        self.to_build_dict_default = {"normalized_slack_muscle_length": 1.4}

    def build(
        self,
        timestep: float,
        max_isometric_force,
        tendon_length,
        optimal_muscle_length,
        normalized_slack_muscle_length,
    ):
        self.n_muscles = int(torch.as_tensor(tendon_length).numel())

        self.dt = float(timestep)
        self.max_iso_force = _to_param(
            max_isometric_force, self.n_muscles, dtype=self.dtype, device=self.device
        )
        self.l0_ce = _to_param(
            optimal_muscle_length, self.n_muscles, dtype=self.dtype, device=self.device
        )
        self.l0_se = _to_param(
            tendon_length, self.n_muscles, dtype=self.dtype, device=self.device
        )
        self.l0_pe = (
            _to_param(
                normalized_slack_muscle_length,
                self.n_muscles,
                dtype=self.dtype,
                device=self.device,
            )
            * self.l0_ce
        )

        self.k_pe = 1.0 / ((1.66 - self.l0_pe / self.l0_ce) ** 2)
        self.musculotendon_slack_len = self.l0_pe + self.l0_se
        self.vmax = 10.0 * self.l0_ce
        self.built = True

    def _get_initial_muscle_state(
        self, batch_size: int, geometry_state: Tensor
    ) -> Tensor:
        shape = geometry_state[:, :1, :].shape
        muscle_state = torch.ones(
            shape, dtype=self.dtype, device=self.device
        ) * self.min_activation
        state_derivatives = torch.zeros(shape, dtype=self.dtype, device=self.device)
        return self.integrate(self.dt, state_derivatives, muscle_state, geometry_state)

    def _integrate(
        self,
        dt: float,
        state_derivative: Tensor,
        muscle_state: Tensor,
        geometry_state: Tensor,
    ) -> Tensor:
        dt_t = torch.as_tensor(dt, dtype=self.dtype, device=self.device)
        activation = self.clip_activation(
            muscle_state[:, :1, :] + state_derivative * dt_t
        )

        # geometry
        musculotendon_len = geometry_state[:, :1, :]
        muscle_vel = geometry_state[:, 1:2, :]
        muscle_len = _clip(musculotendon_len - self.l0_se, lo=0.0)
        muscle_strain = _clip((muscle_len - self.l0_pe) / self.l0_ce, lo=0.0)
        muscle_len_n = muscle_len / self.l0_ce
        muscle_vel_n = muscle_vel / self.vmax

        # forces
        flpe = self.k_pe * (muscle_strain**2)
        flce = _clip(
            1.0
            + (-(muscle_len_n**2) + 2.0 * muscle_len_n - 1.0) / self.f_iso_n_den,
            lo=self.min_flce,
        )

        a_rel_st = torch.where(
            muscle_len_n > 1.0, 0.41 * flce, torch.full_like(flce, 0.41)
        )
        b_rel_st = torch.where(
            activation < self.q_crit,
            5.2
            * (1.0 - 0.9 * ((activation - self.q_crit) / (5e-3 - self.q_crit))) ** 2,
            torch.full_like(activation, 5.2),
        )
        dfdvcon0 = activation * (flce + a_rel_st) / b_rel_st

        f_x_a = flce * activation
        tmp_p_nom = f_x_a * 0.5
        tmp_p_den = self.s_as - dfdvcon0 * 2.0

        p1 = -tmp_p_nom / tmp_p_den
        p2 = (tmp_p_nom**2) / tmp_p_den
        p3 = -1.5 * f_x_a

        nom = torch.where(
            muscle_vel_n < 0.0,
            muscle_vel_n * activation * a_rel_st + f_x_a * b_rel_st,
            -p1 * p3
            + p1 * self.s_as * muscle_vel_n
            + p2
            - p3 * muscle_vel_n
            + self.s_as * muscle_vel_n**2,
        )
        den = torch.where(
            muscle_vel_n < 0.0, b_rel_st - muscle_vel_n, p1 + muscle_vel_n
        )

        active_force = _clip(nom / den, lo=0.0)
        force = (active_force + flpe) * self.max_iso_force

        return torch.cat(
            [activation, muscle_len, muscle_vel, flpe, flce, active_force, force],
            dim=1,
        )


# ---------------------------------------------------------------------
# RigidTendonHillMuscleThelen
# ---------------------------------------------------------------------


class RigidTendonHillMuscleThelen(Muscle):
    def __init__(self, min_activation: float = 0.001, **kwargs):
        super().__init__(min_activation=min_activation, **kwargs)
        self.__name__ = "RigidTendonHillMuscleThelen"
        self.state_name = [
            "activation",
            "muscle length",
            "muscle velocity",
            "force-length PE",
            "force-length CE",
            "force-velocity CE",
            "force",
        ]
        self.state_dim = len(self.state_name)

        # parameters (as scalars -> broadcast in build)
        self.pe_k = 5.0
        self.pe_1 = self.pe_k / 0.6
        self.pe_den = math.exp(self.pe_k) - 1.0
        self.ce_gamma = 0.45
        self.ce_Af = 0.25
        self.ce_fmlen = 1.4

        # precomputed (in build)
        self.ce_0: Tensor | None = None
        self.ce_1: Tensor | None = None
        self.ce_2: Tensor | None = None
        self.ce_3: Tensor | None = None
        self.ce_4: Tensor | None = None
        self.ce_5: Tensor | None = None

        self.to_build_dict = {
            "max_isometric_force": [],
            "tendon_length": [],
            "optimal_muscle_length": [],
            "normalized_slack_muscle_length": [],
        }
        self.to_build_dict_default = {"normalized_slack_muscle_length": 1.0}

    def build(
        self,
        timestep: float,
        max_isometric_force,
        tendon_length,
        optimal_muscle_length,
        normalized_slack_muscle_length,
    ):
        self.n_muscles = int(torch.as_tensor(tendon_length).numel())
        self.dt = float(timestep)

        # broadcast scalars or accept per-muscle lists
        self.max_iso_force = _to_param(
            max_isometric_force, self.n_muscles, dtype=self.dtype, device=self.device
        )
        self.l0_ce = _to_param(
            optimal_muscle_length, self.n_muscles, dtype=self.dtype, device=self.device
        )
        self.l0_se = _to_param(
            tendon_length, self.n_muscles, dtype=self.dtype, device=self.device
        )
        nsl = _to_param(
            normalized_slack_muscle_length,
            self.n_muscles,
            dtype=self.dtype,
            device=self.device,
        )

        self.l0_pe = self.l0_ce * nsl
        self.musculotendon_slack_len = self.l0_pe + self.l0_se
        self.vmax = 10.0 * self.l0_ce

        # precompute (all broadcastable)
        self.ce_0 = 3.0 * self.vmax
        self.ce_1 = self.ce_Af * self.vmax
        self.ce_2 = (
            3.0 * self.ce_Af * self.vmax * self.ce_fmlen
            - 3.0 * self.ce_Af * self.vmax
        )
        self.ce_3 = 8.0 * self.ce_Af * self.ce_fmlen + 8.0 * self.ce_fmlen
        self.ce_4 = self.ce_Af * self.ce_fmlen * self.vmax - self.ce_1
        self.ce_5 = 8.0 * (self.ce_Af + 1.0)

        self.built = True

    def _get_initial_muscle_state(
        self, batch_size: int, geometry_state: Tensor
    ) -> Tensor:
        shape = geometry_state[:, :1, :].shape
        muscle_state = torch.ones(
            shape, dtype=self.dtype, device=self.device
        ) * self.min_activation
        state_derivatives = torch.zeros(shape, dtype=self.dtype, device=self.device)
        return self.integrate(self.dt, state_derivatives, muscle_state, geometry_state)

    def _integrate(
        self,
        dt: float,
        state_derivative: Tensor,
        muscle_state: Tensor,
        geometry_state: Tensor,
    ) -> Tensor:
        dt_t = torch.as_tensor(dt, dtype=self.dtype, device=self.device)
        activation = self.clip_activation(
            muscle_state[:, :1, :] + state_derivative * dt_t
        )

        # geometry
        musculotendon_len = geometry_state[:, :1, :]
        muscle_len = _clip(musculotendon_len - self.l0_se, lo=0.001)
        muscle_vel = geometry_state[:, 1:2, :]

        a3 = activation * 3.0
        condition = muscle_vel <= 0.0
        nom = torch.where(
            condition,
            self.ce_Af * (activation * self.ce_0 + 4.0 * muscle_vel + self.vmax),
            self.ce_2 * activation + self.ce_3 * muscle_vel + self.ce_4,
        )
        den = torch.where(
            condition,
            a3 * self.ce_1 + self.ce_1 - 4.0 * muscle_vel,
            self.ce_4 * a3 + self.ce_5 * muscle_vel + self.ce_4,
        )
        fvce = _clip(nom / den, lo=0.0)
        flpe = _clip(
            (
                torch.exp(self.pe_1 * (muscle_len - self.l0_pe) / self.l0_ce)
                - 1.0
            )
            / self.pe_den,
            lo=0.0,
        )
        flce = torch.exp(
            -(((muscle_len / self.l0_ce) - 1.0) ** 2) / self.ce_gamma
        )
        force = (activation * flce * fvce + flpe) * self.max_iso_force

        return torch.cat(
            [activation, muscle_len, muscle_vel, flpe, flce, fvce, force], dim=1
        )


# ---------------------------------------------------------------------
# CompliantTendonHillMuscle
# ---------------------------------------------------------------------


class CompliantTendonHillMuscle(RigidTendonHillMuscle):
    def __init__(self, min_activation: float = 0.01, **kwargs):
        super().__init__(min_activation=min_activation, **kwargs)
        self.__name__ = "CompliantTendonHillMuscle"
        self.state_name = [
            "activation",
            "muscle length",
            "muscle velocity",
            "force-length PE",
            "force-length SE",
            "active force",
            "force",
        ]
        self.state_dim = len(self.state_name)

    def _integrate(
        self,
        dt: float,
        state_derivative: Tensor,
        muscle_state: Tensor,
        geometry_state: Tensor,
    ) -> Tensor:
        # Current muscle length and velocity (normalized velocity will be in ODE call)
        muscle_len = muscle_state[:, 1:2, :]
        muscle_len_n = muscle_len / self.l0_ce
        musculotendon_len = geometry_state[:, :1, :]
        tendon_len = musculotendon_len - muscle_len
        tendon_strain = _clip((tendon_len - self.l0_se) / self.l0_se, lo=0.0)
        muscle_strain = _clip((muscle_len - self.l0_pe) / self.l0_ce, lo=0.0)

        flse = _clip(self.k_se * (tendon_strain**2), hi=1.0)
        flpe = self.k_pe * (muscle_strain**2)
        active_force = _clip(flse - flpe, lo=0.0)

        # Integrate activation and normalized muscle velocity
        dt_t = torch.as_tensor(dt, dtype=self.dtype, device=self.device)
        d_activation = state_derivative[:, 0:1, :]
        muscle_vel_n = state_derivative[:, 1:2, :]
        activation = self.clip_activation(
            muscle_state[:, 0:1, :] + d_activation * dt_t
        )
        new_muscle_len = (muscle_len_n + dt_t * muscle_vel_n) * self.l0_ce

        muscle_vel = muscle_vel_n * self.vmax
        force = flse * self.max_iso_force
        return torch.cat(
            [activation, new_muscle_len, muscle_vel, flpe, flse, active_force, force],
            dim=1,
        )

    def _ode(self, excitation: Tensor, muscle_state: Tensor) -> Tensor:
        activation = muscle_state[:, 0:1, :]
        d_activation = self.activation_ode(excitation, activation)
        muscle_len_n = muscle_state[:, 1:2, :] / self.l0_ce
        active_force = muscle_state[:, 5:6, :]
        new_muscle_vel_n = self._normalized_muscle_vel(
            muscle_len_n, activation, active_force
        )
        return torch.cat([d_activation, new_muscle_vel_n], dim=1)

    def _get_initial_muscle_state(
        self, batch_size: int, geometry_state: Tensor
    ) -> Tensor:
        musculotendon_len = geometry_state[:, 0:1, :]
        activation = torch.ones_like(
            musculotendon_len, dtype=self.dtype, device=self.device
        ) * self.min_activation
        d_activation = torch.zeros_like(
            musculotendon_len, dtype=self.dtype, device=self.device
        )

        # initial muscle length via piecewise conditions (same as original)
        cond_neg = musculotendon_len < 0.0
        cond_lt_l0se = musculotendon_len < self.l0_se
        cond_lt_sum = musculotendon_len < (self.l0_se + self.l0_pe)

        num = (
            self.k_pe * self.l0_pe * self.l0_se**2
            - self.k_se * (self.l0_ce**2) * musculotendon_len
            + self.k_se * self.l0_ce**2 * self.l0_se
            - self.l0_ce
            * self.l0_se
            * torch.sqrt(self.k_pe * self.k_se)
            * (-musculotendon_len + self.l0_pe + self.l0_se)
        )
        den = self.k_pe * self.l0_se**2 - self.k_se * self.l0_ce**2
        expr = num / den

        muscle_len = torch.where(
            cond_neg,
            torch.full_like(musculotendon_len, -1.0),
            torch.where(
                cond_lt_l0se,
                0.001 * self.l0_ce,
                torch.where(
                    cond_lt_sum,
                    musculotendon_len - self.l0_se,
                    expr,
                ),
            ),
        )

        tendon_len = musculotendon_len - muscle_len
        tendon_strain = _clip((tendon_len - self.l0_se) / self.l0_se, lo=0.0)
        muscle_strain = _clip((muscle_len - self.l0_pe) / self.l0_ce, lo=0.0)

        flse = _clip(self.k_se * (tendon_strain**2), hi=1.0)
        flpe = _clip(self.k_pe * (muscle_strain**2), hi=1.0)
        active_force = _clip(flse - flpe, lo=0.0)

        muscle_vel_n = self._normalized_muscle_vel(
            muscle_len / self.l0_ce, activation, active_force
        )
        muscle_state0 = torch.cat([activation, muscle_len], dim=1)
        state_derivative0 = torch.cat([d_activation, muscle_vel_n], dim=1)
        return self.integrate(self.dt, state_derivative0, muscle_state0, geometry_state)

    def _normalized_muscle_vel(
        self, muscle_len_n: Tensor, activation: Tensor, active_force: Tensor
    ) -> Tensor:
        flce = _clip(
            1.0
            + (-(muscle_len_n**2) + 2.0 * muscle_len_n - 1.0) / (0.66**2),
            lo=self.min_flce,
        )
        a_rel_st = torch.where(
            muscle_len_n < 1.0, 0.41 * flce, torch.full_like(flce, 0.41)
        )
        b_rel_st = torch.where(
            activation < self.q_crit,
            5.2
            * (1.0 - 0.9 * ((activation - self.q_crit) / (5e-3 - self.q_crit))) ** 2,
            torch.full_like(activation, 5.2),
        )
        f_x_a = flce * activation
        dfdvcon0 = (f_x_a + activation * a_rel_st) / b_rel_st

        p1 = -f_x_a * 0.5 / (self.s_as - dfdvcon0 * 2.0)
        p3 = -1.5 * f_x_a
        p2_containing_term = (4.0 * ((f_x_a * 0.5) ** 2) * (-self.s_as)) / (
            self.s_as - dfdvcon0 * 2.0
        )

        sqrt_term = (
            active_force**2
            + 2.0 * active_force * p1 * self.s_as
            + 2.0 * active_force * p3
            + p1**2 * self.s_as**2
            + 2.0 * p1 * p3 * self.s_as
            + p2_containing_term
            + p3**2
        )
        sqrt_term = _clip(sqrt_term, lo=0.0)

        new_muscle_vel_nom = torch.where(
            active_force < f_x_a,
            b_rel_st * (active_force - f_x_a),
            -active_force + p1 * self.s_as - p3 - torch.sqrt(sqrt_term),
        )
        new_muscle_vel_den = torch.where(
            active_force < f_x_a,
            active_force + activation * a_rel_st,
            torch.full_like(active_force, -2.0 * self.s_as),
        )
        return new_muscle_vel_nom / new_muscle_vel_den


# ---------------------------------------------------------------------
# Smoke tests (safe to run directly)
# ---------------------------------------------------------------------


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    batch = 2
    n = 3
    dt = 0.002

    # geometry_state: [:,0,:]=path_length (m), [:,1,:]=path_velocity (m/s)
    geometry_state = torch.zeros((batch, 2, n), dtype=torch.float64)
    geometry_state[:, 0, :] = 0.20 + 0.05 * torch.rand((batch, n), dtype=torch.float64)
    geometry_state[:, 1, :] = -0.05 + 0.10 * torch.rand((batch, n), dtype=torch.float64)

    # sample excitation / drive: shape (batch,1,n)
    action = torch.clamp(
        0.3 + 0.1 * torch.randn((batch, 1, n), dtype=torch.float64), 0.0, 1.0
    )

    print("\n=== ReluMuscle (Torch) ===")
    relu = ReluMuscle(device="cpu", dtype=torch.float64)
    relu.build(timestep=dt, max_isometric_force=[100.0, 120.0, 150.0])
    m0 = relu.get_initial_muscle_state(batch, geometry_state)
    d = relu.ode(action, m0[:, :1, :])
    m1 = relu.integrate(dt, d, m0[:, :1, :], geometry_state)
    print("state0 shape:", m0.shape, "| state1 shape:", m1.shape)
    print("state1[0,:,0] (activation,len,vel,force):", m1[0, :, 0])

    print("\n=== MujocoHillMuscle (Torch) ===")
    muj = MujocoHillMuscle(passive_forces=1.0, device="cpu", dtype=torch.float64)
    muj.build(
        timestep=dt,
        max_isometric_force=[120.0, 110.0, 130.0],
        tendon_length=[0.12, 0.13, 0.11],
        optimal_muscle_length=[0.10, 0.095, 0.105],
        normalized_slack_muscle_length=[1.3, 1.3, 1.3],
        lmin=[0.5, 0.5, 0.5],
        lmax=[1.6, 1.6, 1.6],
        vmax=[1.5, 1.5, 1.5],
        fvmax=[1.2, 1.2, 1.2],
    )
    m0 = muj.get_initial_muscle_state(batch, geometry_state)
    d = muj.ode(action, m0[:, :1, :])
    m1 = muj.integrate(dt, d, m0[:, :1, :], geometry_state)
    print("state0 shape:", m0.shape, "| state1 shape:", m1.shape)
    print("channels:", muj.state_name)
    print("state1[0,:,0]:", m1[0, :, 0])

    print("\n=== RigidTendonHillMuscle (Torch) ===")
    rth = RigidTendonHillMuscle(min_activation=0.001, device="cpu", dtype=torch.float64)
    rth.build(
        timestep=dt,
        max_isometric_force=[140.0, 135.0, 145.0],
        tendon_length=[0.14, 0.13, 0.12],
        optimal_muscle_length=[0.10, 0.10, 0.10],
        normalized_slack_muscle_length=[1.4, 1.4, 1.4],
    )
    m0 = rth.get_initial_muscle_state(batch, geometry_state)
    d = rth.ode(action, m0[:, :1, :])
    m1 = rth.integrate(dt, d, m0[:, :1, :], geometry_state)
    print("state0 shape:", m0.shape, "| state1 shape:", m1.shape)
    print("channels:", rth.state_name)
    print("state1[0,:,0]:", m1[0, :, 0])

    print("\n=== RigidTendonHillMuscleThelen (Torch) ===")
    thl = RigidTendonHillMuscleThelen(
        min_activation=0.001, device="cpu", dtype=torch.float64
    )
    thl.build(
        timestep=dt,
        max_isometric_force=[150.0, 150.0, 150.0],
        tendon_length=[0.13, 0.13, 0.13],
        optimal_muscle_length=[0.10, 0.10, 0.10],
        normalized_slack_muscle_length=1.0,
    )
    m0 = thl.get_initial_muscle_state(batch, geometry_state)
    d = thl.ode(action, m0[:, :1, :])
    m1 = thl.integrate(dt, d, m0[:, :1, :], geometry_state)
    print("state0 shape:", m0.shape, "| state1 shape:", m1.shape)
    print("channels:", thl.state_name)
    print("state1[0,:,0]:", m1[0, :, 0])

    print("\n=== CompliantTendonHillMuscle (Torch) ===")
    cth = CompliantTendonHillMuscle(
        min_activation=0.01, device="cpu", dtype=torch.float64
    )
    cth.build(
        timestep=dt,
        max_isometric_force=[160.0, 150.0, 140.0],
        tendon_length=[0.12, 0.12, 0.12],
        optimal_muscle_length=[0.11, 0.10, 0.105],
        normalized_slack_muscle_length=[1.4, 1.4, 1.4],
    )
    m0 = cth.get_initial_muscle_state(batch, geometry_state)
    d = cth.ode(action, m0)
    m1 = cth.integrate(dt, d, m0, geometry_state)
    print("state0 shape:", m0.shape, "| state1 shape:", m1.shape)
    print("channels:", cth.state_name)
    print("state1[0,:,0]:", m1[0, :, 0])

    print("\n[Muscles Torch] Smoke tests complete ✓")
