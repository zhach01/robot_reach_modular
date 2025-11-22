# -*- coding: utf-8 -*-
"""
Torch-free muscle models implemented with NumPy only.

Shapes follow the original conventions:
- geometry_state: (batch, 2, n_muscles) where [:,0,:] = path_length, [:,1,:] = path_velocity
- muscle_state: model-specific, but always (batch, state_dim, n_muscles)
- actions / excitations: (batch, 1, n_muscles) or broadcastable to that shape

Classes:
- Muscle (base)
- ReluMuscle
- MujocoHillMuscle
- RigidTendonHillMuscle
- RigidTendonHillMuscleThelen
- CompliantTendonHillMuscle
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
import numpy as np


# -----------------------
# Utilities
# -----------------------


def _as_3d(arr, n_muscles: int = None) -> np.ndarray:
    """Ensure arr is 3D (batch, channels, n_muscles)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        # (n_muscles,)
        arr = arr.reshape(1, 1, -1)
    elif arr.ndim == 2:
        # (batch, channels)
        if n_muscles is None:
            raise ValueError(
                "Cannot promote a (batch,channels) array to 3D without n_muscles."
            )
        arr = arr.reshape(arr.shape[0], arr.shape[1], n_muscles)
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError(f"Expected 1–3D array, got {arr.ndim}D")
    return arr


def _to_param(x, n_muscles: int) -> np.ndarray:
    """Make a (1,1,n_muscles) array out of scalar/list-like x."""
    a = np.asarray(x, dtype=float).reshape(1, 1, -1)
    if a.size not in (1, n_muscles):
        raise ValueError(
            f"Parameter size must be 1 or n_muscles={n_muscles}, got {a.size}"
        )
    if a.size == 1:
        a = np.ones((1, 1, n_muscles), dtype=float) * float(a.reshape(()))
    return a


def _clip(x, lo=None, hi=None):
    if lo is None and hi is None:
        return x
    if lo is None:
        return np.minimum(x, hi)
    if hi is None:
        return np.maximum(x, lo)
    return np.clip(x, lo, hi)


# -----------------------
# Base Muscle
# -----------------------


@dataclass
class Muscle:
    input_dim: int = 1
    output_dim: int = 1
    min_activation: float = 0.0
    tau_activation: float = 0.015
    tau_deactivation: float = 0.05

    # runtime-populated
    state_name: List[str] = None
    dt: float = None
    n_muscles: int = None
    max_iso_force: np.ndarray = None  # (1,1,nm)
    vmax: np.ndarray = None  # (1,1,nm)
    l0_se: np.ndarray = None  # (1,1,nm)
    l0_ce: np.ndarray = None  # (1,1,nm)
    l0_pe: np.ndarray = None  # (1,1,nm)
    built: bool = False

    # optional knobs recorded like the torch version
    to_build_dict: Dict[str, list] = None
    to_build_dict_default: Dict[str, Any] = None

    def __post_init__(self):
        if self.state_name is None:
            self.state_name = []
        if self.to_build_dict is None:
            self.to_build_dict = {"max_isometric_force": []}
        if self.to_build_dict_default is None:
            self.to_build_dict_default = {}

    # --- helpers

    def clip_activation(self, a: np.ndarray) -> np.ndarray:
        return _clip(a, self.min_activation, 1.0)

    def clip_excitation(self, u):
        """        Clamp *excitations* u to [min_activation, 1.0].
        Note: activation a is the internal state updated by activation dynamics.
        """
        u = np.asarray(u, dtype=float)       
        return np.clip(u, self.min_activation, 1.0)

    # Back-compat alias (older code calls this):
    clip_activation = clip_excitation

    # --- build

    def build(self, timestep, max_isometric_force, **kwargs):
        """Base build: sets common parameters to (1,1,n_muscles) arrays."""
        mif = np.asarray(max_isometric_force, dtype=float).reshape(1, 1, -1)
        self.n_muscles = mif.size
        self.max_iso_force = mif.copy()
        self.dt = float(timestep)
        ones = np.ones((1, 1, self.n_muscles), dtype=float)
        self.vmax = ones.copy()
        self.l0_se = ones.copy()
        self.l0_ce = ones.copy()
        self.l0_pe = ones.copy()
        self.built = True

    # --- API

    def get_initial_muscle_state(
        self, batch_size: int, geometry_state: np.ndarray
    ) -> np.ndarray:
        return self._get_initial_muscle_state(batch_size, geometry_state)

    def _get_initial_muscle_state(
        self, batch_size: int, geometry_state: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    def integrate(
        self,
        dt: float,
        state_derivative: np.ndarray,
        muscle_state: np.ndarray,
        geometry_state: np.ndarray,
    ) -> np.ndarray:
        return self._integrate(dt, state_derivative, muscle_state, geometry_state)

    def _integrate(
        self,
        dt: float,
        state_derivative: np.ndarray,
        muscle_state: np.ndarray,
        geometry_state: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def ode(self, action: np.ndarray, muscle_state: np.ndarray) -> np.ndarray:
        return self._ode(action, muscle_state)

    def _ode(self, action: np.ndarray, muscle_state: np.ndarray) -> np.ndarray:
        activation = muscle_state[:, :1, :]
        return self.activation_ode(action, activation)

    def activation_ode(self, action: np.ndarray, activation: np.ndarray) -> np.ndarray:
        """Thelen-style activation dynamics (same as original Torch logic)."""
        action = self.clip_activation(np.reshape(action, (-1, 1, self.n_muscles)))
        activation = self.clip_activation(activation)
        tmp = 0.5 + 1.5 * activation
        tau = np.where(
            action > activation, self.tau_activation * tmp, self.tau_deactivation / tmp
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


# -----------------------
# ReluMuscle
# -----------------------


class ReluMuscle(Muscle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__name__ = "ReluMuscle"
        self.state_name = ["activation", "muscle length", "muscle velocity", "force"]
        self.state_dim = len(self.state_name)

    def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
        activation = muscle_state[:, :1, :] + state_derivative * dt
        activation = self.clip_activation(activation)
        forces = activation * self.max_iso_force
        len_vel = geometry_state[:, :2, :]
        return np.concatenate([activation, len_vel, forces], axis=1)

    def _get_initial_muscle_state(self, batch_size, geometry_state):
        shape = geometry_state[:, :1, :].shape
        activation0 = np.ones(shape, dtype=float) * self.min_activation
        force0 = np.zeros(shape, dtype=float)
        len_vel = geometry_state[:, 0:2, :]
        return np.concatenate([activation0, len_vel, force0], axis=1)


# -----------------------
# MujocoHillMuscle
# -----------------------


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
        self.b = None
        self.c = None
        self.p1 = None
        self.p2 = None
        self.mid = None

    def build(
        self,
        timestep,
        max_isometric_force,
        tendon_length,
        optimal_muscle_length,
        normalized_slack_muscle_length,
        lmin,
        lmax,
        vmax,
        fvmax,
    ):
        self.n_muscles = np.asarray(tendon_length).size
        self.dt = float(timestep)

        self.max_iso_force = _to_param(max_isometric_force, self.n_muscles)
        self.l0_pe = _to_param(normalized_slack_muscle_length, self.n_muscles)
        self.l0_ce = _to_param(optimal_muscle_length, self.n_muscles)
        self.l0_se = _to_param(tendon_length, self.n_muscles)
        self.lmin = _to_param(lmin, self.n_muscles)
        self.lmax = _to_param(lmax, self.n_muscles)
        self.vmax = _to_param(vmax, self.n_muscles)
        self.fvmax = _to_param(fvmax, self.n_muscles)

        # derived
        self.b = 0.5 * (1 + self.lmax)
        self.c = self.fvmax - 1
        self.p1 = self.b - 1
        self.p2 = 0.25 * self.l0_pe
        self.mid = 0.5 * (self.lmin + 0.95)
        self.built = True

    def _bump(self, L, mid, lmax):
        """Quadratic spline 'bump' as per original logic."""
        left = 0.5 * (self.lmin + mid)
        right = 0.5 * (mid + lmax)

        out_of_range = (L <= self.lmin) | (L >= lmax)
        less_than_left = L < left
        less_than_mid = L < mid
        less_than_right = L < right

        x = np.where(
            out_of_range,
            0.0,
            np.where(
                less_than_left,
                (L - self.lmin) / (left - self.lmin),
                np.where(
                    less_than_mid,
                    (mid - L) / (mid - left),
                    np.where(
                        less_than_right,
                        (L - mid) / (right - mid),
                        (lmax - L) / (lmax - right),
                    ),
                ),
            ),
        )
        pfivexx = 0.5 * x * x
        y = np.where(
            out_of_range,
            0.0,
            np.where(
                less_than_left,
                pfivexx,
                np.where(
                    less_than_mid,
                    1 - pfivexx,
                    np.where(less_than_right, 1 - pfivexx, pfivexx),
                ),
            ),
        )

        return y

    def _get_initial_muscle_state(self, batch_size, geometry_state):
        shape = geometry_state[:, :1, :].shape
        muscle_state = np.ones(shape, dtype=float) * self.min_activation
        state_derivatives = np.zeros(shape, dtype=float)
        return self.integrate(self.dt, state_derivatives, muscle_state, geometry_state)

    def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
        activation = muscle_state[:, :1, :] + state_derivative * dt
        activation = self.clip_activation(activation)

        # geometry
        musculotendon_len = geometry_state[:, :1, :]
        muscle_len = _clip(
            (musculotendon_len - self.l0_se) / self.l0_ce, lo=0.001
        )  # normalized
        muscle_vel = geometry_state[:, 1:2, :] / self.vmax

        # passive length element (flpe)
        # x = piecewise in original code
        x = np.where(
            muscle_len <= 1,
            0.0,
            np.where(
                muscle_len <= self.b,
                (muscle_len - 1) / self.p1,
                (muscle_len - self.b) / self.p1,
            ),
        )
        flpe = np.where(
            muscle_len <= 1,
            0.0,
            np.where(muscle_len <= self.b, self.p2 * x**3, self.p2 * (1 + 3 * x)),
        )

        # active length (flce)
        flce = self._bump(muscle_len, mid=1, lmax=self.lmax) + 0.15 * self._bump(
            muscle_len, mid=self.mid, lmax=0.95
        )

        # force-velocity CE (fvce)
        fvce = np.where(
            muscle_vel <= -1,
            0.0,
            np.where(
                muscle_vel <= 0.0,
                (muscle_vel + 1) * (muscle_vel + 1),
                np.where(
                    muscle_vel <= self.c,
                    self.fvmax - (self.c - muscle_vel) * (self.c - muscle_vel) / self.c,
                    self.fvmax,
                ),
            ),
        )

        force = (
            activation * flce * fvce + self.passive_forces * flpe
        ) * self.max_iso_force
        # return activation, (denormalized length, velocity), flpe, flce, fvce, force
        return np.concatenate(
            [
                activation,
                muscle_len * self.l0_ce,
                muscle_vel * self.vmax,
                flpe,
                flce,
                fvce,
                force,
            ],
            axis=1,
        )


# -----------------------
# RigidTendonHillMuscle
# -----------------------


class RigidTendonHillMuscle(Muscle):
    def __init__(self, min_activation=0.001, **kwargs):
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
        self.pe_den = np.exp(self.pe_k) - 1
        self.ce_gamma = 0.45
        self.ce_Af = 0.25
        self.ce_fmlen = 1.4

        # derived params set in build()
        self.musculotendon_slack_len = None
        self.k_pe = None
        self.s_as = 0.001
        self.f_iso_n_den = 0.66**2
        self.k_se = 1 / (0.04**2)
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
        timestep,
        max_isometric_force,
        tendon_length,
        optimal_muscle_length,
        normalized_slack_muscle_length,
    ):
        self.n_muscles = np.asarray(tendon_length).size
        shape = (1, 1, self.n_muscles)

        self.dt = float(timestep)
        # NEW (broadcast scalars or per-muscle lists):
        self.max_iso_force = _to_param(max_isometric_force, self.n_muscles)
        self.l0_ce = _to_param(optimal_muscle_length, self.n_muscles)
        self.l0_se = _to_param(tendon_length, self.n_muscles)
        self.l0_pe = (
            _to_param(normalized_slack_muscle_length, self.n_muscles) * self.l0_ce
        )

        self.k_pe = 1.0 / ((1.66 - self.l0_pe / self.l0_ce) ** 2)
        self.musculotendon_slack_len = self.l0_pe + self.l0_se
        self.vmax = 10 * self.l0_ce
        self.built = True

    def _get_initial_muscle_state(self, batch_size, geometry_state):
        shape = geometry_state[:, :1, :].shape
        muscle_state = np.ones(shape, dtype=float) * self.min_activation
        state_derivatives = np.zeros(shape, dtype=float)
        return self.integrate(self.dt, state_derivatives, muscle_state, geometry_state)

    def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
        activation = self.clip_activation(
            muscle_state[:, :1, :] + state_derivative * dt
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
            1 + (-(muscle_len_n**2) + 2 * muscle_len_n - 1) / self.f_iso_n_den,
            lo=self.min_flce,
        )

        a_rel_st = np.where(muscle_len_n > 1.0, 0.41 * flce, 0.41)
        b_rel_st = np.where(
            activation < self.q_crit,
            5.2 * (1 - 0.9 * ((activation - self.q_crit) / (5e-3 - self.q_crit))) ** 2,
            5.2,
        )
        dfdvcon0 = activation * (flce + a_rel_st) / b_rel_st

        f_x_a = flce * activation
        tmp_p_nom = f_x_a * 0.5
        tmp_p_den = self.s_as - dfdvcon0 * 2.0

        p1 = -tmp_p_nom / tmp_p_den
        p2 = (tmp_p_nom**2) / tmp_p_den
        p3 = -1.5 * f_x_a

        nom = np.where(
            muscle_vel_n < 0,
            muscle_vel_n * activation * a_rel_st + f_x_a * b_rel_st,
            -p1 * p3
            + p1 * self.s_as * muscle_vel_n
            + p2
            - p3 * muscle_vel_n
            + self.s_as * muscle_vel_n**2,
        )
        den = np.where(muscle_vel_n < 0, b_rel_st - muscle_vel_n, p1 + muscle_vel_n)

        active_force = _clip(nom / den, lo=0.0)
        force = (active_force + flpe) * self.max_iso_force

        return np.concatenate(
            [activation, muscle_len, muscle_vel, flpe, flce, active_force, force],
            axis=1,
        )


# -----------------------
# RigidTendonHillMuscleThelen
# -----------------------


class RigidTendonHillMuscleThelen(Muscle):
    def __init__(self, min_activation=0.001, **kwargs):
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

        # parameters (as scalars -> broadcast in numpy)
        self.pe_k = 5.0
        self.pe_1 = self.pe_k / 0.6
        self.pe_den = np.exp(self.pe_k) - 1.0
        self.ce_gamma = 0.45
        self.ce_Af = 0.25
        self.ce_fmlen = 1.4

        # precomputed (in build)
        self.ce_0 = None
        self.ce_1 = None
        self.ce_2 = None
        self.ce_3 = None
        self.ce_4 = None
        self.ce_5 = None

        self.to_build_dict = {
            "max_isometric_force": [],
            "tendon_length": [],
            "optimal_muscle_length": [],
            "normalized_slack_muscle_length": [],
        }
        self.to_build_dict_default = {"normalized_slack_muscle_length": 1.0}

    def build(
        self,
        timestep,
        max_isometric_force,
        tendon_length,
        optimal_muscle_length,
        normalized_slack_muscle_length,
    ):
        self.n_muscles = np.asarray(tendon_length).size
        self.dt = float(timestep)

        # broadcast scalars or accept per-muscle lists
        self.max_iso_force = _to_param(max_isometric_force, self.n_muscles)
        self.l0_ce = _to_param(optimal_muscle_length, self.n_muscles)
        self.l0_se = _to_param(tendon_length, self.n_muscles)
        nsl = _to_param(normalized_slack_muscle_length, self.n_muscles)

        self.l0_pe = self.l0_ce * nsl
        self.musculotendon_slack_len = self.l0_pe + self.l0_se
        self.vmax = 10 * self.l0_ce

        # precompute
        self.ce_0 = 3 * self.vmax
        self.ce_1 = self.ce_Af * self.vmax
        self.ce_2 = (
            3 * self.ce_Af * self.vmax * self.ce_fmlen - 3.0 * self.ce_Af * self.vmax
        )
        self.ce_3 = 8 * self.ce_Af * self.ce_fmlen + 8.0 * self.ce_fmlen
        self.ce_4 = self.ce_Af * self.ce_fmlen * self.vmax - self.ce_1
        self.ce_5 = 8.0 * (self.ce_Af + 1.0)

        self.built = True

    def _get_initial_muscle_state(self, batch_size, geometry_state):
        shape = geometry_state[:, :1, :].shape
        muscle_state = np.ones(shape, dtype=float) * self.min_activation
        state_derivatives = np.zeros(shape, dtype=float)
        return self.integrate(self.dt, state_derivatives, muscle_state, geometry_state)

    def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
        activation = self.clip_activation(
            muscle_state[:, :1, :] + state_derivative * dt
        )

        # geometry
        musculotendon_len = geometry_state[:, :1, :]
        muscle_len = _clip(musculotendon_len - self.l0_se, lo=0.001)
        muscle_vel = geometry_state[:, 1:2, :]

        a3 = activation * 3.0
        condition = muscle_vel <= 0
        nom = np.where(
            condition,
            self.ce_Af * (activation * self.ce_0 + 4.0 * muscle_vel + self.vmax),
            self.ce_2 * activation + self.ce_3 * muscle_vel + self.ce_4,
        )
        den = np.where(
            condition,
            a3 * self.ce_1 + self.ce_1 - 4.0 * muscle_vel,
            self.ce_4 * a3 + self.ce_5 * muscle_vel + self.ce_4,
        )
        fvce = _clip(nom / den, lo=0.0)
        flpe = _clip(
            (np.exp(self.pe_1 * (muscle_len - self.l0_pe) / self.l0_ce) - 1.0)
            / self.pe_den,
            lo=0.0,
        )
        flce = np.exp(-(((muscle_len / self.l0_ce) - 1.0) ** 2) / self.ce_gamma)
        force = (activation * flce * fvce + flpe) * self.max_iso_force

        return np.concatenate(
            [activation, muscle_len, muscle_vel, flpe, flce, fvce, force], axis=1
        )


# -----------------------
# CompliantTendonHillMuscle
# -----------------------


class CompliantTendonHillMuscle(RigidTendonHillMuscle):
    def __init__(self, min_activation=0.01, **kwargs):
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

    def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
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
        d_activation = state_derivative[:, 0:1, :]
        muscle_vel_n = state_derivative[:, 1:2, :]
        activation = self.clip_activation(muscle_state[:, 0:1, :] + d_activation * dt)
        new_muscle_len = (muscle_len_n + dt * muscle_vel_n) * self.l0_ce

        muscle_vel = muscle_vel_n * self.vmax
        force = flse * self.max_iso_force
        return np.concatenate(
            [activation, new_muscle_len, muscle_vel, flpe, flse, active_force, force],
            axis=1,
        )

    def _ode(self, excitation, muscle_state):
        activation = muscle_state[:, 0:1, :]
        d_activation = self.activation_ode(excitation, activation)
        muscle_len_n = muscle_state[:, 1:2, :] / self.l0_ce
        active_force = muscle_state[:, 5:6, :]
        new_muscle_vel_n = self._normalized_muscle_vel(
            muscle_len_n, activation, active_force
        )
        return np.concatenate([d_activation, new_muscle_vel_n], axis=1)

    def _get_initial_muscle_state(self, batch_size, geometry_state):
        musculotendon_len = geometry_state[:, 0:1, :]
        activation = np.ones_like(musculotendon_len, dtype=float) * self.min_activation
        d_activation = np.zeros_like(musculotendon_len, dtype=float)

        # initial muscle length via piecewise conditions (same as original)
        cond_neg = musculotendon_len < 0.0
        cond_lt_l0se = musculotendon_len < self.l0_se
        cond_lt_sum = musculotendon_len < self.l0_se + self.l0_pe

        # last case (analytical expression) — compute fully then select
        num = (
            self.k_pe * self.l0_pe * self.l0_se**2
            - self.k_se * (self.l0_ce**2) * musculotendon_len
            + self.k_se * self.l0_ce**2 * self.l0_se
            - self.l0_ce
            * self.l0_se
            * np.sqrt(self.k_pe * self.k_se)
            * (-musculotendon_len + self.l0_pe + self.l0_se)
        )
        den = self.k_pe * self.l0_se**2 - self.k_se * self.l0_ce**2
        expr = num / den

        muscle_len = np.where(
            cond_neg,
            -1.0,
            np.where(
                cond_lt_l0se,
                0.001 * self.l0_ce,
                np.where(cond_lt_sum, musculotendon_len - self.l0_se, expr),
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
        muscle_state0 = np.concatenate([activation, muscle_len], axis=1)
        state_derivative0 = np.concatenate([d_activation, muscle_vel_n], axis=1)
        return self.integrate(self.dt, state_derivative0, muscle_state0, geometry_state)

    def _normalized_muscle_vel(self, muscle_len_n, activation, active_force):
        flce = _clip(
            1.0 + (-(muscle_len_n**2) + 2 * muscle_len_n - 1) / (0.66**2),
            lo=self.min_flce,
        )
        a_rel_st = np.where(muscle_len_n < 1.0, 0.41 * flce, 0.41)
        b_rel_st = np.where(
            activation < self.q_crit,
            5.2 * (1 - 0.9 * ((activation - self.q_crit) / (5e-3 - self.q_crit))) ** 2,
            5.2,
        )
        f_x_a = flce * activation
        dfdvcon0 = (f_x_a + activation * a_rel_st) / b_rel_st

        p1 = -f_x_a * 0.5 / (self.s_as - dfdvcon0 * 2.0)
        p3 = -1.5 * f_x_a
        p2_containing_term = (4 * ((f_x_a * 0.5) ** 2) * (-self.s_as)) / (
            self.s_as - dfdvcon0 * 2.0
        )

        sqrt_term = (
            active_force**2
            + 2 * active_force * p1 * self.s_as
            + 2 * active_force * p3
            + p1**2 * self.s_as**2
            + 2 * p1 * p3 * self.s_as
            + p2_containing_term
            + p3**2
        )
        sqrt_term = _clip(sqrt_term, lo=0.0)

        new_muscle_vel_nom = np.where(
            active_force < f_x_a,
            b_rel_st * (active_force - f_x_a),
            -active_force + p1 * self.s_as - p3 - np.sqrt(sqrt_term),
        )
        new_muscle_vel_den = np.where(
            active_force < f_x_a, active_force + activation * a_rel_st, -2.0 * self.s_as
        )
        return new_muscle_vel_nom / new_muscle_vel_den


# ---------------------------------------------------------------------
# Smoke tests (safe to run directly)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    rng = np.random.default_rng(0)

    # Test bed
    batch = 2
    n = 3
    dt = 0.002

    # geometry_state: [:,0,:]=path_length (m), [:,1,:]=path_velocity (m/s)
    # keep lengths reasonable (>0), small velocities
    geometry_state = np.zeros((batch, 2, n), dtype=float)
    geometry_state[:, 0, :] = 0.20 + 0.05 * rng.random((batch, n))  # 0.20–0.25 m
    geometry_state[:, 1, :] = -0.05 + 0.10 * rng.random((batch, n))  # -0.05..+0.05 m/s

    # sample excitation/drive: shape (batch,1,n)
    action = np.clip(rng.normal(loc=0.3, scale=0.1, size=(batch, 1, n)), 0.0, 1.0)

    print("\n=== ReluMuscle ===")
    relu = ReluMuscle()
    relu.build(timestep=dt, max_isometric_force=[100.0, 120.0, 150.0])
    m0 = relu.get_initial_muscle_state(batch, geometry_state)
    d = relu.ode(action, m0[:, :1, :])  # uses activation_ode on first channel
    m1 = relu.integrate(dt, d, m0[:, :1, :], geometry_state)
    print("state0 shape:", m0.shape, "| state1 shape:", m1.shape)
    print("state1[0,:,0] (activation,len,vel,force):", m1[0, :, 0])

    print("\n=== MujocoHillMuscle ===")
    muj = MujocoHillMuscle(passive_forces=1.0)
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

    print("\n=== RigidTendonHillMuscle ===")
    rth = RigidTendonHillMuscle(min_activation=0.001)
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

    print("\n=== RigidTendonHillMuscleThelen ===")
    thl = RigidTendonHillMuscleThelen(min_activation=0.001)
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

    print("\n=== CompliantTendonHillMuscle ===")
    cth = CompliantTendonHillMuscle(min_activation=0.01)
    # uses the RigidTendonHillMuscle build signature
    cth.build(
        timestep=dt,
        max_isometric_force=[160.0, 150.0, 140.0],
        tendon_length=[0.12, 0.12, 0.12],
        optimal_muscle_length=[0.11, 0.10, 0.105],
        normalized_slack_muscle_length=[1.4, 1.4, 1.4],
    )
    m0 = cth.get_initial_muscle_state(batch, geometry_state)  # returns full state
    d = cth.ode(action, m0)  # d_activation & normalized vel from full state
    m1 = cth.integrate(dt, d, m0, geometry_state)  # updates full state
    print("state0 shape:", m0.shape, "| state1 shape:", m1.shape)
    print("channels:", cth.state_name)
    print("state1[0,:,0]:", m1[0, :, 0])

    print("\nSmoke tests complete ✓")
