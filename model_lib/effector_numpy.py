# -*- coding: utf-8 -*-
"""
Torch-free, NumPy-only Effector stack.

Requires:
- from your NumPy skeletons: TwoDofArm
- from your NumPy muscles: ReluMuscle, CompliantTendonHillMuscle

This file provides:
- Effector (base)
- RigidTendonArm26 (lumped 6-muscle model w/ polynomial moment arms)
- CompliantTendonArm26 (compliant tendon, RK4 default)

Everything uses NumPy arrays (no torch).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from gymnasium.utils import seeding  # ok to keep; it’s NumPy-based

# import your NumPy versions
# If they are in different modules, adjust these imports accordingly.
from model_lib.muscles_numpy import ReluMuscle, CompliantTendonHillMuscle
from model_lib.skeleton_numpy import TwoDofArm  # your earlier NumPy rewrite

# -----------------------
# Helpers
# -----------------------


def _as_batch(x: np.ndarray, width: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != width:
        raise ValueError(f"Expected shape (*,{width}), got {x.shape}")
    return x


def _clip(x, lo=None, hi=None):
    if lo is None and hi is None:
        return x
    if lo is None:
        return np.minimum(x, hi)
    if hi is None:
        return np.maximum(x, lo)
    return np.clip(x, lo, hi)


# -----------------------
# Effector (NumPy)
# -----------------------


class Effector:
    """
    Base class for Effectors (NumPy only).

    Args:
      skeleton: a NumPy-based Skeleton with .ode(), .integrate(), .path2cartesian(), .joint2cartesian()
      muscle: a NumPy-based Muscle with .ode(), .integrate()
      name: name
      n_ministeps: mini-integration steps per dt
      timestep: dt (sec)
      integration_method: 'euler' or any of {'rk4','rungekutta4','runge-kutta4','runge-kutta-4'}
      damping: scalar viscous damping on joints (torque = -damping * qdot)
      pos/vel bounds: override skeleton defaults if provided
    """

    def __init__(
        self,
        skeleton,
        muscle,
        name: str = "Effector",
        n_ministeps: int = 1,
        timestep: float = 0.01,
        integration_method: str = "euler",
        damping: float = 0.0,
        pos_lower_bound: Union[float, List[float], Tuple[float, ...]] = None,
        pos_upper_bound: Union[float, List[float], Tuple[float, ...]] = None,
        vel_lower_bound: Union[float, List[float], Tuple[float, ...]] = None,
        vel_upper_bound: Union[float, List[float], Tuple[float, ...]] = None,
    ):
        self.__name__ = name
        self.skeleton = skeleton
        self.muscle = muscle

        self.damping = float(damping)
        self.dof = self.skeleton.dof
        self.space_dim = self.skeleton.space_dim
        self.state_dim = self.skeleton.state_dim
        self.output_dim = self.skeleton.output_dim
        self.n_ministeps = int(n_ministeps)
        self.dt = float(timestep)
        self.minidt = self.dt / self.n_ministeps
        self.half_minidt = self.minidt / 2.0
        self.integration_method = integration_method.casefold()
        self._np_random = None
        self.seed = None

        # Handle bounds
        if pos_lower_bound is None:
            pos_lower_bound = self.skeleton.pos_lower_bound
        if pos_upper_bound is None:
            pos_upper_bound = self.skeleton.pos_upper_bound
        if vel_lower_bound is None:
            vel_lower_bound = self.skeleton.vel_lower_bound
        if vel_upper_bound is None:
            vel_upper_bound = self.skeleton.vel_upper_bound

        pos_bounds = self._set_state_limit_bounds(
            lb=pos_lower_bound, ub=pos_upper_bound
        )
        vel_bounds = self._set_state_limit_bounds(
            lb=vel_lower_bound, ub=vel_upper_bound
        )
        self.pos_lower_bound = pos_bounds[:, 0]
        self.pos_upper_bound = pos_bounds[:, 1]
        self.vel_lower_bound = vel_bounds[:, 0]
        self.vel_upper_bound = vel_bounds[:, 1]

        # skeleton gets dt + bounds
        self.skeleton.build(
            timestep=self.dt,
            pos_upper_bound=self.pos_upper_bound,
            pos_lower_bound=self.pos_lower_bound,
            vel_upper_bound=self.vel_upper_bound,
            vel_lower_bound=self.vel_lower_bound,
        )

        # muscle “API” descriptors
        self.force_index = self.muscle.state_name.index(
            "force"
        )  # channel index containing output force
        self.MusclePaths: List = []
        self.n_muscles = 0
        self.input_dim = 0
        self.muscle_name: List[str] = []
        self.muscle_state_dim = self.muscle.state_dim
        self.geometry_state_dim = 2 + self.dof
        self.geometry_state_name = [
            "musculotendon length",
            "musculotendon velocity",
        ] + [f"moment for joint {d}" for d in range(self.dof)]

        self.tobuild__muscle = dict(
            self.muscle.to_build_dict
        )  # deep-ish copy of keys -> lists
        self.tobuild__default = dict(self.muscle.to_build_dict_default)

        # muscle wrapping accumulators (NumPy)
        self._path_fixation_body = np.empty((1, 1, 0), dtype=np.float32)
        self._path_coordinates = np.empty(
            (1, self.skeleton.space_dim, 0), dtype=np.float32
        )
        self._muscle_index = np.empty(0, dtype=np.float32)
        self._muscle_transitions = None  # (1,1,N-1) bool
        self._row_splits = None
        # final (NumPy) views
        self.path_fixation_body = None
        self.path_coordinates = None
        self.muscle_index = None
        self.muscle_transitions = None
        self.row_splits = None
        self.section_splits = None
        self._muscle_config_is_empty = True

        # default loads
        self.default_endpoint_load = np.zeros((1, self.skeleton.space_dim), dtype=float)
        self.default_joint_load = np.zeros((1, self.skeleton.dof), dtype=float)

        # integration selector
        if self.integration_method == "euler":
            self._integrate = self._euler
        elif self.integration_method in (
            "rk4",
            "rungekutta4",
            "runge-kutta4",
            "runge-kutta-4",
        ):
            self._integrate = self._rungekutta4
        else:
            raise ValueError(
                f"Integration method not recognized: {self.integration_method}"
            )

        # state dict
        self.states: Dict[str, np.ndarray] = {
            k: None for k in ["joint", "cartesian", "muscle", "geometry", "fingertip"]
        }

    # ---------------
    # Core loop
    # ---------------

    def step_OLD(self, action, **kwargs):
        endpoint_load = kwargs.get("endpoint_load", self.default_endpoint_load)
        joint_load = kwargs.get("joint_load", self.default_joint_load)
        a = self.muscle.clip_activation(action)
        for _ in range(self.n_ministeps):
            self.integrate(a, endpoint_load, joint_load)

    def step(self, action, **kwargs):
        """
        action: excitations u. Accepts (B, M) or (B, 1, M). Internally we use (B,1,M).
        """
        endpoint_load = kwargs.get("endpoint_load", self.default_endpoint_load)
        joint_load = kwargs.get("joint_load", self.default_joint_load)

        action = np.asarray(action, dtype=float)
        if action.ndim == 2:
            # (B, M) -> (B,1,M)
            action = action[:, None, :]
        elif action.ndim != 3:
            raise ValueError(f"action must be (B,M) or (B,1,M); got shape {action.shape}")

        # Clamp *excitations* (naming clarity; behavior identical to your old clip_activation)
        a = self.muscle.clip_excitation(action)

        for _ in range(self.n_ministeps):
            self.integrate(a, endpoint_load, joint_load)


    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self._np_random, self.seed = seeding.np_random(seed)
        options = {} if options is None else options
        batch_size: int = int(options.get("batch_size", 1))
        joint_state = options.get("joint_state", None)

        if joint_state is not None:
            j = np.asarray(joint_state, dtype=float)
            if j.ndim == 2 and j.shape[0] > 1:
                batch_size = j.shape[0]

        joint0 = self._parse_initial_joint_state(
            joint_state=joint_state, batch_size=batch_size
        )
        geometry0 = self.get_geometry(joint0)
        muscle0 = self.muscle.get_initial_muscle_state(
            batch_size=batch_size, geometry_state=geometry0
        )
        states = {"joint": joint0, "muscle": muscle0, "geometry": geometry0}
        self._set_state(states)

    # ---------------
    # RNG
    # ---------------

    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self._np_random, _ = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, rng: np.random.Generator):
        self._np_random = rng

    # ---------------
    # Muscle wrapping
    # ---------------

    def add_muscle(
        self,
        path_fixation_body: List[int],
        path_coordinates: List[List[float]],
        name: str = None,
        **kwargs,
    ):
        """
        Add a muscle with a set of fixation points.
        - path_fixation_body: list of body ids per point (0: world, 1: link1, 2: link2)
        - path_coordinates: list of [x,y] (or [x,y,z] if 3D) per point, in the local frame of the selected body.
        """
        pfb = np.asarray(path_fixation_body, dtype=np.float32).reshape(1, 1, -1)
        n_points = pfb.size
        pcs = np.asarray(path_coordinates, dtype=np.float32).T[np.newaxis, :, :]
        assert pcs.shape[1] == self.skeleton.space_dim
        assert pcs.shape[2] == n_points

        self.n_muscles += 1
        self.input_dim += self.muscle.input_dim

        # accumulate
        self._path_fixation_body = np.concatenate(
            [self._path_fixation_body, pfb], axis=-1
        )
        self._path_coordinates = np.concatenate([self._path_coordinates, pcs], axis=-1)
        self._muscle_index = np.hstack(
            [
                self._muscle_index,
                np.full(n_points, float(self.n_muscles), dtype=np.float32),
            ]
        )

        # transitions (difference == 1 means new muscle starts at that next point)
        diff = np.diff(self._muscle_index.reshape(1, 1, -1)) == 1.0
        self._muscle_transitions = diff.astype(bool)

        # ragged splits
        n_total_points = int(len(self._muscle_index))
        split_indices = np.where(np.diff(self._muscle_index) != 0)[0] + 1
        self._row_splits = np.concatenate(
            [np.array([0]), split_indices, np.array([n_total_points])]
        ).astype(np.int64)

        # publish NumPy views
        self.path_fixation_body = self._path_fixation_body
        self.path_coordinates = self._path_coordinates
        self.muscle_index = self._muscle_index
        self.muscle_transitions = self._muscle_transitions
        self.row_splits = self._row_splits
        self.section_splits = np.diff(self._row_splits).astype(int).tolist()

        # collect build kwargs for the muscle
        for key, val in kwargs.items():
            if key in self.tobuild__muscle:
                self.tobuild__muscle[key].append(val)
        for key, val in self.tobuild__muscle.items():
            if len(val) < self.n_muscles:
                if key in self.tobuild__default:
                    self.tobuild__muscle[key].append(self.tobuild__default[key])
                else:
                    raise TypeError(f"Missing keyword argument {key}.")

        # (re)build muscle with accumulated config
        self.muscle.build(timestep=self.minidt, **self.tobuild__muscle)

        # name
        name = name if name is not None else f"muscle_{self.n_muscles}"
        self.muscle_name.append(name)
        self._muscle_config_is_empty = False

    def get_muscle_cfg(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        for m in range(self.n_muscles):
            ix = np.where(self._muscle_index == (m + 1))[0]
            d = {
                "n_fixation_points": len(ix),
                "fixation body": [
                    int(k) for k in self._path_fixation_body.squeeze()[ix].tolist()
                ],
                "coordinates": [
                    self._path_coordinates.squeeze()[:, k].tolist() for k in ix
                ],
            }
            if not self._muscle_config_is_empty:
                for param, value in self.tobuild__muscle.items():
                    d[param] = value[m]
            cfg[self.muscle_name[m]] = d
        if not cfg:
            cfg = {
                "Placeholder Message": "No muscles were added using the `add_muscle` method."
            }
        return cfg

    def print_muscle_wrappings(self):
        cfg = self.get_muscle_cfg()
        if self._muscle_config_is_empty:
            print(cfg)
            return
        for muscle, params in cfg.items():
            print("MUSCLE NAME:", muscle)
            print("-" * (13 + len(muscle)))
            for key, param in params.items():
                print(key + ": ", param)
            print()

    # ---------------
    # Geometry
    # ---------------

    def get_geometry(self, joint_state: np.ndarray) -> np.ndarray:
        return self._get_geometry(joint_state)

    def _get_geometry(self, joint_state: np.ndarray) -> np.ndarray:
        # skeleton.path2cartesian returns (xy, dxy_dt, dxy_ddof)
        xy, dxy_dt, dxy_ddof = self.skeleton.path2cartesian(
            self.path_coordinates, self.path_fixation_body, joint_state
        )

        diff_pos = xy[:, :, 1:] - xy[:, :, :-1]  # (B,2,N-1)
        diff_vel = dxy_dt[:, :, 1:] - dxy_dt[:, :, :-1]  # (B,2,N-1)
        diff_ddof = dxy_ddof[:, :, :, 1:] - dxy_ddof[:, :, :, :-1]  # (B,2,DOF,N-1)

        # segment metrics
        seg_len = np.sqrt(np.sum(diff_pos**2, axis=1, keepdims=True))  # (B,1,N-1)
        # avoid divide by zero
        safe_len = np.where(seg_len == 0.0, 1.0, seg_len)
        seg_vel = np.sum(
            diff_pos * diff_vel / safe_len, axis=1, keepdims=True
        )  # (B,1,N-1)
        seg_mom = (
            np.sum(diff_ddof * diff_pos[:, :, None, :], axis=1) / safe_len
        )  # (B,DOF,N-1)

        # zero out transitions between muscles
        if self.muscle_transitions is not None:
            mt = self.muscle_transitions.astype(bool)  # (1,1,N-1)
            seg_len = np.where(mt, 0.0, seg_len)
            seg_vel = np.where(mt, 0.0, seg_vel)
            seg_mom = np.where(mt, 0.0, seg_mom)

        # collapse per muscle via ragged splits
        pieces_len = np.split(
            seg_len, indices_or_sections=np.cumsum(self.section_splits[:-1]), axis=-1
        )
        pieces_vel = np.split(
            seg_vel, indices_or_sections=np.cumsum(self.section_splits[:-1]), axis=-1
        )
        pieces_mom = np.split(
            seg_mom, indices_or_sections=np.cumsum(self.section_splits[:-1]), axis=-1
        )

        musculotendon_len = np.stack(
            [np.sum(p, axis=-1) for p in pieces_len], axis=-1
        )  # (B,1,M)
        musculotendon_vel = np.stack(
            [np.sum(p, axis=-1) for p in pieces_vel], axis=-1
        )  # (B,1,M)
        moment_arms = np.stack(
            [np.sum(p, axis=-1) for p in pieces_mom], axis=-1
        )  # (B,DOF,M)

        geometry_state = np.concatenate(
            [musculotendon_len, musculotendon_vel, moment_arms], axis=1
        )
        return geometry_state

    # ---------------
    # State plumbing
    # ---------------

    def _set_state(self, states: Dict[str, np.ndarray]):
        for k, v in states.items():
            self.states[k] = v
        self.states["cartesian"] = self.joint2cartesian(joint_state=states["joint"])
        self.states["fingertip"] = self.states["cartesian"][:, :2]  # (B,2)

    def integrate(
        self, action: np.ndarray, endpoint_load: np.ndarray, joint_load: np.ndarray
    ):
        self._integrate(action, endpoint_load, joint_load)

    def _euler(self, action, endpoint_load, joint_load):
        s0 = self.states
        k1 = self.ode(action, s0, endpoint_load, joint_load)
        s1 = self.integration_step(self.minidt, state_derivative=k1, states=s0)
        self._set_state(s1)

    def _rungekutta4(self, action, endpoint_load, joint_load):
        s0 = self.states
        k1 = self.ode(action, s0, endpoint_load, joint_load)
        s = self.integration_step(self.half_minidt, state_derivative=k1, states=s0)
        k2 = self.ode(action, s, endpoint_load, joint_load)
        s = self.integration_step(self.half_minidt, state_derivative=k2, states=s)
        k3 = self.ode(action, s, endpoint_load, joint_load)
        s = self.integration_step(self.minidt, state_derivative=k3, states=s)
        k4 = self.ode(action, s, endpoint_load, joint_load)
        k = {
            key: (k1[key] + 2 * (k2[key] + k3[key]) + k4[key]) / 6.0
            for key in k1.keys()
        }
        s_out = self.integration_step(self.minidt, state_derivative=k, states=s0)
        self._set_state(s_out)

    def integration_step(
        self,
        dt: float,
        state_derivative: Dict[str, np.ndarray],
        states: Dict[str, np.ndarray],
    ):
        new_muscle = self.muscle.integrate(
            dt, state_derivative["muscle"], states["muscle"], states["geometry"]
        )
        new_joint = self.skeleton.integrate(
            dt, state_derivative["joint"], states["joint"]
        )
        new_geometry = self.get_geometry(new_joint)
        return {"muscle": new_muscle, "joint": new_joint, "geometry": new_geometry}

    def ode(
        self,
        action: np.ndarray,
        states: Dict[str, np.ndarray],
        endpoint_load: np.ndarray,
        joint_load: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        moments = states["geometry"][:, 2:, :]  # (B,DOF,M)
        forces = states["muscle"][
            :, self.force_index : self.force_index + 1, :
        ]  # (B,1,M)
        joint_vel = states["joint"][:, self.dof :]  # (B,DOF)

        # τ = -Σ_i F_i * r_i + joint_load - damping * qdot
        generalized_forces = (
            -np.sum(forces * moments, axis=-1) + joint_load - self.damping * joint_vel
        )

        return {
            "muscle": self.muscle.ode(action, states["muscle"]),
            "joint": self.skeleton.ode(
                generalized_forces, states["joint"], endpoint_load=endpoint_load
            ),
        }

    # ---------------
    # State init helpers
    # ---------------

    def draw_random_uniform_states(self, batch_size: int) -> np.ndarray:
        sz = (batch_size, self.dof)
        rnd = self.np_random.uniform(size=sz)
        pos = (
            self.pos_lower_bound[None, :]
            + (self.pos_upper_bound - self.pos_lower_bound)[None, :] * rnd
        )
        vel = np.zeros(sz, dtype=float)
        return np.concatenate([pos, vel], axis=1)

    def _parse_initial_joint_state(self, joint_state, batch_size: int) -> np.ndarray:
        if joint_state is None:
            return self.draw_random_uniform_states(batch_size=batch_size)

        j = np.asarray(joint_state, dtype=float)
        if j.ndim == 1:
            j = j.reshape(1, -1)

        if j.shape[1] == self.state_dim:
            q = j[:, : self.dof]
            qd = j[:, self.dof :]
            return self.draw_fixed_states(
                position=q, velocity=qd, batch_size=batch_size
            )
        elif j.shape[1] == self.dof:
            return self.draw_fixed_states(position=j, batch_size=batch_size)
        else:
            raise ValueError("Initial joint_state has wrong width.")

    def draw_fixed_states(
        self, position: np.ndarray, batch_size: int, velocity: np.ndarray = None
    ) -> np.ndarray:
        pos = np.asarray(position, dtype=float)
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        vel = (
            np.zeros_like(pos)
            if velocity is None
            else np.asarray(velocity, dtype=float).reshape(pos.shape)
        )

        assert pos.shape == vel.shape == (pos.shape[0], self.dof)

        assert np.all(pos >= self.pos_lower_bound)
        assert np.all(pos <= self.pos_upper_bound)
        assert np.all(vel >= self.vel_lower_bound)
        assert np.all(vel <= self.vel_upper_bound)

        s = np.concatenate([pos, vel], axis=1)
        if batch_size == 1 and s.shape[0] == 1:
            return s
        return np.repeat(s, repeats=batch_size, axis=0)

    def _set_state_limit_bounds(self, lb, ub):
        lb = np.asarray(lb, dtype=np.float32).reshape(-1, 1)
        ub = np.asarray(ub, dtype=np.float32).reshape(-1, 1)
        bounds = np.hstack([lb, ub])
        if bounds.shape[0] != self.dof:
            bounds = np.tile(bounds[0:1, :], (self.dof, 1))
        return bounds

    # ---------------
    # Thin proxies
    # ---------------

    def joint2cartesian(self, joint_state: np.ndarray) -> np.ndarray:
        return self.skeleton.joint2cartesian(joint_state=joint_state)

    def setattr(self, name: str, value):
        setattr(self, name, value)

    def get_save_config(self) -> Dict[str, Any]:
        return {
            "muscle": self.muscle.get_save_config(),
            "skeleton": self.skeleton.get_save_config(),
            "dt": self.dt,
            "n_ministeps": self.n_ministeps,
            "minidt": self.minidt,
            "half_minidt": self.half_minidt,
            "muscle_names": self.muscle_name,
            "n_muscles": self.n_muscles,
            "muscle_wrapping_cfg": self.get_muscle_cfg(),
        }


# -----------------------
# RigidTendonArm26 (NumPy)
# -----------------------


class RigidTendonArm26(Effector):
    """
    Lumped 6-muscle model with polynomial moment arm geometry (Nijhof & Kouwenhoven 2000).

    Defaults to TwoDofArm skeleton if none provided.
    """

    def __init__(
        self, muscle, skeleton=None, timestep=0.01, muscle_kwargs: dict = {}, **kwargs
    ):
        sho_limit = np.deg2rad([0, 135])
        elb_limit = np.deg2rad([0, 155])
        pos_lower_bound = kwargs.pop("pos_lower_bound", [sho_limit[0], elb_limit[0]])
        pos_upper_bound = kwargs.pop("pos_upper_bound", [sho_limit[1], elb_limit[1]])

        if skeleton is None:
            skeleton = TwoDofArm(
                m1=1.82,
                m2=1.43,
                l1g=0.135,
                l2g=0.165,
                i1=0.051,
                i2=0.057,
                l1=0.309,
                l2=0.333,
            )

        super().__init__(
            skeleton=skeleton,
            muscle=muscle,
            timestep=timestep,
            pos_lower_bound=pos_lower_bound,
            pos_upper_bound=pos_upper_bound,
            **kwargs,
        )

        self.muscle_state_dim = self.muscle.state_dim
        self.geometry_state_dim = 2 + self.skeleton.dof
        self.n_muscles = 6
        self.input_dim = self.n_muscles
        self.muscle_name = [
            "pectoralis",
            "deltoid",
            "brachioradialis",
            "tricepslat",
            "biceps",
            "tricepslong",
        ]

        # collect/merge kwargs for the muscle
        self._merge_muscle_kwargs(muscle_kwargs)

        # hard-coded params (as in your original)
        self.tobuild__muscle["max_isometric_force"] = [838, 1207, 1422, 1549, 414, 603]
        self.tobuild__muscle["tendon_length"] = [
            0.039,
            0.066,
            0.172,
            0.187,
            0.204,
            0.217,
        ]
        self.tobuild__muscle["optimal_muscle_length"] = [
            0.134,
            0.140,
            0.092,
            0.093,
            0.137,
            0.127,
        ]
        self.muscle.build(timestep=self.dt, **self.tobuild__muscle)

        # Polynomial geometry constants (NumPy arrays)
        a0 = [0.151, 0.2322, 0.2859, 0.2355, 0.3329, 0.2989]
        a1 = [
            -0.03,
            0.03,
            0,
            0,
            -0.03,
            0.03,
            0,
            0,
            -0.014,
            0.025,
            -0.016,
            0.03,
        ]  # shape (2,6)
        a2 = [0, 0, 0, 0, 0, 0, 0, 0, -4e-3, -2.2e-3, -5.7e-3, -3.2e-3]  # shape (2,6)
        a3 = [np.pi / 2, 0.0]

        self.a0 = np.asarray(a0, dtype=np.float32).reshape(1, 1, 6)
        self.a1 = np.asarray(a1, dtype=np.float32).reshape(1, 2, 6)
        self.a2 = np.asarray(a2, dtype=np.float32).reshape(1, 2, 6)
        self.a3 = np.asarray(a3, dtype=np.float32).reshape(1, 2, 1)

    def _merge_muscle_kwargs(self, muscle_kwargs: dict):
        # merge into self.tobuild__muscle respecting defaults
        for key, val in muscle_kwargs.items():
            if key in self.tobuild__muscle.keys():
                self.tobuild__muscle[key].append(val)
            else:
                raise KeyError(f'Unexpected key "{key}" in muscle_kwargs.')
        for key, val in self.tobuild__muscle.items():
            if len(val) == 0 and key in self.tobuild__default:
                self.tobuild__muscle[key].append(self.tobuild__default[key])

    def _get_geometry(self, joint_state: np.ndarray) -> np.ndarray:
        # custom polynomial geometry (like your original)
        j = _as_batch(joint_state, self.state_dim)
        q, qd = np.split(j, 2, axis=1)  # (B,2)

        old_pos = q[:, :, None] - self.a3  # (B,2,1)
        moment_arm = old_pos * self.a2 * 2 + self.a1  # (B,2,6)

        musculotendon_len = (
            np.sum((self.a1 + old_pos * self.a2) * old_pos, axis=1, keepdims=True)
            + self.a0
        )  # (B,1,6)
        musculotendon_vel = np.sum(
            qd[:, :, None] * moment_arm, axis=1, keepdims=True
        )  # (B,1,6)

        return np.concatenate(
            [musculotendon_len, musculotendon_vel, moment_arm], axis=1
        )


# -----------------------
# CompliantTendonArm26 (NumPy)
# -----------------------


class CompliantTendonArm26(RigidTendonArm26):
    """
    Compliant-tendon version of the lumped 6-muscle model (RK4 default).
    """

    def __init__(
        self, timestep=0.0002, skeleton=None, muscle_kwargs: dict = {}, **kwargs
    ):
        integration_method = kwargs.pop("integration_method", "rk4")
        if skeleton is None:
            skeleton = TwoDofArm(
                m1=1.82,
                m2=1.43,
                l1g=0.135,
                l2g=0.165,
                i1=0.051,
                i2=0.057,
                l1=0.309,
                l2=0.333,
            )
        super().__init__(
            muscle=CompliantTendonHillMuscle(),
            skeleton=skeleton,
            timestep=timestep,
            muscle_kwargs=muscle_kwargs,
            integration_method=integration_method,
            **kwargs,
        )
        # adjust a0 per your original note (relax stiffness)
        a0 = [0.182, 0.2362, 0.2859, 0.2355, 0.3329, 0.2989]
        self.a0 = np.asarray(a0, dtype=np.float32).reshape(1, 1, 6)


# ---------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # === Basic Effector with TwoDofArm + ReluMuscle and 2 muscles via points ===
    arm = TwoDofArm()
    relu = ReluMuscle()
    eff = Effector(
        skeleton=arm,
        muscle=relu,
        timestep=0.002,
        integration_method="euler",
        damping=0.0,
    )

    # add two simple “string” muscles: one with 2 points on link1, one with 2 points on link2
    eff.add_muscle(
        path_fixation_body=[1, 1],
        path_coordinates=[[0.0, 0.05], [0.0, 0.00]],  # p0 (x along link), p1
        name="m1",
        max_isometric_force=100.0,
    )
    eff.add_muscle(
        path_fixation_body=[2, 2],
        path_coordinates=[[0.0, 0.05], [0.0, 0.00]],
        name="m2",
        max_isometric_force=120.0,
    )

    # initial joint state
    state0 = np.array([[np.deg2rad(30.0), np.deg2rad(45.0), 0.1, -0.2]])
    eff.reset(options={"joint_state": state0})

    # action and loads
    action = np.array([[0.2, 0.3]])[:, None, :]  # (B,1,M) -> here B=1, M=2
    endpoint_load = np.zeros((1, eff.space_dim))
    joint_load = np.zeros((1, eff.dof))

    # single step
    eff.step(action, endpoint_load=endpoint_load, joint_load=joint_load)

    print("Joint state:", eff.states["joint"])
    print(
        "Muscle state shape:",
        eff.states["muscle"].shape,
        "| channels:",
        eff.muscle.state_name,
    )
    print(
        "Geometry state shape:",
        eff.states["geometry"].shape,
        "| channels:",
        eff.geometry_state_name,
    )
    print("Fingertip (x,y):", eff.states["fingertip"])

    # === Lumped RigidTendonArm26 with ReluMuscle ===
    lumped = RigidTendonArm26(muscle=ReluMuscle(), timestep=0.005)
    lumped.reset(options={"joint_state": state0})
    lumped.step(
        np.ones((1, 1, 6)) * 0.3,
        endpoint_load=np.zeros((1, 2)),
        joint_load=np.zeros((1, 2)),
    )
    print("\n[RigidTendonArm26] joint:", lumped.states["joint"])
    print("muscle state shape:", lumped.states["muscle"].shape)

    # === CompliantTendonArm26 (RK4) ===
    comp = CompliantTendonArm26(timestep=0.0005)
    comp.reset(options={"joint_state": state0})
    comp.step(
        np.ones((1, 1, 6)) * 0.3,
        endpoint_load=np.zeros((1, 2)),
        joint_load=np.zeros((1, 2)),
    )
    print("\n[CompliantTendonArm26] joint:", comp.states["joint"])
    print("muscle state shape:", comp.states["muscle"].shape)

    print("\nSmoke tests complete ✓")
