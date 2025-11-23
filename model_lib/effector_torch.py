# effector_torch.py
# -*- coding: utf-8 -*-
"""
Pure-PyTorch Effector stack.

Torch counterpart of `effector_numpy.py`:
- Effector (base, path-based geometry)
- RigidTendonArm26 (polynomial geometry, 6 muscles)
- CompliantTendonArm26 (compliant tendon, RK4 default)

All math is pure Torch:
- Batchable on leading dimension.
- GPU-safe via device handling.
- Fully differentiable (no .detach(), no .numpy()).

Requires:
- model_lib.skeleton_torch.TwoDofArm
- model_lib.muscles_torch.{ReluMuscle, CompliantTendonHillMuscle}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Optional

import math
import torch
from torch import Tensor

from model_lib.muscles_torch import ReluMuscle, CompliantTendonHillMuscle
from model_lib.skeleton_torch import TwoDofArm


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _clip(
    x: Tensor,
    lo: Optional[Union[float, Tensor]] = None,
    hi: Optional[Union[float, Tensor]] = None,
) -> Tensor:
    """Torch equivalent of the NumPy clip helper."""
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


def _as_batch(
    x: Any, width: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """
    Ensure x is a 2D tensor (B, width).

    - If 1D: reshape to (1,width).
    - If 2D: keep, but check width.
    """
    t = torch.as_tensor(x, dtype=dtype, device=device)
    if t.dim() == 1:
        t = t.view(1, -1)
    if t.dim() != 2 or t.shape[1] != width:
        raise ValueError(f"Expected shape (*,{width}), got {tuple(t.shape)}")
    return t


# ---------------------------------------------------------------------
# Effector (Torch)
# ---------------------------------------------------------------------


class Effector:
    """
    Base class for Effectors (Torch only).

    Args:
      skeleton: Torch-based Skeleton (TwoDofArm or similar)
                with .ode(), .integrate(), .path2cartesian(), .joint2cartesian()
      muscle:   Torch-based Muscle with .ode(), .integrate()
      name:     descriptive name
      n_ministeps: number of mini-integration steps per dt
      timestep: main dt (seconds)
      integration_method: 'euler' or one of
            {'rk4','rungekutta4','runge-kutta4','runge-kutta-4'}
      damping: scalar viscous damping on joints (τ = -damping * q̇)
      pos/vel bounds: optional overrides for skeleton limits

    All tensors are created on `device` with given `dtype`.
    """

    def __init__(
        self,
        skeleton,
        muscle,
        name: str = "EffectorTorch",
        n_ministeps: int = 1,
        timestep: float = 0.01,
        integration_method: str = "euler",
        damping: float = 0.0,
        pos_lower_bound: Union[float, List[float], Tuple[float, ...]] = None,
        pos_upper_bound: Union[float, List[float], Tuple[float, ...]] = None,
        vel_lower_bound: Union[float, List[float], Tuple[float, ...]] = None,
        vel_upper_bound: Union[float, List[float], Tuple[float, ...]] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
    ):
        self.__name__ = name
        self.skeleton = skeleton
        self.muscle = muscle

        self.device = torch.device(device)
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

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

        # Torch RNG
        self._torch_generator: Optional[torch.Generator] = None
        self.seed: Optional[int] = None

        # Bounds
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

        # Skeleton gets dt + bounds (Torch version of build)
        self.skeleton.build(
            timestep=self.dt,
            pos_upper_bound=self.pos_upper_bound,
            pos_lower_bound=self.pos_lower_bound,
            vel_upper_bound=self.vel_upper_bound,
            vel_lower_bound=self.vel_lower_bound,
        )

        # muscle “API” descriptors
        self.force_index = self.muscle.state_name.index("force")
        self.MusclePaths: List = []
        self.n_muscles: int = 0
        self.input_dim: int = 0
        self.muscle_name: List[str] = []
        self.muscle_state_dim: int = self.muscle.state_dim
        self.geometry_state_dim: int = 2 + self.dof
        self.geometry_state_name = [
            "musculotendon length",
            "musculotendon velocity",
        ] + [f"moment for joint {d}" for d in range(self.dof)]

        # muscle build accumulators
        self.tobuild__muscle = dict(self.muscle.to_build_dict)
        self.tobuild__default = dict(self.muscle.to_build_dict_default)

        # muscle wrapping accumulators (Torch)
        # path_fixation_body is (1, N_points) so skeleton_torch.path2cartesian
        # sees a 2D array (B, N_points).
        self._path_fixation_body = torch.empty(
            (1, 0), dtype=torch.long, device=self.device
        )
        self._path_coordinates = torch.empty(
            (1, self.skeleton.space_dim, 0), dtype=self.dtype, device=self.device
        )
        self._muscle_index = torch.empty(0, dtype=torch.float32, device=self.device)
        self.muscle_transitions: Optional[Tensor] = None  # (1,1,N-1) bool
        self.row_splits: Optional[Tensor] = None  # point splits
        self.section_splits: Optional[List[int]] = None
        # final views
        self.path_fixation_body: Optional[Tensor] = None
        self.path_coordinates: Optional[Tensor] = None
        self.muscle_index: Optional[Tensor] = None
        self._muscle_config_is_empty: bool = True

        # default loads (B=1, broadcastable)
        self.default_endpoint_load = torch.zeros(
            (1, self.skeleton.space_dim), dtype=self.dtype, device=self.device
        )
        self.default_joint_load = torch.zeros(
            (1, self.skeleton.dof), dtype=self.dtype, device=self.device
        )

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
        self.states: Dict[str, Optional[Tensor]] = {
            k: None for k in ["joint", "cartesian", "muscle", "geometry", "fingertip"]
        }

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def step(self, action, **kwargs):
        """
        action: excitations u.
          Accepts (B, M) or (B, 1, M). Internally we use (B,1,M).
        """
        endpoint_load = kwargs.get("endpoint_load", self.default_endpoint_load)
        joint_load = kwargs.get("joint_load", self.default_joint_load)

        action_t = torch.as_tensor(action, dtype=self.dtype, device=self.device)
        if action_t.dim() == 2:
            # (B,M) -> (B,1,M)
            action_t = action_t.unsqueeze(1)
        elif action_t.dim() != 3:
            raise ValueError(
                f"action must be (B,M) or (B,1,M); got shape {tuple(action_t.shape)}"
            )

        # clamp *excitations* (behavior same as clip_activation)
        a = self.muscle.clip_excitation(action_t)

        endpoint_load_t = torch.as_tensor(
            endpoint_load, dtype=self.dtype, device=self.device
        )
        joint_load_t = torch.as_tensor(
            joint_load, dtype=self.dtype, device=self.device
        )

        for _ in range(self.n_ministeps):
            self.integrate(a, endpoint_load_t, joint_load_t)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Reset internal states.

        options:
          - batch_size: int
          - joint_state: (B, state_dim) or (B, dof) or (state_dim,) or (dof,)
        """
        if seed is not None:
            self._torch_generator = torch.Generator(device=self.device)
            self._torch_generator.manual_seed(int(seed))
            self.seed = int(seed)
        options = {} if options is None else options
        batch_size: int = int(options.get("batch_size", 1))
        joint_state = options.get("joint_state", None)

        if joint_state is not None:
            j = torch.as_tensor(joint_state, dtype=self.dtype, device=self.device)
            if j.dim() == 2 and j.shape[0] > 1:
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

    # ------------------------------------------------------------------
    # RNG helpers
    # ------------------------------------------------------------------

    @property
    def torch_generator(self) -> torch.Generator:
        if self._torch_generator is None:
            self._torch_generator = torch.Generator(device=self.device)
        return self._torch_generator

    @torch_generator.setter
    def torch_generator(self, gen: torch.Generator):
        self._torch_generator = gen

    # ------------------------------------------------------------------
    # Muscle wrapping
    # ------------------------------------------------------------------

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
        - path_coordinates: list of [x,y] (or [x,y,z]) per point, in the local frame
                            of the selected body.
        """
        # shape (1, n_points) so skeleton_torch.path2cartesian sees a 2D array
        pfb = torch.as_tensor(
            path_fixation_body, dtype=torch.long, device=self.device
        ).view(1, -1)
        n_points = pfb.shape[-1]

        # local coordinates -> (1, space_dim, n_points)
        pcs = torch.as_tensor(
            path_coordinates, dtype=self.dtype, device=self.device
        ).T.unsqueeze(0)
        assert pcs.shape[1] == self.skeleton.space_dim
        assert pcs.shape[2] == n_points

        self.n_muscles += 1
        self.input_dim += self.muscle.input_dim

        # accumulate along the points dimension
        self._path_fixation_body = torch.cat(
            [self._path_fixation_body, pfb], dim=1
        )  # (1, N_total)
        self._path_coordinates = torch.cat(
            [self._path_coordinates, pcs], dim=-1
        )  # (1, space_dim, N_total)
        self._muscle_index = torch.cat(
            [
                self._muscle_index,
                torch.full(
                    (n_points,),
                    float(self.n_muscles),
                    dtype=torch.float32,
                    device=self.device,
                ),
            ],
            dim=0,
        )

        # transitions (difference == 1 -> new muscle starts at that next point)
        diff = torch.diff(
            self._muscle_index.view(1, 1, -1), dim=-1
        ) == 1.0  # (1,1,N-1)
        self.muscle_transitions = diff.to(dtype=torch.bool)

        # ragged splits over points
        n_total_points = int(self._muscle_index.numel())
        diff_idx = torch.nonzero(
            torch.diff(self._muscle_index) != 0, as_tuple=False
        ).flatten()
        split_indices = diff_idx + 1  # point indices
        row_splits = torch.cat(
            [
                torch.tensor([0], device=self.device, dtype=torch.long),
                split_indices.to(dtype=torch.long),
                torch.tensor([n_total_points], device=self.device, dtype=torch.long),
            ],
            dim=0,
        )
        self.row_splits = row_splits
        self.section_splits = (
            row_splits[1:] - row_splits[:-1]
        ).tolist()  # points per muscle

        # publish views
        self.path_fixation_body = self._path_fixation_body
        self.path_coordinates = self._path_coordinates
        self.muscle_index = self._muscle_index
        self._muscle_config_is_empty = False

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

    def get_muscle_cfg(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        if self._muscle_config_is_empty:
            return {
                "Placeholder Message": "No muscles were added using the `add_muscle` method."
            }
        idx_cpu = self._muscle_index.detach().cpu().numpy()
        pfb_cpu = self._path_fixation_body.detach().cpu().numpy().squeeze()
        pcs_cpu = self._path_coordinates.detach().cpu().numpy().squeeze()
        for m in range(self.n_muscles):
            ix = (idx_cpu == float(m + 1)).nonzero()[0]
            d = {
                "n_fixation_points": len(ix),
                "fixation body": [int(k) for k in pfb_cpu[ix].tolist()],
                "coordinates": [pcs_cpu[:, k].tolist() for k in ix],
            }
            for param, value in self.tobuild__muscle.items():
                d[param] = value[m]
            cfg[self.muscle_name[m]] = d
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

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def get_geometry(self, joint_state: Tensor) -> Tensor:
        return self._get_geometry(joint_state)

    def _get_geometry(self, joint_state: Tensor) -> Tensor:
        """
        Path-based musculotendon geometry using skeleton.path2cartesian.

        geometry_state: (B, 2 + DOF, M)
          [:,0,:]      = musculotendon length
          [:,1,:]      = musculotendon velocity
          [:,2:2+DOF,:]= moment arms (per joint) = ∂ℓ/∂q
        """
        if self._muscle_config_is_empty:
            raise RuntimeError("No muscles added; cannot compute geometry.")

        j = _as_batch(joint_state, self.state_dim, device=self.device, dtype=self.dtype)

        xy, dxy_dt, dxy_ddof = self.skeleton.path2cartesian(
            self.path_coordinates, self.path_fixation_body, j
        )
        # xy:      (B,2,N)
        # dxy_dt:  (B,2,N)
        # dxy_ddof:(B,2,DOF,N)

        diff_pos = xy[:, :, 1:] - xy[:, :, :-1]  # (B,2,N-1)
        diff_vel = dxy_dt[:, :, 1:] - dxy_dt[:, :, :-1]  # (B,2,N-1)
        diff_ddof = dxy_ddof[:, :, :, 1:] - dxy_ddof[:, :, :, :-1]  # (B,2,DOF,N-1)

        seg_len = torch.sqrt(torch.sum(diff_pos**2, dim=1, keepdim=True))  # (B,1,N-1)
        safe_len = torch.where(seg_len == 0.0, torch.ones_like(seg_len), seg_len)
        seg_vel = torch.sum(
            diff_pos * diff_vel / safe_len, dim=1, keepdim=True
        )  # (B,1,N-1)
        seg_mom = (
            torch.sum(diff_ddof * diff_pos[:, :, None, :], dim=1) / safe_len
        )  # (B,DOF,N-1)

        if self.muscle_transitions is not None:
            mt = self.muscle_transitions  # (1,1,N-1) bool
            mt_len = mt.expand(seg_len.shape[0], 1, mt.shape[-1])
            seg_len = torch.where(mt_len, torch.zeros_like(seg_len), seg_len)
            seg_vel = torch.where(mt_len, torch.zeros_like(seg_vel), seg_vel)
            mt_mom = mt.expand(seg_mom.shape[0], seg_mom.shape[1], mt.shape[-1])
            seg_mom = torch.where(mt_mom, torch.zeros_like(seg_mom), seg_mom)

        if self.section_splits is None or len(self.section_splits) == 0:
            raise RuntimeError("section_splits is empty; add_muscle was not called.")

        # split by point-count cumsum (same trick as NumPy version)
        if len(self.section_splits) == 1:
            pieces_len = [seg_len]
            pieces_vel = [seg_vel]
            pieces_mom = [seg_mom]
        else:
            idx = torch.cumsum(
                torch.tensor(
                    self.section_splits[:-1], dtype=torch.long, device=seg_len.device
                ),
                dim=0,
            )
            idx_list = idx.tolist()
            pieces_len = torch.tensor_split(seg_len, idx_list, dim=-1)
            pieces_vel = torch.tensor_split(seg_vel, idx_list, dim=-1)
            pieces_mom = torch.tensor_split(seg_mom, idx_list, dim=-1)

        musculotendon_len = torch.stack(
            [p.sum(dim=-1) for p in pieces_len], dim=-1
        )  # (B,1,M)
        musculotendon_vel = torch.stack(
            [p.sum(dim=-1) for p in pieces_vel], dim=-1
        )  # (B,1,M)
        moment_arms = torch.stack(
            [p.sum(dim=-1) for p in pieces_mom], dim=-1
        )  # (B,DOF,M)

        geometry_state = torch.cat(
            [musculotendon_len, musculotendon_vel, moment_arms], dim=1
        )  # (B,2+DOF,M)
        return geometry_state

    # ------------------------------------------------------------------
    # State plumbing
    # ------------------------------------------------------------------

    def _set_state(self, states: Dict[str, Tensor]):
        for k, v in states.items():
            self.states[k] = v
        self.states["cartesian"] = self.joint2cartesian(joint_state=states["joint"])
        self.states["fingertip"] = self.states["cartesian"][:, :2]  # (B,2)

    def integrate(
        self, action: Tensor, endpoint_load: Tensor, joint_load: Tensor
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
        state_derivative: Dict[str, Tensor],
        states: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
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
        action: Tensor,
        states: Dict[str, Tensor],
        endpoint_load: Tensor,
        joint_load: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute time derivative of (muscle_state, joint_state).

        Moments r:  (B,DOF,M) = ∂ℓ/∂q
        Forces F:  (B,1,M)
        τ_muscles = -Σ_i F_i * r_i
        τ_total   = τ_muscles + joint_load - damping * q̇
        """
        moments = states["geometry"][:, 2:, :]  # (B,DOF,M)
        forces = states["muscle"][
            :, self.force_index : self.force_index + 1, :
        ]  # (B,1,M)
        joint_vel = states["joint"][:, self.dof :]  # (B,DOF)
        B = states["joint"].shape[0]

        endpoint_load = torch.as_tensor(
            endpoint_load, dtype=self.dtype, device=self.device
        )
        joint_load = torch.as_tensor(
            joint_load, dtype=self.dtype, device=self.device
        )

        if endpoint_load.dim() == 1:
            endpoint_load = endpoint_load.view(1, -1)
        if endpoint_load.shape[0] == 1 and B > 1:
            endpoint_load = endpoint_load.expand(B, -1)

        if joint_load.dim() == 1:
            joint_load = joint_load.view(1, -1)
        if joint_load.shape[0] == 1 and B > 1:
            joint_load = joint_load.expand(B, -1)

        generalized_forces = (
            -torch.sum(forces * moments, dim=-1)
            + joint_load
            - self.damping * joint_vel
        )

        return {
            "muscle": self.muscle.ode(action, states["muscle"]),
            "joint": self.skeleton.ode(
                generalized_forces, states["joint"], endpoint_load=endpoint_load
            ),
        }

    # ------------------------------------------------------------------
    # State init helpers
    # ------------------------------------------------------------------

    def draw_random_uniform_states(self, batch_size: int) -> Tensor:
        sz = (batch_size, self.dof)
        rnd = torch.rand(
            sz, generator=self.torch_generator, device=self.device, dtype=self.dtype
        )
        pos = self.pos_lower_bound.unsqueeze(0) + (
            self.pos_upper_bound - self.pos_lower_bound
        ).unsqueeze(0) * rnd
        vel = torch.zeros_like(pos)
        return torch.cat([pos, vel], dim=1)

    def _parse_initial_joint_state(
        self, joint_state, batch_size: int
    ) -> Tensor:
        if joint_state is None:
            return self.draw_random_uniform_states(batch_size=batch_size)

        j = torch.as_tensor(joint_state, dtype=self.dtype, device=self.device)
        if j.dim() == 1:
            j = j.view(1, -1)

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
        self, position: Tensor, batch_size: int, velocity: Optional[Tensor] = None
    ) -> Tensor:
        pos = torch.as_tensor(position, dtype=self.dtype, device=self.device)
        if pos.dim() == 1:
            pos = pos.view(1, -1)
        if velocity is None:
            vel = torch.zeros_like(pos)
        else:
            vel = torch.as_tensor(velocity, dtype=self.dtype, device=self.device)
            vel = vel.view_as(pos)

        assert pos.shape == vel.shape == (pos.shape[0], self.dof)

        assert torch.all(pos >= self.pos_lower_bound)
        assert torch.all(pos <= self.pos_upper_bound)
        assert torch.all(vel >= self.vel_lower_bound)
        assert torch.all(vel <= self.vel_upper_bound)

        s = torch.cat([pos, vel], dim=1)
        if batch_size == 1 and s.shape[0] == 1:
            return s
        return s.repeat(batch_size, 1)

    def _set_state_limit_bounds(self, lb, ub) -> Tensor:
        lb_t = torch.as_tensor(lb, dtype=self.dtype, device=self.device).view(-1, 1)
        ub_t = torch.as_tensor(ub, dtype=self.dtype, device=self.device).view(-1, 1)
        bounds = torch.cat([lb_t, ub_t], dim=1)  # (N,2)
        if bounds.shape[0] != self.dof:
            bounds = bounds[0:1, :].expand(self.dof, -1)
        return bounds

    # ------------------------------------------------------------------
    # Thin proxies
    # ------------------------------------------------------------------

    def joint2cartesian(self, joint_state: Tensor) -> Tensor:
        j = _as_batch(joint_state, self.state_dim, device=self.device, dtype=self.dtype)
        return self.skeleton.joint2cartesian(joint_state=j)

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


# ---------------------------------------------------------------------
# RigidTendonArm26 (Torch)
# ---------------------------------------------------------------------


class RigidTendonArm26(Effector):
    """
    Lumped 6-muscle model with polynomial moment arm geometry
    (Nijhof & Kouwenhoven 2000) – Torch version.

    Defaults to TwoDofArm skeleton if none provided.
    """

    def __init__(
        self,
        muscle,
        skeleton=None,
        timestep=0.01,
        muscle_kwargs: dict = {},
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        sho_limit = torch.deg2rad(torch.tensor([0.0, 135.0]))
        elb_limit = torch.deg2rad(torch.tensor([0.0, 155.0]))
        pos_lower_bound = kwargs.pop(
            "pos_lower_bound", [sho_limit[0].item(), elb_limit[0].item()]
        )
        pos_upper_bound = kwargs.pop(
            "pos_upper_bound", [sho_limit[1].item(), elb_limit[1].item()]
        )

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
                device=device,
                dtype=dtype,
            )

        super().__init__(
            skeleton=skeleton,
            muscle=muscle,
            timestep=timestep,
            pos_lower_bound=pos_lower_bound,
            pos_upper_bound=pos_upper_bound,
            device=device,
            dtype=dtype,
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

        # merge kwargs for the muscle
        self._merge_muscle_kwargs(muscle_kwargs)

        # hard-coded params (same as NumPy version)
        self.tobuild__muscle["max_isometric_force"] = [
            838,
            1207,
            1422,
            1549,
            414,
            603,
        ]
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

        # Polynomial geometry constants (Torch tensors)
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
        a2 = [0, 0, 0, 0, 0, 0, 0, 0, -4e-3, -2.2e-3, -5.7e-3, -3.2e-3]  # (2,6)
        a3 = [math.pi / 2, 0.0]

        self.a0 = torch.as_tensor(a0, dtype=self.dtype, device=self.device).view(
            1, 1, 6
        )
        self.a1 = torch.as_tensor(a1, dtype=self.dtype, device=self.device).view(
            1, 2, 6
        )
        self.a2 = torch.as_tensor(a2, dtype=self.dtype, device=self.device).view(
            1, 2, 6
        )
        self.a3 = torch.as_tensor(a3, dtype=self.dtype, device=self.device).view(
            1, 2, 1
        )

    def _merge_muscle_kwargs(self, muscle_kwargs: dict):
        for key, val in muscle_kwargs.items():
            if key in self.tobuild__muscle.keys():
                self.tobuild__muscle[key].append(val)
            else:
                raise KeyError(f'Unexpected key "{key}" in muscle_kwargs.')
        for key, val in self.tobuild__muscle.items():
            if len(val) == 0 and key in self.tobuild__default:
                self.tobuild__muscle[key].append(self.tobuild__default[key])

    def _get_geometry(self, joint_state: Tensor) -> Tensor:
        """
        Custom polynomial geometry (same as original NumPy RigidTendonArm26).
        """
        j = _as_batch(joint_state, self.state_dim, device=self.device, dtype=self.dtype)
        q, qd = torch.split(j, [self.dof, self.dof], dim=1)  # (B,2)

        old_pos = q.unsqueeze(-1) - self.a3  # (B,2,1)
        moment_arm = old_pos * self.a2 * 2.0 + self.a1  # (B,2,6)

        musculotendon_len = (
            torch.sum((self.a1 + old_pos * self.a2) * old_pos, dim=1, keepdim=True)
            + self.a0
        )  # (B,1,6)
        musculotendon_vel = torch.sum(
            qd.unsqueeze(-1) * moment_arm, dim=1, keepdim=True
        )  # (B,1,6)

        return torch.cat(
            [musculotendon_len, musculotendon_vel, moment_arm], dim=1
        )  # (B,2+DOF,6)


# ---------------------------------------------------------------------
# CompliantTendonArm26 (Torch)
# ---------------------------------------------------------------------


class CompliantTendonArm26(RigidTendonArm26):
    """
    Compliant-tendon version of the lumped 6-muscle model (RK4 default).
    """

    def __init__(
        self,
        timestep=0.0002,
        skeleton=None,
        muscle_kwargs: dict = {},
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
        **kwargs,
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
                device=device,
                dtype=dtype,
            )
        super().__init__(
            muscle=CompliantTendonHillMuscle(device=device, dtype=dtype),
            skeleton=skeleton,
            timestep=timestep,
            muscle_kwargs=muscle_kwargs,
            device=device,
            dtype=dtype,
            integration_method=integration_method,
            **kwargs,
        )
        # adjust a0 per original note (relax stiffness)
        a0 = [0.182, 0.2362, 0.2859, 0.2355, 0.3329, 0.2989]
        self.a0 = torch.as_tensor(a0, dtype=self.dtype, device=self.device).view(
            1, 1, 6
        )


# ---------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    print("[effector_torch] Simple smoke test...")

    # === Basic Effector with TwoDofArm + ReluMuscle and 2 muscles via points ===
    arm = TwoDofArm(device=device, dtype=torch.get_default_dtype())
    relu = ReluMuscle(device=device, dtype=torch.get_default_dtype())
    eff = Effector(
        skeleton=arm,
        muscle=relu,
        timestep=0.002,
        integration_method="euler",
        damping=0.0,
        device=device,
        dtype=torch.get_default_dtype(),
    )

    # add two simple “string” muscles: one with 2 points on link1, one with 2 on link2
    eff.add_muscle(
        path_fixation_body=[1, 1],
        path_coordinates=[[0.0, 0.05], [0.0, 0.00]],
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
    state0 = torch.tensor(
        [[math.radians(30.0), math.radians(45.0), 0.1, -0.2]],
        dtype=torch.get_default_dtype(),
        device=device,
    )
    eff.reset(options={"joint_state": state0})

    # action and loads
    action = torch.tensor([[0.2, 0.3]], dtype=torch.get_default_dtype(), device=device)[
        :, None, :
    ]  # (B,1,M)
    endpoint_load = torch.zeros(
        (1, eff.space_dim), dtype=torch.get_default_dtype(), device=device
    )
    joint_load = torch.zeros(
        (1, eff.dof), dtype=torch.get_default_dtype(), device=device
    )

    eff.step(action, endpoint_load=endpoint_load, joint_load=joint_load)

    print("  [Effector] Joint state:", eff.states["joint"])
    print(
        "  [Effector] Muscle state shape:",
        eff.states["muscle"].shape,
        "| channels:",
        eff.muscle.state_name,
    )
    print(
        "  [Effector] Geometry state shape:",
        eff.states["geometry"].shape,
        "| channels:",
        eff.geometry_state_name,
    )
    print("  [Effector] Fingertip (x,y):", eff.states["fingertip"])

    # === Lumped RigidTendonArm26 with ReluMuscle ===
    lumped = RigidTendonArm26(
        muscle=ReluMuscle(device=device, dtype=torch.get_default_dtype()),
        timestep=0.005,
        device=device,
        dtype=torch.get_default_dtype(),
    )
    lumped.reset(options={"joint_state": state0})
    lumped.step(
        torch.ones((1, 1, 6), dtype=torch.get_default_dtype(), device=device) * 0.3,
        endpoint_load=torch.zeros(
            (1, 2), dtype=torch.get_default_dtype(), device=device
        ),
        joint_load=torch.zeros((1, 2), dtype=torch.get_default_dtype(), device=device),
    )
    print("\n  [RigidTendonArm26] joint:", lumped.states["joint"])
    print("  [RigidTendonArm26] muscle state shape:", lumped.states["muscle"].shape)

    # === CompliantTendonArm26 (RK4) ===
    comp = CompliantTendonArm26(
        timestep=0.0005,
        device=device,
        dtype=torch.get_default_dtype(),
    )
    comp.reset(options={"joint_state": state0})
    comp.step(
        torch.ones((1, 1, 6), dtype=torch.get_default_dtype(), device=device) * 0.3,
        endpoint_load=torch.zeros(
            (1, 2), dtype=torch.get_default_dtype(), device=device
        ),
        joint_load=torch.zeros((1, 2), dtype=torch.get_default_dtype(), device=device),
    )
    print("\n  [CompliantTendonArm26] joint:", comp.states["joint"])
    print("  [CompliantTendonArm26] muscle state shape:", comp.states["muscle"].shape)

    print("\n[effector_torch] Smoke tests complete ✓")
