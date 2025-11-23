# skeleton_torch.py
# -*- coding: utf-8 -*-
"""
Two-DOF planar arm built purely on the Torch robotics stack (HTM formulation).

Torch counterpart of skeleton_numpy.TwoDofArm:
- Uses model_lib.Robot_torch.Serial for the model container
- Uses lib.kinematics.HTM_kinematics_torch.{forwardHTM, geometricJacobian}
- Uses lib.dynamics.DynamicsHTM_torch.{inertiaMatrixCOM, centrifugalCoriolisCOM,
  gravitationalCOM}

Differences vs skeleton_numpy.py:
- Pure PyTorch: no NumPy, no SymPy, no pickle/disk caching.
- Fully differentiable: all math stays in Torch (no .detach().cpu().numpy() in the core).
- Proper batching support:
    * joint_state : (B, 2*dof)
    * inputs (τ)  : (B, dof)
    * endpoint_load: (B, space_dim)
  or 1D versions which are upgraded to B=1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor

from  lib.Robot_torch import Serial as RobotSerial  # adjust import root if needed
import lib.kinematics.HTM_kinematics_torch as _kin
import lib.dynamics.DynamicsHTM_torch as _dyn


# ---------------------------------------------------------------------
# Base Skeleton (Torch-only)
# ---------------------------------------------------------------------


@dataclass
class Skeleton:
    dof: int
    space_dim: int
    name: str = "skeleton"
    pos_lower_bound: Union[float, Tuple[float, ...]] = -1.0
    pos_upper_bound: Union[float, Tuple[float, ...]] = +1.0
    vel_lower_bound: Union[float, Tuple[float, ...]] = -1000.0
    vel_upper_bound: Union[float, Tuple[float, ...]] = +1000.0
    dt: float = 0.001  # default time step
    input_dim: Optional[int] = None
    state_dim: Optional[int] = None
    output_dim: Optional[int] = None
    device: Union[str, torch.device] = "cpu"
    dtype: Optional[torch.dtype] = None

    def __post_init__(self):
        # Device / dtype
        self.device = torch.device(self.device)
        if self.dtype is None:
            self.dtype = torch.get_default_dtype()

        # Dimensionalities
        self.input_dim = self.input_dim or self.dof
        self.state_dim = self.state_dim or (2 * self.dof)
        self.output_dim = self.output_dim or self.state_dim

        # Normalization helper: broadcast scalars to length dof
        def _norm_bound(val, name: str) -> Tensor:
            t = torch.as_tensor(val, dtype=self.dtype, device=self.device)
            if t.numel() == 1:
                t = t.expand(self.dof)
            elif t.numel() != self.dof:
                raise ValueError(
                    f"{name} must have length {self.dof} or be scalar, "
                    f"got shape {tuple(t.shape)} (numel={t.numel()})"
                )
            return t.view(1, self.dof)

        self.pos_lower_bound = _norm_bound(self.pos_lower_bound, "pos_lower_bound")
        self.pos_upper_bound = _norm_bound(self.pos_upper_bound, "pos_upper_bound")
        self.vel_lower_bound = _norm_bound(self.vel_lower_bound, "vel_lower_bound")
        self.vel_upper_bound = _norm_bound(self.vel_upper_bound, "vel_upper_bound")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def build(
        self,
        timestep: float,
        pos_upper_bound,
        pos_lower_bound,
        vel_upper_bound,
        vel_lower_bound,
    ):
        """
        Configure timestep and joint/velocity bounds (Torch version of skeleton_numpy.Skeleton.build).
        """
        self.dt = float(timestep)

        def _norm_bound(val, name: str) -> Tensor:
            t = torch.as_tensor(val, dtype=self.dtype, device=self.device)
            if t.numel() == 1:
                t = t.expand(self.dof)
            elif t.numel() != self.dof:
                raise ValueError(
                    f"{name} must have length {self.dof} or be scalar, "
                    f"got shape {tuple(t.shape)} (numel={t.numel()})"
                )
            return t.view(1, self.dof)

        self.pos_upper_bound = _norm_bound(pos_upper_bound, "pos_upper_bound")
        self.pos_lower_bound = _norm_bound(pos_lower_bound, "pos_lower_bound")
        self.vel_upper_bound = _norm_bound(vel_upper_bound, "vel_upper_bound")
        self.vel_lower_bound = _norm_bound(vel_lower_bound, "vel_lower_bound")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _as_batch(self, x: Any, d: int) -> Tensor:
        """
        Ensure x is a 2D batch of width d: shape (B, d).
        """
        x_t = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if x_t.dim() == 1:
            x_t = x_t.view(1, -1)
        if x_t.dim() != 2 or x_t.shape[1] != d:
            raise ValueError(f"Expected shape (*, {d}), got {tuple(x_t.shape)}")
        return x_t

    @staticmethod
    def _clip(x: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
        """
        Clip x between lb and ub (broadcast on batch).
        """
        return torch.clamp(x, lb, ub)

    def clip_position(self, q: Tensor) -> Tensor:
        """
        Clip joint positions to [pos_lower_bound, pos_upper_bound].
        q: (B, dof)
        """
        return self._clip(q, self.pos_lower_bound, self.pos_upper_bound)

    def clip_velocity(self, pos: Tensor, vel: Tensor) -> Tensor:
        """
        Clip velocities and zero them when they try to push beyond joint limits.
        pos, vel: (B, dof)
        """
        vel = self._clip(vel, self.vel_lower_bound, self.vel_upper_bound)

        # If at/over lower bound and velocity < 0, zero it
        mask_low = (vel < 0.0) & (pos <= self.pos_lower_bound)
        # If at/over upper bound and velocity > 0, zero it
        mask_high = (vel > 0.0) & (pos >= self.pos_upper_bound)

        mask = mask_low | mask_high
        vel = torch.where(mask, torch.zeros_like(vel), vel)
        return vel

    # ------------------------------------------------------------------
    # Core API (to be overridden)
    # ------------------------------------------------------------------
    def ode(
        self, inputs: Tensor, joint_state: Tensor, endpoint_load: Tensor
    ) -> Tensor:
        raise NotImplementedError

    def integrate(
        self, dt: float, state_derivative: Tensor, joint_state: Tensor
    ) -> Tensor:
        """
        Semi-implicit Euler (matches original behavior):

            qd_{k+1} = qd_k + qdd * dt
            q_{k+1}  = q_k  + qd_k * dt
        """
        dt_t = torch.as_tensor(dt, dtype=self.dtype, device=self.device)

        # Ensure batch shapes
        joint_state = self._as_batch(joint_state, self.state_dim)
        qdd = self._as_batch(state_derivative, self.dof)

        q, qd = torch.split(joint_state, self.dof, dim=1)  # (B, dof), (B, dof)

        new_qd = qd + qdd * dt_t
        new_q = q + qd * dt_t

        new_qd = self.clip_velocity(new_q, new_qd)
        new_q = self.clip_position(new_q)
        return torch.cat([new_q, new_qd], dim=1)

    def joint2cartesian(self, joint_state: Tensor) -> Tensor:
        raise NotImplementedError

    def path2cartesian(
        self,
        path_coordinates: Tensor,     # (B, 2, n_points) or (2, n_points)
        path_fixation_body: Tensor,   # (1, n_points) with {0: world, 1: link1, 2: link2}
        joint_state: Tensor,          # (B, 2*dof)
    ):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Device / dtype management
    # ------------------------------------------------------------------
    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Move bounds and any internal Torch tensors to a new device/dtype.
        """
        if device is None:
            device = self.device
        device = torch.device(device)
        if dtype is None:
            dtype = self.dtype

        def _move(x):
            return x.to(device=device, dtype=dtype) if isinstance(x, Tensor) else x

        self.pos_lower_bound = _move(self.pos_lower_bound)
        self.pos_upper_bound = _move(self.pos_upper_bound)
        self.vel_lower_bound = _move(self.vel_lower_bound)
        self.vel_upper_bound = _move(self.vel_upper_bound)

        self.device = device
        self.dtype = dtype
        return self


# ---------------------------------------------------------------------
# Two-DOF planar arm (HTM-based, Torch)
# ---------------------------------------------------------------------


class TwoDofArm(Skeleton):
    """
    A two-DOF planar arm powered by the Torch robotics library.

    Dynamics:
        qdd = D(q)^{-1} [ τ + J(q)^T f - C(q,qd) qd - g(q) ]

    where:
        - D, C, g come from lib.dynamics.DynamicsHTM_torch
        - J is lib.kinematics.HTM_kinematics_torch.geometricJacobian (linear x-y rows)
    """

    def __init__(
        self,
        name: str = "two_dof_arm_torch",
        m1: float = 1.864572,
        m2: float = 1.534315,
        l1g: float = 0.180496,
        l2g: float = 0.181479,
        i1: float = 0.013193,
        i2: float = 0.020062,
        l1: float = 0.309,
        l2: float = 0.26,
        viscosity: float = 0.0,
        gravity_vec: Any = (0.0, 0.0, -9.81),
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        # Joint limits (radians), same spirit as original NumPy code
        sho_limit = (math.radians(0.0), math.radians(140.0))
        elb_limit = (math.radians(0.0), math.radians(160.0))
        lb = [sho_limit[0], elb_limit[0]]
        ub = [sho_limit[1], elb_limit[1]]

        super().__init__(
            dof=2,
            space_dim=2,
            name=name,
            pos_lower_bound=lb,
            pos_upper_bound=ub,
            device=device,
            dtype=dtype,
            **kwargs,
        )

        # Physical params (stored as floats; the Robot_torch carries Torch tensors)
        self.m1, self.m2 = float(m1), float(m2)
        self.L1g, self.L2g = float(l1g), float(l2g)
        self.I1, self.I2 = float(i1), float(i2)
        self.L1, self.L2 = float(l1), float(l2)
        self.c_viscosity = float(viscosity)

        # Gravity vector (3x1 Tensor)
        self._gravity_vec = torch.as_tensor(
            gravity_vec, dtype=self.dtype, device=self.device
        ).reshape(3, 1)

        # Underlying Torch robot (Serial from Robot_torch)
        I1_tensor = torch.diag(
            torch.tensor([1e-6, 1e-6, self.I1], dtype=self.dtype, device=self.device)
        )
        I2_tensor = torch.diag(
            torch.tensor([1e-6, 1e-6, self.I2], dtype=self.dtype, device=self.device)
        )

        self._robot = RobotSerial(
            q=torch.zeros(2, 1, dtype=self.dtype, device=self.device),
            qd=torch.zeros(2, 1, dtype=self.dtype, device=self.device),
            qdd=torch.zeros(2, 1, dtype=self.dtype, device=self.device),
            linksLengths=[self.L1, self.L2],
            COMs=[self.L1g, self.L2g],
            mass=[self.m1, self.m2],
            inertia=[I1_tensor, I2_tensor],
            gravity=self._gravity_vec.reshape(3),  # Robot_torch expects 3-vector
            name=f"{name}_robot",
        )

    # ------------------------------------------------------------------
    # Internal helper: push state into underlying robot
    # ------------------------------------------------------------------
    def _set_state(self, q: Tensor, qd: Tensor):
        """
        Push q, qd (B,2) into the underlying Torch robot.
        """
        q = self._as_batch(q, self.dof)   # (B,2)
        qd = self._as_batch(qd, self.dof) # (B,2)

        self._robot.q = q
        self._robot.qd = qd

    def _fk_frames(self) -> Tensor:
        """
        Call _kin.forwardHTM(self._robot) and normalize result to a tensor of shape:
            (B, n_frames, 4, 4)

        Handles both:
        - our Torch canonical version (tensor),
        - your older/list-based version (list of frames).
        """
        T_raw = _kin.forwardHTM(self._robot)
        dt = self.dtype
        dev = self.device

        # Case 1: list of frames
        if isinstance(T_raw, list):
            if len(T_raw) == 0:
                raise ValueError("forwardHTM returned empty list")

            first = T_raw[0]
            T0 = torch.as_tensor(first, dtype=dt, device=dev)

            if T0.dim() == 2:
                # Each element is (4,4) -> stack to (frames,4,4) then add batch
                Ts = [
                    torch.as_tensor(T, dtype=dt, device=dev) for T in T_raw
                ]  # list of (4,4)
                T_all = torch.stack(Ts, dim=0)  # (frames,4,4)
                T_all = T_all.unsqueeze(0)      # (1,frames,4,4)
            elif T0.dim() == 3:
                # Each element is (B,4,4) -> stack along frame dim -> (B,frames,4,4)
                Ts = [
                    torch.as_tensor(T, dtype=dt, device=dev) for T in T_raw
                ]  # list of (B,4,4)
                T_all = torch.stack(Ts, dim=1)  # (B,frames,4,4)
            else:
                raise ValueError(
                    f"Unexpected frame shape in forwardHTM list: {tuple(T0.shape)}"
                )

        # Case 2: already a tensor
        elif torch.is_tensor(T_raw):
            T_all = T_raw.to(device=dev, dtype=dt)
            if T_all.dim() == 3:
                # (frames,4,4) -> (1,frames,4,4)
                T_all = T_all.unsqueeze(0)
            elif T_all.dim() == 4:
                # (B,frames,4,4) already
                pass
            else:
                raise ValueError(
                    f"Unexpected tensor shape from forwardHTM: {tuple(T_all.shape)}"
                )
        else:
            raise TypeError(
                f"forwardHTM returned unsupported type: {type(T_raw)}"
            )

        return T_all  # (B,frames,4,4)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def ode(
        self, inputs: Tensor, joint_state: Tensor, endpoint_load: Tensor
    ) -> Tensor:
        """
        Compute qdd for a batch of states/inputs/endpoint loads (Torch).

        Shapes:
          joint_state   : (B, 2*dof)
          inputs (τ)    : (B, dof)
          endpoint_load : (B, space_dim)  # planar force [fx, fy]
          returns       : (B, dof)
        """
        # Batchify
        joint_state = self._as_batch(joint_state, self.state_dim)
        inputs = self._as_batch(inputs, self.input_dim)
        endpoint_load = self._as_batch(endpoint_load, self.space_dim)

        q, qd = torch.split(joint_state, self.dof, dim=1)  # (B,2), (B,2)
        B = q.shape[0]

        # Set robot state
        self._set_state(q, qd)

        # Library dynamics (Torch, fully differentiable)
        D = _dyn.inertiaMatrixCOM(self._robot)           # (B,2,2) or (2,2)
        C = _dyn.centrifugalCoriolisCOM(self._robot)     # (B,2,2) or (2,2)
        g = _dyn.gravitationalCOM(self._robot, g=self._gravity_vec)  # (B,2,1) or (2,1)

        # Normalize shapes to batched (B,2,2) and (B,2,1)
        if D.dim() == 2:
            D = D.unsqueeze(0)            # (1,2,2)
        if C.dim() == 2:
            C = C.unsqueeze(0)            # (1,2,2)
        if g.dim() == 2:
            g = g.unsqueeze(0)            # (1,2,1)

        # Geometric Jacobian: take linear x-y rows
        J = _kin.geometricJacobian(self._robot)          # (B,6,2) or (6,2)
        if J.dim() == 2:
            J = J.unsqueeze(0)                           # (1,6,2)
        J_xy = J[:, 0:2, :]                              # (B,2,2)

        # External endpoint force -> joint torques via J^T f
        f = endpoint_load.view(B, self.space_dim, 1)     # (B,2,1)
        tau_ext = torch.matmul(J_xy.transpose(1, 2), f).squeeze(-1)  # (B,2)
        tau = inputs + tau_ext                           # (B,2)

        # qdd = D^{-1} [ tau - C qd - g ]
        qd_vec = qd.view(B, self.dof, 1)                 # (B,2,1)
        C_qd = torch.matmul(C, qd_vec).squeeze(-1)       # (B,2)
        g_vec = g.squeeze(-1)                            # (B,2)

        rhs = tau - C_qd - g_vec                         # (B,2)

        # Stable linear solve instead of explicit inverse
        qdd = torch.linalg.solve(D, rhs.unsqueeze(-1)).squeeze(-1)  # (B,2)
        return qdd

    def integrate(
        self, dt: float, state_derivative: Tensor, joint_state: Tensor
    ) -> Tensor:
        """Same semi-implicit Euler as base Skeleton, preserved for compatibility."""
        return super().integrate(dt, state_derivative, joint_state)

    def joint2cartesian(self, joint_state: Tensor) -> Tensor:
        """
        Returns end-effector cartesian state [x, y, xd, yd] for each batch row.
        Output shape: (B, 4)
        """
        joint_state = self._as_batch(joint_state, self.state_dim)
        q, qd = torch.split(joint_state, self.dof, dim=1)
        B = q.shape[0]

        self._set_state(q, qd)

        # FK frames: normalize to (B, n_frames, 4,4)
        T_all = self._fk_frames()
        # end-effector is last frame
        T_ee = T_all[:, -1, :, :]             # (B,4,4)
        pos = T_ee[:, 0:2, 3]                 # (B,2) -> x,y

        # Linear Jacobian for velocities
        J = _kin.geometricJacobian(self._robot)  # (B,6,2) or (6,2)
        if J.dim() == 2:
            J = J.unsqueeze(0)                   # (1,6,2)
        J_xy = J[:, 0:2, :]                      # (B,2,2)

        qd_vec = qd.view(B, self.dof, 1)         # (B,2,1)
        v_xy = torch.matmul(J_xy, qd_vec).squeeze(-1)  # (B,2)

        return torch.cat([pos, v_xy], dim=1)     # (B,4) = [x,y,xd,yd]

    def path2cartesian(
        self,
        path_coordinates: Tensor,    # (B, 2, n_points) or (2, n_points)
        path_fixation_body: Tensor,  # (1, n_points) or (n_points,) in {0,1,2}
        joint_state: Tensor,         # (B, 2*dof)
    ):
        """
        Transform local fixation points into world XY, along with velocities and d(x,y)/dq.

        Position:
        - Transform local points by world/link1/link2 frames from FK (per batch).
          We use frame indices 0->world(base), 1->link1, 2->link2.

        Velocity (approx.):
        - Use the end-effector linear Jacobian for all points (uniform approx).

        Derivative wrt q:
        - Use linear block of the Jacobian as an approximation per point.

        Returns
        -------
        xy      : (B, 2, n_points)
        dxy_dt  : (B, 2, n_points)
        dxy_dq  : (B, 2, 2, n_points)
        """
        # Ensure joint_state is batched
        joint_state = self._as_batch(joint_state, self.state_dim)
        q, qd = torch.split(joint_state, self.dof, dim=1)
        B = q.shape[0]

        # Normalize path_fixation_body -> (n_points,) long
        body = torch.as_tensor(
            path_fixation_body, dtype=torch.long, device=self.device
        )
        if body.dim() == 2:
            body = body.view(-1)
        elif body.dim() == 1:
            pass
        else:
            raise ValueError(
                f"path_fixation_body must be 1D or 2D, got {tuple(body.shape)}"
            )
        n_points = body.numel()

        # Normalize path_coordinates -> (B, 2, n_points)
        pc = torch.as_tensor(path_coordinates, dtype=self.dtype, device=self.device)
        if pc.dim() == 2:
            # (2, n_points) => (1, 2, n_points)
            if pc.shape[0] != 2:
                raise ValueError(
                    f"path_coordinates 2D must be (2, n_points), got {tuple(pc.shape)}"
                )
            pc = pc.unsqueeze(0)
        elif pc.dim() == 3:
            if pc.shape[1] != 2:
                raise ValueError(
                    f"path_coordinates 3D must be (B, 2, n_points), got {tuple(pc.shape)}"
                )
        else:
            raise ValueError(
                f"path_coordinates must be 2D or 3D, got {tuple(pc.shape)}"
            )

        if pc.shape[2] != n_points:
            raise ValueError(
                f"path_coordinates has n_points={pc.shape[2]}, "
                f"but path_fixation_body has {n_points}"
            )

        if pc.shape[0] == 1 and B > 1:
            pc = pc.expand(B, -1, -1)  # broadcast same local points to all batch rows
        elif pc.shape[0] != B:
            raise ValueError(
                f"Batch size mismatch: joint_state B={B}, path_coordinates B={pc.shape[0]}"
            )

        # Set robot state
        self._set_state(q, qd)

        # FK frames: (B, n_frames, 4,4)
        T_all = self._fk_frames()          # frames 0..n (0 is world/base)
        # Use first three frames: 0->world, 1->link1, 2->link2
        if T_all.shape[1] < 3:
            raise ValueError(
                f"Expected at least 3 frames (world, link1, link2), got {T_all.shape[1]}"
            )
        T_body = T_all[:, 0:3, :, :]       # (B,3,4,4)

        # Select transform per point index -> Ts_all: (B, n_points, 4,4)
        Ts_all = T_body[:, body, :, :]

        # Local homogeneous points: (B,4,n_points)
        ones = torch.ones((B, 1, n_points), dtype=self.dtype, device=self.device)
        zeros = torch.zeros((B, 1, n_points), dtype=self.dtype, device=self.device)
        local_h = torch.cat([pc, zeros, ones], dim=1)   # (B,4,n_points)

        # Apply transforms: world_h shape (B, n_points, 4)
        local_h_vec = local_h.permute(0, 2, 1).unsqueeze(-1)  # (B,n_points,4,1)
        world_vec = torch.matmul(Ts_all, local_h_vec)         # (B,n_points,4,1)
        world = world_vec.squeeze(-1)                         # (B,n_points,4)

        xy = world[..., 0:2]          # (B,n_points,2)
        xy = xy.permute(0, 2, 1)      # (B,2,n_points)

        # Approx velocities using EE linear Jacobian
        J = _kin.geometricJacobian(self._robot)  # (B,6,2) or (6,2)
        if J.dim() == 2:
            J = J.unsqueeze(0)
        J_lin = J[:, 0:2, :]          # (B,2,2)

        qd_vec = qd.view(B, self.dof, 1)           # (B,2,1)
        v_xy = torch.matmul(J_lin, qd_vec).squeeze(-1)    # (B,2)
        dxy_dt = v_xy.unsqueeze(-1).expand(B, 2, n_points)  # (B,2,n_points)

        # d(x,y)/dq approximation per point
        dxy_dq = J_lin.unsqueeze(-1).expand(B, 2, self.dof, n_points)  # (B,2,2,P)

        return xy, dxy_dt, dxy_dq

    # ------------------------------------------------------------------
    # Device / dtype management (also move underlying robot)
    # ------------------------------------------------------------------
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

        super().to(device=device, dtype=dtype)

        def _move(x):
            return x.to(device=device, dtype=dtype) if isinstance(x, Tensor) else x

        self._gravity_vec = _move(self._gravity_vec)
        self._robot.to(device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype
        return self


# ---------------------------------------------------------------------
# Minimal Torch demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    print("[skeleton_torch] Simple smoke test...")

    arm = TwoDofArm(device="cpu", dtype=torch.float64)

    # Configure bounds & dt (mirrors NumPy demo)
    arm.build(
        timestep=0.002,
        pos_upper_bound=[math.radians(140.0), math.radians(160.0)],
        pos_lower_bound=[0.0, 0.0],
        vel_upper_bound=[+10.0, +10.0],
        vel_lower_bound=[-10.0, -10.0],
    )

    # State [q1, q2, qd1, qd2] — radians and rad/s
    state = torch.tensor(
        [[math.radians(30.0), math.radians(45.0), 0.1, -0.2]],
        dtype=arm.dtype,
        device=arm.device,
    )

    # Inputs (joint torques) and endpoint force [fx, fy]
    u = torch.zeros((1, 2), dtype=arm.dtype, device=arm.device)
    load = torch.zeros((1, 2), dtype=arm.dtype, device=arm.device)

    # qdd from dynamics
    qdd = arm.ode(u, state, load)

    # integrate one step
    new_state = arm.integrate(arm.dt, qdd, state)

    # EE cartesian [x, y, xd, yd]
    cart = arm.joint2cartesian(state)

    # Example path points (two points: one on link1, one on link2)
    path_coords = torch.tensor(
        [[[0.05, 0.02], [0.00, 0.00]]],  # shape (1,2,2)
        dtype=arm.dtype,
        device=arm.device,
    )
    path_bodies = torch.tensor([[1, 2]], dtype=torch.long, device=arm.device)
    xy, dxy_dt, dxy_dq = arm.path2cartesian(path_coords, path_bodies, state)

    print("  qdd:", qdd)
    print("  new_state:", new_state)
    print("  cartesian [x,y,xd,yd]:", cart)
    print(
        "  path xy shape:",
        xy.shape,
        "dxy_dt shape:",
        dxy_dt.shape,
        "dxy_dq shape:",
        dxy_dq.shape,
    )
    print("  path xy:", xy)
    print("  path dxy_dt:", dxy_dt)
    print("  path dxy_dq:", dxy_dq)
    print("[skeleton_torch] Smoke test done.")
