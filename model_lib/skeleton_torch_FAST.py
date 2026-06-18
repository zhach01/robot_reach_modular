# skeleton_torch_FAST.py
# -*- coding: utf-8 -*-
"""
FAST 2-DOF planar arm with ANALYTIC dynamics.

This replaces the slow HTM-based dynamics with closed-form equations.
For a 2-DOF arm, we can compute M, C, g analytically without loops or autograd.

Performance: ~50-100x faster than HTM-based dynamics.

The dynamics equations are:
    M(q) * qdd + C(q, qd) * qd + g(q) = tau + J^T * f_ext

where M, C, g are computed analytically (no numerical differentiation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor


@dataclass
class Skeleton:
    """Base skeleton class (same interface as original)."""
    dof: int
    space_dim: int
    name: str = "skeleton"
    pos_lower_bound: Union[float, Tuple[float, ...]] = -1.0
    pos_upper_bound: Union[float, Tuple[float, ...]] = +1.0
    vel_lower_bound: Union[float, Tuple[float, ...]] = -1000.0
    vel_upper_bound: Union[float, Tuple[float, ...]] = +1000.0
    dt: float = 0.001
    input_dim: Optional[int] = None
    state_dim: Optional[int] = None
    output_dim: Optional[int] = None
    device: Union[str, torch.device] = "cpu"
    dtype: Optional[torch.dtype] = None

    def __post_init__(self):
        self.device = torch.device(self.device)
        if self.dtype is None:
            self.dtype = torch.get_default_dtype()

        self.input_dim = self.input_dim or self.dof
        self.state_dim = self.state_dim or (2 * self.dof)
        self.output_dim = self.output_dim or self.state_dim

        def _norm_bound(val, name: str) -> Tensor:
            t = torch.as_tensor(val, dtype=self.dtype, device=self.device)
            if t.numel() == 1:
                t = t.expand(self.dof)
            elif t.numel() != self.dof:
                raise ValueError(f"{name} must have length {self.dof}")
            return t.view(1, self.dof)

        self.pos_lower_bound = _norm_bound(self.pos_lower_bound, "pos_lower_bound")
        self.pos_upper_bound = _norm_bound(self.pos_upper_bound, "pos_upper_bound")
        self.vel_lower_bound = _norm_bound(self.vel_lower_bound, "vel_lower_bound")
        self.vel_upper_bound = _norm_bound(self.vel_upper_bound, "vel_upper_bound")

    def _as_batch(self, x: Tensor, expected_dim: int) -> Tensor:
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def clip_position(self, q: Tensor) -> Tensor:
        return torch.clamp(q, self.pos_lower_bound, self.pos_upper_bound)

    def clip_velocity(self, qd: Tensor) -> Tensor:
        return torch.clamp(qd, self.vel_lower_bound, self.vel_upper_bound)

    def integrate(self, dt: float, state_derivative: Tensor, joint_state: Tensor) -> Tensor:
        joint_state = self._as_batch(joint_state, self.state_dim)
        state_derivative = self._as_batch(state_derivative, self.dof)
        q = joint_state[:, : self.dof]
        qd = joint_state[:, self.dof :]
        new_qd = self.clip_velocity(qd + dt * state_derivative)
        new_q = self.clip_position(q + dt * new_qd)
        return torch.cat([new_q, new_qd], dim=1)


class TwoDofArm(Skeleton):
    """
    FAST 2-DOF planar arm with ANALYTIC dynamics.
    
    Uses closed-form M, C, g matrices instead of HTM-based computation.
    ~50-100x faster than the original HTM-based TwoDofArm.
    
    Dynamics:
        qdd = M(q)^{-1} * [tau + J(q)^T * f - C(q,qd) * qd - g(q)]
    
    Parameters follow the standard 2-link planar arm model.
    """

    def __init__(
        self,
        name: str = "two_dof_arm_fast",
        m1: float = 1.864572,
        m2: float = 1.534315,
        l1g: float = 0.180496,  # COM distance from joint 1
        l2g: float = 0.181479,  # COM distance from joint 2
        i1: float = 0.013193,   # Inertia about COM 1
        i2: float = 0.020062,   # Inertia about COM 2
        l1: float = 0.309,      # Link 1 length
        l2: float = 0.26,       # Link 2 length
        viscosity: float = 0.0,
        gravity: float = 0.0,   # Set to 0 for planar arm in horizontal plane
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        # Joint limits (radians)
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

        # Store physical parameters as tensors for fast computation
        self.m1 = torch.tensor(m1, dtype=self.dtype, device=self.device)
        self.m2 = torch.tensor(m2, dtype=self.dtype, device=self.device)
        self.L1 = torch.tensor(l1, dtype=self.dtype, device=self.device)
        self.L2 = torch.tensor(l2, dtype=self.dtype, device=self.device)
        self.L1g = torch.tensor(l1g, dtype=self.dtype, device=self.device)
        self.L2g = torch.tensor(l2g, dtype=self.dtype, device=self.device)
        self.I1 = torch.tensor(i1, dtype=self.dtype, device=self.device)
        self.I2 = torch.tensor(i2, dtype=self.dtype, device=self.device)
        self.c_viscosity = torch.tensor(viscosity, dtype=self.dtype, device=self.device)
        self.gravity = torch.tensor(gravity, dtype=self.dtype, device=self.device)
        
        # Pre-compute constants for mass matrix
        # M11 = I1 + I2 + m1*L1g^2 + m2*(L1^2 + L2g^2 + 2*L1*L2g*cos(q2))
        # M12 = M21 = I2 + m2*L2g^2 + m2*L1*L2g*cos(q2)
        # M22 = I2 + m2*L2g^2
        self._alpha = self.I1 + self.I2 + self.m1 * self.L1g**2 + self.m2 * (self.L1**2 + self.L2g**2)
        self._beta = self.m2 * self.L1 * self.L2g
        self._delta = self.I2 + self.m2 * self.L2g**2

    def _set_state(self, q: Tensor, qd: Tensor):
        """Set state (for compatibility - not actually needed for analytic dynamics)."""
        pass

    def mass_matrix(self, q: Tensor) -> Tensor:
        """
        Compute mass matrix M(q) analytically.
        
        Args:
            q: (B, 2) joint angles
            
        Returns:
            M: (B, 2, 2) mass matrix
        """
        c2 = torch.cos(q[:, 1])  # cos(q2)
        bc2 = self._beta * c2
        # built via stack (bit-identical to zeros + index-assign, fewer autograd
        # nodes, vectorizes better at the large batch sizes used in training).
        m00 = self._alpha + 2 * bc2
        m01 = self._delta + bc2
        m11 = self._delta * torch.ones_like(c2)
        row0 = torch.stack([m00, m01], dim=-1)
        row1 = torch.stack([m01, m11], dim=-1)
        return torch.stack([row0, row1], dim=-2)  # (B,2,2)

    def coriolis_matrix(self, q: Tensor, qd: Tensor) -> Tensor:
        """
        Compute Coriolis/centrifugal matrix C(q, qd) analytically.
        
        Using Christoffel symbols, C_ij = sum_k c_ijk * qd_k
        where c_ijk = (1/2) * (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)
        
        For 2-DOF arm, this simplifies to:
        C = [-beta*s2*qd2,  -beta*s2*(qd1+qd2)]
            [beta*s2*qd1,   0                 ]
            
        Args:
            q: (B, 2) joint angles
            qd: (B, 2) joint velocities
            
        Returns:
            C: (B, 2, 2) Coriolis matrix
        """
        s2 = torch.sin(q[:, 1])  # sin(q2)
        qd1 = qd[:, 0]
        qd2 = qd[:, 1]
        h = self._beta * s2  # Common term
        z = torch.zeros_like(h)
        row0 = torch.stack([-h * qd2, -h * (qd1 + qd2)], dim=-1)
        row1 = torch.stack([h * qd1, z], dim=-1)
        return torch.stack([row0, row1], dim=-2)  # (B,2,2)

    def gravity_vector(self, q: Tensor) -> Tensor:
        """
        Compute gravity vector g(q).
        
        For planar arm in horizontal plane, g = 0.
        For vertical plane: g1 = (m1*L1g + m2*L1)*g*cos(q1) + m2*L2g*g*cos(q1+q2)
                           g2 = m2*L2g*g*cos(q1+q2)
        
        Args:
            q: (B, 2) joint angles
            
        Returns:
            g: (B, 2) gravity torques
        """
        B = q.shape[0]
        
        if self.gravity == 0:
            return torch.zeros(B, 2, dtype=self.dtype, device=self.device)
        
        q1 = q[:, 0]
        q12 = q[:, 0] + q[:, 1]
        
        g_vec = torch.zeros(B, 2, dtype=self.dtype, device=self.device)
        g_vec[:, 0] = (self.m1 * self.L1g + self.m2 * self.L1) * self.gravity * torch.cos(q1) + \
                      self.m2 * self.L2g * self.gravity * torch.cos(q12)
        g_vec[:, 1] = self.m2 * self.L2g * self.gravity * torch.cos(q12)
        
        return g_vec

    def jacobian(self, q: Tensor) -> Tensor:
        """
        Compute Jacobian J = d(fingertip)/d(q) analytically.
        
        For 2-DOF planar arm:
        J = [-L1*sin(q1) - L2*sin(q1+q2),  -L2*sin(q1+q2)]
            [L1*cos(q1) + L2*cos(q1+q2),   L2*cos(q1+q2)]
        
        Args:
            q: (B, 2) joint angles
            
        Returns:
            J: (B, 2, 2) Jacobian matrix
        """
        B = q.shape[0]
        q1 = q[:, 0]
        q12 = q[:, 0] + q[:, 1]
        
        s1 = torch.sin(q1)
        c1 = torch.cos(q1)
        s12 = torch.sin(q12)
        c12 = torch.cos(q12)
        row0 = torch.stack([-self.L1 * s1 - self.L2 * s12, -self.L2 * s12], dim=-1)
        row1 = torch.stack([self.L1 * c1 + self.L2 * c12, self.L2 * c12], dim=-1)
        return torch.stack([row0, row1], dim=-2)  # (B,2,2)

    def forward_kinematics(self, q: Tensor) -> Tensor:
        """
        Compute fingertip position from joint angles.
        
        Args:
            q: (B, 2) joint angles
            
        Returns:
            xy: (B, 2) fingertip position [x, y]
        """
        q1 = q[:, 0]
        q12 = q[:, 0] + q[:, 1]
        
        x = self.L1 * torch.cos(q1) + self.L2 * torch.cos(q12)
        y = self.L1 * torch.sin(q1) + self.L2 * torch.sin(q12)
        
        return torch.stack([x, y], dim=-1)

    def ode(
        self, 
        inputs: Tensor, 
        joint_state: Tensor, 
        endpoint_load: Tensor
    ) -> Tensor:
        """
        Compute qdd for a batch of states/inputs (FAST analytic version).
        
        qdd = M^{-1} * (tau + J^T * f - C * qd - g - viscosity * qd)
        
        Args:
            inputs: (B, 2) joint torques
            joint_state: (B, 4) joint state [q1, q2, qd1, qd2]
            endpoint_load: (B, 2) external force at fingertip [fx, fy]
            
        Returns:
            qdd: (B, 2) joint accelerations
        """
        joint_state = self._as_batch(joint_state, self.state_dim)
        inputs = self._as_batch(inputs, self.input_dim)
        endpoint_load = self._as_batch(endpoint_load, self.space_dim)
        
        B = joint_state.shape[0]
        q = joint_state[:, :2]
        qd = joint_state[:, 2:]
        
        # Broadcast inputs if needed
        if inputs.shape[0] == 1 and B > 1:
            inputs = inputs.expand(B, -1)
        if endpoint_load.shape[0] == 1 and B > 1:
            endpoint_load = endpoint_load.expand(B, -1)
        
        # Compute dynamics matrices (all analytic, no loops!)
        M = self.mass_matrix(q)      # (B, 2, 2)
        C = self.coriolis_matrix(q, qd)  # (B, 2, 2)
        g = self.gravity_vector(q)   # (B, 2)
        J = self.jacobian(q)         # (B, 2, 2)
        
        # External force contribution: J^T * f
        f = endpoint_load.unsqueeze(-1)  # (B, 2, 1)
        tau_ext = torch.bmm(J.transpose(1, 2), f).squeeze(-1)  # (B, 2)
        
        # Total torque
        tau = inputs + tau_ext
        
        # Coriolis term: C * qd
        C_qd = torch.bmm(C, qd.unsqueeze(-1)).squeeze(-1)  # (B, 2)
        
        # Viscous damping
        viscous = self.c_viscosity * qd
        
        # Right-hand side: tau - C*qd - g - viscosity*qd
        rhs = tau - C_qd - g - viscous
        
        # Solve M * qdd = rhs
        qdd = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)
        
        return qdd

    def joint2cartesian(self, joint_state: Tensor) -> Tensor:
        """
        Convert joint state to Cartesian state.
        
        Args:
            joint_state: (B, 4) [q1, q2, qd1, qd2]
            
        Returns:
            cartesian: (B, 4) [x, y, xd, yd]
        """
        joint_state = self._as_batch(joint_state, self.state_dim)
        q = joint_state[:, :2]
        qd = joint_state[:, 2:]
        
        # Position
        xy = self.forward_kinematics(q)  # (B, 2)
        
        # Velocity: xd = J * qd
        J = self.jacobian(q)  # (B, 2, 2)
        xyd = torch.bmm(J, qd.unsqueeze(-1)).squeeze(-1)  # (B, 2)
        
        return torch.cat([xy, xyd], dim=-1)

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Move skeleton to device/dtype."""
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        
        # Move all tensors
        for attr in ['m1', 'm2', 'L1', 'L2', 'L1g', 'L2g', 'I1', 'I2', 
                     'c_viscosity', 'gravity', '_alpha', '_beta', '_delta',
                     'pos_lower_bound', 'pos_upper_bound', 
                     'vel_lower_bound', 'vel_upper_bound']:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if isinstance(val, Tensor):
                    setattr(self, attr, val.to(device=self.device, dtype=self.dtype))
        
        return self


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("FAST TwoDofArm Smoke Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    
    print(f"Device: {device}")
    
    # Create arm
    arm = TwoDofArm(device=device, dtype=dtype)
    
    # Test different batch sizes
    for B in [1, 8, 32, 128]:
        q = torch.deg2rad(torch.tensor([[45.0, 90.0]], device=device, dtype=dtype))
        q = q.expand(B, -1).contiguous()
        qd = torch.zeros(B, 2, device=device, dtype=dtype)
        joint_state = torch.cat([q, qd], dim=-1)
        tau = torch.zeros(B, 2, device=device, dtype=dtype)
        f_ext = torch.zeros(B, 2, device=device, dtype=dtype)
        
        # Warm up
        for _ in range(10):
            qdd = arm.ode(tau, joint_state, f_ext)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Time it
        t0 = time.perf_counter()
        n_calls = 1000
        for _ in range(n_calls):
            qdd = arm.ode(tau, joint_state, f_ext)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_total = time.perf_counter() - t0
        
        print(f"B={B:3d}: ode() = {t_total/n_calls*1000:.4f} ms/call")
    
    # Test with gradients
    print("\nWith gradients (BPTT):")
    for B in [32, 128]:
        q = torch.deg2rad(torch.tensor([[45.0, 90.0]], device=device, dtype=dtype))
        q = q.expand(B, -1).contiguous()
        qd = torch.zeros(B, 2, device=device, dtype=dtype)
        joint_state = torch.cat([q, qd], dim=-1)
        tau = torch.zeros(B, 2, device=device, dtype=dtype, requires_grad=True)
        f_ext = torch.zeros(B, 2, device=device, dtype=dtype)
        
        # Warm up with grads
        qdd = arm.ode(tau, joint_state, f_ext)
        loss = qdd.pow(2).sum()
        loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Time 60-step rollout
        t0 = time.perf_counter()
        states = [joint_state]
        for _ in range(60):
            qdd = arm.ode(tau, states[-1], f_ext)
            new_state = arm.integrate(0.01, qdd, states[-1])
            states.append(new_state)
        final_pos = arm.forward_kinematics(states[-1][:, :2])
        loss = final_pos.pow(2).sum()
        loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_bptt = time.perf_counter() - t0
        
        print(f"B={B:3d}: 60-step BPTT = {t_bptt*1000:.1f} ms")
    
    print("\n✅ Smoke test passed!")
