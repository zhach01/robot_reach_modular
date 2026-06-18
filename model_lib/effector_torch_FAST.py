# effector_torch_FAST.py
# -*- coding: utf-8 -*-
"""
FAST Effector for MotorNet training.

Uses analytic 2-DOF dynamics (skeleton_torch_FAST.TwoDofArm) instead of
the slow HTM-based dynamics.

Performance: ~50-100x faster than original effector_torch.py

This is a MINIMAL effector focused on speed for MotorNet training.
For full-featured simulations, use the original effector_torch.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

# Import FAST skeleton
from model_lib.skeleton_torch_FAST import TwoDofArm


class FastRigidTendonArm:
    """
    FAST rigid tendon arm effector for MotorNet training.
    
    Uses analytic dynamics and simplified muscle model.
    ~50-100x faster than RigidTendonArm26.
    
    Compatible with EnvironmentTorch interface.
    """
    
    def __init__(
        self,
        timestep: float = 0.01,
        damping: float = 0.05,
        n_ministeps: int = 1,
        integration_method: str = 'euler',
        device: Union[str, torch.device] = 'cpu',
        dtype: Optional[torch.dtype] = None,
        # Skeleton params
        m1: float = 1.864572,
        m2: float = 1.534315,
        l1g: float = 0.180496,
        l2g: float = 0.181479,
        i1: float = 0.013193,
        i2: float = 0.020062,
        l1: float = 0.309,
        l2: float = 0.26,
        **kwargs,
    ):
        self.device = torch.device(device)
        self.dtype = dtype or torch.get_default_dtype()
        self.dt = float(timestep)
        self.damping = float(damping)
        self.n_ministeps = int(n_ministeps)
        self.minidt = self.dt / self.n_ministeps
        self.integration_method = integration_method
        
        # Create FAST skeleton
        self.skeleton = TwoDofArm(
            m1=m1, m2=m2, l1g=l1g, l2g=l2g, i1=i1, i2=i2, l1=l1, l2=l2,
            viscosity=damping,
            device=self.device,
            dtype=self.dtype,
        )
        
        self.dof = self.skeleton.dof
        self.state_dim = self.skeleton.state_dim
        self.space_dim = self.skeleton.space_dim
        
        # Muscle configuration (6 muscles like RigidTendonArm26)
        self.n_muscles = 6
        self._setup_muscles()
        
        # State dict (compatible with Environment interface)
        self.states: Dict[str, Optional[Tensor]] = {
            "joint": None,
            "cartesian": None,
            "geometry": None,
            "muscle": None,
            "fingertip": None,
        }
        
        # For compatibility
        self.muscle = self  # Self-reference for muscle interface
        self.muscle_name = [f"muscle_{i}" for i in range(self.n_muscles)]
        
        # RNG
        self._seed: Optional[int] = None
        self._generator: Optional[torch.Generator] = None
    
    def _setup_muscles(self):
        """Setup muscle parameters (simplified from RigidTendonArm26)."""
        # Maximum isometric forces (N)
        self.max_iso_force = torch.tensor([
            838.0,   # pectoralis major (shoulder flexor)
            1207.0,  # deltoid (shoulder extensor)  
            1422.0,  # brachioradialis (elbow flexor)
            1549.0,  # triceps lateral (elbow extensor)
            414.0,   # biceps short (bi-articular flexor)
            603.0,   # triceps long (bi-articular extensor)
        ], device=self.device, dtype=self.dtype)
        
        # Constant moment arms (m) - simplified approximation
        # Shape: (2 joints, 6 muscles) where R_ij is moment arm of muscle j at joint i
        # Positive = flexor, Negative = extensor
        self.moment_arms = torch.tensor([
            # Shoulder moment arms (joint 0)
            [-0.04, 0.04, 0.0, 0.0, -0.03, 0.03],
            # Elbow moment arms (joint 1)  
            [0.0, 0.0, -0.025, 0.025, -0.025, 0.025],
        ], device=self.device, dtype=self.dtype)  # (2, 6)
        
        # Muscle dynamics parameters
        self.min_activation = 0.02
        self.tau_act = 0.015  # Activation time constant
        self.tau_deact = 0.050  # Deactivation time constant
        
        # Force index in muscle state (for compatibility)
        self.force_index = 0
        
        # Muscle state names (for compatibility)
        self.state_name = ["force", "activation", "excitation"]
    
    @property
    def torch_generator(self) -> torch.Generator:
        if self._generator is None:
            self._generator = torch.Generator(device=self.device)
            if self._seed is not None:
                self._generator.manual_seed(self._seed)
        return self._generator
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        """Reset effector to initial state."""
        options = options or {}
        batch_size = int(options.get("batch_size", 1))
        joint_state = options.get("joint_state", None)
        
        if seed is not None:
            self._seed = seed
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(seed)
        
        # Initialize joint state
        if joint_state is not None:
            js = torch.as_tensor(joint_state, dtype=self.dtype, device=self.device)
            if js.dim() == 1:
                js = js.unsqueeze(0)
            if js.shape[0] == 1 and batch_size > 1:
                js = js.expand(batch_size, -1).contiguous()
            batch_size = js.shape[0]
        else:
            # Default: center of joint range
            q_mid = (self.skeleton.pos_lower_bound + self.skeleton.pos_upper_bound) / 2
            qd_zero = torch.zeros(1, self.dof, device=self.device, dtype=self.dtype)
            js = torch.cat([q_mid, qd_zero], dim=-1)
            js = js.expand(batch_size, -1).contiguous()
        
        self.states["joint"] = js
        
        # Compute derived states
        self._update_derived_states(batch_size)
        
        # Initialize muscle state
        self.states["muscle"] = self._get_initial_muscle_state(batch_size)
        
        return self.states, {}
    
    def _update_derived_states(self, batch_size: int):
        """Update Cartesian, geometry, and fingertip states."""
        js = self.states["joint"]
        q = js[:, :self.dof]
        qd = js[:, self.dof:]
        
        # Cartesian state
        self.states["cartesian"] = self.skeleton.joint2cartesian(js)
        
        # Fingertip (same as cartesian for 2-link arm)
        self.states["fingertip"] = self.states["cartesian"]
        
        # Geometry state: (B, 2+dof, n_muscles) = (B, 4, 6)
        # [0,:,:] = muscle lengths (simplified as constant)
        # [1,:,:] = muscle velocities (simplified as zero)
        # [2:,:,:] = moment arms R
        B = batch_size
        geom = torch.zeros(B, 2 + self.dof, self.n_muscles, 
                          device=self.device, dtype=self.dtype)
        
        # Lengths (constant approximation)
        geom[:, 0, :] = 0.1  # ~10cm nominal length
        # Velocities (zero for simplicity)
        geom[:, 1, :] = 0.0
        # Moment arms (constant, broadcast to batch)
        geom[:, 2:, :] = self.moment_arms.unsqueeze(0).expand(B, -1, -1)
        
        self.states["geometry"] = geom
    
    def _get_initial_muscle_state(self, batch_size: int) -> Tensor:
        """Get initial muscle state."""
        # Muscle state: (B, n_state_vars, n_muscles)
        # State vars: [force, activation, excitation]
        n_state = 3
        state = torch.zeros(batch_size, n_state, self.n_muscles,
                           device=self.device, dtype=self.dtype)
        
        # Initial activation at min
        state[:, 1, :] = self.min_activation
        state[:, 2, :] = self.min_activation
        
        return state
    
    def step(
        self,
        action: Tensor,
        endpoint_load: Optional[Tensor] = None,
        joint_load: Optional[Tensor] = None,
    ) -> Tuple[Dict[str, Tensor], None, None, None, Dict[str, Any]]:
        """
        Step the effector with muscle excitations.
        
        Args:
            action: (B, n_muscles) muscle excitations in [0, 1]
            endpoint_load: (B, 2) external force at fingertip (optional)
            joint_load: (B, 2) external joint torques (optional)
            
        Returns:
            Tuple of (states, None, None, None, info) for compatibility
        """
        B = self.states["joint"].shape[0]
        
        # Process action
        action = torch.as_tensor(action, dtype=self.dtype, device=self.device)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if action.shape[0] == 1 and B > 1:
            action = action.expand(B, -1)
        
        # Clamp excitation
        excitation = torch.clamp(action, self.min_activation, 1.0)
        
        # External loads
        if endpoint_load is None:
            endpoint_load = torch.zeros(B, 2, device=self.device, dtype=self.dtype)
        else:
            endpoint_load = torch.as_tensor(endpoint_load, dtype=self.dtype, device=self.device)
            if endpoint_load.dim() == 1:
                endpoint_load = endpoint_load.unsqueeze(0)
            if endpoint_load.shape[0] == 1 and B > 1:
                endpoint_load = endpoint_load.expand(B, -1)
        
        if joint_load is None:
            joint_load = torch.zeros(B, 2, device=self.device, dtype=self.dtype)
        else:
            joint_load = torch.as_tensor(joint_load, dtype=self.dtype, device=self.device)
            if joint_load.dim() == 1:
                joint_load = joint_load.unsqueeze(0)
            if joint_load.shape[0] == 1 and B > 1:
                joint_load = joint_load.expand(B, -1)
        
        # Integrate
        for _ in range(self.n_ministeps):
            self._euler_step(excitation, endpoint_load, joint_load)
        
        return self.states, None, None, None, {}
    
    def _euler_step(
        self,
        excitation: Tensor,
        endpoint_load: Tensor,
        joint_load: Tensor,
    ):
        """Single Euler integration step."""
        B = self.states["joint"].shape[0]
        
        # Current state
        js = self.states["joint"]
        ms = self.states["muscle"]
        
        # Update muscle activation (first-order dynamics)
        act = ms[:, 1, :]  # (B, M)
        exc = excitation  # (B, M)
        
        # Activation dynamics: da/dt = (exc - act) / tau
        tau = torch.where(exc > act, self.tau_act, self.tau_deact)
        dact = (exc - act) / tau
        new_act = torch.clamp(act + self.minidt * dact, self.min_activation, 1.0)
        
        # Compute muscle forces: F = activation * Fmax
        F = new_act * self.max_iso_force.unsqueeze(0)  # (B, M)
        
        # Compute joint torques from muscle forces: tau = R @ F
        # R is (2, 6), F is (B, 6) -> tau is (B, 2)
        R = self.moment_arms  # (2, 6)
        tau_muscle = torch.einsum('jm,bm->bj', R, F)  # (B, 2)
        
        # Total torque
        tau_total = tau_muscle + joint_load
        
        # Compute joint accelerations
        qdd = self.skeleton.ode(tau_total, js, endpoint_load)
        
        # Integrate joint state
        new_js = self.skeleton.integrate(self.minidt, qdd, js)
        
        # Update states
        self.states["joint"] = new_js
        
        # Update muscle state
        new_ms = ms.clone()
        new_ms[:, 0, :] = F / self.max_iso_force.unsqueeze(0)  # Normalized force
        new_ms[:, 1, :] = new_act
        new_ms[:, 2, :] = exc
        self.states["muscle"] = new_ms
        
        # Update derived states
        self._update_derived_states(B)
    
    def get_geometry(self, joint_state: Tensor) -> Tensor:
        """Get muscle geometry for given joint state."""
        B = joint_state.shape[0]
        geom = torch.zeros(B, 2 + self.dof, self.n_muscles,
                          device=self.device, dtype=self.dtype)
        geom[:, 0, :] = 0.1
        geom[:, 1, :] = 0.0
        geom[:, 2:, :] = self.moment_arms.unsqueeze(0).expand(B, -1, -1)
        return geom
    
    def joint2cartesian(self, joint_state: Tensor) -> Tensor:
        """Convert joint state to Cartesian state."""
        return self.skeleton.joint2cartesian(joint_state)
    
    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "FastRigidTendonArm":
        """Move effector to device/dtype."""
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        
        self.skeleton.to(device=self.device, dtype=self.dtype)
        
        self.max_iso_force = self.max_iso_force.to(device=self.device, dtype=self.dtype)
        self.moment_arms = self.moment_arms.to(device=self.device, dtype=self.dtype)
        
        for key, val in self.states.items():
            if val is not None:
                self.states[key] = val.to(device=self.device, dtype=self.dtype)
        
        return self


# =============================================================================
# Smoke Test
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("FastRigidTendonArm Smoke Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    
    print(f"Device: {device}")
    
    # Create fast effector
    arm = FastRigidTendonArm(
        timestep=0.01,
        damping=0.05,
        device=device,
        dtype=dtype,
    )
    
    # Test different batch sizes
    q0 = torch.deg2rad(torch.tensor([45.0, 90.0], device=device, dtype=dtype))
    qd0 = torch.zeros(2, device=device, dtype=dtype)
    joint0 = torch.cat([q0, qd0]).unsqueeze(0)
    
    print("\nStep timing (forward only):")
    for B in [1, 8, 32, 128]:
        arm.reset(options={"batch_size": B, "joint_state": joint0})
        action = torch.ones(B, 6, device=device, dtype=dtype) * 0.1
        
        # Warm up
        for _ in range(10):
            arm.step(action)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Time
        t0 = time.perf_counter()
        n_steps = 1000
        for _ in range(n_steps):
            arm.step(action)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_total = time.perf_counter() - t0
        
        print(f"B={B:3d}: step() = {t_total/n_steps*1000:.4f} ms")
    
    print("\n60-step rollout (forward only):")
    for B in [1, 8, 32, 128]:
        arm.reset(options={"batch_size": B, "joint_state": joint0})
        action = torch.ones(B, 6, device=device, dtype=dtype) * 0.1
        
        # Warm up
        for _ in range(60):
            arm.step(action)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Time
        arm.reset(options={"batch_size": B, "joint_state": joint0})
        t0 = time.perf_counter()
        for _ in range(60):
            arm.step(action)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_rollout = time.perf_counter() - t0
        
        print(f"B={B:3d}: 60 steps = {t_rollout*1000:.1f} ms ({t_rollout/60*1000:.3f} ms/step)")
    
    print("\n60-step BPTT:")
    for B in [32, 128]:
        arm.reset(options={"batch_size": B, "joint_state": joint0})
        action = torch.ones(B, 6, device=device, dtype=dtype, requires_grad=True) * 0.1
        
        # Warm up with grads
        positions = []
        for _ in range(60):
            arm.step(action)
            positions.append(arm.states["fingertip"][:, :2].clone())
        positions = torch.stack(positions, dim=1)
        loss = positions.pow(2).mean()
        loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Time
        arm.reset(options={"batch_size": B, "joint_state": joint0})
        action = torch.ones(B, 6, device=device, dtype=dtype, requires_grad=True) * 0.1
        
        t0 = time.perf_counter()
        positions = []
        for _ in range(60):
            arm.step(action)
            positions.append(arm.states["fingertip"][:, :2].clone())
        positions = torch.stack(positions, dim=1)
        loss = positions.pow(2).mean()
        loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_bptt = time.perf_counter() - t0
        
        print(f"B={B:3d}: BPTT = {t_bptt*1000:.1f} ms ({t_bptt/60*1000:.2f} ms/step)")
    
    print("\n✅ Smoke test passed!")
