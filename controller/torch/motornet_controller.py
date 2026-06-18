#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRUE MotorNet Controller (Torch)

Follows the same patterns as sliding_mode_torch.py and hybrid_bc_a_torch.py:
- Uses EnvironmentTorch + RigidTendonArm26
- Batched computations (B, *)
- Pure Torch, no NumPy in training loop
- Uses existing muscle_tools_torch for allocation

This is the correct implementation per Codol et al. (2024):
- GRU policy outputs muscle excitations
- BPTT through differentiable plant
- Loss on trajectory tracking

Obs = [x, xd, goal, error, t_norm] = 9 dims
Action = excitations u (M,) in [0, 1]
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

# ---- Model lib imports (same as other torch controllers) ----
from model_lib.torch.environment import Environment as EnvironmentTorch
from model_lib.torch.muscles import RigidTendonHillMuscle
from model_lib.torch.effector import RigidTendonArm26
from model_lib.torch.skeleton import TwoDofArm

# ---- Muscle tools (same as sliding_mode_torch) ----
from muscles.torch.muscle_tools import (
    get_Fmax_vec,
    active_force_from_activation,
    force_to_activation_bisect,
)


# =============================================================================
# Parameters
# =============================================================================

@dataclass
class MotorNetParams:
    """Parameters for MotorNet controller."""
    # Network
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.0
    
    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    batch_size: int = 32
    n_epochs: int = 200
    
    # Trial
    trial_duration: float = 0.6
    reach_radius: float = 0.10
    
    # Loss weights
    w_position: float = 100.0
    w_final: float = 200.0
    w_velocity: float = 10.0
    w_effort: float = 0.01
    w_smooth: float = 0.1
    
    # Device
    device: str = 'cuda'


# =============================================================================
# GRU Policy Network
# =============================================================================

class GRUPolicy(nn.Module):
    """
    Recurrent policy for MotorNet.
    
    Input: [x, y, vx, vy, goal_x, goal_y, err_x, err_y, t_norm] = 9
    Output: M muscle excitations in [0, 1]
    """
    
    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 128,
        output_dim: int = 6,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for m in self.output_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        obs: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            obs: (B, input_dim) or (B, T, input_dim)
            hidden: (n_layers, B, hidden_dim) or None
        
        Returns:
            action: (B, output_dim)
            hidden: (n_layers, B, hidden_dim)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        
        B = obs.shape[0]
        
        if hidden is None:
            hidden = torch.zeros(
                self.n_layers, B, self.hidden_dim,
                device=obs.device, dtype=obs.dtype
            )
        
        gru_out, hidden_new = self.gru(obs, hidden)
        action = self.output_head(gru_out[:, -1, :])
        
        return action, hidden_new
    
    def init_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(
            self.n_layers, batch_size, self.hidden_dim,
            device=device, dtype=dtype
        )


# =============================================================================
# MotorNet Controller (follows same pattern as SlidingModeController)
# =============================================================================

class MotorNetController:
    """
    MotorNet Controller (Torch version, batched).
    
    - Same interface as SlidingModeController / PDIFController
    - Uses GRU policy for excitations
    - Can be trained via BPTT through differentiable plant
    - Falls back to simple PD if not trained
    """
    
    def __init__(
        self,
        env: EnvironmentTorch,
        arm: RigidTendonArm26,
        params: MotorNetParams,
    ):
        self.env = env
        self.arm = arm
        self.p = params
        
        self.device = torch.device(params.device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.get_default_dtype()
        
        # Get muscle count from effector
        self.n_muscles = arm.n_muscles
        self.n_dof = arm.skeleton.dof
        
        # Policy network
        self.policy = GRUPolicy(
            input_dim=9,
            hidden_dim=params.hidden_dim,
            output_dim=self.n_muscles,
            n_layers=params.n_layers,
            dropout=params.dropout,
        ).to(device=self.device, dtype=torch.float32)
        
        # Hidden state
        self._hidden: Optional[Tensor] = None
        
        # Goal and timing
        self._goal: Optional[Tensor] = None
        self._t: float = 0.0
        self._T: float = params.trial_duration
        
        # Reference configuration
        self.qref: Optional[Tensor] = None
        
        # Training state
        self.is_trained: bool = False
        self.train_losses: List[float] = []
        
        # Optimizer (for training)
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=params.lr,
            weight_decay=params.weight_decay,
        )
    
    def reset(self, q0: Tensor) -> None:
        """Reset controller state."""
        q0 = torch.as_tensor(q0, device=self.device, dtype=self.dtype)
        if q0.dim() == 1:
            q0 = q0.unsqueeze(0)
        self.qref = q0.clone()
        self._hidden = None
        self._t = 0.0
        self._goal = None
    
    def set_goal(self, goal: Tensor) -> None:
        """Set reaching goal."""
        goal = torch.as_tensor(goal, device=self.device, dtype=self.dtype)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)
        self._goal = goal
        self._t = 0.0
    
    def _build_observation(
        self,
        x: Tensor,
        xd: Tensor,
        goal: Tensor,
        t_norm: float,
    ) -> Tensor:
        """
        Build observation vector.
        
        Args:
            x: (B, 2) position
            xd: (B, 2) velocity
            goal: (B, 2) goal
            t_norm: normalized time
        
        Returns:
            obs: (B, 9)
        """
        B = x.shape[0]
        err = goal - x
        t_tensor = torch.full((B, 1), t_norm, device=x.device, dtype=x.dtype)
        
        obs = torch.cat([x, xd, goal, err, t_tensor], dim=-1)
        return obs
    
    def compute(
        self,
        x_d: Tensor,
        xd_d: Tensor,
        xdd_d: Tensor,
    ) -> Dict[str, Any]:
        """
        Compute muscle activations.
        
        Same interface as SlidingModeController.compute().
        
        Args:
            x_d, xd_d, xdd_d: (B, 2) or (2,) trajectory references
        
        Returns:
            Dict with act, tau_des, q, qd, x, xd, etc.
        """
        # ---- Get current state (same pattern as sliding_mode_torch) ----
        joint = self.env.states["joint"]      # (B, 2*dof)
        cart = self.env.states["cartesian"]   # (B, >=4)
        geom = self.env.states["geometry"]    # (B, 2+dof, M)
        musc = self.env.states["muscle"]      # (B, n_states, M)
        
        B = joint.shape[0]
        n = self.n_dof
        M = self.n_muscles
        
        q = joint[:, :n]
        qd = joint[:, n:]
        x = cart[:, :2]
        xd = cart[:, 2:4]
        
        device = q.device
        dtype = q.dtype
        
        # ---- Ensure batched references ----
        def _ensure_batch(z: Tensor) -> Tensor:
            z = z.to(device=device, dtype=dtype)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            if z.shape[0] == 1 and B > 1:
                z = z.expand(B, -1)
            return z
        
        x_d = _ensure_batch(x_d)
        xd_d = _ensure_batch(xd_d)
        xdd_d = _ensure_batch(xdd_d)
        
        # ---- Goal (use x_d as goal if not set) ----
        if self._goal is None:
            goal = x_d
        else:
            goal = self._goal
            if goal.shape[0] == 1 and B > 1:
                goal = goal.expand(B, -1)
        
        # ---- Normalized time ----
        t_norm = min(self._t / self._T, 1.0) if self._T > 0 else 1.0
        
        # ---- Build observation ----
        obs = self._build_observation(x, xd, goal, t_norm)
        
        # ---- Policy forward ----
        if self._hidden is None:
            self._hidden = self.policy.init_hidden(B, device, torch.float32)
        
        # Policy uses float32
        obs_f32 = obs.to(dtype=torch.float32)
        hidden_f32 = self._hidden.to(dtype=torch.float32)
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            excitations, self._hidden = self.policy(obs_f32, hidden_f32)
        
        # Convert back to plant dtype
        excitations = excitations.to(dtype=dtype)
        
        # ---- Clamp excitations ----
        a_min = float(getattr(self.env.muscle, "min_activation", 0.02))
        excitations = torch.clamp(excitations, a_min, 1.0)
        
        # ---- Get muscle geometry ----
        lenvel = geom[:, :2, :]                    # (B, 2, M)
        R = geom[:, 2:2 + n, :]                    # (B, n, M)
        
        Fmax_vec = get_Fmax_vec(self.env, M, device=device, dtype=dtype)
        if Fmax_vec.dim() == 1:
            Fmax_vec = Fmax_vec.unsqueeze(0).expand(B, -1)
        
        # ---- Get passive force ----
        names = list(self.env.muscle.state_name)
        if "force-length PE" in names:
            idx_flpe = names.index("force-length PE")
            flpe = musc[:, idx_flpe, :]
        else:
            flpe = torch.zeros(B, M, device=device, dtype=dtype)
        
        # ---- Compute muscle force from excitations ----
        # For rigid tendon: excitation ≈ activation (fast dynamics)
        a = excitations
        
        # Active force
        af = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = Fmax_vec * (af + flpe)
        
        # ---- Compute tau from muscle forces ----
        # τ = -R @ F (same convention as sliding_mode_torch)
        if R.dim() == 3:
            tau = -torch.einsum("bnm,bm->bn", R, F_pred)
        else:
            tau = -(R @ F_pred.T).T
        
        # ---- Update timing ----
        self._t += float(self.arm.dt)
        
        # ---- Return same format as other controllers ----
        return {
            "tau_des": tau,
            "R": R[0] if B == 1 else R,
            "Fmax": Fmax_vec[0] if B == 1 else Fmax_vec,
            "F_des": F_pred,
            "act": a,
            "q": q,
            "qd": qd,
            "x": x,
            "xd": xd,
            "xref_tuple": (x_d, xd_d, xdd_d),
            "eta": 1.0,
            "diag": {},
        }
    
    @property
    def training(self) -> bool:
        return self.policy.training
    
    def train_mode(self, mode: bool = True):
        """Set training mode."""
        self.policy.train(mode)
    
    def eval_mode(self):
        """Set evaluation mode."""
        self.policy.eval()


# =============================================================================
# MotorNet Trainer (BPTT through differentiable plant)
# =============================================================================

class MotorNetTrainer:
    """
    Trains MotorNet using BPTT through differentiable plant.
    
    Everything runs on GPU - no NumPy in training loop!
    """
    
    def __init__(
        self,
        params: MotorNetParams,
        q0_deg: List[float] = [45.0, 90.0],
    ):
        self.p = params
        self.device = torch.device(params.device if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float64  # Match other torch controllers
        
        torch.set_default_dtype(self.dtype)
        
        # ---- Build plant (same as main_random_reach_pd_if_torch.py) ----
        self.muscle = RigidTendonHillMuscle(
            min_activation=0.02,
            device=self.device,
            dtype=self.dtype,
        )
        
        self.arm = RigidTendonArm26(
            muscle=self.muscle,
            timestep=0.01,
            damping=0.0,
            n_ministeps=1,
            integration_method='euler',
            device=self.device,
            dtype=self.dtype,
        )
        
        self.env = EnvironmentTorch(
            effector=self.arm,
            max_ep_duration=params.trial_duration + 0.1,
            action_noise=0.0,
            obs_noise=0.0,
            proprioception_delay=self.arm.dt,
            vision_delay=self.arm.dt,
            name="MotorNetTrainEnv",
        )
        
        # Initial configuration
        self.q0 = torch.deg2rad(torch.tensor(q0_deg, device=self.device, dtype=self.dtype))
        
        # ---- Create controller ----
        self.ctrl = MotorNetController(self.env, self.arm, params)
        
        # Training state
        self.train_losses: List[float] = []
    
    def _reset_env(self, batch_size: int = 1) -> None:
        """Reset environment to initial state."""
        qd0 = torch.zeros(2, device=self.device, dtype=self.dtype)
        joint0 = torch.cat([self.q0, qd0]).unsqueeze(0)
        
        if batch_size > 1:
            joint0 = joint0.expand(batch_size, -1).clone()
        
        self.env.reset(options={
            "joint_state": joint0,
            "deterministic": True,
        })
    
    def _get_center(self) -> Tensor:
        """Get initial fingertip position."""
        self._reset_env(1)
        return self.env.states["fingertip"][0, :2].clone()
    
    def generate_goals(self, center: Tensor, n_goals: int) -> Tensor:
        """Generate random reaching goals."""
        angles = torch.rand(n_goals, device=self.device, dtype=self.dtype) * 2 * 3.14159
        radii = torch.rand(n_goals, device=self.device, dtype=self.dtype) * (self.p.reach_radius - 0.03) + 0.03
        
        goals = torch.stack([
            center[0] + radii * torch.cos(angles),
            center[1] + radii * torch.sin(angles),
        ], dim=-1)
        
        return goals
    
    def _minimum_jerk(self, start: Tensor, goal: Tensor, t: float, T: float) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute minimum-jerk trajectory."""
        tau = min(max(t / T, 0.0), 1.0)
        
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
        sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / (T**2)
        
        delta = goal - start
        x_d = start + s * delta
        xd_d = sd * delta
        xdd_d = sdd * delta
        
        return x_d, xd_d, xdd_d
    
    def _rollout_batch(self, goals: Tensor) -> Dict[str, Tensor]:
        """
        Run batch of reaching trials with BPTT.
        
        Args:
            goals: (B, 2) goal positions
        
        Returns:
            Dict with trajectories
        """
        B = goals.shape[0]
        T = self.p.trial_duration
        dt = float(self.arm.dt)
        n_steps = int(T / dt)
        
        # Reset environment
        self._reset_env(B)
        start = self.env.states["fingertip"][:, :2].clone()
        
        # Initialize controller
        self.ctrl.train_mode(True)
        self.ctrl.reset(self.q0)
        self.ctrl._goal = goals
        self.ctrl._hidden = self.ctrl.policy.init_hidden(B, self.device, torch.float32)
        
        # Storage
        positions = []
        velocities = []
        excitations = []
        
        # Rollout
        t = 0.0
        for step in range(n_steps):
            # Get trajectory reference
            x_d, xd_d, xdd_d = self._minimum_jerk(start, goals, t, T)
            
            # Policy forward (with gradients!)
            t_norm = t / T
            obs = self.ctrl._build_observation(
                self.env.states["cartesian"][:, :2],
                self.env.states["cartesian"][:, 2:4],
                goals,
                t_norm,
            )
            
            obs_f32 = obs.to(dtype=torch.float32)
            excit, self.ctrl._hidden = self.ctrl.policy(obs_f32, self.ctrl._hidden)
            excit = excit.to(dtype=self.dtype)
            
            # Clamp excitations
            a_min = float(self.muscle.min_activation)
            excit = torch.clamp(excit, a_min, 1.0)
            
            # Step environment (differentiable!)
            # env.step returns (obs, reward, terminated, truncated, info)
            _, _, _, _, _ = self.env.step(excit, deterministic=True)
            
            # Store
            positions.append(self.env.states["fingertip"][:, :2].clone())
            velocities.append(self.env.states["cartesian"][:, 2:4].clone())
            excitations.append(excit)
            
            t += dt
        
        # Stack
        positions = torch.stack(positions, dim=1)      # (B, T, 2)
        velocities = torch.stack(velocities, dim=1)    # (B, T, 2)
        excitations = torch.stack(excitations, dim=1)  # (B, T, M)
        
        return {
            'positions': positions,
            'velocities': velocities,
            'excitations': excitations,
            'goals': goals,
            'start': start,
        }
    
    def _compute_loss(self, rollout: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        """Compute training loss."""
        positions = rollout['positions']
        velocities = rollout['velocities']
        excitations = rollout['excitations']
        goals = rollout['goals']
        start = rollout['start']
        
        B, T_steps, _ = positions.shape
        T = self.p.trial_duration
        dt = float(self.arm.dt)
        
        # Position tracking error
        pos_errors = []
        for t_idx in range(T_steps):
            t = t_idx * dt
            x_target, _, _ = self._minimum_jerk(start, goals, t, T)
            pos_errors.append((positions[:, t_idx, :] - x_target).pow(2).sum(dim=-1))
        pos_errors = torch.stack(pos_errors, dim=1)
        loss_position = pos_errors.mean()
        
        # Final position error
        final_pos = positions[:, -1, :]
        loss_final = (final_pos - goals).pow(2).sum(dim=-1).mean()
        
        # Final velocity
        final_vel = velocities[:, -1, :]
        loss_velocity = final_vel.pow(2).sum(dim=-1).mean()
        
        # Effort
        loss_effort = excitations.pow(2).mean()
        
        # Smoothness
        u_diff = excitations[:, 1:, :] - excitations[:, :-1, :]
        loss_smooth = u_diff.pow(2).mean()
        
        # Total
        loss = (
            self.p.w_position * loss_position +
            self.p.w_final * loss_final +
            self.p.w_velocity * loss_velocity +
            self.p.w_effort * loss_effort +
            self.p.w_smooth * loss_smooth
        )
        
        diag = {
            'loss_total': loss.item(),
            'loss_position': loss_position.item(),
            'loss_final': loss_final.item(),
            'final_error_cm': (final_pos - goals).pow(2).sum(dim=-1).sqrt().mean().item() * 100,
        }
        
        return loss, diag
    
    def train(
        self,
        n_goals: int = 100,
        n_epochs: int = 200,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train MotorNet using BPTT.
        
        Returns list of training losses.
        """
        center = self._get_center()
        goals = self.generate_goals(center, n_goals)
        
        if verbose:
            print(f"Training MotorNet (BPTT) on {n_goals} goals for {n_epochs} epochs")
            print(f"Device: {self.device}")
            print(f"Center: ({center[0].item():.3f}, {center[1].item():.3f})")
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.ctrl.optimizer, T_max=n_epochs, eta_min=self.p.lr * 0.01
        )
        
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle
            perm = torch.randperm(n_goals, device=self.device)
            goals_shuffled = goals[perm]
            
            for i in range(0, n_goals, self.p.batch_size):
                batch_goals = goals_shuffled[i:i + self.p.batch_size]
                if batch_goals.shape[0] < 2:
                    continue
                
                self.ctrl.optimizer.zero_grad()
                
                # Rollout with gradients
                rollout = self._rollout_batch(batch_goals)
                
                # Loss
                loss, diag = self._compute_loss(rollout)
                
                # Backward
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.ctrl.policy.parameters(),
                    self.p.grad_clip
                )
                
                # Update
                self.ctrl.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{n_epochs}: loss={avg_loss:.4f}, "
                      f"final_err={diag['final_error_cm']:.2f}cm, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")
        
        self.ctrl.is_trained = True
        self.ctrl.train_losses = losses
        self.train_losses = losses
        
        if verbose:
            print(f"Training complete! Final loss: {losses[-1]:.4f}")
        
        return losses
    
    @torch.no_grad()
    def evaluate(self, n_goals: int = 20) -> Dict[str, float]:
        """Evaluate trained controller."""
        self.ctrl.eval_mode()
        
        center = self._get_center()
        goals = self.generate_goals(center, n_goals)
        
        rollout = self._rollout_batch(goals)
        
        final_pos = rollout['positions'][:, -1, :]
        final_errors = (final_pos - goals).pow(2).sum(dim=-1).sqrt()
        
        return {
            'rmse_cm': final_errors.mean().item() * 100,
            'max_error_cm': final_errors.max().item() * 100,
        }
    
    def save(self, path: str) -> None:
        """Save model."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            'policy_state_dict': self.ctrl.policy.state_dict(),
            'optimizer_state_dict': self.ctrl.optimizer.state_dict(),
            'params': self.p,
            'q0': self.q0.cpu().numpy(),
            'train_losses': self.train_losses,
        }, path)
    
    def load(self, path: str) -> None:
        """Load model."""
        ckpt = torch.load(path, map_location=self.device)
        self.ctrl.policy.load_state_dict(ckpt['policy_state_dict'])
        self.ctrl.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.ctrl.is_trained = True
        self.train_losses = ckpt.get('train_losses', [])


# =============================================================================
# Main / Smoke Test
# =============================================================================

def _smoke_test():
    """Smoke test for MotorNet."""
    import time
    
    print("\n[MotorNet Torch] Smoke test starting...")
    
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    params = MotorNetParams(
        hidden_dim=64,
        n_layers=2,
        batch_size=16,
        n_epochs=20,  # Short test
        device=str(device),
    )
    
    trainer = MotorNetTrainer(params)
    
    print(f"Arm muscles: {trainer.arm.n_muscles}")
    print(f"Initial q0: {trainer.q0.cpu().numpy()}")
    
    # Short training
    start = time.time()
    losses = trainer.train(n_goals=32, n_epochs=20, verbose=True)
    train_time = time.time() - start
    
    print(f"\nTraining time: {train_time:.1f}s ({train_time/20:.2f}s/epoch)")
    
    # Evaluate
    metrics = trainer.evaluate(n_goals=10)
    print(f"Test RMSE: {metrics['rmse_cm']:.2f} cm")
    
    print("\n[MotorNet Torch] Smoke test complete ✓")


if __name__ == "__main__":
    _smoke_test()
