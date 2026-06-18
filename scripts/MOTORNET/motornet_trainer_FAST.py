#!/usr/bin/env python3
# motornet_trainer_FAST.py
"""
FAST MotorNet trainer using analytic dynamics.

Uses FastRigidTendonArm with analytic M, C, g computation
instead of the slow HTM-based dynamics.

Expected speedup: 50-100x faster than original.

Usage:
    python -m scripts.MOTORNET.motornet_trainer_FAST --epochs 200 --n_goals 100
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

# Add project root
script_dir = Path(__file__).parent.resolve()
project_root = script_dir
for _ in range(5):
    if (project_root / 'model_lib').exists():
        break
    project_root = project_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import FAST components
from model_lib.skeleton_torch_FAST import TwoDofArm
from model_lib.effector_torch_FAST import FastRigidTendonArm


# =============================================================================
# GRU Policy Network
# =============================================================================

class GRUPolicy(nn.Module):
    """
    GRU policy network for MotorNet.
    
    Input: observation (position, velocity, goal, error, time)
    Output: muscle excitations
    """
    
    def __init__(
        self,
        obs_dim: int = 9,
        n_muscles: int = 6,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.n_muscles = n_muscles
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # GRU with float32 for numerical stability
        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, n_muscles)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(
        self, 
        obs: Tensor, 
        hidden: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            obs: (B, obs_dim) observation
            hidden: (n_layers, B, hidden_dim) GRU hidden state
            
        Returns:
            excitation: (B, n_muscles) muscle excitations in [0, 1]
            hidden: (n_layers, B, hidden_dim) new hidden state
        """
        B = obs.shape[0]
        
        # Convert to float32 for GRU
        obs_f32 = obs.float()
        
        if hidden is None:
            hidden = torch.zeros(
                self.n_layers, B, self.hidden_dim,
                device=obs.device, dtype=torch.float32
            )
        
        # GRU expects (B, seq_len, input_size)
        obs_seq = obs_f32.unsqueeze(1)  # (B, 1, obs_dim)
        
        out, hidden = self.gru(obs_seq, hidden)  # out: (B, 1, hidden)
        out = out.squeeze(1)  # (B, hidden)
        
        # Output
        excitation = torch.sigmoid(self.fc_out(out))  # (B, n_muscles) in [0, 1]
        
        # Convert back to original dtype
        excitation = excitation.to(dtype=obs.dtype)
        
        return excitation, hidden
    
    def get_initial_hidden(self, batch_size: int, device: torch.device) -> Tensor:
        """Get initial hidden state."""
        return torch.zeros(
            self.n_layers, batch_size, self.hidden_dim,
            device=device, dtype=torch.float32
        )


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""
    # Network
    hidden_dim: int = 128
    n_layers: int = 2
    
    # Training
    n_goals: int = 100
    n_epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    
    # Simulation
    dt: float = 0.01
    episode_duration: float = 0.6
    
    # Loss weights
    w_pos: float = 100.0     # Position tracking
    w_final: float = 200.0   # Final position
    w_vel: float = 10.0      # Final velocity
    w_effort: float = 0.01   # Effort penalty
    w_smooth: float = 0.1    # Smoothness penalty
    
    # Device
    device: str = "cuda"
    
    # Initial state
    q0_deg: Tuple[float, float] = (45.0, 90.0)
    
    # Output
    save_path: str = "models/motornet_fast.pt"


# =============================================================================
# Trainer
# =============================================================================

class MotorNetTrainerFast:
    """
    FAST MotorNet trainer using analytic dynamics.
    """
    
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        
        # Create arm
        self.arm = FastRigidTendonArm(
            timestep=config.dt,
            damping=0.05,
            device=self.device,
            dtype=self.dtype,
        )
        
        self.n_muscles = self.arm.n_muscles
        self.n_steps = int(config.episode_duration / config.dt)
        
        # Create policy
        self.policy = GRUPolicy(
            obs_dim=9,
            n_muscles=self.n_muscles,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
        ).to(device=self.device, dtype=torch.float32)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.n_epochs,
            eta_min=1e-6,
        )
        
        # Initial state
        q0 = torch.deg2rad(torch.tensor(config.q0_deg, device=self.device, dtype=self.dtype))
        qd0 = torch.zeros(2, device=self.device, dtype=self.dtype)
        self.initial_joint = torch.cat([q0, qd0])
        
        # Workspace center (approximate)
        with torch.no_grad():
            self.arm.reset(options={"batch_size": 1, "joint_state": self.initial_joint.unsqueeze(0)})
            self.workspace_center = self.arm.states["fingertip"][0, :2].clone()
    
    def _generate_goals(self, n_goals: int) -> Tensor:
        """Generate random reaching goals."""
        # Sample in a circle around workspace center
        radius = 0.15  # 15cm radius
        
        angles = torch.rand(n_goals, device=self.device, dtype=self.dtype) * 2 * torch.pi
        radii = torch.sqrt(torch.rand(n_goals, device=self.device, dtype=self.dtype)) * radius
        
        dx = radii * torch.cos(angles)
        dy = radii * torch.sin(angles)
        
        goals = self.workspace_center.unsqueeze(0) + torch.stack([dx, dy], dim=-1)
        
        return goals
    
    def _minimum_jerk(
        self, 
        start: Tensor, 
        goal: Tensor, 
        t: float, 
        T: float
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute minimum-jerk trajectory.
        
        Args:
            start: (B, 2) starting position
            goal: (B, 2) goal position
            t: current time
            T: total duration
            
        Returns:
            x_d, xd_d, xdd_d: desired position, velocity, acceleration
        """
        tau = t / T
        tau = min(max(tau, 0.0), 1.0)
        
        # Minimum jerk polynomial
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
        sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / (T**2)
        
        delta = goal - start
        
        x_d = start + s * delta
        xd_d = sd * delta
        xdd_d = sdd * delta
        
        return x_d, xd_d, xdd_d
    
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
            x: (B, 2) current position
            xd: (B, 2) current velocity
            goal: (B, 2) goal position
            t_norm: normalized time [0, 1]
            
        Returns:
            obs: (B, 9) observation
        """
        B = x.shape[0]
        error = goal - x
        t_tensor = torch.full((B, 1), t_norm, device=self.device, dtype=self.dtype)
        
        obs = torch.cat([x, xd, goal, error, t_tensor], dim=-1)  # (B, 9)
        
        return obs
    
    def _rollout(
        self,
        goals: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Run a batched rollout.
        
        Args:
            goals: (B, 2) goal positions
            
        Returns:
            Dictionary with positions, excitations, losses
        """
        B = goals.shape[0]
        T = self.config.episode_duration
        
        # Reset arm
        self.arm.reset(options={
            "batch_size": B,
            "joint_state": self.initial_joint.unsqueeze(0),
        })
        
        # Get starting position
        start = self.arm.states["fingertip"][:, :2].clone()
        
        # Initialize policy hidden state
        hidden = self.policy.get_initial_hidden(B, self.device)
        
        # Storage
        positions = []
        excitations = []
        
        # Rollout
        for step in range(self.n_steps):
            t = step * self.config.dt
            t_norm = t / T
            
            # Get current state
            x = self.arm.states["fingertip"][:, :2]
            xd = self.arm.states["fingertip"][:, 2:4]
            
            # Build observation
            obs = self._build_observation(x, xd, goals, t_norm)
            
            # Get action from policy
            excit, hidden = self.policy(obs, hidden)
            
            # Step arm
            self.arm.step(excit)
            
            # Store
            positions.append(self.arm.states["fingertip"][:, :2].clone())
            excitations.append(excit)
        
        # Stack
        positions = torch.stack(positions, dim=1)  # (B, T, 2)
        excitations = torch.stack(excitations, dim=1)  # (B, T, M)
        
        return {
            "positions": positions,
            "excitations": excitations,
            "start": start,
            "goals": goals,
        }
    
    def _compute_loss(
        self,
        rollout: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute training loss.
        
        Returns:
            loss: scalar loss
            metrics: dictionary of loss components
        """
        positions = rollout["positions"]  # (B, T, 2)
        excitations = rollout["excitations"]  # (B, T, M)
        start = rollout["start"]  # (B, 2)
        goals = rollout["goals"]  # (B, 2)
        
        B, T, _ = positions.shape
        
        # Generate reference trajectory
        ref_positions = []
        for step in range(T):
            t = step * self.config.dt
            x_d, _, _ = self._minimum_jerk(start, goals, t, self.config.episode_duration)
            ref_positions.append(x_d)
        ref_positions = torch.stack(ref_positions, dim=1)  # (B, T, 2)
        
        # Position tracking loss
        pos_error = positions - ref_positions
        L_pos = self.config.w_pos * (pos_error ** 2).mean()
        
        # Final position loss
        final_error = positions[:, -1, :] - goals
        L_final = self.config.w_final * (final_error ** 2).mean()
        
        # Final velocity loss (want to stop at goal)
        # Approximate velocity from last two positions
        final_vel = (positions[:, -1, :] - positions[:, -2, :]) / self.config.dt
        L_vel = self.config.w_vel * (final_vel ** 2).mean()
        
        # Effort loss
        L_effort = self.config.w_effort * (excitations ** 2).mean()
        
        # Smoothness loss
        exc_diff = excitations[:, 1:, :] - excitations[:, :-1, :]
        L_smooth = self.config.w_smooth * (exc_diff ** 2).mean()
        
        # Total loss
        loss = L_pos + L_final + L_vel + L_effort + L_smooth
        
        # Metrics
        final_dist = torch.sqrt((final_error ** 2).sum(dim=-1)).mean()
        
        metrics = {
            "loss": loss.item(),
            "L_pos": L_pos.item(),
            "L_final": L_final.item(),
            "L_vel": L_vel.item(),
            "L_effort": L_effort.item(),
            "L_smooth": L_smooth.item(),
            "final_dist_cm": final_dist.item() * 100,
        }
        
        return loss, metrics
    
    def train(self):
        """Run training."""
        print(f"Training MotorNet (FAST) on {self.config.n_goals} goals for {self.config.n_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Workspace center: ({self.workspace_center[0]:.3f}, {self.workspace_center[1]:.3f})")
        print()
        
        # Generate training goals
        all_goals = self._generate_goals(self.config.n_goals)
        
        n_batches = (self.config.n_goals + self.config.batch_size - 1) // self.config.batch_size
        
        best_loss = float('inf')
        
        for epoch in range(1, self.config.n_epochs + 1):
            epoch_loss = 0.0
            epoch_dist = 0.0
            
            # Shuffle goals
            perm = torch.randperm(self.config.n_goals, device=self.device)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, self.config.n_goals)
                batch_indices = perm[start_idx:end_idx]
                goals = all_goals[batch_indices]
                
                # Forward pass
                self.optimizer.zero_grad()
                rollout = self._rollout(goals)
                loss, metrics = self._compute_loss(rollout)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.grad_clip
                )
                
                # Update
                self.optimizer.step()
                
                epoch_loss += metrics["loss"]
                epoch_dist += metrics["final_dist_cm"]
            
            # Scheduler step
            self.scheduler.step()
            
            # Average metrics
            epoch_loss /= n_batches
            epoch_dist /= n_batches
            
            # Log
            if epoch % 10 == 0 or epoch == 1:
                lr = self.scheduler.get_last_lr()[0]
                print(f"Epoch {epoch:3d}/{self.config.n_epochs}: "
                      f"loss={epoch_loss:.4f}, "
                      f"final_err={epoch_dist:.2f}cm, "
                      f"lr={lr:.2e}")
            
            # Save best
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self._save_checkpoint(epoch, epoch_loss, epoch_dist)
        
        print(f"\nTraining complete. Best loss: {best_loss:.4f}")
        print(f"Model saved to: {self.config.save_path}")
    
    def _save_checkpoint(self, epoch: int, loss: float, dist: float):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(self.config.save_path) or ".", exist_ok=True)
        
        torch.save({
            "epoch": epoch,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "final_dist_cm": dist,
            "config": self.config,
        }, self.config.save_path)


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="FAST MotorNet Training")
    parser.add_argument("--n_goals", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="models/motornet_fast.pt")
    parser.add_argument("--q0", type=float, nargs=2, default=[45.0, 90.0])
    args = parser.parse_args()
    
    print("=" * 70)
    print("FAST MotorNet Training (Analytic Dynamics)")
    print("=" * 70)
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    config = TrainConfig(
        n_goals=args.n_goals,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        lr=args.lr,
        device=args.device,
        save_path=args.out,
        q0_deg=tuple(args.q0),
    )
    
    print(f"\nConfiguration:")
    print(f"  Goals: {config.n_goals}")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  GRU layers: {config.n_layers}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Initial joint angles: {config.q0_deg} deg")
    print()
    
    torch.set_default_dtype(torch.float64)
    
    trainer = MotorNetTrainerFast(config)
    
    t0 = time.perf_counter()
    trainer.train()
    t_total = time.perf_counter() - t0
    
    print(f"\nTotal training time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"Time per epoch: {t_total/config.n_epochs:.2f}s")


if __name__ == "__main__":
    main()
