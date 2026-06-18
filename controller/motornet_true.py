# controller/motornet_true.py
"""
TRUE MotorNet: Differentiable Optimal Control for Musculoskeletal Systems

This implementation follows the actual MotorNet framework (Codol et al. 2024):
- The effector (arm + muscles) is implemented in PyTorch and is FULLY DIFFERENTIABLE
- Training uses BPTT (backpropagation through time) through the ACTUAL plant dynamics
- No toy surrogate dynamics - gradients flow through real M(q), J(q), muscle F-L-V curves

Key difference from previous "motornet_fixed.py":
- OLD: Training used toy point-mass dynamics (xdd = 50*e_x + muscle_force)
- NEW: Training uses actual RigidTendonArm26 torch effector with real dynamics

Two training modes:
1. TRUE MotorNet (Option 2): BPTT through differentiable plant
2. Behavioral Cloning (Option 1): Imitation learning from expert (OSC)

References:
- Codol et al. (2024): MotorNet, a Python toolbox for controlling differentiable
  biomechanical effectors with artificial neural networks
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import math

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# GRU Policy Network
# =============================================================================

if TORCH_AVAILABLE:
    class GRUPolicy(nn.Module):
        """
        GRU-based policy network for muscle control.
        
        Input: [x, y, xd, yd, goal_x, goal_y, e_x, e_y, t_norm] = 9 dims
        Output: 6 muscle excitations in [0, 1]
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
            
            # Input normalization
            self.input_norm = nn.LayerNorm(input_dim)
            
            # GRU
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0,
            )
            
            # Output MLP
            self.output_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, output_dim),
                nn.Sigmoid(),  # Excitations in [0, 1]
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
            
            for layer in self.output_net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        def forward(
            self,
            x: Tensor,
            h: Optional[Tensor] = None,
        ) -> Tuple[Tensor, Tensor]:
            """
            Forward pass.
            
            Args:
                x: Input (batch, seq_len, input_dim) or (batch, input_dim)
                h: Hidden state (n_layers, batch, hidden_dim)
            
            Returns:
                u: Excitations (batch, seq_len, output_dim)
                h_new: New hidden state
            """
            if x.dim() == 2:
                x = x.unsqueeze(1)
            
            batch_size = x.size(0)
            
            if h is None:
                h = torch.zeros(
                    self.n_layers, batch_size, self.hidden_dim,
                    device=x.device, dtype=x.dtype
                )
            
            x_norm = self.input_norm(x)
            gru_out, h_new = self.gru(x_norm, h)
            u = self.output_net(gru_out)
            
            return u, h_new
        
        def init_hidden(self, batch_size: int = 1, device='cpu') -> Tensor:
            return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)


# =============================================================================
# Parameters
# =============================================================================

@dataclass
class MotorNetTrueParams:
    """Parameters for TRUE MotorNet controller."""
    
    # Network architecture
    hidden_dim: int = 128
    n_layers: int = 2
    
    # Training mode: 'bptt' (true MotorNet) or 'bc' (behavioral cloning)
    training_mode: str = 'bptt'
    
    # BPTT training parameters
    bptt_epochs: int = 200
    bptt_lr: float = 1e-3
    bptt_trials_per_epoch: int = 32
    bptt_trial_steps: int = 60  # Steps per trial (60 * 0.01 = 0.6s)
    
    # BC training parameters
    bc_epochs: int = 200
    bc_lr: float = 1e-3
    bc_batch_size: int = 64
    n_expert_demos: int = 200
    
    # Cost weights (for BPTT)
    w_position: float = 1.0
    w_velocity: float = 0.1
    w_effort: float = 0.001
    w_smoothness: float = 0.01
    w_final: float = 5.0
    
    # Task parameters
    reach_radius: float = 0.10
    trial_duration: float = 0.6
    
    # Arm parameters (for torch effector)
    timestep: float = 0.01
    damping: float = 0.05
    
    # Device
    device: str = 'cpu'


# =============================================================================
# TRUE MotorNet Controller
# =============================================================================

class MotorNetTrue:
    """
    TRUE MotorNet Controller with Differentiable Training.
    
    This implementation:
    1. Uses PyTorch effector (RigidTendonArm26) for BPTT training
    2. Backpropagates through REAL arm dynamics (M(q), J(q), muscle model)
    3. Falls back to NumPy OSC for deployment if needed
    
    Training modes:
    - 'bptt': TRUE MotorNet - backprop through differentiable plant
    - 'bc': Behavioral Cloning - imitate expert (OSC) demonstrations
    """
    
    def __init__(self, params: Optional[MotorNetTrueParams] = None):
        self.p = params or MotorNetTrueParams()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for MotorNet. Install with: pip install torch")
        
        self.device = torch.device(self.p.device)
        self.n_muscles = 6
        self.n_dof = 2
        
        # Create policy network
        self.policy = GRUPolicy(
            input_dim=9,
            hidden_dim=self.p.hidden_dim,
            output_dim=self.n_muscles,
            n_layers=self.p.n_layers,
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=self.p.bptt_lr,
            weight_decay=1e-5,
        )
        
        # State
        self.hidden = None
        self.trained = False
        self.goal = None
        self.t_current = 0.0
        
        # Torch effector (created on first use)
        self._torch_effector = None
    
    def _create_torch_effector(self):
        """Create the differentiable PyTorch arm effector."""
        from model_lib.effector_torch import RigidTendonArm26
        from model_lib.muscles_torch import ReluMuscle
        from model_lib.skeleton_torch import TwoDofArm
        
        skeleton = TwoDofArm(
            m1=1.82, m2=1.43,
            l1g=0.135, l2g=0.165,
            i1=0.051, i2=0.057,
            l1=0.309, l2=0.333,
            device=self.device,
            dtype=torch.float32,
        )
        
        muscle = ReluMuscle(device=self.device, dtype=torch.float32)
        
        effector = RigidTendonArm26(
            muscle=muscle,
            skeleton=skeleton,
            timestep=self.p.timestep,
            damping=self.p.damping,
            device=self.device,
            dtype=torch.float32,
        )
        
        return effector
    
    @property
    def torch_effector(self):
        """Lazy initialization of torch effector."""
        if self._torch_effector is None:
            self._torch_effector = self._create_torch_effector()
        return self._torch_effector
    
    def reset(self, q0: np.ndarray = None):
        """Reset controller state."""
        self.hidden = self.policy.init_hidden(1, self.device)
        self.t_current = 0.0
        self.goal = None
        
        if q0 is not None:
            self.q0 = np.asarray(q0, dtype=np.float32)
        else:
            self.q0 = np.array([0.8, 2.1], dtype=np.float32)  # Default
    
    def set_goal(self, goal: np.ndarray, T_trial: float = 0.6):
        """Set goal position for reaching."""
        self.goal = np.asarray(goal, dtype=np.float32)
        self.T_trial = T_trial
        self.t_current = 0.0
    
    def _build_observation(
        self,
        x: Tensor,
        xd: Tensor,
        goal: Tensor,
        t_norm: float,
    ) -> Tensor:
        """Build observation tensor for policy."""
        e_x = goal - x
        t_tensor = torch.tensor([t_norm], device=self.device, dtype=x.dtype)
        obs = torch.cat([x.squeeze(), xd.squeeze(), goal.squeeze(), e_x.squeeze(), t_tensor])
        return obs.unsqueeze(0)  # (1, 9)
    
    # =========================================================================
    # TRUE MotorNet Training (BPTT through real plant)
    # =========================================================================
    
    def train_bptt(
        self,
        goals: np.ndarray,
        n_epochs: int = None,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train using BPTT through differentiable plant (TRUE MotorNet).
        
        This is the core MotorNet training:
        - Unroll the REAL arm dynamics in PyTorch
        - Compute task loss over trajectory
        - Backpropagate gradients through entire simulation
        
        Args:
            goals: Array of goal positions (N, 2)
            n_epochs: Number of training epochs
            verbose: Print progress
        
        Returns:
            List of training losses
        """
        n_epochs = n_epochs or self.p.bptt_epochs
        
        if verbose:
            print("="*60)
            print("TRUE MotorNet Training (BPTT through differentiable plant)")
            print("="*60)
        
        effector = self.torch_effector
        self.policy.train()
        
        losses = []
        
        # Initial joint state
        q0_tensor = torch.tensor(
            [self.q0[0], self.q0[1], 0.0, 0.0],
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        for epoch in range(n_epochs):
            print("epoch",epoch)
            epoch_loss = 0.0
            n_trials = min(len(goals), self.p.bptt_trials_per_epoch)
            
            # Shuffle goals
            indices = np.random.choice(len(goals), n_trials, replace=False)
            
            for idx in indices:
                print("idx",idx)
                goal = torch.tensor(goals[idx], dtype=torch.float32, device=self.device)
                
                self.optimizer.zero_grad()
                
                # Reset effector
                effector.reset(options={'joint_state': q0_tensor})
                h = self.policy.init_hidden(1, self.device)
                
                # Storage for trajectory
                positions = []
                velocities = []
                excitations = []
                
                # Rollout through REAL differentiable dynamics
                for t in range(self.p.bptt_trial_steps):
                    t_norm = t / self.p.bptt_trial_steps
                    
                    # Get current state from effector
                    x = effector.states['fingertip'].squeeze()  # (2,)
                    joint = effector.states['joint'].squeeze()  # (4,)
                    xd = self._compute_endpoint_velocity(effector)  # (2,)
                    
                    # Build observation
                    obs = self._build_observation(x, xd, goal, t_norm)
                    
                    # Get action from policy
                    u, h = self.policy(obs, h)
                    u = u.squeeze()  # (6,)
                    
                    # Step the REAL differentiable effector
                    # This is where gradients flow through M(q), J(q), muscle model!
                    action = u.unsqueeze(0).unsqueeze(0)  # (1, 1, 6)
                    effector.step(action)
                    
                    # Store for loss computation
                    positions.append(effector.states['fingertip'].squeeze())
                    velocities.append(self._compute_endpoint_velocity(effector))
                    excitations.append(u)
                
                # Compute loss
                loss = self._compute_bptt_loss(
                    positions, velocities, excitations, goal
                )
                
                # Backpropagate through entire trajectory!
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / n_trials
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        self.policy.eval()
        self.trained = True
        
        if verbose:
            print(f"BPTT Training complete. Final loss: {losses[-1]:.4f}")
        
        return losses
    
    def _compute_endpoint_velocity(self, effector) -> Tensor:
        """Compute endpoint velocity from cartesian state."""
        # The effector's cartesian state is [x, y, xd, yd]
        # Updated by skeleton.joint2cartesian which uses J @ qd
        cartesian = effector.states.get('cartesian')
        if cartesian is not None and cartesian.shape[-1] >= 4:
            return cartesian[:, 2:4].squeeze()  # (2,)
        
        # Fallback if cartesian not available
        joint = effector.states['joint']
        qd = joint[:, 2:]
        return qd.squeeze() * 0.3  # Rough approximation
    
    def _compute_bptt_loss(
        self,
        positions: List[Tensor],
        velocities: List[Tensor],
        excitations: List[Tensor],
        goal: Tensor,
    ) -> Tensor:
        """Compute task loss for BPTT."""
        T = len(positions)
        
        # Stack trajectories
        x_traj = torch.stack(positions)  # (T, 2)
        xd_traj = torch.stack(velocities)  # (T, 2)
        u_traj = torch.stack(excitations)  # (T, 6)
        
        # Position error (weighted more at end)
        time_weights = torch.linspace(0.5, 1.5, T, device=self.device)
        pos_error = torch.sum((x_traj - goal) ** 2, dim=1)  # (T,)
        L_pos = self.p.w_position * torch.mean(time_weights * pos_error)
        
        # Final position error (extra penalty)
        L_final = self.p.w_final * torch.sum((x_traj[-1] - goal) ** 2)
        
        # Final velocity should be zero
        L_vel = self.p.w_velocity * torch.sum(xd_traj[-1] ** 2)
        
        # Effort (metabolic cost proxy)
        L_effort = self.p.w_effort * torch.mean(u_traj ** 2)
        
        # Smoothness (excitation changes)
        du = u_traj[1:] - u_traj[:-1]
        L_smooth = self.p.w_smoothness * torch.mean(du ** 2)
        
        return L_pos + L_final + L_vel + L_effort + L_smooth
    
    # =========================================================================
    # Behavioral Cloning Training (Option 1)
    # =========================================================================
    
    def train_bc(
        self,
        env_numpy,
        arm_numpy,
        n_demos: int = None,
        n_epochs: int = None,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train using Behavioral Cloning from expert (OSC) demonstrations.
        
        This is NOT true MotorNet, but a practical alternative:
        - Collect expert trajectories using OSC on real (NumPy) environment
        - Train policy to imitate expert actions
        - Deploy learned policy
        
        Args:
            env_numpy: NumPy environment for expert demonstrations
            arm_numpy: NumPy arm for expert demonstrations
            n_demos: Number of expert demonstrations to collect
            n_epochs: Number of training epochs
            verbose: Print progress
        
        Returns:
            List of training losses
        """
        n_demos = n_demos or self.p.n_expert_demos
        n_epochs = n_epochs or self.p.bc_epochs
        
        if verbose:
            print("="*60)
            print("Behavioral Cloning Training (Imitation Learning)")
            print("="*60)
        
        # Collect expert demonstrations
        observations, expert_actions = self._collect_expert_demos(
            env_numpy, arm_numpy, n_demos, verbose
        )
        
        # Convert to tensors
        obs_tensor = torch.tensor(observations, dtype=torch.float32, device=self.device)
        act_tensor = torch.tensor(expert_actions, dtype=torch.float32, device=self.device)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(obs_tensor, act_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.p.bc_batch_size, shuffle=True
        )
        
        # Training
        self.policy.train()
        criterion = nn.MSELoss()
        
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for obs_batch, act_batch in dataloader:
                self.optimizer.zero_grad()
                
                # Forward (no hidden state for BC)
                pred_act, _ = self.policy(obs_batch, None)
                pred_act = pred_act.squeeze(1)
                
                loss = criterion(pred_act, act_batch)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
        
        self.policy.eval()
        self.trained = True
        
        if verbose:
            print(f"BC Training complete. Final loss: {losses[-1]:.6f}")
        
        return losses
    
    def _collect_expert_demos(
        self,
        env_numpy,
        arm_numpy,
        n_demos: int,
        verbose: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect expert demonstrations using OSC controller."""
        from controller.osc_controller import OSCController, OSCParams
        from trajectory.minjerk_numpy import MinJerkLinearTrajectory, MinJerkParams
        from config import TrajectoryConfig
        
        if verbose:
            print(f"Collecting {n_demos} expert demonstrations...")
        
        tc = TrajectoryConfig()
        osc = OSCController(env_numpy, arm_numpy, OSCParams(Kp=2000.0, Kv=100.0))
        
        all_obs = []
        all_act = []
        
        # Get initial position
        q0 = self.q0
        env_numpy.reset(options={
            'joint_state': np.concatenate([q0, np.zeros(2)])[None, :],
            'deterministic': True
        })
        center = env_numpy.states['cartesian'][0, :2].copy()
        
        for i in range(n_demos):
            # Random goal
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0.05, self.p.reach_radius)
            goal = center + radius * np.array([np.cos(angle), np.sin(angle)])
            
            # Reset
            env_numpy.reset(options={
                'joint_state': np.concatenate([q0, np.zeros(2)])[None, :],
                'deterministic': True
            })
            osc.reset(q0)
            
            # Create trajectory
            start = env_numpy.states['cartesian'][0, :2].copy()
            waypoints = [
                {'xy': start, 't_hold': 0.0},
                {'xy': goal, 't_hold': 0.1},
            ]
            traj = MinJerkLinearTrajectory(
                waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
            )
            
            # Collect trajectory
            t = 0.0
            T = self.p.trial_duration
            while t < T:
                x_d, xd_d, xdd_d = traj.sample(t)
                
                # Get expert action
                result = osc.compute(x_d, xd_d, xdd_d)
                expert_act = result['act']
                
                # Build observation
                cart = env_numpy.states['cartesian'][0]
                x, xd = cart[:2], cart[2:4]
                e_x = goal - x
                t_norm = t / T
                obs = np.concatenate([x, xd, goal, e_x, [t_norm]])
                
                all_obs.append(obs)
                all_act.append(expert_act)
                
                # Step
                env_numpy.step(expert_act[None, :])
                t += arm_numpy.dt
            
            if verbose and (i + 1) % 50 == 0:
                print(f"  Collected {i+1}/{n_demos} demos")
        
        if verbose:
            print(f"  Total samples: {len(all_obs)}")
        
        return np.array(all_obs), np.array(all_act)
    
    # =========================================================================
    # Inference (Deployment)
    # =========================================================================
    
    def compute_torch(
        self,
        x: Tensor,
        xd: Tensor,
        goal: Tensor,
        t_norm: float,
    ) -> Tensor:
        """Compute action using trained policy (PyTorch)."""
        obs = self._build_observation(x, xd, goal, t_norm)
        
        with torch.no_grad():
            u, self.hidden = self.policy(obs, self.hidden)
        
        return u.squeeze()
    
    def compute_numpy(
        self,
        x: np.ndarray,
        xd: np.ndarray,
        goal: np.ndarray,
        t_norm: float,
    ) -> np.ndarray:
        """Compute action using trained policy (NumPy interface)."""
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        xd_t = torch.tensor(xd, dtype=torch.float32, device=self.device)
        goal_t = torch.tensor(goal, dtype=torch.float32, device=self.device)
        
        u = self.compute_torch(x_t, xd_t, goal_t, t_norm)
        return u.cpu().numpy()
    
    # =========================================================================
    # Model Save/Load
    # =========================================================================
    
    def save(self, path: str):
        """Save trained model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'params': self.p,
            'trained': self.trained,
        }, path)
    
    def load(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.trained = checkpoint.get('trained', True)
        self.policy.eval()


# =============================================================================
# Wrapper for NumPy Environment Compatibility
# =============================================================================

class MotorNetTrueController:
    """
    Controller wrapper for compatibility with NumPy environment.
    
    Uses trained MotorNet policy for inference,
    with OSC fallback if not trained.
    """
    
    def __init__(self, env, arm, params: Optional[MotorNetTrueParams] = None):
        self.env = env
        self.arm = arm
        self.p = params or MotorNetTrueParams()
        
        # Create MotorNet
        self.motornet = MotorNetTrue(self.p)
        
        # For OSC fallback
        self._osc = None
        
        # State
        self.qref = None
        self.goal = None
        self.t_current = 0.0
        self.T_trial = self.p.trial_duration
    
    def reset(self, q0: np.ndarray):
        """Reset controller."""
        self.qref = np.asarray(q0, dtype=np.float32)
        self.motornet.reset(q0)
        self.goal = None
        self.t_current = 0.0
    
    def set_goal(self, goal: np.ndarray, T_trial: float = 0.6):
        """Set goal for reaching."""
        self.goal = np.asarray(goal, dtype=np.float32)
        self.T_trial = T_trial
        self.motornet.set_goal(goal, T_trial)
    
    def compute(self, x_d: np.ndarray, xd_d: np.ndarray, xdd_d: np.ndarray) -> dict:
        """Compute control action."""
        # Get current state
        cart = self.env.states['cartesian'][0]
        x, xd = cart[:2], cart[2:4]
        
        # Use x_d as goal if not set
        if self.goal is None:
            self.goal = x_d.copy()
        
        # Normalized time
        t_norm = min(self.t_current / self.T_trial, 1.0) if self.T_trial > 0 else 0.5
        self.t_current += self.arm.dt
        
        if self.motornet.trained:
            # Use trained policy
            act = self.motornet.compute_numpy(x, xd, self.goal, t_norm)
        else:
            # Fallback to OSC
            act = self._compute_osc_fallback(x_d, xd_d, xdd_d)
        
        return {
            'act': act,
            'x': x,
            'xd': xd,
            'goal': self.goal,
            't_norm': t_norm,
        }
    
    def _compute_osc_fallback(self, x_d, xd_d, xdd_d) -> np.ndarray:
        """Fallback to OSC if not trained."""
        if self._osc is None:
            from controller.osc_controller import OSCController, OSCParams
            self._osc = OSCController(self.env, self.arm, OSCParams(Kp=2000.0, Kv=100.0))
            self._osc.reset(self.qref)
        
        result = self._osc.compute(x_d, xd_d, xdd_d)
        return result['act']
    
    def train_bptt(self, goals: np.ndarray, **kwargs) -> List[float]:
        """Train using BPTT (TRUE MotorNet)."""
        self.motornet.q0 = self.qref
        return self.motornet.train_bptt(goals, **kwargs)
    
    def train_bc(self, **kwargs) -> List[float]:
        """Train using Behavioral Cloning."""
        self.motornet.q0 = self.qref
        return self.motornet.train_bc(self.env, self.arm, **kwargs)
    
    def save(self, path: str):
        """Save trained model."""
        self.motornet.save(path)
    
    def load(self, path: str):
        """Load trained model."""
        self.motornet.load(path)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_motornet_true(
    env=None,
    arm=None,
    **kwargs
) -> MotorNetTrueController:
    """Create MotorNet controller."""
    params = MotorNetTrueParams(**kwargs)
    if env is not None and arm is not None:
        return MotorNetTrueController(env, arm, params)
    else:
        return MotorNetTrue(params)
