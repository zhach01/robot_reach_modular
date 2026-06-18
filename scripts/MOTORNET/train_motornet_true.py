#!/usr/bin/env python3
"""
TRUE MotorNet Training Script

Train MotorNet using BPTT through differentiable plant dynamics.

Training modes:
1. bptt: TRUE MotorNet - backprop through differentiable arm (Codol et al. 2024)
2. bc: Behavioral Cloning - imitate OSC expert demonstrations

Usage:
    python train_motornet_true.py --mode bptt --epochs 200
    python train_motornet_true.py --mode bc --epochs 200 --n_demos 200
    
    # Or as module:
    python -m scripts.MOTORNET.train_motornet_true --mode bptt --epochs 200
"""

import numpy as np
import argparse
import os
import sys
from pathlib import Path

# Add project root to path (handles both direct run and module run)
script_dir = Path(__file__).parent.resolve()
project_root = script_dir
# Walk up until we find config.py or model_lib
for _ in range(5):
    if (project_root / 'config.py').exists() or (project_root / 'model_lib').exists():
        break
    project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Check PyTorch
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("ERROR: PyTorch required! Install with: pip install torch")
    sys.exit(1)

from model_lib.environment_numpy import Environment
from model_lib.muscles_numpy import RigidTendonHillMuscle
from model_lib.effector_numpy import RigidTendonArm26
from config import PlantConfig, TrajectoryConfig

# Try to import tasks (optional)
try:
    from tasks.center_out import Task as CenterOutTask
except ImportError:
    CenterOutTask = None

# Trajectory imports are optional - we have a fallback
try:
    from trajectory.minjerk import MinJerkLinearTrajectory, MinJerkParams
except ImportError:
    MinJerkLinearTrajectory = None
    MinJerkParams = None

# Try different import paths for MotorNet
try:
    from controller.motornet_true import (
        MotorNetTrue,
        MotorNetTrueParams,
        MotorNetTrueController,
    )
except ImportError:
    # Try relative import for module execution
    from scripts.MOTORNET.motornet_true import (
        MotorNetTrue,
        MotorNetTrueParams,
        MotorNetTrueController,
    )


# =============================================================================
# Simple Trajectory Helper (fallback if MinJerkLinearTrajectory format differs)
# =============================================================================

class SimpleMinJerkTrajectory:
    """Simple minimum-jerk trajectory between two points."""
    
    def __init__(self, start: np.ndarray, goal: np.ndarray, duration: float = 0.5):
        self.start = np.asarray(start)
        self.goal = np.asarray(goal)
        self.T = duration
    
    def sample(self, t: float):
        """Sample trajectory at time t."""
        # Normalized time
        tau = np.clip(t / self.T, 0.0, 1.0)
        
        # Minimum-jerk profile: s(τ) = 10τ³ - 15τ⁴ + 6τ⁵
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.T
        sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / (self.T**2)
        
        # Position, velocity, acceleration
        delta = self.goal - self.start
        x_d = self.start + s * delta
        xd_d = sd * delta
        xdd_d = sdd * delta
        
        return x_d, xd_d, xdd_d


def create_numpy_env():
    """Create NumPy environment for BC training and evaluation."""
    pc = PlantConfig()
    tc = TrajectoryConfig()
    
    muscle = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(
        muscle=muscle,
        timestep=pc.timestep,
        damping=pc.damping,
        n_ministeps=pc.n_ministeps,
        integration_method=pc.integration_method
    )
    env = Environment(
        effector=arm,
        max_ep_duration=pc.max_ep_duration,
        action_noise=0.0,
        obs_noise=0.0,
        proprioception_delay=arm.dt,
        vision_delay=arm.dt,
        name='MotorNetEnv'
    )
    
    return env, arm, pc, tc


def generate_goals(center: np.ndarray, n_goals: int, radius: float = 0.10) -> np.ndarray:
    """Generate random reaching goals."""
    angles = np.random.uniform(0, 2 * np.pi, n_goals)
    radii = np.random.uniform(0.05, radius, n_goals)
    goals = np.array([
        center + r * np.array([np.cos(a), np.sin(a)])
        for a, r in zip(angles, radii)
    ])
    return goals


def evaluate(ctrl, env, arm, goals: np.ndarray, q0: np.ndarray, tc) -> dict:
    """Evaluate controller on goals."""
    # Try multiple import paths for OSC
    OSCController = None
    OSCParams = None
    
    import_attempts = [
        'controller.osc_controller',
        'controller.osc',
        'controllers.osc_controller',
        'controllers.osc',
        'scripts.controllers.osc_controller',
        'scripts.MOTORNET.osc_controller',
    ]
    
    for module_path in import_attempts:
        try:
            module = __import__(module_path, fromlist=['OSCController', 'OSCParams'])
            OSCController = getattr(module, 'OSCController', None)
            OSCParams = getattr(module, 'OSCParams', None)
            if OSCController is not None and OSCParams is not None:
                print(f"Loaded OSC from: {module_path}")
                break
        except (ImportError, AttributeError):
            continue
    
    if OSCController is None:
        print("Warning: OSCController not found, evaluating MotorNet only")
        print(f"  Tried: {import_attempts}")
    
    errors_motornet = []
    errors_osc = []
    
    osc = None
    if OSCController is not None:
        osc = OSCController(env, arm, OSCParams(Kp=2000.0, Kv=100.0))
    
    for goal in goals:
        # Create trajectory
        env.reset(options={'joint_state': np.concatenate([q0, np.zeros(2)])[None, :], 'deterministic': True})
        start = env.states['cartesian'][0, :2].copy()
        
        # Use simple trajectory (works with any format)
        traj = SimpleMinJerkTrajectory(start, goal, duration=0.5)
        
        # Test MotorNet
        env.reset(options={'joint_state': np.concatenate([q0, np.zeros(2)])[None, :], 'deterministic': True})
        ctrl.reset(q0)
        ctrl.set_goal(goal)
        
        trial_errors = []
        t = 0.0
        T = 0.6
        while t < T:
            x_d, xd_d, xdd_d = traj.sample(t)
            result = ctrl.compute(x_d, xd_d, xdd_d)
            cart = env.states['cartesian'][0]
            trial_errors.append(np.linalg.norm(cart[:2] - x_d))
            env.step(result['act'][None, :])
            t += arm.dt
        
        errors_motornet.append(np.sqrt(np.mean(np.array(trial_errors)**2)))
        
        # Test OSC (if available)
        if osc is not None:
            env.reset(options={'joint_state': np.concatenate([q0, np.zeros(2)])[None, :], 'deterministic': True})
            osc.reset(q0)
            
            trial_errors = []
            t = 0.0
            while t < T:
                x_d, xd_d, xdd_d = traj.sample(t)
                result = osc.compute(x_d, xd_d, xdd_d)
                cart = env.states['cartesian'][0]
                trial_errors.append(np.linalg.norm(cart[:2] - x_d))
                env.step(result['act'][None, :])
                t += arm.dt
            
            errors_osc.append(np.sqrt(np.mean(np.array(trial_errors)**2)))
    
    result = {
        'motornet_rmse_cm': np.mean(errors_motornet) * 100,
    }
    if errors_osc:
        result['osc_rmse_cm'] = np.mean(errors_osc) * 100
    else:
        result['osc_rmse_cm'] = None
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Train TRUE MotorNet')
    parser.add_argument('--mode', type=str, default='bptt', choices=['bptt', 'bc'],
                        help='Training mode: bptt (TRUE MotorNet) or bc (Behavioral Cloning)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--n_demos', type=int, default=200, help='Number of expert demos for BC')
    parser.add_argument('--n_goals', type=int, default=100, help='Number of training goals')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./motornet_checkpoints')
    parser.add_argument('--skip_initial_eval', action='store_true',
                        help='Skip initial evaluation (useful if OSC not available)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("TRUE MotorNet Training")
    print("="*70)
    print(f"Mode: {args.mode.upper()}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    
    # Create environment
    env, arm, pc, tc = create_numpy_env()
    q0 = np.deg2rad(np.array(pc.q0_deg))
    
    # Reset to get center position
    env.reset(options={'joint_state': np.concatenate([q0, np.zeros(2)])[None, :], 'deterministic': True})
    center = env.states['cartesian'][0, :2].copy()
    
    # Generate training goals
    training_goals = generate_goals(center, args.n_goals)
    test_goals = generate_goals(center, 20)
    
    print(f"Training goals: {len(training_goals)}")
    print(f"Test goals: {len(test_goals)}")
    
    # Create controller
    params = MotorNetTrueParams(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        training_mode=args.mode,
        bptt_epochs=args.epochs,
        bc_epochs=args.epochs,
        n_expert_demos=args.n_demos,
        device=args.device,
    )
    
    ctrl = MotorNetTrueController(env, arm, params)
    ctrl.reset(q0)
    
    # Evaluate before training (OSC fallback)
    results_before = None
    if not args.skip_initial_eval:
        print("\n--- Before Training (OSC Fallback) ---")
        try:
            results_before = evaluate(ctrl, env, arm, test_goals, q0, tc)
            print(f"MotorNet RMSE: {results_before['motornet_rmse_cm']:.2f} cm (using OSC fallback)")
            if results_before['osc_rmse_cm'] is not None:
                print(f"OSC RMSE:      {results_before['osc_rmse_cm']:.2f} cm")
            else:
                print("OSC RMSE:      N/A (OSCController not found)")
        except Exception as e:
            print(f"Initial evaluation failed: {e}")
            print("Tip: Use --skip_initial_eval to skip this step")
            if args.mode == 'bc':
                print("ERROR: BC mode requires OSCController for expert demos")
                return
    else:
        print("\n--- Skipping Initial Evaluation ---")
    
    # Train
    print(f"\n--- Training ({args.mode.upper()}) ---")
    
    if args.mode == 'bptt':
        losses = ctrl.train_bptt(training_goals, n_epochs=args.epochs, verbose=True)
    else:
        losses = ctrl.train_bc(n_demos=args.n_demos, n_epochs=args.epochs, verbose=True)
    
    # Evaluate after training
    print("\n--- After Training ---")
    try:
        results_after = evaluate(ctrl, env, arm, test_goals, q0, tc)
        print(f"MotorNet RMSE: {results_after['motornet_rmse_cm']:.2f} cm")
        if results_after['osc_rmse_cm'] is not None:
            print(f"OSC RMSE:      {results_after['osc_rmse_cm']:.2f} cm")
        else:
            print("OSC RMSE:      N/A (OSCController not found)")
        
        if results_before is not None and results_after['motornet_rmse_cm'] > 0:
            improvement = results_before['motornet_rmse_cm'] / results_after['motornet_rmse_cm']
            print(f"Improvement:   {improvement:.2f}x")
    except Exception as e:
        print(f"Post-training evaluation failed: {e}")
    
    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'motornet_{args.mode}.pt')
    ctrl.save(save_path)
    print(f"\nModel saved to: {save_path}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
