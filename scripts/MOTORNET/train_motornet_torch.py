#!/usr/bin/env python3
"""
Train TRUE MotorNet (Torch)

Following the same patterns as train_bc_random_reach.py and main_random_reach_pd_if_torch.py.

Usage:
    python train_motornet_torch.py --epochs 200 --n_goals 100
    python -m scripts.MOTORNET.train_motornet_torch --epochs 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root
script_dir = Path(__file__).parent.resolve()
project_root = script_dir
for _ in range(5):
    if (project_root / 'model_lib').exists():
        break
    project_root = project_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Import trainer
try:
    from controller.torch.motornet_controller import (
        MotorNetParams,
        MotorNetTrainer,
        MotorNetController,
    )
except ImportError:
    from scripts.MOTORNET.motornet_controller_torch import (
        MotorNetParams,
        MotorNetTrainer,
        MotorNetController,
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Train TRUE MotorNet (Torch)")
    ap.add_argument("--n_goals", type=int, default=100, help="Number of training goals")
    ap.add_argument("--epochs", type=int, default=200, help="Training epochs")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size")
    ap.add_argument("--hidden_dim", type=int, default=128, help="GRU hidden dim")
    ap.add_argument("--n_layers", type=int, default=2, help="GRU layers")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--device", type=str, default="cuda", help="Device")
    ap.add_argument("--out", type=str, default="models/motornet_torch.pt", help="Save path")
    ap.add_argument("--q0", type=float, nargs=2, default=[45.0, 90.0], 
                    help="Initial joint angles (degrees)")
    return ap.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("TRUE MotorNet Training (Torch)")
    print("="*70)
    
    # Use float64 like other torch controllers
    torch.set_default_dtype(torch.float64)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Parameters
    params = MotorNetParams(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        lr=args.lr,
        device=str(device),
    )
    
    # Create trainer
    trainer = MotorNetTrainer(params, q0_deg=args.q0)
    
    print(f"\nConfiguration:")
    print(f"  q0: {args.q0} deg")
    print(f"  Goals: {args.n_goals}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  GRU layers: {args.n_layers}")
    print(f"  Learning rate: {args.lr}")
    
    # Train
    print(f"\n--- Training ---")
    start_time = time.time()
    
    losses = trainer.train(
        n_goals=args.n_goals,
        n_epochs=args.epochs,
        verbose=True,
    )
    
    train_time = time.time() - start_time
    print(f"\nTraining time: {train_time:.1f}s ({train_time/args.epochs:.2f}s/epoch)")
    
    # Evaluate
    print(f"\n--- Evaluation ---")
    metrics = trainer.evaluate(n_goals=50)
    print(f"Test RMSE:   {metrics['rmse_cm']:.2f} cm")
    print(f"Max error:   {metrics['max_error_cm']:.2f} cm")
    
    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    trainer.save(args.out)
    print(f"\nModel saved: {args.out}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
