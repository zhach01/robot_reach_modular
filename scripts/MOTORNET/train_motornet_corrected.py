#!/usr/bin/env python3
"""
train_motornet_corrected.py

Behavioral cloning (BC) + optional PPO fine-tuning for MotorNetFixed.

Core correction:
- Expert controllers output activations a*(t).
- MotorNet policy outputs excitations u(t), and internal activation dynamics produce a(t).
- We therefore convert each expert activation a*(t) into an equivalent excitation target u*(t)
  using activation_target_to_excitation(...) before training.

Dataset:
- episodes: N
- each episode stores sequences (T,9) -> (T,6)
  obs_t = [x, y, xd, yd, x_d, y_d, e_x, e_y, t_norm]
  u*_t  = target excitations that reproduce a*(t) under activation dynamics.

Usage (from repo root):
  python -m scripts.MOTORNET.train_motornet_corrected --collect --train_bc --out motornet_policy.pt

Notes:
- This script assumes your repo provides:
    - config.PlantConfig, TrajectoryConfig, ...
    - tasks.numpy.random_reach.Task
    - trajectory.numpy.minjerk.MinJerkLinearTrajectory
    - controller.numpy.pd_if_optimized (recommended expert) OR controller.numpy.pd_if_controller
    - model_lib Environment + Arm26

If any import fails, run from your project root (so python can find the package).
"""

import argparse
from pathlib import Path
import numpy as np

# --- Project imports ---
from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26

from config import PlantConfig, ControlToggles, ControlGains, Numerics, InternalForceConfig, TrajectoryConfig
from tasks.numpy.random_reach import Task as RandomReachTask
from trajectory.numpy.minjerk import MinJerkLinearTrajectory, MinJerkParams

# Expert controller: optimized PD+IF (canonical)
from controller.numpy.pd_if_controller import PDIFController, PDIFParams

# MotorNet controller + helper
from controller.numpy.motornet_fixed import MotorNetFixed, MotorNetFixedParams, activation_target_to_excitation

# Torch is optional, but training needs it
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception:
    TORCH_OK = False


def build_env(pc: PlantConfig):
    muscle = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(
        muscle=muscle,
        timestep=pc.timestep,
        damping=pc.damping,
        n_ministeps=pc.n_ministeps,
        integration_method=pc.integration_method,
    )
    env = Environment(
        effector=arm,
        max_ep_duration=pc.max_ep_duration,
        action_noise=0.0,
        obs_noise=0.0,
        proprioception_delay=arm.dt,
        vision_delay=arm.dt,
        name="MotorNetTrainingEnv",
    )
    q0 = np.deg2rad(np.array(pc.q0_deg))
    qd0 = np.array(pc.qd0)
    env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})
    return env, arm, q0


def _step_env(env, action_1x6: np.ndarray):
    """Step wrapper tolerant to gym/gymnasium API variants."""
    out = env.step(action_1x6)
    done = False
    info = {}
    if isinstance(out, tuple):
        if len(out) == 5:
            _, _, terminated, truncated, info = out
            done = bool(terminated or truncated)
        elif len(out) == 4:
            _, _, done, info = out
            done = bool(done)
        elif len(out) == 2:
            _, info = out
            done = False
    return done, info


def collect_dataset(
    episodes: int,
    seed0: int,
    radius: float,
    n_points: int,
    subtract_min_activation: bool,
    out_npz: Path,
    max_steps_override: int = 0,
):
    pc = PlantConfig()
    toggles = ControlToggles()
    gains = ControlGains()
    num = Numerics()
    ifc = InternalForceConfig()
    tc = TrajectoryConfig()

    rng = np.random.RandomState(seed0)

    env, arm, q0 = build_env(pc)

    # Expert controller (optimized PD+IF)
    p_exp = PDIFParams(
        Kp_task=np.array([1600.0, 1600.0]),
        damping_ratio=0.7,
        Kff=1.0,
        enable_inertia_comp=bool(getattr(toggles, "enable_inertia_comp", True)),
        enable_gravity_comp=bool(getattr(toggles, "enable_gravity_comp", True)),
        enable_coriolis_comp=bool(getattr(toggles, "enable_velocity_comp", True)),
        use_critical_damping=True,
        enable_nullspace=True,
        Kp_null=20.0,
        Kd_null=5.0,
        eps=float(getattr(num, "eps", 1e-6)),
        lam_os_max=float(getattr(num, "lam_os_max", 200.0)),
        sigma_thresh=float(getattr(num, "sigma_thresh", 1e-4)),
        gate_pow=float(getattr(num, "gate_pow", 2.0)),
        bisect_iters=int(getattr(ifc, "bisect_iters", 12)),
        enable_internal_force=False,
        cocon_level=0.0,
    )
    expert = PDIFController(env, arm, p_exp)

    expert.reset(q0)

    # Task/trajectory generator
    task = RandomReachTask(n_points=n_points, radius=radius, seed=seed0)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale))

    dt = float(arm.dt)
    steps = int(pc.max_ep_duration / dt)
    if max_steps_override and max_steps_override > 0:
        steps = int(max_steps_override)

    # storage: list of episodes (T,9) and (T,6)
    obs_eps = []
    u_eps = []

    # initial activation history for conversion a* -> u*
    a_prev = np.full(6, 0.05, dtype=float)

    for ep in range(episodes):
        # reset env to nominal start each episode (deterministic, but different target via task seed offset)
        ep_seed = int(seed0 + ep)
        task_ep = RandomReachTask(n_points=n_points, radius=radius, seed=ep_seed)
        waypoints_ep = task_ep.build_waypoints(env)
        traj_ep = MinJerkLinearTrajectory(waypoints_ep, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale))

        env.reset(options={"joint_state": np.concatenate([q0, np.zeros_like(q0)])[None, :], "deterministic": True})
        expert.reset(q0)

        a_prev = np.full(6, 0.05, dtype=float)

        obs_T = np.zeros((steps, 9), dtype=np.float32)
        u_T = np.zeros((steps, 6), dtype=np.float32)

        for k in range(steps):
            # desired reference from trajectory
            #x_d, xd_d, xdd_d = traj_ep(k * dt)
            x_d, xd_d, xdd_d = traj_ep.sample(k * dt)
            x_d = np.asarray(x_d, dtype=float).reshape(2)
            xd_d = np.asarray(xd_d, dtype=float).reshape(2)
            xdd_d = np.asarray(xdd_d, dtype=float).reshape(2)

            # read current state for obs construction
            cart = env.states["cartesian"][0]
            x = cart[:2].astype(float)
            xd = cart[2:4].astype(float)

            e_x = x_d - x
            t_norm = float(min(k / max(steps - 1, 1), 1.0))
            obs = np.concatenate([x, xd, x_d, e_x, np.array([t_norm])]).astype(np.float32)

            # expert activation
            out = expert.compute(x_d, xd_d, xdd_d)
            a_star = np.asarray(out["act"], dtype=float).reshape(6)

            if subtract_min_activation:
                a_star = np.clip(a_star - 0.02, 0.0, 1.0)

            # convert activation target -> excitation target
            u_star = activation_target_to_excitation(
                a_target=a_star,
                a_prev=a_prev,
                dt=dt,
                tau_rise=0.015,
                tau_fall=0.060,
            )

            # update for next conversion step (assume expert activation was applied)
            a_prev = a_star.copy()

            # record
            obs_T[k, :] = obs
            u_T[k, :] = u_star.astype(np.float32)

            # apply expert activation to env
            done, _ = _step_env(env, a_star[None, :])
            if done:
                # pad remaining with last values (keep fixed length)
                obs_T[k+1:, :] = obs_T[k, :]
                u_T[k+1:, :] = u_T[k, :]
                break

        obs_eps.append(obs_T)
        u_eps.append(u_T)
        print(f"[collect] ep {ep+1:03d}/{episodes}  T={steps} samples")

    obs_arr = np.stack(obs_eps, axis=0)  # (E,T,9)
    u_arr = np.stack(u_eps, axis=0)      # (E,T,6)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, obs=obs_arr, u=u_arr, dt=dt, radius=radius, n_points=n_points, seed0=seed0)
    print(f"[done] saved dataset: {out_npz}  obs={obs_arr.shape}  u={u_arr.shape}")


def train_bc(
    dataset_npz: Path,
    out_path: Path,
    device: str = "cpu",
    hidden_dim: int = 128,
    n_layers: int = 2,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    epochs: int = 200,
    batch_size: int = 16,
):
    if not TORCH_OK:
        raise RuntimeError("PyTorch is required for training, but it was not found.")

    data = np.load(dataset_npz)
    obs = data["obs"].astype(np.float32)  # (E,T,9)
    u = data["u"].astype(np.float32)      # (E,T,6)

    E, T, _ = obs.shape

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from controller.numpy.motornet_fixed import GRUPolicyNetwork

    device_t = torch.device(device)
    net = GRUPolicyNetwork(input_dim=9, hidden_dim=hidden_dim, output_dim=6, n_layers=n_layers).to(device_t)
    opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # create episode indices mini-batches (sequence supervision)
    idx = np.arange(E)

    net.train()
    for ep in range(epochs):
        np.random.shuffle(idx)
        ep_loss = 0.0
        n_batches = 0

        for i0 in range(0, E, batch_size):
            ii = idx[i0:i0+batch_size]
            xb = torch.tensor(obs[ii], device=device_t)  # (B,T,9)
            yb = torch.tensor(u[ii], device=device_t)    # (B,T,6)

            # reset hidden each episode batch
            h0 = net.init_hidden(batch_size=len(ii), device=device_t)
            pred, _ = net(xb, h0)  # (B,T,6)

            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            ep_loss += float(loss.item())
            n_batches += 1

        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"[BC] epoch {ep+1:04d}/{epochs}  loss={ep_loss/max(n_batches,1):.6f}")

    # save checkpoint
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": net.state_dict(),
        "trained": True,
        "input_dim": 9,
        "hidden_dim": hidden_dim,
        "output_dim": 6,
        "n_layers": n_layers,
    }
    torch.save(ckpt, out_path)
    print(f"[BC] saved policy: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="collect expert dataset")
    ap.add_argument("--train_bc", action="store_true", help="train by behavioral cloning")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--radius", type=float, default=0.10)
    ap.add_argument("--n_points", type=int, default=1)
    ap.add_argument("--subtract_min_activation", action="store_true")

    ap.add_argument("--dataset", type=str, default="motornet/training/dataset_u.npz")
    ap.add_argument("--out", type=str, default="motornet/saved_model/motornet_policy.pt")

    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)

    args = ap.parse_args()

    dataset_npz = Path(args.dataset)
    out_path = Path(args.out)

    if args.collect:
        collect_dataset(
            episodes=int(args.episodes),
            seed0=int(args.seed0),
            radius=float(args.radius),
            n_points=int(args.n_points),
            subtract_min_activation=bool(args.subtract_min_activation),
            out_npz=dataset_npz,
        )

    if args.train_bc:
        train_bc(
            dataset_npz=dataset_npz,
            out_path=out_path,
            device=str(args.device),
            hidden_dim=int(args.hidden_dim),
            n_layers=int(args.n_layers),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
        )


if __name__ == "__main__":
    main()
