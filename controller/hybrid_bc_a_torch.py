#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Torch version of hybrid_bc_a.py

Behavior-cloned activation policy (a-policy).

Obs = [q, qd, x, xd, e, e_dot, xdd_d, l_norm(m), v_norm(m), R(2*m), a_prev(m)]
   -> 14 + 5m dims (m = number of muscles)

Action = a (m,)  -- final muscle activations in [a_min, 1].

This file provides:
- RLPolicyParams, MLP, RLPolicy           : generic MLP with robust .load()
- BehaviorCloner                          : MSE trainer with whitening
- HookedControllerATorch                  : wraps a teacher, records (O, a)
- RLControllerATorch                      : deploys a-policy (outputs activations)

Works with EnvironmentTorch / SimulatorTorch as long as controller.compute(...)
returns a dict containing at least {"act": a}. Here we also reconstruct
tau_des from a so the logger can compute residuals.
"""
from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Any, List, Dict

import torch
import torch.nn as nn
from torch import Tensor

# For τ reconstruction from activations (Torch version)
from muscles.muscle_tools_torch import (
    get_Fmax_vec,
    active_force_from_activation,
)

# ---------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------
# Policy & trainer (Torch-native)
# ---------------------------------------------------------------------


@dataclass
class RLPolicyParams:
    obs_dim: int
    act_dim: int
    hidden: Tuple[int, int] = (128, 128)
    device: str = "cpu"
    out_clip_min: float = 0.0    # clamp to [min_activation, 1] later
    out_clip_max: float = 1.0


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, int] = (128, 128)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RLPolicy:
    """
    a = π(obs). Stores mean/std for whitening, and rebuilds the network on load
    if checkpoint architecture (obs_dim/hidden/act_dim) differs.

    The internal network is always kept in float32 for numerical stability,
    even if the plant runs in float64.
    """

    def __init__(
        self,
        params: RLPolicyParams,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.p = params
        self.device = torch.device(self.p.device)

        # Always keep model in float32
        self.model = MLP(self.p.obs_dim, self.p.act_dim, self.p.hidden).to(self.device)
        self.model = self.model.to(torch.float32)

        self.mean = (
            torch.zeros(self.p.obs_dim, dtype=torch.float32, device=self.device)
            if mean is None
            else torch.tensor(mean, dtype=torch.float32, device=self.device)
        )
        self.std = (
            torch.ones(self.p.obs_dim, dtype=torch.float32, device=self.device)
            if std is None
            else torch.tensor(std, dtype=torch.float32, device=self.device)
        )

    # ---------- inference ----------

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> np.ndarray:
        """
        NumPy API (for legacy / offline use).
        """
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).view(
            -1, self.p.obs_dim
        )
        xn = (x - self.mean) / (self.std + 1e-6)
        a = self.model(xn)
        a = torch.clamp(a, self.p.out_clip_min, self.p.out_clip_max)
        return a.cpu().numpy().reshape(-1)

    @torch.no_grad()
    def act_torch(self, obs_t: Tensor) -> Tensor:
        """
        Torch API: obs_t shape (..., obs_dim) on any device.
        Returns activations as a Tensor with same leading dims (float32).
        """
        x = obs_t.to(device=self.device, dtype=torch.float32)
        if x.ndim == 1:
            x = x.view(1, -1)
        xn = (x - self.mean) / (self.std + 1e-6)
        a = self.model(xn)
        a = torch.clamp(a, self.p.out_clip_min, self.p.out_clip_max)
        return a.squeeze(0)

    # ---------- I/O ----------

    def save(self, path: str) -> None:
        _ensure_dir(path)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "mean": self.mean.cpu().numpy(),
                "std": self.std.cpu().numpy(),
                "params": {
                    "obs_dim": self.p.obs_dim,
                    "act_dim": self.p.act_dim,
                    "hidden": list(self.p.hidden),
                    "out_clip_min": float(self.p.out_clip_min),
                    "out_clip_max": float(self.p.out_clip_max),
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        ck_p = ckpt.get("params", {})
        in_dim = int(ck_p.get("obs_dim", self.p.obs_dim))
        out_dim = int(ck_p.get("act_dim", self.p.act_dim))
        hidden = tuple(ck_p.get("hidden", list(self.p.hidden)))

        # Rebuild if architecture differs
        if (in_dim != self.p.obs_dim) or (hidden != self.p.hidden) or (out_dim != self.p.act_dim):
            self.p = RLPolicyParams(
                obs_dim=in_dim,
                act_dim=out_dim,
                hidden=hidden,
                device=str(self.device),
                out_clip_min=float(ck_p.get("out_clip_min", 0.0)),
                out_clip_max=float(ck_p.get("out_clip_max", 1.0)),
            )
            self.model = MLP(in_dim, out_dim, hidden).to(self.device)
            self.model = self.model.to(torch.float32)
            self.mean = torch.zeros(in_dim, dtype=torch.float32, device=self.device)
            self.std = torch.ones(in_dim, dtype=torch.float32, device=self.device)

        self.model.load_state_dict(ckpt["state_dict"])
        self.mean = torch.tensor(ckpt["mean"], dtype=torch.float32, device=self.device)
        self.std = torch.tensor(ckpt["std"], dtype=torch.float32, device=self.device)


class BehaviorCloner:
    """
    Simple BC trainer: given (O, A) NumPy datasets, fits a Torch MLP with MSE
    and internal whitening (mean/std stored in the policy).
    """

    def __init__(self, policy: RLPolicy, lr: float = 1e-3, wd: float = 0.0):
        self.policy = policy
        self.opt = torch.optim.Adam(self.policy.model.parameters(), lr=lr, weight_decay=wd)
        self.loss = nn.MSELoss()

    def fit(
        self,
        O: np.ndarray,
        A: np.ndarray,
        epochs: int = 20,
        batch_size: int = 1024,
        shuffle: bool = True,
    ) -> None:
        device = self.policy.device
        O_t = torch.as_tensor(O, dtype=torch.float32, device=device)
        A_t = torch.as_tensor(A, dtype=torch.float32, device=device)

        # whitening
        mean = O_t.mean(dim=0)
        std = O_t.std(dim=0).clamp_min(1e-6)
        self.policy.mean.copy_(mean)
        self.policy.std.copy_(std)

        N = O_t.shape[0]
        idx = torch.arange(N, device=device)
        for _ in range(epochs):
            if shuffle:
                idx = idx[torch.randperm(N, device=device)]
            for i in range(0, N, batch_size):
                b = idx[i:min(i + batch_size, N)]
                x = (O_t[b] - mean) / (std + 1e-6)
                y = A_t[b]
                yhat = self.policy.model(x)
                l = self.loss(yhat, y)
                self.opt.zero_grad()
                l.backward()
                self.opt.step()


# ---------------------------------------------------------------------
# Teacher wrapper (Torch env)
# ---------------------------------------------------------------------


class HookedControllerATorch:
    """
    Wraps a 'teacher' that already computes final activations (diag['act']).
    Records (O, a) pairs for dataset creation using the a-policy obs:

      O = [ q(2) qd(2) x(2) xd(2) e(2) e_dot(2) xdd_d(2)  l_norm(m) v_norm(m)  R(2*m)  a_prev(m) ]

    Exposes .qref for compatibility with your simulator logger.
    """

    def __init__(self, env, teacher):
        self.env = env
        self.teacher = teacher
        self.obs: List[np.ndarray] = []
        self.act: List[np.ndarray] = []
        self._qref: Optional[Tensor] = None

    @property
    def qref(self):
        return getattr(self.teacher, "qref", self._qref)

    def reset(self, q0: Tensor) -> None:
        self.obs.clear()
        self.act.clear()
        self._qref = None
        if hasattr(self.teacher, "reset"):
            self.teacher.reset(q0)

    def compute(self, x_d: Any, xd_d: Any, xdd_d: Any) -> Dict[str, Any]:
        diag = self.teacher.compute(x_d, xd_d, xdd_d)

        joint = self.env.states["joint"]      # (B, 2*dof)
        cart = self.env.states["cartesian"]   # (B, >=4)
        geom = self.env.states["geometry"]    # (B, 2 + dof, M)
        musc = self.env.states["muscle"]      # (B, state_dim, M)

        if joint.ndim == 1:
            joint = joint.unsqueeze(0)
        if cart.ndim == 1:
            cart = cart.unsqueeze(0)

        B = joint.shape[0]
        assert B >= 1, "Environment must have at least one batch element."

        n = self.env.skeleton.dof
        q = joint[0, :n]
        qd = joint[0, n:]
        x = cart[0, :2]
        xd = cart[0, 2:4]

        device = q.device
        dtype = q.dtype

        x_d_t = torch.as_tensor(x_d, device=device, dtype=dtype).view(-1)[:2]
        xd_d_t = torch.as_tensor(xd_d, device=device, dtype=dtype).view(-1)[:2]
        xdd_d_t = torch.as_tensor(xdd_d, device=device, dtype=dtype).view(-1)[:2]

        e = x_d_t - x
        e_dot = xd_d_t - xd

        # Geometry / muscle state
        if geom.ndim != 3:
            raise ValueError(
                f"env.states['geometry'] must be (B, 2 + dof, M), got {tuple(geom.shape)}"
            )
        lenvel = geom[:, :2, :]            # (B,2,M)
        Rm = geom[:, 2:2 + n, :][0]        # (n,M)
        M_muscles = int(Rm.shape[-1])

        names = list(getattr(self.env.muscle, "state_name", []))
        if "activation" in names:
            idx_act = names.index("activation")
            a_prev = musc[0, idx_act, :]   # (M,)
        else:
            a_prev = torch.zeros(M_muscles, device=device, dtype=dtype)

        base14 = torch.cat([q, qd, x, xd, e, e_dot, xdd_d_t], dim=0)  # (14,)
        O_t = torch.cat(
            [
                base14,
                lenvel[0].reshape(-1),
                Rm.reshape(-1),
                a_prev.reshape(-1),
            ],
            dim=0,
        ).to(torch.float32)

        O = O_t.detach().cpu().numpy()

        a = diag["act"]
        if isinstance(a, torch.Tensor):
            a_np = a.detach().cpu().numpy().reshape(-1).astype(np.float32)
        else:
            a_np = np.asarray(a, dtype=np.float32).reshape(-1)

        self.obs.append(O)
        self.act.append(a_np)

        self._qref = getattr(self.teacher, "qref", None)
        return diag


# ---------------------------------------------------------------------
# RL controller (Torch env)
# ---------------------------------------------------------------------


@dataclass
class RLControllerAParams:
    """
    Reserved for future options; currently unused.
    """
    pass


class RLControllerATorch:
    """
    Deployment controller: a = π(obs_a) in a Torch env.

    Returns a dict compatible with your Simulator/Logger.
    Reconstructs tau_des from predicted activations for logging.

    Interface:
      - compute(x_d, xd_d, xdd_d) -> diag dict
      - exposes .qref for simulator compatibility
    """

    def __init__(self, env, arm, policy: RLPolicy, params: RLControllerAParams):
        self.env = env
        self.arm = arm
        self.pi = policy
        self.p = params
        self._qref: Optional[Tensor] = None

    @property
    def qref(self):
        return self._qref

    def reset(self, q0: Tensor) -> None:
        self._qref = None

    def compute(self, x_d: Any, xd_d: Any, xdd_d: Any) -> Dict[str, Any]:
        # --- State from Torch env (use first batch element) ---
        joint = self.env.states["joint"]
        cart = self.env.states["cartesian"]
        geom = self.env.states["geometry"]
        musc = self.env.states["muscle"]

        if joint.ndim == 1:
            joint = joint.unsqueeze(0)
        if cart.ndim == 1:
            cart = cart.unsqueeze(0)

        n = self.env.skeleton.dof
        q = joint[0, :n]
        qd = joint[0, n:]
        x = cart[0, :2]
        xd = cart[0, 2:4]

        device = q.device
        dtype = q.dtype

        # --- Build obs_a: [q, qd, x, xd, e, e_dot, xdd_d, l_norm, v_norm, R, a_prev] ---
        x_d_t = torch.as_tensor(x_d, device=device, dtype=dtype).view(-1)[:2]
        xd_d_t = torch.as_tensor(xd_d, device=device, dtype=dtype).view(-1)[:2]
        xdd_d_t = torch.as_tensor(xdd_d, device=device, dtype=dtype).view(-1)[:2]

        e = x_d_t - x
        e_dot = xd_d_t - xd

        if geom.ndim != 3:
            raise ValueError(
                f"env.states['geometry'] must be (B, 2 + dof, M), got {tuple(geom.shape)}"
            )
        lenvel = geom[:, :2, :]            # (B,2,M)
        Rm = geom[:, 2:2 + n, :][0]        # (n,M)
        M_muscles = int(Rm.shape[-1])

        names = list(getattr(self.env.muscle, "state_name", []))
        if "activation" in names:
            idx_act = names.index("activation")
            a_prev = musc[0, idx_act, :]   # (M,)
        else:
            a_prev = torch.zeros(M_muscles, device=device, dtype=dtype)

        base14 = torch.cat([q, qd, x, xd, e, e_dot, xdd_d_t], dim=0)  # (14,)
        O_t = torch.cat(
            [
                base14,
                lenvel[0].reshape(-1),
                Rm.reshape(-1),
                a_prev.reshape(-1),
            ],
            dim=0,
        ).to(torch.float32)

        # --- Policy: obs -> activations (Torch) ---
        a = self.pi.act_torch(O_t).to(device=device, dtype=dtype)     # (M,)
        # Clamp to [min_activation, 1]
        a_min = float(getattr(self.env.muscle, "min_activation", 0.0))
        a = torch.clamp(a, a_min, 1.0)

        # --- Reconstruct τ from activations (keep signs consistent with hybrid_bc_a.py) ---
        lenvel_b = lenvel  # (B,2,M)

        names = list(getattr(self.env.muscle, "state_name", []))
        if "force-length PE" in names:
            idx_flpe = names.index("force-length PE")
            flpe = musc[0, idx_flpe, :]  # (M,)
        else:
            flpe = torch.zeros(M_muscles, device=device, dtype=dtype)

        Fmax_v = get_Fmax_vec(
            self.env,
            M_muscles,
            device=Rm.device,
            dtype=Rm.dtype,
        )  # (M,)

        # active_force_from_activation is Torch-native and supports these shapes
        af_now = active_force_from_activation(a, lenvel_b, self.env.muscle)  # (M,)
        F_pred = Fmax_v * (af_now + flpe)                                    # (M,)

        # Note: original hybrid_bc_a.py uses tau_hat = -(Rm @ F_pred)
        tau_hat = - (Rm @ F_pred)                                            # (n,)

        self._qref = q.clone()

        return {
            "tau_des": tau_hat,
            "R": Rm,
            "Fmax": Fmax_v,
            "F_des": F_pred,
            "act": a,
            "q": q,
            "qd": qd,
            "x": x,
            "xd": xd,
            "xref_tuple": (x_d_t, xd_d_t, xdd_d_t),
            "eta": 1.0,
            "diag": {},
        }


# ---------------------------------------------------------------------
# Smoke test: activation BC controller on Torch plant/env
# ---------------------------------------------------------------------


def _smoke_test_bc_a_torch() -> None:
    """
    Smoke test for RLControllerATorch on RigidTendonArm26 + Hill muscles.

    - Builds a Torch Env/Arm/Muscle stack.
    - Builds obs_dim = 14 + 5m from geometry shape.
    - Loads models/random_reach_bc_a.pt if present (else random policy).
    - Runs a single min-jerk reach and prints tracking + activation stats.
    """
    import torch as _torch
    from model_lib.environment_torch import Environment as EnvTorch
    from model_lib.muscles_torch import RigidTendonHillMuscle
    from model_lib.effector_torch import RigidTendonArm26
    from trajectory.minjerk_torch import MinJerkLinearTrajectoryTorch, MinJerkParams
    from sim.simulator_torch import TargetReachSimulatorTorch
    from config import (
        PlantConfig,
        ControlToggles,
        ControlGains,
        Numerics,
        InternalForceConfig,
        TrajectoryConfig,
    )

    print("\n[BC-a Torch] smoke test starting ...")
    _torch.set_default_dtype(_torch.float64)
    _torch.set_printoptions(precision=6, sci_mode=False)

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

    # ---------- configs ----------
    pc = PlantConfig()
    toggles = ControlToggles()
    gains = ControlGains()
    num = Numerics()
    ifc = InternalForceConfig()
    tc = TrajectoryConfig()

    # ---------- build effector + env ----------
    muscle = RigidTendonHillMuscle(
        min_activation=0.02,
        device=device,
        dtype=_torch.get_default_dtype(),
    )

    arm = RigidTendonArm26(
        muscle=muscle,
        timestep=pc.timestep,
        damping=pc.damping,
        n_ministeps=pc.n_ministeps,
        integration_method=pc.integration_method,
        device=device,
        dtype=_torch.get_default_dtype(),
    )

    env = EnvTorch(
        effector=arm,
        max_ep_duration=pc.max_ep_duration,
        action_noise=0.0,
        obs_noise=0.0,
        action_frame_stacking=1,
        proprioception_delay=arm.dt,
        vision_delay=arm.dt,
        name="BC_a_ReachEnvTorch",
    )

    # ---------- initial joint state (B=1) ----------
    q0 = _torch.deg2rad(
        _torch.tensor(pc.q0_deg, dtype=_torch.get_default_dtype(), device=device)
    )  # (2,)
    qd0 = _torch.tensor(pc.qd0, dtype=_torch.get_default_dtype(), device=device)  # (2,)
    joint0 = _torch.cat([q0, qd0]).unsqueeze(0)  # (1, 4)

    env.reset(
        options={
            "joint_state": joint0,
            "deterministic": True,
        }
    )

    # ---------- simple min-jerk trajectory ----------
    fingertip0 = env.states["fingertip"][0, :2]  # (2,)
    center = fingertip0
    radius = 0.10
    target = center + _torch.tensor([radius, 0.0], dtype=center.dtype, device=device)

    waypoints = _torch.stack([center, target], dim=0)  # (2, 2)

    mj_params = MinJerkParams(
        Vmax=tc.Vmax,
        Amax=tc.Amax,
        Jmax=tc.Jmax,
        gamma=tc.gamma_time_scale,
    )
    traj = MinJerkLinearTrajectoryTorch(waypoints, mj_params)

    # ---------- infer obs_dim / act_dim from current env ----------
    geom = env.states["geometry"]  # (B, 2 + dof, M)
    if geom.ndim != 3:
        raise RuntimeError(f"[BC-a Torch] unexpected geometry shape: {tuple(geom.shape)}")
    B, rows, M_muscles = geom.shape
    n_dof = env.skeleton.dof
    if rows != 2 + n_dof:
        raise RuntimeError(
            f"[BC-a Torch] geometry second dim should be 2 + dof = {2 + n_dof}, got {rows}"
        )

    # 14 base + 2*m (len) + 2*m (vel) + m (R per dof=2) + m (a_prev) = 14 + 5m
    obs_dim = 14 + 5 * M_muscles
    act_dim = M_muscles

    # ---------- RL policy ----------
    pi_params = RLPolicyParams(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=str(device),
        out_clip_min=0.0,
        out_clip_max=1.0,
    )
    pi = RLPolicy(pi_params)

    ckpt_path = "models/random_reach_bc_a.pt"
    try:
        pi.load(ckpt_path)
        print(f"[BC-a Torch] loaded policy from {ckpt_path}")
    except FileNotFoundError:
        print(f"[BC-a Torch] WARNING: no checkpoint at {ckpt_path}, using random policy.")
    except Exception as e:
        print(f"[BC-a Torch] WARNING: failed to load checkpoint ({e}), using random policy.")

    # ---------- RL controller ----------
    ctrl_params = RLControllerAParams()
    ctrl = RLControllerATorch(env, arm, pi, ctrl_params)

    # ---------- simulate ----------
    steps = int(pc.max_ep_duration / arm.dt)

    try:
        sim = TargetReachSimulatorTorch(env, arm, ctrl, traj, steps)
        logs = sim.run()

        k, tvec = logs.time(arm.dt)
        x_log = logs.x_log[:k]  # (T, 4) or (T, >=2)

        print("  fingertip (before):", center.detach().cpu().numpy())
        print("  target x_d:        ", target.detach().cpu().numpy())

        for idx in [0, 1, 2, 4, 9, 19, 29]:
            if idx < k:
                x = x_log[idx, :2]
                print(f"    step {idx:3d}: x = {x.detach().cpu().numpy()}")

        x_final = x_log[k - 1, :2]
        err_final = _torch.linalg.norm(x_final - target)
        print("  final fingertip:   ", x_final.detach().cpu().numpy())
        print(f"  final |x - x_d|:    {float(err_final):.6f} m")

    except Exception as e:
        print(f"[BC-a Torch] simulator failed: {e}")
        print("[BC-a Torch] running manual loop...")
        for step in range(min(50, steps)):
            t = step * arm.dt
            t_tensor = _torch.tensor([t], device=traj.device, dtype=traj.dtype)
            x_d, xd_d, xdd_d = traj.sample(t_tensor)
            out = ctrl.compute(x_d, xd_d, xdd_d)
            if step in (0, 1, 2, 5, 10, 20, 30, 40, 49):
                x_cur = out["x"]
                act = out["act"]
                if isinstance(x_cur, _torch.Tensor):
                    x_cur_np = x_cur.detach().cpu().numpy()
                else:
                    x_cur_np = np.asarray(x_cur)
                if isinstance(act, _torch.Tensor):
                    act_mean = float(act.mean().item())
                else:
                    act_mean = float(np.mean(act))
                print(f"  step {step:3d}: x = {x_cur_np}, avg_act = {act_mean:.3f}")

    print("[BC-a Torch] smoke test complete ✓")


if __name__ == "__main__":
    _smoke_test_bc_a_torch()
