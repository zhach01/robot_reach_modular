# controller/hybrid_mpc_rl_torch.py
# -*- coding: utf-8 -*-
"""
Torch version of hybrid_mpc_rl.py

- Uses Torch env (EnvironmentTorch) with:
    env.states["joint"]     : (B, 2*dof)  -> [q, qd]
    env.states["cartesian"] : (B, >=4)    -> [x, y, xd, yd, ...]
    env.states["geometry"]  : (B, 2 + dof, M)
    env.states["muscle"]    : (B, state_dim, M)

- Uses muscles.muscle_tools_torch + utils.muscle_guard_torch for τ→F→a.
- RL policy is the same small MLP, but everything stays in torch on the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Any

import os
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

from utils.muscle_guard_torch import MuscleGuardParams, solve_muscle_forces
from muscles.muscle_tools_torch import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
)


# ---------------------------------------------------------------------
# Small filesystem utility
# ---------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------
# RL policy (unchanged logic, torch-native)
# ---------------------------------------------------------------------


@dataclass
class RLPolicyParams:
    obs_dim: int = 14        # 14-D obs: [q, qd, x, xd, e, e_dot, xdd_d]
    act_dim: int = 2         # joint torques
    hidden: Tuple[int, int] = (128, 128)
    device: str = "cpu"
    tau_clip: float = 600.0
    use_muscles: bool = True   # deployment decides τ→a mapping


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, int] = (128, 128)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RLPolicy:
    """
    τ = π(obs). Stores mean/std for whitening; robust .load() that rebuilds the
    network if checkpoint architecture differs (obs_dim or hidden sizes).
    """

    def __init__(
        self,
        params: RLPolicyParams,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.p = params
        self.device = torch.device(self.p.device)

        # always keep the policy in float32
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
        NumPy API (for legacy CPU envs / offline use).
        """
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).view(
            -1, self.p.obs_dim
        )
        xn = (x - self.mean) / (self.std + 1e-6)
        tau = self.model(xn)
        tau = torch.clamp(tau, -self.p.tau_clip, self.p.tau_clip)
        return tau.cpu().numpy().reshape(-1)

    @torch.no_grad()
    def act_torch(self, obs_t: Tensor) -> Tensor:
        """
        Torch API: obs_t shape (..., obs_dim) on any device.
        Returns τ as a Tensor with same leading dims (float32).
        """
        x = obs_t.to(device=self.device, dtype=torch.float32)
        if x.ndim == 1:
            x = x.view(1, -1)
        xn = (x - self.mean) / (self.std + 1e-6)
        tau = self.model(xn)
        tau = torch.clamp(tau, -self.p.tau_clip, self.p.tau_clip)
        return tau.squeeze(0)

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
                    "tau_clip": self.p.tau_clip,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        meta = ckpt.get("params", {})
        in_dim = int(meta.get("obs_dim", self.p.obs_dim))
        out_dim = int(meta.get("act_dim", self.p.act_dim))
        hidden = tuple(meta.get("hidden", list(self.p.hidden)))

        # Rebuild if architecture differs
        if (in_dim != self.p.obs_dim) or (hidden != self.p.hidden) or (
            out_dim != self.p.act_dim
        ):
            self.p = RLPolicyParams(
                obs_dim=in_dim,
                act_dim=out_dim,
                hidden=hidden,
                device=str(self.device),
                tau_clip=self.p.tau_clip,
                use_muscles=self.p.use_muscles,
            )
            self.model = MLP(in_dim, out_dim, hidden).to(self.device)
            self.model = self.model.to(torch.float32)
            self.mean = torch.zeros(in_dim, dtype=torch.float32, device=self.device)
            self.std = torch.ones(in_dim, dtype=torch.float32, device=self.device)

        self.model.load_state_dict(ckpt["state_dict"])
        self.mean = torch.tensor(ckpt["mean"], dtype=torch.float32, device=self.device)
        self.std = torch.tensor(ckpt["std"], dtype=torch.float32, device=self.device)


# ---------------------------------------------------------------------
# Simple BC trainer (unchanged logic; accepts NumPy datasets)
# ---------------------------------------------------------------------


class BehaviorCloner:
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

        mean = O_t.mean(dim=0)
        std = O_t.std(dim=0).clamp_min(1e-6)

        # update policy whitening stats
        self.policy.mean.copy_(mean)
        self.policy.std.copy_(std)

        N = O_t.shape[0]
        idx = torch.arange(N, device=device)

        for _ in range(epochs):
            if shuffle:
                idx = idx[torch.randperm(N, device=device)]
            for i in range(0, N, batch_size):
                b = idx[i : min(i + batch_size, N)]
                x = (O_t[b] - mean) / (std + 1e-6)
                y = A_t[b]
                yhat = self.policy.model(x)
                l = self.loss(yhat, y)
                self.opt.zero_grad()
                l.backward()
                self.opt.step()


# ---------------------------------------------------------------------
# Teacher hook for Torch envs
# ---------------------------------------------------------------------


class HookedControllerTorch:
    """
    Wraps a teacher controller; records (O,A) using the 14-D obs:
      O = [q, qd, x, xd, e, e_dot, xdd_d]
    with e = x_d - x, e_dot = xd_d - xd.
    Stores obs/act as NumPy arrays for BC.
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
        if hasattr(self.teacher, "reset"):
            self.teacher.reset(q0)
        self._qref = None

    def compute(self, x_d: Any, xd_d: Any, xdd_d: Any) -> Dict[str, Any]:
        diag = self.teacher.compute(x_d, xd_d, xdd_d)

        # Build 14-D obs from Torch env, then store as NumPy
        joint = self.env.states["joint"][0]     # (2*dof,)
        cart = self.env.states["cartesian"][0]  # (>=4,)
        n = self.env.skeleton.dof

        q = joint[:n]
        qd = joint[n:]
        x = cart[:2]
        xd = cart[2:4]

        device = q.device
        dtype = q.dtype

        x_d_t = torch.as_tensor(x_d, device=device, dtype=dtype).view(-1)[:2]
        xd_d_t = torch.as_tensor(xd_d, device=device, dtype=dtype).view(-1)[:2]
        xdd_d_t = torch.as_tensor(xdd_d, device=device, dtype=dtype).view(-1)[:2]

        e = x_d_t - x
        e_dot = xd_d_t - xd

        O_t = torch.cat([q, qd, x, xd, e, e_dot, xdd_d_t], dim=0).to(torch.float32)
        O = O_t.detach().cpu().numpy()

        tau_des = diag["tau_des"]
        if isinstance(tau_des, torch.Tensor):
            tau_np = tau_des.detach().cpu().numpy().reshape(-1).astype(np.float32)
        else:
            tau_np = np.asarray(tau_des, dtype=np.float32).reshape(-1)

        self.obs.append(O)
        self.act.append(tau_np)

        self._qref = getattr(self.teacher, "qref", None)
        return diag


# ---------------------------------------------------------------------
# RL controller for Torch envs
# ---------------------------------------------------------------------


@dataclass
class RLControllerParams:
    tau_clip: float = 600.0
    use_muscles: bool = True
    bisect_iters: int = 18


class RLControllerTorch:
    """
    Torch env controller that uses an RLPolicy to produce τ, then maps τ→F→a
    through the same muscle stack as analytic controllers.

    Interface:
      - compute(x_d, xd_d, xdd_d) -> diag dict
      - exposes .qref for simulator compatibility
    """

    def __init__(self, env, arm, policy: RLPolicy, params: RLControllerParams):
        self.env = env
        self.arm = arm
        self.pi = policy
        self.p = params
        self.mp = MuscleGuardParams()
        self._qref: Optional[Tensor] = None

    @property
    def qref(self):
        return self._qref

    def reset(self, q0: Tensor) -> None:
        """
        Keep qref compatible with other controllers:
        store the joint-position part of q0.
        """
        if q0 is not None:
            if q0.ndim > 1:
                q0 = q0[0]
            n = self.env.skeleton.dof
            self._qref = q0[:n].detach().clone()
        else:
            self._qref = None

    def compute(self, x_d: Any, xd_d: Any, xdd_d: Any) -> Dict[str, Any]:
        # --- State from Torch env (use first batch element) ---
        joint = self.env.states["joint"]
        cart = self.env.states["cartesian"]

        if joint.ndim == 1:
            joint = joint.unsqueeze(0)
        if cart.ndim == 1:
            cart = cart.unsqueeze(0)

        n = self.env.skeleton.dof
        q = joint[0, :n]
        qd = joint[0, n:]
        x = cart[0, :2]
        xd = cart[0, 2:4]

        # keep qref as current posture so simulator/logger never sees None
        self._qref = q.detach().clone()

        device = q.device
        dtype = q.dtype

        # --- Build 14-D obs O = [q, qd, x, xd, e, e_dot, xdd_d] ---
        x_d_t = torch.as_tensor(x_d, device=device, dtype=dtype).view(-1)[:2]
        xd_d_t = torch.as_tensor(xd_d, device=device, dtype=dtype).view(-1)[:2]
        xdd_d_t = torch.as_tensor(xdd_d, device=device, dtype=dtype).view(-1)[:2]

        e = x_d_t - x
        e_dot = xd_d_t - xd

        O_t = torch.cat([q, qd, x, xd, e, e_dot, xdd_d_t], dim=0).to(torch.float32)

        # --- Policy: obs -> τ (Torch) ---
        tau_des = self.pi.act_torch(O_t).to(device=device, dtype=dtype)
        tau_des = torch.clamp(tau_des, -self.p.tau_clip, self.p.tau_clip)

        # optional torque-only mode (no muscles)
        if not self.p.use_muscles:
            return {
                "tau_des": tau_des,
                "R": None,
                "Fmax": None,
                "F_des": None,
                "act": None,
                "q": q,
                "qd": qd,
                "x": x,
                "xd": xd,
                "xref_tuple": (x_d_t, xd_d_t, xdd_d_t),
                "eta": 1.0,
                "diag": {},
            }

        # --- τ -> F -> a, using Torch muscle stack ---
        geom = self.env.states["geometry"]              # (B, 2 + dof, M)
        if geom.ndim != 3:
            raise ValueError(
                f"env.states['geometry'] must be (B, 2 + dof, M), got {tuple(geom.shape)}"
            )

        lenvel = geom[:, :2, :]                         # (B,2,M)
        Rm = geom[:, 2 : 2 + n, :][0]                   # (n,M)
        M_muscles = int(Rm.shape[-1])

        Fmax_v = get_Fmax_vec(
            self.env,
            M_muscles,
            device=Rm.device,
            dtype=Rm.dtype,
        )  # (M,)

        # allocation (no internal-force regulation here)
        F_des, _mus_diag = solve_muscle_forces(
            tau_des, Rm, Fmax_v, 1.0, self.mp
        )  # (M,)

        names = list(getattr(self.env.muscle, "state_name", []))
        if "force-length PE" in names:
            idx_flpe = names.index("force-length PE")
            flpe = self.env.states["muscle"][0, idx_flpe, :]  # (M,)
        else:
            flpe = torch.zeros(M_muscles, device=device, dtype=dtype)

        a = force_to_activation_bisect(
            F_des,
            lenvel,
            self.env.muscle,
            flpe,
            Fmax_v,
            iters=int(self.p.bisect_iters),
        )  # (M,)

        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = Fmax_v * (af_now + flpe)

        F_corr = saturation_repair_tau(
            -Rm,
            F_pred,
            a,
            self.env.muscle.min_activation,
            1.0,
            Fmax_v,
            tau_des=tau_des,
        )

        if torch.any(torch.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(
                F_corr,
                lenvel,
                self.env.muscle,
                flpe,
                Fmax_v,
                iters=max(4, int(self.p.bisect_iters) - 4),
            )

        return {
            "tau_des": tau_des,
            "R": Rm,
            "Fmax": Fmax_v,
            "F_des": F_des,
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
# Smoke test: RL-only controller on Torch plant/env
# ---------------------------------------------------------------------


def _smoke_test_hybrid_rl():
    """
    Smoke test for RLControllerTorch on RigidTendonArm26 + Hill muscles.
    Mirrors the NMPC Torch smoke test: single min-jerk reach.
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

    print("\n[Hybrid RL Torch] smoke test starting ...")
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
        name="HybridRL_ReachEnvTorch",
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

    # ---------- RL policy ----------
    pi_params = RLPolicyParams(device=str(device))
    pi = RLPolicy(pi_params)

    ckpt_path = "models/random_reach_bc.pt"
    try:
        pi.load(ckpt_path)
        print(f"[Hybrid RL Torch] loaded policy from {ckpt_path}")
    except FileNotFoundError:
        print(f"[Hybrid RL Torch] WARNING: no checkpoint at {ckpt_path}, using random policy.")
    except Exception as e:
        print(f"[Hybrid RL Torch] WARNING: failed to load checkpoint ({e}), using random policy.")

    # ---------- RL controller ----------
    rl_params = RLControllerParams(
        tau_clip=600.0,
        use_muscles=True,
        bisect_iters=ifc.bisect_iters,
    )
    ctrl = RLControllerTorch(env, arm, pi, rl_params)

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
        print(f"[Hybrid RL Torch] simulator failed: {e}")
        print("[Hybrid RL Torch] running manual loop...")
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
                print(
                    f"  step {step:3d}: x = {x_cur_np}, avg_act = {act_mean:.3f}"
                )

    print("[Hybrid RL Torch] smoke test complete ✓")


if __name__ == "__main__":
    _smoke_test_hybrid_rl()
