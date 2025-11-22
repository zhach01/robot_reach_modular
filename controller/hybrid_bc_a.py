#!/usr/bin/env python3
# controller/hybrid_bc_a.py
"""
Behavior-cloned activation policy (a-policy).

Obs = [q, qd, x, xd, e, e_dot, xdd_d, l_norm(m), v_norm(m), R(2*m), a_prev(m)]
   -> 14 + 5m dims (m = number of muscles)

Action = a (m,)  -- final muscle activations in [a_min, 1].

This file provides:
- RLPolicyParams, MLP, RLPolicy       : generic MLP with robust .load()
- BehaviorCloner                      : MSE trainer with whitening
- obs builders                        : _build_obs14(), _build_obs_a()
- HookedControllerA                   : wraps a teacher, records (O, a)
- RLControllerA                       : deploys a-policy (outputs activations)

Works with your Environment/Simulator as long as controller.compute(...)
returns a dict containing at least {"act": a}. Here we also reconstruct
tau_des from a so the logger can compute residuals.
"""
from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn

# For τ reconstruction from activations
from muscles.muscle_tools import (
    get_Fmax_vec,
    active_force_from_activation,
)

# ----------------------- small utils -----------------------

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _build_obs14(env, x_d, xd_d, xdd_d) -> np.ndarray:
    """14-D task+state features used by both τ- and a-pipelines."""
    joint = env.states["joint"][0]       # [q1,q2,qd1,qd2]
    cart  = env.states["cartesian"][0]   # [x,y,xd,yd]
    q, qd = joint[:2], joint[2:]
    x, xd = cart[:2],  cart[2:]

    x_d   = np.asarray(x_d,   np.float32).reshape(2)
    xd_d  = np.asarray(xd_d,  np.float32).reshape(2)
    xdd_d = np.asarray(xdd_d, np.float32).reshape(2)
    e     = x_d  - x
    e_dot = xd_d - xd

    return np.concatenate([q, qd, x, xd, e, e_dot, xdd_d], axis=0).astype(np.float32)


def _build_obs_a(env, x_d, xd_d, xdd_d) -> np.ndarray:
    """
    14 + 5m features to resolve redundancy and activation dynamics:
      [ q(2) qd(2) x(2) xd(2) e(2) e_dot(2) xdd_d(2)  l_norm(m) v_norm(m)  R(2*m)  a_prev(m) ]
    """
    base = _build_obs14(env, x_d, xd_d, xdd_d)  # 14
    geom0 = env.states["geometry"][0]           # (2 + DOF, m)
    lenvel = geom0[:2, :]                       # (2, m) -> [l_norm; v_norm]
    Rm = geom0[2:2 + env.skeleton.dof, :]       # (DOF, m)

    names = env.muscle.state_name
    idx_act = names.index("activation")
    a_prev = env.states["muscle"][0, idx_act, :]  # (m,)

    obs = np.concatenate([
        base,
        lenvel.reshape(-1),
        Rm.reshape(-1),
        a_prev.reshape(-1),
    ]).astype(np.float32)
    return obs


# ----------------------- policy & trainer -----------------------

@dataclass
class RLPolicyParams:
    obs_dim: int
    act_dim: int
    hidden: Tuple[int, int] = (128, 128)
    device: str = "cpu"
    out_clip_min: float = 0.0    # clamp to [min_activation, 1] later
    out_clip_max: float = 1.0


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(128, 128)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class RLPolicy:
    """
    a = π(obs). Stores mean/std for whitening, and rebuilds the network on load
    if checkpoint architecture (obs_dim/hidden/act_dim) differs.
    """
    def __init__(self, params: RLPolicyParams, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.p = params
        self.device = torch.device(self.p.device)
        self.model = MLP(self.p.obs_dim, self.p.act_dim, self.p.hidden).to(self.device)
        self.mean = torch.zeros(self.p.obs_dim, dtype=torch.float32, device=self.device) if mean is None else torch.tensor(mean, dtype=torch.float32, device=self.device)
        self.std  = torch.ones (self.p.obs_dim, dtype=torch.float32, device=self.device) if std  is None else torch.tensor(std , dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> np.ndarray:
        x = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        xn = (x - self.mean) / (self.std + 1e-6)
        a = self.model(xn)
        a = torch.clamp(a, self.p.out_clip_min, self.p.out_clip_max)
        return a.cpu().numpy()

    def save(self, path: str):
        _ensure_dir(path)
        torch.save({
            "state_dict": self.model.state_dict(),
            "mean": self.mean.cpu().numpy(),
            "std":  self.std.cpu().numpy(),
            "params": {
                "obs_dim": self.p.obs_dim,
                "act_dim": self.p.act_dim,
                "hidden":  list(self.p.hidden),
                "out_clip_min": float(self.p.out_clip_min),
                "out_clip_max": float(self.p.out_clip_max),
            },
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        ck_p = ckpt.get("params", {})
        in_dim  = int(ck_p.get("obs_dim",  self.p.obs_dim))
        out_dim = int(ck_p.get("act_dim",  self.p.act_dim))
        hidden  = tuple(ck_p.get("hidden", list(self.p.hidden)))
        if (in_dim != self.p.obs_dim) or (hidden != self.p.hidden) or (out_dim != self.p.act_dim):
            self.p = RLPolicyParams(
                obs_dim=in_dim, act_dim=out_dim, hidden=hidden,
                device=str(self.device),
                out_clip_min=float(ck_p.get("out_clip_min", 0.0)),
                out_clip_max=float(ck_p.get("out_clip_max", 1.0)),
            )
            self.model = MLP(in_dim, out_dim, hidden).to(self.device)
            self.mean = torch.zeros(in_dim, dtype=torch.float32, device=self.device)
            self.std  = torch.ones (in_dim, dtype=torch.float32, device=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.mean = torch.tensor(ckpt["mean"], dtype=torch.float32, device=self.device)
        self.std  = torch.tensor(ckpt["std"],  dtype=torch.float32, device=self.device)


class BehaviorCloner:
    def __init__(self, policy: RLPolicy, lr=1e-3, wd=0.0):
        self.policy = policy
        self.opt = torch.optim.Adam(self.policy.model.parameters(), lr=lr, weight_decay=wd)
        self.loss = nn.MSELoss()

    def fit(self, O: np.ndarray, A: np.ndarray, epochs=20, batch_size=1024, shuffle=True):
        device = self.policy.device
        O = torch.tensor(O, dtype=torch.float32, device=device)
        A = torch.tensor(A, dtype=torch.float32, device=device)

        # whitening
        mean = O.mean(dim=0)
        std  = O.std(dim=0).clamp_min(1e-6)
        self.policy.mean.copy_(mean)
        self.policy.std.copy_(std)

        N = O.shape[0]
        idx = torch.arange(N, device=device)
        for ep in range(epochs):
            if shuffle:
                idx = idx[torch.randperm(N, device=device)]
            for i in range(0, N, batch_size):
                b = idx[i:min(i+batch_size, N)]
                x = (O[b] - mean) / (std + 1e-6)
                y = A[b]
                yhat = self.policy.model(x)
                l = self.loss(yhat, y)
                self.opt.zero_grad(); l.backward(); self.opt.step()


# ----------------------- teacher wrapper & RL controller -----------------------

class HookedControllerA:
    """
    Wraps a 'teacher' that already computes final activations (diag['act']).
    Records (O, a) pairs for dataset creation using the a-policy obs.
    Exposes .qref for compatibility with your simulator logger.
    """
    def __init__(self, env, teacher):
        self.env = env
        self.teacher = teacher
        self.obs: list[np.ndarray] = []
        self.act: list[np.ndarray] = []
        self._qref = None

    @property
    def qref(self):
        return getattr(self.teacher, "qref", self._qref)

    def reset(self, q0):
        self.obs.clear(); self.act.clear()
        self._qref = None
        if hasattr(self.teacher, "reset"):
            self.teacher.reset(q0)

    def compute(self, x_d, xd_d, xdd_d):
        diag = self.teacher.compute(x_d, xd_d, xdd_d)
        O = _build_obs_a(self.env, x_d, xd_d, xdd_d)
        a = np.asarray(diag["act"], dtype=np.float32).reshape(-1)
        self.obs.append(O); self.act.append(a)
        self._qref = getattr(self.teacher, "qref", None)
        return diag


@dataclass
class RLControllerAParams:
    pass  # reserved for future use


class RLControllerA:
    """
    Deployment controller: a = π(obs_a).
    Returns a dict compatible with your Simulator/Logger.
    Reconstructs tau_des from predicted activations for logging.
    """
    def __init__(self, env, arm, policy: RLPolicy, params: RLControllerAParams):
        self.env = env
        self.arm = arm
        self.pi  = policy
        self.p   = params
        self._qref = None  # for logger

    @property
    def qref(self):
        return self._qref

    def reset(self, q0):
        self._qref = None

    def compute(self, x_d, xd_d, xdd_d):
        # 1) Build observation and predict activations
        obs = _build_obs_a(self.env, x_d, xd_d, xdd_d)
        a = self.pi.act(obs).astype(np.float32)                      # (m,)
        a = np.clip(a, self.env.muscle.min_activation, 1.0)          # safety clamp

        # 2) Reconstruct τ from activations (keep batch dims where required)
        geom = self.env.states["geometry"]                           # (B=1, 2+DOF, m)
        lenvel_b = geom[:, :2, :]                                    # (1, 2, m)
        Rm = geom[0, 2:2 + self.env.skeleton.dof, :]                 # (DOF, m)
        m = Rm.shape[1]

        # passive FL-PE (fallback to zeros if absent)
        names = self.env.muscle.state_name
        try:
            idx_flpe = names.index("force-length PE")
            flpe = self.env.states["muscle"][0, idx_flpe, :].astype(np.float32)  # (m,)
        except (ValueError, IndexError):
            flpe = np.zeros(m, dtype=np.float32)

        Fmax_v = get_Fmax_vec(self.env, m).astype(np.float32)        # (m,)

        # ---- IMPORTANT: active_force_from_activation expects batch dims ----
        a_b = a.reshape(1, -1)                                       # (1, m)
        af_now = active_force_from_activation(a_b, lenvel_b, self.env.muscle)[0]  # (m,)

        F_pred = Fmax_v * (af_now + flpe)                            # (m,)
        tau_hat = - (Rm @ F_pred).astype(np.float32)                 # (DOF,)

        # 3) Common diag payload
        base = _build_obs14(self.env, x_d, xd_d, xdd_d)
        q, qd, x, xd = base[0:2], base[2:4], base[4:6], base[6:8]

        return {
            "tau_des": tau_hat, "R": Rm, "Fmax": Fmax_v, "F_des": F_pred,
            "act": a,
            "q": q, "qd": qd, "x": x, "xd": xd,
            "xref_tuple": (
                np.asarray(x_d,  dtype=np.float32)[:2],
                np.asarray(xd_d, dtype=np.float32)[:2],
                np.asarray(xdd_d,dtype=np.float32)[:2],
            ),
            "eta": 1.0, "diag": {}
        }
