# controller/hybrid_mpc_rl.py
import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

import torch
import torch.nn as nn

from utils.muscle_guard import MuscleGuardParams, solve_muscle_forces
from muscles.muscle_tools import (
    get_Fmax_vec, force_to_activation_bisect, active_force_from_activation,
    saturation_repair_tau
)

# ----------------------- small utilities -----------------------

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _build_obs(env, x_d: np.ndarray, xd_d: np.ndarray, xdd_d: np.ndarray) -> np.ndarray:
    """
    14-D observation used for BC/RL:
      O = [q(2), qd(2), x(2), xd(2), e(2), e_dot(2), xdd_d(2)]
    where e = x_d - x, e_dot = xd_d - xd.
    """
    joint = env.states["joint"][0]     # [q1,q2,qd1,qd2]
    cart  = env.states["cartesian"][0] # [x,y,xd,yd]
    q, qd = joint[:2], joint[2:]
    x, xd = cart[:2],  cart[2:]

    x_d   = np.asarray(x_d,   dtype=np.float32).reshape(2)
    xd_d  = np.asarray(xd_d,  dtype=np.float32).reshape(2)
    xdd_d = np.asarray(xdd_d, dtype=np.float32).reshape(2)
    e     = x_d - x
    e_dot = xd_d - xd

    O = np.concatenate([q, qd, x, xd, e, e_dot, xdd_d], axis=0).astype(np.float32)
    return O

# ----------------------- policy & trainer -----------------------

@dataclass
class RLPolicyParams:
    obs_dim: int = 14           # <- updated to 14
    act_dim: int = 2            # joint torques
    hidden: Tuple[int, int] = (128, 128)
    device: str = "cpu"
    tau_clip: float = 600.0
    use_muscles: bool = True    # retained; deployment decides τ→a mapping

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(128,128)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class RLPolicy:
    """
    τ = π(obs). Stores mean/std for whitening; robust .load() that rebuilds the
    network if checkpoint architecture differs (obs_dim or hidden sizes).
    """
    def __init__(self, params: RLPolicyParams,
                 mean: Optional[np.ndarray]=None,
                 std: Optional[np.ndarray]=None):
        self.p = params
        self.device = torch.device(self.p.device)
        self.model = MLP(self.p.obs_dim, self.p.act_dim, self.p.hidden).to(self.device)
        self.mean = torch.zeros(self.p.obs_dim, dtype=torch.float32, device=self.device) if mean is None else torch.tensor(mean, dtype=torch.float32, device=self.device)
        self.std  = torch.ones (self.p.obs_dim, dtype=torch.float32, device=self.device) if std  is None else torch.tensor(std , dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> np.ndarray:
        x = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        xn = (x - self.mean) / (self.std + 1e-6)
        tau = self.model(xn)
        tau = torch.clamp(tau, -self.p.tau_clip, self.p.tau_clip)
        return tau.cpu().numpy()

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
                "tau_clip": self.p.tau_clip,
            },
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        meta = ckpt.get("params", {})
        in_dim  = int(meta.get("obs_dim",  self.p.obs_dim))
        out_dim = int(meta.get("act_dim",  self.p.act_dim))
        hidden  = tuple(meta.get("hidden", list(self.p.hidden)))
        # Rebuild if architecture differs
        if (in_dim != self.p.obs_dim) or (hidden != self.p.hidden) or (out_dim != self.p.act_dim):
            self.p = RLPolicyParams(obs_dim=in_dim, act_dim=out_dim, hidden=hidden,
                                    device=str(self.device), tau_clip=self.p.tau_clip,
                                    use_muscles=self.p.use_muscles)
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
        mean = O.mean(dim=0)
        std  = O.std(dim=0).clamp_min(1e-6)
        self.policy.mean.copy_(mean)
        self.policy.std .copy_(std)

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

# ----------------------- teacher hook -----------------------

class HookedController:
    """
    Wraps the teacher; records (O,A) using the 14-D obs above.
    Also exposes .qref so your simulator logger never breaks.
    """
    def __init__(self, env, teacher):
        self.env = env
        self.teacher = teacher
        self.obs: List[np.ndarray] = []
        self.act: List[np.ndarray] = []
        self._qref = None

    @property
    def qref(self):
        return getattr(self.teacher, "qref", self._qref)

    def reset(self, q0):
        self.obs.clear(); self.act.clear()
        if hasattr(self.teacher, "reset"):
            self.teacher.reset(q0)
        self._qref = None

    def compute(self, x_d, xd_d, xdd_d):
        diag = self.teacher.compute(x_d, xd_d, xdd_d)
        O = _build_obs(self.env, x_d, xd_d, xdd_d)                 # 14-D
        tau_des = np.asarray(diag["tau_des"], dtype=np.float32).reshape(2)
        self.obs.append(O); self.act.append(tau_des)
        self._qref = getattr(self.teacher, "qref", None)
        return diag

# ----------------------- RL controller (deployment) -----------------------

@dataclass
class RLControllerParams:
    tau_clip: float = 600.0
    use_muscles: bool = True
    bisect_iters: int = 18

class RLController:
    """
    Same interface as analytic controllers; produces τ via π(O).
    Always exposes .qref for the simulator.
    """
    def __init__(self, env, arm, policy: RLPolicy, params: RLControllerParams):
        self.env = env
        self.arm = arm
        self.pi  = policy
        self.p   = params
        self.mp  = MuscleGuardParams()
        self._qref = None

    @property
    def qref(self):
        return self._qref

    def reset(self, q0):
        self._qref = None

    def compute(self, x_d, xd_d, xdd_d):
        # 1) 14-D obs (state + reference)
        obs = _build_obs(self.env, x_d, xd_d, xdd_d)
        # 2) policy → τ
        tau_des = self.pi.act(obs).astype(np.float32)
        tau_des = np.clip(tau_des, -self.p.tau_clip, self.p.tau_clip)

        if not self.p.use_muscles:
            q, qd, x, xd = obs[0:2], obs[2:4], obs[4:6], obs[6:8]
            return {
                "tau_des": tau_des, "R": None, "Fmax": None, "F_des": None, "act": None,
                "q": q, "qd": qd, "x": x, "xd": xd,
                "xref_tuple": (np.asarray(x_d)[:2], np.asarray(xd_d)[:2], np.asarray(xdd_d)[:2]),
                "eta": 1.0, "diag": {}
            }

        # 3) τ → F → a  (same robust path you use elsewhere)
        geom   = self.env.states["geometry"]
        lenvel = geom[:, :2, :]
        Rm     = geom[:, 2:2 + self.env.skeleton.dof, :][0]
        Fmax_v = get_Fmax_vec(self.env, Rm.shape[1])

        F_des, _ = solve_muscle_forces(tau_des, Rm, Fmax_v, 1.0, self.mp)
        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe  = self.env.states["muscle"][0, idx_flpe, :]
        a     = force_to_activation_bisect(F_des, lenvel, self.env.muscle, flpe, Fmax_v,
                                           iters=self.p.bisect_iters)
        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = Fmax_v * (af_now + flpe)
        F_corr = saturation_repair_tau(-Rm, F_pred, a,
                                       self.env.muscle.min_activation, 1.0,
                                       Fmax_v, tau_des=tau_des)
        if np.any(np.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(F_corr, lenvel, self.env.muscle, flpe, Fmax_v,
                                           iters=max(4, self.p.bisect_iters - 4))

        q, qd, x, xd = obs[0:2], obs[2:4], obs[4:6], obs[6:8]
        return {
            "tau_des": tau_des, "R": Rm, "Fmax": Fmax_v, "F_des": F_des, "act": a,
            "q": q, "qd": qd, "x": x, "xd": xd,
            "xref_tuple": (np.asarray(x_d)[:2], np.asarray(xd_d)[:2], np.asarray(xdd_d)[:2]),
            "eta": 1.0, "diag": {}
        }
