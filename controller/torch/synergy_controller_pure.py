# controller/torch/synergy_controller_pure.py
# -*- coding: utf-8 -*-
"""
Pure Muscle Synergy Controller (STRICT) -- RESEARCH BASELINE, torch port of
controller/numpy/synergy_controller_pure.py.

Canonical synergy controller = controller/torch/synergy_controller.py (full:
modules + modulation + residual correction). This *pure* variant deliberately has
NO residual correction: the command is constrained to the non-negative synergy
subspace  a = clip(W c), c >= 0 (NNLS projection of the optimal full-space
activation a_ref onto span(W)). It can transiently STALL on reach directions whose
required activation lies outside the non-negative synergy cone -- that limit is
the whole point of the baseline (see the numpy file's header for the analysis).

Faithful B=1 torch mirror of the numpy controller. Uses the torch HTM dynamics
(env.skeleton.*) and torch guard/muscle helpers.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from utils.torch.kinematics_guard import KinGuardParams, adaptive_dls_pinv
from utils.torch.dynamics_guard import DynGuardParams, op_space_guard_and_gate
from utils.torch.muscle_guard import MuscleGuardParams, solve_muscle_forces
from utils.torch.telemetry import pack_diag, merge_diag
from muscles.torch.muscle_tools import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
)


def _t(x, like: Tensor) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(x, device=like.device, dtype=like.dtype)


def _first(x: Tensor) -> Tensor:
    return x[0] if x.dim() >= 3 else x


def nnls_pgd(W: Tensor, a: Tensor, iters: int = 60) -> Tensor:
    """min_{c>=0} ||Wc - a||^2 via projected gradient descent (faithful to numpy)."""
    a = a.reshape(-1)
    K = W.shape[1]
    c = torch.zeros(K, device=W.device, dtype=W.dtype)
    try:
        L = float(torch.linalg.norm(W, 2)) ** 2 + 1e-12
    except Exception:
        L = float(torch.sum(W * W)) + 1e-12
    lr = 1.0 / L
    WT = W.T
    for _ in range(int(iters)):
        grad = 2.0 * (WT @ (W @ c - a))
        c = torch.clamp(c - lr * grad, min=0.0)
    return c


@dataclass
class SynergyPureParams:
    Kp_task: float = 800.0
    Kv_task: float = 60.0
    Kff_x: float = 1.0

    eps: float = 1e-6
    lam_os_max: float = 1e6
    sigma_thresh: float = 1e-4
    gate_pow: float = 2.0

    enable_gravity_comp: bool = True
    enable_velocity_comp: bool = False
    enable_inertia_comp: bool = False

    W_init: Optional[Any] = None     # (6,K)
    n_synergies: int = 3

    c_max: float = 5.0
    nnls_iters: int = 60
    bisect_iters: int = 12


class SynergyPureController:
    def __init__(self, env, arm, params: SynergyPureParams = None):
        self.env = env
        self.arm = arm
        self.p = params or SynergyPureParams()
        self.n_dof = int(env.skeleton.dof)
        self.dtype = getattr(env, "dtype", torch.get_default_dtype())
        self.device = getattr(env, "device", torch.device("cpu"))

        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=self.p.eps, lam_os_max=self.p.lam_os_max, gate_pow=self.p.gate_pow,
            sigma_thresh_S=max(self.p.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()
        self.qref: Optional[Tensor] = None
        self._Fmax_cache: Optional[Tensor] = None
        self._idx_flpe: Optional[int] = None

        if self.p.W_init is None:
            K = int(self.p.n_synergies)
            g = torch.Generator(device="cpu").manual_seed(0)
            W = torch.abs(torch.randn(6, K, generator=g)).to(device=self.device, dtype=self.dtype)
            W = W / (torch.linalg.norm(W, dim=0, keepdim=True) + 1e-12)
            self.W = W
        else:
            self.W = _t(self.p.W_init, torch.zeros(1, device=self.device, dtype=self.dtype))

    def reset(self, q0: Tensor):
        q0 = q0.to(device=self.device, dtype=self.dtype)
        self.qref = (q0[0] if q0.ndim == 2 else q0).clone()

    def set_W(self, W):
        self.W = _t(W, self.W)

    def get_W(self) -> Tensor:
        return self.W.clone()

    def compute(self, x_d: Tensor, xd_d: Tensor, xdd_d: Tensor) -> Dict[str, Any]:
        joint = self.env.states["joint"]
        n = self.n_dof
        q, qd = joint[0, :n], joint[0, n:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:4]
        self.env.skeleton._set_state(q, qd)
        if self.qref is None:
            self.qref = q.clone()

        x_d = _t(x_d, x).reshape(-1)[:2]
        xd_d = _t(xd_d, x).reshape(-1)[:2]
        xdd_d = _t(xdd_d, x).reshape(-1)[:2]

        # kinematics + guards
        J_xy = _first(self.env.skeleton.geometric_jacobian(q))[0:2, :]
        _, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)

        M = _first(self.env.skeleton.mass_matrix(q))
        Minv = torch.linalg.inv(M)
        S = J_xy @ Minv @ J_xy.T
        _, lam_os, eta, eta2, xd_d_g, xdd_d_g, dyn_diag = op_space_guard_and_gate(
            S, xd_d.clone(), xdd_d.clone(), self.dp
        )
        xd_d_g = xd_d_g.reshape(-1)[:2]
        xdd_d_g = xdd_d_g.reshape(-1)[:2]
        eta = torch.as_tensor(eta, device=q.device, dtype=q.dtype).reshape(())
        eta2 = torch.as_tensor(eta2, device=q.device, dtype=q.dtype).reshape(())

        # task PD force -> joint torque (+ gravity)
        e_x = x_d - x
        e_v = xd_d_g - xd
        F_task = self.p.Kp_task * e_x + self.p.Kv_task * e_v + (self.p.Kff_x * xdd_d_g)
        tau_task = J_xy.T @ F_task
        if self.p.enable_gravity_comp:
            g_any = self.env.skeleton.gravity_vector(q)
            g = (g_any[0] if g_any.dim() >= 3 else g_any).reshape(-1)[:n]
            tau_task = tau_task + g

        # muscle geometry
        geom = self.env.states["geometry"]
        lenvel = geom[:, :2, :]
        R = geom[0, 2:2 + n, :]
        if self._Fmax_cache is None:
            self._Fmax_cache = get_Fmax_vec(self.env, R.shape[1]).to(device=q.device, dtype=q.dtype)
            names = list(self.env.muscle.state_name)
            self._idx_flpe = names.index("force-length PE") if "force-length PE" in names else -1
        Fmax_vec = self._Fmax_cache
        flpe = self.env.states["muscle"][:, self._idx_flpe, :] if self._idx_flpe >= 0 \
            else torch.zeros((1, R.shape[1]), device=q.device, dtype=q.dtype)

        # full-space optimal activation a_ref
        F_ref, mus_diag = solve_muscle_forces(tau_task.unsqueeze(0), R.unsqueeze(0), Fmax_vec, eta.reshape(1), self.mp)
        a_ref = force_to_activation_bisect(F_ref, lenvel, self.env.muscle, flpe, Fmax_vec, iters=int(self.p.bisect_iters))

        # PURE synergy projection: a = clip(W c), c = NNLS(W, a_ref)
        c = nnls_pgd(self.W, a_ref.reshape(-1), iters=int(self.p.nnls_iters))
        c = torch.clamp(c, 0.0, float(self.p.c_max))
        a_min = float(getattr(self.env.muscle, "min_activation", 0.0))
        a_cmd = torch.clamp((self.W @ c).unsqueeze(0), a_min, 1.0)

        # predicted forces/torques (logging)
        af = active_force_from_activation(a_cmd, lenvel, self.env.muscle)
        F_cmd = torch.clamp(Fmax_vec * (af + flpe), min=0.0)
        tau_cmd = (-R @ F_cmd[0]).reshape(-1)

        diag = merge_diag(
            pack_diag(sminJ=sminJ, lamJ=lamJ), dyn_diag, mus_diag,
            pack_diag(lam_os=lam_os, eta=eta, eta2=eta2,
                      c_max=float(torch.max(c)) if c.numel() else 0.0,
                      a_ref_mean=float(torch.mean(a_ref))),
        )
        return {
            "tau_des": tau_cmd.unsqueeze(0), "R": R.unsqueeze(0), "Fmax": Fmax_vec,
            "F_des": F_cmd, "act": a_cmd, "synergy_coeff": c.unsqueeze(0),
            "q": q.unsqueeze(0), "qd": qd.unsqueeze(0), "x": x.unsqueeze(0), "xd": xd.unsqueeze(0),
            "xref_tuple": (x_d, xd_d_g, xdd_d_g), "eta": eta2.reshape(1), "diag": diag,
        }
