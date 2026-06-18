# controller/torch/osc_controller.py
# -*- coding: utf-8 -*-
"""
Pure-Torch Operational-Space Control (OSC) controller -- a faithful port of the
numpy controller (controller/numpy/osc_controller.py).

Control law (Khatib operational-space):
  Lambda = (J M^-1 J^T + d I)^-1        (damped op-space inertia)
  J_bar  = M^-1 J^T Lambda              (dynamically-consistent inverse)
  N      = I - J_bar J                  (nullspace projector)
  xdd_cmd = Kp e_x + Kv e_v + xdd_d
  F_task  = Lambda xdd_cmd + mu,  mu = Lambda (J M^-1 h - Jdot qd)
  tau = J^T F_task + N^T (tau_posture + tau_limits)
then torque-feasibility cap + robust muscle inversion.

Operates on B=1 state from the torch simulator. Uses the torch HTM dynamics
(env.skeleton.*) and the torch muscle/guard helpers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import torch
from torch import Tensor

from muscles.torch.muscle_tools import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
)
from utils.torch.muscle_guard import MuscleGuardParams, solve_muscle_forces


def _t(x, like: Tensor) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(x, device=like.device, dtype=like.dtype)


def _first(x: Tensor) -> Tensor:
    return x[0] if x.dim() >= 3 else x


@dataclass
class OSCParams:
    Kp: float = 1000.0
    Kv: float = 80.0
    Kp_posture: float = 50.0
    Kv_posture: float = 10.0
    enable_joint_limits: bool = True
    joint_limit_gain: float = 20.0
    joint_limit_buffer: float = 0.1
    damping: float = 0.01
    eps: float = 1e-6
    bisect_iters: int = 12
    q_min: Any = field(default_factory=lambda: [-0.5, 0.1])
    q_max: Any = field(default_factory=lambda: [2.5, 2.8])


class OSCController:
    def __init__(self, env, arm, params: OSCParams = None):
        self.env = env
        self.arm = arm
        self.p = params or OSCParams()
        self.dt = float(arm.dt)
        self.n_dof = int(env.skeleton.dof)
        self.qref: Tensor | None = None
        self.q_posture: Tensor | None = None
        self.J_prev: Tensor | None = None
        self._Fmax_cache: Tensor | None = None
        self._idx_flpe: int | None = None
        self.mp = MuscleGuardParams()

    def reset(self, q0: Tensor) -> None:
        q0 = q0.to(dtype=torch.get_default_dtype())
        self.qref = (q0[0] if q0.ndim == 2 else q0).clone()
        self.q_posture = self.qref.clone()
        self.J_prev = None

    def _posture_torque(self, q: Tensor, qd: Tensor) -> Tensor:
        if self.q_posture is None:
            return torch.zeros_like(q)
        return self.p.Kp_posture * (self.q_posture - q) - self.p.Kv_posture * qd

    def _joint_limit_torque(self, q: Tensor, qd: Tensor) -> Tensor:
        tau = torch.zeros_like(q)
        if not self.p.enable_joint_limits:
            return tau
        q_min = _t(self.p.q_min, q)
        q_max = _t(self.p.q_max, q)
        buf, k = float(self.p.joint_limit_buffer), float(self.p.joint_limit_gain)
        dist_min = q - q_min
        dist_max = q_max - q
        tau = tau + torch.where(dist_min < buf, k * (buf - dist_min), torch.zeros_like(q))
        tau = tau - torch.where(dist_max < buf, k * (buf - dist_max), torch.zeros_like(q))
        return tau

    def compute(self, x_d: Tensor, xd_d: Tensor, xdd_d: Tensor) -> Dict[str, Any]:
        joint = self.env.states["joint"]
        n = self.n_dof
        q, qd = joint[0, :n], joint[0, n:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:4]
        self.env.skeleton._set_state(q, qd)
        if self.qref is None:
            self.qref = q.clone()
        if self.q_posture is None:
            self.q_posture = self.qref.clone()

        x_d = _t(x_d, x).reshape(-1)[:2]
        xd_d = _t(xd_d, x).reshape(-1)[:2]
        xdd_d = _t(xdd_d, x).reshape(-1)[:2]

        # --- kinematics / dynamics ---
        J_xy = _first(self.env.skeleton.geometric_jacobian(q))[0:2, :]   # (2,n)
        M = _first(self.env.skeleton.mass_matrix(q))                     # (n,n)
        Minv = torch.linalg.inv(M)
        C = _first(self.env.skeleton.coriolis_matrix(q, qd))             # (n,n)
        g_any = self.env.skeleton.gravity_vector(q)
        g = (g_any[0] if g_any.dim() >= 3 else g_any).reshape(-1)[:n]
        h = C @ qd + g

        # op-space inertia (damped), dyn-consistent inverse, nullspace projector
        I2 = torch.eye(2, device=q.device, dtype=q.dtype)
        Lambda = torch.linalg.inv(J_xy @ Minv @ J_xy.T + self.p.damping * I2)
        J_bar = Minv @ J_xy.T @ Lambda
        N = torch.eye(n, device=q.device, dtype=q.dtype) - J_bar @ J_xy

        # Jacobian derivative (finite difference)
        if self.J_prev is None:
            Jdot = torch.zeros_like(J_xy)
        else:
            Jdot = (J_xy - self.J_prev) / self.dt
        self.J_prev = J_xy.clone()

        # --- operational-space control law ---
        e_x = x_d - x
        e_v = xd_d - xd
        xdd_cmd = self.p.Kp * e_x + self.p.Kv * e_v + xdd_d
        mu = Lambda @ (J_xy @ (Minv @ h) - Jdot @ qd)
        F_task = Lambda @ xdd_cmd + mu

        tau_null = self._posture_torque(q, qd) + self._joint_limit_torque(q, qd)
        tau_total = J_xy.T @ F_task + N.T @ tau_null

        # --- muscle geometry + torque-feasibility cap ---
        geom = self.env.states["geometry"]
        lenvel = geom[:, :2, :]
        R = geom[0, 2:2 + n, :]                                          # (n,M)
        if self._Fmax_cache is None:
            self._Fmax_cache = get_Fmax_vec(self.env, R.shape[1]).to(device=q.device, dtype=q.dtype)
            names = list(self.env.muscle.state_name)
            self._idx_flpe = names.index("force-length PE") if "force-length PE" in names else -1
        Fmax_vec = self._Fmax_cache
        tau_cap = 0.95 * (torch.abs(R) @ Fmax_vec)
        tau_capped = torch.clamp(tau_total, -tau_cap, tau_cap)

        flpe = self.env.states["muscle"][:, self._idx_flpe, :] if self._idx_flpe >= 0 \
            else torch.zeros((1, R.shape[1]), device=q.device, dtype=q.dtype)
        eta_b = torch.ones(1, device=q.device, dtype=q.dtype)
        F_des, mus_diag = solve_muscle_forces(tau_capped.unsqueeze(0), R.unsqueeze(0), Fmax_vec, eta_b, self.mp)
        a = force_to_activation_bisect(F_des, lenvel, self.env.muscle, flpe, Fmax_vec, iters=int(self.p.bisect_iters))
        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = torch.clamp(Fmax_vec * (af_now + flpe), min=0.0)
        F_corr = saturation_repair_tau(-R.unsqueeze(0), F_pred, a, self.env.muscle.min_activation, 1.0,
                                       Fmax_vec, tau_des=tau_capped.unsqueeze(0))
        if torch.any(torch.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(F_corr, lenvel, self.env.muscle, flpe, Fmax_vec,
                                           iters=max(4, int(self.p.bisect_iters) - 4))

        return {
            "tau_des": tau_total.unsqueeze(0), "R": R.unsqueeze(0), "Fmax": Fmax_vec,
            "F_des": F_des, "act": a, "q": q.unsqueeze(0), "qd": qd.unsqueeze(0),
            "x": x.unsqueeze(0), "xd": xd.unsqueeze(0),
            "xref_tuple": (x_d, xd_d, xdd_d), "eta": eta_b, "diag": {},
        }
