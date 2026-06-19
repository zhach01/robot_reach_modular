# controller/torch/synergy_controller.py
# -*- coding: utf-8 -*-
"""
Pure-Torch muscle-synergy controller -- a faithful B=1 mirror of the canonical
numpy controller (controller/numpy/synergy_controller.py, the bake-off winner).

Structure (modules + modulation + residual correction):
  1) task-level OSC/PD torque target tau_cmd (same guards as the other controllers)
  2) synergy activation a_syn = clip(W c), c = NNLS-projection of the optimal
     full-space activation a_opt into span(W>=0)
  3) residual torque tau_res = tau_cmd - tau(a_syn), solved in full muscle space
  4) F_final = (beta*F_syn + (1-beta)*F_opt) + F_res -> activation + saturation repair

The residual correction is what makes tracking robust even when W is imperfect
(see archive/.. note: the *pure* synergy variant has no residual and can stall on
reach directions outside the non-negative synergy cone).

Operates on B=1 state from the torch simulator. Uses the torch HTM dynamics
(env.skeleton.*) and the torch guard/muscle helpers.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from utils.torch.kinematics_guard import KinGuardParams, adaptive_dls_pinv, scale_task_by_J
from utils.torch.dynamics_guard import DynGuardParams, op_space_guard_and_gate
from utils.torch.muscle_guard import MuscleGuardParams, solve_muscle_forces
from utils.torch.telemetry import pack_diag, merge_diag
from muscles.torch.muscle_tools import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
)


def _t(x, like: Tensor) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(x, device=like.device, dtype=like.dtype)


def _first(x: Tensor) -> Tensor:
    return x[0] if x.dim() >= 3 else x


def get_default_synergies(device, dtype) -> Tensor:
    """Heuristic default synergy patterns for a 6-muscle Arm26-like model (6,3)."""
    W = torch.tensor([
        [0.91, 0.85, 0.88, 0.05, 0.08, 0.35],   # S1 flexor-biased
        [0.08, 0.12, 0.05, 0.85, 0.79, 0.81],   # S2 extensor/abductor-biased
        [0.45, 0.40, 0.42, 0.38, 0.35, 0.50],   # S3 co-contraction
    ], device=device, dtype=dtype).T  # (6,3)
    W = torch.clamp(W, min=0.0)
    W = W / (torch.max(W, dim=0, keepdim=True).values + 1e-12)
    return W


def _nnls_small(A: Tensor, b: Tensor, reg: float = 1e-10) -> Tensor:
    """min ||A x - b||^2 + reg||x||^2  s.t. x>=0, brute-force active-set (2^K). K<=6."""
    K = A.shape[1]
    AtA = A.T @ A
    Atb = A.T @ b
    best_x = torch.zeros(K, device=A.device, dtype=A.dtype)
    best_cost = float("inf")
    for mask in range(1 << K):
        active = [i for i in range(K) if (mask >> i) & 1]
        if len(active) == 0:
            cost = float(b @ b)
            if cost < best_cost:
                best_cost = cost
                best_x = torch.zeros(K, device=A.device, dtype=A.dtype)
            continue
        idx = torch.tensor(active, device=A.device, dtype=torch.long)
        Msub = AtA[idx][:, idx] + reg * torch.eye(len(active), device=A.device, dtype=A.dtype)
        y = Atb[idx]
        try:
            x_act = torch.linalg.solve(Msub, y)
        except Exception:
            continue
        if torch.any(x_act < -1e-12):
            continue
        x = torch.zeros(K, device=A.device, dtype=A.dtype)
        x[idx] = torch.clamp(x_act, min=0.0)
        r = A @ x - b
        cost = float(r @ r + reg * (x @ x))
        if cost < best_cost:
            best_cost = cost
            best_x = x
    return best_x


@dataclass
class SynergyParams:
    W_preset: Optional[Any] = None
    n_synergies: Optional[int] = None
    c_max: float = 3.0
    B_preset: Optional[Any] = None
    use_prior: bool = False
    prior_weight: float = 1e-2

    Kp_task: float = 800.0
    Kv_task: float = 60.0
    Kff_x: float = 1.0

    enable_inertia_comp: bool = True
    enable_gravity_comp: bool = True
    enable_velocity_comp: bool = True

    eps: float = 1e-6
    lam_os_max: float = 1e6
    sigma_thresh: float = 1e-4
    gate_pow: float = 2.0

    bisect_iters: int = 12
    synergy_strength: float = 0.7

    online_B_adapt: bool = False
    rls_lambda: float = 0.995
    rls_delta: float = 100.0


class SynergyController:
    def __init__(self, env, arm, params: SynergyParams = None):
        self.env = env
        self.arm = arm
        self.p = params or SynergyParams()
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

        if self.p.W_preset is not None:
            W = _t(self.p.W_preset, torch.zeros(1, device=self.device, dtype=self.dtype))
        else:
            W = get_default_synergies(self.device, self.dtype)
        W = torch.clamp(W, min=0.0)
        W = W / (torch.max(W, dim=0, keepdim=True).values + 1e-12)
        if self.p.n_synergies is not None:
            W = W[:, : int(self.p.n_synergies)]
        self.W = W
        self.K = W.shape[1]
        self.n_muscles = W.shape[0]

        if self.p.B_preset is not None:
            B = _t(self.p.B_preset, W)
        else:
            B = torch.zeros((self.K, 5), device=self.device, dtype=self.dtype)
            B[:, 0] = 0.4; B[:, 1] = 0.4; B[:, 2] = 0.1; B[:, 3] = 0.1; B[:, 4] = 0.05
        self.B = B[: self.K, :]

    def reset(self, q0: Tensor):
        q0 = q0.to(device=self.device, dtype=self.dtype)
        self.qref = (q0[0] if q0.ndim == 2 else q0).clone()

    def set_synergy_matrix(self, W):
        W = torch.clamp(_t(W, self.W), min=0.0)
        W = W / (torch.max(W, dim=0, keepdim=True).values + 1e-12)
        self.W = W
        self.K = W.shape[1]
        self.n_muscles = W.shape[0]
        if self.B.shape[0] != self.K:
            self.B = self.B[: self.K, :]

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

        # kinematics
        J_xy = _first(self.env.skeleton.geometric_jacobian(q))[0:2, :]
        Jdot_xy = _first(self.env.skeleton.geometric_jacobian_dot(q, qd))[0:2, :]
        _, sminJ, lamJ = adaptive_dls_pinv(J_xy, 2, self.kp)
        sminJ = torch.as_tensor(sminJ, device=q.device, dtype=q.dtype).reshape(())
        xd_d_s, xdd_d_s, alpha_J = scale_task_by_J(xd_d.clone(), xdd_d.clone(), sminJ=sminJ, P=self.kp)

        # dynamics + op-space guard
        M = _first(self.env.skeleton.mass_matrix(q))
        Minv = torch.linalg.inv(M)
        S = J_xy @ Minv @ J_xy.T
        Lambda_reg, lam_os, eta, eta2, xd_d_g, xdd_d_g, dyn_diag = op_space_guard_and_gate(
            S, xd_d_s.clone(), xdd_d_s.clone(), self.dp
        )
        xd_d_g = xd_d_g.reshape(-1)[:2]
        xdd_d_g = xdd_d_g.reshape(-1)[:2]
        eta = torch.as_tensor(eta, device=q.device, dtype=q.dtype).reshape(())
        eta2 = torch.as_tensor(eta2, device=q.device, dtype=q.dtype).reshape(())

        e_x = x_d - x
        e_v = xd_d_g - xd

        # nonlinear terms h
        h = torch.zeros_like(q)
        if self.p.enable_velocity_comp:
            C = _first(self.env.skeleton.coriolis_matrix(q, qd))
            h = h + C @ qd
        if self.p.enable_gravity_comp:
            g_any = self.env.skeleton.gravity_vector(q)
            g = (g_any[0] if g_any.dim() >= 3 else g_any).reshape(-1)[:n]
            h = h + g

        # OSC/PD torque target
        Kp, Kv = float(self.p.Kp_task), float(self.p.Kv_task)
        xdd_cmd = (self.p.Kff_x * xdd_d_g) + (Kp * e_x) + (Kv * e_v)
        if self.p.enable_inertia_comp:
            mu = Lambda_reg @ (J_xy @ (Minv @ h)) - Lambda_reg @ (Jdot_xy @ qd)
            F_task = (Lambda_reg @ xdd_cmd) + mu
            tau_cmd = (J_xy.T @ F_task).reshape(-1)
        else:
            F_task = (Kp * e_x) + (Kv * e_v)
            tau_cmd = (J_xy.T @ F_task).reshape(-1)
            if self.p.enable_gravity_comp:
                tau_cmd = tau_cmd + h.reshape(-1)
        tau_cmd = tau_cmd * torch.clamp(eta, 0.0, 1.0)

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
        a_min = float(getattr(self.env.muscle, "min_activation", 0.0))
        eta_b = eta.reshape(1)

        # STEP 1: optimal full-space forces / activation
        F_opt, mus_diag = solve_muscle_forces(tau_cmd.unsqueeze(0), R.unsqueeze(0), Fmax_vec, eta_b, self.mp)
        a_opt = force_to_activation_bisect(F_opt, lenvel, self.env.muscle, flpe, Fmax_vec, iters=int(self.p.bisect_iters))
        a_opt = torch.clamp(a_opt, a_min, 1.0)

        # STEP 2: project into synergy space  a_syn = W c
        c0 = torch.clamp(self.B @ torch.stack([e_x[0], e_x[1], e_v[0], e_v[1],
                                               torch.ones((), device=q.device, dtype=q.dtype)]),
                         0.0, self.p.c_max)
        if self.p.use_prior and self.p.prior_weight > 0.0:
            w = float(self.p.prior_weight)
            A_aug = torch.cat([self.W, (w ** 0.5) * torch.eye(self.K, device=q.device, dtype=q.dtype)], dim=0)
            b_aug = torch.cat([a_opt[0], (w ** 0.5) * c0])
            c = _nnls_small(A_aug, b_aug)
        else:
            c = _nnls_small(self.W, a_opt[0])
        c = torch.clamp(c, 0.0, self.p.c_max)
        a_syn = torch.clamp((self.W @ c).unsqueeze(0), a_min, 1.0)

        # STEP 3: residual correction in torque space
        af_syn = active_force_from_activation(a_syn, lenvel, self.env.muscle)
        F_syn = torch.clamp(Fmax_vec * (af_syn + flpe), min=0.0)
        tau_syn = (-R @ F_syn[0]).reshape(-1)
        tau_res = (tau_cmd - tau_syn).reshape(-1)
        F_res, mus_diag2 = solve_muscle_forces(tau_res.unsqueeze(0), R.unsqueeze(0), Fmax_vec, eta_b, self.mp)
        beta = float(min(max(self.p.synergy_strength, 0.0), 1.0))
        F_mix = beta * F_syn + (1.0 - beta) * F_opt
        F_final = torch.clamp(F_mix + F_res, min=0.0)

        a_final = force_to_activation_bisect(F_final, lenvel, self.env.muscle, flpe, Fmax_vec, iters=int(self.p.bisect_iters))
        a_final = torch.clamp(a_final, a_min, 1.0)

        af_now = active_force_from_activation(a_final, lenvel, self.env.muscle)
        F_pred = torch.clamp(Fmax_vec * (af_now + flpe), min=0.0)
        F_corr = saturation_repair_tau(-R.unsqueeze(0), F_pred, a_final, a_min, 1.0,
                                       Fmax_vec, tau_des=tau_cmd.unsqueeze(0))
        if torch.any(torch.abs(F_corr - F_pred) > 1e-9):
            a_final = force_to_activation_bisect(F_corr, lenvel, self.env.muscle, flpe, Fmax_vec,
                                                 iters=max(4, int(self.p.bisect_iters) - 4))
            a_final = torch.clamp(a_final, a_min, 1.0)
            af_now = active_force_from_activation(a_final, lenvel, self.env.muscle)
            F_pred = torch.clamp(Fmax_vec * (af_now + flpe), min=0.0)

        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ, alpha_J=alpha_J)
        diag = merge_diag(
            kin_diag, dyn_diag, mus_diag, mus_diag2,
            pack_diag(lam_os=lam_os, eta=eta, eta2=eta2,
                      c_max=float(torch.max(c)) if c.numel() else 0.0,
                      beta=beta, synergy_K=int(self.K)),
        )
        return {
            "tau_des": tau_cmd.unsqueeze(0), "R": R.unsqueeze(0), "Fmax": Fmax_vec,
            "F_des": F_pred, "act": a_final, "synergy_coeff": c.unsqueeze(0),
            "q": q.unsqueeze(0), "qd": qd.unsqueeze(0), "x": x.unsqueeze(0), "xd": xd.unsqueeze(0),
            "xref_tuple": (x_d, xd_d_g, xdd_d_g), "eta": eta2.reshape(1), "diag": diag,
        }
