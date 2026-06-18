"""
MotorNet Fixed (Corrected): Differentiable / Learning-Based Muscle Control

This version fixes the common "doesn't learn / doesn't work" mismatch:

- Your classical controllers (PDIF/OSC/SMC) typically output **muscle activations** a(t) ∈ [0,1]^6
  that are applied directly to the environment.
- MotorNet-style policies typically output **muscle excitations** u(t) ∈ [0,1]^6, and then
  activation dynamics produce a(t).

If you train a network to imitate a(t) but deploy it as u(t) (or vice-versa), control fails.

✅ Fix implemented here:
- The policy outputs **excitations** u(t)
- The controller integrates activation dynamics internally to produce **activations** a(t)
- The environment receives the activation vector a(t)

Additionally:
- Adds a consistent expert->policy conversion helper: activation_target_to_excitation(...)
- Provides safe save/load utilities
- Keeps a robust NumPy fallback (inverse-dynamics-like) when no trained policy is loaded

References (conceptual):
- Codol et al., 2024 (MotorNet)
- Standard muscle activation dynamics (first-order rise/fall)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# Try to import torch (optional)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from model_lib.numpy.skeleton import (
    geometricJacobian_cached,
    inertiaMatrixCOM_cached,
    centrifugalCoriolisCOM_cached,
    gravityCOM_cached,
)
from utils.numpy.kinematics_guard import KinGuardParams, adaptive_dls_pinv
from utils.numpy.dynamics_guard import DynGuardParams, op_space_guard_and_gate
from utils.numpy.muscle_guard import MuscleGuardParams, solve_muscle_forces
from muscles.numpy.muscle_tools import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
)


# =============================================================================
# Policy network (Torch)
# =============================================================================

if TORCH_AVAILABLE:
    class GRUPolicyNetwork(nn.Module):
        """
        GRU policy:
          input  = [x, y, xd, yd, x_d, y_d, e_x, e_y, t_norm]  (9)
          output = excitations u ∈ [0,1]^6
        """
        def __init__(
            self,
            input_dim: int = 9,
            hidden_dim: int = 128,
            output_dim: int = 6,
            n_layers: int = 2,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.input_dim = int(input_dim)
            self.hidden_dim = int(hidden_dim)
            self.output_dim = int(output_dim)
            self.n_layers = int(n_layers)

            self.input_norm = nn.LayerNorm(self.input_dim)

            self.gru = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.n_layers,
                batch_first=True,
                dropout=dropout if self.n_layers > 1 else 0.0,
            )

            self.head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, self.output_dim),
                nn.Sigmoid(),
            )

            self._init_weights()

        def _init_weights(self):
            for name, param in self.gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

            for layer in self.head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        def init_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None):
            device = device or torch.device("cpu")
            return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)

        def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
            # Accept (B,9) or (B,T,9)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            B = x.size(0)
            if h is None:
                h = self.init_hidden(B, x.device)

            x = self.input_norm(x)
            y, h_new = self.gru(x, h)
            u = self.head(y)  # (B,T,6)
            return u, h_new


# =============================================================================
# Params
# =============================================================================

@dataclass
class MotorNetFixedParams:
    # Network
    hidden_dim: int = 128
    n_layers: int = 2

    # Activation dynamics (seconds)
    tau_rise: float = 0.015
    tau_fall: float = 0.060

    # Fallback gains (for inverse-dynamics-like numpy fallback)
    Kp: float = 2000.0
    Kv: float = 100.0
    damping_lambda: float = 0.01  # damping for Lambda inversion in fallback

    # Guards
    eps: float = 1e-6
    lam_os_max: float = 200.0
    sigma_thresh: float = 1e-4
    gate_pow: float = 2.0

    # Muscle inversion
    bisect_iters: int = 12

    # Device
    device: str = "cpu"


# =============================================================================
# Helpers
# =============================================================================

def activation_target_to_excitation(
    a_target: np.ndarray,
    a_prev: np.ndarray,
    dt: float,
    tau_rise: float,
    tau_fall: float,
) -> np.ndarray:
    """
    Convert an *activation* target a_target into an equivalent *excitation* u such that,
    after one Euler step of the activation dynamics, the activation reaches a_target.

    Activation dynamics:
      a_{k+1} = a_k + (u_k - a_k)/tau * dt

    Solve for u_k given a_{k+1} = a_target:
      u_k = a_k + (a_target - a_k) * (tau/dt)

    tau uses rise vs fall depending on whether a_target > a_prev.
    """
    a_target = np.asarray(a_target, dtype=float).reshape(-1)
    a_prev = np.asarray(a_prev, dtype=float).reshape(-1)
    dt = float(dt)

    dt_safe = max(dt, 1e-12)
    tau = np.where(a_target > a_prev, float(tau_rise), float(tau_fall))
    u = a_prev + (a_target - a_prev) * (tau / dt_safe)
    return np.clip(u, 0.0, 1.0)


# =============================================================================
# Controller
# =============================================================================

class MotorNetFixed:
    """
    A controller that can run in 2 modes:

    1) Torch (learned): policy outputs excitations u; we integrate activation dynamics to get a.
    2) NumPy fallback: compute a directly via inverse-dynamics-like solve, so it "works out of the box".

    The simulator should send env.step(result["act"][None,:]) as usual.
    """

    def __init__(self, env, arm, params: Optional[MotorNetFixedParams] = None):
        self.env = env
        self.arm = arm
        self.p = params or MotorNetFixedParams()

        self.n_muscles = 6
        self.n_dof = 2

        # Guards
        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=self.p.eps,
            lam_os_max=self.p.lam_os_max,
            gate_pow=self.p.gate_pow,
            sigma_thresh_S=max(self.p.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()

        # State
        self.qref: Optional[np.ndarray] = None
        self.activations = np.full(self.n_muscles, 0.05, dtype=float)  # internal activation state
        self.u_prev = np.zeros(self.n_muscles, dtype=float)
        self.t_current = 0.0
        self.T_trial = 0.6
        self.goal: Optional[np.ndarray] = None

        # Torch bits
        self.use_torch = bool(TORCH_AVAILABLE)
        self.trained = False
        self.policy = None
        self.optimizer = None
        self.hidden = None

        if self.use_torch:
            self.device = torch.device(self.p.device)
            self.policy = GRUPolicyNetwork(
                input_dim=9,
                hidden_dim=self.p.hidden_dim,
                output_dim=self.n_muscles,
                n_layers=self.p.n_layers,
            ).to(self.device)
            self.optimizer = optim.AdamW(self.policy.parameters(), lr=3e-4, weight_decay=1e-5)
            self.hidden = self.policy.init_hidden(1, self.device)

        # For numerical Jdot estimate in fallback
        self._J_prev = None

    def reset(self, q0: np.ndarray):
        self.qref = np.asarray(q0, dtype=float).copy()
        self.activations = np.full(self.n_muscles, 0.05, dtype=float)
        self.u_prev = np.zeros(self.n_muscles, dtype=float)
        self.goal = None
        self.t_current = 0.0
        self.T_trial = 0.6
        self._J_prev = None

        if self.use_torch and self.policy is not None:
            self.hidden = self.policy.init_hidden(1, self.device)

    def set_goal(self, goal: np.ndarray, T_trial: float = 0.6):
        self.goal = np.asarray(goal, dtype=float).copy()
        self.T_trial = float(T_trial)
        self.t_current = 0.0

    # ----------------------- activation dynamics -----------------------

    def _activation_dynamics(self, u: np.ndarray, a: np.ndarray, dt: float) -> np.ndarray:
        u = np.asarray(u, dtype=float).reshape(-1)
        a = np.asarray(a, dtype=float).reshape(-1)
        dt = float(dt)
        tau = np.where(u > a, self.p.tau_rise, self.p.tau_fall)
        da = (u - a) / (tau + 1e-12)
        a_new = a + da * dt
        # keep min_activation consistent with your muscles (commonly 0.02)
        return np.clip(a_new, 0.02, 1.0)

    # ----------------------- torch policy -----------------------

    def _build_obs(self, x: np.ndarray, xd: np.ndarray, x_d: np.ndarray, t_norm: float) -> np.ndarray:
        e_x = np.asarray(x_d, dtype=float).reshape(2) - np.asarray(x, dtype=float).reshape(2)
        obs = np.concatenate([
            np.asarray(x, dtype=float).reshape(2),
            np.asarray(xd, dtype=float).reshape(2),
            np.asarray(x_d, dtype=float).reshape(2),
            e_x.reshape(2),
            np.array([float(t_norm)], dtype=float),
        ])
        return obs

    def _compute_torch_excitation(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,9)
        with torch.no_grad():
            u_t, self.hidden = self.policy(obs_t, self.hidden)  # (1,1,6)
        return u_t.squeeze(0).squeeze(0).cpu().numpy()

    # ----------------------- numpy fallback -----------------------

    def _compute_inverse_dynamics_activation(
        self,
        q: np.ndarray, qd: np.ndarray,
        x: np.ndarray, xd: np.ndarray,
        x_d: np.ndarray, xd_d: np.ndarray, xdd_d: np.ndarray,
    ) -> np.ndarray:
        """
        Inverse-dynamics-like activation solve:
          - compute task force via operational-space dynamics (damped Lambda)
          - map to tau via J^T
          - solve muscle forces and invert force->activation

        This is intended as a robust "works without training" fallback.
        """
        J = geometricJacobian_cached(self.env.skeleton._robot, symbolic=False)
        J_xy = J[0:2, :]

        M = inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False)
        C_any = centrifugalCoriolisCOM_cached(self.env.skeleton._robot, symbolic=False)
        g = gravityCOM_cached(
            self.env.skeleton._robot, self.env.skeleton._gravity_vec, symbolic=False
        ).reshape(-1)

        Minv = np.linalg.inv(M)
        S = J_xy @ Minv @ J_xy.T

        # op-space guard (gives Lambda_reg + gating)
        Lambda_reg, _, eta, _, xd_d_g, xdd_d_g, _ = op_space_guard_and_gate(
            S, np.asarray(xd_d, float).copy(), np.asarray(xdd_d, float).copy(), self.dp
        )

        # numerical Jdot
        if self._J_prev is None:
            Jdot = np.zeros_like(J_xy)
        else:
            Jdot = (J_xy - self._J_prev) / float(self.arm.dt)
        self._J_prev = J_xy.copy()

        # task error
        e_x = np.asarray(x_d, float).reshape(2) - np.asarray(x, float).reshape(2)
        e_v = np.asarray(xd_d_g, float).reshape(2) - np.asarray(xd, float).reshape(2)

        xdd_cmd = self.p.Kp * e_x + self.p.Kv * e_v + np.asarray(xdd_d_g, float).reshape(2)

        # bias term
        C_any = np.asarray(C_any)
        if C_any.ndim == 2:
            h = C_any @ qd + g
        else:
            h = C_any.reshape(-1) + g

        mu = Lambda_reg @ (J_xy @ (Minv @ h) - Jdot @ qd)
        F_task = Lambda_reg @ xdd_cmd + mu
        tau_des = J_xy.T @ F_task

        # muscle allocation
        geom = self.env.states["geometry"]
        R = geom[:, 2:2 + self.n_dof, :][0]
        Fmax_vec = get_Fmax_vec(self.env, R.shape[1])

        # conservative torque cap
        tau_cap = 0.95 * (np.abs(R) @ Fmax_vec)
        tau_des = np.clip(tau_des, -tau_cap, tau_cap)

        F_des, _ = solve_muscle_forces(tau_des, R, Fmax_vec, eta, self.mp)

        # invert to activation
        lenvel = geom[:, :2, :]
        idx_flpe = self.env.muscle.state_name.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]

        a = force_to_activation_bisect(
            F_des, lenvel, self.env.muscle, flpe, Fmax_vec,
            iters=self.p.bisect_iters
        )

        # repair
        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = Fmax_vec * (af_now + flpe)
        F_corr = saturation_repair_tau(
            -R,
            F_pred,
            a,
            self.env.muscle.min_activation,
            1.0,
            Fmax_vec,
            tau_des=tau_des,
        )
        if np.any(np.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(
                F_corr, lenvel, self.env.muscle, flpe, Fmax_vec, iters=max(6, self.p.bisect_iters - 4)
            )
        return np.clip(a, 0.02, 1.0)

    # ----------------------- main API -----------------------

    def compute(self, x_d: np.ndarray, xd_d: np.ndarray, xdd_d: np.ndarray) -> Dict[str, Any]:
        """
        Returns a dict compatible with your simulator:
          - "act": muscle activations (what env.step should receive)
          - "excitations": excitations u (for logging)
          - "tau_des": torque produced by current activation estimate
        """
        dt = float(self.arm.dt)

        joint = self.env.states["joint"][0]
        q, qd = joint[:2], joint[2:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:4]
        self.env.skeleton._set_state(q, qd)

        if self.qref is None:
            self.qref = q.copy()

        # goal for diagnostics / time norm
        if self.goal is None:
            self.goal = np.asarray(x_d, dtype=float).copy()

        t_norm = min(self.t_current / max(self.T_trial, 1e-12), 1.0)
        self.t_current += dt

        if self.use_torch and self.trained and self.policy is not None:
            # Policy outputs u; integrate activation dynamics to obtain a.
            obs = self._build_obs(x, xd, x_d, t_norm)
            u = self._compute_torch_excitation(obs)
            u = np.clip(u, 0.0, 1.0)
            self.u_prev = u.copy()
            self.activations = self._activation_dynamics(u, self.activations, dt)
        else:
            # Fallback: compute activation directly
            self.activations = self._compute_inverse_dynamics_activation(q, qd, x, xd, x_d, xd_d, xdd_d)
            self.u_prev = self.activations.copy()

        # diagnostics torque
        geom = self.env.states["geometry"]
        R = geom[:, 2:2 + self.n_dof, :][0]
        Fmax_vec = get_Fmax_vec(self.env, R.shape[1])
        lenvel = geom[:, :2, :]
        idx_flpe = self.env.muscle.state_name.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]

        af = active_force_from_activation(self.activations, lenvel, self.env.muscle)
        F_muscle = Fmax_vec * (af + flpe)
        F_muscle = np.maximum(F_muscle, 0.0)
        tau_actual = -R @ F_muscle

        return {
            "act": self.activations.copy(),
            "excitations": self.u_prev.copy(),
            "tau_des": tau_actual,
            "R": R,
            "Fmax": Fmax_vec,
            "F_des": F_muscle,
            "q": q,
            "qd": qd,
            "x": x,
            "xd": xd,
            "t_norm": float(t_norm),
            "eta": 1.0,
            "xref_tuple": (x_d, xd_d, xdd_d),
        }

    # ----------------------- persistence -----------------------

    def save(self, path: str):
        if not (self.use_torch and self.policy is not None):
            raise RuntimeError("Torch not available; cannot save policy weights.")
        ckpt = {
            "state_dict": self.policy.state_dict(),
            "trained": bool(self.trained),
            "input_dim": int(self.policy.input_dim),
            "hidden_dim": int(self.policy.hidden_dim),
            "output_dim": int(self.policy.output_dim),
            "n_layers": int(self.policy.n_layers),
            "params": self.p.__dict__,
        }
        torch.save(ckpt, path)

    def load(self, path: str):
        if not (self.use_torch and self.policy is not None):
            raise RuntimeError("Torch not available; cannot load policy weights.")
        ckpt = torch.load(path, map_location=self.device)

        # Rebuild if needed
        hidden_dim = int(ckpt.get("hidden_dim", self.p.hidden_dim))
        n_layers = int(ckpt.get("n_layers", self.p.n_layers))
        if (hidden_dim != self.p.hidden_dim) or (n_layers != self.p.n_layers):
            self.p.hidden_dim = hidden_dim
            self.p.n_layers = n_layers
            self.policy = GRUPolicyNetwork(
                input_dim=int(ckpt.get("input_dim", 9)),
                hidden_dim=hidden_dim,
                output_dim=int(ckpt.get("output_dim", self.n_muscles)),
                n_layers=n_layers,
            ).to(self.device)

        self.policy.load_state_dict(ckpt["state_dict"])
        self.trained = bool(ckpt.get("trained", True))
        self.policy.eval()
