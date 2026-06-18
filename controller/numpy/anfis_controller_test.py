"""
ANFIS (Adaptive Neuro-Fuzzy Inference System) Controller - STABILIZED
=====================================================================

Stability upgrades vs naive ANFIS:
- Input scaling (e, ed) to match MF ranges (prevents saturation).
- Teacher scaling/clipping (prevents huge LSE targets).
- Ridge LSE with conditioning + theta clipping.
- Exponential smoothing of consequent updates (prevents violent jumps).
- Update throttling and min batch.
- Optional premise SGD with gradient clipping.

Only dependencies: numpy + existing project modules.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from model_lib.numpy.skeleton import (
    geometricJacobian_cached,
    inertiaMatrixCOM_cached,
    centrifugalCoriolisCOM_cached,
    gravityCOM_cached,
)
from utils.numpy.kinematics_guard import KinGuardParams, adaptive_dls_pinv
from utils.numpy.dynamics_guard import DynGuardParams, op_space_guard_and_gate
from utils.numpy.muscle_guard import MuscleGuardParams, solve_muscle_forces
from utils.numpy.telemetry import pack_diag, merge_diag
from muscles.numpy.muscle_tools import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
)


# ----------------------------- Params ---------------------------------


@dataclass
class ANFISParams:
    # ANFIS structure
    n_mf: int = 5
    mf_type: str = "gaussian"

    # Consequent learning (LSE)
    online_adapt: bool = False
    lse_reg: float = 1e-2               # ✅ more damping (was 1e-4)
    adapt_every: int = 20               # ✅ slower updates (was 5)
    min_fit_samples: int = 80           # ✅ more data before fitting
    buffer_size: int = 600              # ✅ larger replay buffer
    teacher_mode: str = "id_residual"   # 'id_residual' or 'id_full'

    # Premise learning (MF centres/widths) via replay + SGD
    lr_premise: float = 0.0             # ✅ default OFF (turn on carefully)
    premise_batch_size: int = 64
    premise_update_every: int = 50

    # Freeze / anti-forgetting
    freeze_after_steps: int = 0
    freeze_err_thresh: float = 0.0
    freeze_patience: int = 80
    err_ema_beta: float = 0.98

    # Save/load
    rules_path: str = "ANFIS/saved_model/anfis_rules.npz"
    autosave_every: int = 400
    save_on_freeze: bool = True

    # Dynamics compensation
    enable_gravity_comp: bool = True
    enable_coriolis_comp: bool = True

    # Guards / numerics
    eps: float = 1e-6
    gate_pow: float = 2.0
    sigma_thresh: float = 1e-4
    lam_os_max: float = 1e6

    # Muscle inversion
    bisect_iters: int = 16

    # ------------------ ✅ NEW STABILITY KNOBS ------------------
    # Scale ANFIS inputs so they live in MF range
    input_scale_e: float = 0.25         # e_scaled = input_scale_e * e_joint
    input_scale_ed: float = 0.10        # ed_scaled = input_scale_ed * ed_joint

    # Scale & clip teacher (prevents huge y)
    teacher_scale: float = 0.01         # y_scaled = teacher_scale * y
    teacher_clip: float = 5.0           # clip y_scaled to ±teacher_clip

    # Clip + smooth consequent parameters after LSE
    theta_clip: float = 2000.0          # clip each consequent param
    theta_smooth: float = 0.15          # 0=no smoothing, 1=instant replace

    # Numerical safety for LSE
    max_cond: float = 1e10              # if cond(A) too high -> skip update

    # Premise SGD stability
    premise_grad_clip: float = 5.0      # clip MF parameter gradients
    mf_width_min: float = 1e-3
    mf_width_max: float = 10.0


# ------------------------ Membership Functions -------------------------


class MembershipFunction:
    def __init__(self, mf_type: str, center: float, width: float, slope: float = 2.0):
        self.mf_type = mf_type
        self.c = float(center)
        self.a = float(max(width, 1e-3))
        self.b = float(slope)

    def __call__(self, x: float) -> float:
        x = float(x)
        if self.mf_type == "gaussian":
            return float(np.exp(-((x - self.c) ** 2) / (2.0 * self.a ** 2 + 1e-12)))
        elif self.mf_type == "bell":
            diff = (x - self.c) / (self.a + 1e-12)
            return float(1.0 / (1.0 + np.abs(diff) ** (2.0 * self.b)))
        elif self.mf_type == "triangular":
            return float(max(0.0, 1.0 - abs(x - self.c) / (self.a + 1e-12)))
        else:
            raise ValueError(f"Unknown MF type: {self.mf_type}")

    def grad_c(self, x: float, mu: float) -> float:
        if self.mf_type != "gaussian":
            return 0.0
        return mu * (x - self.c) / (self.a ** 2 + 1e-12)

    def grad_a(self, x: float, mu: float) -> float:
        if self.mf_type != "gaussian":
            return 0.0
        return mu * ((x - self.c) ** 2) / (self.a ** 3 + 1e-12)

    def pack(self) -> Tuple[float, float, float, str]:
        return (self.c, self.a, self.b, self.mf_type)

    def unpack(self, t: Tuple[float, float, float, str]):
        self.c = float(t[0])
        self.a = float(max(t[1], 1e-3))
        self.b = float(t[2])
        self.mf_type = str(t[3])


# ------------------------------ ANFIS Layer ----------------------------


class ANFISLayer:
    def __init__(self, n_mf: int, mf_type: str, input_ranges: List[Tuple[float, float]]):
        self.n_mf = int(n_mf)
        self.n_inputs = len(input_ranges)
        if self.n_inputs != 2:
            raise ValueError("ANFISLayer expects exactly 2 inputs (e, edot).")
        self.n_rules = self.n_mf ** self.n_inputs

        self.mfs: List[List[MembershipFunction]] = []
        for (lo, hi) in input_ranges:
            lo, hi = float(lo), float(hi)
            centers = np.linspace(lo, hi, self.n_mf)
            width = (hi - lo) / max(self.n_mf - 1, 1) * 0.5
            self.mfs.append([MembershipFunction(mf_type, c, width) for c in centers])

        self.consequent = np.zeros((self.n_rules, self.n_inputs + 1), dtype=float)
        self.consequent[:, -1] = np.random.randn(self.n_rules) * 0.01

    def forward(self, inputs: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        e = float(inputs[0])
        ed = float(inputs[1])

        mu0 = np.array([mf(e) for mf in self.mfs[0]], dtype=float)
        mu1 = np.array([mf(ed) for mf in self.mfs[1]], dtype=float)

        w = np.zeros(self.n_rules, dtype=float)
        idx = 0
        for i in range(self.n_mf):
            for j in range(self.n_mf):
                w[idx] = mu0[i] * mu1[j]
                idx += 1

        w_sum = float(np.sum(w) + 1e-12)
        wbar = w / w_sum

        x_ext = np.array([e, ed, 1.0], dtype=float)
        f = self.consequent @ x_ext
        out = float(np.dot(wbar, f))
        return out, wbar, f

    def compute_phi(self, inputs: np.ndarray, wbar: np.ndarray) -> np.ndarray:
        e = float(inputs[0])
        ed = float(inputs[1])
        x_ext = np.array([e, ed, 1.0], dtype=float)
        return np.outer(wbar, x_ext).reshape(-1)

    def init_pd(self, Kp: float, Kd: float, bias: float = 0.0):
        self.consequent[:, 0] = float(Kp)
        self.consequent[:, 1] = float(Kd)
        self.consequent[:, 2] = float(bias)

    # ✅ stabilized LSE: cond-check + clip + smooth
    def update_consequent_lse_stable(
        self,
        Phi: np.ndarray,
        y: np.ndarray,
        reg: float,
        theta_clip: float,
        theta_smooth: float,
        max_cond: float,
    ):
        Phi = np.asarray(Phi, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n_params = self.n_rules * (self.n_inputs + 1)
        if Phi.ndim != 2 or Phi.shape[1] != n_params:
            return
        if Phi.shape[0] != y.shape[0]:
            return

        try:
            A = Phi.T @ Phi + float(reg) * np.eye(n_params)
            cA = np.linalg.cond(A)
            if not np.isfinite(cA) or (cA > float(max_cond)):
                return

            b = Phi.T @ y
            theta_new = np.linalg.solve(A, b)
            theta_new = np.clip(theta_new, -float(theta_clip), +float(theta_clip))

            C_new = theta_new.reshape(self.n_rules, self.n_inputs + 1)

            a = float(theta_smooth)
            a = float(np.clip(a, 0.0, 1.0))
            self.consequent = (1.0 - a) * self.consequent + a * C_new
        except np.linalg.LinAlgError:
            pass

    # Premise SGD (kept, but caller should keep lr small and clipped)
    def premise_sgd_step(
        self,
        inputs_batch: np.ndarray,
        targets_batch: np.ndarray,
        lr: float,
        grad_clip: float,
        width_min: float,
        width_max: float,
    ):
        if lr <= 0.0:
            return

        inputs_batch = np.asarray(inputs_batch, dtype=float)
        targets_batch = np.asarray(targets_batch, dtype=float).reshape(-1)
        B = inputs_batch.shape[0]
        gc = float(max(grad_clip, 0.0))

        for b in range(B):
            e = float(inputs_batch[b, 0])
            ed = float(inputs_batch[b, 1])
            y_t = float(targets_batch[b])

            mu0 = np.array([mf(e) for mf in self.mfs[0]], dtype=float)
            mu1 = np.array([mf(ed) for mf in self.mfs[1]], dtype=float)

            w = np.zeros(self.n_rules, dtype=float)
            idx = 0
            for i in range(self.n_mf):
                for j in range(self.n_mf):
                    w[idx] = mu0[i] * mu1[j]
                    idx += 1
            w_sum = float(np.sum(w) + 1e-12)
            wbar = w / w_sum

            x_ext = np.array([e, ed, 1.0], dtype=float)
            f = self.consequent @ x_ext
            y_hat = float(np.dot(wbar, f))
            delta = y_hat - y_t

            # update mfs[0]
            for i in range(self.n_mf):
                dw_dp = np.zeros_like(w)
                idx = 0
                for ii in range(self.n_mf):
                    for jj in range(self.n_mf):
                        if ii == i:
                            dw_dp[idx] = mu1[jj]
                        idx += 1
                dw_sum_dp = float(np.sum(dw_dp))

                dy_dmu = 0.0
                for k in range(self.n_rules):
                    d_wbar_k_dp = (w_sum * dw_dp[k] - w[k] * dw_sum_dp) / (w_sum ** 2)
                    dy_dmu += d_wbar_k_dp * f[k]

                mf = self.mfs[0][i]
                dmu_dc = mf.grad_c(e, mu0[i])
                dmu_da = mf.grad_a(e, mu0[i])

                grad_c = delta * dy_dmu * dmu_dc
                grad_a = delta * dy_dmu * dmu_da

                if gc > 0.0:
                    grad_c = float(np.clip(grad_c, -gc, +gc))
                    grad_a = float(np.clip(grad_a, -gc, +gc))

                mf.c -= lr * grad_c
                mf.a -= lr * grad_a
                mf.a = float(np.clip(mf.a, width_min, width_max))

            # update mfs[1]
            for j in range(self.n_mf):
                dw_dp = np.zeros_like(w)
                idx = 0
                for ii in range(self.n_mf):
                    for jj in range(self.n_mf):
                        if jj == j:
                            dw_dp[idx] = mu0[ii]
                        idx += 1
                dw_sum_dp = float(np.sum(dw_dp))

                dy_dmu = 0.0
                for k in range(self.n_rules):
                    d_wbar_k_dp = (w_sum * dw_dp[k] - w[k] * dw_sum_dp) / (w_sum ** 2)
                    dy_dmu += d_wbar_k_dp * f[k]

                mf = self.mfs[1][j]
                dmu_dc = mf.grad_c(ed, mu1[j])
                dmu_da = mf.grad_a(ed, mu1[j])

                grad_c = delta * dy_dmu * dmu_dc
                grad_a = delta * dy_dmu * dmu_da

                if gc > 0.0:
                    grad_c = float(np.clip(grad_c, -gc, +gc))
                    grad_a = float(np.clip(grad_a, -gc, +gc))

                mf.c -= lr * grad_c
                mf.a -= lr * grad_a
                mf.a = float(np.clip(mf.a, width_min, width_max))


# ------------------------------ Controller -----------------------------


class ANFISController:
    def __init__(self, env, arm, params: ANFISParams):
        self.env = env
        self.arm = arm
        self.p = params

        # ✅ Since we scale inputs, keep MF ranges modest & consistent
        error_range = (-1.5, 1.5)
        error_dot_range = (-8.0, 8.0)

        self.anfis_joints = [
            ANFISLayer(params.n_mf, params.mf_type, [error_range, error_dot_range]),
            ANFISLayer(params.n_mf, params.mf_type, [error_range, error_dot_range]),
        ]

        self.Phi_buffer: List[List[np.ndarray]] = [[], []]
        self.y_buffer: List[List[float]] = [[], []]
        self.input_buffer: List[List[np.ndarray]] = [[], []]

        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=params.eps,
            lam_os_max=params.lam_os_max,
            gate_pow=params.gate_pow,
            sigma_thresh_S=max(params.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()

        self.qref = None
        self._dt = float(getattr(self.arm, "dt", 0.01))
        self._qd_prev: Optional[np.ndarray] = None
        self._step = 0
        self._adapt_enabled = bool(params.online_adapt)
        self._freeze_reason = ""
        self._err_ema = None
        self._freeze_counter = 0
        self._last_comp = None

        # ✅ cached scales
        self._in_scale = np.array([self.p.input_scale_e, self.p.input_scale_ed], dtype=float)

    # -------------------------- Save / Load --------------------------

    def save_rules(self, path: Optional[str] = None):
        path = path or self.p.rules_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data: Dict[str, Any] = {"n_mf": self.p.n_mf, "mf_type": self.p.mf_type}
        for j, layer in enumerate(self.anfis_joints):
            data[f"consequent_{j}"] = layer.consequent.copy()
            packed = []
            for in_idx in range(len(layer.mfs)):
                packed.append([mf.pack() for mf in layer.mfs[in_idx]])
            data[f"mfs_{j}"] = np.array(packed, dtype=object)
        np.savez(path, **data)

    def load_rules(self, path: Optional[str] = None) -> bool:
        path = path or self.p.rules_path
        if not os.path.exists(path):
            return False
        z = np.load(path, allow_pickle=True)
        for j, layer in enumerate(self.anfis_joints):
            keyc = f"consequent_{j}"
            keym = f"mfs_{j}"
            if keyc in z:
                layer.consequent = z[keyc].copy()
            if keym in z:
                blob = z[keym]
                for in_idx in range(len(layer.mfs)):
                    for mf_idx in range(len(layer.mfs[in_idx])):
                        layer.mfs[in_idx][mf_idx].unpack(tuple(blob[in_idx][mf_idx]))
        return True

    # -------------------------- Utilities ---------------------------

    def init_pd(self, Kp_q: np.ndarray, Kd_q: np.ndarray, bias: float = 0.0):
        Kp_q = np.asarray(Kp_q, dtype=float).reshape(-1)
        Kd_q = np.asarray(Kd_q, dtype=float).reshape(-1)
        if Kp_q.size == 1:
            Kp_q = np.repeat(Kp_q, 2)
        if Kd_q.size == 1:
            Kd_q = np.repeat(Kd_q, 2)
        for j in range(2):
            self.anfis_joints[j].init_pd(Kp_q[j], Kd_q[j], bias=bias)

    def reset(self, q0: np.ndarray):
        self.qref = q0.copy()
        self.Phi_buffer = [[], []]
        self.y_buffer = [[], []]
        self.input_buffer = [[], []]
        self._qd_prev = None
        self._step = 0
        self._adapt_enabled = bool(self.p.online_adapt)
        self._freeze_reason = ""
        self._err_ema = None
        self._freeze_counter = 0
        self._last_comp = None

    def _compute_dynamics_compensation(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        n = len(q)
        comp = np.zeros(n, dtype=float)
        if self.p.enable_gravity_comp:
            g = gravityCOM_cached(
                self.env.skeleton._robot,
                self.env.skeleton._gravity_vec,
                symbolic=False,
            ).reshape(-1)
            comp += g
        if self.p.enable_coriolis_comp:
            C = centrifugalCoriolisCOM_cached(self.env.skeleton._robot, symbolic=False)
            C = np.asarray(C)
            if C.ndim == 2:
                comp += C @ qd
            else:
                comp += C.reshape(-1)
        return comp

    def _compute_tau_id(self, q: np.ndarray, qd: np.ndarray) -> Optional[np.ndarray]:
        if self._qd_prev is None:
            self._qd_prev = qd.copy()
            return None
        qdd = (qd - self._qd_prev) / (self._dt + 1e-12)
        self._qd_prev = qd.copy()

        M = inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False)
        M = np.asarray(M, dtype=float)

        g = gravityCOM_cached(
            self.env.skeleton._robot,
            self.env.skeleton._gravity_vec,
            symbolic=False,
        ).reshape(-1)

        C = centrifugalCoriolisCOM_cached(self.env.skeleton._robot, symbolic=False)
        C = np.asarray(C, dtype=float)
        cqd = (C @ qd) if (C.ndim == 2) else C.reshape(-1)

        return M @ qdd + cqd + g

    def _maybe_freeze(self, err_norm: float):
        if not self._adapt_enabled:
            return

        if self.p.freeze_after_steps and (self._step >= self.p.freeze_after_steps):
            self._adapt_enabled = False
            self._freeze_reason = f"freeze_after_steps={self.p.freeze_after_steps}"
            if self.p.save_on_freeze:
                self.save_rules()
            return

        if self.p.freeze_err_thresh and self.p.freeze_err_thresh > 0.0:
            beta = float(self.p.err_ema_beta)
            if self._err_ema is None:
                self._err_ema = float(err_norm)
            else:
                self._err_ema = beta * float(self._err_ema) + (1.0 - beta) * float(err_norm)

            if self._err_ema <= self.p.freeze_err_thresh:
                self._freeze_counter += 1
            else:
                self._freeze_counter = 0

            if self._freeze_counter >= int(self.p.freeze_patience):
                self._adapt_enabled = False
                self._freeze_reason = (
                    f"freeze_err_thresh={self.p.freeze_err_thresh}, "
                    f"patience={self.p.freeze_patience}"
                )
                if self.p.save_on_freeze:
                    self.save_rules()

    # -------------------------- Main compute -------------------------

    def compute(self, x_d: np.ndarray, xd_d: np.ndarray, xdd_d: np.ndarray):
        joint = self.env.states["joint"][0]
        q, qd = joint[:2], joint[2:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:]
        self.env.skeleton._set_state(q, qd)

        e = x_d - x
        ed = xd_d - xd
        err_norm = float(np.linalg.norm(e))

        J = geometricJacobian_cached(self.env.skeleton._robot, symbolic=False)
        J_xy = J[0:2, :]
        n = q.shape[0]
        J_pinv, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)

        # joint proxy errors
        e_joint = J_pinv @ e
        ed_joint = J_pinv @ ed

        # ✅ scale inputs to match MF ranges (big stability win)
        e_joint_s = float(self._in_scale[0]) * e_joint
        ed_joint_s = float(self._in_scale[1]) * ed_joint

        tau_anfis = np.zeros(n, dtype=float)

        # forward + replay
        for j in range(n):
            inp = np.array([e_joint_s[j], ed_joint_s[j]], dtype=float)
            tau_anfis[j], wbar, _ = self.anfis_joints[j].forward(inp)

            if self.p.online_adapt and self._adapt_enabled:
                phi = self.anfis_joints[j].compute_phi(inp, wbar)
                self.Phi_buffer[j].append(phi)
                self.input_buffer[j].append(inp.copy())
                if len(self.Phi_buffer[j]) > self.p.buffer_size:
                    self.Phi_buffer[j].pop(0)
                if len(self.input_buffer[j]) > self.p.buffer_size:
                    self.input_buffer[j].pop(0)

        comp = self._compute_dynamics_compensation(q, qd)
        self._last_comp = comp.copy()
        tau_des = tau_anfis + comp

        # online teacher
        tau_id = None
        if self.p.online_adapt and self._adapt_enabled:
            tau_id = self._compute_tau_id(q, qd)

        if (tau_id is not None) and self.p.online_adapt and self._adapt_enabled:
            if self.p.teacher_mode == "id_full":
                y = tau_id
            else:
                y = tau_id - comp

            # ✅ teacher scale + clip (big stability win)
            y = float(self.p.teacher_scale) * np.asarray(y, dtype=float)
            y = np.clip(y, -float(self.p.teacher_clip), +float(self.p.teacher_clip))

            for j in range(n):
                self.y_buffer[j].append(float(y[j]))
                if len(self.y_buffer[j]) > self.p.buffer_size:
                    self.y_buffer[j].pop(0)

            # ✅ LSE update throttled + stabilized + smoothed
            if (self._step % max(int(self.p.adapt_every), 1)) == 0:
                for j in range(n):
                    m = min(len(self.Phi_buffer[j]), len(self.y_buffer[j]))
                    if m < int(self.p.min_fit_samples):
                        continue
                    Phi = np.asarray(self.Phi_buffer[j][-m:], dtype=float)
                    yy = np.asarray(self.y_buffer[j][-m:], dtype=float)

                    self.anfis_joints[j].update_consequent_lse_stable(
                        Phi, yy,
                        reg=float(self.p.lse_reg),
                        theta_clip=float(self.p.theta_clip),
                        theta_smooth=float(self.p.theta_smooth),
                        max_cond=float(self.p.max_cond),
                    )

            # ✅ Premise update (optional, clipped)
            if (self.p.lr_premise > 0.0) and (
                self._step % max(int(self.p.premise_update_every), 1) == 0
            ):
                for j in range(n):
                    m = min(len(self.input_buffer[j]), len(self.y_buffer[j]))
                    if m < int(self.p.premise_batch_size):
                        continue
                    B = int(self.p.premise_batch_size)
                    batch_inputs = np.asarray(self.input_buffer[j][-B:], dtype=float)
                    batch_targets = np.asarray(self.y_buffer[j][-B:], dtype=float)
                    self.anfis_joints[j].premise_sgd_step(
                        batch_inputs,
                        batch_targets,
                        lr=float(self.p.lr_premise),
                        grad_clip=float(self.p.premise_grad_clip),
                        width_min=float(self.p.mf_width_min),
                        width_max=float(self.p.mf_width_max),
                    )

            self._maybe_freeze(err_norm)
            if self.p.autosave_every and self.p.autosave_every > 0:
                if (self._step % int(self.p.autosave_every)) == 0:
                    self.save_rules()

        # Operational-space guard and gating
        M = inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False)
        Minv = np.linalg.inv(np.asarray(M, dtype=float))
        S = J_xy @ Minv @ J_xy.T

        Lambda, lam_os, eta, eta2, xd_d_scaled, xdd_d_scaled, dyn_diag = op_space_guard_and_gate(
            S, xd_d.copy(), xdd_d.copy(), self.dp
        )

        # Muscle allocation
        geom = self.env.states["geometry"]
        lenvel = geom[:, :2, :]
        R = geom[:, 2:2 + self.env.skeleton.dof, :][0]
        Fmax_vec = get_Fmax_vec(self.env, R.shape[1])

        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta, self.mp)

        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]

        a = force_to_activation_bisect(
            F_des, lenvel, self.env.muscle, flpe, Fmax_vec,
            iters=self.p.bisect_iters
        )

        af_now = active_force_from_activation(a, lenvel, self.env.muscle)
        F_pred = Fmax_vec * (af_now + flpe)
        F_corr = saturation_repair_tau(
            -R, F_pred, a,
            self.env.muscle.min_activation, 1.0,
            Fmax_vec, tau_des=tau_des
        )
        if np.any(np.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(
                F_corr, lenvel, self.env.muscle, flpe, Fmax_vec,
                iters=max(4, self.p.bisect_iters - 4)
            )

        # NOTE: q_ref uses *unscaled* errors (keep behavior consistent)
        q_ref = q + e_joint
        qd_ref = qd + ed_joint

        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ, alpha_J=None, k_manip=None)
        extra = pack_diag(
            lam_os=lam_os,
            eta=eta,
            eta2=eta2,
            anfis_adapt_enabled=float(1.0 if self._adapt_enabled else 0.0),
            anfis_freeze_reason=self._freeze_reason,
            err_norm=err_norm,
            err_ema=float(self._err_ema) if (self._err_ema is not None) else np.nan,
        )
        diag = merge_diag(kin_diag, dyn_diag, mus_diag, extra)

        self._step += 1

        return {
            "tau_des": tau_des,
            "R": R,
            "Fmax": Fmax_vec,
            "F_des": F_des,
            "act": a,
            "q": q,
            "qd": qd,
            "q_ref": q_ref,
            "qd_ref": qd_ref,
            "x": x,
            "xd": xd,
            "xref_tuple": (x_d, xd_d_scaled, xdd_d_scaled),
            "eta": eta2,
            "diag": diag,
        }

    def get_rule_base(self, joint_idx: int = 0) -> np.ndarray:
        return self.anfis_joints[joint_idx].consequent.copy()

    def set_rule_base(self, consequent: np.ndarray, joint_idx: int = 0):
        self.anfis_joints[joint_idx].consequent = consequent.copy()

