# controller/anfis_controller.py
"""
ANFIS Controller (Takagi-Sugeno) with SAFE true-online consequent learning.

Important notes (why previous online_adapt sometimes “didn’t work”):
1) If online_adapt=True but you never call adapt_online(), nothing learns
   unless learning happens inside compute(). This file DOES adapt inside compute().

2) If your teacher is ID based on noisy qdd (finite differences), the LSE target
   becomes extremely noisy and can explode. This file avoids that by default:
   - Teacher = PD/PID in joint-error proxy space (stable)
   - Optional ID teacher exists but is filtered + gated

3) To prevent blow-up from learning torques that are not physically feasible,
   this file can "anchor" the teacher target toward tau_real_est (estimated from
   the current feasible muscle forces). This keeps LSE targets realistic.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

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


@dataclass
class ANFISParams:
    # ANFIS structure
    n_mf: int = 5
    mf_type: str = "gaussian"

    # Online adaptation (consequents only)
    online_adapt: bool = True
    lse_reg: float = 1e-2
    adapt_every: int = 50
    min_fit_samples: int = 150
    buffer_size: int = 400

    # Teacher inside controller
    teacher_mode: str = "pd"   # 'pd', 'pid', or 'id' (optional)
    Kp_teacher: float = 20.0
    Kd_teacher: float = 4.0
    Ki_teacher: float = 0.0
    integ_clip: float = 0.5
    torque_clip: float = 200.0

    # Alpha schedule: tau_mix = (1-alpha)*teacher + alpha*anfis
    alpha_final: float = 1.0
    alpha_warmup_steps: int = 1500

    # Dynamics compensation (added after mixing)
    enable_gravity_comp: bool = True
    enable_coriolis_comp: bool = True

    # Optional ID teacher filtering (only used if teacher_mode='id')
    id_qdd_ema: float = 0.95     # EMA on qdd
    id_gate_max_qdd: float = 100.0  # if |qdd| too large -> ignore ID target

    # τ_real anchoring (prevents LSE from learning impossible torques)
    use_tau_real_anchor: bool = True
    tau_real_anchor_w: float = 0.5   # y := (1-w)*y + w*clip(y toward tau_real_est)
    tau_real_clip: float = 150.0     # clip teacher target magnitude before anchoring

    # Save/load
    rules_path: str = "ANFIS/saved_model/anfis_rules.npz"
    autosave_every: int = 200

    # Freeze adaptation (optional)
    freeze_after_steps: int = 0
    freeze_err_thresh: float = 0.0
    freeze_patience: int = 80
    err_ema_beta: float = 0.98

    # Guards / numerics
    eps: float = 1e-6
    gate_pow: float = 2.0
    sigma_thresh: float = 1e-4
    lam_os_max: float = 1e6

    # Muscle inversion
    bisect_iters: int = 16


class MembershipFunction:
    def __init__(self, mf_type: str, center: float, width: float, slope: float = 2.0):
        self.mf_type = mf_type
        self.c = float(center)
        self.a = float(width)
        self.b = float(slope)

    def __call__(self, x: float) -> float:
        x = float(x)
        if self.mf_type == "gaussian":
            return float(np.exp(-((x - self.c) ** 2) / (2.0 * self.a ** 2 + 1e-12)))
        if self.mf_type == "bell":
            diff = (x - self.c) / (self.a + 1e-12)
            return float(1.0 / (1.0 + np.abs(diff) ** (2.0 * self.b)))
        if self.mf_type == "triangular":
            return float(max(0.0, 1.0 - abs(x - self.c) / (self.a + 1e-12)))
        raise ValueError(f"Unknown MF type: {self.mf_type}")

    def pack(self):
        return (self.c, self.a, self.b, self.mf_type)

    def unpack(self, t):
        self.c = float(t[0])
        self.a = float(t[1])
        self.b = float(t[2])
        self.mf_type = str(t[3])


class ANFISLayer:
    def __init__(self, n_mf: int, mf_type: str, input_ranges: List[Tuple[float, float]]):
        self.n_mf = int(n_mf)
        self.n_inputs = len(input_ranges)
        self.n_rules = self.n_mf ** self.n_inputs

        self.mfs: List[List[MembershipFunction]] = []
        for (lo, hi) in input_ranges:
            centers = np.linspace(lo, hi, self.n_mf)
            width = (hi - lo) / max(self.n_mf - 1, 1) * 0.5
            self.mfs.append([MembershipFunction(mf_type, c, width) for c in centers])

        self.consequent = np.zeros((self.n_rules, self.n_inputs + 1), dtype=float)
        self.consequent[:, -1] = np.random.randn(self.n_rules) * 0.01

    def forward(self, inputs: np.ndarray):
        mu = []
        for i, x in enumerate(inputs):
            mu.append(np.array([mf(float(x)) for mf in self.mfs[i]], dtype=float))

        w = np.zeros(self.n_rules, dtype=float)
        idx = 0
        for i in range(self.n_mf):
            for j in range(self.n_mf):
                w[idx] = mu[0][i] * mu[1][j]
                idx += 1

        w_sum = float(np.sum(w) + 1e-12)
        w_bar = w / w_sum

        inputs_ext = np.append(inputs, 1.0)
        f = self.consequent @ inputs_ext
        out = float(np.dot(w_bar, f))
        return out, w_bar, f

    def compute_phi(self, inputs: np.ndarray, w_bar: np.ndarray) -> np.ndarray:
        inputs_ext = np.append(inputs, 1.0)
        return np.outer(w_bar, inputs_ext).flatten()

    def update_consequent_lse(self, Phi: np.ndarray, y: np.ndarray, reg: float):
        n_params = self.n_rules * (self.n_inputs + 1)
        Phi = np.asarray(Phi, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if Phi.ndim != 2 or Phi.shape[1] != n_params or Phi.shape[0] != y.shape[0]:
            return
        try:
            A = Phi.T @ Phi + float(reg) * np.eye(n_params)
            b = Phi.T @ y
            theta = np.linalg.solve(A, b)
            self.consequent = theta.reshape(self.n_rules, self.n_inputs + 1)
        except np.linalg.LinAlgError:
            pass

    def init_pd(self, Kp: float, Kd: float, bias: float = 0.0):
        self.consequent[:, 0] = float(Kp)
        self.consequent[:, 1] = float(Kd)
        self.consequent[:, 2] = float(bias)


class ANFISController:
    def __init__(self, env, arm, params: ANFISParams):
        self.env = env
        self.arm = arm
        self.p = params

        dof = int(getattr(self.env.skeleton, "dof", 2))
        self.qref = np.zeros(dof, dtype=float)  # simulator expects this always

        error_range = (-1.5, 1.5)
        error_dot_range = (-8.0, 8.0)

        self.anfis_joints = [
            ANFISLayer(params.n_mf, params.mf_type, [error_range, error_dot_range]),
            ANFISLayer(params.n_mf, params.mf_type, [error_range, error_dot_range]),
        ]

        self.Phi_buffer = [[], []]
        self.y_buffer = [[], []]
        self.buffer_size = int(params.buffer_size)

        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=params.eps,
            lam_os_max=params.lam_os_max,
            gate_pow=params.gate_pow,
            sigma_thresh_S=max(params.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()

        self._dt = float(getattr(self.arm, "dt", 0.01))
        self._step = 0

        self._ei = np.zeros(dof, dtype=float)

        # ID filtering state (only used in teacher_mode='id')
        self._qd_prev: Optional[np.ndarray] = None
        self._qdd_ema: Optional[np.ndarray] = None

        # freeze logic
        self._adapt_enabled = bool(params.online_adapt)
        self._err_ema = None
        self._freeze_counter = 0
        self._freeze_reason = ""

    # -------------------- save/load --------------------

    def save_rules(self, path: Optional[str] = None):
        path = path or self.p.rules_path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

        data = {"n_mf": int(self.p.n_mf), "mf_type": str(self.p.mf_type)}
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
            kc = f"consequent_{j}"
            km = f"mfs_{j}"
            if kc in z:
                layer.consequent = np.asarray(z[kc], dtype=float).copy()
            if km in z:
                blob = z[km]
                for in_idx in range(len(layer.mfs)):
                    for mf_idx in range(len(layer.mfs[in_idx])):
                        layer.mfs[in_idx][mf_idx].unpack(tuple(blob[in_idx][mf_idx]))
        return True

    def init_pd(self, Kp_q, Kd_q, bias: float = 0.0):
        Kp_q = np.asarray(Kp_q, dtype=float).reshape(-1)
        Kd_q = np.asarray(Kd_q, dtype=float).reshape(-1)
        if Kp_q.size == 1:
            Kp_q = np.repeat(Kp_q, 2)
        if Kd_q.size == 1:
            Kd_q = np.repeat(Kd_q, 2)
        for j in range(2):
            self.anfis_joints[j].init_pd(Kp_q[j], Kd_q[j], bias=bias)

    # -------------------- reset --------------------

    def reset(self, q0: np.ndarray):
        self.qref = np.asarray(q0, dtype=float).copy()
        self.Phi_buffer = [[], []]
        self.y_buffer = [[], []]
        self._step = 0
        self._ei[:] = 0.0
        self._qd_prev = None
        self._qdd_ema = None

        self._adapt_enabled = bool(self.p.online_adapt)
        self._err_ema = None
        self._freeze_counter = 0
        self._freeze_reason = ""

    # -------------------- helpers --------------------

    def _alpha(self) -> float:
        a_fin = float(np.clip(self.p.alpha_final, 0.0, 1.0))
        warm = int(self.p.alpha_warmup_steps)
        if warm <= 0:
            return a_fin
        return a_fin * min(1.0, self._step / float(warm))

    def _compute_tau_comp(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        n = len(q)
        tau_comp = np.zeros(n, dtype=float)

        if self.p.enable_gravity_comp:
            g = gravityCOM_cached(self.env.skeleton._robot, self.env.skeleton._gravity_vec, symbolic=False).reshape(-1)
            tau_comp += g

        if self.p.enable_coriolis_comp:
            C = centrifugalCoriolisCOM_cached(self.env.skeleton._robot, symbolic=False)
            C = np.asarray(C)
            if C.ndim == 2:
                tau_comp += C @ qd
            else:
                tau_comp += C.reshape(-1)

        return tau_comp

    def _tau_from_R_F(self, R: np.ndarray, F: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        F = np.asarray(F, dtype=float).reshape(-1)
        dof = int(getattr(self.env.skeleton, "dof", 2))

        if R.ndim != 2:
            return np.zeros(dof, dtype=float)

        if R.shape[0] == dof and R.shape[1] == F.shape[0]:
            return (R @ F).reshape(-1)
        if R.shape[1] == dof and R.shape[0] == F.shape[0]:
            return (R.T @ F).reshape(-1)

        try:
            return (R @ F).reshape(-1)
        except Exception:
            try:
                return (R.T @ F).reshape(-1)
            except Exception:
                return np.zeros(dof, dtype=float)

    def _maybe_freeze(self, err_norm: float):
        if not self._adapt_enabled:
            return

        if self.p.freeze_after_steps and self._step >= int(self.p.freeze_after_steps):
            self._adapt_enabled = False
            self._freeze_reason = f"freeze_after_steps={self.p.freeze_after_steps}"
            return

        if self.p.freeze_err_thresh and self.p.freeze_err_thresh > 0.0:
            beta = float(self.p.err_ema_beta)
            if self._err_ema is None:
                self._err_ema = float(err_norm)
            else:
                self._err_ema = beta * float(self._err_ema) + (1.0 - beta) * float(err_norm)

            if self._err_ema <= float(self.p.freeze_err_thresh):
                self._freeze_counter += 1
            else:
                self._freeze_counter = 0

            if self._freeze_counter >= int(self.p.freeze_patience):
                self._adapt_enabled = False
                self._freeze_reason = f"freeze_err_thresh={self.p.freeze_err_thresh}"

    def _teacher_tau_pd_pid(self, e_joint: np.ndarray, ed_joint: np.ndarray) -> np.ndarray:
        Kp = float(self.p.Kp_teacher)
        Kd = float(self.p.Kd_teacher)
        Ki = float(self.p.Ki_teacher)

        if self.p.teacher_mode.lower() == "pid" and Ki > 0.0:
            self._ei = np.clip(self._ei + e_joint * self._dt, -self.p.integ_clip, self.p.integ_clip)
        else:
            self._ei[:] = 0.0

        tau = Kp * e_joint + Kd * ed_joint + Ki * self._ei
        tau = np.clip(tau, -self.p.torque_clip, self.p.torque_clip)
        return tau

    def _teacher_tau_id_filtered(self, q: np.ndarray, qd: np.ndarray) -> Optional[np.ndarray]:
        # ID teacher is optional; we filter qdd hard.
        if self._qd_prev is None:
            self._qd_prev = qd.copy()
            self._qdd_ema = np.zeros_like(qd)
            return None

        qdd = (qd - self._qd_prev) / (self._dt + 1e-12)
        self._qd_prev = qd.copy()

        if np.any(np.abs(qdd) > float(self.p.id_gate_max_qdd)):
            return None  # ignore insane qdd spikes

        beta = float(self.p.id_qdd_ema)
        if self._qdd_ema is None:
            self._qdd_ema = qdd.copy()
        else:
            self._qdd_ema = beta * self._qdd_ema + (1.0 - beta) * qdd

        M = inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False)
        M = np.asarray(M, dtype=float)
        tau_comp = self._compute_tau_comp(q, qd)
        return (M @ self._qdd_ema + tau_comp).reshape(-1)

    def _anchor_teacher_to_tau_real(self, y: np.ndarray, tau_real_est: np.ndarray) -> np.ndarray:
        if not self.p.use_tau_real_anchor:
            return y

        y = np.asarray(y, dtype=float).reshape(-1)
        tau_real_est = np.asarray(tau_real_est, dtype=float).reshape(-1)

        # step 1: clip raw teacher target
        y_clip = np.clip(y, -float(self.p.tau_real_clip), float(self.p.tau_real_clip))

        # step 2: softly pull toward physically feasible torque estimate
        w = float(np.clip(self.p.tau_real_anchor_w, 0.0, 1.0))
        # clamp tau_real_est too (avoid numerical junk)
        tr = np.clip(tau_real_est, -float(self.p.tau_real_clip), float(self.p.tau_real_clip))
        return (1.0 - w) * y_clip + w * tr

    # -------------------- compute --------------------

    def compute(self, x_d: np.ndarray, xd_d: np.ndarray, xdd_d: np.ndarray):
        joint = self.env.states["joint"][0]
        q, qd = joint[:2], joint[2:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:]
        self.env.skeleton._set_state(q, qd)

        self.qref = q.copy()

        e = x_d - x
        ed = xd_d - xd
        err_norm = float(np.linalg.norm(e))

        J = geometricJacobian_cached(self.env.skeleton._robot, symbolic=False)
        J_xy = J[0:2, :]
        n = q.shape[0]

        J_pinv, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)

        e_joint = J_pinv @ e
        ed_joint = J_pinv @ ed

        # ---------------- teacher (default PD/PID) ----------------
        mode = self.p.teacher_mode.lower().strip()
        if mode in ("pd", "pid"):
            tau_teacher = self._teacher_tau_pd_pid(e_joint, ed_joint)
        elif mode == "id":
            tau_id = self._teacher_tau_id_filtered(q, qd)
            if tau_id is None:
                tau_teacher = self._teacher_tau_pd_pid(e_joint, ed_joint)
            else:
                tau_teacher = np.clip(tau_id, -self.p.torque_clip, self.p.torque_clip)
        else:
            tau_teacher = self._teacher_tau_pd_pid(e_joint, ed_joint)

        # ---------------- ANFIS forward ----------------
        tau_anfis = np.zeros(n, dtype=float)
        for i in range(n):
            inp = np.array([e_joint[i], ed_joint[i]], dtype=float)
            tau_anfis[i], w_bar, _ = self.anfis_joints[i].forward(inp)

            if self.p.online_adapt and self._adapt_enabled:
                phi = self.anfis_joints[i].compute_phi(inp, w_bar)
                self.Phi_buffer[i].append(phi)
                if len(self.Phi_buffer[i]) > self.buffer_size:
                    self.Phi_buffer[i].pop(0)

        # ---------------- mix teacher + anfis ----------------
        alpha = self._alpha()
        tau_mix = (1.0 - alpha) * tau_teacher + alpha * tau_anfis

        tau_comp = self._compute_tau_comp(q, qd)
        tau_des = tau_mix + tau_comp

        # ---------------- operational-space guard ----------------
        M = inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False)
        Minv = np.linalg.inv(np.asarray(M, dtype=float))
        S = J_xy @ Minv @ J_xy.T

        Lambda, lam_os, eta, eta2, xd_d_scaled, xdd_d_scaled, dyn_diag = op_space_guard_and_gate(
            S, xd_d.copy(), xdd_d.copy(), self.dp
        )

        # ---------------- muscle allocation ----------------
        geom = self.env.states["geometry"]
        lenvel = geom[:, :2, :]
        R = geom[:, 2:2 + self.env.skeleton.dof, :][0]

        Fmax_vec = get_Fmax_vec(self.env, R.shape[-1])
        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta, self.mp)

        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]

        a = force_to_activation_bisect(F_des, lenvel, self.env.muscle, flpe, Fmax_vec, iters=self.p.bisect_iters)

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
            af_now = active_force_from_activation(a, lenvel, self.env.muscle)
            F_pred = Fmax_vec * (af_now + flpe)

        # ---------------- τ_real estimate (feasible torque) ----------------
        tau_real_est = self._tau_from_R_F(R, F_pred)

        # ---------------- ONLINE LSE UPDATE (INSIDE compute) ----------------
        if self.p.online_adapt and self._adapt_enabled:
            # target torque is teacher (optionally anchored to feasible tau_real_est)
            y = tau_teacher.copy()
            y = self._anchor_teacher_to_tau_real(y, tau_real_est)

            for j in range(n):
                self.y_buffer[j].append(float(y[j]))
                if len(self.y_buffer[j]) > self.buffer_size:
                    self.y_buffer[j].pop(0)

            if (self._step % max(int(self.p.adapt_every), 1)) == 0:
                for j in range(n):
                    m = min(len(self.Phi_buffer[j]), len(self.y_buffer[j]))
                    if m < int(self.p.min_fit_samples):
                        continue
                    Phi = np.asarray(self.Phi_buffer[j][-m:], dtype=float)
                    yy = np.asarray(self.y_buffer[j][-m:], dtype=float)
                    self.anfis_joints[j].update_consequent_lse(Phi, yy, reg=float(self.p.lse_reg))

            self._maybe_freeze(err_norm)

            if self.p.autosave_every and self.p.autosave_every > 0:
                if (self._step % int(self.p.autosave_every)) == 0:
                    try:
                        self.save_rules(self.p.rules_path)
                    except Exception:
                        pass

        # ---------------- diagnostics ----------------
        kin_diag = pack_diag(sminJ=sminJ, lamJ=lamJ, alpha_J=None, k_manip=None)
        extra = pack_diag(
            alpha=alpha,
            teacher_mode=str(self.p.teacher_mode),
            anfis_adapt_enabled=float(1.0 if self._adapt_enabled else 0.0),
            anfis_freeze_reason=self._freeze_reason,
            err_norm=err_norm,
            err_ema=float(self._err_ema) if self._err_ema is not None else np.nan,
            tau_teacher_0=float(tau_teacher[0]),
            tau_teacher_1=float(tau_teacher[1]),
            tau_real_est_0=float(tau_real_est[0]) if tau_real_est.size > 0 else np.nan,
            tau_real_est_1=float(tau_real_est[1]) if tau_real_est.size > 1 else np.nan,
        )
        diag = merge_diag(
            kin_diag, dyn_diag, mus_diag,
            pack_diag(lam_os=lam_os, eta=eta, eta2=eta2),
            extra
        )

        self._step += 1

        return {
            "tau_des": tau_des,
            "R": R,
            "Fmax": Fmax_vec,
            "F_des": F_des,
            "act": a,
            "q": q,
            "qd": qd,
            "x": x,
            "xd": xd,
            "xref_tuple": (x_d, xd_d_scaled, xdd_d_scaled),
            "eta": eta2,
            "diag": diag,
        }

