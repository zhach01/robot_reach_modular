# controller/nmpc_task.py
import numpy as np
from dataclasses import dataclass, field

from model_lib.skeleton_numpy import (
    geometricJacobian_cached,
    geometricJacobianDot_cached,
    inertiaMatrixCOM_cached,
    centrifugalCoriolisCOM_cached,
    gravityCOM_cached,
)

from utils.kinematics_guard import KinGuardParams, adaptive_dls_pinv, scale_task_by_J
from utils.dynamics_guard import DynGuardParams, op_space_guard_and_gate

# τ→activations shim to keep env API happy when muscles are off, and allocator
from utils.muscle_guard import MuscleGuardParams, solve_muscle_forces
from muscles.muscle_tools import (
    get_Fmax_vec, force_to_activation_bisect, active_force_from_activation,
    saturation_repair_tau, apply_internal_force_regulation
)

# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------

def _vec2(x, default=(0.0, 0.0)):
    """Return (2,), or last row of (N,2); also accepts scalar -> (a,0)."""
    if x is None:
        return np.array(default, dtype=float)
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        return np.array([float(a), 0.0])
    if a.ndim == 1:
        return a[:2] if a.size >= 2 else np.pad(a, (0, 2 - a.size))
    if a.ndim == 2 and a.shape[1] == 2:
        return a[-1]
    return a.reshape(-1)[:2]


def _preview_2(N, maybe):
    """
    Ensure an (N,2) preview:
      - if maybe is (N,2) -> return as-is (trim/pad to N)
      - if maybe is (2,)   -> tile to (N,2)
      - if None            -> zeros
    """
    if maybe is None:
        row = np.zeros(2)
        return np.tile(row, (N, 1))
    a = np.asarray(maybe, dtype=float)
    if a.ndim == 1:
        row = _vec2(a)
        return np.tile(row, (N, 1))
    if a.ndim == 2 and a.shape[1] == 2:
        if a.shape[0] >= N:
            return a[:N]
        out = np.zeros((N, 2))
        out[:a.shape[0]] = a
        out[a.shape[0]:] = a[-1]
        return out
    row = _vec2(a)
    return np.tile(row, (N, 1))


def _activation_to_excitation(a_star, a_now, dt, tau_up, tau_down):
    """
    Invert a' = (u - a)/tau  -> choose u so a_{k+1} ≈ a_star in one step.
    Uses separate time constants for up/down.
    """
    a_star = np.asarray(a_star, dtype=float)
    a_now  = np.asarray(a_now, dtype=float)
    tau = np.where(a_star >= a_now, tau_up, tau_down)
    tau = np.clip(tau, 1e-6, 10.0)
    dt  = float(max(dt, 1e-6))
    u = a_star + tau * (a_star - a_now) / dt
    return np.clip(u, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Params
# -----------------------------------------------------------------------------

@dataclass
class NMPCParams:
    # horizon & timing
    N: int = 30
    dt_mpc: float | None = None  # if None, use arm.dt

    # stage / terminal weights
    Wx: np.ndarray = field(default_factory=lambda: np.diag([1500.0, 1500.0]))
    Wv: np.ndarray = field(default_factory=lambda: np.diag([20.0, 20.0]))
    Wu: np.ndarray = field(default_factory=lambda: np.diag([2e-3, 2e-3]))
    # terminal: make the goal sticky (x_N≈x*, xdot_N≈0)
    WN: np.ndarray = field(default_factory=lambda: np.diag([40e4, 40e4, 80e2, 80e2]))

    # regularization / limits
    lam_reg: float = 5e-4          # small ridge on inputs
    lam_du: float  = 2e-3          # input slew penalty (0 disables)
    Fmax: float = 600.0            # task-force clip (higher helps fast tracking)
    tau_clip: float = 600.0        # joint torque clip

    # guards (original)
    eps: float = 1e-8
    lam_os_max: float = 40.0
    sigma_thresh: float = 1e-4
    gate_pow: float = 2.0

    # plant comp
    enable_inertia_comp: bool = True
    enable_gravity_comp: bool = True
    enable_velocity_comp: bool = True
    enable_joint_damping: bool = True

    # Analytic IK for qref (L1, L2 can be read from skeleton)
    ik_use: bool = True
    L1: float | None = None
    L2: float | None = None
    elbow_pref: str = "closest"  # "closest", "up", "down"

    # muscles (τ→a path)
    use_muscles: bool = False

    # --- (A) passive-torque compensation ---
    compensate_passive: bool = True

    # --- (B) excitation prediction to cancel activation lag ---
    send_excitation: bool = True           # send u (not a) to env if it expects excitation
    tau_up: float = 0.030                  # s
    tau_down: float = 0.050                # s

    # --- (C) internal force (co-contraction) for conditioning ---
    enable_internal_force: bool = True
    cocon_a0: float = 0.05                 # nominal co-contraction level (0.03–0.08)
    internal_force_scale: float = 0.30     # blending scale (0..1)

    # --- (D) allocator / scaling knobs ---
    allocator_scale: float = 1.0           # pass-through (kept for compatibility)
    residual_weight: float = 100.0         # if allocator supports residual slack (hint)

    # --- (E) lower activation floor (optional) ---
    min_activation_override: float | None = 0.005

    # solver bits
    bisect_iters: int = 18
    linesearch_eps: float = 1e-6
    linesearch_safety: float = 0.2

    # ======= singularity regularisation & avoidance (new) =======
    sigma_target: float = 0.10          # [m/rad]-ish; tune to your arm scale
    lam_sing_gain: float = 60.0         # scales the added damping
    lam_sing_max: float = 80.0          # cap on extra damping

    # gate the final task-force with the same alpha_J used to scale (xd, xdd)
    force_gate_with_alpha: bool = True

    # nullspace posture to keep away from singular shapes
    ns_avoid_enable: bool = True
    ns_q_rest: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8]))  # radians
    ns_kp_base: float = 2.0         # Nm/rad baseline
    ns_kp_boost: float = 40.0       # extra gain near singularity
    ns_kp_max: float = 60.0         # cap
    ns_sigma_target: float = 0.10   # where boosting begins


# -----------------------------------------------------------------------------
# Controller
# -----------------------------------------------------------------------------

class NonlinearMPCController:
    """
    Task-space MPC on the discrete op-space model with full preview:
      xdd = Λ_reg(F - μ)
      y = [x; xdot]
      y_{k+1} = A y_k + B (F_k - F_ff,k) + c_k,   τ = Jᵀ F
      where  F_ff,k = μ + Λ_reg^{-1} xdd_ref,k  and  c_k = [½dt² xdd_ref,k; dt xdd_ref,k]
    """

    def __init__(self, env, arm, params: NMPCParams):
        self.env = env
        self.arm = arm
        self.p = params
        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=params.eps, lam_os_max=params.lam_os_max,
            gate_pow=params.gate_pow, sigma_thresh_S=max(params.sigma_thresh, 1e-9)
        )
        self.mp = MuscleGuardParams()
        self.qref = None

    # ---------- dynamics helpers ----------
    def _dyn_terms(self, q, qd):
        n = len(q)
        M = inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False) if self.p.enable_inertia_comp else np.eye(n)
        if self.p.enable_gravity_comp:
            g = gravityCOM_cached(self.env.skeleton._robot, self.env.skeleton._gravity_vec, symbolic=False).reshape(-1)
        else:
            g = np.zeros_like(q)
        if self.p.enable_velocity_comp:
            C_any = centrifugalCoriolisCOM_cached(self.env.skeleton._robot, symbolic=False)
        else:
            C_any = np.zeros((n, n))
        D = float(self.arm.damping) if self.p.enable_joint_damping else 0.0
        C_any = np.asarray(C_any)
        h = (C_any @ qd if C_any.ndim == 2 else C_any.reshape(-1)) + g + D * qd
        return M, h

    # ---- Analytic IK (2-link planar) ----
    def _get_links(self):
        if self.p.L1 is not None and self.p.L2 is not None:
            return float(self.p.L1), float(self.p.L2)
        sk = self.env.skeleton
        for cand in [getattr(sk, "link_lengths", None), getattr(sk, "L", None), getattr(sk, "links", None)]:
            if cand is not None:
                arr = np.asarray(cand, dtype=float).reshape(-1)
                if arr.size >= 2:
                    return float(arr[0]), float(arr[1])
        for n1, n2 in [("L1", "L2"), ("l1", "l2"), ("upper_len", "fore_len")]:
            if hasattr(sk, n1) and hasattr(sk, n2):
                return float(getattr(sk, n1)), float(getattr(sk, n2))
        return 0.30, 0.25  # conservative fallback

    @staticmethod
    def _angle_wrap(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    def _ik_analytic(self, x_d, q_curr):
        L1, L2 = self._get_links()
        px, py = float(x_d[0]), float(x_d[1])
        r = np.sqrt(px*px + py*py)
        r = np.clip(r, abs(L1 - L2) + 1e-9, (L1 + L2) - 1e-9)
        if r > 0:
            s = r / np.sqrt(px*px + py*py + 1e-12)
            px, py = px * s, py * s
        c2 = (px*px + py*py - L1*L1 - L2*L2) / (2.0 * L1 * L2)
        c2 = float(np.clip(c2, -1.0, 1.0))
        s2 = np.sqrt(max(0.0, 1.0 - c2*c2))
        th2_up, th2_dn = np.arctan2(+s2, c2), np.arctan2(-s2, c2)
        k1, k2 = L1 + L2 * c2, L2 * s2
        th1_up = np.arctan2(py, px) - np.arctan2(+k2, k1)
        th1_dn = np.arctan2(py, px) - np.arctan2(-k2, k1)
        up, dn = np.array([th1_up, th2_up]), np.array([th1_dn, th2_dn])
        if   self.p.elbow_pref == "up":   qref = up
        elif self.p.elbow_pref == "down": qref = dn
        else:
            qref = up if np.linalg.norm(self._angle_wrap(up - q_curr)) <= np.linalg.norm(self._angle_wrap(dn - q_curr)) else dn
        if hasattr(self.env.skeleton, "joint_limits"):
            qmin, qmax = self.env.skeleton.joint_limits
            qref = np.minimum(np.maximum(qref, qmin), qmax)
        return qref

    # ---------- MPC (full preview) ----------
    def _solve_horizon(self, y0, Xref, Xdref, Xddref, J_xy, Jdot_xy, M, h, dt, lam_extra=0.0):
        """
        Solve min_u Σ_k ||Q½(y_k - y^ref_k)||² + ||R½ u_k||²  + terminal,
        with y_{k+1} = A y_k + B u_k + c_k (where c_k depends on xdd_ref,k),
        and F_k = u_k + F_ff,k,  F_ff,k = μ + Λ_reg^{-1} xdd_ref,k.
        lam_extra: extra isotropic damping for singularity regularisation.
        """
        # Op-space metric & regularization
        S_os = J_xy @ np.linalg.solve(M, J_xy.T)               # Λ^{-1}
        _, lam_os, _, _, _, _, _ = op_space_guard_and_gate(S_os, Xdref[0], Xdref[0], self.dp)

        # singularity regularisation added here
        lam_eff = float(lam_os) + float(lam_extra)

        S_reg      = S_os + lam_eff * np.eye(2)                 # Λ_reg^{-1}
        Lambda_reg = np.linalg.inv(S_reg)                       # Λ_reg
        mu_task    = Lambda_reg @ (J_xy @ np.linalg.solve(M, h) - Jdot_xy @ y0[2:])

        I2 = np.eye(2)
        A  = np.block([[I2, dt*I2],
                       [np.zeros((2,2)), I2]])
        B  = np.block([[0.5*(dt**2) * Lambda_reg],
                       [dt * Lambda_reg]])

        # stage-wise c_k and F_ff,k from Xddref
        N = self.p.N
        c_list   = [np.hstack([0.5*(dt**2) * Xddref[k], dt * Xddref[k]]) for k in range(N)]
        Fff_list = [mu_task + S_reg @ Xddref[k] for k in range(N)]
        Fff_stack = np.concatenate(Fff_list, axis=0)  # (2N,)

        # build reachability T
        nY, nU = 4 * N, 2 * N
        As = [np.eye(4)]
        for _ in range(1, N+1):
            As.append(A @ As[-1])

        T = np.zeros((nY, nU))
        for k in range(N):
            for j in range(k):
                T[4*k:4*k+4, 2*j:2*j+2] = As[k-1-j] @ B

        # affine stack with stage-wise c_k
        d_stack   = np.zeros(nY)
        Sy0_stack = np.zeros(nY)
        for k in range(N):
            d_k = np.zeros(4)
            for i in range(k):
                d_k += As[k-1-i] @ c_list[i]
            idx = 4*k
            d_stack[idx:idx+4]   = d_k
            Sy0_stack[idx:idx+4] = As[k] @ y0

        # reference stack (full preview)
        Yref = np.zeros(nY)
        for k in range(N):
            idx = 4*k
            Yref[idx:idx+2]     = Xref[k]
            Yref[idx+2:idx+4]   = Xdref[k]

        # quadratic
        Qblk = np.block([[self.p.Wx, np.zeros((2,2))],
                         [np.zeros((2,2)), self.p.Wv]])
        Q = np.kron(np.eye(N), Qblk)
        Q[-4:, -4:] += self.p.WN
        R = np.kron(np.eye(N), self.p.Wu)

        if self.p.lam_du > 0.0:
            Dm = np.eye(2*N) - np.roll(np.eye(2*N), 2, axis=1)  # Δu
            Dm[:2, :] = 0.0
            H = T.T @ Q @ T + R + self.p.lam_reg*np.eye(nU) + self.p.lam_du*(Dm.T @ Dm)
        else:
            H = T.T @ Q @ T + R + self.p.lam_reg*np.eye(nU)

        b = T.T @ Q @ (Yref - Sy0_stack - d_stack)
        ustack = np.linalg.solve(H, b)

        # back to forces, clip first control
        Fstack = ustack + Fff_stack
        F0 = np.clip(Fstack[:2], -self.p.Fmax, self.p.Fmax)
        return F0, Lambda_reg, mu_task, lam_eff

    # ---------- API ----------
    def reset(self, q0):
        self.qref = q0.copy()

    def compute(self, x_d, xd_d, xdd_d):
        # state & kinematics
        joint = self.env.states["joint"][0]
        q, qd = joint[:2], joint[2:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:]
        self.env.skeleton._set_state(q, qd)

        # Jacobian & scaling (for safety near singularities)
        J = geometricJacobian_cached(self.env.skeleton._robot, symbolic=False)
        J_xy = J[0:2, :]
        Jdot = geometricJacobianDot_cached(self.env.skeleton._robot, symbolic=False)
        Jdot_xy = Jdot[0:2, :]

        n = q.shape[0]
        _, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, KinGuardParams())

        # Scale task rates/accels; keep alpha_J for later force gating
        xd_d, xdd_d, alpha_J = scale_task_by_J(_vec2(xd_d), _vec2(xdd_d), sminJ, self.kp)

        # IK posture (closed-form)
        if self.p.ik_use:
            self.qref = self._ik_analytic(_vec2(x_d), q)

        # dynamics
        M, h = self._dyn_terms(q, qd)
        y0 = np.hstack([x, xd])

        # singularity regularisation amount (extra damping)
        if sminJ <= self.p.sigma_target:
            ratio = (self.p.sigma_target / max(sminJ, 1e-9)) - 1.0  # >=0 near singular
            lam_sing = np.clip(self.p.lam_sing_gain * (ratio ** 2), 0.0, self.p.lam_sing_max)
        else:
            lam_sing = 0.0

        # full preview stacks (if caller gives (N,2), we use it; else we tile)
        N = self.p.N
        dt = self.p.dt_mpc if (self.p.dt_mpc is not None) else self.arm.dt
        Xref  = _preview_2(N, x_d)
        Xdref = _preview_2(N, xd_d)
        Xddref= _preview_2(N, xdd_d)

        # MPC solve with lam_extra
        F0, _, _, lam_eff = self._solve_horizon(y0, Xref, Xdref, Xddref,
                                                J_xy, Jdot_xy, M, h, dt,
                                                lam_extra=lam_sing)

        # gate the force with alpha_J near singularities (optional, conservative)
        if self.p.force_gate_with_alpha:
            F0 = alpha_J * F0

        # base task torque
        tau_task = J_xy.T @ F0

        # -------- Nullspace singularity avoidance (posture term) --------
        tau_ns = 0.0
        if self.p.ns_avoid_enable:
            # manipulability-aware gain
            if sminJ <= self.p.ns_sigma_target:
                r = (self.p.ns_sigma_target / max(sminJ, 1e-9)) - 1.0
                kp_eff = min(self.p.ns_kp_base + self.p.ns_kp_boost * (r ** 2), self.p.ns_kp_max)
            else:
                kp_eff = self.p.ns_kp_base

            q_rest = np.asarray(self.p.ns_q_rest, dtype=float).reshape(-1)[:n]
            e_post = q_rest - q  # pull toward safe posture
            # nullspace projector N = I - J^T (JJ^T)^-1 J
            JJt = J_xy @ J_xy.T
            JJt_inv = np.linalg.inv(JJt + 1e-9 * np.eye(2))
            N_null = np.eye(n) - J_xy.T @ JJt_inv @ J_xy
            tau_ns = kp_eff * (N_null @ e_post)
        else:
            kp_eff = 0.0

        # total torque and clip
        tau_des = np.clip(tau_task + tau_ns, -self.p.tau_clip, self.p.tau_clip)

        # ===================== τ → activation/excitation path =====================
        # Geometry/state used by both branches
        geom   = self.env.states["geometry"]
        lenvel = geom[:, :2, :]                                   # (1,2,m)
        Rm     = geom[:, 2:2 + self.env.skeleton.dof, :][0]       # (dof,m)
        Fmax_v = get_Fmax_vec(self.env, Rm.shape[1])              # (m,)

        # Passive force & torque (A: passive torque compensation)
        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe  = self.env.states["muscle"][0, idx_flpe, :]         # (m,)
        F_pass = Fmax_v * flpe                                    # (m,)
        # FIX: use -Rm @ F_pass (Rm is (dof,m)), NOT -Rm.T @ F_pass
        tau_passive = -Rm @ F_pass if self.p.compensate_passive else 0.0  # (dof,)

        # Allocate for ACTIVE torque only
        tau_need = tau_des - tau_passive

        # Branch 1: muscles OFF (shim only; still compute a/u for env compatibility)
        if not self.p.use_muscles:
            F_des, _ = solve_muscle_forces(tau_need, Rm, Fmax_v, self.p.allocator_scale, self.mp)

            a_des = force_to_activation_bisect(
                F_des, lenvel, self.env.muscle, flpe, Fmax_v, iters=self.p.bisect_iters
            )

            # (E) lower activation floor if requested
            if self.p.min_activation_override is not None:
                a_min = float(self.p.min_activation_override)
            else:
                a_min = float(getattr(self.env.muscle, "min_activation", 0.0))
            a_des = np.clip(a_des, a_min, 1.0)

            # (B) predict excitation to reach a_des next step if the env consumes "u"
            if self.p.send_excitation:
                try:
                    idx_act = names.index("activation")
                    a_now = self.env.states["muscle"][0, idx_act, :]
                except Exception:
                    a_now = a_des
                payload_act = _activation_to_excitation(a_des, a_now, dt, self.p.tau_up, self.p.tau_down)
            else:
                payload_act = a_des

            af_now = active_force_from_activation(a_des, lenvel, self.env.muscle)
            F_pred = Fmax_v * (af_now + flpe)
            F_corr = saturation_repair_tau(-Rm, F_pred, a_des, a_min, 1.0,
                                           Fmax_v, tau_des=tau_des)
            if np.any(np.abs(F_corr - F_pred) > 1e-9):
                a_des = force_to_activation_bisect(F_corr, lenvel, self.env.muscle, flpe, Fmax_v,
                                                   iters=max(4, self.p.bisect_iters - 4))
                if self.p.send_excitation:
                    try:
                        idx_act = names.index("activation")
                        a_now = self.env.states["muscle"][0, idx_act, :]
                    except Exception:
                        a_now = a_des
                    payload_act = _activation_to_excitation(a_des, a_now, dt, self.p.tau_up, self.p.tau_down)
                else:
                    payload_act = a_des

            return {
                "tau_des": tau_des, "R": Rm, "Fmax": Fmax_v, "F_des": F_des, "act": payload_act,
                "q": q, "qd": qd, "x": x, "xd": xd,
                "xref_tuple": (Xref[-1], Xdref[-1], Xddref[-1]),
                "eta": 1.0,
                "diag": {
                    "sminJ": float(sminJ), "lamJ": float(lamJ),
                    "lam_sing": float(lam_sing), "lam_eff": float(lam_eff),
                    "alpha_J": float(alpha_J),
                    "kp_ns": float(kp_eff),
                    "comp_passive": bool(self.p.compensate_passive),
                    "send_excitation": bool(self.p.send_excitation)
                }
            }

        # Branch 2: muscles ON  ---------------------------------------------------

        F_des, _ = solve_muscle_forces(tau_need, Rm, Fmax_v, self.p.allocator_scale, self.mp)

        # (C) small internal-force (co-contraction) to improve conditioning
        if self.p.enable_internal_force:
            a0 = np.full(F_des.shape[0], self.p.cocon_a0)
            af0 = active_force_from_activation(a0, lenvel, self.env.muscle)
            F_bias = Fmax_v * (af0 + flpe)                       # keep passive bias
            F_des = apply_internal_force_regulation(
                -Rm, F_des, F_bias, Fmax_v,
                eps=self.p.eps,
                linesearch_eps=self.p.linesearch_eps,
                linesearch_safety=self.p.linesearch_safety,
                scale=float(np.clip(self.p.internal_force_scale, 0.0, 1.0))
            )

        a_des = force_to_activation_bisect(
            F_des, lenvel, self.env.muscle, flpe, Fmax_v, iters=self.p.bisect_iters
        )

        # (E) lower activation floor if requested
        if self.p.min_activation_override is not None:
            a_min = float(self.p.min_activation_override)
        else:
            a_min = float(getattr(self.env.muscle, "min_activation", 0.0))
        a_des = np.clip(a_des, a_min, 1.0)

        # (B) excitation prediction if the plant reads "u"
        if self.p.send_excitation:
            try:
                idx_act = names.index("activation")
                a_now = self.env.states["muscle"][0, idx_act, :]
            except Exception:
                a_now = a_des
            act_field = _activation_to_excitation(a_des, a_now, dt, self.p.tau_up, self.p.tau_down)
        else:
            act_field = a_des

        af_now = active_force_from_activation(a_des, lenvel, self.env.muscle)
        F_pred = Fmax_v * af_now
        F_corr = saturation_repair_tau(-Rm, F_pred, a_des, a_min, 1.0,
                                       Fmax_v, tau_des=tau_des)
        if np.any(np.abs(F_corr - F_pred) > 1e-9):
            a_des = force_to_activation_bisect(F_corr, lenvel, self.env.muscle, flpe, Fmax_v,
                                               iters=max(4, self.p.bisect_iters - 4))
            if self.p.send_excitation:
                try:
                    idx_act = names.index("activation")
                    a_now = self.env.states["muscle"][0, idx_act, :]
                except Exception:
                    a_now = a_des
                act_field = _activation_to_excitation(a_des, a_now, dt, self.p.tau_up, self.p.tau_down)
            else:
                act_field = a_des

        return {
            "tau_des": tau_des, "R": Rm, "Fmax": Fmax_v, "F_des": F_des, "act": act_field,
            "q": q, "qd": qd, "x": x, "xd": xd,
            "xref_tuple": (Xref[-1], Xdref[-1], Xddref[-1]), "eta": 1.0,
            "diag": {
                "sminJ": float(sminJ), "lamJ": float(lamJ),
                "lam_sing": float(lam_sing), "lam_eff": float(lam_eff),
                "alpha_J": float(alpha_J),
                "kp_ns": float(kp_eff),
                "comp_passive": bool(self.p.compensate_passive),
                "send_excitation": bool(self.p.send_excitation)
            }
        }

