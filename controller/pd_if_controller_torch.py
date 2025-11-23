# controller/pd_if_controller_torch.py
# -----------------------------------
# Pure-Torch PD + Internal Force (PD/IF) controller.
#
# This is the Torch counterpart of controller/pd_if_controller.py.
# It expects a Torch-based environment with:
#   - env.skeleton: TwoDofArm (or similar) from model_lib.skeleton_torch
#   - env.states: dict with keys "joint", "cartesian", "geometry", "muscle"
#   - env.muscle: muscle model used by utils.muscle_tools_torch
#
# All math is done in torch.Tensor (CPU or CUDA), fully differentiable.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from utils.math_utils_torch import matrix_sqrt_spd, matrix_isqrt_spd
from utils.kinematics_guard_torch import (
    KinGuardParams,
    adaptive_dls_pinv,
    scale_task_by_J,
    add_nullspace_manip,
)
from utils.dynamics_guard_torch import DynGuardParams, op_space_guard_and_gate
from utils.muscle_guard_torch import MuscleGuardParams, solve_muscle_forces
from utils.telemetry_torch import pack_diag, merge_diag
from muscles.muscle_tools_torch import (
    get_Fmax_vec,
    force_to_activation_bisect,
    active_force_from_activation,
    saturation_repair_tau,
    apply_internal_force_regulation,
)
from model_lib.skeleton_torch import (
    geometricJacobian_cached,
    geometricJacobianDot_cached,
    inertiaMatrixCOM_cached,
    centrifugalCoriolisCOM_cached,
    gravityCOM_cached,
)


Tensor = torch.Tensor


@dataclass
class PDIFParamsTorch:
    """
    Torch version of PDIFParams.

    Kp_x, Kff_x, Kp_q, Kd_q can be Python lists, NumPy arrays, or Tensors.
    They are converted to Tensors on the fly on the correct device/dtype.
    """

    Kp_x: Any
    Kff_x: Any
    Kp_q: Any
    Kd_q: Any
    eps: float
    lam_os_smin_target: float
    lam_os_max: float
    sigma_thresh: float
    gate_pow: float
    enable_internal_force: bool
    enable_inertia_comp: bool
    enable_gravity_comp: bool
    enable_velocity_comp: bool
    enable_joint_damping: bool
    cocon_a0: float
    bisect_iters: int
    linesearch_eps: float
    linesearch_safety: float


class PDIFControllerTorch:
    """
    Pure-Torch PD + Internal Force controller.

    API is intentionally close to the NumPy version:

        ctrl = PDIFControllerTorch(env_torch, arm_torch, params)
        out = ctrl.compute(x_d, xd_d, xdd_d)

    with:
      - x_d, xd_d, xdd_d: 1D tensors of length 2 (task-space x,y).
      - env_torch: environment exposing states and skeleton/muscle fields.
      - arm_torch: object with dt (time step) and damping (joint viscous).
    """

    def __init__(self, env, arm, params: PDIFParamsTorch):
        self.env = env
        self.arm = arm
        self.p = params
        self.qref: Optional[Tensor] = None

        # guard params (you can expose/tune these)
        self.kp = KinGuardParams()
        self.dp = DynGuardParams(
            eps=self.p.eps,
            lam_os_max=self.p.lam_os_max,
            gate_pow=self.p.gate_pow,
            sigma_thresh_S=max(self.p.sigma_thresh, 1e-9),
        )
        self.mp = MuscleGuardParams()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_tensor_like(x: Any, like: Tensor) -> Tensor:
        """Convert x to a tensor on like.device / like.dtype."""
        return torch.as_tensor(x, dtype=like.dtype, device=like.device)

    def reset(self, q0: Tensor) -> None:
        """
        Reset the reference joint configuration.

        q0 is expected to be a 1D tensor of shape (n,) or (n,1).
        """
        q0 = torch.as_tensor(q0)
        self.qref = q0.detach().clone().reshape(-1)

    # ------------------------------------------------------------------ #
    # Dynamics
    # ------------------------------------------------------------------ #
    def _compute_dynamics(self, q: Tensor, qd: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute joint-space inertia matrix M(q) and bias term h(q, qd).

        h(q, qd) = C(q, qd) qd + g(q) + D qd
        where D is viscous joint damping (if enabled).
        """
        p = self.p
        n = q.shape[0]
        dtype = q.dtype
        device = q.device

        # Inertia
        if p.enable_inertia_comp:
            M = inertiaMatrixCOM_cached(self.env.skeleton._robot, symbolic=False)
        else:
            M = torch.eye(n, dtype=dtype, device=device)

        # Gravity
        if p.enable_gravity_comp:
            g = gravityCOM_cached(
                self.env.skeleton._robot,
                self.env.skeleton._gravity_vec,
                symbolic=False,
            ).reshape(-1)
        else:
            g = torch.zeros_like(q)

        # Centrifugal/Coriolis
        if p.enable_velocity_comp:
            C_any = centrifugalCoriolisCOM_cached(
                self.env.skeleton._robot, symbolic=False
            )
        else:
            C_any = torch.zeros((n, n), dtype=dtype, device=device)

        # Joint damping
        if p.enable_joint_damping:
            D = torch.diag(
                torch.full((n,), float(self.arm.damping), dtype=dtype, device=device)
            )
        else:
            D = torch.zeros((n, n), dtype=dtype, device=device)

        # Bias h(q, qd)
        if C_any.dim() == 2:
            h = C_any @ qd + g + D @ qd
        else:
            h = C_any.reshape(-1) + g + D @ qd

        return M, h

    # ------------------------------------------------------------------ #
    # Main compute
    # ------------------------------------------------------------------ #
    def compute(self, x_d: Tensor, xd_d: Tensor, xdd_d: Tensor) -> Dict[str, Any]:
        """
        Compute muscle activations for a given task-space reference.

        Inputs x_d, xd_d, xdd_d must be 1D tensors (len=2) on the same
        device/dtype as the environment tensors.
        """
        # ---------- current state ----------
        joint = self.env.states["joint"][0]
        q, qd = joint[:2], joint[2:]
        cart = self.env.states["cartesian"][0]
        x, xd = cart[:2], cart[2:]

        q = q.reshape(-1)
        qd = qd.reshape(-1)
        x = x.reshape(-1)
        xd = xd.reshape(-1)

        # ensure robot state is up-to-date
        self.env.skeleton._set_state(q, qd)

        # ---------- Jacobians ----------
        J = geometricJacobian_cached(self.env.skeleton._robot, symbolic=False)
        J_xy = J[0:2, :]
        Jdot = geometricJacobianDot_cached(self.env.skeleton._robot, symbolic=False)
        Jdot_xy = Jdot[0:2, :]
        n = q.shape[0]

        # ---------- gains as tensors ----------
        # task-space gains
        Kp_x = self._to_tensor_like(self.p.Kp_x, x)
        Kff_x = self._to_tensor_like(self.p.Kff_x, x)
        # joint-space gains
        Kp_q = self._to_tensor_like(self.p.Kp_q, q)
        Kd_q = self._to_tensor_like(self.p.Kd_q, q)

        # ensure qref is initialized
        if self.qref is None:
            self.qref = q.detach().clone()

        # ===== [1a] adaptive DLS on J; [1b] kinematic scaling
        J_pinv_dls, sminJ, lamJ = adaptive_dls_pinv(J_xy, n, self.kp)
        xd_d, xdd_d, alpha_J = scale_task_by_J(xd_d, xdd_d, sminJ, self.kp)

        # reference update
        qd_des = J_pinv_dls @ xd_d
        self.qref = self.qref + qd_des * float(self.arm.dt)

        # --- dynamics terms
        M, h = self._compute_dynamics(q, qd)
        Minv = torch.linalg.inv(M)

        # ===== [2a]/[2b]/[2c] op-space guard + gate (returns Λ, scaled cmds, and gate)
        S = J_xy @ Minv @ J_xy.transpose(-1, -2)
        Lambda, lam_os, eta, eta2, xd_d, xdd_d, dyn_diag = op_space_guard_and_gate(
            S, xd_d, xdd_d, self.dp
        )

        # --- operational-space control law
        mu = Lambda @ (J_xy @ Minv @ h - Jdot_xy @ qd)

        # textbook tracking error: e_x = x_d - x, ė_x = ẋ_d - ẋ
        e_x = x_d - x
        ed_x = xd_d - xd

        Kp_mat = torch.diag(Kp_x)
        Lam_sqrt = matrix_sqrt_spd(Lambda)
        Lam_isqrt = matrix_isqrt_spd(Lambda)
        Kv_mat = (
            2.0
            * Lam_sqrt
            @ matrix_sqrt_spd(Lam_isqrt @ Kp_mat @ Lam_isqrt)
            @ Lam_sqrt
        )

        xdd_r = (torch.diag(Kff_x) @ xdd_d) + (Kv_mat @ ed_x) + (Kp_mat @ e_x)
        F_task = Lambda @ xdd_r + mu
        tau_task = J_xy.transpose(-1, -2) @ F_task

        # --- nullspace policy
        J_dyn_pinv = Minv @ J_xy.transpose(-1, -2) @ Lambda
        N = torch.eye(n, dtype=q.dtype, device=q.device) - J_dyn_pinv @ J_xy

        qdd_post = Kp_q * (self.qref - q) - Kd_q * qd

        # ===== [1c] manipulability gradient in nullspace (faded by eta)
        qdd_post, k_manip_used = add_nullspace_manip(
            qdd_post, self.env, q, qd, J_xy, self.kp, eta
        )

        tau_post = M @ qdd_post + h
        # NOTE: keep the minus (back off nullspace near singularities with eta2)
        tau_des = tau_task - eta2 * (N.transpose(-1, -2) @ tau_post)

        # ---------- muscle geometry & forces ----------
        geom = self.env.states["geometry"]
        # len/vel channels: shape (1, 2, M) for unbatched
        lenvel = geom[:, :2, :]
        # R has shape (n, M)
        R = geom[:, 2 : 2 + self.env.skeleton.dof, :][0]
        Fmax_vec = get_Fmax_vec(self.env, R.shape[1], device=q.device, dtype=q.dtype)

        names = self.env.muscle.state_name
        idx_flpe = names.index("force-length PE")
        flpe = self.env.states["muscle"][0, idx_flpe, :]

        # ===== [3a]/[3b]/[3c] robust muscle solve (WLS/NNLS) with shared gate
        F_des, mus_diag = solve_muscle_forces(tau_des, R, Fmax_vec, eta, self.mp)

        # --- optional internal force regulation (scaled by eta2)
        if self.p.enable_internal_force:
            a0_vec = torch.full(
                (F_des.shape[0],),
                float(self.p.cocon_a0),
                dtype=q.dtype,
                device=q.device,
            )
            af = active_force_from_activation(a0_vec, lenvel, self.env.muscle)
            F_bias = Fmax_vec * (af + flpe)
            F_des = apply_internal_force_regulation(
                -R,
                F_des,
                F_bias,
                Fmax_vec,
                eps=self.p.eps,
                linesearch_eps=self.p.linesearch_eps,
                linesearch_safety=self.p.linesearch_safety,
                scale=eta2,
            )

        # --- activation back-projection & saturation repair
        a = force_to_activation_bisect(
            F_des, lenvel, self.env.muscle, flpe, Fmax_vec, iters=self.p.bisect_iters
        )

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

        if torch.any(torch.abs(F_corr - F_pred) > 1e-9):
            a = force_to_activation_bisect(
                F_corr,
                lenvel,
                self.env.muscle,
                flpe,
                Fmax_vec,
                iters=max(4, int(self.p.bisect_iters) - 4),
            )

        # ===== [5] diagnostics
        kin_diag = pack_diag(
            sminJ=sminJ, lamJ=lamJ, alpha_J=alpha_J, k_manip=k_manip_used
        )
        diag = merge_diag(
            kin_diag, dyn_diag, mus_diag, pack_diag(lam_os=lam_os, eta=eta, eta2=eta2)
        )

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
            "xref_tuple": (x_d, xd_d, xdd_d),
            "eta": eta2,
            "diag": diag,
        }


if __name__ == "__main__":
    # Very lightweight smoke test: just construct the controller and make
    # sure the module imports. Full closed-loop tests should be
    # done in your main simulation scripts.
    print("[pd_if_controller_torch] Module import OK (no full smoke test here).")
