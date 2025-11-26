#!/usr/bin/env python3
# controller/mpc_rl_hybrid_torch.py
# -*- coding: utf-8 -*-
"""
Torch version of mpc_rl_hybrid.py (MPC + RL a-policy hybrid).

- Works with EnvironmentTorch + RigidTendonArm26 + Hill muscles.
- Teacher: NonlinearMPCControllerTorch (τ from NMPC with muscles).
- Student: behavior-cloned a-policy RL (activations) from hybrid_bc_a_torch.RLPolicy.
- Blends torques τ_rl and τ_mpc using a confidence-based β(c_rl, c_mpc).
- Final τ_des is always passed through the robust τ→F→a muscle stack
  (solve_muscle_forces + force_to_activation_bisect + saturation_repair_tau).

Obs for the RL a-policy is the same as in the NumPy version:
    Obs_a = [ q(2) qd(2) x(2) xd(2) e(2) e_dot(2) xdd_d(2)
              l_norm(m) v_norm(m) R(2*m) a_prev(m) ]
          → 14 + 5m dims (m = n_muscles)

All heavy computations are torch-native; DAgger dataset recording is NumPy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from controller.nmpc_task_torch import NonlinearMPCControllerTorch, NMPCParams
from controller.hybrid_bc_a_torch import RLPolicy, RLPolicyParams

from utils.muscle_guard_torch import MuscleGuardParams, solve_muscle_forces
from muscles.muscle_tools_torch import (
    get_Fmax_vec,
    active_force_from_activation,
    force_to_activation_bisect,
    saturation_repair_tau,
)


# ---------------------------------------------------------------------
# DAgger recorder (NumPy, unchanged logic)
# ---------------------------------------------------------------------


class OnlineDatasetRecorder:
    def __init__(self, save_path: str = "data/random_reach_a_ds.npz", flush_every: int = 1000):
        self.save_path, self.flush_every = save_path, int(flush_every)
        self._O, self._A, self._since = [], [], 0
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)

    def add(self, obs: np.ndarray, act_teacher: np.ndarray) -> None:
        self._O.append(np.asarray(obs, np.float32))
        self._A.append(np.asarray(act_teacher, np.float32))
        self._since += 1
        if self._since >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._O:
            return
        O_new = np.asarray(self._O, np.float32)
        A_new = np.asarray(self._A, np.float32)
        self._O.clear()
        self._A.clear()
        self._since = 0

        if os.path.exists(self.save_path):
            data = np.load(self.save_path)
            O_all = np.concatenate([data["O"], O_new], axis=0)
            A_all = np.concatenate([data["A"], A_new], axis=0)
        else:
            O_all, A_all = O_new, A_new

        mean = O_all.mean(axis=0).astype(np.float32)
        std = np.clip(O_all.std(axis=0), 1e-6, None).astype(np.float32)
        np.savez_compressed(self.save_path, O=O_all, A=A_all, mean=mean, std=std)
        print(
            f"[DAgger] appended {len(O_new)} → O={O_all.shape} A={A_all.shape} @ {self.save_path}"
        )

    def close(self) -> None:
        self.flush()


# ---------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------


@dataclass
class HybridParams:
    z_ref: float = 1.6
    w_rl: float = 1.0
    w_mpc: float = 1.0
    beta_min: float = 0.0
    beta_max: float = 1.0
    sigma_good: float = 0.15
    bound_eps: float = 0.015
    sat_penalty: float = 0.25
    dagger_enabled: bool = True
    dagger_beta_thresh: float = 0.35
    dagger_path: str = "data/random_reach_a_ds.npz"
    dagger_flush_every: int = 1000
    print_every: int = 50
    name: str = "Hybrid(MPC+RL_a Torch)"


# ---------------------------------------------------------------------
# Hybrid controller (Torch env)
# ---------------------------------------------------------------------


class MPC_RL_ControllerHybridTorch:
    """
    Torch env controller that blends MPC (teacher) and RL a-policy (student).

    - Teacher: NonlinearMPCControllerTorch (τ_tch).
    - Student: RL a-policy (a_rl) → τ_rl via active forces + FL-PE.
    - Blend:
        β = β(c_rl, c_mpc)
        τ_des = (1 - β) τ_rl + β τ_tch

    - Final τ_des is mapped to muscle activations a_cmd via the same
      robust τ→F→a pipeline as other controllers.

    Public API:
      - compute(x_d, xd_d, xdd_d) -> diag dict
      - qref property for logger compatibility
      - close() to flush DAgger recorder
    """

    def __init__(
        self,
        env,
        arm,
        teacher_params: NMPCParams,
        policy: RLPolicy,
        hparams: HybridParams,
    ):
        self.env = env
        self.arm = arm
        self.hp = hparams
        self.pi = policy

        # Teacher NMPC (muscle-based, no excitation output)
        fields = getattr(NMPCParams, "__dataclass_fields__", {})
        d = dict(teacher_params.__dict__)
        #d["use_muscles"] = True
        if "send_excitation" in fields:
            d["send_excitation"] = False
        self.teacher = NonlinearMPCControllerTorch(env, arm, NMPCParams(**d))

        # DAgger recorder
        self.rec: Optional[OnlineDatasetRecorder] = (
            OnlineDatasetRecorder(self.hp.dagger_path, self.hp.dagger_flush_every)
            if self.hp.dagger_enabled
            else None
        )

        self._mp = MuscleGuardParams()
        self._step = 0
        self._beta_sum = 0.0
        self._qref: Optional[Tensor] = None

    # ---------------- properties & reset ----------------

    @property
    def qref(self):
        return getattr(self.teacher, "qref", self._qref)

    def reset(self, q0: Tensor) -> None:
        self._qref = None
        self._step = 0
        self._beta_sum = 0.0
        if hasattr(self.teacher, "reset"):
            self.teacher.reset(q0)

    # ---------------- helpers ----------------

    def _geometry(self):
        g = self.env.states["geometry"]  # (B, 2 + dof, M) or (2 + dof, M)
        if g.ndim == 2:
            g = g.unsqueeze(0)
        lenvel = g[:, :2, :]  # (B, 2, M)
        R = g[0, 2 : 2 + self.env.skeleton.dof, :]  # (dof, M)
        return lenvel, R

    def _flpe(self, m: int) -> Tensor:
        geom = self.env.states["geometry"]
        if geom.ndim == 2:
            device = geom.device
            dtype = geom.dtype
        else:
            device = geom.device
            dtype = geom.dtype
        names = list(getattr(self.env.muscle, "state_name", []))
        if "force-length PE" in names:
            idx = names.index("force-length PE")
            return self.env.states["muscle"][0, idx, :].to(device=device, dtype=dtype)
        return torch.zeros(m, device=device, dtype=dtype)

    def _build_obs14_torch(self, x_d: Any, xd_d: Any, xdd_d: Any) -> Tensor:
        """
        14-D task+state features:
          [q(2), qd(2), x(2), xd(2), e(2), e_dot(2), xdd_d(2)]
        """
        joint = self.env.states["joint"]
        cart = self.env.states["cartesian"]
        if joint.ndim == 1:
            joint = joint.unsqueeze(0)
        if cart.ndim == 1:
            cart = cart.unsqueeze(0)

        n = self.env.skeleton.dof
        q = joint[0, :n]
        qd = joint[0, n:]
        x = cart[0, :2]
        xd = cart[0, 2:4]

        device = joint.device
        dtype = joint.dtype

        x_d_t = torch.as_tensor(x_d, device=device, dtype=dtype).view(-1)[:2]
        xd_d_t = torch.as_tensor(xd_d, device=device, dtype=dtype).view(-1)[:2]
        xdd_d_t = torch.as_tensor(xdd_d, device=device, dtype=dtype).view(-1)[:2]

        e = x_d_t - x
        e_dot = xd_d_t - xd

        base = torch.cat([q, qd, x, xd, e, e_dot, xdd_d_t], dim=0)
        return base.to(torch.float32)

    def _build_obs_a_torch(self, x_d: Any, xd_d: Any, xdd_d: Any) -> Tensor:
        """
        14 + 5m features (same as NumPy a-policy):
          [ q(2) qd(2) x(2) xd(2) e(2) e_dot(2) xdd_d(2)
            l_norm(m) v_norm(m) R(2*m) a_prev(m) ]
        """
        base = self._build_obs14_torch(x_d, xd_d, xdd_d)  # (14,)

        geom = self.env.states["geometry"]
        if geom.ndim == 2:
            geom = geom.unsqueeze(0)
        geom0 = geom[0]  # (2 + dof, M)
        lenvel = geom0[:2, :]  # (2, M)
        n = self.env.skeleton.dof
        R = geom0[2 : 2 + n, :]  # (dof, M)

        names = list(getattr(self.env.muscle, "state_name", []))
        if "activation" in names:
            idx_act = names.index("activation")
            a_prev = self.env.states["muscle"][0, idx_act, :]  # (M,)
        else:
            M = R.shape[1]
            a_prev = torch.zeros(M, device=geom0.device, dtype=geom0.dtype)

        extra = torch.cat(
            [lenvel.reshape(-1), R.reshape(-1), a_prev.reshape(-1)],
            dim=0,
        ).to(base.dtype)

        return torch.cat([base, extra], dim=0)

    def _conf_rl(self, obs_a_t: Tensor, a_rl_t: Tensor) -> float:
        """
        Confidence in RL a-policy: combine feature z-score and saturation.
        Mirrors NumPy logic but stays in torch.
        """
        # Use RL policy's whitening stats
        mean = self.pi.mean  # (obs_dim,)
        std = self.pi.std.clamp_min(1e-6)

        obs = obs_a_t.to(device=mean.device, dtype=mean.dtype).view_as(mean)
        z = (obs - mean) / std
        z_norm = torch.sqrt(torch.mean(z * z)) / max(self.hp.z_ref, 1e-6)
        c_z = 1.0 / (1.0 + z_norm)

        a = a_rl_t.to(device=mean.device, dtype=mean.dtype).view(-1)
        a_min = float(getattr(self.env.muscle, "min_activation", 0.0))
        eps = self.hp.bound_eps
        sat_lo = torch.mean((a <= (a_min + eps)).to(torch.float32))
        sat_hi = torch.mean((a >= (1.0 - eps)).to(torch.float32))
        c_sat = 1.0 - self.hp.sat_penalty * 0.5 * (sat_lo + sat_hi)

        c = c_z * c_sat
        c = torch.clamp(c, 0.0, 1.0)
        return float(c.item())

    def _conf_mpc(self, diag_t: Dict[str, Any]) -> float:
        sminJ = diag_t.get("diag", {}).get("sminJ", None)
        if sminJ is None:
            return 1.0
        try:
            smin_val = float(sminJ)
        except Exception:
            smin_val = float(getattr(sminJ, "item", lambda: 0.0)())
        return float(
            np.clip(smin_val / max(self.hp.sigma_good, 1e-6), 0.0, 1.0)
        )

    def _beta(self, c_rl: float, c_mpc: float) -> float:
        num = self.hp.w_mpc * (1.0 - c_rl) * c_mpc
        den = num + self.hp.w_rl * c_rl + 1e-8
        return float(
            np.clip(num / den, self.hp.beta_min, self.hp.beta_max)
        )

    # ---------------- main API ----------------

    def compute(self, x_d: Any, xd_d: Any, xdd_d: Any) -> Dict[str, Any]:
        self._step += 1

        # Build a-policy observation (torch)
        obs_a_t = self._build_obs_a_torch(x_d, xd_d, xdd_d)  # (obs_dim,)

        # Geometry + muscle constants
        lenvel, R = self._geometry()
        m = int(R.shape[1])
        Fmax = get_Fmax_vec(
            self.env, m, device=R.device, dtype=R.dtype
        )  # (m,)
        flpe = self._flpe(m)  # (m,)

        # -------- RL branch: a_rl → τ_rl --------
        a_rl = self.pi.act_torch(obs_a_t)  # (m,)
        a_rl = a_rl.to(device=R.device, dtype=R.dtype).view(-1)
        a_min = float(getattr(self.env.muscle, "min_activation", 0.0))
        a_rl = torch.clamp(a_rl, min=a_min, max=1.0)

        af_rl = active_force_from_activation(a_rl, lenvel, self.env.muscle)  # (m,)
        F_rl = Fmax * (af_rl + flpe)  # (m,)
        tau_rl = -(R @ F_rl)  # (dof,)

        # -------- Teacher branch --------
        diag_t = self.teacher.compute(x_d, xd_d, xdd_d)

        a_tch_raw = diag_t.get("act", None)
        a_tch: Optional[Tensor]
        if isinstance(a_tch_raw, torch.Tensor):
            if a_tch_raw.ndim > 1:
                a_tch_raw = a_tch_raw[0]
            a_tch = a_tch_raw.to(device=R.device, dtype=R.dtype).view(-1)
        elif a_tch_raw is not None:
            a_tch = torch.as_tensor(a_tch_raw, device=R.device, dtype=R.dtype).view(-1)
        else:
            a_tch = None

        tau_tch: Tensor
        if (
            "R" in diag_t
            and "F_des" in diag_t
            and diag_t["R"] is not None
            and diag_t["F_des"] is not None
        ):
            R_t = diag_t["R"]
            if not isinstance(R_t, torch.Tensor):
                R_t = torch.as_tensor(R_t, device=R.device, dtype=R.dtype)
            else:
                R_t = R_t.to(device=R.device, dtype=R.dtype)

            F_des_t = diag_t["F_des"]
            if not isinstance(F_des_t, torch.Tensor):
                F_des_t = torch.as_tensor(
                    F_des_t, device=R.device, dtype=R.dtype
                ).view(-1)
            else:
                F_des_t = F_des_t.to(device=R.device, dtype=R.dtype).view(-1)

            tau_tch = -(R_t @ F_des_t)
        elif "tau_des" in diag_t:
            tau_raw = diag_t["tau_des"]
            if isinstance(tau_raw, torch.Tensor):
                tau_tch = tau_raw.to(device=R.device, dtype=R.dtype).view(-1)
            else:
                tau_tch = torch.as_tensor(
                    tau_raw, device=R.device, dtype=R.dtype
                ).view(-1)
        else:
            if a_tch is None:
                raise RuntimeError(
                    "Teacher diag lacks 'act' and 'tau_des'; cannot compute tau_tch."
                )
            af_t = active_force_from_activation(a_tch, lenvel, self.env.muscle)
            F_t = Fmax * (af_t + flpe)
            tau_tch = -(R @ F_t)

        # -------- Confidences & torque blend --------
        c_rl = self._conf_rl(obs_a_t, a_rl)
        c_mpc = self._conf_mpc(diag_t)
        beta = self._beta(c_rl, c_mpc)

        tau_des = (1.0 - beta) * tau_rl + beta * tau_tch  # (dof,)

        # -------- Robust τ → F → a pipeline --------
        F_des, _ = solve_muscle_forces(tau_des, R, Fmax, 1.0, self._mp)  # (m,)
        a_cmd = force_to_activation_bisect(
            F_des, lenvel, self.env.muscle, flpe, Fmax, iters=18
        )  # (m,)
        af_cmd = active_force_from_activation(a_cmd, lenvel, self.env.muscle)  # (m,)
        F_pred = Fmax * (af_cmd + flpe)  # (m,)

        F_fixed = saturation_repair_tau(
            -R,
            F_pred,
            a_cmd,
            self.env.muscle.min_activation,
            1.0,
            Fmax,
            tau_des=tau_des,
        )
        if torch.any(torch.abs(F_fixed - F_pred) > 1e-9):
            a_cmd = force_to_activation_bisect(
                F_fixed, lenvel, self.env.muscle, flpe, Fmax, iters=12
            )

        # Diagnostic τ from final activations
        af_final = active_force_from_activation(a_cmd, lenvel, self.env.muscle)
        F_final = Fmax * (af_final + flpe)
        tau_from_act = -(R @ F_final)

        # -------- DAgger: log (obs_a, a_teacher) when teacher dominates --------
        if (
            self.rec is not None
            and beta <= self.hp.dagger_beta_thresh
            and a_tch is not None
        ):
            obs_np = obs_a_t.detach().cpu().numpy().astype(np.float32)
            a_tch_np = a_tch.detach().cpu().numpy().astype(np.float32)
            self.rec.add(obs_np, a_tch_np)

        # -------- prints --------
        self._beta_sum += beta
        if (self._step % max(1, self.hp.print_every)) == 0:
            avg_beta = self._beta_sum / self._step
            recon_err = float(
                torch.linalg.norm(tau_from_act - tau_des).item()
            )
            print(
                f"[{self.hp.name}] step {self._step:5d} | beta={beta:5.2f} | "
                f"avg_beta={100*avg_beta:4.1f}% | c_rl={c_rl:4.2f} c_mpc={c_mpc:4.2f} | "
                f"|τ| rl={float(torch.linalg.norm(tau_rl).item()):.3f} "
                f"teach={float(torch.linalg.norm(tau_tch).item()):.3f} "
                f"mix={float(torch.linalg.norm(tau_des).item()):.3f} | "
                f"recon_err={recon_err: .3e}"
            )

        # -------- logger payload --------
        joint = self.env.states["joint"]
        cart = self.env.states["cartesian"]
        if joint.ndim == 1:
            joint = joint.unsqueeze(0)
        if cart.ndim == 1:
            cart = cart.unsqueeze(0)
        n = self.env.skeleton.dof
        q = joint[0, :n]
        qd = joint[0, n:]
        x = cart[0, :2]
        xd = cart[0, 2:4]

        device = joint.device
        dtype = joint.dtype
        x_d_t = torch.as_tensor(x_d, device=device, dtype=dtype).view(-1)[:2]
        xd_d_t = torch.as_tensor(xd_d, device=device, dtype=dtype).view(-1)[:2]
        xdd_d_t = torch.as_tensor(xdd_d, device=device, dtype=dtype).view(-1)[:2]

        return {
            "tau_des": tau_des,
            "R": R,
            "Fmax": Fmax,
            "F_des": F_des,
            "act": a_cmd,
            "q": q,
            "qd": qd,
            "x": x,
            "xd": xd,
            "xref_tuple": (x_d_t, xd_d_t, xdd_d_t),
            "eta": 1.0,
            "diag": {
                "beta": float(beta),
                "c_rl": float(c_rl),
                "c_mpc": float(c_mpc),
                "tau_rl": tau_rl,
                "tau_teacher": tau_tch,
                "tau_from_act": tau_from_act,
            },
        }

    # ---------------- teardown ----------------

    def close(self) -> None:
        if self.rec is not None:
            self.rec.close()


# ---------------------------------------------------------------------
# Smoke test: MPC + RL a-policy hybrid on Torch plant/env
# ---------------------------------------------------------------------


def _smoke_test_mpc_rl_hybrid_torch():
    """
    Smoke test for MPC_RL_ControllerHybridTorch on RigidTendonArm26 + Hill muscles.
    Single min-jerk reach, same setup as other Torch controllers.
    """
    import torch as _torch
    from model_lib.environment_torch import Environment as EnvTorch
    from model_lib.muscles_torch import RigidTendonHillMuscle
    from model_lib.effector_torch import RigidTendonArm26
    from trajectory.minjerk_torch import (
        MinJerkLinearTrajectoryTorch,
        MinJerkParams,
    )
    from sim.simulator_torch import TargetReachSimulatorTorch
    from config import (
        PlantConfig,
        ControlToggles,
        ControlGains,
        Numerics,
        InternalForceConfig,
        TrajectoryConfig,
    )
    from controller.nmpc_task_torch import NMPCParams

    print("\n[MPC+RL_a Torch] smoke test starting ...")
    _torch.set_default_dtype(_torch.float64)
    _torch.set_printoptions(precision=6, sci_mode=False)

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

    # ---------- configs ----------
    pc = PlantConfig()
    toggles = ControlToggles()
    gains = ControlGains()
    num = Numerics()
    ifc = InternalForceConfig()
    tc = TrajectoryConfig()

    # ---------- build effector + env ----------
    muscle = RigidTendonHillMuscle(
        min_activation=0.02,
        device=device,
        dtype=_torch.get_default_dtype(),
    )

    arm = RigidTendonArm26(
        muscle=muscle,
        timestep=pc.timestep,
        damping=pc.damping,
        n_ministeps=pc.n_ministeps,
        integration_method=pc.integration_method,
        device=device,
        dtype=_torch.get_default_dtype(),
    )

    env = EnvTorch(
        effector=arm,
        max_ep_duration=pc.max_ep_duration,
        action_noise=0.0,
        obs_noise=0.0,
        action_frame_stacking=1,
        proprioception_delay=arm.dt,
        vision_delay=arm.dt,
        name="MPC_RL_Hybrid_EnvTorch",
    )

    # ---------- initial joint state (B=1) ----------
    q0 = _torch.deg2rad(
        _torch.tensor(pc.q0_deg, dtype=_torch.get_default_dtype(), device=device)
    )  # (2,)
    qd0 = _torch.tensor(pc.qd0, dtype=_torch.get_default_dtype(), device=device)  # (2,)
    joint0 = _torch.cat([q0, qd0]).unsqueeze(0)  # (1, 4)

    env.reset(
        options={
            "joint_state": joint0,
            "deterministic": True,
        }
    )

    # ---------- simple min-jerk trajectory ----------
    fingertip0 = env.states["fingertip"][0, :2]  # (2,)
    center = fingertip0
    radius = 0.10
    target = center + _torch.tensor([radius, 0.0], dtype=center.dtype, device=device)

    waypoints = _torch.stack([center, target], dim=0)  # (2, 2)

    mj_params = MinJerkParams(
        Vmax=tc.Vmax,
        Amax=tc.Amax,
        Jmax=tc.Jmax,
        gamma=tc.gamma_time_scale,
    )
    traj = MinJerkLinearTrajectoryTorch(waypoints, mj_params)

    # ---------- RL a-policy ----------
    geom = env.states["geometry"]
    if geom.ndim == 2:
        geom = geom.unsqueeze(0)
    M = int(geom.shape[-1])
    obs_dim = 14 + 5 * M  # as in NumPy a-policy
    pi_params = RLPolicyParams(
        obs_dim=obs_dim,
        act_dim=M,
        device=str(device),
    )
    pi = RLPolicy(pi_params)

    ckpt_path = "models/random_reach_bc_a.pt"
    try:
        pi.load(ckpt_path)
        print(f"[BC-a Torch] loaded policy from {ckpt_path}")
    except FileNotFoundError:
        print(
            f"[BC-a Torch] WARNING: no checkpoint at {ckpt_path}, using random policy."
        )
    except Exception as e:
        print(
            f"[BC-a Torch] WARNING: failed to load checkpoint ({e}), using random policy."
        )

    # ---------- teacher NMPC params ----------
    teacher_params = NMPCParams()

    # ---------- hybrid controller ----------
    hp = HybridParams(name="Hybrid(MPC+RL_a Torch)")
    ctrl = MPC_RL_ControllerHybridTorch(env, arm, teacher_params, pi, hp)

    # ---------- simulate ----------
    steps = int(pc.max_ep_duration / arm.dt)

    try:
        sim = TargetReachSimulatorTorch(env, arm, ctrl, traj, steps)
        logs = sim.run()

        k, tvec = logs.time(arm.dt)
        x_log = logs.x_log[:k]  # (T, >=2)

        print("  fingertip (before):", center.detach().cpu().numpy())
        print("  target x_d:        ", target.detach().cpu().numpy())

        for idx in [0, 1, 2, 4, 9, 19, 29]:
            if idx < k:
                x = x_log[idx, :2]
                print(f"    step {idx:3d}: x = {x.detach().cpu().numpy()}")

        x_final = x_log[k - 1, :2]
        err_final = _torch.linalg.norm(x_final - target)
        print("  final fingertip:   ", x_final.detach().cpu().numpy())
        print(f"  final |x - x_d|:    {float(err_final):.6f} m")

    except Exception as e:
        print(f"[MPC+RL_a Torch] simulator failed: {e}")
        print("[MPC+RL_a Torch] running manual loop...")
        for step in range(min(50, steps)):
            t = step * arm.dt
            t_tensor = _torch.tensor(
                [t], device=traj.device, dtype=traj.dtype
            )
            x_d, xd_d, xdd_d = traj.sample(t_tensor)
            out = ctrl.compute(x_d, xd_d, xdd_d)
            if step in (0, 1, 2, 5, 10, 20, 30, 40, 49):
                x_cur = out["x"]
                act = out["act"]
                if isinstance(x_cur, _torch.Tensor):
                    x_cur_np = x_cur.detach().cpu().numpy()
                else:
                    x_cur_np = np.asarray(x_cur)
                if isinstance(act, _torch.Tensor):
                    act_mean = float(act.mean().item())
                else:
                    act_mean = float(np.mean(act))
                print(
                    f"  step {step:3d}: x = {x_cur_np}, avg_act = {act_mean:.3f}"
                )

    print("[MPC+RL_a Torch] smoke test complete ✓")


if __name__ == "__main__":
    _smoke_test_mpc_rl_hybrid_torch()
