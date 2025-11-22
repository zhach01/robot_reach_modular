#!/usr/bin/env python3
# controller/mpc_rl_hybrid.py
from __future__ import annotations
import os, numpy as np
from dataclasses import dataclass

from controller.nmpc_task import NonlinearMPCController, NMPCParams
from controller.hybrid_bc_a import RLPolicy, _build_obs_a, _build_obs14

from utils.muscle_guard import MuscleGuardParams, solve_muscle_forces
from muscles.muscle_tools import (
    get_Fmax_vec,
    active_force_from_activation,
    force_to_activation_bisect,
    saturation_repair_tau,
)

# --------- DAgger recorder ----------
class OnlineDatasetRecorder:
    def __init__(self, save_path="data/random_reach_a_ds.npz", flush_every=1000):
        self.save_path, self.flush_every = save_path, int(flush_every)
        self._O, self._A, self._since = [], [], 0
        d = os.path.dirname(save_path)
        if d: os.makedirs(d, exist_ok=True)
    def add(self, obs, act_teacher):
        self._O.append(np.asarray(obs, np.float32))
        self._A.append(np.asarray(act_teacher, np.float32))
        self._since += 1
        if self._since >= self.flush_every: self.flush()
    def flush(self):
        if not self._O: return
        O_new = np.asarray(self._O, np.float32); self._O.clear()
        A_new = np.asarray(self._A, np.float32); self._A.clear()
        self._since = 0
        if os.path.exists(self.save_path):
            data = np.load(self.save_path)
            O_all = np.concatenate([data["O"], O_new], 0)
            A_all = np.concatenate([data["A"], A_new], 0)
        else:
            O_all, A_all = O_new, A_new
        mean = O_all.mean(0).astype(np.float32)
        std  = np.clip(O_all.std(0), 1e-6, None).astype(np.float32)
        np.savez_compressed(self.save_path, O=O_all, A=A_all, mean=mean, std=std)
        print(f"[DAgger] appended {len(O_new)} → O={O_all.shape} A={A_all.shape} @ {self.save_path}")
    def close(self): self.flush()

# --------- params ----------
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
    name: str = "Hybrid(MPC+RL_a)"

# --------- controller ----------
class MPC_RL_ControllerHybrid:
    def __init__(self, env, arm, teacher_params: NMPCParams, policy: RLPolicy, hparams: HybridParams):
        self.env, self.arm, self.hp, self.pi = env, arm, hparams, policy

        fields = getattr(NMPCParams, "__dataclass_fields__", {})
        d = dict(teacher_params.__dict__); d["use_muscles"] = True
        if "send_excitation" in fields: d["send_excitation"] = False
        self.teacher = NonlinearMPCController(env, arm, NMPCParams(**d))

        self._mean = self.pi.mean.detach().cpu().numpy()
        self._std  = np.clip(self.pi.std.detach().cpu().numpy(), 1e-6, None)

        self.rec = OnlineDatasetRecorder(self.hp.dagger_path, self.hp.dagger_flush_every) if self.hp.dagger_enabled else None
        self._mp = MuscleGuardParams()
        self._step, self._beta_sum = 0, 0.0
        self._qref = None

    @property
    def qref(self): return getattr(self.teacher, "qref", self._qref)

    def reset(self, q0):
        self._qref = None; self._step = 0; self._beta_sum = 0.0
        if hasattr(self.teacher, "reset"): self.teacher.reset(q0)

    # --- helpers ---
    def _geometry(self):
        g = self.env.states["geometry"]
        if g.ndim == 2: g = g[None, ...]
        lenvel = g[:, :2, :]                             # (1,2,m)
        R = g[0, 2:2 + self.env.skeleton.dof, :]         # (DOF,m)
        return lenvel, R
    def _flpe(self, m: int):
        try:
            idx = self.env.muscle.state_name.index("force-length PE")
            return self.env.states["muscle"][0, idx, :].astype(np.float32)
        except Exception:
            return np.zeros(m, np.float32)
    def _conf_rl(self, obs_a, a_rl):
        z = (obs_a - self._mean) / self._std
        z_norm = float(np.sqrt(np.mean(z*z)) / max(self.hp.z_ref, 1e-6))
        c_z = 1.0 / (1.0 + z_norm)
        a_min = float(getattr(self.env.muscle, "min_activation", 0.0))
        sat_lo = (a_rl <= a_min + self.hp.bound_eps).mean()
        sat_hi = (a_rl >= 1.0 - self.hp.bound_eps).mean()
        c_sat = 1.0 - self.hp.sat_penalty * float(0.5*(sat_lo + sat_hi))
        return float(np.clip(c_z * c_sat, 0.0, 1.0))
    def _conf_mpc(self, diag_t):
        sminJ = diag_t.get("diag", {}).get("sminJ", None)
        if sminJ is None: return 1.0
        return float(np.clip(float(sminJ) / max(self.hp.sigma_good, 1e-6), 0.0, 1.0))
    def _beta(self, c_rl, c_mpc):
        num = self.hp.w_mpc * (1.0 - c_rl) * c_mpc
        den = num + self.hp.w_rl * c_rl + 1e-8
        return float(np.clip(num / den, self.hp.beta_min, self.hp.beta_max))

    # --- main API ---
    def compute(self, x_d, xd_d, xdd_d):
        self._step += 1
        obs_a = _build_obs_a(self.env, x_d, xd_d, xdd_d)

        lenvel, R = self._geometry()
        m = R.shape[1]
        Fmax = get_Fmax_vec(self.env, m).astype(np.float32)
        flpe = self._flpe(m)

        # RL activations -> τ estimate (include PE)
        a_rl = self.pi.act(obs_a).astype(np.float32)
        a_rl = np.clip(a_rl, self.env.muscle.min_activation, 1.0)
        af_rl = active_force_from_activation(a_rl, lenvel, self.env.muscle)   # (m,) in your codebase
        F_rl  = Fmax * (af_rl + flpe)
        tau_rl = - (R @ F_rl).astype(np.float32)

        # Teacher branch
        diag_t = self.teacher.compute(x_d, xd_d, xdd_d)
        a_tch  = np.asarray(diag_t["act"], np.float32).reshape(-1)

        # teacher torque preference
        if ("R" in diag_t) and ("F_des" in diag_t) and diag_t["R"] is not None and diag_t["F_des"] is not None:
            tau_tch = - (np.asarray(diag_t["R"], np.float32) @ np.asarray(diag_t["F_des"], np.float32).reshape(-1))
        elif "tau_des" in diag_t:
            tau_tch = np.asarray(diag_t["tau_des"], np.float32).reshape(-1)
        else:
            af_t = active_force_from_activation(a_tch, lenvel, self.env.muscle)
            F_t  = Fmax * (af_t + flpe)
            tau_tch = - (R @ F_t).astype(np.float32)

        # confidences and torque blend
        c_rl = self._conf_rl(obs_a, a_rl); c_mpc = self._conf_mpc(diag_t)
        beta = self._beta(c_rl, c_mpc)
        tau_des = (1.0 - beta) * tau_rl + beta * tau_tch          # <- τ_des used by logger

        # τ → F → a  (robust path; all 1-D vectors)
        F_des, _ = solve_muscle_forces(tau_des, R, Fmax, 1.0, self._mp)
        a_cmd = force_to_activation_bisect(F_des, lenvel, self.env.muscle, flpe, Fmax, iters=18)   # (m,)
        af_cmd = active_force_from_activation(a_cmd, lenvel, self.env.muscle)                     # (m,)
        F_pred = Fmax * (af_cmd + flpe)                                                           # (m,)

        F_fixed = saturation_repair_tau(-R, F_pred, a_cmd,
                                        self.env.muscle.min_activation, 1.0,
                                        Fmax, tau_des=tau_des)
        if np.any(np.abs(F_fixed - F_pred) > 1e-9):
            a_cmd = force_to_activation_bisect(F_fixed, lenvel, self.env.muscle, flpe, Fmax, iters=12)

        # diagnostic: τ from final activations
        af_final = active_force_from_activation(a_cmd, lenvel, self.env.muscle)
        F_final  = Fmax * (af_final + flpe)
        tau_from_act = - (R @ F_final).astype(np.float32)

        # DAgger
        if self.rec is not None and beta <= self.hp.dagger_beta_thresh:
            self.rec.add(obs_a, a_tch)

        # prints
        self._beta_sum += beta
        if (self._step % max(1, self.hp.print_every)) == 0:
            avg_beta = self._beta_sum / self._step
            recon_err = float(np.linalg.norm(tau_from_act - tau_des))
            print(f"[{self.hp.name}] step {self._step:5d} | beta={beta:5.2f} | avg_beta={100*avg_beta:4.1f}% "
                  f"| c_rl={c_rl:4.2f} c_mpc={c_mpc:4.2f} | |τ| rl={np.linalg.norm(tau_rl):.3f} "
                  f"teach={np.linalg.norm(tau_tch):.3f} mix={np.linalg.norm(tau_des):.3f} | recon_err={recon_err: .3e}")

        # logger payload
        base = _build_obs14(self.env, x_d, xd_d, xdd_d)
        q, qd, x, xd = base[0:2], base[2:4], base[4:6], base[6:8]

        return {
            "tau_des": tau_des,                      # (2,)
            "R": R, "Fmax": Fmax, "F_des": F_des,    # (DOF,m), (m,), (m,)
            "act": a_cmd.astype(np.float32),         # (m,)  <-- 1-D, not 0-D
            "q": q, "qd": qd, "x": x, "xd": xd,
            "xref_tuple": (
                np.asarray(x_d,  np.float32)[:2],
                np.asarray(xd_d, np.float32)[:2],
                np.asarray(xdd_d,np.float32)[:2],
            ),
            "eta": 1.0,
            "diag": {
                "beta": float(beta),
                "c_rl": float(c_rl),
                "c_mpc": float(c_mpc),
                "tau_rl": tau_rl,
                "tau_teacher": tau_tch,
                "tau_from_act": tau_from_act,
            }
        }

    def close(self):
        if self.rec is not None: self.rec.close()
