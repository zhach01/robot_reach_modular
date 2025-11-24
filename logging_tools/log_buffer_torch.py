# log_buffer_torch.py
# -*- coding: utf-8 -*-
"""
Torch version of LogBuffer.

- Same public API:
    LogBufferTorch(steps, n_muscles)
    .record(env, diag, tau_real, qref)
    .time(dt) -> (k, t_vec)

- Internally uses Torch tensors for all logs (CPU by default).
- Accepts NumPy or Torch arrays in `record`.
- Computes link positions from 2-DoF arm geometry (l1, l2) in Torch
  instead of calling forwardHTM_cached from the NumPy stack.

Batch-safe:
- If controller/env produce batched tensors with leading dim B, this logger
  will record ONLY THE FIRST SAMPLE (batch 0) into the per-step logs,
  to keep the same (steps, …) shape as the NumPy version.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
from torch import Tensor


class LogBufferTorch:
    def __init__(
        self,
        steps: int,
        n_muscles: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.steps = int(steps)
        self.n_muscles = int(n_muscles)
        self.k = 0

        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

        # state / reference logs
        self.x_log = torch.zeros((steps, 2), device=self.device, dtype=self.dtype)
        self.xd_log = torch.zeros((steps, 2), device=self.device, dtype=self.dtype)
        self.xref_log = torch.zeros((steps, 2), device=self.device, dtype=self.dtype)
        self.q_log = torch.zeros((steps, 2), device=self.device, dtype=self.dtype)
        self.qref_log = torch.zeros((steps, 2), device=self.device, dtype=self.dtype)

        # torque & residual logs
        self.tau_des_log = torch.zeros((steps, 2), device=self.device, dtype=self.dtype)
        self.tau_real_log = torch.zeros((steps, 2), device=self.device, dtype=self.dtype)
        self.res_log = torch.zeros((steps, 2), device=self.device, dtype=self.dtype)

        # muscle activation logs
        self.act_log = torch.zeros(
            (steps, n_muscles), device=self.device, dtype=self.dtype
        )
        self.act_norm = torch.zeros(steps, device=self.device, dtype=self.dtype)
        self.sat_min_pct = torch.zeros(steps, device=self.device, dtype=self.dtype)
        self.sat_max_pct = torch.zeros(steps, device=self.device, dtype=self.dtype)

        # conditioning logs
        self.condR_log = torch.zeros(steps, device=self.device, dtype=self.dtype)
        self.condM_log = torch.zeros(steps, device=self.device, dtype=self.dtype)

        # link positions: (steps, 4 points, 2 coordinates)
        # order: base, joint1, joint2, end-effector
        self.links_xy = torch.zeros(
            (steps, 4, 2), device=self.device, dtype=self.dtype
        )

    # ----------------- helpers -----------------

    def _to_1d(self, x: Any, expected_len: Optional[int] = None) -> Tensor:
        """
        Convert x to a 1D tensor on the log device/dtype.

        Accepts:
          - scalar / list / np array
          - 1D tensor (d,)
          - batched tensor (B, d)  -> logs the FIRST batch sample

        expected_len:
          If not None, we check that the final 1D vector has that length.
        """
        t = torch.as_tensor(x, device=self.device, dtype=self.dtype)

        # If batched, keep only the first sample for logging.
        # Example: (B,2) -> (2,)
        if t.ndim > 1:
            t = t[0]

        # Use reshape instead of view to support non-contiguous inputs
        t = t.reshape(-1)

        if expected_len is not None and t.numel() != expected_len:
            raise ValueError(
                f"_to_1d: expected length {expected_len}, got {t.numel()} "
                f"for tensor with shape {tuple(t.shape)}"
            )

        return t

    def _to_2d_vec(self, x: Any) -> Tensor:
        """
        Convert x to a 2D vector (length-2 1D tensor) on the log device/dtype.

        Accepts:
          - shape (2,)
          - shape (1,2)
          - shape (B,2)  -> logs the FIRST batch sample
        """
        return self._to_1d(x, expected_len=2)

    # ----------------- main logging -----------------

    def record(self, env, diag: Dict[str, Any], tau_real: Any, qref: Any):
        """
        env  : environment object, with .muscle, .skeleton, .states.
        diag : diagnostic dict from controller (Torch or NumPy arrays allowed):
               keys: 'xref_tuple', 'x', 'xd', 'q', 'tau_des', 'act', 'R', 'Fmax'
               These can be either unbatched or batched on leading dim B.
        tau_real : actual joint torque (2D or (B,2))
        qref     : reference joint angles (2D or (B,2))
        """
        k = self.k
        if k >= self.steps:
            # silently ignore if we overflow (like ring buffer could be added later)
            return

        # --- unpack diagnostics ---
        x_d_raw, xd_d_raw, _ = diag["xref_tuple"]  # each (2,) or (B,2)
        # x_d, xd_d are mostly for plotting error vs ref, so we just take batch 0
        x = self._to_2d_vec(diag["x"])       # (2,)
        xd = self._to_2d_vec(diag["xd"])     # (2,)
        x_d = self._to_2d_vec(x_d_raw)       # (2,)
        q = self._to_2d_vec(diag["q"])       # (2,)
        qref_vec = self._to_2d_vec(qref)     # (2,)

        # tau_real & tau_des
        tau_real_t = self._to_2d_vec(tau_real)
        tau_des_val = diag.get("tau_des", None)
        if tau_des_val is None:
            tau_des_t = torch.zeros(2, device=self.device, dtype=self.dtype)
        else:
            tau_des_t = self._to_2d_vec(tau_des_val)

        # activations: allow (M,) or (B,M), log first sample
        act_vec = self._to_1d(diag["act"], expected_len=self.n_muscles)

        # --- write state logs ---
        self.x_log[k] = x
        self.xd_log[k] = xd
        self.xref_log[k] = x_d
        self.q_log[k] = q
        self.qref_log[k] = qref_vec

        self.tau_des_log[k] = tau_des_t
        self.tau_real_log[k] = tau_real_t
        self.res_log[k] = tau_des_t - tau_real_t

        # --- activations & saturation stats ---
        self.act_log[k] = act_vec
        self.act_norm[k] = act_vec.norm()

        a_min = float(getattr(env.muscle, "min_activation", 0.0))
        eps = 1e-3
        self.sat_min_pct[k] = (
            (act_vec <= a_min + eps).to(self.dtype).mean() * 100.0
        )
        self.sat_max_pct[k] = ((act_vec >= 1.0 - eps).to(self.dtype).mean() * 100.0)

        # --- conditioning of R and R*diag(Fmax) ---
        R_raw = diag["R"]
        R_t = torch.as_tensor(R_raw, device=self.device, dtype=self.dtype)
        # R can be (n,M) or (B,n,M) -> log first batch
        if R_t.ndim == 3:
            R_t = R_t[0]
        # Now R_t is (n,M) (2 x n_muscles for 2-DoF)
        try:
            self.condR_log[k] = torch.linalg.cond(R_t)
        except RuntimeError:
            self.condR_log[k] = float("inf")

        Fmax_t = torch.as_tensor(
            diag["Fmax"], device=self.device, dtype=self.dtype
        ).reshape(-1)  # (M,)

        try:
            M = R_t @ torch.diag(Fmax_t)  # (n,M)
            self.condM_log[k] = torch.linalg.cond(M)
        except RuntimeError:
            self.condM_log[k] = float("inf")

        # --- link positions via 2-DoF planar geometry (log first sample) ---
        # Assume env.skeleton is a 2-link planar arm with attributes l1, l2
        arm = env.skeleton
        l1 = float(getattr(arm, "l1", 1.0))
        l2 = float(getattr(arm, "l2", 1.0))

        q1, q2 = q[0], q[1]
        c1, s1 = torch.cos(q1), torch.sin(q1)
        c12, s12 = torch.cos(q1 + q2), torch.sin(q1 + q2)

        base = torch.tensor([0.0, 0.0], device=self.device, dtype=self.dtype)
        joint1 = torch.stack([l1 * c1, l1 * s1])
        joint2 = torch.stack([joint1[0] + l2 * c12, joint1[1] + l2 * s12])

        # fingertip from env.states["fingertip"] (B,2) or (B,>=2) -> take first batch
        ft = env.states.get("fingertip", None)
        if ft is not None:
            ft_t = torch.as_tensor(
                ft, device=self.device, dtype=self.dtype
            )
            if ft_t.ndim == 1:
                # (2,) or (>=2,)
                ft_vec = ft_t.reshape(-1)[:2]
            else:
                # (B,2) or (B,>=2)
                ft_vec = ft_t.reshape(ft_t.shape[0], -1)[0, :2]
        else:
            ft_vec = joint2  # fallback

        self.links_xy[k, 0] = base
        self.links_xy[k, 1] = joint1
        self.links_xy[k, 2] = joint2
        self.links_xy[k, 3] = ft_vec

        self.k += 1

    def time(self, dt: float) -> Tuple[int, np.ndarray]:
        """
        Return (k, t_vec) where t_vec is a NumPy array of time stamps [0, dt, 2dt, ...].
        """
        k = self.k
        t = np.arange(k) * float(dt)
        return k, t


# ---------------------------------------------------------------------
# Simple smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    print("[log_buffer_torch] Simple smoke test...")

    # dummy env with minimal attributes
    class DummyMuscle:
        def __init__(self):
            self.min_activation = 0.01

    class DummySkeleton:
        def __init__(self):
            self.l1 = 0.3
            self.l2 = 0.3

    class DummyEnv:
        def __init__(self, B: int = 1):
            self.muscle = DummyMuscle()
            self.skeleton = DummySkeleton()
            # fingertip: batched (B,2)
            self.states = {
                "fingertip": torch.tensor(
                    [[0.5, 0.4]] * B, dtype=torch.get_default_dtype()
                )
            }

    steps = 10
    nm = 3
    logs = LogBufferTorch(steps, nm, device=torch.device("cpu"))

    # ---- unbatched diag ----
    env1 = DummyEnv(B=1)
    diag1 = {
        "xref_tuple": (
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 0.0]),
        ),
        "x": torch.tensor([0.0, 0.0]),
        "xd": torch.tensor([0.0, 0.0]),
        "q": torch.tensor([0.1, 0.2]),
        "tau_des": torch.tensor([0.5, 0.6]),
        "act": torch.tensor([0.3, 0.4, 0.5]),
        "R": torch.eye(2, nm),
        "Fmax": torch.tensor([10.0, 20.0, 30.0]),
    }
    tau_real1 = torch.tensor([0.4, 0.5])
    qref1 = torch.tensor([0.0, 0.0])

    logs.record(env1, diag1, tau_real1, qref1)

    # ---- batched diag (B=4) ----
    B = 4
    envB = DummyEnv(B=B)

    diagB = {
        "xref_tuple": (
            torch.stack([torch.tensor([0.1, 0.2])] * B, dim=0),  # (B,2)
            torch.zeros(B, 2),
            torch.zeros(B, 2),
        ),
        "x": torch.stack([torch.tensor([0.0, 0.0])] * B, dim=0),   # (B,2)
        "xd": torch.stack([torch.tensor([0.0, 0.0])] * B, dim=0),  # (B,2)
        "q": torch.stack([torch.tensor([0.1, 0.2])] * B, dim=0),   # (B,2)
        "tau_des": torch.stack([torch.tensor([0.5, 0.6])] * B, dim=0),  # (B,2)
        "act": torch.stack([torch.tensor([0.3, 0.4, 0.5])] * B, dim=0), # (B,3)
        "R": torch.stack([torch.eye(2, nm)] * B, dim=0),                # (B,2,3)
        "Fmax": torch.tensor([10.0, 20.0, 30.0]),
    }
    tau_realB = torch.stack([torch.tensor([0.4, 0.5])] * B, dim=0)  # (B,2)
    qrefB = torch.stack([torch.tensor([0.0, 0.0])] * B, dim=0)      # (B,2)

    logs.record(envB, diagB, tau_realB, qrefB)

    k, t = logs.time(dt=0.01)

    print("  k:", k)
    print("  t[:5]:", t[:5])
    print("  x_log[0]:", logs.x_log[0])
    print("  x_log[1]:", logs.x_log[1])
    print("  tau_des_log[0]:", logs.tau_des_log[0])
    print("  tau_real_log[0]:", logs.tau_real_log[0])
    print("  res_log[0]:", logs.res_log[0])
    print("  act_log[0]:", logs.act_log[0])
    print("  act_log[1]:", logs.act_log[1])
    print("  sat_min_pct[0]:", logs.sat_min_pct[0].item())
    print("  sat_max_pct[0]:", logs.sat_max_pct[0].item())
    print("  condR_log[0]:", logs.condR_log[0].item())
    print("  condM_log[0]:", logs.condM_log[0].item())
    print("  links_xy[0]:", logs.links_xy[0])
    print("  links_xy[1]:", logs.links_xy[1])

    print("[log_buffer_torch] Smoke test done.")
