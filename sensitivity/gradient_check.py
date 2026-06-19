# sensitivity/gradient_check.py
"""
Automatic-differentiation vs finite-difference gradient check, Section VIII.F.

The sensitivity derivatives d(tau_j)/d(p_k) are computed two independent ways and
compared: (1) reverse-mode automatic differentiation (torch autograd, float64) and
(2) central finite differences on the same torque map. The maximum absolute
mismatch over all parameters/outputs is reported; agreement to ~1e-8 confirms the
hand-derived/AD gradients used in the sensitivity study are correct.
"""
from __future__ import annotations

import numpy as np
import torch

from .lhc_analysis import DEFAULT_PARAMS, torque_model

torch.set_default_dtype(torch.float64)

# parameters that actually enter the rigid-body + moment-arm torque map
_ACT = np.array([0.3, 0.3, 0.2, 0.4, 0.2, 0.2])
_THETA = np.array([np.radians(35), np.radians(50)])
_THETA_D = np.array([0.0, 0.0])
_THETA_DD = np.array([1.0, 0.5])


def torque_model_torch(p: dict, theta=_THETA, theta_dot=_THETA_D, theta_ddot=_THETA_DD,
                       act=_ACT):
    """Differentiable torch mirror of lhc_analysis.torque_model (returns tau (2,))."""
    g = lambda k, d: p[k] if k in p else torch.tensor(d)
    L1, L2 = g("L1", 0.27), g("L2", 0.26)
    Lg1, Lg2 = g("Lg1", 0.135), g("Lg2", 0.165)
    m1, m2 = g("m1", 1.82), g("m2", 1.44)
    I1, I2 = g("I1", 0.012), g("I2", 0.018)
    a1, a2 = g("a1", 0.035), g("a2", 0.028)
    a61, a62 = g("a61", 0.025), g("a62", 0.024)
    Fmax = torch.stack([g("Fmax_DEL", 1142.6), g("Fmax_PEC", 699.8), g("Fmax_BRA", 987.3),
                        g("Fmax_BIC", 780.0), g("Fmax_TRI_long", 798.5), g("Fmax_TRI_lat", 624.3)])
    q2 = float(theta[1]); qd1, qd2 = theta_dot; qdd1, qdd2 = theta_ddot
    c2, s2 = np.cos(q2), np.sin(q2)
    M11 = I1 + I2 + m1 * Lg1**2 + m2 * (L1**2 + Lg2**2 + 2 * L1 * Lg2 * c2)
    M12 = I2 + m2 * (Lg2**2 + L1 * Lg2 * c2)
    M22 = I2 + m2 * Lg2**2
    h = -m2 * L1 * Lg2 * s2
    tau_in0 = M11 * qdd1 + M12 * qdd2 + h * qd2 * (2 * qd1 + qd2)
    tau_in1 = M12 * qdd1 + M22 * qdd2 - h * qd1**2
    act_t = torch.as_tensor(act)
    F = act_t * Fmax
    z = torch.zeros((), dtype=Fmax.dtype)
    R0 = torch.stack([a1, a1, z, a1 * 0.85, -a61, z])
    R1 = torch.stack([z, z, a2, a2, -a62, -a62])
    return torch.stack([(R0 * F).sum() - tau_in0, (R1 * F).sum() - tau_in1])


def gradient_check(param_specs=DEFAULT_PARAMS, output_idx=0, fd_delta=1e-6, verbose=True):
    names = [s.name for s in param_specs]
    nominal = {s.name: s.nominal for s in param_specs}

    # --- AD gradient ---
    leaves = {k: torch.tensor(float(v), requires_grad=True) for k, v in nominal.items()}
    tau = torque_model_torch(leaves)[output_idx]
    tau.backward()
    ad = np.array([(leaves[n].grad.item() if leaves[n].grad is not None else 0.0) for n in names])

    # --- central finite difference on the numpy model ---
    fd = np.zeros(len(names))
    for i, n in enumerate(names):
        p0 = nominal[n]
        if p0 == 0:
            continue
        pp = dict(nominal); pp[n] = p0 * (1 + fd_delta)
        pm = dict(nominal); pm[n] = p0 * (1 - fd_delta)
        fd[i] = (torque_model(pp)[output_idx] - torque_model(pm)[output_idx]) / (2 * fd_delta * p0)

    mism = np.abs(ad - fd)
    kmax = int(np.argmax(mism))
    if verbose:
        joint = "shoulder" if output_idx == 0 else "elbow"
        print(f"=== AD vs finite-difference gradient check ({joint} torque) ===")
        print(f"{'param':14s}{'AD':>14s}{'FD':>14s}{'|diff|':>12s}")
        order = np.argsort(-np.abs(ad))
        for i in order[:8]:
            print(f"{names[i]:14s}{ad[i]:>14.6f}{fd[i]:>14.6f}{mism[i]:>12.2e}")
        print(f"max |AD - FD| = {mism[kmax]:.2e}  (at {names[kmax]})")
    return float(mism[kmax]), dict(names=names, ad=ad, fd=fd)


if __name__ == "__main__":
    gradient_check()
