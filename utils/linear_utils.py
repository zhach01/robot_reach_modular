import numpy as np


def nnls_small_active_set(tau_des, R, iters=12):
    """Solve min ||R f + tau_des|| s.t. f>=0 (tiny active-set, OK for M≈6–12)."""
    _, M = R.shape
    active = np.ones(M, dtype=bool)
    b = -tau_des
    for _ in range(iters):
        if not np.any(active):
            return np.zeros(M)
        Ra = R[:, active]
        fa, *_ = np.linalg.lstsq(Ra, b, rcond=None)
        f = np.zeros(M)
        f[active] = fa
        neg = f < 0
        if not np.any(neg):
            return f
        active[neg] = False
    f = np.zeros(M)
    Ra = R[:, active]
    fa, *_ = np.linalg.lstsq(Ra, b, rcond=None)
    f[active] = fa
    return np.clip(f, 0.0, None)
