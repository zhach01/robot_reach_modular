# trajectory/numpy/lissajous.py
# Smooth Lissajous / figure-8 trajectory for benchmark task T4 (paper Fig. 31).
import numpy as np
from dataclasses import dataclass


@dataclass
class LissajousParams:
    A: float = 0.08          # x amplitude (m)
    B: float = 0.06          # y amplitude (m)
    wx: float = 2.0 * np.pi / 4.0   # x angular freq (rad/s) -> 4 s period
    wy: float = 4.0 * np.pi / 4.0   # y angular freq (= 2*wx -> figure-8)
    delta: float = 0.0       # phase offset
    ramp_tau: float = 0.25   # s, smooth amplitude ramp-in to avoid a start velocity jump


class LissajousTrajectory:
    """
    x(t) = cx + A*r(t)*sin(wx t),   y(t) = cy + B*r(t)*sin(wy t + delta)
    with a smooth amplitude ramp r(t)=1-exp(-t/ramp_tau) so the path eases out of
    the start pose (fingertip at the center) without a velocity discontinuity.
    sample(t) -> (x, xd, xdd) matching the MinJerk trajectory API.
    """

    def __init__(self, center, params: LissajousParams = None):
        self.c = np.asarray(center, dtype=float)[:2]
        self.p = params or LissajousParams()

    def sample(self, t):
        p = self.p
        t = float(max(t, 0.0))
        # amplitude ramp r, r', r''
        tau = max(p.ramp_tau, 1e-6)
        e = np.exp(-t / tau)
        r = 1.0 - e
        rd = e / tau
        rdd = -e / (tau * tau)

        sx, cx = np.sin(p.wx * t), np.cos(p.wx * t)
        sy, cy = np.sin(p.wy * t + p.delta), np.cos(p.wy * t + p.delta)

        # f = r*sin(w t); f' = r'*sin + r*w*cos; f'' = r''*sin + 2 r' w cos - r w^2 sin
        def comp(amp, w, s, c):
            f = amp * r * s
            fd = amp * (rd * s + r * w * c)
            fdd = amp * (rdd * s + 2.0 * rd * w * c - r * w * w * s)
            return f, fd, fdd

        fx, fxd, fxdd = comp(p.A, p.wx, sx, cx)
        fy, fyd, fydd = comp(p.B, p.wy, sy, cy)

        x = self.c + np.array([fx, fy])
        xd = np.array([fxd, fyd])
        xdd = np.array([fxdd, fydd])
        return x, xd, xdd
