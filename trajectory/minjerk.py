# trajectory/minijerk.py
import numpy as np
from dataclasses import dataclass


@dataclass
class MinJerkParams:
    Vmax: float
    Amax: float
    Jmax: float
    gamma: float = 1.10


def _minjerk_profile(T, t):
    T = float(max(T, 1e-9))
    tau = np.clip(t / T, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
    sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
    sddd = (60 - 360 * tau + 360 * tau**2) / T**3
    return s, sd, sdd, sddd


def _segment_time(L, p: MinJerkParams):
    Vfac, Afac, Jfac = 1.875, 5.7735026919, 60.0
    T_v = Vfac * L / max(p.Vmax, 1e-9)
    T_a = np.sqrt(Afac * L / max(p.Amax, 1e-9))
    T_j = np.cbrt(Jfac * L / max(p.Jmax, 1e-9))
    return float(p.gamma * max(T_v, T_a, T_j, 1e-3))


class MinJerkLinearTrajectory:
    def __init__(self, waypoints, params: MinJerkParams):
        self.segs, self.tgrid = self._plan(waypoints, params)

    def _plan(self, waypoints, params):
        segs, tgrid = [], [0.0]
        for k in range(len(waypoints) - 1):
            P0 = np.asarray(waypoints[k], float)
            P1 = np.asarray(waypoints[k + 1], float)
            L = float(np.linalg.norm(P1 - P0))
            T = _segment_time(L, params)
            segs.append((P0, P1, T))
            tgrid.append(tgrid[-1] + T)
        return segs, np.array(tgrid)

    def sample(self, t):
        if len(self.segs) == 0:
            raise ValueError("Empty trajectory")
        if t <= self.tgrid[0]:
            k, tau = 0, 0.0
        elif t >= self.tgrid[-1]:
            k, tau = len(self.segs) - 1, self.segs[-1][2]
        else:
            k = int(np.searchsorted(self.tgrid, t) - 1)
            tau = float(t - self.tgrid[k])
        P0, P1, T = self.segs[k]
        d = (P1 - P0).astype(float)
        s, sd, sdd, _ = _minjerk_profile(T, tau)
        x = P0 + d * s
        xd = d * sd
        xdd = d * sdd
        return x, xd, xdd
