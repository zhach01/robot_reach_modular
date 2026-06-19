# sensitivity/anthropometry.py
"""
Anthropometric rescaling (subject-specific), Section VIII.B.

Geometric-similarity scaling (Zajac 1989) from population-average baseline
parameters to an individual subject given body height h_b [m] and mass m_b [kg]:

    L_i, l0, ls  <- (h_b / h0)            * baseline      (lengths)
    m_i          <- (m_b / m0)            * baseline      (segment masses)
    I_i          <- (m_b / m0)(h_b/h0)^2  * baseline      (segment inertias)
    Fmax         <- (m_b / m0)^(2/3)      * baseline      (force capacity ~ PCSA)

with population reference (h0, m0) = (1.75 m, 73 kg). A subject is described by a
small JSON "anthro" block; apply() returns scaled parameters for any baseline set
(e.g. lhc_analysis.DEFAULT_PARAMS).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

# parameter-name -> scaling category (anything unlisted is left unscaled)
LENGTH_KEYS = {"L1", "L2", "Lg1", "Lg2", "a1", "a2", "a61", "a62", "l0", "ls"}
MASS_KEYS = {"m1", "m2"}
INERTIA_KEYS = {"I1", "I2"}
FORCE_PREFIX = "Fmax"

H0 = 1.75   # reference height [m]
M0 = 73.0   # reference mass [kg]


@dataclass
class Anthro:
    """Subject anthropometry ('anthro' JSON block)."""
    height: float = 1.75      # m
    mass: float = 73.0        # kg
    h0: float = H0
    m0: float = M0

    def factors(self) -> dict:
        """Multiplicative scale factors per category (Zajac geometric similarity)."""
        rh = self.height / self.h0
        rm = self.mass / self.m0
        return {
            "length":  rh,
            "mass":    rm,
            "inertia": rm * rh ** 2,
            "force":   rm ** (2.0 / 3.0),
        }

    def category(self, name: str) -> str:
        if name in LENGTH_KEYS:
            return "length"
        if name in MASS_KEYS:
            return "mass"
        if name in INERTIA_KEYS:
            return "inertia"
        if name.startswith(FORCE_PREFIX):
            return "force"
        return "none"

    def apply(self, params: dict) -> dict:
        """Return a scaled copy of a {name: value} baseline parameter dict."""
        f = self.factors()
        out = {}
        for k, v in params.items():
            c = self.category(k)
            out[k] = v * f[c] if c != "none" else v
        return out

    def to_json(self, path: str | None = None) -> str:
        s = json.dumps({"anthro": asdict(self)}, indent=2)
        if path:
            with open(path, "w") as fh:
                fh.write(s)
        return s

    @classmethod
    def from_json(cls, path_or_str: str) -> "Anthro":
        try:
            with open(path_or_str) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            d = json.loads(path_or_str)
        d = d.get("anthro", d)
        return cls(**{k: d[k] for k in ("height", "mass", "h0", "m0") if k in d})


if __name__ == "__main__":
    a = Anthro(height=1.90, mass=90.0)
    f = a.factors()
    print("Subject (1.90 m, 90 kg) scale factors vs reference (1.75 m, 73 kg):")
    for k, v in f.items():
        print(f"  {k:8s} x{v:.3f}")
    # paper Section VIII.B example: m x1.23, I x1.45, Fmax x1.15, L x1.09
    print("anthro JSON block:")
    print(a.to_json())
