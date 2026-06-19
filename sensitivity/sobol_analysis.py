# sensitivity/sobol_analysis.py
"""
Global (variance-based) Sobol sensitivity analysis.

Section VIII.E of the paper. Uses a Saltelli design (SALib) over +-uncertainty
parameter ranges and reports first-order (S1) and total-order (ST) Sobol indices
for the peak shoulder torque, plus the leading second-order interaction.

The model evaluated is the same vectorized rigid-body + moment-arm torque map used
for the local sensitivity / Monte-Carlo study (lhc_analysis.torque_model_batch), so
the global and local rankings are directly comparable.

References:
- Saltelli et al. (2010), Comput. Phys. Commun. 181:259-270
- Herman & Usher (2017), SALib, J. Open Source Softw. 2(9):97
"""
from __future__ import annotations

import numpy as np

from .lhc_analysis import DEFAULT_PARAMS, torque_model_batch, ParameterSpec

# SALib >=1.4 moved the Saltelli sampler under SALib.sample.sobol
try:                                            # SALib >= 1.4.5
    from SALib.sample import sobol as _sobol_sample
    from SALib.analyze import sobol as _sobol_analyze
    _sample = _sobol_sample.sample
except Exception:                               # older layout
    from SALib.sample import saltelli as _sobol_sample
    from SALib.analyze import sobol as _sobol_analyze
    _sample = _sobol_sample.sample


def build_problem(param_specs=DEFAULT_PARAMS):
    """SALib problem dict with +-uncertainty bounds, in param_specs column order."""
    return {
        "num_vars": len(param_specs),
        "names": [s.name for s in param_specs],
        "bounds": [[s.nominal * (1.0 - s.uncertainty), s.nominal * (1.0 + s.uncertainty)]
                   for s in param_specs],
    }


def run_sobol(n_base: int = 512, output_idx: int = 0, param_specs=DEFAULT_PARAMS,
              seed: int = 0, second_order: bool = True, verbose: bool = True):
    """
    Saltelli sample + Sobol analysis on |joint torque| (output_idx 0=shoulder).

    n_base -> N*(2D+2) model evaluations (D = num params). With the vectorized
    torque model the whole design runs in one array op.
    """
    problem = build_problem(param_specs)
    X = _sample(problem, n_base, calc_second_order=second_order, seed=seed)
    Y = np.abs(torque_model_batch(X, param_specs)[:, output_idx])     # peak |tau_j|
    Si = _sobol_analyze.analyze(problem, Y, calc_second_order=second_order,
                                print_to_console=False, seed=seed)

    names = problem["names"]
    total = sorted(zip(names, Si["ST"]), key=lambda t: -t[1])
    first = sorted(zip(names, Si["S1"]), key=lambda t: -t[1])

    # leading pairwise interaction (|S2|)
    top_pair = None
    if second_order and "S2" in Si:
        S2 = Si["S2"]
        best = -np.inf
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                v = S2[i, j]
                if np.isfinite(v) and abs(v) > best:
                    best = abs(v); top_pair = (names[i], names[j], float(v))

    if verbose:
        joint = "shoulder" if output_idx == 0 else "elbow"
        print(f"=== Global Sobol sensitivity ({joint} torque) ===")
        print(f"Saltelli design: N_base={n_base} -> {X.shape[0]} evaluations, "
              f"D={problem['num_vars']} params, +-uncertainty bounds")
        print("Top-5 total-order indices (ST):")
        for nm, v in total[:5]:
            print(f"  {nm:14s} ST={v:.3f}")
        print("Top-3 first-order indices (S1):")
        for nm, v in first[:3]:
            print(f"  {nm:14s} S1={v:.3f}")
        if top_pair:
            interactions = float(np.nansum(np.clip(Si["ST"] - Si["S1"], 0, None)))
            print(f"Leading interaction: {top_pair[0]} x {top_pair[1]} "
                  f"S2={top_pair[2]:.3f}; total interaction share ~{interactions:.2f}")
    return Si, total, first, top_pair


if __name__ == "__main__":
    run_sobol(n_base=512, output_idx=0)
