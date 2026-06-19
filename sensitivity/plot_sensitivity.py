# sensitivity/plot_sensitivity.py
"""
Regenerate figures/sensitivity_bars.pdf from the actual local sensitivity
coefficients (so the figure, its caption, and Table tab:sensCoeffs agree).
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .lhc_analysis import run_sensitivity_analysis, DEFAULT_PARAMS

# symbol labels for the figure
SYMBOL = {
    "a1": r"$a_1$ (sh. flexor lever)", "Fmax_DEL": r"$F_{\max,1}$ (deltoid)",
    "Fmax_BIC": r"$F_{\max,5}$ (biceps)", "Fmax_PEC": r"$F_{\max,2}$ (pectoralis)",
    "a61": r"$a_{61}$ (tri-long lever)", "Fmax_TRI_long": r"$F_{\max,6}$ (tri-long)",
    "Fmax_BRA": r"$F_{\max,3}$ (brachialis)", "Fmax_TRI_lat": r"$F_{\max,4}$ (tri-lat)",
    "L1": r"$L_1$", "m2": r"$m_2$", "Lg2": r"$L_{g2}$", "I1": r"$I_1$",
    "l0": r"$l_0$", "ls": r"$l_s$", "tau_act": r"$\tau_{\rm act}$",
    "D1": r"$D_1$", "D2": r"$D_2$",
}


def make_figure(out_path: str, n_show: int = 10, threshold: float = 0.2):
    sens = run_sensitivity_analysis(verbose=False)
    items = sorted(sens.items(), key=lambda kv: kv[1])      # ascending for barh
    items = items[-n_show:]
    names = [SYMBOL.get(k, k) for k, _ in items]
    vals = np.array([v for _, v in items])
    colors = ["#c0392b" if v >= threshold else "#7f8c8d" for v in vals]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=9)
    ax.axvline(threshold, ls="--", color="red", lw=1.0)
    ax.text(threshold, len(vals) - 0.4, "  high-sensitivity (0.2)", color="red",
            fontsize=8, va="center")
    for i, v in enumerate(vals):
        ax.text(v + 0.01 * max(vals), i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel(r"$|S_{1k}|$  (dimensionless shoulder-torque sensitivity)")
    ax.set_title(r"Parameter sensitivity ranking ($\theta_1{=}35^\circ,\ \theta_2{=}50^\circ$)")
    ax.set_xlim(0, max(vals) * 1.15)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}  (top: {names[-1]}={vals[-1]:.2f})")


if __name__ == "__main__":
    default = os.path.join(os.path.dirname(__file__), "..", "..",
                           "PAPER", "overleaf_model_based", "figures", "sensitivity_bars.pdf")
    make_figure(os.path.abspath(default))
