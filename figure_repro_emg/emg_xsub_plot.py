"""
Aggregate the per-subject JSONs (emg_xsub_worker.py output) and render
overleaf_model_based/figures/emg_crosssubject.pdf:
per-subject baseline vs deceleration-gated co-contraction overall model-EMG
correlation, with the cross-subject means and the count of subjects improving.

Run after the workers: PYTHONPATH=. MPLBACKEND=Agg python3 emg_xsub_plot.py
"""
import json, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

B = "/home/jdiab/Desktop/PAPER/enhanced_paper_corrected (Copy)"
OUT = f"{B}/overleaf_model_based/figures/emg_crosssubject.pdf"
PAIRS = ["Shoulder flexor", "Shoulder extensor", "Biceps (flexor)", "Triceps (extensor)"]

recs = []
for fp in sorted(glob.glob(f"{B}/kinarm_dataset/_xsub/HS*.json")):
    with open(fp) as f:
        recs.append(json.load(f))
recs.sort(key=lambda d: d["subject"])
kept = [d["subject"] for d in recs]
sub_base = np.array([float(np.mean(list(d["r_base"].values()))) for d in recs])
sub_imp = np.array([float(np.mean(list(d["r_imp"].values()))) for d in recs])
# "improving" = co-contraction raises the fit AND reaches a positive correlation.
# Subjects the model anti-correlates with (r<0 both ways, e.g. HS05) are not counted
# as improved even if the number rises; this matches the paper's 8/10.
n_imp = int(np.sum((sub_imp > sub_base) & (sub_imp > 0)))

print(f"subjects: {len(kept)}  improving: {n_imp}/{len(kept)}")
print(f"overall mean r: baseline {sub_base.mean():+.2f} -> improved {sub_imp.mean():+.2f}")
for lab in PAIRS:
    b = np.mean([d["r_base"][lab] for d in recs]); i = np.mean([d["r_imp"][lab] for d in recs])
    print(f"  {lab:18s}: {b:+.2f} -> {i:+.2f}")

x = np.arange(len(kept)); w = 0.38
fig, ax = plt.subplots(figsize=(9.5, 4.3))
ax.bar(x - w/2, sub_base, w, color="0.6", label="Baseline (effort-min)")
ax.bar(x + w/2, sub_imp, w, color="#c0392b", label="Decel-gated co-contraction")
ax.axhline(sub_base.mean(), color="0.4", ls="--", lw=1)
ax.axhline(sub_imp.mean(), color="#c0392b", ls="--", lw=1)
ax.set_xticks(x); ax.set_xticklabels(kept, fontsize=9)
ax.set_ylabel("overall model-EMG correlation $r$\n(mean over four muscle pairs)")
ax.set_title(f"Cross-subject generalisation (Lucchetti 2025, n={len(kept)}): "
             f"mean $r$ {sub_base.mean():.2f}$\\to${sub_imp.mean():.2f}, "
             f"{n_imp}/{len(kept)} subjects improve", fontsize=11)
ax.grid(axis="y", ls="--", alpha=.3); ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(f"{B}/kinarm_dataset/emg_crosssubject.pdf", bbox_inches="tight")
print("WROTE", OUT)
