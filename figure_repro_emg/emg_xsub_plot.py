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

# per-muscle means +/- SD across subjects (Left panel)
mb_m = np.array([np.mean([d["r_base"][lab] for d in recs]) for lab in PAIRS])
mb_s = np.array([np.std ([d["r_base"][lab] for d in recs]) for lab in PAIRS])
mi_m = np.array([np.mean([d["r_imp"][lab]  for d in recs]) for lab in PAIRS])
mi_s = np.array([np.std ([d["r_imp"][lab]  for d in recs]) for lab in PAIRS])

w = 0.38
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.3), gridspec_kw={"width_ratios": [1, 1.45]})

# Left: per-muscle, baseline vs decel-gated (mean +/- SD over subjects)
xm = np.arange(len(PAIRS))
axL.bar(xm - w/2, mb_m, w, yerr=mb_s, capsize=3, color="0.6", label="Baseline (effort-min)")
axL.bar(xm + w/2, mi_m, w, yerr=mi_s, capsize=3, color="#c0392b", label="Decel-gated co-contraction")
axL.axhline(0, color="k", lw=0.6)
axL.set_xticks(xm); axL.set_xticklabels(["Shoulder\nflexor", "Shoulder\nextensor", "Biceps", "Triceps"], fontsize=9)
axL.set_ylabel("model-EMG correlation $r$ (mean $\\pm$ SD)")
axL.set_title(f"Per-muscle (n={len(kept)} subjects)", fontsize=11)
axL.grid(axis="y", ls="--", alpha=.3); axL.legend(fontsize=8, loc="upper left")

# Right: per-subject overall correlation
x = np.arange(len(kept))
axR.bar(x - w/2, sub_base, w, color="0.6", label="Baseline (effort-min)")
axR.bar(x + w/2, sub_imp, w, color="#c0392b", label="Decel-gated co-contraction")
axR.axhline(sub_base.mean(), color="0.4", ls="--", lw=1)
axR.axhline(sub_imp.mean(), color="#c0392b", ls="--", lw=1)
axR.set_xticks(x); axR.set_xticklabels(kept, fontsize=9, rotation=45)
axR.set_ylabel("overall correlation $r$ (mean over 4 muscles)")
axR.set_title(f"Per-subject: mean $r$ {sub_base.mean():.2f}$\\to${sub_imp.mean():.2f}, "
              f"{n_imp}/{len(kept)} improve", fontsize=11)
axR.grid(axis="y", ls="--", alpha=.3); axR.legend(fontsize=8, loc="upper right")

fig.suptitle("Cross-subject generalisation (Lucchetti 2025)", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(f"{B}/kinarm_dataset/emg_crosssubject.pdf", bbox_inches="tight")
print("WROTE", OUT)
