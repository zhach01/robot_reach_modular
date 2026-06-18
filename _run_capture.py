import os, sys
os.environ["MPLBACKEND"] = "Agg"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [HERE] + [p for p in sys.path if p not in ("", HERE, "/tmp")]
SCRIPT, OUTDIR, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUTDIR, exist_ok=True)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib
PP = importlib.import_module("plotting.plots")
_orig_plot_all = PP.plot_all
def _wrap(logs, time_vec, center=None, targets=None, *a, **k):
    x = getattr(logs, "x_log", None); xr = getattr(logs, "xref_log", None)
    if x is not None and xr is not None:
        x = np.asarray(x); xr = np.asarray(xr); T = min(len(x), len(xr))
        e = np.linalg.norm(x[:T] - xr[:T], axis=1); n = max(1, int(0.2*T))
        with open(os.path.join(OUTDIR, "metric.txt"), "w") as f:
            f.write(f"{TAG}  task_RMSE={1000*float(np.sqrt(np.mean(e**2))):.2f}mm  "
                    f"final={1000*e[-1]:.2f}mm  steadystate={1000*np.mean(e[-n:]):.2f}mm  max={1000*e.max():.2f}mm\n")
        print(f"[METRIC] {TAG}  RMSE={1000*float(np.sqrt(np.mean(e**2))):.2f}mm  steadystate={1000*np.mean(e[-n:]):.2f}mm  final={1000*e[-1]:.2f}mm")
    try: return _orig_plot_all(logs, time_vec, center=center, targets=targets, *a, **k)
    except Exception as ex: print("plot_all err:", ex)
PP.plot_all = _wrap
def _save_show(*a, **k):
    for i in plt.get_fignums():
        try: plt.figure(i).savefig(os.path.join(OUTDIR, f"fig{i:02d}.png"), dpi=110, bbox_inches="tight")
        except Exception: pass
    plt.close("all")
plt.show = _save_show
import matplotlib.animation as animation
_fa = animation.FuncAnimation.__init__
def _fa2(self, fig, func, frames=None, *a, **k):
    _fa(self, fig, func, frames, *a, **k)
    try:
        last = frames-1 if isinstance(frames,int) else (list(frames)[-1] if frames is not None else 0)
        func(last)
    except Exception: pass
animation.FuncAnimation.__init__ = _fa2
animation.FuncAnimation.save = lambda *a, **k: None
sys.argv = [SCRIPT]
import runpy
runpy.run_path(SCRIPT, run_name="__main__")
_save_show()
