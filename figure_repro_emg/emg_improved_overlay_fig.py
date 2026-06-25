import numpy as np, scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26
import model_lib.numpy.skeleton as _sk; _sk.USE_CACHE = False  # deterministic, reproducible (no rounded htm cache)
from controller.numpy.pd_if_controller import PDIFController, PDIFParams
from sim.numpy.simulator import TargetReachSimulator

B = "/home/jdiab/Desktop/PAPER/enhanced_paper_corrected (Copy)"
OUT = f"{B}/kinarm_dataset/emg_improved_overlay.png"
SHOI, ELBI = 3, 6
SHO = np.deg2rad([0, 140]); ELB = np.deg2rad([0, 160])
N = 100; ph = np.linspace(0, 1, N); DT = 0.002

def build():
    mu = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(muscle=mu, timestep=0.002, damping=2., n_ministeps=1, integration_method="Euler")
    env = Environment(effector=arm, max_ep_duration=20., action_noise=0., obs_noise=0.,
                      proprioception_delay=arm.dt, vision_delay=arm.dt, name="L")
    return env, arm

class T:
    def __init__(s, x, v, a, dt): s.x, s.v, s.a, s.dt = x, v, a, dt
    def sample(s, t):
        k = min(int(round(t / s.dt)), len(s.x) - 1)
        return s.x[k], s.v[k], s.a[k]

b5, a5 = butter(2, 5/500, "low")
bm, am = butter(2, 3/(0.5/DT), "low")
def envf(x):
    x = np.asarray(x, float) - np.median(x)
    return np.clip(filtfilt(b5, a5, np.abs(x)), 0, None)
def smo(x):
    return np.clip(filtfilt(bm, am, x), 0, None)

def params(improved):
    common = dict(Kp_task=np.array([2400., 2400.]), damping_ratio=0.7, Kff=1.,
                  use_critical_damping=True, enable_inertia_comp=True,
                  enable_gravity_comp=False, enable_coriolis_comp=True,
                  enable_nullspace=True, Kp_null=20., Kd_null=5., eps=1e-6,
                  lam_os_max=200., sigma_thresh=1e-4, gate_pow=2., bisect_iters=12)
    if improved:
        return PDIFParams(enable_internal_force=True, cocon_level=0.03,
                          cocon_vel_gain=0.30, cocon_vel_ref=2.0, cocon_decel_only=True, **common)
    return PDIFParams(enable_internal_force=False, **common)

m = sio.loadmat(f"{B}/kinarm_dataset/raw_lucchetti/HS01.mat", struct_as_record=False, squeeze_me=True)
D = m["s"].DataULdom; KF, EF = m["s"].KinFreq, m["s"].EmgFreq

RE = []
for tk in D:
    A = np.atleast_2d(tk.Angles); E = np.atleast_2d(tk.EMG)
    sh = A[SHOI]; el = A[ELBI]
    if (sh.max() - sh.min()) < 40 or (el.max() - el.min()) < 40:
        continue
    st = np.atleast_1d(tk.Events.Start).astype(int); en = np.atleast_1d(tk.Events.End).astype(int)
    for j in range(min(len(st), len(en))):
        a, b = st[j], en[j]
        if b - a < 40:
            continue
        RE.append((sh[a:b], el[a:b], E[:, a*EF//KF:b*EF//KF]))

PAIRS = [(0, 2, "Shoulder flexor"), (1, 0, "Shoulder extensor"),
         (4, 4, "Biceps (flexor)"), (3, 3, "Triceps (extensor)")]

def runcfg(improved):
    # accumulate per-reach model + emg profiles
    accm = {p[2]: [] for p in PAIRS}
    acce = {p[2]: [] for p in PAIRS}
    for sh, el, Ew in RE:
        th = np.vstack([np.clip(np.deg2rad(sh), *SHO), np.clip(np.deg2rad(el), *ELB)])
        env_, arm = build(); dt = arm.dt
        tt = np.arange(th.shape[1]) / KF; tn = np.arange(0, tt[-1], dt)
        ths = np.vstack([np.interp(tn, tt, th[i]) for i in range(2)])
        if ths.shape[1] < 20:
            continue
        xd = np.array([arm.skeleton.joint2cartesian(
            joint_state=np.concatenate([ths[:, k], [0, 0]])[None, :])[0, :2]
            for k in range(ths.shape[1])])
        xv = np.gradient(xd, dt, axis=0); xa = np.gradient(xv, dt, axis=0)
        env_.reset(options={"joint_state": np.concatenate([ths[:, 0], [0, 0]])[None, :],
                            "deterministic": True})
        ctrl = PDIFController(env_, arm, params(improved))
        logs = TargetReachSimulator(env_, arm, ctrl, T(xd, xv, xa, dt), ths.shape[1]).run()
        act = logs.act_log[:logs.k]
        te = np.linspace(0, 1, Ew.shape[1])
        for col, row, lab in PAIRS:
            mo = smo(act[:, col]); mo = np.interp(ph, np.linspace(0, 1, len(mo)), mo); mo /= (mo.max() or 1)
            me = np.interp(ph, te, envf(Ew[row])); me /= (me.max() or 1)
            accm[lab].append(mo); acce[lab].append(me)
    meanm = {lab: np.array(accm[lab]).mean(0) for _, _, lab in PAIRS}
    meane = {lab: np.array(acce[lab]).mean(0) for _, _, lab in PAIRS}
    r = {lab: np.corrcoef(meanm[lab], meane[lab])[0, 1] for _, _, lab in PAIRS}
    return meanm, meane, r

mm_b, me_b, r_b = runcfg(False)
mm_i, me_i, r_i = runcfg(True)

labels = [p[2] for p in PAIRS]
print("config    | " + "  ".join(f"{l[:8]:>8s}" for l in labels) + " | mean")
print("baseline  | " + "  ".join(f"{r_b[l]:+8.2f}" for l in labels) + f" | {np.mean([r_b[l] for l in labels]):+.2f}")
print("improved  | " + "  ".join(f"{r_i[l]:+8.2f}" for l in labels) + f" | {np.mean([r_i[l] for l in labels]):+.2f}")

fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for ax, lab in zip(axes, labels):
    ax.plot(ph * 100, me_i[lab], color="black", lw=3, label="Measured EMG")
    ax.plot(ph * 100, mm_i[lab], color="red", lw=2, ls="-", label="Improved model")
    ax.plot(ph * 100, mm_b[lab], color="gray", lw=2, ls="--", label="Baseline model")
    ax.set_title(f"{lab}\nr: base={r_b[lab]:+.2f} -> improved={r_i[lab]:+.2f}", fontsize=10)
    ax.set_xlabel("reach %"); ax.set_ylabel("normalised activation")
axes[0].legend(fontsize=8, loc="best")
fig.suptitle("Deceleration-gated co-contraction improves model-EMG agreement (HS01, mean over reaches)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=120, bbox_inches="tight")
print("WROTE", OUT)
print(f"baseline mean r = {np.mean([r_b[l] for l in labels]):+.3f}")
print(f"improved mean r = {np.mean([r_i[l] for l in labels]):+.3f}")
print("improved beats baseline:", np.mean([r_i[l] for l in labels]) > np.mean([r_b[l] for l in labels]))
