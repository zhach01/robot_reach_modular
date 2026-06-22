#!/usr/bin/env python3
"""Multi-controller KINARM replay validation through robot_reach_modular.
Measured KINARM reaching (KiPy/KU Leuven, 12 reaches, 2 subjects) -> Arm-26 FK
-> desired hand path -> each classical controller + RigidTendon Hill-muscle
Arm-26 plant. Benchmarks PD+IF, passivity (energy-tank), sliding-mode, OSC."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26
from sim.numpy.simulator import TargetReachSimulator
from controller.numpy.pd_if_controller import PDIFController, PDIFParams
from controller.numpy.energy_tank_controller import EnergyTankController, EnergyTankParams
from controller.numpy.sliding_mode import SlidingModeController, SlidingModeParams
from controller.numpy.osc_controller import OSCController, OSCParams

BASE="/home/jdiab/Desktop/PAPER/enhanced_paper_corrected (Copy)"
CSV=f"{BASE}/kinarm_dataset/kinarm_reaching_multisession.csv"
OUT=f"{BASE}/overleaf_model_based/figures/joint_overlay_mb.pdf"
df=pd.read_csv(CSV); SHO=np.deg2rad([0,140]); ELB=np.deg2rad([0,160])

def build():
    m=RigidTendonHillMuscle(min_activation=0.02)
    arm=RigidTendonArm26(muscle=m,timestep=0.002,damping=2.0,n_ministeps=1,integration_method="Euler")
    env=Environment(effector=arm,max_ep_duration=30.0,action_noise=0.0,obs_noise=0.0,
                    proprioception_delay=arm.dt,vision_delay=arm.dt,name="K")
    return env,arm
class Traj:
    def __init__(s,x,v,a,dt): s.x,s.v,s.a,s.dt=x,v,a,dt
    def sample(s,t): k=min(int(round(t/s.dt)),len(s.x)-1); return s.x[k],s.v[k],s.a[k]
def r2(y,yh):
    y=np.asarray(y);yh=np.asarray(yh);sr=np.sum((y-yh)**2);st=np.sum((y-y.mean())**2)
    return 1-sr/st if st>0 else float("nan")

def make_pdif(env,arm):
    return PDIFController(env,arm,PDIFParams(Kp_task=np.array([2400.,2400.]),damping_ratio=0.7,Kff=1.0,
        use_critical_damping=True,enable_inertia_comp=True,enable_gravity_comp=False,enable_coriolis_comp=True,
        enable_nullspace=True,Kp_null=20.,Kd_null=5.,eps=1e-6,lam_os_max=200.,sigma_thresh=1e-4,gate_pow=2.,
        bisect_iters=12,enable_internal_force=False,cocon_level=0.))
def make_pass(env,arm):
    return EnergyTankController(env,arm,EnergyTankParams(D0=np.diag([25.,25.]),K0=np.diag([800.,800.]),enable_inertia_comp=True,
        enable_gravity_comp=False,enable_joint_damping=True,enable_internal_force=False,cocon_a0=0.0,
        bisect_iters=16,linesearch_eps=1e-6,linesearch_safety=0.99,E0=0.15,Emin=1e-4,Emax=1.0))
def make_smc(env,arm):
    return SlidingModeController(env,arm,SlidingModeParams(lambda_surf=np.array([18.,18.]),
        K_switch=np.array([18.,18.]),phi=np.array([0.075,0.075]),Kff_x=1.0,enable_inertia_comp=True,
        enable_gravity_comp=False,enable_velocity_comp=True,bisect_iters=16))
def make_osc(env,arm):
    return OSCController(env,arm,OSCParams(Kp=4000.0,Kv=126.0,Kp_posture=50.0,Kv_posture=10.0))
CTRLS={"Impedance (PD+IF)":make_pdif,"Passivity":make_pass,"Sliding-mode":make_smc,"OSC":make_osc}

rows=[]; plotc=None
for (sess,gt),g in df.groupby(["session","gtrial"]):
    g=g.reset_index(drop=True)
    th=np.vstack([g.shoulder_angle_rad.values,g.elbow_angle_rad.values])[:,::2]
    if th.shape[1]<80: continue
    w=51 if th.shape[1]>60 else (th.shape[1]//2)*2+1
    ths=np.vstack([savgol_filter(th[i],w,3) for i in range(2)])
    ths[0]=np.clip(ths[0],*SHO); ths[1]=np.clip(ths[1],*ELB)
    e0,a0=build(); dt=e0.arm.dt if hasattr(e0,'arm') else 0.002
    arm0=a0; dt=arm0.dt
    xd=np.array([arm0.skeleton.joint2cartesian(joint_state=np.concatenate([ths[:,k],[0,0]])[None,:])[0,:2] for k in range(ths.shape[1])])
    xv=np.gradient(xd,dt,axis=0); xa=np.gradient(xv,dt,axis=0); steps=ths.shape[1]
    perctrl={}
    for cname,mk in CTRLS.items():
        try:
            env,arm=build()
            env.reset(options={"joint_state":np.concatenate([ths[:,0],[0.,0.]])[None,:],"deterministic":True})
            logs=TargetReachSimulator(env,arm,mk(env,arm),Traj(xd,xv,xa,dt),steps).run()
            k=logs.k; x=logs.x_log[:k]; xref=logs.xref_log[:k]; q=logs.q_log[:k]; tau=logs.tau_real_log[:k]; act=logs.act_log[:k]
            rows.append(dict(controller=cname,session=sess,gt=gt,
                ee_rmse_mm=float(np.sqrt(np.mean(np.sum((x-xref)**2,1)))*1000),
                ee_R2=float(min(r2(xref[:,0],x[:,0]),r2(xref[:,1],x[:,1]))),
                q_R2=float(min(r2(ths[0,:k],q[:,0]),r2(ths[1,:k],q[:,1]))),
                mean_act=float(act.mean()),peak_act=float(act.max()),tau_abs_max=float(np.abs(tau).max())))
            perctrl[cname]=tau[:,0]
        except Exception as ex:
            rows.append(dict(controller=cname,session=sess,gt=gt,ee_rmse_mm=np.nan,ee_R2=np.nan,q_R2=np.nan,
                mean_act=np.nan,peak_act=np.nan,tau_abs_max=np.nan)); print("FAIL",cname,sess,gt,repr(ex)[:80])
    if (plotc is None or steps>plotc["k"]) and len(perctrl)==4:
        plotc=dict(k=steps,t=np.arange(steps)*dt,qref=ths,tau=perctrl)
res=pd.DataFrame(rows); res.to_csv(f"{BASE}/kinarm_dataset/allctrl_replay_metrics.csv",index=False)
print("=== per-controller aggregate over reaches (mean +/- SD) ===")
for c in CTRLS:
    s=res[(res.controller==c)].dropna()
    if len(s)==0: print(f"  {c}: NO SUCCESSFUL RUNS"); continue
    print(f"  {c:18s} n={len(s):2d}  RMSE={s.ee_rmse_mm.mean():.2f}+/-{s.ee_rmse_mm.std():.2f}mm  "
          f"eeR2={s.ee_R2.mean():.3f}  qR2={s.q_R2.mean():.3f}  meanAct={s.mean_act.mean():.3f}  "
          f"peakAct={s.peak_act.mean():.3f}  |tau|max={s.tau_abs_max.mean():.2f}")
# figure
if plotc:
    p=plotc; plt.rcParams.update({"font.size":12,"legend.fontsize":9})
    fig,ax=plt.subplots(2,1,figsize=(7.0,5.6))
    ax[0].plot(p["t"],np.degrees(p["qref"][0]),color="tab:blue",lw=1.8,label="Shoulder (measured)")
    ax[0].plot(p["t"],np.degrees(p["qref"][1]),color="tab:orange",lw=1.8,label="Elbow (measured)")
    ax[0].set_ylabel("Joint angle [deg]"); ax[0].grid(True,ls="--",alpha=.3); ax[0].legend(frameon=False)
    ax[0].set_title("KINARM reaching replayed through Arm-26 (representative reach)")
    cols={"Impedance (PD+IF)":"tab:red","Passivity":"tab:cyan","Sliding-mode":"tab:green","OSC":"tab:purple"}
    for c,tau in p["tau"].items(): ax[1].plot(p["t"],tau,lw=1.1,alpha=.85,color=cols[c],label=c)
    ax[1].set_xlabel("Time [s]"); ax[1].set_ylabel("Shoulder torque [N·m]"); ax[1].grid(True,ls="--",alpha=.3)
    ax[1].legend(loc="upper center",bbox_to_anchor=(0.5,-0.22),ncol=4,frameon=False)
    ax[1].set_title("Shoulder torque: four model-based controllers")
    fig.tight_layout(); fig.savefig(OUT,bbox_inches="tight"); print("wrote",OUT)
