import numpy as np, scipy.io as sio
from scipy.signal import butter, filtfilt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26
import model_lib.numpy.skeleton as _sk; _sk.USE_CACHE = False  # deterministic, reproducible (no rounded htm cache)
from controller.numpy.pd_if_controller import PDIFController, PDIFParams
from sim.numpy.simulator import TargetReachSimulator
B="/home/jdiab/Desktop/PAPER/enhanced_paper_corrected (Copy)"
SHOI,ELBI=3,6;SHO=np.deg2rad([0,140]);ELB=np.deg2rad([0,160]);N=100;ph=np.linspace(0,1,N)
def build():
    mu=RigidTendonHillMuscle(min_activation=0.02);arm=RigidTendonArm26(muscle=mu,timestep=0.002,damping=2.,n_ministeps=1,integration_method="Euler")
    return Environment(effector=arm,max_ep_duration=20.,action_noise=0.,obs_noise=0.,proprioception_delay=arm.dt,vision_delay=arm.dt,name="L"),arm
class T:
    def __init__(s,x,v,a,dt):s.x,s.v,s.a,s.dt=x,v,a,dt
    def sample(s,t):k=min(int(round(t/s.dt)),len(s.x)-1);return s.x[k],s.v[k],s.a[k]
b5,a5=butter(2,5/500,"low")
def envf(x):x=np.asarray(x,float)-np.median(x);return np.clip(filtfilt(b5,a5,np.abs(x)),0,None)
def pid(e,a):return PDIFController(e,a,PDIFParams(Kp_task=np.array([2400.,2400.]),damping_ratio=0.7,Kff=1.,use_critical_damping=True,enable_inertia_comp=True,enable_gravity_comp=False,enable_coriolis_comp=True,enable_nullspace=True,Kp_null=20.,Kd_null=5.,eps=1e-6,lam_os_max=200.,sigma_thresh=1e-4,gate_pow=2.,bisect_iters=12,enable_internal_force=False))
m=sio.loadmat(f"{B}/kinarm_dataset/raw_lucchetti/HS01.mat",struct_as_record=False,squeeze_me=True);D=m["s"].DataULdom;KF,EF=m["s"].KinFreq,m["s"].EmgFreq
PAIRS=[(0,2,"Shoulder flexor"),(1,0,"Shoulder extensor"),(4,4,"Biceps (flexor)"),(3,3,"Triceps (extensor)")]
M={p[2]:[] for p in PAIRS};E_={p[2]:[] for p in PAIRS}
for tk in D:
    A=np.atleast_2d(tk.Angles);E=np.atleast_2d(tk.EMG);sh=A[SHOI];el=A[ELBI]
    if (sh.max()-sh.min())<40 or (el.max()-el.min())<40: continue
    st=np.atleast_1d(tk.Events.Start).astype(int);en=np.atleast_1d(tk.Events.End).astype(int)
    for j in range(min(len(st),len(en))):
        a,b=st[j],en[j]
        if b-a<40:continue
        th=np.vstack([np.clip(np.deg2rad(sh[a:b]),*SHO),np.clip(np.deg2rad(el[a:b]),*ELB)])
        env_,arm=build();dt=arm.dt;tt=np.arange(th.shape[1])/KF;tn=np.arange(0,tt[-1],dt)
        ths=np.vstack([np.interp(tn,tt,th[i]) for i in range(2)])
        if ths.shape[1]<20:continue
        xd=np.array([arm.skeleton.joint2cartesian(joint_state=np.concatenate([ths[:,k],[0,0]])[None,:])[0,:2] for k in range(ths.shape[1])])
        xv=np.gradient(xd,dt,axis=0);xa=np.gradient(xv,dt,axis=0)
        env_.reset(options={"joint_state":np.concatenate([ths[:,0],[0,0]])[None,:],"deterministic":True})
        logs=TargetReachSimulator(env_,arm,pid(env_,arm),T(xd,xv,xa,dt),ths.shape[1]).run();act=logs.act_log[:logs.k]
        tm=np.linspace(0,1,act.shape[0]);te=np.linspace(0,1,b*EF//KF-a*EF//KF)
        for col,row,lab in PAIRS:
            mo=np.interp(ph,tm,act[:,col]);mo/=(mo.max() or 1);me=np.interp(ph,te,envf(E[row][a*EF//KF:b*EF//KF]));me/=(me.max() or 1)
            M[lab].append(mo);E_[lab].append(me)
# FIG 1: emg_diagnostic = single representative reach (raw, noisy) 2x2
fig,ax=plt.subplots(2,2,figsize=(6.9,3.4))
for a,(c,r,lab) in zip(ax.ravel(),PAIRS):
    a.plot(ph*100,E_[lab][0],"k",lw=1.6,label="EMG (1 reach)");a.plot(ph*100,M[lab][0],"r--",lw=1.6,label="Model (1 reach)")
    a.set_title(lab,fontsize=9);a.grid(ls="--",alpha=.3);a.set_ylim(-0.05,1.15)
ax[0,0].legend(fontsize=7);[a.set_xlabel("reach %") for a in ax[1,:]];[a.set_ylabel("norm. act.") for a in ax[:,0]]
fig.tight_layout()
fig.savefig(f"{B}/overleaf_model_based/figures/emg_diagnostic.png",dpi=130,bbox_inches="tight");fig.savefig(f"{B}/kinarm_dataset/emg_diagnostic.png",dpi=130,bbox_inches="tight")
# FIG 2: emg_meanprofile = mean over reaches + r, 2x2
fig,ax=plt.subplots(2,2,figsize=(6.9,3.4))
for a,(c,r,lab) in zip(ax.ravel(),PAIRS):
    Mm=np.array(M[lab]).mean(0);Em=np.array(E_[lab]).mean(0);rr=np.corrcoef(Mm,Em)[0,1]
    a.plot(ph*100,Em,"k",lw=2,label="EMG (mean)");a.plot(ph*100,Mm,"r--",lw=2,label="Model (mean)")
    a.set_title(f"{lab}  r={rr:+.2f}",fontsize=9);a.grid(ls="--",alpha=.3);a.set_ylim(-0.05,1.15)
ax[0,0].legend(fontsize=7);[a.set_xlabel("reach %") for a in ax[1,:]];[a.set_ylabel("norm. act.") for a in ax[:,0]]
fig.tight_layout()
fig.savefig(f"{B}/overleaf_model_based/figures/emg_meanprofile.png",dpi=130,bbox_inches="tight");fig.savefig(f"{B}/kinarm_dataset/emg_meanprofile.png",dpi=130,bbox_inches="tight")
print("wrote compact emg_diagnostic.png and emg_meanprofile.png (2x2)")
