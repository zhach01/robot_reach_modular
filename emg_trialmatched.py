#!/usr/bin/env python3
"""Trial-matched activation<->EMG validation (Lucchetti 2025, decoupled from KiPy).
For each measured reach: drive the model FORWARD with the SAME trial's measured
shoulder/elbow joint angles (PD+IF), get predicted muscle activations, and
compare to that trial's recorded EMG envelopes."""
import numpy as np, scipy.io as sio
from scipy.signal import butter, filtfilt
from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26
from controller.numpy.pd_if_controller import PDIFController, PDIFParams
from sim.numpy.simulator import TargetReachSimulator
BASE="/home/jdiab/Desktop/PAPER/enhanced_paper_corrected (Copy)"
SHOI,ELBI=3,6                  # Angles rows: ShoulderFlx, ElbowFlx (degrees)
EMG_POSTD,EMG_ANTD,EMG_TRI,EMG_BIC=0,2,3,4
def build():
    m=RigidTendonHillMuscle(min_activation=0.02)
    arm=RigidTendonArm26(muscle=m,timestep=0.002,damping=2.0,n_ministeps=1,integration_method="Euler")
    env=Environment(effector=arm,max_ep_duration=20.,action_noise=0.,obs_noise=0.,proprioception_delay=arm.dt,vision_delay=arm.dt,name="L")
    return env,arm
class T:
    def __init__(s,x,v,a,dt):s.x,s.v,s.a,s.dt=x,v,a,dt
    def sample(s,t):k=min(int(round(t/s.dt)),len(s.x)-1);return s.x[k],s.v[k],s.a[k]
def r2corr(a,b): return np.corrcoef(a,b)[0,1]
b5,a5=butter(2,5/(1000/2),"low")
def env_emg(x): x=np.asarray(x,float)-np.median(x); return np.clip(filtfilt(b5,a5,np.abs(x)),0,None)
def pdif(env,arm):
    return PDIFController(env,arm,PDIFParams(Kp_task=np.array([2400.,2400.]),damping_ratio=0.7,Kff=1.,use_critical_damping=True,
     enable_inertia_comp=True,enable_gravity_comp=False,enable_coriolis_comp=True,enable_nullspace=True,Kp_null=20.,Kd_null=5.,
     eps=1e-6,lam_os_max=200.,sigma_thresh=1e-4,gate_pow=2.,bisect_iters=12,enable_internal_force=False,cocon_level=0.))
SHO=np.deg2rad([0,140]); ELB=np.deg2rad([0,160])
m=sio.loadmat(f"{BASE}/kinarm_dataset/raw_lucchetti/HS01.mat",struct_as_record=False,squeeze_me=True)
D=m["s"].DataULdom; KF,EF=m["s"].KinFreq,m["s"].EmgFreq
N=100; ph=np.linspace(0,1,N)
# muscle pairs: (model_act_col, measured_emg_row, label)
PAIRS=[(0,EMG_ANTD,"Shoulder flexor (ant.deltoid)"),(1,EMG_POSTD,"Shoulder extensor (post.deltoid)"),
       (4,EMG_BIC,"Biceps (flexor)"),(3,EMG_TRI,"Triceps (extensor)")]
corr={lab:[] for _,_,lab in PAIRS}; nreach=0
for ti,tk in enumerate(D):
    A=np.atleast_2d(tk.Angles); E=np.atleast_2d(tk.EMG)
    sh=A[SHOI]; el=A[ELBI]
    if (sh.max()-sh.min())<40 or (el.max()-el.min())<40: continue   # reaching tasks only
    st=np.atleast_1d(tk.Events.Start).astype(int); en=np.atleast_1d(tk.Events.End).astype(int)
    for j in range(min(len(st),len(en))):
        a,b=st[j],en[j]
        if b-a<30: continue
        th=np.vstack([np.deg2rad(sh[a:b]),np.deg2rad(el[a:b])])
        th[0]=np.clip(th[0],*SHO); th[1]=np.clip(th[1],*ELB)
        # upsample 125Hz -> arm.dt
        env_,arm=build(); dt=arm.dt
        tt=np.arange(th.shape[1])/KF; tnew=np.arange(0,tt[-1],dt)
        ths=np.vstack([np.interp(tnew,tt,th[i]) for i in range(2)])
        if ths.shape[1]<20: continue
        xd=np.array([arm.skeleton.joint2cartesian(joint_state=np.concatenate([ths[:,k],[0,0]])[None,:])[0,:2] for k in range(ths.shape[1])])
        xv=np.gradient(xd,dt,axis=0); xa=np.gradient(xv,dt,axis=0)
        env_.reset(options={"joint_state":np.concatenate([ths[:,0],[0,0]])[None,:],"deterministic":True})
        logs=TargetReachSimulator(env_,arm,pdif(env_,arm),T(xd,xv,xa,dt),ths.shape[1]).run()
        act=logs.act_log[:logs.k]
        ttm=np.linspace(0,1,act.shape[0])
        ea,eb=a*EF//KF, b*EF//KF
        teg=np.linspace(0,1,eb-ea)
        for col,emrow,lab in PAIRS:
            pa=np.interp(ph,ttm,act[:,col])
            me=np.interp(ph,teg,env_emg(E[emrow][ea:eb]))
            corr[lab].append(r2corr(pa,me))
        nreach+=1
print(f"HS01: {nreach} trial-matched reaches")
print("Trial-matched correlation (predicted activation vs measured EMG):")
for _,_,lab in PAIRS:
    c=np.array(corr[lab]); c=c[np.isfinite(c)]
    print(f"  {lab:34s} r = {c.mean():+.2f} +/- {c.std():.2f}  (n={len(c)})")
