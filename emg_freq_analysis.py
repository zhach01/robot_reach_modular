import numpy as np, scipy.io as sio
from scipy.signal import butter, filtfilt, coherence, correlate, resample
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26
from controller.numpy.pd_if_controller import PDIFController, PDIFParams
from sim.numpy.simulator import TargetReachSimulator
BASE="/home/jdiab/Desktop/PAPER/enhanced_paper_corrected (Copy)"
SHOI,ELBI=3,6; EMG={"postD":0,"antD":2,"tri":3,"bic":4}
SHO=np.deg2rad([0,140]); ELB=np.deg2rad([0,160]); FS=100.0   # common resample rate
def build():
    mu=RigidTendonHillMuscle(min_activation=0.02)
    arm=RigidTendonArm26(muscle=mu,timestep=0.002,damping=2.,n_ministeps=1,integration_method="Euler")
    return Environment(effector=arm,max_ep_duration=20.,action_noise=0.,obs_noise=0.,proprioception_delay=arm.dt,vision_delay=arm.dt,name="L"),arm
class T:
    def __init__(s,x,v,a,dt):s.x,s.v,s.a,s.dt=x,v,a,dt
    def sample(s,t):k=min(int(round(t/s.dt)),len(s.x)-1);return s.x[k],s.v[k],s.a[k]
b5,a5=butter(2,5/500,"low")
def envf(x):x=np.asarray(x,float)-np.median(x);return np.clip(filtfilt(b5,a5,np.abs(x)),0,None)
def pdif(e,a):return PDIFController(e,a,PDIFParams(Kp_task=np.array([2400.,2400.]),damping_ratio=0.7,Kff=1.,use_critical_damping=True,enable_inertia_comp=True,enable_gravity_comp=False,enable_coriolis_comp=True,enable_nullspace=True,Kp_null=20.,Kd_null=5.,eps=1e-6,lam_os_max=200.,sigma_thresh=1e-4,gate_pow=2.,bisect_iters=12,enable_internal_force=False,cocon_level=0.))
def norm(x): x=x-x.mean(); s=x.std(); return x/s if s>0 else x
def rs(x,n): return resample(x,n)
m=sio.loadmat(f"{BASE}/kinarm_dataset/raw_lucchetti/HS01.mat",struct_as_record=False,squeeze_me=True)
D=m["s"].DataULdom; KF,EF=m["s"].KinFreq,m["s"].EmgFreq
PAIRS=[(0,"antD","ShoFlex"),(1,"postD","ShoExt"),(4,"bic","Biceps"),(3,"tri","Triceps")]
res={lab:{"r0":[],"rmax":[],"lag":[],"coh":[]} for _,_,lab in PAIRS}
for tk in D:
    A=np.atleast_2d(tk.Angles); E=np.atleast_2d(tk.EMG); sh=A[SHOI]; el=A[ELBI]
    if (sh.max()-sh.min())<40 or (el.max()-el.min())<40: continue
    st=np.atleast_1d(tk.Events.Start).astype(int); en=np.atleast_1d(tk.Events.End).astype(int)
    for j in range(min(len(st),len(en))):
        a,b=st[j],en[j]
        if b-a<40: continue
        th=np.vstack([np.clip(np.deg2rad(sh[a:b]),*SHO),np.clip(np.deg2rad(el[a:b]),*ELB)])
        env_,arm=build(); dt=arm.dt; tt=np.arange(th.shape[1])/KF; tn=np.arange(0,tt[-1],dt)
        ths=np.vstack([np.interp(tn,tt,th[i]) for i in range(2)])
        if ths.shape[1]<20: continue
        xd=np.array([arm.skeleton.joint2cartesian(joint_state=np.concatenate([ths[:,k],[0,0]])[None,:])[0,:2] for k in range(ths.shape[1])])
        xv=np.gradient(xd,dt,axis=0); xa=np.gradient(xv,dt,axis=0)
        env_.reset(options={"joint_state":np.concatenate([ths[:,0],[0,0]])[None,:],"deterministic":True})
        logs=TargetReachSimulator(env_,arm,pdif(env_,arm),T(xd,xv,xa,dt),ths.shape[1]).run()
        act=logs.act_log[:logs.k]
        dur=(b-a)/KF; n=max(int(dur*FS),32)
        ea,eb=a*EF//KF,b*EF//KF
        for col,emk,lab in PAIRS:
            mo=rs(act[:,col],n); me=rs(envf(E[EMG[emk]][ea:eb]),n)
            mo_n,me_n=norm(mo),norm(me)
            res[lab]["r0"].append(np.corrcoef(mo_n,me_n)[0,1])
            xc=correlate(me_n,mo_n,"full")/len(mo_n)   # EMG vs model
            lags=np.arange(-len(mo_n)+1,len(mo_n)); maxlag=int(0.3*FS)  # +-300ms
            mid=np.abs(lags)<=maxlag
            k=np.argmax(xc[mid]); res[lab]["rmax"].append(xc[mid][k]); res[lab]["lag"].append(lags[mid][k]/FS*1000)
            f,Cxy=coherence(mo,me,fs=FS,nperseg=min(64,n)); band=(f>=0.3)&(f<=4)
            res[lab]["coh"].append(np.nanmean(Cxy[band]))
print("Muscle    zero-lag r |  max xcorr (lag ms) |  mean coherence(0.3-4Hz)")
for _,_,lab in PAIRS:
    R0=np.array(res[lab]["r0"]); RM=np.array(res[lab]["rmax"]); LG=np.array(res[lab]["lag"]); CO=np.array(res[lab]["coh"])
    print(f"  {lab:8s}  {np.nanmean(R0):+.2f}      |  {np.nanmean(RM):+.2f}  (lag {np.nanmedian(LG):+.0f} ms) |  {np.nanmean(CO):.2f}")
