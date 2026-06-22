import numpy as np, scipy.io as sio
from scipy.signal import butter, filtfilt, coherence, correlate, resample
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26
from controller.numpy.pd_if_controller import PDIFController, PDIFParams
from sim.numpy.simulator import TargetReachSimulator
B="/home/jdiab/Desktop/PAPER/enhanced_paper_corrected (Copy)"
SHOI,ELBI=3,6; EM={"antD":2,"postD":0,"bic":4,"tri":3}; SHO=np.deg2rad([0,140]);ELB=np.deg2rad([0,160]);FS=100.
def build():
    mu=RigidTendonHillMuscle(min_activation=0.02);arm=RigidTendonArm26(muscle=mu,timestep=0.002,damping=2.,n_ministeps=1,integration_method="Euler")
    return Environment(effector=arm,max_ep_duration=20.,action_noise=0.,obs_noise=0.,proprioception_delay=arm.dt,vision_delay=arm.dt,name="L"),arm
class T:
    def __init__(s,x,v,a,dt):s.x,s.v,s.a,s.dt=x,v,a,dt
    def sample(s,t):k=min(int(round(t/s.dt)),len(s.x)-1);return s.x[k],s.v[k],s.a[k]
b5,a5=butter(2,5/500,"low")
def envf(x):x=np.asarray(x,float)-np.median(x);return np.clip(filtfilt(b5,a5,np.abs(x)),0,None)
def pid(e,a):return PDIFController(e,a,PDIFParams(Kp_task=np.array([2400.,2400.]),damping_ratio=0.7,Kff=1.,use_critical_damping=True,enable_inertia_comp=True,enable_gravity_comp=False,enable_coriolis_comp=True,enable_nullspace=True,Kp_null=20.,Kd_null=5.,eps=1e-6,lam_os_max=200.,sigma_thresh=1e-4,gate_pow=2.,bisect_iters=12,enable_internal_force=False,cocon_level=0.))
def nz(x):x=x-x.mean();s=x.std();return x/s if s>0 else x
m=sio.loadmat(f"{B}/kinarm_dataset/raw_lucchetti/HS01.mat",struct_as_record=False,squeeze_me=True);D=m["s"].DataULdom;KF,EF=m["s"].KinFreq,m["s"].EmgFreq
P=[("antD","Shoulder flexor"),("postD","Shoulder extensor"),("bic","Biceps"),("tri","Triceps")]
NL=int(0.3*FS); lags=np.arange(-NL,NL+1)
XC={k:[] for k,_ in P}; CO={k:[] for k,_ in P}; FR=None; SIG={k:None for k,_ in P}
for tk in D:
    A=np.atleast_2d(tk.Angles);E=np.atleast_2d(tk.EMG);sh=A[SHOI];el=A[ELBI]
    if (sh.max()-sh.min())<40 or (el.max()-el.min())<40: continue
    st=np.atleast_1d(tk.Events.Start).astype(int);en=np.atleast_1d(tk.Events.End).astype(int)
    for j in range(min(len(st),len(en))):
        a,b=st[j],en[j]
        if b-a<40: continue
        th=np.vstack([np.clip(np.deg2rad(sh[a:b]),*SHO),np.clip(np.deg2rad(el[a:b]),*ELB)])
        env_,arm=build();dt=arm.dt;tt=np.arange(th.shape[1])/KF;tn=np.arange(0,tt[-1],dt)
        ths=np.vstack([np.interp(tn,tt,th[i]) for i in range(2)])
        if ths.shape[1]<20:continue
        xd=np.array([arm.skeleton.joint2cartesian(joint_state=np.concatenate([ths[:,k],[0,0]])[None,:])[0,:2] for k in range(ths.shape[1])])
        xv=np.gradient(xd,dt,axis=0);xa=np.gradient(xv,dt,axis=0)
        env_.reset(options={"joint_state":np.concatenate([ths[:,0],[0,0]])[None,:],"deterministic":True})
        logs=TargetReachSimulator(env_,arm,pid(env_,arm),T(xd,xv,xa,dt),ths.shape[1]).run();act=logs.act_log[:logs.k]
        dur=(b-a)/KF;n=max(int(dur*FS),48);ea,eb=a*EF//KF,b*EF//KF
        colmap={"antD":0,"postD":1,"bic":4,"tri":3}
        for k,_ in P:
            mo=resample(act[:,colmap[k]],n);me=resample(envf(E[EM[k]][ea:eb]),n)
            mn,en_=nz(mo),nz(me);xc=correlate(en_,mn,"full")/len(mn)
            L=np.arange(-len(mn)+1,len(mn));sel=(L>=-NL)&(L<=NL)
            if sel.sum()==len(lags): XC[k].append(xc[sel])
            f,C=coherence(mo,me,fs=FS,nperseg=min(48,n));CO[k].append(np.interp(np.linspace(0,10,40),f,C));FR=np.linspace(0,10,40)
            if SIG[k] is None: SIG[k]=(np.linspace(0,100,n),nz(mo),nz(me))
fig,ax=plt.subplots(3,4,figsize=(14,7.5))
for i,(k,lab) in enumerate(P):
    tphase,mo,me=SIG[k]
    ax[0,i].plot(tphase,me,"k",lw=1.6,label="EMG (norm)");ax[0,i].plot(tphase,mo,"r--",lw=1.6,label="Model act (norm)")
    ax[0,i].set_title(lab);ax[0,i].set_xlabel("reach %");ax[0,i].grid(ls="--",alpha=.3)
    xc=np.array(XC[k]).mean(0);ax[1,i].plot(lags/FS*1000,xc,"b");ax[1,i].axvline(0,color="0.6",ls=":")
    ax[1,i].set_title(f"cross-corr vs lag (peak {xc.max():+.2f})",fontsize=9);ax[1,i].set_xlabel("lag [ms]");ax[1,i].set_ylim(-0.5,1.0);ax[1,i].grid(ls="--",alpha=.3)
    co=np.array(CO[k]).mean(0);ax[2,i].plot(FR,co,"g");ax[2,i].axhline(0.6,color="0.6",ls=":",label="0.6")
    ax[2,i].set_title(f"coherence (mean {co[(FR>=.3)&(FR<=4)].mean():.2f})",fontsize=9);ax[2,i].set_xlabel("Hz");ax[2,i].set_ylim(0,1);ax[2,i].grid(ls="--",alpha=.3)
ax[0,0].legend(fontsize=8);ax[0,0].set_ylabel("normalised");ax[1,0].set_ylabel("cross-corr");ax[2,0].set_ylabel("coherence")
fig.suptitle("Frequency/lag analysis: model activation vs EMG (HS01). No sharp cross-corr peak, low coherence -> not a phase shift.",fontsize=11)
fig.tight_layout();fig.savefig(f"{B}/kinarm_dataset/emg_freq_diagnostic.png",dpi=105,bbox_inches="tight");print("wrote emg_freq_diagnostic.png")
