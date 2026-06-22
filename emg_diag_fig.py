import numpy as np, scipy.io as sio
from scipy.signal import butter, filtfilt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26
from controller.numpy.pd_if_controller import PDIFController, PDIFParams
from sim.numpy.simulator import TargetReachSimulator
BASE="/home/jdiab/Desktop/PAPER/enhanced_paper_corrected (Copy)"
SHOI,ELBI=3,6; EMG={"postD":0,"antD":2,"tri":3,"bic":4}
SHO=np.deg2rad([0,140]); ELB=np.deg2rad([0,160]); N=100; ph=np.linspace(0,1,N)
def build():
    mu=RigidTendonHillMuscle(min_activation=0.02)
    arm=RigidTendonArm26(muscle=mu,timestep=0.002,damping=2.0,n_ministeps=1,integration_method="Euler")
    env=Environment(effector=arm,max_ep_duration=20.,action_noise=0.,obs_noise=0.,proprioception_delay=arm.dt,vision_delay=arm.dt,name="L")
    return env,arm
class T:
    def __init__(s,x,v,a,dt):s.x,s.v,s.a,s.dt=x,v,a,dt
    def sample(s,t):k=min(int(round(t/s.dt)),len(s.x)-1);return s.x[k],s.v[k],s.a[k]
b5,a5=butter(2,5/500,"low")
def envf(x): x=np.asarray(x,float)-np.median(x); return np.clip(filtfilt(b5,a5,np.abs(x)),0,None)
def pdif(env,arm): return PDIFController(env,arm,PDIFParams(Kp_task=np.array([2400.,2400.]),damping_ratio=0.7,Kff=1.,use_critical_damping=True,enable_inertia_comp=True,enable_gravity_comp=False,enable_coriolis_comp=True,enable_nullspace=True,Kp_null=20.,Kd_null=5.,eps=1e-6,lam_os_max=200.,sigma_thresh=1e-4,gate_pow=2.,bisect_iters=12,enable_internal_force=False,cocon_level=0.))
m=sio.loadmat(f"{BASE}/kinarm_dataset/raw_lucchetti/HS01.mat",struct_as_record=False,squeeze_me=True)
D=m["s"].DataULdom; KF,EF=m["s"].KinFreq,m["s"].EmgFreq
PAIRS=[(0,"antD","Shoulder flexor\n(model pec. vs ant. deltoid EMG)"),(1,"postD","Shoulder extensor\n(model delt. vs post. deltoid EMG)"),
       (4,"bic","Biceps / flexor"),(3,"tri","Triceps / extensor")]
acc={lab:{"mod":[],"emg":[]} for _,_,lab in PAIRS}; shp=[]; elp=[]
for tk in D:
    A=np.atleast_2d(tk.Angles); E=np.atleast_2d(tk.EMG); sh=A[SHOI]; el=A[ELBI]
    if (sh.max()-sh.min())<40 or (el.max()-el.min())<40: continue
    st=np.atleast_1d(tk.Events.Start).astype(int); en=np.atleast_1d(tk.Events.End).astype(int)
    for j in range(min(len(st),len(en))):
        a,b=st[j],en[j]
        if b-a<30: continue
        th=np.vstack([np.clip(np.deg2rad(sh[a:b]),*SHO),np.clip(np.deg2rad(el[a:b]),*ELB)])
        env_,arm=build(); dt=arm.dt
        tt=np.arange(th.shape[1])/KF; tn=np.arange(0,tt[-1],dt)
        ths=np.vstack([np.interp(tn,tt,th[i]) for i in range(2)])
        if ths.shape[1]<20: continue
        xd=np.array([arm.skeleton.joint2cartesian(joint_state=np.concatenate([ths[:,k],[0,0]])[None,:])[0,:2] for k in range(ths.shape[1])])
        xv=np.gradient(xd,dt,axis=0); xa=np.gradient(xv,dt,axis=0)
        env_.reset(options={"joint_state":np.concatenate([ths[:,0],[0,0]])[None,:],"deterministic":True})
        logs=TargetReachSimulator(env_,arm,pdif(env_,arm),T(xd,xv,xa,dt),ths.shape[1]).run()
        act=logs.act_log[:logs.k]; tm=np.linspace(0,1,act.shape[0])
        shp.append(np.interp(ph,np.linspace(0,1,b-a),np.rad2deg(th[0]))); elp.append(np.interp(ph,np.linspace(0,1,b-a),np.rad2deg(th[1])))
        ea,eb=a*EF//KF,b*EF//KF; te=np.linspace(0,1,eb-ea)
        for col,emk,lab in PAIRS:
            pa=np.interp(ph,tm,act[:,col]); pa=pa/(pa.max() or 1)
            me=np.interp(ph,te,envf(E[EMG[emk]][ea:eb])); me=me/(me.max() or 1)
            acc[lab]["mod"].append(pa); acc[lab]["emg"].append(me)
plt.rcParams.update({"font.size":10})
fig,ax=plt.subplots(1,5,figsize=(15,3.2))
ax[0].plot(ph*100,np.mean(shp,0),label="Shoulder flex",color="tab:blue")
ax[0].plot(ph*100,np.mean(elp,0),label="Elbow flex",color="tab:orange")
ax[0].set_title("Measured joint angles\n(model input)"); ax[0].set_ylabel("deg"); ax[0].legend(fontsize=8); ax[0].grid(ls="--",alpha=.3)
for i,(col,emk,lab) in enumerate(PAIRS):
    M=np.array(acc[lab]["mod"]); EE=np.array(acc[lab]["emg"])
    r=np.corrcoef(M.mean(0),EE.mean(0))[0,1]
    a=ax[i+1]
    a.plot(ph*100,EE.mean(0),"k",lw=2,label="Measured EMG")
    a.fill_between(ph*100,EE.mean(0)-EE.std(0),EE.mean(0)+EE.std(0),color="k",alpha=.12)
    a.plot(ph*100,M.mean(0),"r--",lw=2,label="Model activation")
    a.fill_between(ph*100,M.mean(0)-M.std(0),M.mean(0)+M.std(0),color="r",alpha=.12)
    a.set_title(f"{lab}\nr={r:+.2f}",fontsize=9); a.set_ylim(-0.1,1.2); a.grid(ls="--",alpha=.3); a.set_xlabel("reach %")
ax[1].legend(fontsize=8,loc="upper right")
fig.suptitle("Why activation↔EMG correlation is low: model (effort-minimising, planar) vs measured EMG (3D task, co-contraction). HS01, mean±SD over reaches.",fontsize=10)
fig.tight_layout()
fig.savefig(f"{BASE}/kinarm_dataset/emg_diagnostic.png",dpi=110,bbox_inches="tight")
print("wrote emg_diagnostic.png")
