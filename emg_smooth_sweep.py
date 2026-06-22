import numpy as np, scipy.io as sio
from scipy.signal import butter, filtfilt
from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26
from controller.numpy.pd_if_controller import PDIFController, PDIFParams
from sim.numpy.simulator import TargetReachSimulator
B="/home/jdiab/Desktop/PAPER/enhanced_paper_corrected (Copy)"
SHOI,ELBI=3,6; SHO=np.deg2rad([0,140]);ELB=np.deg2rad([0,160]);N=100;ph=np.linspace(0,1,N)
def build():
    mu=RigidTendonHillMuscle(min_activation=0.02);arm=RigidTendonArm26(muscle=mu,timestep=0.002,damping=2.,n_ministeps=1,integration_method="Euler")
    return Environment(effector=arm,max_ep_duration=20.,action_noise=0.,obs_noise=0.,proprioception_delay=arm.dt,vision_delay=arm.dt,name="L"),arm
class T:
    def __init__(s,x,v,a,dt):s.x,s.v,s.a,s.dt=x,v,a,dt
    def sample(s,t):k=min(int(round(t/s.dt)),len(s.x)-1);return s.x[k],s.v[k],s.a[k]
b5,a5=butter(2,5/500,"low")
def envf(x):x=np.asarray(x,float)-np.median(x);return np.clip(filtfilt(b5,a5,np.abs(x)),0,None)
def pid(e,a):return PDIFController(e,a,PDIFParams(Kp_task=np.array([2400.,2400.]),damping_ratio=0.7,Kff=1.,use_critical_damping=True,enable_inertia_comp=True,enable_gravity_comp=False,enable_coriolis_comp=True,enable_nullspace=True,Kp_null=20.,Kd_null=5.,eps=1e-6,lam_os_max=200.,sigma_thresh=1e-4,gate_pow=2.,bisect_iters=12,enable_internal_force=True,cocon_level=0.05))
def actdyn(u,dt,ta=0.015,td=0.05):  # 1st-order activation dynamics
    a=np.zeros_like(u); a[0]=u[0]
    for k in range(1,len(u)):
        tau=ta if u[k]>=a[k-1] else td
        a[k]=a[k-1]+dt*(u[k]-a[k-1])/tau
    return a
m=sio.loadmat(f"{B}/kinarm_dataset/raw_lucchetti/HS01.mat",struct_as_record=False,squeeze_me=True);D=m["s"].DataULdom;KF,EF=m["s"].KinFreq,m["s"].EmgFreq
PAIRS=[(0,2,"ShoFlex"),(1,0,"ShoExt"),(4,4,"Biceps"),(3,3,"Triceps")]
# accumulate raw model act + emg per reach once
RAW={p[2]:([],[]) for p in PAIRS}; DT=0.002
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
        ea,eb=a*EF//KF,b*EF//KF;te=np.linspace(0,1,eb-ea)
        for col,row,lab in PAIRS:
            RAW[lab][0].append(act[:,col]);RAW[lab][1].append(np.interp(np.linspace(0,1,len(act[:,col])),te,envf(E[row][ea:eb])))
def meanprof(transform):
    out={}
    for _,_,lab in PAIRS:
        Ms=[];Es=[]
        for mo,me in zip(*RAW[lab]):
            mt=transform(mo); mt=np.interp(ph,np.linspace(0,1,len(mt)),mt); mt=mt/(mt.max() or 1)
            mee=np.interp(ph,np.linspace(0,1,len(me)),me); mee=mee/(mee.max() or 1)
            Ms.append(mt);Es.append(mee)
        out[lab]=np.corrcoef(np.array(Ms).mean(0),np.array(Es).mean(0))[0,1]
    return out
def lp(fc):
    bb,aa=butter(2,fc/(0.5/DT),"low"); return lambda x: np.clip(filtfilt(bb,aa,x),0,None)
variants={"raw":lambda x:x,"lp5Hz":lp(5),"lp3Hz":lp(3),"actdyn":lambda x:actdyn(x,DT)}
print("smoothing | ShoFlex ShoExt Biceps Triceps | mean   (all at cocon=0.05)")
for name,tr in variants.items():
    r=meanprof(tr);mn=np.mean(list(r.values()))
    print(f"{name:9s} |  {r['ShoFlex']:+.2f}   {r['ShoExt']:+.2f}  {r['Biceps']:+.2f}  {r['Triceps']:+.2f}  | {mn:+.2f}")
