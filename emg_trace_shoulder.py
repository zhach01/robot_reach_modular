import numpy as np, scipy.io as sio
from scipy.signal import butter, filtfilt, correlate, resample
from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26
from controller.numpy.pd_if_controller import PDIFController, PDIFParams
from sim.numpy.simulator import TargetReachSimulator
B="/home/jdiab/Desktop/PAPER/enhanced_paper_corrected (Copy)"
SHOI,ELBI=3,6; SHO=np.deg2rad([0,140]);ELB=np.deg2rad([0,160]);FS=100.
# EMG channel indices: 0 postD,1 midD,2 antD,3 tri,4 bic
def build():
    mu=RigidTendonHillMuscle(min_activation=0.02);arm=RigidTendonArm26(muscle=mu,timestep=0.002,damping=2.,n_ministeps=1,integration_method="Euler")
    return Environment(effector=arm,max_ep_duration=20.,action_noise=0.,obs_noise=0.,proprioception_delay=arm.dt,vision_delay=arm.dt,name="L"),arm
class T:
    def __init__(s,x,v,a,dt):s.x,s.v,s.a,s.dt=x,v,a,dt
    def sample(s,t):k=min(int(round(t/s.dt)),len(s.x)-1);return s.x[k],s.v[k],s.a[k]
def pid(e,a):return PDIFController(e,a,PDIFParams(Kp_task=np.array([2400.,2400.]),damping_ratio=0.7,Kff=1.,use_critical_damping=True,enable_inertia_comp=True,enable_gravity_comp=False,enable_coriolis_comp=True,enable_nullspace=True,Kp_null=20.,Kd_null=5.,eps=1e-6,lam_os_max=200.,sigma_thresh=1e-4,gate_pow=2.,bisect_iters=12,enable_internal_force=False,cocon_level=0.))
m=sio.loadmat(f"{B}/kinarm_dataset/raw_lucchetti/HS01.mat",struct_as_record=False,squeeze_me=True);D=m["s"].DataULdom;KF,EF=m["s"].KinFreq,m["s"].EmgFreq
# run sims ONCE; store model act (6) + raw EMG windows (1000Hz) per reach
REACH=[]
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
        logs=TargetReachSimulator(env_,arm,pid(env_,arm),T(xd,xv,xa,dt),ths.shape[1]).run()
        ea,eb=a*EF//KF,b*EF//KF
        REACH.append((logs.act_log[:logs.k], E[:, ea:eb]))
print("reaches:",len(REACH))
def nz(x):x=x-x.mean();s=x.std();return x/s if s>0 else x
def envf(x,fc):
    bb,aa=butter(2,fc/500,"low");x=np.asarray(x,float)-np.median(x);return np.clip(filtfilt(bb,aa,np.abs(x)),0,None)
def score(modcols, emgrows, fc, use_maxxc):
    rs=[]
    for act,Ew in REACH:
        n=max(int(act.shape[0]*0.5),48)
        mo=resample(act[:,modcols].sum(1),n)
        me=resample(envf(Ew[emgrows].sum(0),fc),n)
        mn,en_=nz(mo),nz(me)
        if use_maxxc:
            xc=correlate(en_,mn,"full")/len(mn);NL=int(0.3*FS);L=np.arange(-len(mn)+1,len(mn));sel=(L>=-NL)&(L<=NL);rs.append(xc[sel].max())
        else: rs.append(np.corrcoef(mn,en_)[0,1])
    return np.nanmean(rs)
print("\n=== SHOULDER FLEXOR trace (model col vs EMG ch, cutoff, metric) ===")
variants=[
 ("pec(0) vs antD",[0],[2]),("delt(1) vs antD",[1],[2]),("pec+delt(0,1) vs antD",[0,1],[2]),
 ("pec(0) vs antD+midD",[0],[2,1]),("pec(0) vs postD",[0],[0]),("delt(1) vs postD",[1],[0]),
 ("pec+delt vs all-delt(0,1,2)",[0,1],[0,1,2]),
]
print(f"{'variant':32s} {'r@2Hz':>7} {'r@5Hz':>7} {'r@10Hz':>7} {'maxXC@5Hz':>10}")
for name,mc,er in variants:
    print(f"{name:32s} {score(mc,er,2,0):+7.2f} {score(mc,er,5,0):+7.2f} {score(mc,er,10,0):+7.2f} {score(mc,er,5,1):+10.2f}")
