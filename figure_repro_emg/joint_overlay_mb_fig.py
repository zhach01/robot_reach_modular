import numpy as np, pandas as pd
import model_lib.numpy.skeleton as sk; sk.USE_CACHE=False
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter, butter, filtfilt
from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26
from sim.numpy.simulator import TargetReachSimulator
from controller.numpy.pd_if_controller import PDIFController, PDIFParams
from controller.numpy.energy_tank_controller import EnergyTankController, EnergyTankParams
from controller.numpy.sliding_mode import SlidingModeController, SlidingModeParams
from controller.numpy.osc_controller import OSCController, OSCParams
BASE="/home/jdiab/Desktop/PAPER/enhanced_paper_corrected (Copy)"
OUT=f"{BASE}/overleaf_model_based/figures/joint_overlay_mb.pdf"
df=pd.read_csv(f"{BASE}/kinarm_dataset/kinarm_reaching_multisession.csv"); SHO=np.deg2rad([0,140]); ELB=np.deg2rad([0,160])
def build():
    m=RigidTendonHillMuscle(min_activation=0.02); arm=RigidTendonArm26(muscle=m,timestep=0.002,damping=2.0,n_ministeps=1,integration_method="Euler")
    return Environment(effector=arm,max_ep_duration=30.0,action_noise=0.,obs_noise=0.,proprioception_delay=arm.dt,vision_delay=arm.dt,name="K"),arm
def mk(c,env,arm):
    if c=="Impedance (PD+IF)": return PDIFController(env,arm,PDIFParams(Kp_task=np.array([2400.,2400.]),damping_ratio=0.7,Kff=1.0,use_critical_damping=True,enable_inertia_comp=True,enable_gravity_comp=False,enable_coriolis_comp=True,enable_nullspace=True,Kp_null=20.,Kd_null=5.,eps=1e-6,lam_os_max=200.,sigma_thresh=1e-4,gate_pow=2.,bisect_iters=12,enable_internal_force=False,cocon_level=0.))
    if c=="Passivity": return EnergyTankController(env,arm,EnergyTankParams(D0=np.diag([25.,25.]),K0=np.diag([800.,800.]),enable_inertia_comp=True,enable_gravity_comp=False,enable_joint_damping=True,enable_internal_force=False,cocon_a0=0.0,bisect_iters=16,linesearch_eps=1e-6,linesearch_safety=0.99,E0=0.15,Emin=1e-4,Emax=1.0))
    if c=="Sliding-mode": return SlidingModeController(env,arm,SlidingModeParams(lambda_surf=np.array([18.,18.]),K_switch=np.array([18.,18.]),phi=np.array([0.075,0.075]),Kff_x=1.0,enable_inertia_comp=True,enable_gravity_comp=False,enable_velocity_comp=True,bisect_iters=16))
    if c=="OSC": return OSCController(env,arm,OSCParams(Kp=4000.,Kv=126.,Kp_posture=50.,Kv_posture=10.))
class Traj:
    def __init__(s,x,v,a,dt): s.x,s.v,s.a,s.dt=x,v,a,dt
    def sample(s,t): k=min(int(round(t/s.dt)),len(s.x)-1); return s.x[k],s.v[k],s.a[k]
# representative reach = most steps
best=None
for (sess,gt),g in df.groupby(["session","gtrial"]):
    n=len(g)//2
    if best is None or n>best[0]: best=(n,sess,gt,g)
_,sess,gt,g=best; g=g.reset_index(drop=True)
th=np.vstack([g.shoulder_angle_rad.values,g.elbow_angle_rad.values])[:,::2]
w=51; ths=np.vstack([savgol_filter(th[i],w,3) for i in range(2)]); ths[0]=np.clip(ths[0],*SHO); ths[1]=np.clip(ths[1],*ELB)
e0,a0=build(); dt=a0.dt
xd=np.array([a0.skeleton.joint2cartesian(joint_state=np.concatenate([ths[:,k],[0,0]])[None,:])[0,:2] for k in range(ths.shape[1])])
xv=np.gradient(xd,dt,axis=0); xa=np.gradient(xv,dt,axis=0); steps=ths.shape[1]; t=np.arange(steps)*dt
order=["Impedance (PD+IF)","Passivity","Sliding-mode","OSC"]; cols={"Impedance (PD+IF)":"tab:red","Passivity":"tab:cyan","Sliding-mode":"tab:green","OSC":"tab:purple"}
taus={}; qrep={}
for c in order:
    env,arm=build(); env.reset(options={"joint_state":np.concatenate([ths[:,0],[0.,0.]])[None,:],"deterministic":True})
    logs=TargetReachSimulator(env,arm,mk(c,env,arm),Traj(xd,xv,xa,dt),steps).run(); k=logs.k
    taus[c]=logs.tau_real_log[:k,0]; qrep[c]=logs.q_log[:k]
b,a=butter(2,4/(0.5/dt),"low")  # 4 Hz low-pass for the smoothed (physiological) torque profile
plt.rcParams.update({"font.size":11,"legend.fontsize":8})
fig=plt.figure(figsize=(7.4,6.6)); gs=gridspec.GridSpec(3,2,height_ratios=[1.15,1,1],hspace=0.62,wspace=0.28)
axA=fig.add_subplot(gs[0,:])
axA.plot(t,np.degrees(ths[0]),color="tab:blue",lw=2.0,label="Shoulder (measured)")
axA.plot(t,np.degrees(ths[1]),color="tab:orange",lw=2.0,label="Elbow (measured)")
axA.set_ylabel("Joint angle [deg]"); axA.grid(True,ls="--",alpha=.3); axA.legend(frameon=False,ncol=2,loc="upper right")
axA.set_title("KINARM reaching replayed through Arm-26 (representative reach)")
ymax=max(np.abs(taus[c]).max() for c in order)*1.1
for i,c in enumerate(order):
    ax=fig.add_subplot(gs[1+i//2,i%2]); tau=taus[c]; ts=filtfilt(b,a,tau)
    ax.plot(t,tau,color=cols[c],alpha=0.22,lw=0.6)
    ax.plot(t,ts,color=cols[c],lw=1.8)
    ax.axhline(0,color="k",lw=0.5,alpha=.4); ax.set_ylim(-ymax,ymax)
    ax.set_title(c,fontsize=10)
    ax.grid(True,ls="--",alpha=.3)
    if i%2==0: ax.set_ylabel("Shoulder $\\tau$ [N·m]")
    if i//2==1: ax.set_xlabel("Time [s]")
fig.suptitle("",y=0.995)
fig.tight_layout(); fig.savefig(OUT,bbox_inches="tight"); print("wrote",OUT,"| peaks:",{c:round(float(np.abs(taus[c]).max()),1) for c in order})
