#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt
from model_lib.environment_numpy import Environment
from model_lib.muscles_numpy import RigidTendonHillMuscle
from model_lib.effector_numpy import RigidTendonArm26
from config import PlantConfig, ControlToggles, Numerics, TrajectoryConfig, RunConfig
from tasks.center_out import Task as CenterOutTask
from trajectory.minjerk import MinJerkLinearTrajectory, MinJerkParams
from controller.nmpc_task import NonlinearMPCController, NMPCParams
from sim.simulator import TargetReachSimulator
from plotting.plots import plot_all, make_animations, hold_anims

def build_env(pc):
    muscle = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(muscle=muscle, timestep=pc.timestep, damping=pc.damping,
                           n_ministeps=pc.n_ministeps, integration_method=pc.integration_method)
    env = Environment(effector=arm, max_ep_duration=pc.max_ep_duration,
                      action_noise=0.0, obs_noise=0.0,
                      proprioception_delay=arm.dt, vision_delay=arm.dt,
                      name="CenterOutEnv(NMPC)")
    q0 = np.deg2rad(np.array(pc.q0_deg)); qd0 = np.array(pc.qd0)
    env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})
    return env, arm

def main():
    pc, tc, run = PlantConfig(), TrajectoryConfig(), RunConfig()
    env, arm = build_env(pc)
    task = CenterOutTask(n_targets=4, radius=0.10)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale))
    
    p = NMPCParams(N=12, dt_mpc=arm.dt, use_muscles=True)  # no muscles
    ctrl = NonlinearMPCController(env, arm, p)
    steps = int(pc.max_ep_duration / arm.dt)
    logs = TargetReachSimulator(env, arm, ctrl, traj, steps).run()

    k, tvec = logs.time(arm.dt)
    plot_all(logs, tvec, center=task.center, targets=task.targets)
    if run.animate:
        anims = make_animations(logs, tvec, env, playback=run.playback,
                                downsample=run.downsample_anim, center=task.center, targets=task.targets)
        hold_anims(anims)
    print("Center-out (NMPC) complete."); plt.show()

if __name__ == "__main__":
    main()

