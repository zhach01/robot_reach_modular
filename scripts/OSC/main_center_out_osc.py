#!/usr/bin/env python3
# scripts/OSC/main_center_out_osc.py
# Operational-Space Control (OSC) center-out demo.

import numpy as np
import matplotlib.pyplot as plt

from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26

from config import PlantConfig, TrajectoryConfig, RunConfig

from tasks.numpy.base_task import CenterOutTask
from trajectory.numpy.minjerk import MinJerkLinearTrajectory, MinJerkParams
from sim.numpy.simulator import TargetReachSimulator
from controller.numpy.osc_controller import OSCController, OSCParams
from plotting.plots import plot_all, make_animations, hold_anims


def build_env(pc: PlantConfig):
    muscle = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(
        muscle=muscle, timestep=pc.timestep, damping=pc.damping,
        n_ministeps=pc.n_ministeps, integration_method=pc.integration_method,
    )
    env = Environment(
        effector=arm, max_ep_duration=pc.max_ep_duration, action_noise=0.0,
        obs_noise=0.0, proprioception_delay=arm.dt, vision_delay=arm.dt,
        name="CenterOutOSCEnv",
    )
    q0 = np.deg2rad(np.array(pc.q0_deg))
    qd0 = np.array(pc.qd0)
    env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})
    return env, arm, q0


def main():
    print("[CenterOut OSC] demo starting ...")
    pc = PlantConfig(); tc = TrajectoryConfig(); run = RunConfig()

    env, arm, q0 = build_env(pc)
    task = CenterOutTask(n_targets=4, radius=0.10)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(
        waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
    )

    p = OSCParams(Kp=4000.0, Kv=126.0, Kp_posture=50.0, Kv_posture=10.0)
    ctrl = OSCController(env, arm, p)
    steps = int(pc.max_ep_duration / arm.dt)

    sim = TargetReachSimulator(env, arm, ctrl, traj, steps)
    logs = sim.run()

    k, tvec = logs.time(arm.dt)
    plot_all(logs, tvec, center=task.center, targets=task.targets)
    if run.animate:
        anims = make_animations(logs, tvec, env, playback=run.playback,
                                downsample=run.downsample_anim,
                                center=task.center, targets=task.targets)
        hold_anims(anims)

    print("[CenterOut OSC] demo complete.")
    plt.show()


if __name__ == "__main__":
    main()
