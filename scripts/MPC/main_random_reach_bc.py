#!/usr/bin/env python3
import argparse, numpy as np, matplotlib.pyplot as plt
from model_lib.environment_numpy import Environment
from model_lib.muscles_numpy import RigidTendonHillMuscle
from model_lib.effector_numpy import RigidTendonArm26
from config import PlantConfig, TrajectoryConfig, RunConfig
from tasks.random_reach import Task as RandomReachTask
from trajectory.minjerk import MinJerkLinearTrajectory, MinJerkParams
from controller.hybrid_mpc_rl import RLPolicy, RLPolicyParams, RLController, RLControllerParams
from sim.simulator import TargetReachSimulator
from plotting.plots import plot_all, make_animations, hold_anims

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--goals", type=int, default=6)
    p.add_argument("--radius", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hidden", type=str, default="256,256")  # ignored if ckpt differs; kept for safety
    p.add_argument("--no_muscles", action="store_true")
    p.add_argument("--animate", action="store_true")
    return p.parse_args()

def build_env(pc: PlantConfig):
    muscle = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(muscle=muscle, timestep=pc.timestep, damping=pc.damping,
                           n_ministeps=pc.n_ministeps, integration_method=pc.integration_method)
    env = Environment(effector=arm, max_ep_duration=pc.max_ep_duration,
                      action_noise=0.0, obs_noise=0.0,
                      proprioception_delay=arm.dt, vision_delay=arm.dt,
                      name="RandomReach(BC-policy)")
    q0 = np.deg2rad(np.array(pc.q0_deg)); qd0 = np.array(pc.qd0)
    env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})
    return env, arm

def main():
    args = parse()
    pc, tc, run = PlantConfig(), TrajectoryConfig(), RunConfig()
    env, arm = build_env(pc)

    # targets & trajectory
    task = RandomReachTask(n_points=args.goals, radius=args.radius, seed=args.seed)
    try:
        waypoints = task.build_waypoints(env, reach_T=1.2, settle_T=0.3)
    except TypeError:
        waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale))

    # policy
    hidden = tuple(int(x) for x in args.hidden.split(",") if x)
    ppi = RLPolicyParams(obs_dim=14, hidden=hidden, device=args.device)  # ckpt may override
    pi = RLPolicy(ppi); pi.load(args.ckpt)

    # controller
    pctrl = RLControllerParams(tau_clip=600.0, use_muscles=(not args.no_muscles))
    ctrl = RLController(env, arm, pi, pctrl)

    steps = int(pc.max_ep_duration / arm.dt)
    logs  = TargetReachSimulator(env, arm, ctrl, traj, steps).run()

    k, tvec = logs.time(arm.dt)
    plot_all(logs, tvec, center=task.center, targets=task.targets)
    if args.animate or run.animate:
        anims = make_animations(logs, tvec, env, playback=run.playback,
                                downsample=run.downsample_anim, center=task.center, targets=task.targets)
        hold_anims(anims)
    print("Random reach (BC policy) complete."); plt.show()

if __name__ == "__main__":
    main()
