# -------- main_center_out_tank.py --------

#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from model_lib.environment_numpy import Environment
from model_lib.muscles_numpy import RigidTendonHillMuscle
from model_lib.effector_numpy import RigidTendonArm26
from config import (
    PlantConfig,
    ControlToggles,
    ControlGains,
    Numerics,
    InternalForceConfig,
    TrajectoryConfig,
    RunConfig,
)
from tasks.center_out import Task as CenterOutTask
from trajectory.minjerk import MinJerkLinearTrajectory, MinJerkParams
from controller.energy_tank_controller import EnergyTankController, EnergyTankParams
from sim.simulator import TargetReachSimulator
from plotting.plots import plot_all, make_animations, hold_anims


def build_env(pc: PlantConfig):
    muscle = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(
        muscle=muscle,
        timestep=pc.timestep,
        damping=pc.damping,
        n_ministeps=pc.n_ministeps,
        integration_method=pc.integration_method,
    )
    env = Environment(
        effector=arm,
        max_ep_duration=pc.max_ep_duration,
        action_noise=0.0,
        obs_noise=0.0,
        proprioception_delay=arm.dt,
        vision_delay=arm.dt,
        name="CenterOutTankEnv",
    )
    q0 = np.deg2rad(np.array(pc.q0_deg))
    qd0 = np.array(pc.qd0)
    env.reset(
        options={
            "joint_state": np.concatenate([q0, qd0])[None, :],
            "deterministic": True,
        }
    )
    return env, arm, q0


def main():
    pc = PlantConfig()
    toggles = ControlToggles()
    gains = ControlGains()
    num = Numerics()
    ifc = InternalForceConfig()
    tc = TrajectoryConfig()
    run = RunConfig()
    env, arm, q0 = build_env(pc)

    task = CenterOutTask(n_targets=4, radius=0.10)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(
        waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
    )

    p = EnergyTankParams(
        D0=np.diag([15.0, 15.0]),
        K0=np.diag(gains.Kp_x),  # reuse stiffness magnitudes from config
        KI=np.array([80.0, 80.0]),
        Imax=np.array([0.03, 0.03]),
        eps=num.eps,
        lam_os_smin_target=num.lam_os_smin_target,
        lam_os_max=num.lam_os_max,
        sigma_thresh=num.sigma_thresh,
        gate_pow=num.gate_pow,
        enable_inertia_comp=toggles.enable_inertia_comp,
        enable_gravity_comp=toggles.enable_gravity_comp,
        enable_velocity_comp=toggles.enable_velocity_comp,
        enable_joint_damping=toggles.enable_joint_damping,
        enable_internal_force=toggles.enable_internal_force,
        cocon_a0=ifc.cocon_a0,
        bisect_iters=ifc.bisect_iters,
        linesearch_eps=num.linesearch_eps,
        linesearch_safety=num.linesearch_safety,
        E0=0.08,
        Emin=1e-4,
        Emax=0.5,
    )
    ctrl = EnergyTankController(env, arm, p)
    steps = int(pc.max_ep_duration / arm.dt)

    sim = TargetReachSimulator(env, arm, ctrl, traj, steps)
    logs = sim.run()

    k, tvec = logs.time(arm.dt)
    plot_all(logs, tvec, center=task.center, targets=task.targets)
    if run.animate:
        anims = make_animations(
            logs,
            tvec,
            env,
            playback=run.playback,
            downsample=run.downsample_anim,
            center=task.center,
            targets=task.targets,
        )
        hold_anims(anims)
    print("Center-out (Energy Tank) demo complete.")
    plt.show()


if __name__ == "__main__":
    main()
