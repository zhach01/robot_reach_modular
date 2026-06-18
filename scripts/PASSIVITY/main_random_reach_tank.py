# -------- main_random_reach_tank.py --------
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from model_lib.numpy.environment import Environment
from model_lib.numpy.muscles import RigidTendonHillMuscle
from model_lib.numpy.effector import RigidTendonArm26
from config import (
    PlantConfig,
    ControlToggles,
    ControlGains,
    Numerics,
    InternalForceConfig,
    TrajectoryConfig,
    RunConfig,
)
from tasks.numpy.random_reach import Task as RandomReachTask
from trajectory.numpy.minjerk import MinJerkLinearTrajectory, MinJerkParams
from controller.numpy.energy_tank_controller import EnergyTankController, EnergyTankParams
from sim.numpy.simulator import TargetReachSimulator
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
        name="RandomReachTankEnv",
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

    task = RandomReachTask(n_points=1, radius=0.10, seed=0)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(
        waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale)
    )

    # Tight path-following (same tuning as center-out): match_torch=False applies
    # the inverse-dynamics feedforward ungated, and the stiffer task gain +
    # feedforward (K0=4000, Kff=2.0, critical zeta) keep the actual endpoint on
    # the straight reference line instead of bowing ~5-8 mm off it. RMSE 2.17 -> 1.05 mm.
    p = EnergyTankParams(
        D0=np.diag([25.0, 25.0]),
        K0=np.diag([4000.0, 4000.0]),
        Kff=2.0,
        zeta=1.0,
        match_torch=False,
        KI=np.array([0.0, 0.0]),
        Imax=np.array([0.0, 0.0]),
        eps=num.eps,
        lam_os_max=num.lam_os_max,
        sigma_thresh=num.sigma_thresh,
        gate_pow=num.gate_pow,
        enable_inertia_comp=toggles.enable_inertia_comp,
        enable_gravity_comp=toggles.enable_gravity_comp,
        enable_joint_damping=toggles.enable_joint_damping,
        enable_internal_force=False,
        cocon_a0=0,
        bisect_iters=ifc.bisect_iters,
        linesearch_eps=num.linesearch_eps,
        linesearch_safety=num.linesearch_safety,
        E0=0.15,
        Emin=1e-4,
        Emax=1,

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
    print("Random reach (Energy Tank) demo complete.")
    plt.show()


if __name__ == "__main__":
    main()
