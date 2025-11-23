#!/usr/bin/env python3
import numpy as np
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
from tasks.random_reach import Task as RandomReachTask
from trajectory.minjerk import MinJerkLinearTrajectory, MinJerkParams
from controller.sliding_mode import SlidingModeController, SlidingModeParams
from sim.simulator import TargetReachSimulator
from plotting.plots import plot_all, make_animations, hold_anims
import matplotlib.pyplot as plt


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
        name="RandomReachEnv(SMC)",
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


def _vec2(x, fallback=(1.0, 1.0)):
    v = np.array(x, dtype=float).ravel()
    if v.size == 2:
        return v
    if v.size == 1:
        return np.array([v[0], v[0]], dtype=float)
    return np.array(fallback, dtype=float)


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


    # --- Sliding-Mode gains (task-space = x,y) ---
    lambda_surf = np.array([26.0,26.0])   # np.array([16.0, 16.0]) slope of sliding surface on position error (e_x)
                                            # ↑ larger => stiffer position feedback, faster convergence,
                                            #             but more high-freq content (risk of chatter).
                                            # ↓ smaller => softer, slower, smoother.

    K_switch    = np.array([4.0, 4.0])    # amplitude of switching/robust term
                                            # ↑ larger => stronger “push” against disturbances/model errors,
                                            #             quicker error kill, but more torque spikes & allocator load.
                                            # ↓ smaller => gentler, less chatter, may leave small steady ripples.

    phi         = np.array([0.004, 0.004])    # boundary layer half-width for sat(s/phi)
                                            # ↑ larger => wider linear zone → smoother, less chatter, but more
                                            #             steady error near target.
                                            # ↓ smaller => crisper tracking, but risk of buzz/sawtooth torques.

    # --- Nullspace posture (joint space). We keep it OFF for pure tracking ---
    Kp_q        = np.array([0.0, 0.0])      # posture proportional gain
    Kd_q        = np.array([0.0, 0.0])      # posture damping gain
                                            # If you enable posture: 
                                            # ↑ gains => joints get pulled to qref stronger/faster; can fight the task.
                                            # ↓ gains => weaker posture influence; safer for tracking.        

    # --- Build params (kept 1:1 with your config objects) ---
    p = SlidingModeParams(
        lambda_surf=lambda_surf,             # see above
        K_switch=K_switch,                   # see above
        phi=phi,                             # see above

        Kff_x=_vec2(gains.Kff_x, (1.0, 1.0)),# feedforward on desired ẍ
                                            # ↑ larger => follows the reference acceleration more; can reduce lag,
                                            #             but if dynamics are mismatched it can excite oscillations.
                                            # ↓ smaller => relies more on feedback; more robust, possibly slower.

        Kp_q=_vec2(Kp_q, (0.0, 0.0)),        # posture P (we pass our explicit values)
        Kd_q=_vec2(Kd_q, (0.0, 0.0)),        # posture D

        # --- Guards / gates (stability near singularities & poor conditioning) ---
        eps=num.eps,                         # small positive for numeric safety in divisions/LS solves
                                                # ↑ larger => more conservative numerics; ↓ smaller => more precise but risk NaNs.
        lam_os_max=num.lam_os_max,           # cap on operational-space regularization
                                            # ↑ larger => adds more damping near singular J; safer but less authority.
        sigma_thresh=num.sigma_thresh,       # threshold for “danger zone” in S/J conditioning
                                            # ↑ larger => earlier gating (more conservative); ↓ smaller => later gating.
        gate_pow=num.gate_pow,               # how aggressively the gate fades control near singularity
                                            # ↑ larger => gate drops faster; ↓ smaller => gate drops gently.
    
        # --- Plant compensation toggles (enable all for best tracking) ---
        enable_internal_force=toggles.enable_internal_force,  # keep False for pure tracking (frees capacity)
        enable_inertia_comp=toggles.enable_inertia_comp,       # True: better authority; False: more lag
        enable_gravity_comp=toggles.enable_gravity_comp,       # True: removes load bias
        enable_velocity_comp=toggles.enable_velocity_comp,     # True: cancels Coriolis/centrifugal
        enable_joint_damping=toggles.enable_joint_damping,     # True: adds viscous joint damping in h term

        # --- Allocation / inversion settings ---
        cocon_a0=ifc.cocon_a0,               # baseline co-contraction (active force bias)
                                            # ↑ larger => stiffer feel, higher effort; ↓ smaller => freer, less robust.
        bisect_iters=ifc.bisect_iters,       # activation-from-force solver iterations
                                            # ↑ larger => more accurate a(F) but slower.
        linesearch_eps=num.linesearch_eps,   # internal line search tolerance (muscle IF regulation)
        linesearch_safety=num.linesearch_safety, # step shrink factor (0..1): smaller => safer but slower
    )




    ctrl = SlidingModeController(env, arm, p)
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
    print("Random reach (Sliding-Mode) complete.")
    plt.show()


if __name__ == "__main__":
    main()
