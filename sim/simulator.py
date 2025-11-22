# sim/simulator.py
import numpy as np
from logging_tools.log_buffer import LogBuffer


class TargetReachSimulator:
    """Generic target-reaching simulator: controller + trajectory."""

    def __init__(self, env, arm, controller, trajectory, steps):
        self.env = env
        self.arm = arm
        self.controller = controller
        self.trajectory = trajectory
        self.steps = steps
        self.logs = LogBuffer(steps, env.n_muscles)

    def run(self):
        dt = self.arm.dt
        self.controller.reset(self.env.states["joint"][0][:2].copy())
        terminated = False
        for k in range(self.steps):
            t_now = k * dt
            x_d, xd_d, xdd_d = self.trajectory.sample(t_now)

            diag = self.controller.compute(x_d, xd_d, xdd_d)
            act = diag["act"]

            obs, reward, terminated, truncated, info = self.env.step(
                act,
                deterministic=True,
                endpoint_load=np.zeros((1, 2)),
                joint_load=np.zeros((1, 2)),
            )

            names = self.env.muscle.state_name
            idx_force = names.index("force")
            geom = self.env.states["geometry"]
            R = geom[:, 2 : 2 + self.env.skeleton.dof, :][0]
            forces = self.env.states["muscle"][0, idx_force, :]
            tau_real = -(R @ forces)

            self.logs.record(self.env, diag, tau_real, self.controller.qref)

            if terminated:
                break
        return self.logs
