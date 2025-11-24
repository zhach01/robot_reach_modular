# simulator_torch.py
# -*- coding: utf-8 -*-
"""
Torch-only unified simulator.

Mirrors simulator.py (mixed NumPy/Torch) but:
- Pure Torch backend (no NumPy branches).
- Policy mode:
    * RL rollout on any Torch Environment (e.g. RandomTargetReach).
    * Uses PolicyGRU_Torch.
- Controller mode:
    * Classical controller + trajectory tracking.
    * Uses LogBufferTorch and Torch env/effector stack.

Public API:

    @dataclass
    class SimConfig:
        mode: "policy" or "controller"
        horizon, hidden_size, deterministic, seed, steps

    class SimulatorTorch:
        run(batch_size=1)

    class TargetReachSimulatorTorch(SimulatorTorch):
        # controller mode wrapper (like old TargetReachSimulator)

    class PolicyRolloutSimulatorTorch(SimulatorTorch):
        # policy mode wrapper

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from logging_tools.log_buffer_torch import LogBufferTorch
from model_lib.policies_torch import PolicyGRU_Torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """
    Torch-only simulation config.

    mode:
        "policy"     -> RL policy rollout on a Torch Environment
        "controller" -> classical controller + trajectory tracking

    horizon:
        Number of rollout steps in policy mode.

    hidden_size:
        GRU hidden size in policy mode.

    deterministic:
        Passed to env.step(..., deterministic=...) in policy mode.

    seed:
        Forwarded to env.reset(seed=...) in policy mode.

    steps:
        Number of simulation steps in controller mode.
    """
    mode: str = "policy"          # "policy" or "controller"

    # Policy-mode options
    horizon: int = 5
    hidden_size: int = 32
    deterministic: bool = True
    seed: Optional[int] = None

    # Controller-mode options
    steps: int = 1000


# ---------------------------------------------------------------------------
# Unified Torch-only simulator
# ---------------------------------------------------------------------------

class SimulatorTorch:
    """
    Unified simulator for Torch stack:
      - RL policy rollouts (mode = "policy")
      - Controller-based target reaching (mode = "controller")

    Public methods:
      - run(...)
        * policy mode    -> dict with obs/act trajectories (Torch tensors)
        * controller mode-> LogBufferTorch with detailed physics logs

      - reset(...), step(...) are only meaningful in policy mode.
    """

    def __init__(
        self,
        *,
        config: SimConfig,
        env: Optional[Any] = None,
        arm: Optional[Any] = None,
        controller: Optional[Any] = None,
        trajectory: Optional[Any] = None,
        policy: Optional[PolicyGRU_Torch] = None,
    ):
        self.config = config
        self.mode = config.mode

        if self.mode not in ("policy", "controller"):
            raise ValueError("SimConfig.mode must be 'policy' or 'controller'")

        if self.mode == "policy":
            self._init_policy_mode(env=env, policy=policy)
        else:
            self._init_controller_mode(env=env, arm=arm, controller=controller, trajectory=trajectory)

    # ------------------------------------------------------------------
    # Policy mode: init + helpers
    # ------------------------------------------------------------------
    def _init_policy_mode(self, env: Optional[Any], policy: Optional[PolicyGRU_Torch]):
        """
        Initialise policy mode with a Torch environment and Torch policy.

        env:
            Torch Environment (e.g. RandomTargetReach from random_target_reach_torch)
        policy:
            PolicyGRU_Torch instance, or None (lazy construction)
        """
        if env is None:
            raise ValueError("Policy mode requires a Torch environment (env) instance")

        self.env = env
        self.policy = policy
        self.h: Optional[Tensor] = None  # GRU hidden state

    def _ensure_policy(self, obs_dim: int, act_dim: int):
        """Create a PolicyGRU_Torch if none is supplied."""
        if self.policy is not None:
            return

        device = getattr(self.env, "device", torch.device("cpu"))
        dtype = getattr(self.env, "dtype", torch.get_default_dtype())

        self.policy = PolicyGRU_Torch(
            input_dim=obs_dim,
            hidden_dim=self.config.hidden_size,
            output_dim=act_dim,
            device=device,
            dtype=dtype,
        )

    def _init_hidden(self, batch_size: int):
        """Initialise GRU hidden state."""
        if self.policy is None:
            raise RuntimeError("Policy must be created before init_hidden")
        self.h = self.policy.init_hidden(batch_size=batch_size)

    @staticmethod
    def _clip_action01(a: Tensor) -> Tensor:
        """Clamp actions to [0,1] for muscle activations."""
        return torch.clamp(a, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Policy mode: public API
    # ------------------------------------------------------------------
    def reset(self, *, batch_size: int = 1) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Reset environment and policy (policy mode only).

        Returns:
          obs, info  (from env.reset) as Torch tensor + info dict
        """
        if self.mode != "policy":
            raise RuntimeError("reset(...) is only valid in policy mode")

        # forward seed into env if provided
        if self.config.seed is not None:
            obs, info = self.env.reset(
                seed=self.config.seed,
                options={"batch_size": batch_size},
            )
        else:
            obs, info = self.env.reset(options={"batch_size": batch_size})

        obs = torch.as_tensor(obs, device=self.env.device, dtype=self.env.dtype)
        obs_dim = int(obs.shape[-1])
        act_dim = int(self.env.effector.input_dim)  # muscles

        self._ensure_policy(obs_dim, act_dim)
        self._init_hidden(batch_size)

        return obs, info

    def step(self, obs: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """
        One policy + environment step (policy mode only).

        Inputs:
          obs: current observation (Torch tensor, shape (B, obs_dim))

        Returns:
          obs_next, info
        """
        if self.mode != "policy":
            raise RuntimeError("step(...) is only valid in policy mode")

        if self.policy is None or self.h is None:
            raise RuntimeError("Call reset(...) before step(...) in policy mode")

        obs_t = torch.as_tensor(obs, device=self.env.device, dtype=self.env.dtype)

        # policy forward
        a, self.h = self.policy.forward(obs_t, self.h)
        a = self._clip_action01(a)

        # env step
        obs_next, _, _, _, info = self.env.step(
            a,
            deterministic=self.config.deterministic,
        )
        obs_next = torch.as_tensor(obs_next, device=self.env.device, dtype=self.env.dtype)
        return obs_next, info

    def _rollout_policy(self, *, batch_size: int = 1) -> Dict[str, Any]:
        """
        Policy mode rollout.

        Returns:
          {
            "obs": (T+1, B, obs_dim)  Torch tensor
            "act": (T,   B, act_dim)  Torch tensor
            "info_first": info from reset,
            "info_last":  info from final step
          }
        """
        T = int(self.config.horizon)
        obs, info0 = self.reset(batch_size=batch_size)

        obs_traj = [obs]
        act_traj = []
        info = {}

        for _ in range(T):
            o_last = obs_traj[-1]
            if self.policy is None or self.h is None:
                raise RuntimeError("Policy/hidden state not initialised")

            a_t, self.h = self.policy.forward(o_last, self.h)
            a_t = self._clip_action01(a_t)
            act_traj.append(a_t)

            o_next, _, _, _, info = self.env.step(
                a_t,
                deterministic=self.config.deterministic,
            )
            o_next = torch.as_tensor(o_next, device=self.env.device, dtype=self.env.dtype)
            obs_traj.append(o_next)

        obs_stack = torch.stack(obs_traj, dim=0)  # (T+1,B,obs_dim)
        act_stack = torch.stack(act_traj, dim=0)  # (T,B,act_dim)
        return {
            "obs": obs_stack,
            "act": act_stack,
            "info_first": info0,
            "info_last": info,
        }

    # ------------------------------------------------------------------
    # Controller mode: init + run
    # ------------------------------------------------------------------
    def _init_controller_mode(
        self,
        env: Any,
        arm: Any,
        controller: Any,
        trajectory: Any,
    ):
        """
        Initialise classical target-reaching simulator (controller mode).

        Torch stack requirements:
        - env: Environment (Torch) with .dt, .states, .step(...)
        - arm: skeleton object (kept for API symmetry)
        - controller: has .reset(q0) and .compute(x_d, xd_d, xdd_d)
        - trajectory: object with .sample(t) -> (x_d, xd_d, xdd_d) in Torch
        """
        if env is None or arm is None or controller is None or trajectory is None:
            raise ValueError(
                "Controller mode requires env, arm, controller, and trajectory"
            )

        self.env = env
        self.arm = arm
        self.controller = controller
        self.trajectory = trajectory
        self.steps = int(self.config.steps)

        device = getattr(env, "device", torch.device("cpu"))
        dtype = getattr(env, "dtype", torch.get_default_dtype())
        self.logs = LogBufferTorch(self.steps, env.n_muscles, device=device, dtype=dtype)

    def _run_controller(self):
        """
        Controller-mode simulation loop (Torch version of TargetReachSimulator.run).
        """
        dt = float(self.env.dt)

        # initialise controller with current q
        joint0 = self.env.states["joint"][0]   # [q1,q2,qd1,qd2]
        dof = int(self.env.skeleton.dof)
        q0 = joint0[:dof].clone()
        self.controller.reset(q0)

        terminated = False

        for k in range(self.steps):
            t_now = k * dt
            x_d, xd_d, xdd_d = self.trajectory.sample(t_now)

            diag = self.controller.compute(x_d, xd_d, xdd_d)
            act = diag["act"]
            if k == 0:
                print(f"[sim_torch] batch size = {act.shape[0]}")

            # optional debug
            if k in (0, 5, 20, 50, 100):
                if isinstance(act, Tensor):
                    a_min = float(act.min())
                    a_max = float(act.max())
                else:
                    a_t = torch.as_tensor(act)
                    a_min = float(a_t.min())
                    a_max = float(a_t.max())
                print(f"[sim_torch] step {k}: act min/max = {a_min:.3f} / {a_max:.3f}")

            # ensure batch shape for env.step
            act_t = torch.as_tensor(act, device=self.env.device, dtype=self.env.dtype)
            if act_t.dim() == 1:
                act_t = act_t.unsqueeze(0)  # (1,n_muscles)

            B = act_t.shape[0]
            device = self.env.device
            dtype = self.env.dtype

            endpoint_load = torch.zeros((B, 2), device=device, dtype=dtype)
            joint_load = torch.zeros((B, 2), device=device, dtype=dtype)

            obs, reward, terminated, truncated, info = self.env.step(
                act_t,
                deterministic=True,
                endpoint_load=endpoint_load,
                joint_load=joint_load,
            )

            # real torques from current muscle forces
            names = self.env.muscle.state_name
            idx_force = names.index("force")
            geom = self.env.states["geometry"]             # (B, 2+DOF, M)
            R = geom[:, 2 : 2 + dof, :][0]                 # (DOF,M)
            muscle_state = self.env.states["muscle"]       # (B, channels, M)
            forces = muscle_state[0, idx_force, :]         # (M,)
            tau_real = -(R @ forces)                       # (DOF,)

            self.logs.record(self.env, diag, tau_real, self.controller.qref)

            if bool(terminated):
                break

        return self.logs

    # ------------------------------------------------------------------
    # Unified public entry point
    # ------------------------------------------------------------------
    def run(self, *, batch_size: int = 1):
        """
        Unified entry point:

        - If mode == "policy":
            returns dict with trajectories (obs, act, info_first, info_last)

        - If mode == "controller":
            returns LogBufferTorch (same semantics as old TargetReachSimulator.run())
        """
        if self.mode == "policy":
            return self._rollout_policy(batch_size=batch_size)
        else:
            return self._run_controller()


# ---------------------------------------------------------------------------
# Backwards-compatible-style wrappers (Torch versions)
# ---------------------------------------------------------------------------

class TargetReachSimulatorTorch(SimulatorTorch):
    """
    Torch wrapper for controller-based target reaching.

    Usage:

        sim = TargetReachSimulatorTorch(env, arm, controller, trajectory, steps)
        logs = sim.run()
    """

    def __init__(self, env, arm, controller, trajectory, steps: int):
        cfg = SimConfig(mode="controller", steps=int(steps))
        super().__init__(
            config=cfg,
            env=env,
            arm=arm,
            controller=controller,
            trajectory=trajectory,
        )


class PolicyRolloutSimulatorTorch(SimulatorTorch):
    """
    Convenience wrapper for policy mode (Torch-only).

    Usage:

        cfg = SimConfig(mode="policy", horizon=100, hidden_size=64)
        sim = PolicyRolloutSimulatorTorch(config=cfg, env=env_pol, policy=None)
        traj = sim.run(batch_size=32)
    """

    def __init__(
        self,
        config: Optional[SimConfig] = None,
        env: Optional[Any] = None,
        policy: Optional[PolicyGRU_Torch] = None,
    ):
        if config is None:
            config = SimConfig(mode="policy")
        else:
            config.mode = "policy"
        super().__init__(config=config, env=env, policy=policy)


# ---------------------------------------------------------------------------
# Tiny smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    torch.set_printoptions(precision=6, sci_mode=False)
    torch.set_default_dtype(torch.float64)

    print("=== Policy mode (Torch backend) ===")
    from model_lib.skeleton_torch import TwoDofArm
    from model_lib.muscles_torch import ReluMuscle
    from model_lib.effector_torch import Effector
    from model_lib.environment_torch import Environment
    from model_lib.random_target_reach_torch import RandomTargetReach
    from trajectory.minjerk_torch import MinJerkLinearTrajectoryTorch, MinJerkParams

    device = torch.device("cpu")

    # Common arm + muscles
    arm_pol = TwoDofArm(
        m1=1.82,
        m2=1.43,
        l1g=0.135,
        l2g=0.165,
        i1=0.051,
        i2=0.057,
        l1=0.309,
        l2=0.333,
        device=device,
        dtype=torch.get_default_dtype(),
    )
    mus_pol = ReluMuscle(device=device, dtype=torch.get_default_dtype())
    eff_pol = Effector(
        skeleton=arm_pol,
        muscle=mus_pol,
        timestep=0.002,
        integration_method="euler",
        damping=0.0,
        device=device,
        dtype=torch.get_default_dtype(),
    )

    # Add two simple muscles
    eff_pol.add_muscle(
        path_fixation_body=[1, 1],
        path_coordinates=[[0.0, 0.05], [0.0, 0.0]],
        name="m1",
        max_isometric_force=100.0,
    )
    eff_pol.add_muscle(
        path_fixation_body=[2, 2],
        path_coordinates=[[0.0, 0.05], [0.0, 0.0]],
        name="m2",
        max_isometric_force=120.0,
    )

    env_pol = RandomTargetReach(
        effector=eff_pol,
        max_ep_duration=0.1,
        action_noise=0.0,
        obs_noise=0.0,
        action_frame_stacking=1,
        proprioception_delay=eff_pol.dt,
        vision_delay=eff_pol.dt,
    )

    cfg_pol_t = SimConfig(
        mode="policy",
        horizon=3,
        hidden_size=16,
        seed=123,
        deterministic=True,
    )
    sim_pol_t = SimulatorTorch(config=cfg_pol_t, env=env_pol)
    traj_pol_t = sim_pol_t.run(batch_size=2)
    o_t, a_t = traj_pol_t["obs"], traj_pol_t["act"]
    print("obs:", o_t)
    print("act:", a_t)
    print("T  obs shape:", tuple(o_t.shape), "act shape:", tuple(a_t.shape))

    print("\n=== Controller mode (Torch backend) ===")

    # Separate arm/env for controller smoke
    arm_ctrl = TwoDofArm(
        m1=1.82,
        m2=1.43,
        l1g=0.135,
        l2g=0.165,
        i1=0.051,
        i2=0.057,
        l1=0.309,
        l2=0.333,
        device=device,
        dtype=torch.get_default_dtype(),
    )
    mus_ctrl = ReluMuscle(device=device, dtype=torch.get_default_dtype())
    eff_ctrl = Effector(
        skeleton=arm_ctrl,
        muscle=mus_ctrl,
        timestep=0.002,
        integration_method="euler",
        damping=0.0,
        device=device,
        dtype=torch.get_default_dtype(),
    )

    eff_ctrl.add_muscle(
        path_fixation_body=[1, 1],
        path_coordinates=[[0.0, 0.05], [0.0, 0.0]],
        name="m1",
        max_isometric_force=100.0,
    )
    eff_ctrl.add_muscle(
        path_fixation_body=[2, 2],
        path_coordinates=[[0.0, 0.05], [0.0, 0.0]],
        name="m2",
        max_isometric_force=120.0,
    )

    env_ctrl = Environment(
        effector=eff_ctrl,
        max_ep_duration=0.1,
        action_noise=0.0,
        obs_noise=0.0,
        action_frame_stacking=1,
        proprioception_delay=eff_ctrl.dt,
        vision_delay=eff_ctrl.dt,
    )

    # Reset to initialise states
    env_ctrl.reset(options={"batch_size": 1, "deterministic": True})

    # Dummy controller & trajectory (like in earlier tests)
    class DummyController:
        def __init__(self, env_):
            self.env = env_
            self.qref = torch.zeros(2, dtype=env_.dtype, device=env_.device)

        def reset(self, q0: Tensor):
            self.qref = q0.detach().clone()

        def compute(self, x_d, xd_d, xdd_d):
            joint = self.env.states["joint"][0]
            q = joint[:2]

            fingertip = self.env.states["fingertip"][0]
            if fingertip.numel() >= 4:
                x = fingertip[:2]
                xd = fingertip[2:4]
            else:
                x = fingertip[:2]
                xd = torch.zeros_like(x)

            nm = self.env.n_muscles
            act = torch.zeros(nm, dtype=self.env.dtype, device=self.env.device)

            geom = self.env.states["geometry"]
            dof = self.env.skeleton.dof
            R = geom[:, 2 : 2 + dof, :][0]
            Fmax = torch.ones(nm, dtype=self.env.dtype, device=self.env.device)

            diag = {
                "xref_tuple": (x_d, xd_d, xdd_d),
                "x": x,
                "xd": xd,
                "q": q,
                "tau_des": torch.zeros(dof, dtype=self.env.dtype, device=self.env.device),
                "act": act,
                "R": R,
                "Fmax": Fmax,
            }
            return diag

    ctrl = DummyController(env_ctrl)

    # simple 2D min-jerk trajectory from current fingertip to a point to the right
    center = env_ctrl.states["fingertip"][0, :2]
    target = center + torch.tensor([0.05, 0.0], device=device, dtype=torch.get_default_dtype())
    waypoints = torch.stack([center, target], dim=0)

    params = MinJerkParams(Vmax=1.0, Amax=10.0, Jmax=100.0, gamma=1.1)
    traj = MinJerkLinearTrajectoryTorch(waypoints, params)

    cfg_ctrl = SimConfig(mode="controller", steps=50)
    sim_ctrl = SimulatorTorch(
        config=cfg_ctrl,
        env=env_ctrl,
        arm=arm_ctrl,
        controller=ctrl,
        trajectory=traj,
    )
    logs = sim_ctrl.run()

    k, t = logs.time(dt=env_ctrl.dt)
    print("steps logged:", k)
    print("t[:5]:", t[:5])
    print("x_log shape:", logs.x_log[:k].shape)
    print("tau_real_log[0]:", logs.tau_real_log[0])

    print("\n[simulator_torch] Unified Torch simulator smoke ✓")
