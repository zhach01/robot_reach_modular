# random_target_reach_torch.py
# -*- coding: utf-8 -*-
"""
RandomTargetReach (Torch version)

Torch counterpart of random_target_reach_numpy.py:
- Inherits from Environment (environment_torch).
- Draws a random joint state, maps it to fingertip (x,y) and uses that as goal.
- Ensures target components in the observation are noiseless.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, Union, List

import torch
from torch import Tensor

from model_lib.environment_torch import Environment


class RandomTargetReach(Environment):
    """Reach to a random target from a random starting position (Torch version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make sure target components (first `space_dim` dims of obs) are noiseless.
        # After Environment.__init__ + _build_spaces, self.obs_noise is a 1D Tensor.
        if isinstance(self.obs_noise, Tensor) and self.obs_noise.numel() >= self.space_dim:
            self.obs_noise[: self.space_dim] = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Reset env and draw a random target in joint space, projected to fingertip (x,y).

        Returns:
          obs:  (batch, obs_dim)  Torch tensor
          info: dict with 'states', 'action', 'noisy action', 'goal'
        """
        if seed is not None:
            # delegate RNG seeding to effector
            self.effector.reset(seed=seed)

        options = {} if options is None else options
        batch_size: int = int(options.get("batch_size", 1))
        joint_state = options.get("joint_state", None)
        deterministic: bool = bool(options.get("deterministic", False))

        if joint_state is not None:
            js = torch.as_tensor(joint_state, dtype=self.dtype, device=self.device)
            if js.dim() == 1:
                js = js.view(1, -1)
            if js.shape[0] > 1:
                batch_size = js.shape[0]
        else:
            joint_state = self.q_init  # may be None -> effector chooses its own default

        # Initialize effector states for this episode
        self.effector.reset(
            options={"batch_size": batch_size, "joint_state": joint_state}
        )

        # Draw a random **joint** state and convert to fingertip to define a random target
        # draw_random_uniform_states should return (batch, dof*2) with zero velocities
        q_rand = self.effector.draw_random_uniform_states(batch_size)  # (B, dof*2)
        xy = self.joint2cartesian(q_rand)  # (B, 4) = [x, y, xd, yd]
        self.goal = xy[:, : self.space_dim]  # (B, 2) → goal in task space

        # Initialize buffers
        action0 = torch.zeros(
            (batch_size, self.n_muscles),
            dtype=self.dtype,
            device=self.device,
        )

        self.obs_buffer["proprioception"] = [
            self.get_proprioception()
        ] * max(1, len(self.obs_buffer["proprioception"]))

        self.obs_buffer["vision"] = [
            self.get_vision()
        ] * max(1, len(self.obs_buffer["vision"]))

        self.obs_buffer["action"] = [action0] * self.action_frame_stacking

        self.elapsed = 0.0

        obs = self.get_obs(deterministic=deterministic)
        info = {
            "states": self.effector.states,
            "action": action0,
            "noisy action": action0,
            "goal": self.goal,
        }
        return obs, info


# ---------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    from model_lib.skeleton_torch import TwoDofArm
    from model_lib.muscles_torch import ReluMuscle
    from model_lib.effector_torch import Effector

    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    print("[random_target_reach_torch] Simple smoke test...")

    # Build a tiny effector (TwoDofArm + ReluMuscle with 2 simple muscles)
    arm = TwoDofArm(
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
    mus = ReluMuscle(device=device, dtype=torch.get_default_dtype())
    eff = Effector(
        skeleton=arm,
        muscle=mus,
        timestep=0.002,
        integration_method="euler",
        damping=0.0,
        device=device,
        dtype=torch.get_default_dtype(),
    )

    # Add two simple straight-line muscles
    eff.add_muscle(
        path_fixation_body=[1, 1],
        path_coordinates=[[0.0, 0.05], [0.0, 0.0]],
        name="m1",
        max_isometric_force=100.0,
    )
    eff.add_muscle(
        path_fixation_body=[2, 2],
        path_coordinates=[[0.0, 0.05], [0.0, 0.0]],
        name="m2",
        max_isometric_force=120.0,
    )

    # Create Torch random-target environment
    env = RandomTargetReach(
        effector=eff,
        max_ep_duration=0.1,
        action_noise=0.0,
        obs_noise=0.0,
        action_frame_stacking=1,
        proprioception_delay=eff.dt,
        vision_delay=eff.dt,
    )

    # Reset with batch_size=2
    obs0, info0 = env.reset(options={"batch_size": 2, "deterministic": True})
    print("  obs0 shape:", obs0.shape)
    print("  goal shape:", info0["goal"].shape)
    print("  goal[0]:", info0["goal"][0])

    # Step once with random actions
    a = torch.rand(2, env.n_muscles, dtype=env.dtype, device=device)
    obs1, rew, terminated, truncated, info1 = env.step(a, deterministic=False)
    print("  obs1 shape:", obs1.shape)
    print("  terminated:", terminated, "truncated:", truncated)
    print("  noisy action shape:", info1["noisy action"].shape)

    print("[random_target_reach_torch] Smoke test done.")
