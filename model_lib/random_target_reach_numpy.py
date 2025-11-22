# random_target_reach_numpy.py

import numpy as np
from typing import Any, Dict, Tuple
from model_lib.environment_numpy import Environment


class RandomTargetReach(Environment):
    """Reach to a random target from a random starting position (NumPy version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make sure target components (first 2 dims of obs) are noiseless
        # After _build_spaces(), self.obs_noise is a per-dim vector
        # We need to rebuild if obs space changed. Call this AFTER __init__/reset once.
        if (
            isinstance(self.obs_noise, np.ndarray)
            and self.obs_noise.size >= self.space_dim
        ):
            self.obs_noise[: self.space_dim] = 0.0

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.effector.reset(seed=seed)

        options = {} if options is None else options
        batch_size: int = int(options.get("batch_size", 1))
        joint_state = options.get("joint_state", None)
        deterministic: bool = bool(options.get("deterministic", False))

        if joint_state is not None:
            js = np.asarray(joint_state, dtype=float)
            if js.ndim == 1:
                js = js.reshape(1, -1)
            if js.shape[0] > 1:
                batch_size = js.shape[0]
        else:
            joint_state = self.q_init

        # Initialize effector states
        self.effector.reset(
            options={"batch_size": batch_size, "joint_state": joint_state}
        )

        # Draw a random **joint** state and convert to fingertip to define a random target
        q_rand = self.effector.draw_random_uniform_states(
            batch_size
        )  # (batch, dof*2) with zero vel
        xy = self.joint2cartesian(q_rand)  # (batch, 4) = [x,y,xd,yd]
        self.goal = xy[:, : self.space_dim]  # (batch, 2)

        # initialize buffers
        action0 = np.zeros((batch_size, self.effector.n_muscles), dtype=float)
        self.obs_buffer["proprioception"] = [self.get_proprioception()] * max(
            1, len(self.obs_buffer["proprioception"])
        )
        self.obs_buffer["vision"] = [self.get_vision()] * max(
            1, len(self.obs_buffer["vision"])
        )
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
