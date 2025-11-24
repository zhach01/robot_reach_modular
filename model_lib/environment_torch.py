# environment_torch.py
# -*- coding: utf-8 -*-
"""
Pure-PyTorch Environment.

Torch counterpart of environment_numpy.py:
- Wraps a Torch Effector (e.g. Effector, RigidTendonArm26, CompliantTendonArm26).
- Batchable on leading dimension.
- GPU-safe: all tensors created on effector.device.
- Fully differentiable from actions -> states/observations (noise can be disabled).

Requirements from effector:
  - .dt
  - .space_dim
  - .states: dict with "joint", "muscle", "geometry", "cartesian", "fingertip"
  - .step(action, **kwargs)
  - .reset(options={"batch_size": ..., "joint_state": ...})
  - .joint2cartesian(joint_states)
  - .skeleton, .muscle
  - .torch_generator (torch.Generator)
"""

from __future__ import annotations

from typing import Any, Tuple, Dict, List, Union, Optional

import math
import numpy as np
import gymnasium as gym
import torch
from torch import Tensor


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _as_2d(
    x: Union[Tensor, float, List[float], np.ndarray],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Ensure a (batch, features) 2D Torch tensor."""
    t = torch.as_tensor(x, dtype=dtype, device=device)
    if t.dim() == 1:
        t = t.view(1, -1)
    elif t.dim() == 2:
        pass
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {t.dim()}D with shape {tuple(t.shape)}")
    return t


# ---------------------------------------------------------------------
# Environment (Torch)
# ---------------------------------------------------------------------


class Environment(gym.Env):
    """
    Torch version of the base environment.

    Observation = [goal (2D), vision (2D), proprioception (2*nm), past actions if stacking]
      * proprioception = [normalized muscle length, normalized muscle velocity]
      * vision         = fingertip (x,y)

    All internal math is Torch; `step` and `reset` return Torch tensors.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        effector,
        q_init: Optional[Union[Tensor, np.ndarray, List[float]]] = None,
        name: str = "EnvTorch",
        differentiable: bool = True,  # kept for API compatibility (ignored here)
        max_ep_duration: float = 1.0,
        action_noise: float = 0.0,
        obs_noise: Union[float, List[float]] = 0.0,
        action_frame_stacking: int = 0,
        proprioception_delay: Optional[float] = None,
        vision_delay: Optional[float] = None,
        proprioception_noise: float = 0.0,
        vision_noise: float = 0.0,
        **_,
    ):
        super().__init__()

        self.__name__ = name
        self.effector = effector
        self.device = effector.device
        self.dtype = effector.dtype if hasattr(effector, "dtype") else torch.get_default_dtype()

        self.dt = float(self.effector.dt)
        self.max_ep_duration = float(max_ep_duration)
        self.elapsed = 0.0

        # Initial joint state template (optional) – stored as Torch
        if q_init is not None:
            q_init_t = torch.as_tensor(q_init, dtype=self.dtype, device=self.device)
            if q_init_t.dim() == 1:
                q_init_t = q_init_t.view(1, -1)
            self.nq_init = q_init_t.shape[0]
            self.q_init = q_init_t
        else:
            self.nq_init = None
            self.q_init = None

        # noise settings (store original + expanded versions later)
        self._action_noise = action_noise
        self._obs_noise = obs_noise
        self.action_noise: Union[float, Tensor] = 0.0  # will be 1D tensor after _build_spaces
        self.obs_noise: Union[float, Tensor] = 0.0
        # proprioception & vision noise are typically scalars; we keep them as simple scalars/lists
        self.proprioception_noise: Union[float, List[float]] = float(proprioception_noise)
        self.vision_noise: Union[float, List[float]] = float(vision_noise)

        self.action_frame_stacking = int(action_frame_stacking)

        # delays (in seconds -> steps)
        proprioception_delay = self.dt if proprioception_delay is None else float(proprioception_delay)
        vision_delay = self.dt if vision_delay is None else float(vision_delay)

        def _is_multiple_of_dt(delay_s: float) -> bool:
            frac = (delay_s / self.dt) % 1.0
            return math.isclose(frac, 0.0, rel_tol=0.0, abs_tol=1e-10)

        assert _is_multiple_of_dt(
            vision_delay
        ), f"vision_delay={vision_delay} must be a multiple of dt={self.dt}"
        assert _is_multiple_of_dt(
            proprioception_delay
        ), f"proprioception_delay={proprioception_delay} must be a multiple of dt={self.dt}"

        self.proprioception_delay = int(round(proprioception_delay / self.dt))
        self.vision_delay = int(round(vision_delay / self.dt))

        # rolling buffers; elements will be Torch tensors
        self.obs_buffer: Dict[str, List[Optional[Tensor]]] = {
            "proprioception": [None] * max(1, self.proprioception_delay),
            "vision": [None] * max(1, self.vision_delay),
            "action": [None] * self.action_frame_stacking,
        }

        # goal (batch, 2); set in reset()
        self.goal: Optional[Tensor] = None

        # spaces depend on n_muscles and obs dim; we build them here
        self._build_spaces()

    # -------------------- Spaces --------------------

    @property
    def torch_generator(self) -> torch.Generator:
        # delegate RNG to effector
        return self.effector.torch_generator

    def _build_spaces(self):
        # Action: one scalar per muscle (range [0,1])
        nm = self.n_muscles
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(nm,),
            dtype=np.float32,
        )

        # Build an initial observation by doing a deterministic reset (no obs noise yet)
        obs, _ = self.reset(options={"deterministic": True})
        obs_dim = obs.shape[-1]

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Expand configured noises to per-dimension Torch vectors
        def _expand_noise(noise, dim: int) -> Tensor:
            if isinstance(noise, (int, float)):
                return torch.full(
                    (dim,),
                    float(noise),
                    dtype=self.dtype,
                    device=self.device,
                )
            if isinstance(noise, torch.Tensor):
                n = noise.to(self.device, self.dtype).view(-1)
            else:
                n = torch.as_tensor(noise, dtype=self.dtype, device=self.device).view(-1)
            if n.numel() == 1:
                return n.expand(dim)
            if n.numel() != dim:
                raise ValueError(f"Noise must have {dim} elements, got {n.numel()}")
            return n

        self.action_noise = _expand_noise(self._action_noise, nm)
        self.obs_noise = _expand_noise(self._obs_noise, obs_dim)

    # -------------------- Observation helpers --------------------

    def _apply_noise(self, loc: Tensor, noise: Union[float, List[float], Tensor]) -> Tensor:
        """
        Add element-wise Gaussian noise to (batch, features).
        Accepts:
        - scalar
        - 1D torch tensor (len==features or 1)
        - list/tuple convertible to torch 1D
        """
        loc = loc.to(dtype=self.dtype, device=self.device)
        if loc.numel() == 0:
            return loc
        if loc.dim() != 2:
            raise ValueError(f"_apply_noise expects (B,F) tensor, got shape {tuple(loc.shape)}")
        B, F = loc.shape

        # normalize noise -> 1D tensor length == F
        if isinstance(noise, torch.Tensor):
            n = noise.to(self.device, self.dtype).view(-1)
        else:
            n = torch.as_tensor(noise, dtype=self.dtype, device=self.device).view(-1)

        if n.numel() == 1:
            sigma = n.expand(F)
        elif n.numel() == F:
            sigma = n
        else:
            raise ValueError(f"noise length {n.numel()} != features {F}")

        if torch.all(sigma == 0):
            return loc

        eps = torch.randn(
            (B, F),
            generator=self.torch_generator,
            device=self.device,
            dtype=self.dtype,
        )
        return loc + eps * sigma.unsqueeze(0)

    def get_proprioception(self) -> Tensor:
        """
        (batch, 2*nm):
          [ normalized muscle length | normalized muscle velocity ]
        """
        mus = self.effector.states["muscle"]  # (batch, state_dim, nm)
        l0_ce = self.effector.muscle.l0_ce   # (1,1,nm)
        vmax = self.effector.muscle.vmax     # (1,1,nm)

        # channel indices must match muscle_torch models:
        #   'muscle length'   -> index 1
        #   'muscle velocity' -> index 2
        mlen_n = mus[:, 1:2, :] / l0_ce
        mvel_n = mus[:, 2:3, :] / vmax

        prop = torch.cat([mlen_n, mvel_n], dim=1)  # (batch, 2, nm)
        prop = prop.reshape(prop.shape[0], -1)     # (batch, 2*nm)
        return self._apply_noise(prop, self.proprioception_noise)

    def get_vision(self) -> Tensor:
        """(batch, 2): fingertip position (x,y) with noise."""
        vis = self.effector.states["fingertip"]  # (batch, 2)
        return self._apply_noise(vis, self.vision_noise)

    def update_obs_buffer(self, action_2d: Optional[Tensor] = None):
        # roll proprioception/vision
        self.obs_buffer["proprioception"] = self.obs_buffer["proprioception"][1:] + [
            self.get_proprioception()
        ]
        self.obs_buffer["vision"] = self.obs_buffer["vision"][1:] + [self.get_vision()]

        if self.action_frame_stacking > 0:
            if action_2d is None:
                # keep previous content
                self.obs_buffer["action"] = self.obs_buffer["action"][1:] + [
                    self.obs_buffer["action"][-1]
                ]
            else:
                self.obs_buffer["action"] = self.obs_buffer["action"][1:] + [action_2d]

    def get_obs(
        self,
        action_2d: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> Tensor:
        self.update_obs_buffer(action_2d=action_2d)

        parts: List[Tensor] = [
            self.goal,                        # (batch, 2)
            self.obs_buffer["vision"][0],     # (batch, 2)
            self.obs_buffer["proprioception"][0],  # (batch, 2*nm)
        ]

        # append past actions (each (batch, nm))
        for k in range(self.action_frame_stacking):
            a_k = self.obs_buffer["action"][k]
            if a_k is None:
                # if not yet initialized, use zeros
                B = self.goal.shape[0]
                a_k = torch.zeros(B, self.n_muscles, dtype=self.dtype, device=self.device)
            parts.append(a_k)

        obs = torch.cat(parts, dim=1)  # (batch, features)

        if not deterministic:
            obs = self._apply_noise(obs, self.obs_noise)

        return obs

    # -------------------- Gym API --------------------

    def reset_(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        if seed is not None:
            # Delegate seeding to effector; it manages a torch.Generator
            self.effector.reset(seed=seed)

        options = {} if options is None else options
        batch_size: int = int(options.get("batch_size", 1))
        joint_state = options.get("joint_state", None)
        deterministic: bool = bool(options.get("deterministic", False))

        # Determine batch size from joint_state if provided
        if joint_state is not None:
            js = _as_2d(joint_state, device=self.device, dtype=self.dtype)
            if js.shape[0] > 1:
                batch_size = js.shape[0]
        else:
            joint_state = self.q_init  # can be None

        # reset effector (sets self.effector.states)
        self.effector.reset(
            options={"batch_size": batch_size, "joint_state": joint_state}
        )

        # goal defaults to origin (0,0,...)
        self.goal = torch.zeros(
            (batch_size, self.effector.space_dim),
            dtype=self.dtype,
            device=self.device,
        )

        # init buffers
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


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Reset the environment and effector.

        options:
          - batch_size: int (optional)
          - joint_state: (B, 2*dof) or (2*dof,) or None
          - deterministic: bool
        """
        # ------------------------------------------------------
        # Seed effector RNG (if requested)
        # ------------------------------------------------------
        if seed is not None:
            # Delegate seeding to effector; it manages a torch.Generator
            self.effector.reset(seed=seed)

        options = {} if options is None else options
        batch_size: int = int(options.get("batch_size", 1))
        joint_state = options.get("joint_state", None)
        deterministic: bool = bool(options.get("deterministic", False))

        # ------------------------------------------------------
        # Infer batch_size from joint_state (if provided)
        # ------------------------------------------------------
        if joint_state is not None:
            js = _as_2d(joint_state, device=self.device, dtype=self.dtype)
            if js.shape[0] > 1:
                batch_size = js.shape[0]
        else:
            # fall back to stored initial joint config (can be None)
            joint_state = self.q_init

        # ------------------------------------------------------
        # Reset effector (this sets self.effector.states)
        # ------------------------------------------------------
        self.effector.reset(
            options={"batch_size": batch_size, "joint_state": joint_state}
        )

        # 🔴 Important: from here on, trust the effector's batch size
        joint = self.effector.states["joint"]
        if joint.dim() != 2:
            raise ValueError(
                f"Effector joint state must be (B, state_dim), got {tuple(joint.shape)}"
            )
        batch_size = joint.shape[0]
        print(f"[env_torch] reset with batch_size = {batch_size}")
        # ------------------------------------------------------
        # Goal defaults to origin (0, 0, ...) for each batch element
        # ------------------------------------------------------
        self.goal = torch.zeros(
            (batch_size, self.effector.space_dim),
            dtype=self.dtype,
            device=self.device,
        )

        # ------------------------------------------------------
        # Init action buffer and obs buffers
        # ------------------------------------------------------
        action0 = torch.zeros(
            (batch_size, self.n_muscles),
            dtype=self.dtype,
            device=self.device,
        )

        # Fill proprioception / vision buffers with current measurements,
        # replicated to full buffer length
        self.obs_buffer["proprioception"] = [
            self.get_proprioception()
        ] * max(1, len(self.obs_buffer["proprioception"]))

        self.obs_buffer["vision"] = [
            self.get_vision()
        ] * max(1, len(self.obs_buffer["vision"]))

        # Past actions: all zeros at reset
        self.obs_buffer["action"] = [action0] * self.action_frame_stacking

        # Reset time
        self.elapsed = 0.0

        # ------------------------------------------------------
        # Initial observation + info dict
        # ------------------------------------------------------
        obs = self.get_obs(deterministic=deterministic)
        info = {
            "states": self.effector.states,
            "action": action0,
            "noisy action": action0,
            "goal": self.goal,
        }
        return obs, info








    def step(
        self,
        action: Union[Tensor, np.ndarray, List[float]],
        deterministic: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], bool, bool, Dict[str, Any]]:
        """
        action: (batch, n_muscles) OR (n_muscles,)
        Environment will:
          - add Gaussian action noise (if enabled)
          - clamp to [min_activation, 1.0]
          - feed to effector.step(action)
        """
        self.elapsed += self.dt

        a2d = _as_2d(action, device=self.device, dtype=self.dtype)  # (batch, n_muscles)

        noisy_a2d = a2d.clone()
        if not deterministic:
            noisy_a2d = self._apply_noise(noisy_a2d, self.action_noise)

        # keep actions in valid excitation range *before* integration
        a_min = float(self.muscle.min_activation) if hasattr(self.muscle, "min_activation") else 0.0
        noisy_a2d = torch.clamp(noisy_a2d, a_min, 1.0)

        # effector.step accepts (B,M) or (B,1,M) in our Torch implementation
        self.effector.step(noisy_a2d, **kwargs)

        # build obs (optionally noisy)
        obs = self.get_obs(action_2d=noisy_a2d, deterministic=deterministic)

        # default reward/termination behavior:
        reward = None  # for compatibility with original API
        truncated = False
        terminated = bool(self.elapsed >= self.max_ep_duration) or truncated

        info = {
            "states": self.effector.states,
            "action": a2d,
            "noisy action": noisy_a2d,
            "goal": self.goal,
        }

        return obs, reward, terminated, truncated, info

    # -------------------- Utilities / Introspection --------------------

    @property
    def states(self):
        return self.effector.states

    @property
    def skeleton(self):
        return self.effector.skeleton

    @property
    def muscle(self):
        return self.effector.muscle

    @property
    def n_muscles(self) -> int:
        # prefer effector.muscle.n_muscles if available, else effector.n_muscles
        if hasattr(self.effector.muscle, "n_muscles"):
            return int(self.effector.muscle.n_muscles)
        if hasattr(self.effector, "n_muscles"):
            return int(self.effector.n_muscles)
        raise AttributeError("Cannot determine number of muscles.")

    @property
    def space_dim(self) -> int:
        return int(self.effector.space_dim)

    def joint2cartesian(self, joint_states: Tensor) -> Tensor:
        return self.effector.joint2cartesian(joint_states)

    def get_attributes(self):
        """Return (names, values) for non-callable, non-space attributes (JSON-friendly)."""
        blacklist = {
            "effector",
            "muscle",
            "skeleton",
            "states",
            "goal",
            "obs_buffer",
            "unwrapped",
        }
        attrs = [
            a
            for a in dir(self)
            if (not a.startswith("_"))
            and (not callable(getattr(self, a)))
            and (a not in blacklist)
            and (not isinstance(getattr(self, a), gym.spaces.Space))
        ]
        vals: List[Any] = []
        for a in attrs:
            v = getattr(self, a)
            if isinstance(v, Tensor):
                v = v.tolist()
            vals.append(v)
        return attrs, vals

    def get_save_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"name": self.__name__}
        names, vals = self.get_attributes()
        for k, v in zip(names, vals):
            cfg[k] = v
        if hasattr(self.effector, "get_save_config"):
            cfg["effector"] = self.effector.get_save_config()
        return cfg


# ---------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------


if __name__ == "__main__":
    from model_lib.skeleton_torch import TwoDofArm
    from model_lib.muscles_torch import ReluMuscle
    from model_lib.effector_torch import Effector

    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    print("[environment_torch] Simple smoke test...")

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

    env = Environment(
        effector=eff,
        max_ep_duration=0.02,
        action_noise=0.0,
        obs_noise=0.0,
        action_frame_stacking=1,
        proprioception_delay=eff.dt,
        vision_delay=eff.dt,
    )

    # Reset
    obs0, info0 = env.reset(options={"batch_size": 2, "deterministic": True})
    print("  obs0 shape:", obs0.shape)
    print("  goal:", info0["goal"])
    print("  action0:", info0["action"])

    # One step with random actions
    a = torch.rand(2, env.n_muscles, dtype=env.dtype, device=device)
    obs1, rew, terminated, truncated, info1 = env.step(a, deterministic=False)
    print("  obs1 shape:", obs1.shape)
    print("  terminated:", terminated, "truncated:", truncated)
    print("  noisy action:", info1["noisy action"])

    print("[environment_torch] Smoke test done.")
