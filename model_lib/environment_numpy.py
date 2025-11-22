# env_numpy.py

import numpy as np
import gymnasium as gym
from typing import Any, Tuple, Dict, List, Union

Array = np.ndarray


def _as_2d(x: Array) -> Array:
    """Ensure (batch, features) 2D shape."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim == 2:
        pass
    else:
        raise ValueError(f"Expected 1D/2D, got {x.ndim}D")
    return x


class Environment(gym.Env):
    """
    NumPy version of the base environment.

    Works with your NumPy-only Effector/Skeleton/Muscle stack:
      - effector: must expose .dt, .n_muscles, .space_dim, .states, .step(), .reset(), .joint2cartesian(), .np_random
      - states: dict with "joint", "muscle", "geometry", "cartesian", "fingertip" (NumPy arrays)

    Observation = [goal (2D), vision (2D), proprioception (2*nm), past actions if stacking]
      * proprioception = [normalized muscle length, normalized muscle velocity]
      * vision         = fingertip (x,y)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        effector,
        q_init: Array | None = None,
        name: str = "Env",
        differentiable: bool = True,  # kept for API compatibility (ignored; everything is NumPy)
        max_ep_duration: float = 1.0,
        action_noise: float = 0.0,
        obs_noise: float | List[float] = 0.0,
        action_frame_stacking: int = 0,
        proprioception_delay: float | None = None,
        vision_delay: float | None = None,
        proprioception_noise: float = 0.0,
        vision_noise: float = 0.0,
        **_,
    ):
        super().__init__()

        self.__name__ = name
        self.effector = effector
        self.dt = float(self.effector.dt)
        self.max_ep_duration = float(max_ep_duration)
        self.elapsed = 0.0

        # Initial joint state template (optional)
        if q_init is not None:
            q_init = np.asarray(q_init, dtype=float)
            if q_init.ndim == 1:
                q_init = q_init.reshape(1, -1)
            self.nq_init = q_init.shape[0]
        else:
            self.nq_init = None
        self.q_init = q_init

        # noise settings (store original + expanded versions later)
        self._action_noise = action_noise
        self._obs_noise = obs_noise
        self.action_noise = 0.0  # will be array after _build_spaces
        self.obs_noise = 0.0
        self.proprioception_noise = [float(proprioception_noise)]
        self.vision_noise = [float(vision_noise)]

        self.action_frame_stacking = int(action_frame_stacking)

        # delays (in steps)
        proprioception_delay = (
            self.dt if proprioception_delay is None else proprioception_delay
        )
        vision_delay = self.dt if vision_delay is None else vision_delay

        def _is_multiple_of_dt(delay_s: float) -> bool:
            # allow tiny floating precision ≥ gym-level tolerance
            frac = (delay_s / self.dt) % 1.0
            return np.isclose(frac, 0.0, atol=np.finfo(float).resolution * 10)

        assert _is_multiple_of_dt(
            vision_delay
        ), f"vision_delay={vision_delay} must be a multiple of dt={self.dt}"
        assert _is_multiple_of_dt(
            proprioception_delay
        ), f"proprioception_delay={proprioception_delay} must be a multiple of dt={self.dt}"

        self.proprioception_delay = int(round(proprioception_delay / self.dt))
        self.vision_delay = int(round(vision_delay / self.dt))

        # rolling buffers
        self.obs_buffer: Dict[str, List[Array]] = {
            "proprioception": [None] * self.proprioception_delay,
            "vision": [None] * self.vision_delay,
            "action": [None] * self.action_frame_stacking,
        }

        # goal (batch, 2); set in reset()
        self.goal: Array | None = None

        # Spaces depend on n_muscles and assembled observation; we need a reset to compute obs shape
        self._build_spaces()

    # -------------------- Spaces --------------------

    def _build_spaces(self):
        # Action: one scalar per muscle (range [0,1]); Environment will reshape for the effector
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.effector.n_muscles,),
            dtype=np.float32,
        )

        # Build an initial observation by doing a deterministic reset (no noise)
        obs, _ = self.reset(options={"deterministic": True})
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs.shape[-1],),
            dtype=np.float32,
        )

        # Expand configured noises to per-dimension arrays
        def _expand_noise(noise, dim: int) -> Array:
            if isinstance(noise, (int, float)):
                return np.full((dim,), float(noise), dtype=float)
            noise = np.asarray(noise, dtype=float).reshape(-1)
            assert (
                noise.size == dim
            ), f"obs_noise must have {dim} elements, got {noise.size}"
            return noise

        self.action_noise = _expand_noise(
            self._action_noise, self.action_space.shape[0]
        )
        self.obs_noise = _expand_noise(self._obs_noise, self.observation_space.shape[0])

    # -------------------- Observation helpers --------------------

    def get_proprioception(self) -> Array:
        """
        (batch, 2*nm):
          [ normalized muscle length | normalized muscle velocity ]
        """
        mus = self.effector.states["muscle"]  # (batch, state_dim, nm)
        l0_ce = self.effector.muscle.l0_ce  # (1,1,nm)
        vmax = self.effector.muscle.vmax  # (1,1,nm)

        # These channel indices match your models:
        #   'muscle length'  -> index 1
        #   'muscle velocity'-> index 2
        mlen_n = mus[:, 1:2, :] / l0_ce
        mvel_n = mus[:, 2:3, :] / vmax

        prop = np.concatenate([mlen_n, mvel_n], axis=1)  # (batch, 2, nm)
        prop = prop.reshape(prop.shape[0], -1)  # (batch, 2*nm)
        return self._apply_noise(prop, self.proprioception_noise)

    def get_vision(self) -> Array:
        """(batch, 2): fingertip position (x,y) with noise."""
        vis = self.effector.states["fingertip"]  # (batch, 2)
        return self._apply_noise(vis, self.vision_noise)

    def update_obs_buffer(self, action_2d: Array | None = None):
        # roll proprioception/vision
        self.obs_buffer["proprioception"] = self.obs_buffer["proprioception"][1:] + [
            self.get_proprioception()
        ]
        self.obs_buffer["vision"] = self.obs_buffer["vision"][1:] + [self.get_vision()]

        if action_2d is not None:
            # store as (batch, n_muscles)
            self.obs_buffer["action"] = self.obs_buffer["action"][1:] + [action_2d]

    def get_obs(
        self, action_2d: Array | None = None, deterministic: bool = False
    ) -> Array:
        self.update_obs_buffer(action_2d=action_2d)

        parts: List[Array] = [
            self.goal,  # (batch, 2)
            self.obs_buffer["vision"][0],  # (batch, 2)
            self.obs_buffer["proprioception"][0],  # (batch, 2*nm)
        ]
        # append past actions (each (batch, nm))
        for k in range(self.action_frame_stacking):
            parts.append(self.obs_buffer["action"][k])

        obs = np.concatenate(parts, axis=1)  # (batch, features)

        if not deterministic:
            obs = self._apply_noise(obs, self.obs_noise)

        return obs

    # -------------------- Gym API --------------------

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[Array, Dict[str, Any]]:
        if seed is not None:
            # seed the effector's RNG; Environment leverages effector.np_random as the single RNG
            self.effector.reset(seed=seed)

        options = {} if options is None else options
        batch_size: int = int(options.get("batch_size", 1))
        joint_state: Array | None = options.get("joint_state", None)
        deterministic: bool = bool(options.get("deterministic", False))

        if joint_state is not None:
            js = _as_2d(joint_state)
            if js.shape[0] > 1:
                batch_size = js.shape[0]
        else:
            joint_state = self.q_init

        # reset effector (sets self.effector.states)
        self.effector.reset(
            options={"batch_size": batch_size, "joint_state": joint_state}
        )

        # goal defaults to origin (0,0)
        self.goal = np.zeros((batch_size, self.effector.space_dim), dtype=float)

        # init buffers
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
            "states": self.effector.states,  # already NumPy
            "action": action0,
            "noisy action": action0,
            "goal": self.goal,
        }
        return obs, info

    def step(
        self,
        action: Array,
        deterministic: bool = False,
        **kwargs,
    ) -> Tuple[Array, float | Array | None, bool, bool, Dict[str, Any]]:
        """
        action: (batch, n_muscles) OR (n_muscles,)
        Environment will:
          - add Gaussian action noise (if enabled)
          - reshape to (batch, 1, n_muscles) for the effector
        """
        self.elapsed += self.dt

        a2d = _as_2d(action)  # (batch, n_muscles)

        #noisy_a2d = a2d.copy()
        #if not deterministic:
        #    noisy_a2d = self._apply_noise(noisy_a2d, self.action_noise)

        noisy_a2d = a2d.copy()
        if not deterministic:
            noisy_a2d = self._apply_noise(noisy_a2d, self.action_noise)
        # keep actions in valid excitation range *before* integration
        a_min = float(self.muscle.min_activation) if hasattr(self.muscle, "min_activation") else 0.0
        noisy_a2d = np.clip(noisy_a2d, a_min, 1.0)

        # effector expects (batch, 1, n_muscles)
        #noisy_a31 = noisy_a2d[:, None, :]
        # effector accepts (B,1,M) (we also support (B,M) now; either is fine)
        noisy_a31 = noisy_a2d[:, None, :]

        # integrate
        self.effector.step(noisy_a31, **kwargs)

        # build obs (optionally noisy)
        obs = self.get_obs(action_2d=noisy_a2d, deterministic=deterministic)

        # default reward/termination behavior:
        reward = (
            None  # for compatibility with "differentiable" path in your original API
        )
        truncated = False
        terminated = bool(self.elapsed >= self.max_ep_duration) or truncated

        info = {
            "states": self.effector.states,
            "action": a2d,
            "noisy action": noisy_a2d,
            "goal": self.goal,
        }

        return obs, reward, terminated, truncated, info

    # -------------------- RNG + Noise --------------------

    @property
    def np_random(self) -> np.random.Generator:
        return self.effector.np_random

    @np_random.setter
    def np_random(self, rng: np.random.Generator):
        self.effector.np_random = rng

    def _apply_noise(
        self, loc: np.ndarray, noise: float | list | np.ndarray
    ) -> np.ndarray:
        """
        Add element-wise Gaussian noise to (batch, features).
        Accepts:
        - scalar
        - 1D array/list of length==features
        - 1D array/list of length==1 (broadcasted)
        """
        loc = np.asarray(loc, dtype=float)
        feat = loc.shape[1]

        # normalize noise -> 1D vector length==feat
        if isinstance(noise, list) or isinstance(noise, np.ndarray):
            arr = np.asarray(noise, dtype=float).reshape(-1)
            if arr.size == 1:
                sigma = np.full((feat,), float(arr.item()), dtype=float)
            else:
                assert arr.size == feat, f"noise length {arr.size} != features {feat}"
                sigma = arr
        else:
            sigma = np.full((feat,), float(noise), dtype=float)

        eps = self.np_random.normal(loc=0.0, scale=sigma, size=loc.shape)
        return loc + eps

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
    def n_muscles(self):
        return self.effector.muscle.n_muscles

    @property
    def space_dim(self):
        return self.effector.space_dim

    def joint2cartesian(self, joint_states: Array) -> Array:
        return self.effector.joint2cartesian(joint_states)

    def get_attributes(self):
        """Return (names, values) for non-callable, non-space attributes (JSON-friendly)."""
        blacklist = {
            "effector",
            "muscle",
            "skeleton",
            "np_random",
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
        vals = [getattr(self, a) for a in attrs]
        return attrs, vals

    def get_save_config(self) -> Dict[str, Any]:
        cfg = {"name": self.__name__}
        names, vals = self.get_attributes()
        for k, v in zip(names, vals):
            if isinstance(v, np.ndarray):
                v = v.tolist()
            cfg[k] = v
        cfg["effector"] = self.effector.get_save_config()
        return cfg
