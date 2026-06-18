"""
Audit finding H1: the action-frame-stacking history must behave identically in
environment_numpy and environment_torch. The action-history slice of the
observation is just the (clamped) past actions, so it must match exactly across
the two backends for the same action sequence — independent of dynamics.

Run:  python -m pytest tests/test_env_obs_parity.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import model_lib.skeleton_numpy as sk_np
sk_np.USE_CACHE = False

torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)

STACK = 3


def _build_torch():
    from model_lib.environment_torch import Environment
    from model_lib.muscles_torch import RigidTendonHillMuscle
    from model_lib.effector_torch import RigidTendonArm26
    mus = RigidTendonHillMuscle(min_activation=0.02, dtype=torch.float64)
    arm = RigidTendonArm26(muscle=mus, timestep=0.01, damping=0.0, dtype=torch.float64)
    env = Environment(effector=arm, max_ep_duration=2.0, action_noise=0.0,
                      obs_noise=0.0, action_frame_stacking=STACK,
                      proprioception_delay=arm.dt, vision_delay=arm.dt)
    return env, env.n_muscles


def _build_numpy():
    from model_lib.environment_numpy import Environment
    from model_lib.muscles_numpy import RigidTendonHillMuscle
    from model_lib.effector_numpy import RigidTendonArm26
    mus = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(muscle=mus, timestep=0.01, damping=0.0)
    env = Environment(effector=arm, max_ep_duration=2.0, action_noise=0.0,
                      obs_noise=0.0, action_frame_stacking=STACK,
                      proprioception_delay=arm.dt, vision_delay=arm.dt)
    return env, env.n_muscles


def _action_slice(obs, nm):
    obs = np.asarray(obs, dtype=float).reshape(-1)
    return obs[-STACK * nm:]


def test_action_history_parity_numpy_vs_torch():
    env_t, nm = _build_torch()
    env_n, nm2 = _build_numpy()
    assert nm == nm2

    q0 = np.deg2rad([55.0, 65.0])
    joint0 = np.concatenate([q0, [0.0, 0.0]]).reshape(1, 4)
    env_t.reset(options={"joint_state": torch.tensor(joint0, dtype=torch.float64),
                         "deterministic": True})
    env_n.reset(options={"joint_state": joint0, "deterministic": True})

    rng = np.random.default_rng(0)
    for _ in range(8):
        a = rng.uniform(0.05, 0.9, size=(1, nm))
        env_t.step(torch.tensor(a, dtype=torch.float64), deterministic=True)
        env_n.step(a, deterministic=True)
        ot = env_t.get_obs(deterministic=True)
        on = env_n.get_obs(deterministic=True)
        at = _action_slice(ot.detach().numpy() if hasattr(ot, "detach") else ot, nm)
        an = _action_slice(on, nm)
        assert np.max(np.abs(at - an)) < 1e-6, (
            f"action history diverged: max|Δ|={np.max(np.abs(at - an)):.2e}"
        )


def test_none_action_does_not_drop_history_torch():
    """Calling update with action_2d=None must not drop/duplicate past actions."""
    env_t, nm = _build_torch()
    env_t.reset(options={"deterministic": True})
    a1 = torch.full((1, nm), 0.4, dtype=torch.float64)
    a2 = torch.full((1, nm), 0.7, dtype=torch.float64)
    env_t.update_obs_buffer(a1)
    env_t.update_obs_buffer(a2)
    before = [None if x is None else x.clone() for x in env_t.obs_buffer["action"]]
    env_t.update_obs_buffer(None)  # no real action applied
    after = env_t.obs_buffer["action"]
    assert len(after) == len(before)
    for b, a in zip(before, after):
        assert torch.equal(b, a), "None action mutated the action history"
