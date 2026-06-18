"""
Golden regression for the NumPy muscle integrators.

model_lib/muscles_numpy._integrate was optimized (attribute localization +
mask hoisting) under the invariant that the output is bit-identical. These
reference force values were captured from the verified pre-optimization code;
any future change to the muscle math that moves them will fail here.

Run:  python -m pytest tests/test_muscle_numpy_golden.py -q
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model_lib.muscles_numpy import (
    RigidTendonHillMuscle,
    RigidTendonHillMuscleThelen,
)

REFERENCE_FORCE = {
    "RigidTendonHillMuscle": [1790.235733331944, 2000.8105257392212, 799.2802130602324],
    "RigidTendonHillMuscleThelen": [6493.159854887574, 1284.5496374338477, 156.31040751269245],
}


def _build(cls):
    m = cls(min_activation=0.02)
    m.build(
        timestep=0.01,
        max_isometric_force=[100, 200, 150],
        tendon_length=[0.1, 0.2, 0.15],
        optimal_muscle_length=[0.08, 0.09, 0.1],
        normalized_slack_muscle_length=1.4,
    )
    return m


def _integrate_once(m):
    geo = np.array([[[0.30, 0.40, 0.35], [-0.1, 0.2, 0.05]]])  # (1,2,3)
    ms = np.full((1, m.state_dim, 3), 0.3)
    sd = np.array([[[0.5, -0.3, 0.1]]])
    return m._integrate(m.dt, sd, ms, geo)


def test_rigid_tendon_hill_force_matches_reference():
    m = _build(RigidTendonHillMuscle)
    out = _integrate_once(m)
    ref = np.array(REFERENCE_FORCE["RigidTendonHillMuscle"])
    assert np.max(np.abs(out[0, -1, :] - ref)) < 1e-9


def test_thelen_force_matches_reference():
    m = _build(RigidTendonHillMuscleThelen)
    out = _integrate_once(m)
    ref = np.array(REFERENCE_FORCE["RigidTendonHillMuscleThelen"])
    assert np.max(np.abs(out[0, -1, :] - ref)) < 1e-9


def test_integrate_output_shape_and_finite():
    for cls in (RigidTendonHillMuscle, RigidTendonHillMuscleThelen):
        m = _build(cls)
        out = _integrate_once(m)
        assert out.shape == (1, m.state_dim, 3)
        assert np.isfinite(out).all()
