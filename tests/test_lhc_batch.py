"""
sensitivity/lhc_analysis: the vectorized torque_model_batch must match the
per-sample loop (to floating-point tolerance) on the same LHC samples.

Run:  python -m pytest tests/test_lhc_batch.py -q
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sensitivity.lhc_analysis import (
    monte_carlo_propagation, torque_model, torque_model_batch, DEFAULT_PARAMS,
)


def test_batch_matches_loop():
    loop = monte_carlo_propagation(torque_model, DEFAULT_PARAMS, 2000, seed=7, verbose=False)
    batch = monte_carlo_propagation(torque_model, DEFAULT_PARAMS, 2000, seed=7,
                                    verbose=False, batch_model_func=torque_model_batch)
    assert np.max(np.abs(loop.outputs - batch.outputs)) < 1e-10
    assert np.max(np.abs(loop.output_mean - batch.output_mean)) < 1e-10
    assert np.max(np.abs(loop.output_std - batch.output_std)) < 1e-10
