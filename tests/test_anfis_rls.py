"""
Audit H6: ANFIS-v4.1 RLS must regress the teacher onto the model's OWN raw
prediction phi·theta, not onto the attenuated applied torque the caller passes.
The update must therefore (a) reduce the model's prediction error toward the
target and (b) be independent of the y_pred argument.

Run:  python -m pytest tests/test_anfis_rls.py -q
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from controller.numpy.anfis_controller import ANFISModule


def _unit():
    return ANFISModule(n_mf=3, e_range=(-1.5, 1.5), ed_range=(-3.0, 3.0),
                          forgetting_factor=0.995, P0_scale=100.0,
                          inertia=0.05, omega_n=20.0, zeta=1.0)


def test_rls_reduces_model_prediction_error():
    m = _unit()
    e, ed = 0.4, -0.2
    y_target = 7.0
    tau0, phi = m.forward(e, ed)
    err0 = abs(y_target - tau0)
    # one RLS step (pass a deliberately WRONG attenuated y_pred to prove it's unused)
    m.update_rls(phi=phi, y_target=y_target, y_pred=0.123 * tau0, clip=50.0)
    tau1, _ = m.forward(e, ed)
    err1 = abs(y_target - tau1)
    assert err1 < err0, f"RLS did not reduce error: {err0:.3f} -> {err1:.3f}"


def test_rls_independent_of_y_pred_argument():
    e, ed, y_target = -0.3, 0.5, -4.0
    # two identical units, updated with very different y_pred values
    m1, m2 = _unit(), _unit()
    tau, phi = m1.forward(e, ed)
    _, phi2 = m2.forward(e, ed)
    m1.update_rls(phi=phi, y_target=y_target, y_pred=0.0, clip=50.0)
    m2.update_rls(phi=phi2, y_target=y_target, y_pred=999.0, clip=50.0)
    assert np.allclose(m1.consequent, m2.consequent), \
        "RLS update still depends on the (deprecated) y_pred argument"
