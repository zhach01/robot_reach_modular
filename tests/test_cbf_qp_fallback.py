"""
Audit H5: the energy-tank CBF-QP dependency-free fallback must actually enforce
the inequality (passivity) constraint and the box, not just scale toward origin.

Run:  python -m pytest tests/test_cbf_qp_fallback.py -q
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from controller.numpy.energy_tank_cbf_qp import solve_qp_fallback


def test_returns_unconstrained_when_feasible():
    H = np.diag([2.0, 3.0])
    f = np.array([-4.0, -6.0])           # unconstrained min at [2, 2]
    x = solve_qp_fallback(H, f, A_ub=np.array([1.0, 1.0]), b_ub=100.0,
                          lb=np.array([-10, -10]), ub=np.array([10, 10]))
    assert np.allclose(x, [2.0, 2.0], atol=1e-6)


def test_enforces_positive_inequality():
    H = np.eye(2)
    f = np.array([-5.0, -5.0])           # wants [5,5], but xd·F <= 3
    A = np.array([1.0, 1.0])             # constraint x0 + x1 <= 3
    x = solve_qp_fallback(H, f, A_ub=A, b_ub=3.0)
    assert A @ x <= 3.0 + 1e-6, f"inequality violated: {A @ x}"
    # closest feasible point to [5,5] on x0+x1=3 is [1.5,1.5]
    assert np.allclose(x, [1.5, 1.5], atol=1e-6)


def test_enforces_negative_upper_bound():
    """The old scaling could NOT satisfy a negative bound (would flip sign)."""
    H = np.eye(2)
    f = np.array([-2.0, -2.0])           # wants [2,2]
    A = np.array([1.0, 0.0])             # x0 <= -1   (negative bound)
    x = solve_qp_fallback(H, f, A_ub=A, b_ub=-1.0)
    assert A @ x <= -1.0 + 1e-6, f"negative bound violated: x0={x[0]}"


def test_box_and_inequality_together():
    H = np.eye(2)
    f = np.array([-5.0, -5.0])
    A = np.array([1.0, 1.0])
    x = solve_qp_fallback(H, f, A_ub=A, b_ub=3.0,
                          lb=np.array([-10, -10]), ub=np.array([1.0, 10.0]))
    assert x[0] <= 1.0 + 1e-6 and x[1] <= 10.0 + 1e-6
    assert A @ x <= 3.0 + 1e-6


def test_general_dimension_not_zero():
    """n != 2 previously returned zeros; must now actually solve."""
    H = np.diag([1.0, 2.0, 4.0])
    f = np.array([-1.0, -2.0, -4.0])     # unconstrained min [1,1,1]
    x = solve_qp_fallback(H, f)
    assert np.allclose(x, [1.0, 1.0, 1.0], atol=1e-6)
