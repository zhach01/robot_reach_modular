"""
The FAST training variant (model_lib/skeleton_torch_FAST) implements the same
closed-form 2-DOF dynamics as the (verified) canonical skeleton_torch analytic
path. Cross-check them so the FAST path is regression-guarded and any
optimization of it stays correct.

Run:  python -m pytest tests/test_fast_variant.py -q
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)


def _arms():
    from model_lib.skeleton_torch import TwoDofArm as Main
    from model_lib.skeleton_torch_FAST import TwoDofArm as Fast
    # identical inertial params so the closed forms must agree
    kw = dict(m1=1.864572, m2=1.534315, l1g=0.180496, l2g=0.181479,
              i1=0.013193, i2=0.020062, l1=0.309, l2=0.26, dtype=torch.float64)
    return Main(**kw), Fast(**kw)


@pytest.mark.parametrize("q,qd", [
    ([0.3, 0.5], [0.0, 0.0]),
    ([0.9, 1.2], [0.7, -0.4]),
    ([1.0, 0.7], [-0.5, 0.3]),
])
def test_fast_mass_coriolis_match_main(q, qd):
    main, fast = _arms()
    qt = torch.tensor([q], dtype=torch.float64)
    qdt = torch.tensor([qd], dtype=torch.float64)
    Mm = main.mass_matrix(qt)
    Mf = fast.mass_matrix(qt)
    assert torch.max(torch.abs(Mm - Mf)) < 1e-12, "FAST mass_matrix diverged"
    # Coriolis force C@qd (the physically meaningful quantity)
    Cm = main.coriolis_matrix(qt, qdt)
    Cf = fast.coriolis_matrix(qt, qdt)
    cm = torch.einsum("bij,bj->bi", Cm, qdt)
    cf = torch.einsum("bij,bj->bi", Cf, qdt)
    assert torch.max(torch.abs(cm - cf)) < 1e-12, "FAST coriolis diverged"
    # xy Jacobian (FAST.jacobian) vs main's geometric_jacobian linear rows
    Jf = fast.jacobian(qt)
    Jm = main.geometric_jacobian(qt)[:, 0:2, :]
    assert torch.max(torch.abs(Jf - Jm)) < 1e-12, "FAST jacobian diverged"


def test_fast_differentiable():
    _, fast = _arms()
    a = torch.tensor([[0.4, 0.6]], requires_grad=True)
    M = fast.mass_matrix(a)
    M.sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_fast_batched():
    _, fast = _arms()
    q = torch.rand(8, 2, dtype=torch.float64)
    M = fast.mass_matrix(q)
    assert M.shape == (8, 2, 2)
    assert torch.allclose(M, M.transpose(-1, -2))  # symmetric
