"""
The analytic fast-path in skeleton_torch must be ON by default and produce the
SAME results as the generic HTM engine (machine precision for M/g/J/FK; the
small qdd gap is HTM's finite-difference Coriolis bias, where analytic is exact).

Run:  python -m pytest tests/test_fast_analytic.py -q
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)

from model_lib.skeleton_torch import TwoDofArm


def _arm():
    return TwoDofArm(dtype=torch.float64)


def test_analytic_enabled_by_default():
    assert _arm().use_analytic is True


def test_ode_matches_htm():
    arm = _arm()
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        js = torch.cat([(torch.rand(4, 4, generator=g) * 2 - 1)], 0)
        js[:, :2] *= 1.4
        inp = (torch.rand(4, 2, generator=g) * 2 - 1) * 2
        load = (torch.rand(4, 2, generator=g) * 2 - 1) * 0.5
        arm.use_analytic = True
        qa = arm.ode(inp, js, load)
        arm.use_analytic = False
        qh = arm.ode(inp, js, load)
        # analytic Coriolis is exact; HTM uses dq=1e-3 finite differences, so
        # the only gap is HTM's O(dq) bias -> a few e-4 on qdd.
        assert torch.max(torch.abs(qa - qh)) < 5e-3


def test_kinematics_match_htm_to_machine_precision():
    arm = _arm()
    g = torch.Generator().manual_seed(1)
    pc = torch.tensor([[0.0, 0.05, 0.01], [0.0, 0.0, 0.02]])
    body = torch.tensor([[1, 2, 2]])
    for _ in range(20):
        js = (torch.rand(3, 4, generator=g) * 2 - 1)
        js[:, :2] *= 1.4
        arm.use_analytic = True
        c_a = arm.joint2cartesian(js)
        xy_a, v_a, dq_a = arm.path2cartesian(pc, body, js)
        arm.use_analytic = False
        c_h = arm.joint2cartesian(js)
        xy_h, v_h, dq_h = arm.path2cartesian(pc, body, js)
        assert torch.max(torch.abs(c_a - c_h)) < 1e-10
        assert torch.max(torch.abs(xy_a - xy_h)) < 1e-10
        assert torch.max(torch.abs(v_a - v_h)) < 1e-10
        assert torch.max(torch.abs(dq_a - dq_h)) < 1e-10


def test_analytic_differentiable():
    arm = _arm()
    arm.use_analytic = True
    a = torch.tensor([[0.3, 0.5]], requires_grad=True)
    js = torch.cat([a, torch.zeros(1, 2)], dim=1)
    qdd = arm.ode(torch.tensor([[1.0, 0.0]]), js, torch.zeros(1, 2))
    qdd.sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert float(a.grad.abs().sum()) > 0


def test_analytic_faster_than_htm():
    """Sanity: analytic ode is at least 5x faster (typically ~50x)."""
    import time
    arm = _arm()
    inp = torch.tensor([[1.0, 0.5]]); js = torch.tensor([[0.5, 0.7, 0.3, -0.2]])
    load = torch.zeros(1, 2)

    def bench(n=300):
        for _ in range(5):
            arm.ode(inp, js, load)
        t = time.perf_counter()
        for _ in range(n):
            arm.ode(inp, js, load)
        return time.perf_counter() - t

    arm.use_analytic = True
    ta = bench()
    arm.use_analytic = False
    th = bench()
    assert th > 5 * ta, f"expected >=5x speedup, got {th/ta:.1f}x"
