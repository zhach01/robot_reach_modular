"""
Regression tests for the rigid-body dynamics core.

Covers audit finding C1: the numpy `lib.numpy.Robot.Serial` shared `_cache_q` between
`denavitHartenberg` (joint DH) and `denavitHartenbergCOM` (COM DH) froze the
COM-based dynamics (M, C, g) at the construction configuration.

Run:  python -m pytest tests/test_dynamics_parity.py -q
"""
import os
import sys
import math

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Disable the higher-level pickle cache so we exercise the live compute path.
import model_lib.numpy.skeleton as sk_np
sk_np.USE_CACHE = False

from model_lib.numpy.skeleton import TwoDofArm as ArmNP
import lib.dynamics.numpy.DynamicsHTM as DYN_NP
import lib.kinematics.numpy.HTM_kinematics as KIN_NP

torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)
from model_lib.torch.skeleton import TwoDofArm as ArmT
import lib.dynamics.torch.DynamicsHTM as DYN_T


def _M_np(arm, q):
    arm._robot.jointsPositions = np.array(q, float).reshape(-1, 1)
    arm._robot.jointsVelocities = np.zeros((2, 1))
    return np.asarray(DYN_NP.inertiaMatrixCOM(arm._robot), float)


def _com1_np(arm, q):
    arm._robot.jointsPositions = np.array(q, float).reshape(-1, 1)
    frames = KIN_NP.forwardCOMHTM(arm._robot)
    return np.asarray(frames[1], float)[0:2, 3]


def _M_torch(arm, q):
    arm._robot.q = torch.tensor(q, dtype=torch.float64).reshape(1, 2)
    arm._robot.qd = torch.zeros(1, 2, dtype=torch.float64)
    return np.squeeze(DYN_T.inertiaMatrixCOM(arm._robot).detach().cpu().numpy())


# --- C1: the inertia matrix must actually depend on the elbow angle -----------

def test_numpy_inertia_depends_on_elbow_angle():
    """M(q) must change with the elbow angle q2 (frozen -> all equal -> FAIL)."""
    arm = ArmNP()
    M_a = _M_np(arm, [0.3, 0.3])
    M_b = _M_np(arm, [0.3, 0.9])
    M_c = _M_np(arm, [0.3, 1.4])
    # Shoulder inertia M[0,0] must strictly decrease as the elbow flexes.
    assert M_a[0, 0] > M_b[0, 0] > M_c[0, 0], (
        f"M[0,0] not responding to elbow angle: "
        f"{M_a[0,0]:.5f}, {M_b[0,0]:.5f}, {M_c[0,0]:.5f}"
    )


def test_numpy_com_position_depends_on_shoulder_angle():
    """Link-1 COM must rotate with q1 (frozen -> stuck at [L1g, 0] -> FAIL)."""
    arm = ArmNP()
    c0 = _com1_np(arm, [0.0, 0.5])
    c1 = _com1_np(arm, [0.6, 0.5])
    assert np.linalg.norm(c0 - c1) > 1e-3, (
        f"COM1 frozen: {c0} vs {c1}"
    )


# --- numpy <-> torch parity (both must be correct, to machine precision) ------

@pytest.mark.parametrize("q", [[0.3, 0.5], [0.9, 1.2], [1.0, 0.7],
                               [math.radians(55), math.radians(65)]])
def test_numpy_torch_inertia_parity(q):
    arm_np = ArmNP()
    arm_t = ArmT(dtype=torch.float64)
    M_np = _M_np(arm_np, q)
    M_t = _M_torch(arm_t, q)
    assert np.max(np.abs(M_np - M_t)) < 1e-9, (
        f"numpy/torch M disagree at q={q}: max|Δ|={np.max(np.abs(M_np - M_t)):.2e}"
    )


def test_inertia_symmetric_and_pd():
    arm_t = ArmT(dtype=torch.float64)
    for q in ([0.3, 0.5], [1.2, 1.4]):
        M = _M_torch(arm_t, q)
        assert np.max(np.abs(M - M.T)) < 1e-12
        assert np.all(np.linalg.eigvalsh(M) > 0)


def test_energy_conserved_free_swing():
    """Zero input, no damping, in-plane (gravity-free): kinetic energy must be
    conserved -> confirms M and C are mutually consistent (passivity)."""
    arm_t = ArmT(dtype=torch.float64)
    sk = arm_t._robot

    def KE(q, qd):
        sk.q = q.reshape(1, 2)
        sk.qd = qd.reshape(1, 2)
        M = DYN_T.inertiaMatrixCOM(sk).squeeze(0)
        return float(0.5 * qd @ (M @ qd))

    q = torch.tensor([0.6, 0.8], dtype=torch.float64)
    qd = torch.tensor([1.5, -1.0], dtype=torch.float64)
    E0 = KE(q, qd)
    dt = 5e-4
    for _ in range(2000):
        sk.q = q.reshape(1, 2)
        sk.qd = qd.reshape(1, 2)
        M = DYN_T.inertiaMatrixCOM(sk).squeeze(0)
        C = DYN_T.centrifugalCoriolisCOM(sk).squeeze(0)
        qdd = torch.linalg.solve(M, -(C @ qd))
        qd = qd + dt * qdd
        q = q + dt * qd
    drift = abs(KE(q, qd) - E0) / E0
    assert drift < 1e-3, f"energy drift too large: {drift*100:.3f}%"
