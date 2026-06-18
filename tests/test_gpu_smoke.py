"""
GPU smoke test: the differentiable torch stack must run on CUDA and keep
gradients flowing on-device. Skipped automatically when no CUDA GPU is present
(so CPU-only machines still pass the suite).

Run:  python -m pytest tests/test_gpu_smoke.py -q
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA GPU available"
)


def _build_arm(device):
    from model_lib.torch.skeleton import TwoDofArm
    return TwoDofArm(device=device, dtype=torch.float64)


def test_dynamics_on_cuda_matches_cpu():
    import lib.dynamics.torch.DynamicsHTM as DYN
    q = [0.7, 1.1]
    arm_cpu = _build_arm("cpu")
    arm_gpu = _build_arm("cuda")

    arm_cpu._robot.q = torch.tensor(q, dtype=torch.float64).reshape(1, 2)
    arm_gpu._robot.q = torch.tensor(q, dtype=torch.float64, device="cuda").reshape(1, 2)

    M_cpu = DYN.inertiaMatrixCOM(arm_cpu._robot).detach().cpu()
    M_gpu = DYN.inertiaMatrixCOM(arm_gpu._robot).detach().cpu()
    assert torch.allclose(M_cpu, M_gpu, atol=1e-9), "CUDA dynamics disagree with CPU"


def test_gradient_flows_on_cuda():
    """End-to-end gradient (loss -> fingertip -> ... -> activation) on the GPU."""
    from model_lib.torch.environment import Environment
    from model_lib.torch.muscles import RigidTendonHillMuscle
    from model_lib.torch.effector import RigidTendonArm26

    dev = torch.device("cuda")
    mus = RigidTendonHillMuscle(min_activation=0.02, device=dev, dtype=torch.float64)
    arm = RigidTendonArm26(muscle=mus, timestep=0.01, damping=0.0,
                           integration_method="rk4", device=dev, dtype=torch.float64)
    env = Environment(effector=arm, max_ep_duration=1.0, action_noise=0.0,
                      obs_noise=0.0, proprioception_delay=arm.dt, vision_delay=arm.dt)
    q0 = torch.deg2rad(torch.tensor([55.0, 65.0], device=dev, dtype=torch.float64))
    joint0 = torch.cat([q0, torch.zeros(2, device=dev, dtype=torch.float64)]).unsqueeze(0)
    env.reset(options={"joint_state": joint0, "deterministic": True})

    nm = env.n_muscles
    act = torch.full((1, nm), 0.3, device=dev, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0.0, 0.45]], device=dev, dtype=torch.float64)
    for _ in range(10):
        env.step(act, deterministic=True)
    ft = env.states["fingertip"][:, :2]
    loss = ((ft - target) ** 2).sum()
    loss.backward()
    assert act.grad is not None
    assert torch.isfinite(act.grad).all()
    assert float(act.grad.abs().sum()) > 0
    assert act.grad.is_cuda
