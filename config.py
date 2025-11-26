from dataclasses import dataclass, field
import numpy as np


@dataclass
class PlantConfig:
    timestep: float = 0.002
    damping: float = 2.0
    n_ministeps: int = 1
    integration_method: str = "Euler"
    max_ep_duration: float = 5.0
    q0_deg: tuple = (55.0, 65.0)
    qd0: tuple = (0.0, 0.0)


@dataclass
class ControlToggles:
    enable_internal_force: bool = False
    enable_inertia_comp: bool = True
    enable_gravity_comp: bool = False
    enable_velocity_comp: bool = True
    enable_joint_damping: bool = True
    enable_joint_damping: bool = True


@dataclass
class ControlGains:
    # PD/IF controller gains
    Kp_x: np.ndarray = field(default_factory=lambda: np.array([800.0, 800.0]))
    Kff_x: np.ndarray = field(default_factory=lambda: np.array([1.5, 1.5]))
    Kp_q: np.ndarray = field(default_factory=lambda: np.array([14.0, 12.0]))
    Kd_q: np.ndarray = field(default_factory=lambda: np.array([1.6, 1.4]))

    # Energy tank controller gains (task-space)
    # D0: passive damping matrix in task space (2x2, SPD)
    D0: np.ndarray = field(
        default_factory=lambda: np.diag([20.0, 20.0])
    )

    # K0: nominal stiffness matrix in task space (2x2, SPD)
    K0: np.ndarray = field(
        default_factory=lambda: np.diag([800.0, 800.0])
    )

    # KI: integral gain vector (per task dimension)
    KI: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0])
    )

    # Imax: integral windup limits per dimension
    Imax: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.05])
    )


@dataclass
class Numerics:
    eps: float = 1e-8
    lam_os_smin_target: float = 1e-2
    lam_os_max: float = 1e-2
    sigma_thresh: float = 5e-2
    gate_pow: float = 2.0
    linesearch_eps: float = 1e-6
    linesearch_safety: float = 0.99
    lam_dls: float = 1e-4


@dataclass
class InternalForceConfig:
    cocon_a0: float = 0.12
    bisect_iters: int = 22


@dataclass
class TrajectoryConfig:
    Vmax: float = 0.8
    Amax: float = 4.0
    Jmax: float = 30.0
    gamma_time_scale: float = 1.10


@dataclass
class RunConfig:
    seed: int = 0
    deterministic_env: bool = True
    animate: bool = True
    downsample_anim: int = 3
    playback: float = 1.0
