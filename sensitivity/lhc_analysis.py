# sensitivity/lhc_analysis.py
"""
Local Sensitivity Analysis & Latin Hypercube Sampling

Implements first-order (local) sensitivity analysis and Monte Carlo
error propagation using Latin Hypercube Sampling (LHS).

Section VIII.D-F of the paper:
- Dimensionless sensitivity coefficients S_jk
- Latin Hypercube Monte Carlo error propagation
- Parameter ranking and uncertainty quantification

References:
- Saltelli et al. (2008): Global Sensitivity Analysis
- Iman & Conover (1980): Latin Hypercube Sampling
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings


# =============================================================================
# Parameter Configuration
# =============================================================================

@dataclass
class ParameterSpec:
    """Specification for a model parameter."""
    name: str
    nominal: float
    uncertainty: float  # relative uncertainty (e.g., 0.10 for ±10%)
    units: str = ""
    description: str = ""


# Default parameter specifications from Table V
DEFAULT_PARAMS = [
    ParameterSpec("a1", 0.035, 0.10, "m", "Shoulder flexor lever (mono)"),
    ParameterSpec("Lg2", 0.165, 0.10, "m", "Forearm COM distance"),
    ParameterSpec("Fmax_BIC", 414.0, 0.10, "N", "Biceps Fmax (bi-articular)"),
    ParameterSpec("m1", 1.82, 0.10, "kg", "Humerus mass"),
    ParameterSpec("a61", 0.025, 0.10, "m", "Triceps longus shoulder lever"),
    ParameterSpec("L1", 0.309, 0.10, "m", "Humerus length"),
    ParameterSpec("Fmax_PEC", 838.0, 0.10, "N", "Pectoralis Fmax"),
    ParameterSpec("I1", 0.051, 0.10, "kg·m²", "Moment of inertia"),
    ParameterSpec("l0", 0.10, 0.10, "m", "Optimal fibre length"),
    ParameterSpec("ls", 0.10, 0.10, "m", "Tendon slack length"),
    ParameterSpec("L2", 0.333, 0.10, "m", "Forearm length"),
    ParameterSpec("Lg1", 0.135, 0.10, "m", "Humerus COM distance"),
    ParameterSpec("m2", 1.43, 0.10, "kg", "Forearm mass"),
    ParameterSpec("I2", 0.057, 0.10, "kg·m²", "Forearm inertia"),
    ParameterSpec("Fmax_DEL", 1207.0, 0.10, "N", "Deltoid Fmax"),
    ParameterSpec("Fmax_BRA", 1422.0, 0.10, "N", "Brachialis Fmax"),
    ParameterSpec("Fmax_TRI_long", 603.0, 0.10, "N", "Triceps long Fmax"),
    ParameterSpec("Fmax_TRI_lat", 1549.0, 0.10, "N", "Triceps lat Fmax"),
    ParameterSpec("tau_act", 0.015, 0.05, "s", "Activation time constant"),
    ParameterSpec("D1", 0.5, 0.10, "N·m·s/rad", "Shoulder damping"),
    ParameterSpec("D2", 0.3, 0.10, "N·m·s/rad", "Elbow damping"),
]


# =============================================================================
# Local (First-Order) Sensitivity Analysis
# =============================================================================

def compute_local_sensitivity(
    model_func: Callable,
    params: Dict[str, float],
    param_name: str,
    output_idx: int = 0,
    delta: float = 1e-4,
    method: str = 'central',
) -> float:
    """
    Compute dimensionless sensitivity coefficient S_jk.
    
    S_jk = (p_k / τ_j) × (∂τ_j / ∂p_k)
    
    This is the log-derivative ∂ln(τ_j)/∂ln(p_k).
    
    Args:
        model_func: Function that takes params dict and returns outputs
        params: Nominal parameter values
        param_name: Name of parameter to perturb
        output_idx: Index of output to analyze (0=shoulder, 1=elbow)
        delta: Relative perturbation size
        method: 'forward', 'backward', or 'central' differences
    
    Returns:
        S_jk: Dimensionless sensitivity coefficient
    """
    p0 = params[param_name]
    tau0 = model_func(params)
    
    if isinstance(tau0, (list, np.ndarray)):
        tau0 = tau0[output_idx]
    
    # Handle near-zero outputs
    if abs(tau0) < 1e-10:
        warnings.warn(f"Output τ_{output_idx} ≈ 0; using absolute sensitivity")
        tau0 = 1e-6  # Small reference value
    
    # Perturbed evaluations
    params_plus = params.copy()
    params_plus[param_name] = p0 * (1 + delta)
    tau_plus = model_func(params_plus)
    if isinstance(tau_plus, (list, np.ndarray)):
        tau_plus = tau_plus[output_idx]
    
    if method == 'central':
        params_minus = params.copy()
        params_minus[param_name] = p0 * (1 - delta)
        tau_minus = model_func(params_minus)
        if isinstance(tau_minus, (list, np.ndarray)):
            tau_minus = tau_minus[output_idx]
        
        dtau_dp = (tau_plus - tau_minus) / (2 * delta * p0)
    elif method == 'forward':
        dtau_dp = (tau_plus - tau0) / (delta * p0)
    else:  # backward
        params_minus = params.copy()
        params_minus[param_name] = p0 * (1 - delta)
        tau_minus = model_func(params_minus)
        if isinstance(tau_minus, (list, np.ndarray)):
            tau_minus = tau_minus[output_idx]
        dtau_dp = (tau0 - tau_minus) / (delta * p0)
    
    # Dimensionless sensitivity
    S_jk = (p0 / tau0) * dtau_dp
    
    return S_jk


def compute_all_sensitivities(
    model_func: Callable,
    params: Dict[str, float],
    param_specs: List[ParameterSpec],
    output_idx: int = 0,
) -> Dict[str, float]:
    """
    Compute sensitivity coefficients for all parameters.
    
    Args:
        model_func: Model function
        params: Nominal parameters
        param_specs: List of parameter specifications
        output_idx: Output index (0=shoulder torque)
    
    Returns:
        Dictionary mapping parameter names to |S_jk| values
    """
    sensitivities = {}
    
    for spec in param_specs:
        if spec.name in params:
            S = compute_local_sensitivity(
                model_func, params, spec.name, output_idx
            )
            sensitivities[spec.name] = abs(S)
    
    return sensitivities


def rank_sensitivities(
    sensitivities: Dict[str, float],
    threshold: float = 0.2,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Rank parameters by sensitivity and classify as high/low.
    
    Args:
        sensitivities: Dictionary of |S_jk| values
        threshold: High-sensitivity threshold (default 0.2)
    
    Returns:
        high_sens: List of (name, S) for high-sensitivity parameters
        low_sens: List of (name, S) for low-sensitivity parameters
    """
    sorted_sens = sorted(sensitivities.items(), key=lambda x: -x[1])
    
    high_sens = [(name, S) for name, S in sorted_sens if S >= threshold]
    low_sens = [(name, S) for name, S in sorted_sens if S < threshold]
    
    return high_sens, low_sens


# =============================================================================
# Latin Hypercube Sampling
# =============================================================================

def latin_hypercube_sample(
    n_samples: int,
    param_specs: List[ParameterSpec],
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate Latin Hypercube samples for parameters.
    
    Each parameter is sampled uniformly within its ±uncertainty range,
    with LHS ensuring good space coverage.
    
    Args:
        n_samples: Number of samples
        param_specs: Parameter specifications
        seed: Random seed for reproducibility
    
    Returns:
        samples: Array of shape (n_samples, n_params)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_params = len(param_specs)
    samples = np.zeros((n_samples, n_params))
    
    for j, spec in enumerate(param_specs):
        # Generate LHS intervals
        intervals = np.linspace(0, 1, n_samples + 1)
        
        # Sample uniformly within each interval
        lower = intervals[:-1]
        upper = intervals[1:]
        points = np.random.uniform(lower, upper)
        
        # Shuffle to break correlations
        np.random.shuffle(points)
        
        # Map to parameter range [nominal*(1-unc), nominal*(1+unc)]
        low = spec.nominal * (1 - spec.uncertainty)
        high = spec.nominal * (1 + spec.uncertainty)
        samples[:, j] = low + points * (high - low)
    
    return samples


def lhc_sample_dict(
    sample_row: np.ndarray,
    param_specs: List[ParameterSpec],
) -> Dict[str, float]:
    """Convert LHC sample row to parameter dictionary."""
    return {spec.name: sample_row[i] for i, spec in enumerate(param_specs)}


# =============================================================================
# Monte Carlo Error Propagation
# =============================================================================

@dataclass
class MonteCarloResults:
    """Results from Monte Carlo error propagation."""
    n_samples: int
    output_mean: np.ndarray
    output_std: np.ndarray
    output_percentiles: Dict[int, np.ndarray]  # {5: ..., 50: ..., 95: ...}
    samples: np.ndarray
    outputs: np.ndarray


def monte_carlo_propagation(
    model_func: Callable,
    param_specs: List[ParameterSpec],
    n_samples: int = 10000,
    seed: Optional[int] = 42,
    verbose: bool = True,
    batch_model_func: Optional[Callable] = None,
) -> MonteCarloResults:
    """
    Monte Carlo error propagation using Latin Hypercube Sampling.
    
    Args:
        model_func: Model function taking params dict, returning array
        param_specs: Parameter specifications
        n_samples: Number of LHC samples
        seed: Random seed
        verbose: Print progress
    
    Returns:
        MonteCarloResults with statistics
    """
    # Generate LHC samples
    samples = latin_hypercube_sample(n_samples, param_specs, seed)

    # Evaluate the model. If a vectorized batch evaluator is supplied, run all
    # samples in one array op (10k-sample Python loop -> a few numpy ops).
    if batch_model_func is not None:
        outputs = np.atleast_2d(batch_model_func(samples, param_specs))
    else:
        outputs = np.array([
            np.atleast_1d(model_func(lhc_sample_dict(samples[i], param_specs)))
            for i in range(n_samples)
        ])
    
    # Compute statistics
    results = MonteCarloResults(
        n_samples=n_samples,
        output_mean=np.mean(outputs, axis=0),
        output_std=np.std(outputs, axis=0),
        output_percentiles={
            5: np.percentile(outputs, 5, axis=0),
            50: np.percentile(outputs, 50, axis=0),
            95: np.percentile(outputs, 95, axis=0),
        },
        samples=samples,
        outputs=outputs,
    )
    
    return results


# =============================================================================
# Torque Model for Sensitivity Analysis
# =============================================================================

def torque_model(
    params: Dict[str, float],
    theta: np.ndarray = np.array([np.radians(35), np.radians(50)]),
    theta_dot: np.ndarray = np.array([0.0, 0.0]),
    theta_ddot: np.ndarray = np.array([1.0, 0.5]),  # Small acceleration
    activations: np.ndarray = np.array([0.3, 0.3, 0.2, 0.4, 0.2, 0.2]),
) -> np.ndarray:
    """
    Simple torque model for sensitivity analysis.
    
    Computes joint torques given parameters, state, and muscle activations.
    
    Args:
        params: Model parameters
        theta: Joint angles (rad)
        theta_dot: Joint velocities (rad/s)
        theta_ddot: Joint accelerations (rad/s²)
        activations: Muscle activations (6,)
    
    Returns:
        tau: Joint torques (2,)
    """
    # Extract parameters with defaults
    L1 = params.get('L1', 0.27)
    L2 = params.get('L2', 0.26)
    Lg1 = params.get('Lg1', 0.135)
    Lg2 = params.get('Lg2', 0.165)
    m1 = params.get('m1', 1.82)
    m2 = params.get('m2', 1.44)
    I1 = params.get('I1', 0.012)
    I2 = params.get('I2', 0.018)
    
    a1 = params.get('a1', 0.035)
    a2 = params.get('a2', 0.028)
    a61 = params.get('a61', 0.025)
    a62 = params.get('a62', 0.024)
    
    Fmax = np.array([
        params.get('Fmax_DEL', 1142.6),
        params.get('Fmax_PEC', 699.8),
        params.get('Fmax_BRA', 987.3),
        params.get('Fmax_BIC', 780.0),
        params.get('Fmax_TRI_long', 798.5),
        params.get('Fmax_TRI_lat', 624.3),
    ])
    
    q1, q2 = theta
    qd1, qd2 = theta_dot
    qdd1, qdd2 = theta_ddot
    
    # Mass matrix
    c2 = np.cos(q2)
    M11 = I1 + I2 + m1*Lg1**2 + m2*(L1**2 + Lg2**2 + 2*L1*Lg2*c2)
    M12 = I2 + m2*(Lg2**2 + L1*Lg2*c2)
    M22 = I2 + m2*Lg2**2
    
    # Coriolis
    s2 = np.sin(q2)
    h = -m2 * L1 * Lg2 * s2
    
    # Inertial torques
    tau_inertia = np.array([
        M11*qdd1 + M12*qdd2 + h*qd2*(2*qd1 + qd2),
        M12*qdd1 + M22*qdd2 - h*qd1**2
    ])
    
    # Moment arm matrix (simplified)
    R = np.array([
        [a1, a1, 0, a1*0.85, -a61, 0],
        [0, 0, a2, a2, -a62, -a62]
    ])
    
    # Muscle forces
    F_muscle = activations * Fmax
    
    # Muscle torques
    tau_muscle = R @ F_muscle
    
    # Total torque (muscle - required for dynamics)
    tau = tau_muscle - tau_inertia

    return tau


def torque_model_batch(
    samples: np.ndarray,
    param_specs: "List[ParameterSpec]",
    theta: np.ndarray = np.array([np.radians(35), np.radians(50)]),
    theta_dot: np.ndarray = np.array([0.0, 0.0]),
    theta_ddot: np.ndarray = np.array([1.0, 0.5]),
    activations: np.ndarray = np.array([0.3, 0.3, 0.2, 0.4, 0.2, 0.2]),
) -> np.ndarray:
    """
    Vectorized torque_model over a batch of LHC samples.

    samples: (N, n_params) in the param_specs order.
    Returns: (N, 2) joint torques — equals looping torque_model() per row
    (to floating-point tolerance; the only difference is the 6-element muscle
    dot product done as a sum vs np.dot). Replaces the 10k-sample Python loop.
    """
    N = samples.shape[0]
    col = {spec.name: j for j, spec in enumerate(param_specs)}

    def P(name, default):
        return samples[:, col[name]] if name in col else np.full(N, default)

    L1 = P('L1', 0.27); L2 = P('L2', 0.26)
    Lg1 = P('Lg1', 0.135); Lg2 = P('Lg2', 0.165)
    m1 = P('m1', 1.82); m2 = P('m2', 1.44)
    I1 = P('I1', 0.012); I2 = P('I2', 0.018)
    a1 = P('a1', 0.035); a2 = P('a2', 0.028)
    a61 = P('a61', 0.025); a62 = P('a62', 0.024)
    Fmax = np.stack([
        P('Fmax_DEL', 1142.6), P('Fmax_PEC', 699.8), P('Fmax_BRA', 987.3),
        P('Fmax_BIC', 780.0), P('Fmax_TRI_long', 798.5), P('Fmax_TRI_lat', 624.3),
    ], axis=1)  # (N, 6)

    q2 = theta[1]; qd1, qd2 = theta_dot; qdd1, qdd2 = theta_ddot
    c2 = np.cos(q2); s2 = np.sin(q2)
    M11 = I1 + I2 + m1 * Lg1**2 + m2 * (L1**2 + Lg2**2 + 2 * L1 * Lg2 * c2)
    M12 = I2 + m2 * (Lg2**2 + L1 * Lg2 * c2)
    M22 = I2 + m2 * Lg2**2
    h = -m2 * L1 * Lg2 * s2
    tau_in0 = M11 * qdd1 + M12 * qdd2 + h * qd2 * (2 * qd1 + qd2)
    tau_in1 = M12 * qdd1 + M22 * qdd2 - h * qd1**2

    F = activations[None, :] * Fmax                      # (N, 6)
    z = np.zeros(N)
    R0 = np.stack([a1, a1, z, a1 * 0.85, -a61, z], axis=1)
    R1 = np.stack([z, z, a2, a2, -a62, -a62], axis=1)
    tau_mus0 = np.sum(R0 * F, axis=1)
    tau_mus1 = np.sum(R1 * F, axis=1)
    return np.stack([tau_mus0 - tau_in0, tau_mus1 - tau_in1], axis=1)  # (N, 2)


# =============================================================================
# Analysis Functions
# =============================================================================

def run_sensitivity_analysis(
    theta: np.ndarray = np.array([np.radians(35), np.radians(50)]),
    output_idx: int = 0,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Run complete local sensitivity analysis at a pose.
    
    Args:
        theta: Joint pose (rad)
        output_idx: 0 for shoulder, 1 for elbow
        verbose: Print results
    
    Returns:
        Dictionary of sensitivity coefficients
    """
    # Build nominal parameters
    nominal_params = {spec.name: spec.nominal for spec in DEFAULT_PARAMS}
    
    # Create model function for this pose
    def model_at_pose(params):
        return torque_model(params, theta=theta)
    
    # Compute sensitivities
    sensitivities = compute_all_sensitivities(
        model_at_pose, nominal_params, DEFAULT_PARAMS, output_idx
    )
    
    # Rank
    high_sens, low_sens = rank_sensitivities(sensitivities)
    
    if verbose:
        joint_name = "shoulder" if output_idx == 0 else "elbow"
        print(f"\n=== Sensitivity Analysis ({joint_name} torque) ===")
        print(f"Pose: θ₁={np.degrees(theta[0]):.0f}°, θ₂={np.degrees(theta[1]):.0f}°")
        print(f"\nHigh-sensitivity parameters (|S| ≥ 0.2):")
        for name, S in high_sens[:5]:
            print(f"  {name}: {S:.3f}")
        print(f"\nLow-sensitivity parameters (|S| < 0.2):")
        for name, S in low_sens[-3:]:
            print(f"  {name}: {S:.3f}")
    
    return sensitivities


def run_monte_carlo_analysis(
    n_samples: int = 10000,
    seed: int = 42,
    verbose: bool = True,
) -> MonteCarloResults:
    """
    Run Monte Carlo error propagation analysis.
    
    Args:
        n_samples: Number of LHC samples
        seed: Random seed
        verbose: Print progress and results
    
    Returns:
        MonteCarloResults object
    """
    if verbose:
        print(f"\n=== Monte Carlo Error Propagation ({n_samples} samples) ===")
    
    results = monte_carlo_propagation(
        torque_model, DEFAULT_PARAMS, n_samples, seed, verbose,
        batch_model_func=torque_model_batch,   # vectorized: ~50-100x fewer Python calls
    )
    
    if verbose:
        print(f"\nTorque variability:")
        print(f"  σ_τ1 = {results.output_std[0]:.2f} N·m")
        print(f"  σ_τ2 = {results.output_std[1]:.2f} N·m")
        print(f"\n95th percentile ranges:")
        print(f"  τ1: [{results.output_percentiles[5][0]:.2f}, "
              f"{results.output_percentiles[95][0]:.2f}] N·m")
        print(f"  τ2: [{results.output_percentiles[5][1]:.2f}, "
              f"{results.output_percentiles[95][1]:.2f}] N·m")
    
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Run local sensitivity analysis
    sens = run_sensitivity_analysis()
    
    # Run Monte Carlo (reduced samples for demo)
    mc_results = run_monte_carlo_analysis(n_samples=10000)
