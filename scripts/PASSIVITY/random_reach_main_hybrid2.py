#!/usr/bin/env python3
"""
scripts/run_random_reach_comparison_fixed.py
Fixed version with proper NumPy array handling
"""

import os, json, argparse, time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any

from model_lib.environment_numpy import Environment
from model_lib.muscles_numpy import RigidTendonHillMuscle
from model_lib.effector_numpy import RigidTendonArm26

from config import (
    PlantConfig, ControlToggles, ControlGains, Numerics,
    InternalForceConfig, TrajectoryConfig, RunConfig,
)

from tasks.random_reach_numpy import Task as RandomReachTask
from trajectory.minjerk_numpy import MinJerkLinearTrajectory, MinJerkParams
from sim.simulator_numpy import TargetReachSimulator

# Import both controllers
try:
    from controller.energy_tank_hybrid import HybridEnergyTankController, HybridEnergyTankParams
    HYBRID_AVAILABLE = True
except ImportError:
    print("Warning: HybridEnergyTankController not available")
    HYBRID_AVAILABLE = False

try:
    from controller.energy_tank_cbf_qp import CBFQPTankController, CBFQPTankParams
    CBF_QP_AVAILABLE = True
except ImportError:
    print("Warning: CBFQPTankController not available")
    CBF_QP_AVAILABLE = False


class ControllerRunResult:
    """Results from a single controller run."""
    def __init__(self, name, rmse_cm, max_error_cm, mean_energy, mean_s, 
                 energy_violations, computation_time_ms, trace=None, logs=None, 
                 x_log=None, xref_log=None):
        self.name = name
        self.rmse_cm = rmse_cm
        self.max_error_cm = max_error_cm
        self.mean_energy = mean_energy
        self.mean_s = mean_s
        self.energy_violations = energy_violations
        self.computation_time_ms = computation_time_ms
        self.trace = trace if trace is not None else []
        self.logs = logs
        # Handle numpy arrays properly
        self.x_log = x_log if x_log is not None else np.array([])
        self.xref_log = xref_log if xref_log is not None else np.array([])


def build_env(pc: PlantConfig):
    muscle = RigidTendonHillMuscle(min_activation=0.02)
    arm = RigidTendonArm26(
        muscle=muscle,
        timestep=pc.timestep,
        damping=pc.damping,
        n_ministeps=pc.n_ministeps,
        integration_method=pc.integration_method,
    )
    env = Environment(
        effector=arm,
        max_ep_duration=pc.max_ep_duration,
        action_noise=0.0,
        obs_noise=0.0,
        proprioception_delay=arm.dt,
        vision_delay=arm.dt,
        name="RandomReachEnv(Comparison)",
    )
    q0 = np.deg2rad(np.array(pc.q0_deg))
    qd0 = np.array(pc.qd0)
    env.reset(options={"joint_state": np.concatenate([q0, qd0])[None, :], "deterministic": True})
    return env, arm, q0


def _scalar(x, default=1.0) -> float:
    try:
        v = np.array(x, dtype=float).ravel()
        if v.size >= 1:
            return float(v[0])
    except Exception:
        pass
    return float(default)


def filter_params_for_class(params_dict, params_class):
    """Filter dictionary to only include parameters valid for the given class."""
    if hasattr(params_class, '__dataclass_fields__'):
        valid_fields = set(params_class.__dataclass_fields__.keys())
    else:
        import inspect
        valid_fields = set(inspect.signature(params_class.__init__).parameters.keys())
        valid_fields.discard('self')
    
    filtered = {}
    for key, value in params_dict.items():
        if key in valid_fields:
            filtered[key] = value
    
    return filtered


def extract_logs_data(logs) -> tuple[np.ndarray, np.ndarray]:
    """Extract position and reference data from logs."""
    x_log = []
    xref_log = []
    
    # Debug: print available attributes
    if hasattr(logs, '__dict__'):
        print(f"  Available log attributes: {list(logs.__dict__.keys())}")
    
    # Try different ways to get the data
    if hasattr(logs, 'x_log'):
        x_log = logs.x_log
        print(f"  Found x_log with {len(x_log)} entries")
    elif hasattr(logs, 'cart_log'):
        x_log = logs.cart_log
        print(f"  Found cart_log with {len(x_log)} entries")
    elif hasattr(logs, 'cartesian_log'):
        x_log = logs.cartesian_log
        print(f"  Found cartesian_log with {len(x_log)} entries")
    
    if hasattr(logs, 'xref_log'):
        xref_log = logs.xref_log
        print(f"  Found xref_log with {len(xref_log)} entries")
    elif hasattr(logs, 'xref_tuple_log'):
        xref_tuple = logs.xref_tuple_log
        if xref_tuple and len(xref_tuple) > 0:
            # Extract position from tuple (x_d, xd_d_g, xdd_d_g)
            xref_log = [x[0] for x in xref_tuple]
        print(f"  Found xref_tuple_log with {len(xref_tuple)} entries")
    
    # Convert to numpy arrays
    x_log_np = np.array(x_log) if len(x_log) > 0 else np.array([])
    xref_log_np = np.array(xref_log) if len(xref_log) > 0 else np.array([])
    
    # Debug shape
    if x_log_np.size > 0:
        print(f"  x_log shape: {x_log_np.shape}")
    if xref_log_np.size > 0:
        print(f"  xref_log shape: {xref_log_np.shape}")
    
    return x_log_np, xref_log_np


def compute_metrics_from_data(x_log: np.ndarray, xref_log: np.ndarray, 
                             trace: List, dt: float, Emin: float = 0.1) -> Dict:
    """Compute performance metrics from actual data."""
    metrics = {
        "rmse_cm": 0.0,
        "max_error_cm": 0.0,
        "mean_energy": 0.0,
        "mean_s": 1.0,
        "energy_violations_percent": 0.0,
    }
    
    # Compute tracking error from position data
    if x_log.size > 0 and xref_log.size > 0:
        # Ensure 2D arrays
        if x_log.ndim == 1:
            x_log = x_log.reshape(-1, 1)
        if xref_log.ndim == 1:
            xref_log = xref_log.reshape(-1, 1)
        
        # Ensure we have at least 2 columns for XY coordinates
        if x_log.shape[1] >= 2 and xref_log.shape[1] >= 2:
            n = min(len(x_log), len(xref_log))
            if n > 0:
                # Extract XY coordinates
                x = x_log[:n, :2]
                xref = xref_log[:n, :2]
                
                # Compute errors
                errors = np.linalg.norm(x - xref, axis=1)
                rmse_m = np.sqrt(np.mean(errors**2))
                metrics["rmse_cm"] = rmse_m * 100.0
                metrics["max_error_cm"] = np.max(errors) * 100.0
                
                print(f"  Computed RMSE: {metrics['rmse_cm']:.2f} cm from {n} data points")
    
    # Compute passivity metrics from trace
    if trace and len(trace) > 0:
        # Extract energy levels
        E_values = []
        s_values = []
        
        for entry in trace:
            if isinstance(entry, dict):
                if "E" in entry:
                    E_values.append(float(entry["E"]))
                if "s" in entry:
                    s_values.append(float(entry["s"]))
        
        if E_values:
            E_array = np.array(E_values)
            metrics["mean_energy"] = float(np.mean(E_array))
            metrics["energy_violations_percent"] = float(np.sum(E_array < Emin) / len(E_array) * 100.0)
            print(f"  Mean energy: {metrics['mean_energy']:.3f}, violations: {metrics['energy_violations_percent']:.1f}%")
        
        if s_values:
            metrics["mean_s"] = float(np.mean(np.array(s_values)))
            print(f"  Mean authority (s): {metrics['mean_s']:.3f}")
    
    return metrics


def run_controller(controller_name: str, controller_class, params_class, 
                   env, arm, q0, traj, steps, base_params: Dict) -> ControllerRunResult:
    """Run a single controller and return results."""
    print(f"\n{'='*80}")
    print(f"Running {controller_name}...")
    print(f"{'='*80}")
    
    # Filter parameters for this specific class
    filtered_params = filter_params_for_class(base_params, params_class)
    
    # Create controller
    params = params_class(**filtered_params)
    ctrl = controller_class(env, arm, params)
    
    # Reset controller
    ctrl.reset(q0)
    
    # Run simulation
    sim = TargetReachSimulator(env, arm, ctrl, traj, steps)
    start_time = time.time()
    logs = sim.run()
    end_time = time.time()
    
    # Extract position data from logs
    x_log, xref_log = extract_logs_data(logs)
    
    # Get trace (if available)
    trace = []
    if hasattr(ctrl, 'get_trace'):
        trace = ctrl.get_trace()
        print(f"  Got trace with {len(trace)} entries")
    
    # Compute metrics from actual data
    metrics = compute_metrics_from_data(x_log, xref_log, trace, arm.dt, getattr(params, 'Emin', 0.1))
    
    # Computation time per step (in milliseconds)
    total_time_ms = (end_time - start_time) * 1000
    computation_time_ms = total_time_ms / steps if steps > 0 else 0
    print(f"  Computation time: {computation_time_ms:.3f} ms/step")
    
    return ControllerRunResult(
        name=controller_name,
        rmse_cm=metrics["rmse_cm"],
        max_error_cm=metrics["max_error_cm"],
        mean_energy=metrics["mean_energy"],
        mean_s=metrics["mean_s"],
        energy_violations=metrics["energy_violations_percent"],
        computation_time_ms=computation_time_ms,
        trace=trace,
        logs=logs,
        x_log=x_log,
        xref_log=xref_log,
    )


def print_comparison_table(results: List[ControllerRunResult]):
    """Print a formatted comparison table."""
    print("\n" + "="*100)
    print("CONTROLLER COMPARISON RESULTS")
    print("="*100)
    
    if len(results) != 2:
        print(f"Error: Expected 2 controllers, got {len(results)}")
        return None, None, 0
    
    # Identify which is which
    tank_result = None
    cbf_result = None
    
    for result in results:
        if "Hybrid" in result.name:
            tank_result = result
        elif "CBF" in result.name:
            cbf_result = result
    
    if not tank_result or not cbf_result:
        print("Error: Could not identify both controllers")
        return None, None, 0
    
    # Header
    headers = ["Metric", "Hybrid Energy Tank", "CBF-QP Passivity", "Improvement"]
    print(f"{headers[0]:<25} {headers[1]:<25} {headers[2]:<25} {headers[3]:<20}")
    print("-"*100)
    
    # Data rows
    metrics_data = [
        ("RMSE (cm)", tank_result.rmse_cm, cbf_result.rmse_cm, True),
        ("Max Error (cm)", tank_result.max_error_cm, cbf_result.max_error_cm, True),
        ("Mean Energy", tank_result.mean_energy, cbf_result.mean_energy, False),
        ("Mean Authority (s)", tank_result.mean_s, cbf_result.mean_s, False),
        ("Energy Violations %", tank_result.energy_violations, cbf_result.energy_violations, True),
        ("Comp Time/Step (ms)", tank_result.computation_time_ms, cbf_result.computation_time_ms, True),
    ]
    
    for name, tank_val, cbf_val, calc_improvement in metrics_data:
        if calc_improvement:
            if name in ["RMSE (cm)", "Max Error (cm)", "Energy Violations %", "Comp Time/Step (ms)"]:
                # Lower is better
                if tank_val > 0:
                    improvement = ((tank_val - cbf_val) / tank_val * 100)
                else:
                    improvement = 0.0
            else:
                # Higher is better
                if tank_val > 0:
                    improvement = ((cbf_val - tank_val) / tank_val * 100)
                else:
                    improvement = 0.0
            impr_str = f"{improvement:>+19.1f}%"
        else:
            impr_str = "-"
        
        print(f"{name:<25} {tank_val:<25.3f} {cbf_val:<25.3f} {impr_str:<20}")
    
    print("-"*100)
    
    # Calculate overall RMSE improvement
    rmse_improvement = 0.0
    if tank_result.rmse_cm > 0:
        rmse_improvement = ((tank_result.rmse_cm - cbf_result.rmse_cm) / tank_result.rmse_cm * 100)
    
    # Summary
    print("\nSUMMARY:")
    print(f"  • CBF-QP provides {rmse_improvement:+.1f}% better tracking accuracy")
    
    if cbf_result.mean_s > tank_result.mean_s:
        s_improvement = ((cbf_result.mean_s - tank_result.mean_s) / tank_result.mean_s * 100)
        print(f"  • CBF-QP maintains {s_improvement:+.1f}% higher average authority")
    
    if tank_result.energy_violations > 0 or cbf_result.energy_violations > 0:
        violations_improvement = ((tank_result.energy_violations - cbf_result.energy_violations) / max(tank_result.energy_violations, 1) * 100)
        print(f"  • CBF-QP has {violations_improvement:+.1f}% fewer energy violations")
    
    time_improvement = ((tank_result.computation_time_ms - cbf_result.computation_time_ms) / tank_result.computation_time_ms * 100)
    print(f"  • Computational overhead: {time_improvement:+.1f}%")
    
    # Winner determination
    if cbf_result.rmse_cm < 1.0:
        print("\n🎯 ACHIEVED GOAL: CBF-QP controller reached sub-1cm RMSE!")
    elif cbf_result.rmse_cm < tank_result.rmse_cm:
        print(f"\n⚠️  Close but not there: CBF-QP RMSE = {cbf_result.rmse_cm:.2f} cm (target: <1.0 cm)")
    else:
        print(f"\n❌ CBF-QP not better: RMSE = {cbf_result.rmse_cm:.2f} cm vs Tank: {tank_result.rmse_cm:.2f} cm")
    
    return tank_result, cbf_result, rmse_improvement


def plot_trajectory_comparison(tank_result, cbf_result, task_center, task_targets, out_dir):
    """Plot trajectory comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Hybrid Tank trajectory
    ax = axes[0]
    if tank_result.x_log.size > 0 and tank_result.xref_log.size > 0:
        # Ensure we have XY data
        x_tank = tank_result.x_log[:, 0] if tank_result.x_log.shape[1] >= 2 else tank_result.x_log.flatten()
        y_tank = tank_result.x_log[:, 1] if tank_result.x_log.shape[1] >= 2 else np.zeros_like(x_tank)
        
        xref_tank = tank_result.xref_log[:, 0] if tank_result.xref_log.shape[1] >= 2 else tank_result.xref_log.flatten()
        yref_tank = tank_result.xref_log[:, 1] if tank_result.xref_log.shape[1] >= 2 else np.zeros_like(xref_tank)
        
        ax.plot(x_tank, y_tank, 'b-', alpha=0.7, linewidth=2, label='Actual')
        ax.plot(xref_tank, yref_tank, 'r--', alpha=0.7, linewidth=1, label='Desired')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Hybrid Tank: RMSE = {tank_result.rmse_cm:.2f}cm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot CBF-QP trajectory
    ax = axes[1]
    if cbf_result.x_log.size > 0 and cbf_result.xref_log.size > 0:
        # Ensure we have XY data
        x_cbf = cbf_result.x_log[:, 0] if cbf_result.x_log.shape[1] >= 2 else cbf_result.x_log.flatten()
        y_cbf = cbf_result.x_log[:, 1] if cbf_result.x_log.shape[1] >= 2 else np.zeros_like(x_cbf)
        
        xref_cbf = cbf_result.xref_log[:, 0] if cbf_result.xref_log.shape[1] >= 2 else cbf_result.xref_log.flatten()
        yref_cbf = cbf_result.xref_log[:, 1] if cbf_result.xref_log.shape[1] >= 2 else np.zeros_like(xref_cbf)
        
        ax.plot(x_cbf, y_cbf, 'g-', alpha=0.7, linewidth=2, label='Actual')
        ax.plot(xref_cbf, yref_cbf, 'r--', alpha=0.7, linewidth=1, label='Desired')
    
    ax.set_xlabel('X (m)')
    ax.set_title(f'CBF-QP: RMSE = {cbf_result.rmse_cm:.2f}cm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Add targets
    for ax in axes:
        if task_targets is not None:
            for target in task_targets:
                ax.plot(target[0], target[1], 'ro', markersize=8, alpha=0.5)
        if task_center is not None:
            ax.plot(task_center[0], task_center[1], 'kx', markersize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "trajectory_comparison.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved trajectory plot to: {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius", type=float, default=0.10)
    parser.add_argument("--n_points", type=int, default=4)
    parser.add_argument("--Kp", type=float, default=2000.0)
    parser.add_argument("--D", type=float, default=50.0)
    parser.add_argument("--w_base", type=float, default=0.25)
    parser.add_argument("--rho_mu", type=float, default=1.0)
    parser.add_argument("--cbf_alpha", type=float, default=2.0)
    parser.add_argument("--out_dir", type=str, default="controller_comparison_runs")
    parser.add_argument("--skip_plots", action="store_true", help="Skip generating comparison plots")
    args = parser.parse_args()
    
    if not HYBRID_AVAILABLE:
        print("Error: HybridEnergyTankController not available")
        return
    
    if not CBF_QP_AVAILABLE:
        print("Error: CBFQPTankController not available")
        return
    
    pc = PlantConfig()
    toggles = ControlToggles()
    gains = ControlGains()
    num = Numerics()
    ifc = InternalForceConfig()
    tc = TrajectoryConfig()
    run = RunConfig()
    
    # Build environment
    env, arm, q0 = build_env(pc)
    
    # Create trajectory
    task = RandomReachTask(n_points=args.n_points, radius=args.radius, seed=args.seed)
    waypoints = task.build_waypoints(env)
    traj = MinJerkLinearTrajectory(waypoints, MinJerkParams(tc.Vmax, tc.Amax, tc.Jmax, tc.gamma_time_scale))
    
    # Base parameters (common to both controllers)
    lam_os_max = float(getattr(num, "lam_os_max", 200.0))
    if lam_os_max < 1.0:
        lam_os_max = 200.0
    
    base_params_common = {
        "K0": np.diag([args.Kp, args.Kp]),
        "D0": np.diag([args.D, args.D]),
        "Kff": _scalar(getattr(gains, "Kff_x", 1.0), 1.0),
        "zeta": 0.85,
        "eps": float(getattr(num, "eps", 1e-6)),
        "lam_os_max": lam_os_max,
        "sigma_thresh": float(getattr(num, "sigma_thresh", 1e-4)),
        "gate_pow": float(getattr(num, "gate_pow", 2.0)),
        "enable_inertia_comp": bool(getattr(toggles, "enable_inertia_comp", True)),
        "enable_gravity_comp": bool(getattr(toggles, "enable_gravity_comp", True)),
        "enable_coriolis_comp": bool(getattr(toggles, "enable_velocity_comp", True)),
        "enable_joint_damping": bool(getattr(toggles, "enable_joint_damping", True)),
        "E0": 0.4,
        "Emin": 0.1,
        "Emax": 2.0,
        "harvest_gain": 1.0,
        "store_returned_energy": True,
        "eta_min": 0.25,
        "bisect_iters": int(getattr(ifc, "bisect_iters", 12)),
        "clamp_tau_to_muscles": True,
        "tau_cap_margin": 0.95,
        "enable_internal_force": bool(getattr(toggles, "enable_internal_force", False)),
        "cocon_a0": float(getattr(ifc, "cocon_a0", 0.0)),
        "linesearch_eps": float(getattr(num, "linesearch_eps", 1e-6)),
        "linesearch_safety": float(getattr(num, "linesearch_safety", 1.2)),
    }
    
    steps = int(pc.max_ep_duration / arm.dt)
    
    # Print configuration
    print("\n" + "="*80)
    print("CONTROLLER COMPARISON TEST")
    print("="*80)
    print(f"Seed: {args.seed} | Radius: {args.radius}m | Points: {args.n_points}")
    print(f"Kp: {args.Kp} | D: {args.D} | dt: {arm.dt:.4f}s | Steps: {steps}")
    print(f"CBF alpha: {args.cbf_alpha}")
    print(f"Output directory: {args.out_dir}")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Run controllers
    all_results = []
    
    # 1. Hybrid Energy Tank
    hybrid_params = base_params_common.copy()
    hybrid_params.update({
        "w_passivity_base": float(args.w_base),
        "adapt_passivity": True,
        "E_blend_lo": 0.10,
        "E_blend_hi": 0.35,
        "eta_blend_lo": 0.35,
        "eta_blend_hi": 0.85,
        "K_scale_min": 0.35,
        "rho_mu": float(args.rho_mu),
    })
    
    result1 = run_controller(
        "Hybrid Energy Tank",
        HybridEnergyTankController,
        HybridEnergyTankParams,
        env, arm, q0, traj, steps,
        hybrid_params
    )
    all_results.append(result1)
    
    # Reset environment for fair comparison
    env.reset(options={"joint_state": np.concatenate([q0, [0, 0]])[None, :], "deterministic": True})
    
    # 2. CBF-QP Passivity
    cbf_params = base_params_common.copy()
    # Add CBF-QP specific params
    cbf_params.update({
        "cbf_alpha": args.cbf_alpha,
        "cbf_E_min": 0.1,
        "qp_weight_tracking": 10.0,
        "qp_weight_control": 0.1,
        "use_cbf_qp": True,
    })
    
    result2 = run_controller(
        "CBF-QP Passivity",
        CBFQPTankController,
        CBFQPTankParams,
        env, arm, q0, traj, steps,
        cbf_params
    )
    all_results.append(result2)
    
    # Print comparison table
    tank_result, cbf_result, improvement = print_comparison_table(all_results)
    
    # Save detailed results
    tag = f"seed{args.seed}_r{args.radius:.2f}_Kp{args.Kp:.0f}_D{args.D:.0f}"
    results_path = os.path.join(args.out_dir, f"{tag}_comparison_results.json")
    
    results_dict = {
        "test_config": {
            "seed": args.seed,
            "radius": args.radius,
            "n_points": args.n_points,
            "Kp": args.Kp,
            "D": args.D,
            "cbf_alpha": args.cbf_alpha,
            "dt": float(arm.dt),
            "steps": steps,
        },
        "hybrid_energy_tank": {
            "rmse_cm": float(tank_result.rmse_cm),
            "max_error_cm": float(tank_result.max_error_cm),
            "mean_energy": float(tank_result.mean_energy),
            "mean_authority": float(tank_result.mean_s),
            "energy_violations_percent": float(tank_result.energy_violations),
            "computation_time_ms": float(tank_result.computation_time_ms),
            "data_points": int(tank_result.x_log.shape[0]) if tank_result.x_log.size > 0 else 0,
        },
        "cbf_qp_passivity": {
            "rmse_cm": float(cbf_result.rmse_cm),
            "max_error_cm": float(cbf_result.max_error_cm),
            "mean_energy": float(cbf_result.mean_energy),
            "mean_authority": float(cbf_result.mean_s),
            "energy_violations_percent": float(cbf_result.energy_violations),
            "computation_time_ms": float(cbf_result.computation_time_ms),
            "data_points": int(cbf_result.x_log.shape[0]) if cbf_result.x_log.size > 0 else 0,
        },
        "improvement": {
            "rmse_percent": float(improvement),
            "achieved_sub_1cm": cbf_result.rmse_cm < 1.0,
        },
    }
    
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSaved detailed results to: {results_path}")
    
    # Generate comparison plots if we have data
    if not args.skip_plots and tank_result.x_log.size > 0 and cbf_result.x_log.size > 0:
        plot_trajectory_comparison(tank_result, cbf_result, task.center, task.targets, args.out_dir)
    
    # Print final recommendation
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    
    if cbf_result.rmse_cm < 1.0:
        print("✅ USE CBF-QP CONTROLLER:")
        print(f"   • Achieves {cbf_result.rmse_cm:.2f}cm RMSE (sub-1cm goal ✓)")
        print(f"   • {improvement:+.1f}% better tracking than Hybrid Tank")
        print(f"   • Maintains good passivity ({cbf_result.energy_violations:.1f}% violations)")
    elif cbf_result.rmse_cm < tank_result.rmse_cm:
        print("✅ USE CBF-QP CONTROLLER:")
        print(f"   • Achieves {cbf_result.rmse_cm:.2f}cm RMSE")
        print(f"   • {improvement:+.1f}% better tracking than Hybrid Tank")
        if cbf_result.rmse_cm < 1.2:
            print("   • Very close to sub-1cm goal!")
        else:
            print("   • Consider tuning cbf_alpha or increasing Kp for sub-1cm performance")
    else:
        print("✅ USE HYBRID TANK CONTROLLER:")
        print(f"   • Simpler implementation")
        print(f"   • {tank_result.rmse_cm:.2f}cm RMSE")
        print(f"   • Lower computational cost")
        if tank_result.rmse_cm < 1.0:
            print("   • Already achieves sub-1cm goal! ✓")
    
    print("="*80)


if __name__ == "__main__":
    main()
