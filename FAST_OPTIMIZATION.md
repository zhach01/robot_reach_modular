# Torch fast-path optimization (`fast-torch-optimization` branch)

## What changed
The canonical torch plant (`model_lib/skeleton_torch.TwoDofArm`) now computes its
dynamics and kinematics with **closed-form analytic 2-DOF equations** instead of
the generic HTM/DH engine (transform matrices + geometric Jacobians + a
**finite-difference Coriolis**, which alone was ~72% of per-call cost).

- Added analytic `M(q)`, `C(q,q̇)`, `g(q)`, `J(q)`, FK, and frames, behind a
  `use_analytic` flag (default **on** for the 2-DOF arm; set `use_analytic=False`
  to force the generic HTM path). `_robot` is still kept in sync.
- `ode`, `joint2cartesian`, and `path2cartesian` use the analytic path.
- Public skeleton methods (`mass_matrix`, `coriolis_matrix`, `gravity_vector`,
  `geometric_jacobian`, `geometric_jacobian_dot`) return the **exact HTM output
  format**, so controllers can use them as drop-in replacements.
- **All four** torch controllers that computed their own dynamics —
  `pd_if_controller_torch`, `sliding_mode_torch`, `nmpc_task_torch`,
  `energy_tank_controller_torch` — are routed through these methods. The first
  two use the batched form; the latter two use the single-state `q[0]` form
  (the public methods are rank-preserving: 1-D input → unbatched output, mirroring
  HTM, so they are drop-ins for both call patterns).

## Correctness
Everything is verified to machine precision against the HTM engine and guarded
by `tests/` (run `.venv/bin/python -m pytest tests/ -q`):
- `M`, `g`, `J`, `J̇`, FK frames, EE position: **machine precision (1e-16)**.
- Coriolis: the analytic form is **exact**; the only gap vs HTM (~3e-4) is HTM's
  finite-difference bias, where analytic is the more correct value.
- Fully differentiable (CPU + CUDA), energy-conserving, reaches still track.
- `tests/test_fast_analytic.py` asserts analytic↔HTM parity + the speedup.

## Benchmarks (CPU, float64)
| Workload | HTM | Analytic | Speedup |
|---|---|---|---|
| `skeleton.ode` (single dynamics call) | 13.8 ms | 0.24 ms | **58×** |
| `skeleton.ode` (batched B=64) | 18.1 ms | 0.29 ms | **62×** |
| `env.step` (RL/policy rollout, B=1) | 57.7 ms | 2.3 ms | **25×** |
| `env.step` (RL/policy rollout, B=64) | 75.4 ms | 2.9 ms | **26×** |
| PD/IF closed-loop reach (2500 steps) | 64.9 s | 28.1 s | 2.3× |

The huge win lands on **simulation/training throughput** (RL, MotorNet BPTT),
where the plant dominates. A classical-controller reach speeds up less (2.3×)
because there the controller's muscle allocation + op-space guards dominate, not
the dynamics.

## nmpc_task_torch + energy_tank_controller_torch
These were orphaned (no script/test imports them) and use a different
single-state-then-broadcast dynamics pattern. They are now converted via the
rank-preserving public methods (`env.skeleton.mass_matrix(q[0])`, etc.) and
covered by `tests/test_orphan_controllers_reach.py` (short closed-loop reach,
finite + makes progress). Verified behavior-preserving: with the analytic path
the final fingertip matches the pre-swap HTM result to **0.02–0.04 mm** (just the
exact-vs-finite-difference Coriolis), at ~4× lower wall-clock on a short reach.
