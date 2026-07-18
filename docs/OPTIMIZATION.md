# Optimization notes

Canonical record of the performance work. All changes are verified against the
reference implementation and guarded by `tests/` (`.venv/bin/python -m pytest tests/ -q`).

## 1. General n-DOF torch dynamics/kinematics

The optimized HTM dynamics now live in the **canonical** torch modules
(`lib/dynamics/torch/DynamicsHTM`, `lib/kinematics/torch/HTM_kinematics`,
`lib/movements/torch/HTM`) — there is no separate `_OPTIMIZED` copy anymore; the
base modules *are* the optimized implementation, shared by `model_lib/torch/skeleton` and
every torch controller.

Bottlenecks removed (work for any n-DOF, not just 2-DOF):

| Issue (original) | Fix |
|------------------|-----|
| `autograd.functional.jacobian` for Coriolis (~70% of time) | batched finite-difference Christoffel via one vectorized inertia call + einsum |
| repeated `torch.eye()` / `torch.zeros()` | cached identity/zeros buffers |
| separate DH transform matmuls per joint | vectorized closed-form DH for the whole table (`_dh_transforms`) |
| no FK caching (recomputed for M, C, g) | FK memoized on a per-`q` version counter |
| Python loops over links | vectorized |

## 2. Analytic 2-DOF fast-path (`model_lib/torch/skeleton.TwoDofArm`)

For the 2-DOF arm, the plant computes `M(q)`, `C(q,q̇)`, `g(q)`, `J(q)`, FK and
frames from **closed-form analytic equations** behind a `use_analytic` flag
(default on; set `use_analytic=False` for the generic HTM path). The public
methods (`mass_matrix`, `coriolis_matrix`, `gravity_vector`,
`geometric_jacobian`, `geometric_jacobian_dot`) are rank-preserving and return
the exact HTM output format, so they are drop-ins for the torch controllers
(`controller/torch/`: pd_if_controller, sliding_mode, nmpc_task,
energy_tank_controller).

## 3. NumPy simulation backbone

The live per-timestep numpy path (`model_lib/numpy/environment`, `model_lib/numpy/effector`,
`model_lib/numpy/skeleton`, `sim/numpy/simulator`) was profiled and tightened — all
bit-identical:

- `environment_numpy._apply_noise`: zero-noise fast path (skips an obs-sized RNG draw).
- `np.split(x,2,axis=1)` → half-slicing at the per-step state-unpack sites.
- `skeleton_numpy._key`: single-array fast path for the dynamics-cache key.
- `sim/simulator_numpy.run`: hoist zero-load allocs + `state_name.index` out of the loop.
- `model_lib/numpy/muscles._integrate`: attribute localization + shared branch-mask hoist.
- `muscles/numpy/muscle_tools`: memoize muscle channel indices (saves ~66 `list.index`/step in the bisection).
- controllers: cache `get_Fmax_vec` + the flpe index; single eigendecomposition
  for the op-space `Kv` gain (`matrix_sqrt_isqrt_spd`).

Per-step CPU work for `env.step`: 0.714 → 0.539 s / 2000 steps (cProfile tottime).

## 4. Sensitivity Monte-Carlo

`sensitivity/lhc_analysis.torque_model_batch` vectorizes the 10k-sample loop:
116 → 9.2 ms (13×), outputs match the loop to 7e-15.

See `COVERAGE_LEDGER.md` for the per-file accounting and `AUDIT_REPORT.md` for
the correctness audit.
