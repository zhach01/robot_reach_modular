# Coverage Ledger — every Python file accounted for

You said you weren't convinced all the files were read/optimized. Fair. This is
the honest, *verifiable* accounting. "156 files" counted `__pycache__/*.pyc`
build artifacts; the repo has **113 tracked `.py` source files**
(`git ls-files '*.py' | grep -v __pycache__ | wc -l`).

Of those, **86 were touched** (read + edited) across the audit + optimization
work. The remaining 27 are listed below, each with a concrete reason — and
critically, **whether anything live actually imports it**, measured with grep,
not asserted.

## Per-directory coverage

| dir | touched / total |
|-----|-----------------|
| scripts | 16 / 16 |
| lib | 15 / 16 |
| controller | 13 / 18 |
| model_lib | 10 / 14 |
| tests | 11 / 11 (all created) |
| utils | 8 / 15 |
| tasks | 3 / 7 |
| sim | 2 / 3 |
| muscles | 2 / 3 |
| trajectory | 1 / 3 |
| logging_tools | 1 / 3 |
| sensitivity | 1 / 1 |
| plotting | 2 / 2 |
| config.py | touched |

## The 27 untouched files — why

### A. Empty package markers (7) — nothing to optimize
`controller/__init__.py`, `model_lib/__init__.py`, `sim/__init__.py`,
`tasks/__init__.py`, `trajectory/__init__.py`, `utils/__init__.py`,
`logging_tools/__init__.py` — all 0 lines.

### B. Genuinely cold — 0 live importers (superseded by the torch stack)
Measured: nothing under `scripts/`, `tests/`, `controller/`, `model_lib/`,
`sim/` imports these in a live path.

| file | lines | note |
|------|-------|------|
| `lib/kinematics/DifferentialHTM.py` | 659 | numpy differential kinematics; the torch path is what runs |
| `model_lib/policies_numpy.py` | 653 | only its torch counterpart references the name |
| `lib/plots/Plot.py` | 359 | standalone figure script, not imported by any sim |
| `lib/dynamics/fastsymp.py` | 178 | symbolic helper, not on any runtime path |
| `model_lib/random_target_reach_numpy.py` | 73 | superseded by `_torch`; not live |
| `lib/dynamics/Solver.py` | 52 | not imported live |

Optimizing dead code would be noise. Flagged instead of silently "covered".

### C. Live but cold path — setup / once-per-episode / intrinsically cheap
Imported by live scripts, but not per-timestep hot (build-time, task defs, or a
handful of scalar ops). Their torch counterparts were already optimized where a
hot one exists (`minjerk_torch`, `log_buffer_torch`, `*_guard_torch`).

`trajectory/minjerk.py` (scalar sample), `lib/movements/HTM.py` (movement gen,
setup), `muscles/muscle_tools.py` (build-time), `logging_tools/log_buffer.py`
(cheap append/step), `tasks/{random_reach,center_out,base_task}.py` (task
defs), `utils/{kinematics_guard,dynamics_guard,gating,telemetry,linear_utils,
math_utils}.py` (small guards/helpers).

### D. Live numpy controllers — not optimized (low yield vs env.step)
`controller/{pd_if_controller,hybrid_bc_a,hybrid_mpc_rl,mpc_rl_hybrid}.py`.
Their **torch counterparts were optimized**. Per-step control-law cost is small
next to `env.step`; `sliding_mode.py` + `energy_tank_controller.py` carry the
audit's correctness edits.

### E. Live + HOT, but intentionally left — `model_lib/muscles_numpy.py` (925)
This is the one genuinely-hot file I did **not** rewrite, and I want to be
straight about why. `_integrate` (the `RigidTendonHillMuscle` ODE) is the #1
line in the profile (0.078 s / 2000 steps). I optimized the whole backbone
*around* it — `environment_numpy`, `effector_numpy`, `skeleton_numpy`,
`sim/simulator.py` — but the Hill-model math itself is dense, already vectorized
over `(B, 1, M)`, and its `np.where` branches are semantically required (force
differs by fiber regime). A rewrite there is high-risk for the differentiable
physics and ~5% yield. Left correct rather than risk a subtle bug.

## What WAS optimized in the numpy hot path (this round)
Profiled `env.step` with cProfile (2000 steps). All changes bit-identical,
guarded by `test_env_obs_parity` + `test_dynamics_parity`:

- `environment_numpy._apply_noise` — zero-noise fast path (skips an obs-sized
  RNG draw every call; N(0,0)≡0).
- `skeleton_numpy` + `effector_numpy` — `np.split(x,2,axis=1)` → half-slicing at
  5 per-step sites (skips `array_split`'s cumsum + list build).
- `skeleton_numpy._key` — single-array fast path for the M/J/F dynamics-cache
  key (skips concatenate; identical bytes). Cache hit-rate verified unchanged.
- `sim/simulator.py` — hoisted `np.zeros((1,2))` loads + `state_name.index` out
  of the rollout loop.

Per-step CPU work: **0.714 → 0.539 s / 2000 steps** (cProfile tottime).
