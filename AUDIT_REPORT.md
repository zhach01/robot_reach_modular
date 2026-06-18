# Full Audit — Differentiability, Correctness & Controller V&V
_Date: 2026-06-18 · Scope: numpy + torch stacks, all controllers, gradient integrity + physical correctness_

---

## ✅ Fixes applied & verified (this pass)
Regression tests live in `tests/` (run: `python -m pytest tests/ -q`). All green.

| # | Fix | Files | Verified by |
|---|---|---|---|
| C1 | **Separate joint/COM DH cache keys** — unfreezes numpy M/C/g | `lib/Robot.py` | `tests/test_dynamics_parity.py` (numpy now matches torch to 1e-9; M varies with pose) |
| — | **Deleted corrupted `htm_cache.pkl`** + versioned, self-healing cache loader + `.gitignore` | `model_lib/skeleton_numpy.py`, `.gitignore` | cache discards on version mismatch |
| H1 | **Action-frame-stacking parity** — torch no longer fabricates/drops actions when none applied | `model_lib/environment_torch.py` | `tests/test_env_obs_parity.py` (numpy↔torch action history identical) |
| H2 | **Sliding-mode torch: added missing `λ·e_v` velocity feedback** (was feedforward-only → diverged) | `controller/sliding_mode_torch.py` | `tests/test_controllers_reach.py` (370 mm → 0.7 mm) |
| H3 | **Sliding-mode torch: eta floors 0.50/0.35 → 0.0** (restore singularity gate) | `controller/sliding_mode_torch.py` | same reach test |
| M | **muscle_guard `lam_mus`: full-range `(1−gate)`** (was ~constant, never damped at singularities) | `utils/muscle_guard.py`, `utils/muscle_guard_torch.py` | reach tests still track |
| M | **NMPC guard: pass acceleration ref** (was passing velocity for the xdd slot) | `controller/nmpc_task.py`, `controller/nmpc_task_torch.py` | reach tests green |
| H4 | **NMPC clamp-in-loop box constraints** — projected-gradient solve enforces \|F_k\|≤Fmax across the horizon (+ `TODO` for an optional QP solver) | `controller/nmpc_task.py`, `controller/nmpc_task_torch.py` | `tests/test_nmpc_clamp.py` (tracks at full budget; \|F0\|=8.0 held under tight Fmax) |
| H5 | **CBF-QP fallback now enforces the constraint** — orthogonal half-space projection + box (old scaling couldn't satisfy negative bounds, returned zeros for n≠2) | `controller/energy_tank_cbf_qp.py` | `tests/test_cbf_qp_fallback.py` (5 cases) |
| H6 | **ANFIS-v4.1 RLS regresses raw `φ·θ`** instead of the attenuated applied torque | `controller/anfis_controller_v4_1.py` | `tests/test_anfis_rls.py` (error↓, independent of y_pred) |
| L | **Inertia `RᵀIR`→`RIRᵀ`** (world-frame inertia; correct for general n-DOF, no-op for this planar arm) | `lib/dynamics/DynamicsHTM.py`, `lib/dynamics/DynamicsHTM_torch_OPTIMIZED.py` | parity + energy tests green |

**Note on false positives caught by dynamic verification:** the static pass flagged a "CRITICAL torch `forwardCOMHTM` off-by-one" and an "H2 torch switching-sign error"; both were **refuted empirically** — torch dynamics are correct, and the `+K·sat(s)` sign is right (the real sliding bug was the missing `λ·e_v` term).

Energy conservation is now a regression test (`test_energy_conserved_free_swing`, 0.006%/s drift).

## ⏳ Remaining (lower-priority / your review)
- **NMPC**: a dedicated box-QP solver (`qpsolvers`/OSQP) would be exact/faster than the projected-gradient clamp — left as a `TODO` per your call (clamp-in-loop for now, no new deps).
- **`RᵀIR`→`RIRᵀ`** was applied to the canonical `inertiaMatrixCOM` only; the Coriolis-internal recompute and the **dead variant files** still carry the old form (a no-op for planar). Sweep them if the code is ever extended past the planar arm.
- **Coherence (recommend, not done — needs your sign-off to delete):** remove dead variants (`*_GENERAL_OPT`, `*_OPTIMIZED_GENERAL`, `*_RNEA`, `*_FAST`, `*(Copy)`, DQ files), the trailing-underscore dead methods (incl. the **buggy** `nmpc_task_torch._solve_horizon_one_`), and resolve the `EnergyTankController` name collision between `energy_tank_controller.py` and `energy_tank_v3_complete.py`.
- **`pd_if_optimized.py`** global-eta attenuation — non-canonical variant; fix if you intend to use it.

---

## How this was done
- Mapped the **live import graph** to separate canonical code from dead variants.
- 6 parallel static auditors read every in-scope file (dynamics/kinematics, plant, all controllers, glue).
- **Dynamic verification I ran myself** in an isolated venv (torch 2.12 CPU): numpy↔torch parity, energy conservation, end-to-end gradient flow, and a closed-loop reach. Several static claims were **empirically confirmed or refuted** — flagged below.

---

## TL;DR
| Stack | Verdict |
|---|---|
| **Torch** | ✅ **Correct & fully differentiable.** Dynamics match analytics, energy conserved to 0.006%/s, gradients flow end-to-end, PD/IF reach tracks to 0.37 mm. This is the trustworthy stack. |
| **NumPy** | ❌ **Physically broken.** A caching bug freezes M/C/g at the start configuration, so the numpy plant simulates a constant-inertia system regardless of pose. The committed `htm_cache.pkl` is corrupted the same way. **Any result produced by the numpy stack is invalid.** |

---

## CRITICAL (verified by running the code)

### C1 — NumPy COM dynamics are frozen at the initial configuration
**`lib/Robot.py:178` and `:222`** — `denavitHartenberg()` (joint DH) and `denavitHartenbergCOM()` (COM DH) **share `self._cache_q`**. The joint-DH call runs first each step and writes `_cache_q = q_current`; the COM-DH cache check is `_cache_q == qflat and _cache_coms == COMs`, which is then **always satisfied**, so `denavitHartenbergCOM()` never rebuilds after the first call.

Consequence: `M(q)`, `C(q,q̇)`, `g(q)` and all COM positions are stuck at the construction state (q=0). End-effector FK still updates (joint DH rebuilds normally), so reaches still *move* — which **hides** the bug.

Empirical proof (USE_CACHE off, live compute):
```
ode() qdd for tau=[1,0]:   q=[0.3,0.9] → [8.65,-19.20]
                           q=[1.2,0.4] → [8.65,-19.20]   # identical for every pose
                           q=[0.6,1.3] → [8.65,-19.20]
M(q) across elbow angles:  constant [0.4631, 0.1566, 0.1566, 0.0706]
committed htm_cache.pkl:    19,012 M entries → only 1 distinct matrix
```
Fix proven by monkeypatch (give COM its own cache rebuild) → M then varies correctly with elbow angle (`M[0,0]: 0.455 → 0.398 → 0.320`) and matches torch to machine precision.

**Fix:** give the COM DH its own cache key. Replace the single `self._cache_q` with separate `self._cache_q_joint` / `self._cache_q_com` (used by `denavitHartenberg` and `denavitHartenbergCOM` respectively). Then **delete the committed `model_lib/htm_cache.pkl`** (it was generated with the bug and must be regenerated).

> Until C1 is fixed, every numpy controller is being validated against a fake plant. The numpy↔torch "parity" you asked about cannot hold while this stands.

---

## Differentiability verdict (torch) — VERIFIED ✅
- **End-to-end gradient flow** from `loss = ‖fingertip − target‖²` back to muscle activations through 15 differentiable env steps: gradients finite, non-zero, `requires_grad` chain intact.
- **Energy conservation:** free swing, zero input, no damping → KE drift **0.006%/s** → torch `M` and `C` are correct *and mutually consistent* (passivity / Ṁ−2C skew-symmetry holds numerically).
- **Finite-difference Coriolis** in the OPTIMIZED torch path is **still differentiable** (the perturbation `q_plus = q.clone(); q_plus[k]+=dq` stays in the graph). It is a biased O(dq) estimate of the exact Christoffel term, not a gradient break. Fine for learning/control; not for exact-derivative validation. Default `dq=1e-3` is on the large side.
- **No graph breaks** in the action→fingertip path (no stray `.detach()/.item()/float()/.numpy()` on the forward path; in-place writes are confined to fresh non-leaf tensors).
- ⚠️ Gradients **are** severed inside the muscle-force allocator fallback (`utils/muscle_guard_torch.py:246` uses `float(smin.detach())` to branch into NNLS; NNLS active-set is non-differentiable). Callers that need gradients through τ→activations get zeros on the fallback path. Acceptable for control, a gap for learning through the allocator.

### REFUTED false positive
A static pass flagged a "CRITICAL off-by-one in torch `forwardCOMHTM` → torch dynamics wrong." **This is incorrect.** Direct numerical parity shows torch COM positions and `M/C/g` match the (unfrozen) numpy and hand-computed analytics to machine precision. The torch core is correct. (Good illustration of why the dynamic checks mattered.)

---

## HIGH

| ID | File:line | Issue |
|---|---|---|
| H1 | `model_lib/environment_numpy.py:183` vs `environment_torch.py:276` | **Action-frame-stacking buffer rolled under different conditions** (numpy gates on `action is not None`; torch on `stacking>0` and self-duplicates). With `action_frame_stacking>0` the observation vectors diverge between stacks. |
| H2 | `controller/sliding_mode_torch.py:607-608` | **Switching-term sign disagrees with the numpy version** for the same `F_eq+F_sw` structure (torch `+K·sat`, numpy `−K·sat`). They can't both be Lyapunov-decreasing — re-derive against the plant force direction. |
| H3 | `controller/sliding_mode_torch.py:598-605` | **η floors of 0.5/0.35 re-introduce what the numpy "FIX D" removed** — they keep 50% of the equivalent force and 35% of switching force alive *at singularities*, defeating the singularity gate. |
| H4 | `controller/nmpc_task.py:331` (+torch) | **NMPC is unconstrained.** Activation `[0,1]`/`Fmax` limits are **not** in the QP; only the realized first force is clipped post-hoc. It's "MPC-flavored LQR tracking," not constrained NMPC. |
| H5 | `controller/energy_tank_cbf_qp.py:307-321, 68-98` | **Passivity not guaranteed in the `qpsolvers`-absent fallback** (`solve_qp_fallback` can leave `A_ub·x > b_ub`); the barrier constrains signed total power, not injected power. The stated passivity guarantee is loose. |
| H6 | `controller/anfis_controller_v4_1.py:608` | **RLS regresses against the *attenuated/applied* torque**, not the raw rule output, so consequent gains are biased high after warmup. |

---

## MEDIUM (selected — full list in agent notes)

- **`lib/dynamics/DynamicsHTM.py:84-88`** — inertia uses `Rᵀ·I·R` instead of `R·I·Rᵀ`. Masked for this planar arm (only `Izz` is selected, invariant under the swap; confirmed by exact torch/numpy agreement & energy conservation), but wrong for any out-of-plane/3-D extension.
- **`controller/nmpc_task.py:262` (+torch `:501`)** — `op_space_guard_and_gate` receives `Xdref[0]` as **both** velocity and acceleration; the gate sees velocity-as-acceleration.
- **`controller/mpc_rl_hybrid.py:116-119`** — confidence gate has **no minimum teacher authority**; a confidently-wrong RL policy silences the teacher, and the DAgger logger then stops collecting corrections (self-reinforcing blind spot).
- **`utils/muscle_guard*.py`** — adaptive damping `lam_mus` is effectively constant (~0.0027–0.0038) regardless of conditioning, and slightly *decreases* as conditioning worsens. Double-squashing bug; likely intended `lam_min + (lam_max−lam_min)*(1−gate)`.
- **`controller/pd_if_optimized.py:265`** — multiplies the **whole** desired torque by `clip(η,0.3,1.0)`, attenuating tracking even when well-conditioned → steady-state error. (Other PDIF variants gate only the nullspace term.)
- **`controller/synergy_controller.py:50-72`** — "NMF synergies" are **hand-coded**, never extracted; tracking actually comes from a full-space residual force, so the synergy structure is largely cosmetic. `synergy_controller_pure.py` defaults to a *random* W and won't track without a learned W.
- **MotorNet** — three divergent pipelines with **mismatched muscle models** (`motornet_true` trains on ReluMuscle f32; `motornet_controller_torch` on RigidTendonHill f64; `motornet_fixed` deploys on the numpy Hill env). Weights are not interchangeable → train/deploy distribution shift. `motornet_true.py` also has debug `print`s and a fabricated `qd*0.3` velocity fallback (draft quality).
- **`model_lib/skeleton_numpy.py:74`** — even after C1, the numpy plant **quantizes q to 4 decimals** for its cache key, so numpy can never bit-match torch; disable the cache for any parity work.
- **`utils/kinematics_guard_torch.py`** — `manip_grad` mutates `robot.q` in place mid-graph and restores it; latent corruption hazard if `robot.q` is captured elsewhere in the same step.

---

## VERIFIED-OK (checked, correct)
- **Torch rigid-body core** (M symmetric-PSD, geometric Jacobian, Ṁ, gravity, DH) — correct; energy-conserving; differentiable.
- **Min-jerk trajectory** (numpy + torch) — 5th-order `10τ³−15τ⁴+6τ⁵`, exact time-derivatives, zero vel/accel boundary conditions, correct time-scaling and multi-waypoint stitching. Identical across stacks.
- **Torque sign convention** `τ = −R·F` (`A = −R` in the allocator) — consistent across **every** controller and both simulators.
- **Simulator `tau_real = −R·f`**, forward dynamics `qdd = M⁻¹(τ − Cq̇ − g)`, semi-implicit Euler + RK4 — correct and consistent numpy/torch.
- **PD/IF, OSC, energy-tank** error convention `e = x_d − x` with **+K** — correct sign everywhere (no destabilizing sign error in any error definition).
- **dynamics_guard gate** `η = clip(sminS/σ_thresh)`, `η²` — monotonic, in [0,1], NaN-safe at sminS=0.
- **Hill muscle models** (force-length/velocity, activation dynamics, min-activation, Fmax scaling) — formulas match line-for-line numpy↔torch.

---

## Coherence / project-health (you asked about this)
The repo carries **heavy duplication** that is itself a correctness hazard — it's how the same bug hides in some files and not others, and how `_cache_q`-style mistakes creep in:
- **Dead lib variants** never imported by the canonical path: `DynamicsHTM_torch_{FAST,GENERAL_OPT,OPTIMIZED_GENERAL,OPTIMIZED_GENERAL1,RNEA}`, the matching `HTM_kinematics_torch_*` / `HTM_torch_*`, `Robot_torch_GENERAL_OPT`, plus DQ/dual-quaternion files.
- **`(Copy)` / `_FAST` files** in `model_lib` and `controller` (e.g. `effector_torch (Copy).py`, `energy_tank_controller (Copy).py`).
- **Trailing-underscore dead duplicates** inside live modules (e.g. `_coriolis_christoffel_torch_`, `nmpc_task_torch._solve_horizon_one_` — the latter contains a *wrong* `mu_task`; a landmine if ever wired up).
- **Name collisions:** both `energy_tank_controller.py` and `energy_tank_v3_complete.py` export `EnergyTankController`/`EnergyTankParams`; confirm which the sim imports.
- `skeleton_torch.py` docstring names the non-OPTIMIZED modules while it actually imports the `_OPTIMIZED` ones.

Recommend: pick the canonical file per role, delete the rest (git history preserves them), and add a numpy↔torch parity test on the **real `Serial` robot** (the existing `test_*` use a mock with a different indexing convention, which is exactly why they miss C1).

---

## Recommended fix order
1. **C1** — separate joint/COM DH cache keys in `lib/Robot.py`; delete `htm_cache.pkl`. *(Unblocks all numpy validity.)*
2. **H1** — reconcile action-frame-stacking between numpy/torch envs.
3. **H2 / H3** — fix sliding_mode_torch switching sign and remove the η floors.
4. **H4 / H5** — decide whether NMPC/CBF need true constraints/passivity, or relabel them.
5. **H6** + ANFIS/synergy/motornet medium items — correctness vs. their stated claims.
6. **Coherence sweep** — delete dead variants, add the real-robot parity test.

> **Environment:** a project venv lives at `.venv` (gitignored) with CUDA torch
> 2.6.0+cu124 — verified `cuda available: True` on the NVIDIA RTX A1000. Run the
> suite with `.venv/bin/python -m pytest tests/ -q` (see `requirements.txt` and
> `tests/README.md`). The GPU is exercised by `tests/test_gpu_smoke.py`
> (CUDA↔CPU dynamics parity + on-device gradient flow), which auto-skips on
> CPU-only machines.
