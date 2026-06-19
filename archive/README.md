# Archive — superseded variants

Kept for reference; not on any runtime path.

## PD-IF
- `PD_IF/main_random_reach_legacy.py` — the original PD-IF random-reach demo
  (used `PDIFController` from the old `pd_if_controller`). Superseded by the
  optimized variant, which tracks ~1.8× tighter (RMSE 0.97 vs 1.72 mm, max 2.95
  vs 5.53 mm). The optimized controller is now the canonical
  `controller/numpy/pd_if_controller.py`; the old controller was renamed
  `controller/numpy/pd_if_legacy.py` (still used by center-out + synergy/motornet
  training, which depend on its parameter API).

## PD-IF legacy controllers (archived — no longer used anywhere)
- `controller/numpy/pd_if_legacy.py`, `controller/torch/pd_if_legacy.py` — the
  original (pre-optimization) PD-IF controllers with the Kp_x gain API. All
  scripts and tests now use the canonical optimized `pd_if_controller`
  (numpy + torch). Kept here only for historical reference.

## Passivity: faulty / superseded implementations (archived)
Comparison on a reach (lower RMSE = better tracking):
- `controller/numpy/energy_tank_controller.py` (canonical, **2.15 mm**) — kept.
- `controller/numpy/energy_tank_cbf_qp.py` — **FAULTY: diverges to 15.9 cm** (max
  49 cm). Confirmed a control-formulation bug, not the solver: it still diverges
  with a real QP solver (osqp). Its docstring claims <1 cm. Archived.
- `controller/numpy/energy_tank_hybrid.py` — mediocre (2.4 cm). Archived.
- `scripts/PASSIVITY/random_reach_main_hybrid{,2}.py` — demos/comparison of the
  two archived controllers (hybrid2 even recommended Hybrid without comparing
  against the much-better EnergyTankController). Archived.
- `tests/test_cbf_qp_fallback.py` — tested the QP fallback inside the archived
  CBF-QP. Removed.

## Passivity: energy_tank_v3_complete (strict-passivity variant, archived)
A "comprehensive" energy tank (task-energy preview, gain scheduling, variable
damping) that keeps STRICT passivity. Benchmarked vs the canonical
EnergyTankController (relaxed) on the same reaches:
  random-reach: current 1.05mm vs v3 1.99mm RMSE
  center-out:   current 3.89mm vs v3 24mm RMSE
v3 is worse for free-motion reaching (strict passivity costs tracking, as the
literature predicts). Archived; retrieve it if a CONTACT/interaction task needs
the strict-passivity guarantee.
