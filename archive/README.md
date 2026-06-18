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
