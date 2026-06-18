# Tests

Regression tests guarding the audit fixes (see `../AUDIT_REPORT.md`).

## Setup

A project virtualenv lives at `../.venv` (gitignored). To recreate it:

```bash
python3 -m venv .venv
# GPU (CUDA 12.4 — e.g. NVIDIA RTX A1000):
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
# ...or CPU only:
# .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python -m pytest tests/ -q
```

The reach tests (`test_controllers_reach.py`, `test_nmpc_clamp.py`) run short
closed-loop simulations and take a couple of minutes total. The GPU smoke test
(`test_gpu_smoke.py`) auto-skips when no CUDA device is present.

## What each file covers

| File | Audit item |
|------|-----------|
| `test_dynamics_parity.py` | C1 (numpy freeze) + numpy↔torch M parity + energy conservation |
| `test_env_obs_parity.py`  | H1 (action-frame-stacking parity) |
| `test_controllers_reach.py` | H2/H3 (sliding-mode) + PD/IF tracking |
| `test_nmpc_clamp.py`      | H4 (NMPC box-constraint clamp-in-loop) |
| `test_cbf_qp_fallback.py` | H5 (CBF-QP passivity fallback) |
| `test_anfis_rls.py`       | H6 (ANFIS RLS raw-output regression) |
| `test_gpu_smoke.py`       | CUDA dynamics parity + on-device gradient flow |
