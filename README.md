# MuscleDrivenArm — Learning-Based Controllers

Learning-based control of a **planar two-link, six-muscle human arm**. This branch
holds the data-driven and learned controllers built on the same musculoskeletal
plant as the classical controllers — arm anatomy, Hill-type muscle mechanics,
tendon-excursion kinematics, and Euler–Lagrange dynamics — **with the trained
models and datasets bundled in** so everything runs out of the box.

> ### 🌿 Two branches, two controller families
> | Branch | Contents |
> |--------|----------|
> | **`main`** | **Classical / physics-based** controllers: impedance (PD+IF), passivity-based, sliding-mode, operational-space, plus the benchmark, sensitivity and EMG-validation pipelines. Lightweight — no bundled data. |
> | **`learning`** *(this branch)* | **Learning-based** controllers (MotorNet, muscle synergies, ANFIS, behaviour-cloning, MPC/RL) **with the trained models and datasets** bundled in. |
> |
> Both branches share the same plant (`lib/`, `model_lib/`, `muscles/`) and
> simulation infrastructure, so results are directly comparable.
> Switch with `git checkout main`.

## Controllers on this branch

| Controller | Module | What it is |
|------------|--------|------------|
| MotorNet policy | `controller/torch/motornet_controller.py` | recurrent policy trained by BPTT through the differentiable plant |
| Muscle synergy (full) | `controller/*/synergy_controller.py` | NMF synergy modules + modulation + residual correction |
| Muscle synergy (pure) | `controller/*/synergy_controller_pure.py` | fixed *K*-synergy activation basis |
| ANFIS | `controller/*/anfis_controller.py` | adaptive neuro-fuzzy inference with online RLS |
| Behaviour cloning (Hybrid-BC) | `controller/*/hybrid_bc_a.py` | activation policy cloned from an expert |
| Nonlinear MPC | `controller/*/nmpc_task.py` | receding-horizon task-space MPC |

**The classical PD+IF controller is retained here as the *expert***
(`controller/numpy/pd_if_controller.py`): the MotorNet, synergy and
behaviour-cloning pipelines generate their training demonstrations by rolling it
out. The full classical controller suite lives on the `main` branch.

## Bundled models & data

These are committed to this branch so the controllers run without a training step:

```
data/random_reach_a_ds_v2.npz      reach demonstration dataset (BC / MPC)
models/random_reach_bc_a.pt        trained behaviour-cloning policy
motornet/saved_model/*.pt          trained MotorNet policies
motornet/training/dataset_u.npz    MotorNet training set
motornet_checkpoints/*.pt          MotorNet BPTT checkpoint
synergy/saved_model/W_model*.npz   NMF synergy bases (K = 2,3,4 + default)
synergy/training/act_dataset.npz   activation dataset for synergy extraction
synergy_model.npz                  packaged synergy model
```

## Layout

```
controller/        learning controllers (numpy + torch) + PD+IF expert
lib/  model_lib/  muscles/   shared plant: skeleton, Hill muscles, dynamics
sim/  trajectory/  tasks/     simulation loop, min-jerk & Lissajous refs, reach tasks
utils/  logging_tools/  plotting/   helpers, run logging, figures
config.py                     plant / gain / numerics configuration
scripts/
  MOTORNET/   train + run the MotorNet policy
  SYNERGY/    extract synergies (NMF) + run synergy controllers
  ANFIS/      run the ANFIS controller
  MPC/        nonlinear MPC + behaviour-cloning (data gen, training, run)
tests/                        controller API + reach regression tests
docs/                         audit, coverage ledger, optimization notes
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
# torch (CPU build shown; see tests/README.md for CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# run a reach with the bundled trained models
python scripts/MOTORNET/random_reach_main_motornet.py
python scripts/SYNERGY/main_random_reach_synergy.py
python scripts/ANFIS/main_random_reach.py
python scripts/MPC/main_random_reach_mpc.py
```

### Retraining (optional — trained models are already bundled)

```bash
python scripts/MOTORNET/train_motornet_torch.py                 # MotorNet (BPTT)
python scripts/SYNERGY/training/train_synergy_w_nmf.py          # synergy NMF bases
python scripts/MPC/train_rl/train_bc_random_reach_a.py          # behaviour cloning
```

## Citation

If you use this code, please cite the paper (BibTeX to be added on publication).

## License

Released under the [MIT License](LICENSE).
