# MuscleDrivenArm — Classical Controllers

Physics-based control of a **planar two-link, six-muscle human arm**. This is the
reference implementation behind the paper

> **Physics-Based Control of a Planar Two-Link, Six-Muscle Human Arm:
> Classical Controllers with a Common Redundancy Layer.**

It couples a full musculoskeletal model — arm anatomy, Hill-type muscle mechanics,
tendon-excursion kinematics, Euler–Lagrange dynamics, and identified parameters —
to four classical controllers through **one shared redundancy-resolution layer**.
Each controller outputs a desired joint torque; a single muscle-force allocation
problem then maps that torque to admissible muscle forces via the posture-dependent
moment-arm matrix **W(θ)**.

> ### 🌿 Two branches, two controller families
> | Branch | Contents |
> |--------|----------|
> | **`main`** *(this branch)* | **Classical / physics-based** controllers: impedance (PD+IF), passivity-based, sliding-mode, operational-space, plus the benchmark, sensitivity and EMG-validation pipelines. Lightweight — no bundled data. |
> | **`learning`** | **Learning-based** controllers (MotorNet, muscle synergies, ANFIS, behaviour-cloning, MPC/RL) **with the trained models and datasets** bundled in. |
> |
> Both branches share the same plant (`lib/`, `model_lib/`, `muscles/`) and
> simulation infrastructure, so results are directly comparable.
> Switch with `git checkout learning`.

## Controllers on this branch

| Controller | Module | Paper family |
|------------|--------|--------------|
| Impedance / PD+IF | `controller/numpy/pd_if_controller.py` | Impedance control |
| Passivity-based (energy tank) | `controller/numpy/energy_tank_controller.py` | Passivity-based control |
| Sliding-mode | `controller/numpy/sliding_mode.py` | Sliding-mode control |
| Operational-space | `controller/numpy/osc_controller.py` | Operational-space control |
| Predictive (benchmark lead) | `controller/numpy/predictive.py` | — |

Each has a NumPy reference implementation and, where relevant, a Torch counterpart
under `controller/torch/`.

## Layout

```
controller/        classical controllers (numpy + torch)
lib/  model_lib/  muscles/   shared plant: skeleton, Hill muscles, dynamics
sim/  trajectory/  tasks/     simulation loop, min-jerk & Lissajous refs, reach tasks
utils/  logging_tools/  plotting/   helpers, run logging, figures
config.py                     plant / gain / numerics configuration
scripts/
  PD_IF/  PASSIVITY/  SLIDING/  OSC/   per-controller reaching runs
  BENCHMARK/                          cross-controller benchmark + robustness
sensitivity/                          Section-VIII Sobol/Monte-Carlo parameter study
figure_repro_emg/  emg_*.py           Section-XI EMG-validation figures
kinarm_replay_validation.py           closed-loop replay of measured KINARM reaching
tests/                                unit tests for the plant and controllers
docs/                                 audit, coverage ledger, optimization notes
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# a single reaching run with the impedance (PD+IF) controller
python scripts/PD_IF/main_random_reach.py

# the full cross-controller benchmark
python scripts/BENCHMARK/run_benchmark.py

# Section-VIII parameter sensitivity study (Sobol / Monte-Carlo)
python -m sensitivity.sobol_analysis
python -m sensitivity.mc_propagation
```

## Citation

If you use this code, please cite the paper (BibTeX to be added on publication).

## License

Released under the [MIT License](LICENSE).
