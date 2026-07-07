# PINN + FNO: 1D Elastic Wave Equation Solver

A research codebase for solving the 1D elastic wave equation using Physics-Informed Neural Networks (PINNs). Three PINN architectures are benchmarked across homogeneous, two-layer, and multi-layer material models, with separate numerical solvers included for reference and comparison.

---

## Overview

This project trains and evaluates three neural network architectures on the 1D elastic wave PDE:

- **Vanilla PINN** — standard MLP with tanh activations
- **Fourier-Feature PINN** — random Fourier feature embedding to capture high-frequency behavior
- **PirateNet** — adaptive residual blocks (Wang et al. 2024)

Training uses a curriculum approach with LBFGS optimization. Vanilla and Fourier PINNs additionally use **R3 adaptive collocation** (Daw et al. 2023), which resamples collocation points based on residual magnitude. PirateNet uses fixed Sobol sampling with an Adam warmup phase.

A **ForwardCNN** (Task 3) is also included as a data-driven baseline trained on finite difference reference solutions.

---

## Experiments

Three material configurations are tested:

| Experiment | Material | Domain | Notes |
|---|---|---|---|
| `exp1` | Homogeneous | [-1, 1] | Uniform E=80, ρ=100 |
| `exp2` | Two-layer | [-1, 1] | Smooth interface via tanh at x=0 |
| `exp3` | Multi-layer | [-1.5, 1.5] | 6 layers, E from 60 to 150 |

A finite difference solver with absorbing boundary conditions generates the reference wavefield for each experiment. Models are evaluated by mean relative L2 error across several time snapshots.

---

## Numerical Lab

Separately from the PINN experiments, `physics.py` and `solvers.py` implement six classical numerical methods for the wave equation on a 10 km domain with a Ricker wavelet source:

- Finite Difference (FD)
- Pseudo-Spectral (PS)
- Finite Element (FEM)
- Finite Volume (FVM)
- Discontinuous Galerkin (DG)
- Spectral Element (SEM)

Results can be animated and compared via `solve.py`, `animate.py`, and `compare.py`.

---

## File Structure

```
PINN_FNO-rak/
├── pinn_1d_elastic_wave.py   # Main training script: architectures, training loop, evaluation
├── benchmark_PINNs.py        # Load saved models and generate result plots/animations
├── physics.py                # Domain config, material properties, Ricker source
├── solvers.py                # FD, PS, FEM, FVM, DG, SEM implementations
├── solve.py                  # Run all six solvers and save results to .npz
├── animate.py                # Animate solver outputs
├── compare.py                # Side-by-side comparison of numerical methods
├── requirements.txt
├── Models/                   # Saved .pt model weights for all experiments
└── results/                  # Output plots and animations per experiment
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `numpy`, `matplotlib`, `tqdm`, `torch`, `scipy`

---

## Usage

**Train PINN models**

```bash
# Run all three experiments
python pinn_1d_elastic_wave.py --experiment all

# Run a single experiment
python pinn_1d_elastic_wave.py --experiment homogeneous --epochs 700 --n_col 10000

# Skip the ForwardCNN
python pinn_1d_elastic_wave.py --experiment all --no_cnn
```

`--experiment` options: `homogeneous`, `layered`, `multilayer`, `all`

**Benchmark saved models**

```bash
python benchmark_PINNs.py
```

Loads `.pt` files from `Models/`, regenerates metrics and plots into `results/`.

**Run numerical solvers**

```bash
python solve.py      # Runs all six methods, saves numerical_lab_results.npz
python compare.py    # Visualize and compare solver outputs
python animate.py    # Animate wavefields
```

---

## Results

Pre-trained models and result figures are included in `Models/` and `results/`. Baseline L2 errors on the homogeneous case:

| Architecture | Mean L2 Error |
|---|---|
| Vanilla PINN | 0.78% |
| Fourier-Feature PINN | 1.05% |
| PirateNet | 3.56% |

Animated wavefields for each experiment are saved as `.gif` files in the root and `results/` directories.

---

## References

- Wang et al. (2024) — *PirateNet: Adaptive residual networks for PDE solving*
- Daw et al. (2023) — *R3 adaptive collocation for PINNs*
- Raissi et al. (2019) — *Physics-informed neural networks*
- 
