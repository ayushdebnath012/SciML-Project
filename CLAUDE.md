# CLAUDE.md — Wave Equation PINN/KAN Research Project

## Project Overview

Physics-Informed Neural Network (PINN) research comparing five architectures for solving the **1D elastic wave equation** (`ρ u_tt = ∂_x[E(x) u_x]`) across three heterogeneous material profiles. Training uses a two-phase Adam → L-BFGS schedule with causal loss weighting and GradNorm adaptive balancing.

## Repository Layout

```
Exp/
├── src/
│   ├── models.py          # All 5 neural architectures
│   ├── physics.py         # dy_dx autograd helper
│   ├── train.py           # Adam + L-BFGS training loops
│   ├── utils.py           # Error metrics, summary/plot helpers
│   └── losses/
│       └── wave_loss.py   # PDE residuals, causal loss, GradNorm, BC/IC losses
├── wave/
│   ├── problem_data.py    # Gaussian IC, ansatz, FD reference dispatch
│   ├── materials.py       # Homogeneous / TwoLayer / MultiLayer material models
│   ├── fd_solver.py       # Leapfrog finite-difference reference solver
│   ├── run_experiment.py  # Main experiment runner (argparse + SLURM support)
│   ├── setup_sherlock.sh  # One-time Sherlock HPC env setup
│   ├── slurm_array.sh     # SLURM array job (360 configs × 30 concurrent GPUs)
│   └── slurm_h100.sh      # Single H100 sequential sweep (~12–18 h)
├── config.yaml            # HPT sweep settings (legacy; superseded by run_experiment.py CONFIG)
├── requirements.txt       # Pip dependencies (PyTorch installed separately for CUDA)
└── experiment_results/    # Output: model checkpoints, loss curves, JSON metrics
```

## Architectures (`src/models.py`)

| Class | Key idea |
|---|---|
| `VanillaPINN` | MLP with Tanh + Xavier init |
| `FourierFeaturePINN` | Trainable random Fourier embedding → MLP |
| `PirateNet` | Fixed Fourier embed → RWF layers → residual blocks (α init=0) |
| `WavKAN` | Wavelet-based KAN (Mexican Hat / Morlet / DoG / Meyer / Shannon) |
| `KAN` (pykan) | Kolmogorov-Arnold Networks via `pykan` library |

## Physics (`src/losses/wave_loss.py`)

- **`causal_pde_loss`** — splits time domain into `n_chunks=32` slabs; weights slab `m` by `exp(-ε Σ_{k<m} L_k)` (Wang et al. 2022, causal training).
- **`GradNormWeighter`** — EMA-smoothed gradient-norm balancing for `w_pde / w_bc / w_ic` (Wang et al. 2023). Updated every `gradnorm_update_freq` steps (default 100).
- **`ic_loss`** — displacement MSE + velocity MSE at `t=0`.
- **Absorbing BCs** — first-order radiation condition `u_t ± c u_x = 0` at left/right boundaries.
- **Hard-constraint ansatz** — `û = g(x)·exp(-½(15t)²) + tanh²(25t)·NN(x,t)` enforces IC exactly.

## Running Experiments

```bash
# Local test (5 Adam + 5 L-BFGS, all architectures, 3 materials)
python wave/run_experiment.py --test

# Full sweep locally (sequential, ~360 runs)
python wave/run_experiment.py

# Single run by ID (for SLURM array jobs)
python wave/run_experiment.py --run-id 42 --output-dir $SCRATCH/wave_results

# SLURM array (30 GPUs concurrent)
sbatch wave/slurm_array.sh
# or override concurrency:
sbatch --array=0-359%10 wave/slurm_array.sh

# Single H100 sequential sweep
sbatch wave/slurm_h100.sh
```

Key CLI flags: `--adam-iterations`, `--lbfgs-iterations`, `--output-dir`.

## Environment Setup

```bash
# GPU (CUDA 12.1) — install PyTorch first:
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Sherlock HPC (one-time):
bash wave/setup_sherlock.sh
```

Python 3.10 (`.venv` present). Key dependencies: `torch 2.5.1`, `pykan 0.2.8`, `numpy 1.26.4`, `matplotlib 3.10.8`.

## Output Structure

Each run writes to `experiment_results/<Material>/<RunName>/`:
- `model_best_adam.pkl` / `model_best_lbfgs.pkl` — best checkpoints
- `l2_errors.json` — relative L2 error vs. FD reference + IC metrics
- `causal_convergence.json` — `w_min` history + per-chunk weight snapshots
- `loss_curve.png`, `gradnorm_weights.png`, `causal_wmin.png`, `causal_heatmap.png`
- `solution_comparison.png` — 3-panel: FD ref | PINN pred | residuals

## Configuration

Edit `wave/run_experiment.py` directly:
- **`MODEL_CONFIGS`** (~line 37) — list of architecture sweeps
- **`CONFIG`** (~line 80) — `adam_iterations`, `lbfgs_iterations`, `Nx_collocation`, `Nt_collocation`, `lr`, `seed`, `output_base_dir`

## Architecture Notes

- `WavKANLinear` and `WavKAN` accept `use_base=True` (default) to include the linear branch `wavelet_output + F.linear(x, weight1)`. Set `use_base=False` to run wavelet-only ablations.
- `VanillaPINN` accepts `in_dim` / `out_dim` (default 2/1) so `build_mlp` correctly reflects any input dimensionality change.
- The FD reference solver (`fd_solver.py`) uses first-order ABCs matching the PINN training objective. BC: `u_t - c*u_x = 0` (left), `u_t + c*u_x = 0` (right).
