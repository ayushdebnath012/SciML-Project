# CLAUDE.md — Wave Equation PINN/KAN Research Project

## Project Overview

Physics-Informed Neural Network (PINN) research comparing seven architectures for solving the **1D elastic wave equation** (`ρ u_tt = ∂_x[E(x) u_x]`) across three heterogeneous material profiles. Training uses a two-phase Adam → L-BFGS schedule with causal loss weighting and GradNorm adaptive balancing, plus two newer training arms: **SOAP** optimizer (arXiv:2502.00604) and **RBA** residual-based attention weighting (arXiv:2307.00379).

Two compute backends are supported: **JAX** (default, `--backend jax`) and **PyTorch** (`--backend pytorch`). The JAX path is a parallel implementation in `src/*_jax.py`.

## Repository Layout

```
Exp/
├── src/
│   ├── models.py              # PyTorch: all 7 architectures
│   ├── models_jax.py          # JAX/Equinox: all 7 architectures (single-sample interface)
│   ├── optimizers.py          # PyTorch: SOAP optimizer (rotated-Adam, arXiv:2409.11321)
│   ├── optimizers_jax.py      # JAX: SOAP as an optax GradientTransformation
│   ├── physics.py             # dy_dx autograd helper
│   ├── train.py               # PyTorch: Adam/SOAP + L-BFGS training loops
│   ├── train_jax.py           # JAX: optax Adam/SOAP + jaxopt L-BFGS, RBA step
│   └── losses/
│       ├── wave_loss.py       # PyTorch: PDE residuals, causal loss, GradNorm, RBA, BC/IC
│       └── wave_loss_jax.py   # JAX: same physics, jax.grad + jax.vmap derivatives
├── wave/
│   ├── problem_data.py        # Gaussian IC, ansatz, FD reference dispatch
│   ├── materials.py           # Homogeneous / TwoLayer / MultiLayer + JAX methods
│   ├── fd_solver.py           # Leapfrog FD reference solver (conservative form)
│   ├── run_experiment.py      # Main runner — argparse, SLURM, backend switch
│   └── sherlock.sh            # THE Sherlock script: bash wave/sherlock.sh [NUM_GPUS] [WORKERS_PER_GPU]
├── requirements.txt           # All pip deps (PyTorch + JAX + KAN libraries)
└── experiment_results/        # Output: checkpoints, loss curves, JSON metrics
```

## Architectures

### PyTorch (`src/models.py`)

| Class | Key idea |
|---|---|
| `VanillaPINN` | MLP with Tanh + Xavier init |
| `FourierFeaturePINN` | Trainable random Fourier embedding → MLP |
| `PirateNet` | Fixed Fourier embed → RWF layers → residual blocks (α init=0) |
| `WavKAN` | Wavelet-based KAN (Mexican Hat / Morlet / DoG / Meyer / Shannon) |
| `KAN` (pykan) | Kolmogorov-Arnold Networks via `pykan` library |
| `ChebyshevKAN` | cPIKAN basis: tanh-squashed inputs → Chebyshev recurrence per edge (arXiv:2406.02917) |
| `FourierWavKAN` | Trainable random Fourier embedding → WavKAN trunk (spectral-bias fix for sharp interfaces) |

### JAX (`src/models_jax.py`)

Same five architectures, reimplemented with `equinox`. **Single-sample interface**: every model's `__call__` takes `xt: (2,)` → `(1,)`. Use `jax.vmap(model)(xt_batch)` or `_call_model_jax(model, x, t)` for batches.

- `physics_informed_init_jax(model, xt_data, y_data)` — returns a **new** model (equinox is immutable); mirrors PyTorch `model.physics_informed_init()`.
- `build_kan_jax(...)` — wraps `jaxkan` library; tries both `jaxkan.models.KAN` and `jaxkan.KAN` constructors.
- JAX checkpoints saved as `.eqx` files via `eqx.tree_serialise_leaves`.
- `B` matrix in PirateNet is a regular array (trains slightly); mark frozen in optimizer if needed.

## Physics

### PyTorch (`src/losses/wave_loss.py`)

- **`causal_pde_loss`** — splits time domain into `n_chunks=16` slabs; weights slab `m` by `exp(-ε Σ_{k<m} L_k)` (Wang et al. 2022).
- **`GradNormWeighter`** — EMA-smoothed gradient-norm balancing for `w_pde / w_bc / w_ic` (Wang et al. 2023). Updated every `gradnorm_update_freq` steps (default 100).
- **`ic_loss`** — displacement MSE (×5) + velocity MSE (×1) at `t=0`.
- **Absorbing BCs** — first-order radiation condition `u_t ± c u_x = 0` at left/right boundaries.
- **Hard-constraint ansatz** — `û = g(x)·exp(-½(15t)²) + tanh²(25t)·NN(x,t)` enforces IC exactly.
- **`_IC_SCALE = 15000`** — pre-scale applied to IC loss before GradNorm, prevents IC gradient starvation early in training.
- **RBA** (`use_rba=True` in `losses_gradnorm`) — residual-based attention pointwise weights `λ ← 0.999λ + 0.01|r|/max|r|`, `loss = mean((λr)²)`. Replaces causal weighting (reports `w_min=1` so the continuation loop is skipped); λ frozen during L-BFGS; `reset_rba_state()` is called per run.
- **SOAP** (`optimizer_name="soap"` in `train_adam`) — Adam in the Shampoo eigenbasis, 2D params only (others get plain Adam); eigh refresh every 10 steps.

### JAX (`src/losses/wave_loss_jax.py`)

- PDE residual uses `jax.grad(jax.grad(...))` + `jax.vmap` per point — no reverse-over-reverse overhead.
- Causal chunks use `jax.ops.segment_sum` instead of a Python loop.
- `losses_gradnorm_jax(model, x_int, t_int, x_bc, t_bc, x_ic, ...)` — takes **pre-sampled, fixed-shape** BC/IC arrays (no `jnp.unique`; JIT-compatible).
- Returns 5-tuple `(l_pde, l_bc, l_ic, weights, chunk_losses)` — causal state extracted post-JIT and written to `_causal_state_jax` from Python.
- `GradNormWeighterJax` — re-runs individual loss forward passes to get per-loss gradient norms (3× extra FWD every 100 steps; cheap vs total training).
- `gaussian_ic_jax` uses analytical normalization constant `1/sigma * exp(-0.5)` instead of `dfdx.abs().max()`.

## Running Experiments

```bash
# Local test — 5 Adam + 5 L-BFGS, 1 config per arch (incl. soap/rba arms), all 3 materials
python wave/run_experiment.py --test                      # JAX (default backend)
python wave/run_experiment.py --test --backend pytorch    # PyTorch

# Full sweep (sequential, 528 runs)
python wave/run_experiment.py

# Print run-id -> experiment mapping (use to size SLURM arrays)
python wave/run_experiment.py --list-runs

# Single run by ID (SLURM array)
python wave/run_experiment.py --run-id 42 --output-dir $SCRATCH/wave_results_jax

# Sherlock sweep — single self-bootstrapping script (env built on first task):
bash wave/sherlock.sh                 # 30 GPUs, 1 run per GPU
bash wave/sherlock.sh 10 2            # 10 GPUs, 2 runs sharing each GPU
```

Key CLI flags: `--adam-iterations`, `--lbfgs-iterations`, `--output-dir`, `--backend {jax,pytorch}` (default jax), `--list-runs`.

New experiment arms are plain `MODEL_CONFIGS` entries with optional keys `"optimizer": "soap"` and/or `"weighting": "rba"`; variant configs are excluded from the L-BFGS-only half of the sweep (no Adam phase → duplicates). Run names append `_soap` / `_rba`.

## Environment Setup

```bash
# PyTorch GPU (CUDA 12.1):
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# JAX CPU (Windows local dev):
pip install "jax[cpu]" equinox optax jaxopt jaxkan

# JAX GPU (Linux / Sherlock, CUDA 12):
pip install "jax[cuda12]" equinox optax jaxopt jaxkan

# Sherlock: no manual setup — wave/sherlock.sh builds $SCRATCH/wave_jax_env
# automatically on first run (flock-guarded across array tasks).
```

Python 3.10 (`.venv` present). Key deps: `torch 2.5.1`, `pykan 0.2.8`, `numpy 1.26.4`, `equinox>=0.11`, `optax>=0.2`, `jaxopt>=0.8`, `jaxkan`.

## Output Structure

**PyTorch runs** write to `experiment_results/<Material>/<RunName>/`:
- `model_best_adam.pkl` / `model_best_lbfgs.pkl` — PyTorch state-dict checkpoints
- `l2_errors.json`, `causal_convergence.json`
- `loss_curve.png`, `gradnorm_weights.png`, `causal_wmin.png`, `causal_heatmap.png`
- `solution_comparison.png` — 3-panel: FD ref | PINN pred | residuals

**JAX runs** write to `experiment_results/<Material>/<RunName>_jax/`:
- `model_best_adam.eqx` / `model_best_lbfgs.eqx` — equinox serialised leaves

## Configuration

Edit `wave/run_experiment.py` directly:
- **`MODEL_CONFIGS`** (~line 37) — list of architecture sweeps
- **`CONFIG`** (~line 80) — `adam_iterations`, `lbfgs_iterations`, `Nx_collocation`, `Nt_collocation`, `lr`, `seed`, `output_base_dir`

## Architecture & Physics Notes

- FD reference solver (`fd_solver.py`) solves the **conservative form** `ρ u_tt = ∂_x[E u_x]` with harmonic-mean interface stiffness `E_{i+1/2}` on `[material.x_min, material.x_max]`. Do NOT revert to `u_tt = c²u_xx`: drops the `E_x u_x` term.
- FD ABCs use **Mur discretization** (`β = (cΔt-Δx)/(cΔt+Δx)`). Centred-time/one-sided-space ABC is unconditionally unstable.
- All three materials nondimensionalize to `x ∈ [-1, 1]` (MultiLayer: `[-1.5, 1.5]`).
- Checkpoint selection uses a **stationary validation metric** (unweighted per-chunk PDE residual mean + BC + IC/`_IC_SCALE`), not the weighted training loss. `train_adam` returns a 12-tuple; last element is best metric. Causal continuation loop passes it back via `initial_best`.
- L-BFGS phase runs with `use_causal=False`: causal weights change inside the closure and would corrupt the line-search objective.
- `WavKANLinear` / `WavKAN` accept `use_base=True` (default) for `wavelet_output + linear(x, weight1)`. Set `use_base=False` for wavelet-only ablations.
- `VanillaPINN` accepts `in_dim` / `out_dim` so `build_mlp` handles non-default input dims.
- JAX materials: `E_jax(x)`, `rho_jax(x)`, `Vp_jax(x)` on all three material classes — accept scalars or jnp arrays, used inside `jax.grad` chains.

## Key Papers

- Wang et al. 2022 — causal training (arXiv:2203.07404)
- Wang et al. 2023 — "An Expert's Guide to Training PINNs" (arXiv:2308.08468) — `train_pinns.pdf` in repo root
