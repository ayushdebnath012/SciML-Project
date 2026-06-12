# Wave Equation PINN/KAN Experiments

Physics-Informed Neural Networks for the 1D elastic wave equation

```
ρ(x) u_tt = ∂_x [ E(x) u_x ]
```

across three material profiles (**Homogeneous**, **TwoLayer**, **MultiLayer**), comparing seven architectures — VanillaPINN, FourierFeaturePINN, PirateNet, WavKAN, KAN (pykan/jaxkan), ChebyshevKAN, FourierWavKAN — plus two training variants: the **SOAP** optimizer (arXiv:2502.00604) and **RBA** residual-based attention weighting (arXiv:2307.00379).

The full sweep is **528 runs**: every config × 3 materials × {hard-constraint IC, soft IC loss} × {Adam+L-BFGS, L-BFGS-only}. Accuracy is measured as relative L2 error against a finite-difference reference solution.

## How to run on Sherlock (recommended)

Everything is one script. From the repository root on a login node:

```bash
git clone https://github.com/Arulrana31/Stanford-SciML-Project.git
cd Stanford-SciML-Project

bash wave/sherlock.sh                # 30 GPUs, 1 run per GPU (defaults)
```

Options — first argument is concurrent GPUs, second is runs sharing each GPU:

```bash
bash wave/sherlock.sh 10             # at most 10 GPUs at a time
bash wave/sherlock.sh 10 2           # 10 GPUs, 2 experiments per GPU
bash wave/sherlock.sh 8 3            # 8 GPUs, 3 experiments per GPU
```

That's it. The script submits itself as a SLURM array job and, on first use, automatically builds the Python environment at `$SCRATCH/wave_jax_env` (~10 min, one-time; concurrent tasks wait on a lock instead of racing pip). No modules to load, no manual setup.

Notes:
- Workers-per-GPU > 1 splits GPU memory evenly between runs (JAX preallocation is disabled). 2–3 workers is fine on a 40/80 GB card; CPU cores and RAM per task are scaled automatically.
- Each task has an 8 h time limit; a typical run finishes well within it.
- Uses the `serc` partition with `gpu` as fallback.

### Monitoring and results

```bash
squeue -u $USER                                   # job status
tail -f slurm_logs/run_<id>.log                   # live log of one run
```

Results land in `$SCRATCH/wave_results_jax/<Material>/<RunName>_jax/`:

| File | Contents |
|---|---|
| `l2_errors.json` | relative L2 error vs FD reference, IC metrics, config |
| `solution_comparison.png` | FD reference vs prediction vs residuals |
| `loss_curve.png`, `gradnorm_weights.png`, `causal_*.png` | training diagnostics |
| `model_best_adam.eqx`, `model_best_lbfgs.eqx` | best checkpoints (equinox) |

### Re-running specific experiments

```bash
python wave/run_experiment.py --list-runs         # run-id -> experiment mapping
sbatch --array=42 wave/sherlock.sh                # re-run just run-id 42
sbatch --array=100-150%10 wave/sherlock.sh        # re-run a range
```

## Running locally

```bash
# Environment (Python 3.10+)
pip install -r requirements.txt                   # torch, pykan, numpy, ...
pip install "jax[cpu]" equinox optax jaxopt jaxkan   # JAX backend (CPU)

# Fast sanity check: 1 config per architecture, 5 optimizer steps, all materials
python wave/run_experiment.py --test

# Single full run
python wave/run_experiment.py --run-id 0 --output-dir ./experiment_results/

# Full sweep, sequential (slow — use Sherlock instead)
python wave/run_experiment.py
```

Useful flags: `--backend {jax,pytorch}` (default `jax`), `--adam-iterations N`, `--lbfgs-iterations N`, `--output-dir DIR`, `--list-runs`.

## Configuring the sweep

Edit `wave/run_experiment.py`:
- **`MODEL_CONFIGS`** — the list of architecture/hyperparameter configs. Optional per-config keys: `"optimizer": "soap"` and `"weighting": "rba"` (run names get `_soap` / `_rba` suffixes).
- **`CONFIG`** — iterations, learning rate, collocation grid, causal-training schedule, seed, output dir.

## Method summary

- Two-phase training: Adam (or SOAP) with causal time-weighting (Wang et al. 2022) and GradNorm loss balancing (Wang et al. 2023), then L-BFGS fine-tuning on the plain residual.
- IC handled either by a hard-constraint ansatz (exact at t=0) or a soft IC loss with warm-up + scaling; absorbing (radiation) BCs at both ends.
- FD reference solves the conservative form with harmonic-mean interface stiffness and Mur absorbing boundaries.
- Checkpoint selection uses a stationary validation metric (unweighted residuals), not the weighted training loss.
