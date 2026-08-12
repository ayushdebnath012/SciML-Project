# What this project does

A single 1D physics problem — the elastic wave equation in heterogeneous
media — attacked with **two different families of neural PDE solver**, plus the
finite-difference reference that both are measured against.

```
rho(x) u_tt = d/dx [ E(x) u_x ]        x in [-1, 1],  t in [0, 1]
```

The two arms answer different questions and share nothing but the physics and
the reference solver:

| | Arm A — coordinate networks | Arm B — neural operators |
|---|---|---|
| **learns** | `(x, t) -> u(x,t)` for *one* material | `[E(x), rho(x), ...] -> u(x,t)` for *any* material |
| **supervision** | PDE residual (no solution data) | FD solutions (supervised) |
| **cost model** | retrain per problem | train once, infer in one forward pass |
| **question** | which architecture / optimizer / weighting trains a PINN best? | which operator architecture generalizes to unseen media? |
| **models** | VanillaPINN, FourierFeaturePINN, PirateNet, WavKAN, KAN, ChebyshevKAN, FourierWavKAN | FNO, DeepONet, PFNO |
| **status** | infrastructure complete, 528-run sweep runs on Sherlock | **run, measured, documented** |

Arm B is where the results are. Its full write-up — physics, architectures,
diagnosis, per-arm results, visualization rationale, limitations — is
[`docs/operator_simulations.md`](operator_simulations.md) (also built to PDF).
This document is the map: what exists, what was found, and where it lives.

---

## 1. The shared foundation

**The equation is solved in conservative form.** Expanding gives
`rho u_tt = E u_xx + E_x u_x`, so the naive `u_tt = c(x)^2 u_xx` drops the
`E_x u_x` term — exactly the term producing correct reflection and transmission
amplitudes at a material interface. Everything in the repo uses the
conservative form.

**The FD reference** ([`wave/fd_solver.py`](../wave/fd_solver.py)) is an
explicit second-order leapfrog scheme in flux form. Two details are
load-bearing and were both arrived at by fixing a failure:

- **Harmonic-mean interface stiffness** `E_{i+1/2} = 2 E_i E_{i+1} / (E_i + E_{i+1})`,
  which preserves transmission/reflection across a jump in `E`.
- **Mur discretization** of the absorbing boundaries,
  `beta = (c dt - dx)/(c dt + dx)`. The apparently natural centred-time /
  one-sided-space ABC is *unconditionally unstable*.

**The reference has to be grid-converged**, which was not true initially and
turned out to matter (§3.2). Measured against a `refine=32` solve:

| refine | FD grid | rel `L2` vs converged | cost / 512 samples |
|---|---|---|---|
| 1 | 64 | 13.35 % | 1.8 s |
| 4 | 253 | 1.14 % | 8.9 s |
| **8** | **505** | **0.32 %** | **22 s** |
| 16 | 1009 | 0.08 % | 41 s |

The problem definition — initial condition, material sampler, refined solve,
input-channel layout — lives in [`wave/wave_problem.py`](../wave/wave_problem.py),
which is deliberately torch-free so dataset generation runs on machines with no
GPU stack.

---

## 2. Arm A — PINN / KAN coordinate networks

A comparison of **seven architectures** on three canonical materials
(Homogeneous, TwoLayer, MultiLayer), trained purely on PDE residuals.

### Architectures ([`src/models.py`](../src/models.py), [`src/models_jax.py`](../src/models_jax.py))

| model | key idea |
|---|---|
| `VanillaPINN` | MLP, Tanh + Xavier |
| `FourierFeaturePINN` | trainable random Fourier embedding -> MLP |
| `PirateNet` | fixed Fourier embed -> RWF layers -> residual blocks (alpha init 0) |
| `WavKAN` | wavelet KAN (Mexican Hat / Morlet / DoG / Meyer / Shannon) |
| `KAN` | pykan / jaxkan spline KAN |
| `ChebyshevKAN` | cPIKAN — tanh-squashed inputs, Chebyshev recurrence per edge |
| `FourierWavKAN` | Fourier embedding -> WavKAN trunk (spectral-bias fix for sharp interfaces) |

**Both backends implement all seven.** PyTorch is the original; JAX/Equinox is
a parallel reimplementation (`src/*_jax.py`) and is now the default
(`--backend jax`). The JAX path uses `jax.grad(jax.grad(...))` + `vmap` for the
PDE residual instead of reverse-over-reverse autograd, and
`jax.ops.segment_sum` for causal chunking instead of a Python loop.

### Training machinery ([`src/losses/wave_loss.py`](../src/losses/wave_loss.py))

Everything below is implemented and switchable per run:

- **Two-phase schedule** — Adam (15 000 steps) then L-BFGS (2 000), or
  L-BFGS-only (5 000). L-BFGS runs with `use_causal=False`: causal weights
  changing inside the closure would corrupt the line-search objective.
- **Causal weighting** (Wang et al. 2022) — 16 time slabs, slab `m` weighted by
  `exp(-eps * sum_{k<m} L_k)`, with tolerance scheduling and a continuation loop.
- **GradNorm** (Wang et al. 2023) — EMA gradient-norm balancing of
  `w_pde / w_bc / w_ic`, refreshed every 100 steps.
- **RBA** (arXiv:2307.00379) — residual-based attention, pointwise
  `lambda <- 0.999 lambda + 0.01 |r|/max|r|`; an alternative to causal weighting,
  frozen during L-BFGS.
- **SOAP** (arXiv:2409.11321 / 2502.00604) — Adam in the Shampoo eigenbasis,
  2D params only, eigendecomposition refreshed every 10 steps. Implemented for
  both backends ([`src/optimizers.py`](../src/optimizers.py),
  [`src/optimizers_jax.py`](../src/optimizers_jax.py)).
- **IC handling, two ways** — a hard-constraint ansatz
  `u_hat = g(x) exp(-(15t)^2/2) + tanh^2(25t) NN(x,t)` that satisfies the IC
  exactly, or a soft IC loss with warm-up and a `_IC_SCALE = 15000` pre-scale
  that prevents IC gradient starvation early in training.
- **Checkpoint selection on a stationary metric** — unweighted per-chunk PDE
  residual + BC + IC/scale, *not* the weighted training loss, which moves as the
  weights move.

### The sweep

**528 runs** = ~50 architecture configs × 3 materials × {hard ansatz, soft IC}
× {Adam+L-BFGS, L-BFGS-only}, with SOAP/RBA variants excluded from the
L-BFGS-only half (no Adam phase means they would duplicate the base config).

[`wave/sherlock.sh`](../wave/sherlock.sh) is the whole cluster story in one
file: it submits itself as a SLURM array, and on first use builds
`$SCRATCH/wave_jax_env` under a `flock` so concurrent array tasks wait rather
than racing pip. `bash wave/sherlock.sh 10 2` = 10 GPUs, 2 runs sharing each.

### Status

The infrastructure is complete and the sweep runs; **its results are not in
this repository** — they land in `$SCRATCH/wave_results_jax/` on the cluster.
What is here from the coordinate-network side:

- [`final/`](../final/) — the earlier standalone version of this project:
  trained checkpoints for vanilla / Fourier / PirateNet / FNO on the three
  materials (`final/ML/Models/`), rendered comparisons, heatmaps, residual maps
  and animations for three experiments (`final/ML/results/exp1..exp3`), and a
  `Numerical methods/` folder with the FD solver comparison it was validated
  against.
- [`wave/hyperparam_tuning/`](../wave/hyperparam_tuning/) — PINN and PIKAN
  tuning drivers plus a results-viewing notebook.

---

## 3. Arm B — neural operators (the measured results)

Three architectures learn `[E(x), rho(x), g(x), x, t] -> u(x,t)` supervised on
FD solutions — no PDE residual in the loss, unlike Arm A. 512 samples on a
64×64 grid, 410 train / 102 validation, 400 epochs, AdamW + OneCycleLR, one
H100. The metric is relative `L2` in physical units, which has a useful
calibration property: **a model that outputs zeros scores ~100 %**.

- **FNO** — 2D spectral convolution over the joint `(x,t)` spectrum.
- **DeepONet** — branch × trunk inner product; a rank-`p` separable expansion.
- **PFNO** — *paralleled* FNO (not physics-informed): one small 1D FNO per
  temporal frequency bin, reassembled by inverse rFFT. 33–41 independent branches.

### 3.1 The unforced arm, and the failure that started it

The original animations showed noise. The cause was not the plotting code — the
models were **untrained** and the plots were faithfully showing that. The
configuration was 16 samples / 12 epochs / batch 4, about 39 gradient steps
total, at 10–30 k parameters. FNO scored 81.9 %, PFNO 94.8 %, DeepONet 97.3 % —
i.e. near-zero fields.

Three changes fixed it, in order of importance: **512 samples instead of 16**
(FD generation costs ~2 s, so the small dataset was never a compute
constraint), **OneCycleLR** (with a flat LR these models sit at normalized
MSE ≈ 1.0, exactly what predicting the mean achieves), and larger capacity /
400 epochs.

| model | parameters | train time | val rel `L2` | was |
|---|---|---|---|---|
| FNO | 7.39 M | 73 s | **2.07 %** | 81.9 % |
| PFNO | 1.23 M | 819 s | **4.26 %** | 94.8 % |
| DeepONet | 0.89 M | 25 s | **14.43 %** | 97.3 % |

Per material family the ordering is identical for all three models and matches
physical intuition — error grows with the number of interfaces:

| family | FNO | DeepONet | PFNO |
|---|---|---|---|
| homogeneous | 0.13 % | 6.07 % | 0.74 % |
| smooth | 0.93 % | 10.94 % | 2.59 % |
| two-layer | 1.86 % | 16.22 % | 3.96 % |
| layered (3–7) | 5.28 % | 23.78 % | 9.55 % |

### 3.2 The target was not converged

The datasets above solve the FD reference on the *same* 64-point grid as the
output. Regenerating with identical materials (bit-identical `inputs`) and only
a `refine=8` target moves the targets by **9.72 % mean** — and by 7.91 % even on
homogeneous samples, which have no interfaces at all. That part is numerical
dispersion of the initial pulse itself, whose width spans about three cells at
64 points.

| model | vs `refine=1` target | vs `refine=8` target |
|---|---|---|
| FNO | 2.07 % | **1.96 %** |
| PFNO | 4.26 % | **4.03 %** |
| DeepONet | 14.43 % | **24.57 %** |

FNO and PFNO barely move: discretization error is a deterministic function of
the material, so a model with capacity simply learns whichever target it is
given. **DeepONet nearly doubles** — the under-resolved target is smoother, and
a rank-`p` separable expansion represents smooth fields far more cheaply than
sharp ones. One architecture's headline number was materially flattered by a
numerical artefact, and it was invisible until the target was fixed.

### 3.3 The forced arm: a source term and physics residuals

A second, independent arm drives the wave with a Ricker point source instead of
an initial pulse, with quiescent IC and `t` extended to 2 (at `t_max = 1`, 99 %
of the energy is still inside the domain, so the absorbing BC is barely
exercised; at 2, 99.8 % has left). Three inputs now vary — `E(x)`, source
position, peak frequency — making it much harder: FNO 17.13 %, PFNO 31.89 %,
DeepONet 91.82 %.

IC and BC residuals were added as **diagnostics** (the loss is still plain MSE),
with the FD reference's own residual as the floor, and the metric validated on
two controls (a standing wave scores exactly 1.0; an all-zero field scores 0).

The result worth keeping: **DeepONet reports the best boundary score** (0.144,
absolute residual 7× *smaller* than the reference) while being by far the worst
model. Decomposing it, its boundary terms are ~6× smaller than the reference's —
there is barely a wave arriving, so it satisfies the radiation condition by
having nothing to radiate. **These residuals are only interpretable alongside
the field error.**

### 3.4 Making the physics an actual training loss

`L = MSE + lambda (L_ic_u + L_ic_ut + L_bc)`, with the numpy diagnostics
reimplemented in torch (agreeing to ~1e-8 before anything was trusted) and each
term divided by a fixed precomputed scale so one `lambda` balances all three.
`--lambda-physics 0` reproduces the supervised run exactly.

| lambda | field error | IC velocity | BC relative |
|---|---|---|---|
| 0 | 17.13 % | 0.1835 | 0.2494 |
| 1.0 | 14.84 % | 0.0171 | 0.0657 |
| **10.0** | **12.70 %** | **0.0058** | **0.0291** |

Every metric improves monotonically with **no trade-off** — field error down
26 % relative, IC-velocity residual 32× better. Unusual enough to suspect, so
the collapse check was repeated: field RMS rises to 0.982× the reference from
0.969×, so the model is *more* amplitude-faithful. The likely cause is
regularization — with 410 samples the constraints supply information the data
does not pin down.

At `lambda = 1` the three architectures do three different things:

- **FNO improves on every axis.** Its BC residual (0.066) drops below the FD
  reference's own 0.126, which reflects the reference's first-order Mur
  discretization rather than the network being "better than truth".
- **PFNO is essentially unchanged** (31.89 → 31.43 %). It was predicted to
  benefit most, since its initial-velocity residual was its clearest weakness.
  The prediction was wrong, and the reason is the better result: `u_t(x,0) = 0`
  requires 41 independent frequency branches to agree in phase, and a gradient
  on the assembled field cannot repair a coordination failure spread across 41
  sub-networks. The defect is architectural, not an optimization shortfall.
- **DeepONet gets worse** (91.82 → 93.55 %). Zero satisfies every one of these
  homogeneous constraints exactly, so for a model that already cannot fit the
  data the physics terms are a gradient pointing *toward* the trivial solution.

### 3.5 Real velocity models

The synthetic sampler (tanh steps, sums of sinusoids) is replaced by 1D depth
columns from three published benchmark models — **Marmousi, SEG/EAGE
Overthrust, SEG/EAGE Salt** — keeping the problem, solver, architectures and
schedule identical. Two methodological changes are forced by real data:

- **Coarsening by Backus averaging** (arithmetic mean of `rho`, harmonic mean of
  the modulus), the correct long-wavelength effective medium. Point-sampling a
  739-cell Marmousi column to 64 points would alias interfaces into noise —
  3.2 % of its cells jump by more than 200 m/s.
- **A spatial train/validation split.** Neighbouring Marmousi traces are 4 m
  apart and differ by 0.2 % on average; a random split puts near-duplicates of
  training profiles into validation and measures interpolation, not
  generalization. Validation instead takes four contiguous trace blocks with
  320 m buffers discarded on each side.

| arm | heterogeneity | FNO | PFNO | DeepONet |
|---|---|---|---|---|
| synthetic | 0.068 | **1.96 %** | **4.03 %** | **24.57 %** |
| Overthrust | 0.172 | **3.59 %** | **5.65 %** | **26.11 %** |
| Salt | 0.097 | **6.95 %** | **9.30 %** | **18.55 %** |
| Marmousi | 0.280 | **12.33 %** | **16.15 %** | **39.97 %** |

**The architectural ranking survives on real geology** — FNO < PFNO < DeepONet
on every arm without exception, which is the main thing this section was run to
check. But the margins compress (FNO beats PFNO 2.1× on synthetic, 1.3× on
Marmousi), and difficulty tracks **heterogeneity, not contrast**: Marmousi and
Overthrust actually get *easier* as `Vp_max/Vp_min` rises, while Salt reverses
completely because its high-contrast columns are exactly the ones intersecting
a salt body — one sharp isolated reflector, a different and harder problem than
a wide velocity range.

### 3.6 Visualization

20 GIFs in [`simulations/`](../simulations/) (4 synthetic, 9 real, 4 forced, 3
superseded), each a 2×4 panel: velocity model + three line-plot comparisons on
top, FD reference field + three predicted fields below, all on a shared colour
scale. Two defects in the original figure were fixed because they mattered for
interpretability independent of model quality:

1. **The reference is now shown.** The original compared predictions against
   each other only — actively misleading when models are undertrained, since a
   near-zero field looks smooth and plausible.
2. **The colour scale comes from the propagating wave** (98th percentile of
   `|u|`), not the global max. The `t = 0` pulse is ~1.8× taller than anything
   after it, so scaling to it spent half the colormap on a single frame.
3. **The palette was re-stepped** for colour-vision separability — the original
   blue/teal pair sat at ΔE 14.0, below the legibility floor; the worst adjacent
   pair is now ΔE 16.0 under protanopia.

Rendering is deliberately split from training: `train_operators.py` writes no
plots, so the GPU host needs no matplotlib.

---

## 4. What the whole thing found

1. **Operator learning has a hard data threshold.** 16 samples is not a small
   dataset for this problem; it is below the point where anything is learnable.
   A few hundred is the threshold, and generating them costs seconds.
2. **A near-zero prediction is the default failure**, and it looks plausible.
   Every metric here is calibrated so zero scores ~100 %, and every surprising
   result is checked against an amplitude-ratio control.
3. **Architecture ranking is stable and structural.** FNO < PFNO < DeepONet on
   every material family, every real geological arm, forced and unforced. It
   follows from what each architecture can represent: joint `(x,t)` spectrum vs
   per-frequency factorization vs rank-`p` separable expansion.
4. **Physics losses help a model that already fits, and hurt one that doesn't.**
   FNO improves monotonically to `lambda = 10`; DeepONet is pushed further
   toward the trivial solution that satisfies every homogeneous constraint
   exactly.
5. **Residual diagnostics are not standalone evidence.** The worst model had the
   best boundary residual.
6. **The reference solution is part of the experiment.** An unconverged target
   left one architecture's headline number 10 points too good.
7. **Difficulty is about sharp isolated reflectors, not velocity range.**
   `Vp_max/Vp_min` predicts the wrong sign on two of three real models.

---

## 5. Repository map

**Code**

| path | role |
|---|---|
| [`wave/fd_solver.py`](../wave/fd_solver.py) | leapfrog FD reference (conservative form, Mur ABC) |
| [`wave/wave_problem.py`](../wave/wave_problem.py) | torch-free problem definition: IC, material sampler, channel layout |
| [`wave/materials.py`](../wave/materials.py), [`problem_data.py`](../wave/problem_data.py) | Homogeneous / TwoLayer / MultiLayer + JAX methods; ansatz, FD dispatch |
| [`wave/run_experiment.py`](../wave/run_experiment.py) | Arm A runner — argparse, SLURM, backend switch, `MODEL_CONFIGS` / `CONFIG` |
| [`wave/sherlock.sh`](../wave/sherlock.sh) | the one cluster script (self-submitting, self-bootstrapping env) |
| [`wave/run_fno_baseline.py`](../wave/run_fno_baseline.py) | original FNO/CNN baseline; now re-exports `wave_problem` |
| [`wave/operator_sim/`](../wave/operator_sim/) | Arm B: dataset generation (synthetic / real / source), the three models, training, physics losses + metrics, rendering |
| [`src/models*.py`, `optimizers*.py`, `train*.py`, `losses/`](../src/) | Arm A internals, PyTorch and JAX |
| [`kaggle/run_fno_t4.py`](../kaggle/run_fno_t4.py) | T4 preset launcher (smoke / medium / full) |

**Data and results**

| path | contents |
|---|---|
| `operator_data/*.npz` | 5 datasets, 512×5×64×64 in / 512×1×64×64 out (~8.5 MB each): synthetic `refine=1`, synthetic `refine=8`, Marmousi, Overthrust, Salt, plus the forced arm at 64×80 |
| `operator_data/raw/` | raw velocity models (~700 MB, gitignored) |
| `server_outputs/` | unforced synthetic run: predictions, summary, per-epoch histories, log |
| `server_outputs_real/{synthetic_r8,marmousi,overthrust,salt}/` | the four real-model arms |
| `server_outputs_source/`, `server_outputs_pinn/` | forced arm, supervised and physics-informed (+ `lambda_sweep.json`) |
| `simulations/` | 20 GIFs — synthetic, `real/<model>/`, `source_term/`, `superseded_undertrained/` |
| `fno_t4_{smoke,medium,full}/`, `operator_results_*`, `kaggle_outputs/` | earlier baseline runs that bracket the data threshold |
| `final/` | the earlier standalone project: checkpoints, exp1–3 results, numerical-methods comparison |

Model checkpoints (`FNO_state.pt` is 59 MB) stayed on the training host and were
not copied back; predictions and summaries were, which is everything the
renderer and the analysis need.

**Writing and reference material**

| path | contents |
|---|---|
| [`docs/operator_simulations.md`](operator_simulations.md) | the full Arm B write-up (17 sections) + `.tex`/`.pdf` build and 11 figures |
| [`notebooks/`](../notebooks/) | end-to-end FNO/DeepONet/PFNO notebook (`QUICK_RUN` toggle; uses pretrained predictions when present) |
| [`ppt/`](../ppt/) | `FNO_Technical_Reference.tex`/`.pdf`, KAN guide, experiments deck, and `diagrams/` — TikZ sources for spectral conv, FNO2d, Fourier block, ansatz anatomy, data pipeline, architecture evolution |
| [`presentations/`](../presentations/), [`outputs/`](../outputs/) | Neural-operator architecture decks, FNO explainer (several revisions) |
| [`videos/`](../videos/) | two rendered FNO explainer videos |
| [`papers/`](../papers/) | ~30 reference PDFs + `fno_references.bib`, with a curated 7-paper `FNO_reading_order/` |
| [`fno paper/`](../fno%20paper/) | the three papers the Arm B implementations follow |

---

## 6. Status and honest limits

**Done and measured:** the entire operator arm — four material regimes, forced
and unforced, supervised and physics-informed, with converged targets, spatial
splits, calibrated metrics, animations, and a full write-up.

**Built but not reported here:** the 528-run PINN/KAN sweep. Code, both
backends, all seven architectures, SOAP/RBA/causal/GradNorm and the cluster
harness are complete; results live on `$SCRATCH` rather than in the repo, and
no cross-architecture comparison has been written up from them.

**Limits to keep in view when reading Arm B's numbers:**

- **One seed, one split, no error bars.** Tenths of a percent between
  configurations mean nothing; the order-of-magnitude gaps between architectures
  do.
- **Fixed initial condition** (`sigma_g = 0.1`, `x0 = 0`) — the unforced arm
  learns a velocity-model → wavefield map, not a general IC → wavefield map.
- **Fixed 64×64 grid.** FNO and PFNO are in principle discretization-invariant;
  zero-shot super-resolution was never tested.
- **In-distribution evaluation only.** No arm is evaluated on a *different*
  arm — nothing trains on Marmousi and tests on Salt, which is the transfer
  question a practitioner would actually ask. Transfer to Arm A's canonical
  materials is also untested.
- **"Real" means published benchmark model, not measurement.** Marmousi and
  Overthrust are themselves synthetic constructions; the wave field is
  FD-simulated in every arm. Only Marmousi ships a density model.
- **Parameters are not matched.** FNO has 7.4 M against DeepONet's 0.89 M and
  PFNO's 1.2 M, so some of FNO's advantage is capacity. A parameter-matched
  comparison has not been run. Note also that `train_operators.py` defaults to
  `--don-latent 192 --don-hidden 384`, while every DeepONet number quoted here
  uses the larger `256/512` pair passed explicitly — re-running without those
  flags silently produces a smaller, non-comparable model.
- **These are readable reimplementations**, sized to train in minutes, not
  faithful reproductions of the Li et al. or Lu et al. configurations.

**Version-control state:** most of the work described in §3 — `wave/operator_sim/`,
`docs/`, `notebooks/`, `simulations/`, all `server_outputs*` directories, the
`ppt/` technical reference and diagrams — is currently **untracked**. Only the
FNO baseline and Kaggle runner have been committed.
