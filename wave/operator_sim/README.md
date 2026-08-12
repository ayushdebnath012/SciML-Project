# Operator simulation pipeline (FNO / DeepONet / PFNO)

> Quick start. The full write-up — physics, architectures, diagnosis, results
> per material, visualization rationale, and limitations — is in
> [`docs/operator_simulations.md`](../../docs/operator_simulations.md).

Reproduces the animations in `simulations/`. These scripts exist because the
notebook's original 16-sample / 12-epoch setting left every model at 82–97 %
relative L2 — no better than predicting zero — so the animations showed noise
rather than wave propagation.

## Why the old runs failed

Operator learning on this problem needs a few hundred training samples. The
repository's own prior runs show the threshold clearly:

| samples | grid | parameters | val rel L2 |
|---|---|---|---|
| 16 | 48² | 99 k | 58.3 % |
| 32 | 64² | 99 k | 94.8 % |
| 256 | 128² | 4.7 M | 7.4 % |
| 768 | 128² | 13.1 M | 3.2 % |

The notebook was using 16 samples with ~10–30 k parameters for 12 epochs.

## Pipeline

```bash
# 1. Generate FD training data (512 samples, 64x64, ~2 s)
python wave/operator_sim/generate_dataset.py \
    --num-samples 512 --nx 64 --nt 64 --seed 42 \
    --out operator_data/wave_operator_fixedic_n512_nx64_nt64_t1_seed42.npz

# 2. Train all three operators (GPU strongly recommended; ~15 min on one H100,
#    hours on CPU because PFNO instantiates nt//2+1 = 33 parallel 1D branches)
python wave/operator_sim/train_operators.py \
    --data operator_data/wave_operator_fixedic_n512_nx64_nt64_t1_seed42.npz \
    --epochs 400 --outdir server_outputs_v2

# 3. Render the comparison GIFs (CPU, matplotlib; ~5 min per GIF)
python wave/operator_sim/render_simulations.py \
    --pred server_outputs/operator_wave_predictions_v2.npz \
    --summary server_outputs/operator_wave_summary_v2.json \
    --outdir simulations --fps 12 --dpi 95
```

Step 2 writes no plots, so the training host needs no matplotlib.

## Results (512 samples, 64x64, 400 epochs, one H100)

| model | parameters | train time | val rel L2 |
|---|---|---|---|
| FNO | 7.39 M | 73 s | **2.07 %** |
| PFNO | 1.23 M | 819 s | **4.26 %** |
| DeepONet | 0.89 M | 25 s | **14.43 %** |

The ordering is expected: FNO sees the full 2D `(x,t)` spectrum; PFNO solves
each temporal frequency with an independent 1D branch; DeepONet is a low-rank
separable expansion `u ≈ Σ_p b_p(E,ρ,g)·τ_p(x,t)`, which limits how sharply it
can represent a moving front.

Two changes beyond dataset size mattered:

- **`OneCycleLR`.** With a flat learning rate these models stall near the zero
  solution regardless of epoch budget.
- **Capacity.** FNO width 8 → 48, modes 6 → 20, layers 2 → 4.

## Real-material arm (published velocity models)

Same problem and solver, but `E(x)` and `rho(x)` are 1D depth columns from
community-standard velocity models instead of the synthetic tanh/sine sampler.
The material is real; the wave field is still FD-simulated, because no measured
full-field `u(x,t)` exists for this geometry. See `velocity_models.py` for
provenance and for the Backus coarsening that reduces a depth column to the
64-point grid.

| `--model` | geometry | spacing | density? | source |
|---|---|---|---|---|
| `marmousi` | 2301 traces x 739 depth | 4 m | yes | geoazur WIND |
| `overthrust` | 801x801 x 161 depth | 25 m | no | SEG/EAGE (S3) |
| `salt` | 676x676 x 180 depth | 20 m | no | SEG/EAGE (S3) |

```bash
# 0. Fetch the models into operator_data/raw/ (not in git -- operator_data/ is ignored)
mkdir -p operator_data/raw/marmousi
curl -o operator_data/raw/marmousi/vp.bin  https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Marmousi/vp.bin
curl -o operator_data/raw/marmousi/rho.bin https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Marmousi/rho.bin

cd operator_data/raw
curl -LO https://s3.amazonaws.com/open.source.geoscience/open_data/seg_eage_models_cd/Overthrust_3D_CD1.tar.gz
tar xzf Overthrust_3D_CD1.tar.gz --wildcards "*/3D-Velocity-Grid/*"
curl -LO https://s3.amazonaws.com/open.source.geoscience/open_data/seg_eage_models_cd/Salt_Model_3D.tar.gz
tar xzf Salt_Model_3D.tar.gz "Salt_Model_3D/3-D_Salt_Model/VEL_GRIDS/SALTF.ZIP"
unzip -o Salt_Model_3D/3-D_Salt_Model/VEL_GRIDS/SALTF.ZIP -d Salt_Model_3D/3-D_Salt_Model/VEL_GRIDS/
cd ../..

# 1. Generate (CPU, ~40 s for 512 samples)
python wave/operator_sim/generate_dataset_real.py --model marmousi \
    --num-samples 512 --nx 64 --nt 64 --seed 42 \
    --out operator_data/wave_operator_marmousi_n512_nx64_nt64_t1_seed42.npz

# 2, 3. Train and render exactly as above -- the npz is drop-in compatible.
```

Three things differ from the synthetic arm, each forced by the real data:

- **`rho` may vary.** Marmousi ships a density model, so channel 1 is no longer
  all ones. Overthrust and Salt ship velocity only and get `rho = 1`; Gardner's
  relation could synthesise a density, but it is an empirical fit and would add
  fiction while claiming realism.
- **The train/val split ships with the dataset.** Neighbouring traces are metres
  apart and near-identical — in Marmousi, 4 m apart with a mean `|dVp|` of
  6 m/s, 0.2 % of the mean — so a random split scores near duplicates of the
  training profiles as held out. Validation instead takes four contiguous trace
  blocks with 320 m buffers either side. `train_operators.py` honours a `split`
  array when the dataset carries one and falls back to its random split
  otherwise, so the synthetic arms are unaffected.
- **The FD solve is spatially refined** (`--refine`, default 8) and sampled back
  onto the output grid, rather than solved at output resolution.

Contrast bands (`kinds`) are **per-model terciles**, so `high_contrast` means
something different in each model — Marmousi's high band starts at 2.96x,
Salt's at 1.61x. They group samples within a model; for cross-model comparison
use the absolute `contrast` array each dataset carries.

### On `--refine`

Measured rel L2 against a `refine=32` reference, averaged over six Marmousi
profiles:

| refine | FD grid | rel L2 | cost / 512 samples |
|---|---|---|---|
| 1 | 64 | 13.35 % | 1.8 s |
| 2 | 127 | 3.87 % | 2.9 s |
| 4 | 253 | 1.14 % | 8.9 s |
| **8** | **505** | **0.32 %** | **22 s** |
| 16 | 1009 | 0.08 % | 41 s |

`refine=1` is what `generate_dataset.py` and `generate_dataset_source.py` still
do, and on their own `layered` / `two_layer` profiles it carries **10.4 % mean
error** (worst 18.8 %) against the converged solution. That is larger than the
2.07 % rel L2 reported for FNO above, so those headline numbers are measured
against a target that is itself further from the true solution than the models
are from the target. Model-to-model *ordering* is unaffected — every arm was
scored against the same target — but the absolute percentages should not be
read as physical accuracy. Regenerating the synthetic sets with `--refine 8`
would fix this; it has not been done.

## Rendering notes

- Every panel is shown against the **finite-difference reference**. Without it,
  a model that outputs a near-zero field looks like a smooth, plausible result.
- The colour scale is the **98th percentile of |u|**, not the global maximum.
  The `t = 0` pulse is ~1.8× taller than the two waves it splits into, so
  scaling to the global max pushed everything after the first frame into the
  pale middle of the colormap. The initial pulse now saturates for a few
  frames, which is the intended trade.
- `render_simulations.py` picks the median-error sample per material kind, so
  the GIFs are representative rather than cherry-picked. Override with
  `--positions`.
