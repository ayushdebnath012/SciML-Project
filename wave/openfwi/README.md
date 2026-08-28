# OpenFWI forward benchmark — FNO / PFNO / DeepONet / GNO

Four neural operators on the same learned map, over two OpenFWI datasets:

```
velocity model  (1, 70, 70)        ->   shot gathers  (5, 1000, 70)
(depth, offset), 1500-4500 m/s          (shot, time, receiver)
```

This is the **forward** direction — OpenFWI's own leaderboard task is the
inverse one (gathers → velocity), so these numbers are not comparable to
InversionNet/VelocityGAN. Forward is the direction the rest of this repository
learns (`wave/operator_sim/`), and it is the direction the PFNO paper
(Li et al., [arXiv:2209.12340](https://arxiv.org/abs/2209.12340)) uses OpenFWI
for. `docs/operator_simulations.md` notes that the 1D pipeline was a time-domain
analogue of that paper and "not a reproduction of their 2D OpenFWI experiment";
this is that experiment.

## Geometry, and why the models are shaped the way they are

From OpenFWI's `dataset_config.json`, identical across all 2D families:
70×70 grid at `dx = 10 m`, 5 shots, 70 receivers, 1000 time steps at
`dt = 1 ms`, 15 Hz Ricker source, sources and receivers at grid depth 10.

The receiver axis and the velocity map's lateral axis are the **same** 70-point
grid — receivers sit on every surface grid point — so a model only has to turn
the *depth* axis into a *time* axis. That single fact sets the architecture:

| model | body | how depth becomes time |
|---|---|---|
| FNO | spectral convolution on (z,x), then on (t,x) | `DepthToTime` + upsample head |
| GNO | local graph kernel integration, same two stages | `DepthToTime` + upsample head |
| PFNO | one 2D FNO per temporal frequency bin | never — it is frequency-native |
| DeepONet | branch(velocity) · trunk(shot, time, receiver) | never — the trunk is queried directly |

FNO and GNO deliberately share `DepthToTime` and `TimeDecoderHead`, so what
separates them is the block being stacked (global spectral vs local kernel
integral) and nothing else. PFNO and DeepONet do not fit that template and are
left in their own idiom.

## Representation floors

Two numbers are measured before any training, because both architectures make a
representation choice that a loss curve will not show you. They are *not* the
same kind of number, and conflating them would be wrong:

- **PFNO band limit — a hard floor.** One network per frequency means you can
  only afford a truncated band; bins at or above `--pfno-freqs` are set to
  exactly zero, which `test_openfwi.py` verifies. PFNO cannot score below this
  no matter how long it trains. At OpenFWI's 15 Hz source and `dt = 1 ms`, 64
  of the 501 rFFT bins are essentially free — **0.02 %** on real validation
  gathers from both datasets — but 32 bins would cost **4.2 %**. That the
  default is nearly free is a fact about this source, not about PFNO.
- **Time latent — a reference scale, not a bound.** FNO and GNO decode on
  `--t-latent` time points and upsample. Resampling a *single-channel* gather
  through 250 points and back costs **1.74 %** on real data (6.6 % at 125,
  0.46 % at 500). The models do not do that: their latent carries `width`
  channels over those 250 points — 32 × 250 numbers feeding 1000 outputs — so
  they have plenty of information to beat this figure, and the head's temporal
  convolutions exist to use it. Read it as the scale at which the time latent
  starts to matter, not as a floor.

Both are measured on the real validation gathers at run time and written into
`openfwi_summary.json` under `oracles`. FNO and GNO are held at the **same**
`t_latent` so that their comparison stays a comparison of blocks. Standalone:

```bash
python wave/openfwi/openfwi_data.py --root ~/openfwi_data --dataset FlatVel_A \
    --freqs 16 32 64 128 --t-latent 125 250 500
```

## Pipeline

```bash
# 1. Fetch. Chunks are 500 samples / 700 MB each; the official split is
#    chunks 1-48 train, 49-60 val, and --train/--val-chunks take a prefix of
#    each block so a subset never straddles that boundary.
python wave/openfwi/fetch_openfwi.py --datasets FlatVel_A CurveVel_A \
    --train-chunks 4 --val-chunks 1 --root ~/openfwi_data      # ~7 GB

# 2. Verify the models before spending GPU hours on them (CPU, seconds)
python wave/openfwi/test_openfwi.py

# 3. Train all four, one dataset
python wave/openfwi/train_openfwi.py --root ~/openfwi_data --dataset FlatVel_A \
    --train-chunks 4 --val-chunks 1 --epochs 120 --outdir ~/openfwi_results/flatvel_a

# 3'. Or both datasets, one per GPU
bash wave/openfwi/runners/run_openfwi.sh 120 4 FNO,PFNO,DeepONet,GNO
```

Data is fetched from the `ashynf/OpenFWI` Hugging Face mirror, because OpenFWI's
own distribution is per-dataset Google Drive folders that are not scriptable
from a headless box. Every downloaded file's `.npy` header is checked against
the shape `dataset_config.json` declares, and a mismatch deletes the file and
aborts — a mirror is only usable if it is verified.

## Scoring

Relative L2 (%), MAE and RMSE, all on **physical amplitudes**. The OpenFWI
normalization is affine with a non-zero offset, so a relative L2 computed in
`[-1, 1]` units is a different number and flatters a model that predicts near
the middle of the range. A zero prediction scores ~100 %.

Normalization uses the published `data_min`/`data_max` from `dataset_config.json`
rather than statistics refit on whatever subset was fetched, so a number here
stays comparable across chunk counts.

## Design notes

- **The GNO kernel is diagonal in channels.** `k(dz, dx, a_i, a_j)` returns a
  width-vector applied elementwise, not a width×width matrix. A matrix kernel
  costs `width`× more memory per edge — at 4900 nodes × 29 neighbours that
  decides whether the layer fits beside another tenant on the GPU — and the
  channel mixing it would add is what the `W v(x)` term already does. The
  kernel still reads the velocity at both endpoints, which is what separates it
  from a CNN with the same stencil; `test_openfwi.py` checks that dependence.
- **The GNO stencil is applied by shifting the padded feature map**, once per
  offset, rather than by scatter/gather on an edge list. The grid is regular and
  the stencil is identical at every node, so an edge list would store 142k
  redundant indices and force an unsorted scatter. Boundary nodes see a clipped
  ball and are averaged over their true neighbour count.
- **PFNO's branches are evaluated grouped, not looped.** `GroupedSpectralConv2d`
  runs all 64 branches in one einsum. It is checked against a loop of
  independent `SpectralConv2d` in `test_openfwi.py`; the loop version is what
  made the 1D PFNO in `wave/operator_sim` 11× slower than its FNO.
- **DeepONet gets a Fourier-feature trunk** (`--don-fourier`, default 32). A
  plain tanh trunk cannot represent a 15 Hz wavelet across a 1 s window, and
  would report the trunk's frequency ceiling rather than anything about the
  branch/trunk factorization. This favours DeepONet over its textbook form on
  purpose. `--don-fourier 0` recovers the plain version.
- **`DepthToTime` starts as linear interpolation.** The learned (T, nz) matrix
  is initialized to the interpolation operator, so the block begins as a plain
  resampler and departs from it only if the data pays for it.
- **Parameters are reported twice.** `parameters` counts a complex spectral
  weight once; `parameters_real` counts it as the two real scalars it is.
  Compare architectures on `parameters_real` — the nominal count understates
  FNO and PFNO 2× against the all-real DeepONet and GNO.
- **The split is cached, normalized, once** (`--cache gpu`, the default). A
  gather is 1.4 MB, so streaming 2000 of them per epoch means 2.8 GB of reads
  and 2.8 GB of float conversion *per epoch, per model*. Measured on the H100
  box that made FNO take 30 s/epoch against a 3.6 s compute cost — seven
  eighths of the run was the data path, and it would have been paid again for
  every architecture. `--cache ram` keeps it in host memory; `--cache none`
  restores the memmap DataLoader for splits too large to hold.

## Simulations

`report_openfwi.py` shows *where* a model is wrong; `render_openfwi.py` shows
*when*. Each frame is one time sample: the gather panels carry a cursor at the
current time and the wide panel underneath plots amplitude across the whole
receiver line at that instant, reference against every model.

```bash
python wave/openfwi/render_openfwi.py --run results/openfwi/curvevel_a \
    --outdir results/openfwi/simulations --shot 3 --pick median
python wave/openfwi/render_openfwi.py --run results/openfwi/curvevel_a \
    --pick worst --tag curvevel_a_worst      # where it breaks, not where it works
```

`--pick` takes `median` (default, so the animation is representative rather
than cherry-picked), `best`, `worst`, or an export index. `--stride` sets time
samples per frame; the default 8 turns 1000 steps into a 125-frame, ~10 s loop.

Two scaling choices, both for the same reason — the direct arrival is several
times taller than every reflection:

- 2D panels clip at the **98th percentile** of `|reference|`, so the reflections
  are visible instead of being washed into the pale middle of the colormap.
- The trace panel scales to the **99.5th percentile**, letting the direct
  arrival clip for the handful of frames it occupies. Scaling to the global
  maximum leaves ~90 % of frames a flat line near zero, and the reflections are
  the part the models get wrong.

It runs unchanged on a SubsurfaceGen run — the export has the same layout.

## Outputs

`--outdir` receives:

- `openfwi_summary.json` — config, split, oracle floors, per-model parameters,
  timings, rel L2 / MAE / RMSE, and per-sample errors
- `openfwi_histories.json` — per-epoch train MSE and validation rel L2
- `<Model>_state.pt` — best-validation checkpoint per model
- `openfwi_predictions.npz` — velocity, target and per-model predictions for a
  handful of validation samples, for plotting off the GPU box

## Results

Filled in from `openfwi_summary.json` once the sweep has run.
