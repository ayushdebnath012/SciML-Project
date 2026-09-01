# FNO / PFNO benchmark on a rented notebook GPU

`run_bench_gpu.py` runs the forward-operator benchmark — velocity model →
shot gathers — for **FNO** and **PFNO** on one GPU, on either of two targets:

| Target | Map | Split | Notes |
| --- | --- | --- | --- |
| `FlatVel_B`, `CurveVel_B` | `(1,70,70) → (5,1000,70)` | 2000 train / 500 val | OpenFWI chunks 1–4 train, 49 val |
| `SubsurfaceGen` | `(309,500) → (5,572,1000)` | 600 train / 100 val / 80 OOD | field-scale, arXiv:2605.30541 |

The B families are the harder counterpart of the `FlatVel_A` / `CurveVel_A`
numbers already in `results/openfwi/`: same geometry classes, wider velocity
contrast, more layers. SubsurfaceGen adds field scale (10 km × 6.19 km, 8 s
records at 3–25 Hz) and holds Penobscot out of training entirely, so it also
reports an out-of-distribution number.

Model sizing is **not re-tuned here**. The OpenFWI settings are the defaults
the A-family table was produced with, and the SubsurfaceGen settings are the
measured ones from `wave/subsurfacegen/runners/run_ssgen.sh`. That is what
makes a new number comparable to an existing one.

## Run it

Internet must be enabled in the notebook (both datasets stream from Hugging
Face). Clone the repo into the session. The OpenFWI targets need nothing beyond
torch and numpy; **SubsurfaceGen additionally needs**

```bash
pip install pandas pyarrow h5py hdf5plugin
```

because its cubes are HDF5 behind a parquet index. The runner checks for these
before the first download rather than failing 40 minutes in. Then:

```bash
# 1. Always start here: probes the real models on the real GPU and prints
#    projected hours, peak memory and download size. Downloads nothing.
python wave/openfwi/runners/run_bench_gpu.py --plan
python wave/openfwi/runners/run_bench_gpu.py --plan --targets SubsurfaceGen

# 2. Prove the whole path end to end in a few minutes before spending a session
python wave/openfwi/runners/run_bench_gpu.py --preset smoke --targets FlatVel_B

# 3. The benchmark
python wave/openfwi/runners/run_bench_gpu.py                          # both B families
python wave/openfwi/runners/run_bench_gpu.py --targets SubsurfaceGen
```

Paths are chosen from the host: Kaggle puts data in `/kaggle/temp` and results
in `/kaggle/working/bench_results` (data is re-downloadable in minutes and
should not eat the 20 GB output quota); Lightning uses the studio directory.
Override with `--data-root` / `--outdir`.

If Kaggle assigns a Tesla P100 and its preinstalled PyTorch warns that the
wheel starts at `sm_70`, install a Pascal-compatible build before importing
the runner.  This pair was forward/backward tested on Kaggle's P100 image:

```bash
pip install --no-deps torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
pip install nvidia-cusparselt-cu12==0.6.3
```

## Fitting a session

`--plan` checks its projection against `--session-hours` (default 12) and says
so if it does not fit. Four ways out, in order of preference:

1. **`--split-models`.** Trains one model per invocation into its own result
   directory. Each work unit checkpoints at every epoch and automatically
   resumes that exact model, optimizer, scheduler, shuffle generator and RNG
   state when you re-issue the command. The transient checkpoint is removed
   only after the summary and final artifacts have been written, so a session
   that dies during PFNO continues from its last completed epoch and still
   leaves a finished FNO untouched.
   Strongly recommended for SubsurfaceGen, where PFNO is ~14× FNO's per-epoch
   cost. Note it changes the weight-init seed: the trainer seeds as
   `init_seed + 100 * i` for the model's position in the list it is handed, and
   a split invocation always passes a one-model list. A split run is a
   different random initialization from a combined one, not a different
   configuration — valid either way, just don't mix them silently in one table.
2. **One target per session.** Finished work is skipped on the next run, so
   resuming is just re-issuing the same command.
3. `--preset short` — 40 epochs for OpenFWI, 30 for SubsurfaceGen.
4. A faster GPU. PFNO is the expensive one on both targets: on the H100 box it
   was 5–6× FNO's per-epoch cost on OpenFWI and ~14× on SubsurfaceGen.

Reference per-epoch costs measured on an H100 NVL, for scaling your probe
against:

| | FNO | PFNO |
| --- | --- | --- |
| OpenFWI (2000 train, batch 8) | 4–7 s | 23–38 s |
| SubsurfaceGen (600 train, batch 2) | 11 s | 150 s |

## Memory

The runner caches the normalized split on the GPU when the probe says it fits
in 80% of VRAM, and falls back to host RAM otherwise — the A-family run was
I/O bound rather than compute bound until that cache existed, so it is worth
having, but no reported number depends on it. Force it with `--cache
gpu|ram|none`.

SubsurfaceGen is the tight one: PFNO peaked at 13 GB of activations at batch 2,
and its split cache is another ~8 GB. On a 16 GB T4 that means `--cache ram`
(the runner will choose this itself), and possibly `--batch-size 1`.

## Reading the SubsurfaceGen FNO number

The run prints representation floors before training. On SubsurfaceGen the
`fno_time_resample` oracle at `t_latent 286` is **65% relative L2** — resampling
an 8 s, 3–25 Hz record through a half-length time latent destroys most of the
signal. The FNO's latent is multi-channel so this is a reference rather than a
hard bound (the existing run scored 46%, better than the floor), but it means
the published FNO number on this dataset is partly an artefact of the decoder
geometry, not only of the operator.

If you want the architecture's actual ceiling rather than the comparable
number, add `--extra --t-latent 572 --gno-t-latent 572` — but report it as a
separate row, because it is no longer comparable to `results/ssgen/`. PFNO does
not have this problem: its band-limit floor at 220 bins is 3.4%.

## Getting results back

Each target writes `openfwi_summary.json`, `openfwi_histories.json`,
`openfwi_predictions.npz` and per-model `*_state.pt` into `<outdir>/<target>/`
— or into `<outdir>/<target>__<model>/`, one directory per model, under
`--split-models`. While a model is active, that directory also contains a
resumable `*_train_checkpoint.pt`; it is deleted after successful completion.
The final table merges both layouts, and the runner zips the tree at the end.
Download the zip, unpack it under `results/`, then render on any CPU box:

```bash
python wave/openfwi/report_openfwi.py \
    --runs results/openfwi/flatvel_b results/openfwi/curvevel_b \
    --outdir results/openfwi/figures
python wave/openfwi/render_openfwi.py --run results/openfwi/flatvel_b \
    --outdir results/openfwi/simulations
```

Errors are reported on physical amplitudes, never on the normalized surrogate —
the scaling has a non-zero offset, so a relative L2 in normalized units is a
different number and would flatter a model that predicts near the middle of the
range.
