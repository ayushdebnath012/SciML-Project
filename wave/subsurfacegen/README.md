# SubsurfaceGen field-scale forward benchmark

The same four operators as [`wave/openfwi/`](../openfwi/README.md) — FNO, PFNO,
DeepONet, GNO — on the field-scale dataset released with **SubsurfaceGen**
(Stitt et al., [arXiv:2605.30541](https://arxiv.org/abs/2605.30541)):

```
velocity slice (1, 309, 500)   ->   shot gathers (5, 572, 1000)
10 km x 6.19 km, 20 m grid          8 s at 3-25 Hz, 1000 receivers
```

Same code, same scoring, same task direction as the OpenFWI arm — the models
are shared (`wave/openfwi/openfwi_models.py`), parameterized by grid shape.
What changes is that this is real field-scale acquisition geometry, and that
changes what the architectures cost.

## Why this is not just "OpenFWI but bigger"

OpenFWI's 1 s record with a 15 Hz Ricker is heavily **over**sampled. An 8 s
record at 3–25 Hz is **critically** sampled. Measured on a real preview cube:

| operation | OpenFWI | SubsurfaceGen |
|---|---:|---:|
| keep 64 frequency bins | 0.02 % | **98.0 %** |
| keep 220 frequency bins | — | 3.46 % |
| coarse time latent | 1.74 % @ 250 | **65.8 %** @ 286 |
| decimate receivers 2× | — | **17.9 %** |

Every cheap subsampling that OpenFWI tolerates destroys this data. Two
consequences drive the whole configuration:

- **PFNO needs 220 of 287 branches, not 64.** The band a per-frequency model
  must cover is `f_max × T`, so an 8 s / 25 Hz record needs ~4× the branches of
  a 1 s / 15 Hz one — on a grid 30× larger. This is the clearest scaling
  statement the benchmark produces about the architecture.
- **Time and receiver axes are kept native.** `t_latent` is 286 and the head
  upsamples to 572; receivers stay at 1000 throughout.

The velocity map *is* downsampled 2× (area-average, 619×1000 → 309×500). It is
the input, and a velocity model is piecewise smooth where a wavefield is not.
The decoder head upsamples laterally back to 1000 receivers (`nx_out`), so no
output resolution is lost.

## Data

~99 MB per shot-gather cube, so an OpenFWI-sized benchmark would be 250 GB.
`fetch_ssgen.py` downloads each pair, keeps **5 of 64 sources** (matching
OpenFWI's shot count), writes a shard, and deletes the raw cube — peak disk is
one shard, and 780 samples compress to a 9.4 GB cache.

```bash
python wave/subsurfacegen/fetch_ssgen.py --root ~/ssgen_data \
    --train 600 --val 100 --ood 80 --jobs 8      # ~78 GB down, ~35 min
```

Splits are the dataset's own: `train` (5 geologies — f3, fault, gom,
salt_canopy, seam, sampled evenly rather than proportionally), `test_in_dist`,
and `test_out_dist` — **Penobscot, held out of training entirely**. The OOD
block is scored for every model and lands in the summary under `ood`, because
cross-geology generalization is what the dataset was built to measure.

## Running

```bash
bash wave/subsurfacegen/runners/run_ssgen.sh 80 FNO,PFNO,DeepONet,GNO 1
python wave/openfwi/report_openfwi.py --runs results/ssgen/ssgen --outdir results/ssgen/figures
```

Measured cost at this grid (batch 2, 600 training samples, one H100):

| model | real params | min/epoch | peak GB |
|---|---:|---:|---:|
| FNO | 5.40 M | 0.2 | 1.2 |
| PFNO (220 branches) | 2.19 M | 2.1 | 13.0 |
| DeepONet | 39.77 M | 0.5 | 13.1 |
| GNO | 0.11 M | 3.2 | 8.5 |

DeepONet's parameter count is not a tuning choice: its branch is a flat MLP over
a 154,500-pixel velocity map, so the first layer alone is 39.5 M weights. That a
flat branch does not scale to field-scale inputs is itself a result, and it
means a DeepONet loss here cannot be blamed on starved capacity.

## Read the headline numbers with the late-time column

**No model on this benchmark learned wave propagation.** Splitting relative L2
at 1.5 s — the direct arrival before, reflections after — on the exported
validation samples:

| model | full | t < 1.5 s | t > 1.5 s |
|---|---:|---:|---:|
| FNO | 39.8 % | 15.9 % | **100.0 %** |
| DeepONet | 45.2 % | 25.4 % | **107.9 %** |
| PFNO | 54.6 % | 30.0 % | **147.1 %** |
| GNO | 68.9 % | 61.9 % | **100.4 %** |

A zero prediction scores ~100 % by construction, so every model is *at or worse
than silence* after the direct arrival; PFNO and DeepONet emit artifacts that
are worse than predicting nothing. 83 % of the record's energy sits in the
direct arrival, which is why fitting that one loud event alone still yields a
40–70 % headline score.

The same split on OpenFWI CurveVel-A shows the contrast — FNO 22.3 %, GNO
39.5 %, PFNO 56.6 % after 0.25 s — so those models genuinely learned
reflections and that benchmark's ranking means what it appears to mean.

**The field-scale ranking is therefore a ranking of direct-arrival fidelity,
not of operator learning.** The most likely cause is data: 600 training slices
against the paper's 4096, at 80 epochs. Treat this arm as a demonstration that
the pipeline runs at field scale and a measurement of what it costs — not as
evidence about which operator models field-scale propagation best.

`report_openfwi.py` emits this as a `late-time %` column so the headline number
cannot be read without it.

## GNO and receptive field

GNO ranks 2nd of 4 on OpenFWI and **last** here. The likely cause is receptive
field, and the arithmetic is stark. A radius-`r` stencil over `L` encoder layers
reaches `r × L` cells:

| | cells reached | physical | domain | fraction |
|---|---:|---:|---:|---:|
| OpenFWI (r=3, L=4) | 12 | 120 m | 700 m | **17 %** |
| SubsurfaceGen (r=3, L=3) | 9 | 180 m | 10,000 m | **1.8 %** |

Field-scale propagation is global — energy crosses 10 km — and a kernel seeing
1.8 % of the model cannot connect a velocity anomaly to its signature in a
distant gather. FNO's spectral convolution has full receptive field at every
layer, which is exactly the property that survives the scale change.

Widening the stencil the obvious way does not work. Measured on this grid at
batch 2:

| config | reach | s/step | peak | min/epoch |
|---|---:|---:|---:|---:|
| r=3, L=3 (used) | 1.8 % | 0.65 | 8.5 GB | 3.2 |
| r=8, L=5 | 8 % | 4.20 | **54 GB** | 21.0 |
| r=12, L=6 | 14 % | — | **OOM** | — |

Stencil cost grows as `r²`, so buying back the receptive field costs 28 hours
per run and then stops fitting on the card at all. **That GNO's locality cannot
be scaled to field-scale domains within practical compute is the finding**, not
an artifact of the configuration chosen here.

The cheap fix, if this is pursued, is dilation: spacing the stencil by `d`
multiplies reach by `d` at identical cost, and a per-layer schedule
(1, 2, 4, 8, 16) would give r=3/L=5 a reach of 93 cells — 18.6 % of the domain,
matching OpenFWI's ratio — for about 5 min/epoch. Not implemented here.

## Shared implementation

Models, scoring, the training loop and the report generator all live in
`wave/openfwi/`. This directory adds only the fetcher, the runner and this
README. `train_openfwi.py --meta` switches it from OpenFWI's hard-coded chunk
convention and published normalization constants to a cache that carries its own
split layout, grid shapes and measured statistics in `ssgen_meta.json`.
