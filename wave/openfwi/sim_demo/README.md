# Forward-simulation demo — what the operators are actually learning

A standalone 2D acoustic finite-difference solver running **OpenFWI's exact
acquisition geometry**, used to produce explanatory figures for talks and decks:

```
velocity model (70, 70)   ->   shot gathers (5, 1000, 70)
```

This is not part of the benchmark and trains nothing. `wave/openfwi/` is where
the actual experiment lives. What this adds is the picture the benchmark never
shows you: the wavefield in the subsurface, which the operator is never given
and must reproduce the surface trace of.

## Why it exists

`render_openfwi.py` animates *predictions* against reference, and needs
`openfwi_predictions.npz` from a training run. These figures need neither a
trained model nor the OpenFWI download — they run on CPU in about three seconds
and are reproducible anywhere.

## Geometry

Taken from OpenFWI's `dataset_config.json`, identical across all 2D families:

| | |
|---|---|
| grid | 70 × 70 at `dx = 10 m` |
| record | 1000 steps at `dt = 1 ms` |
| source | 15 Hz Ricker, 5 shots, grid depth 10 |
| receivers | 70, grid depth 10 |
| absorbing pad | `nbc = 120`, multiplicative sponge |

CFL is `v_max·dt/dx = 4500 × 0.001 / 10 = 0.45`, comfortably inside the 2D
stability limit of `1/√2 ≈ 0.707`.

The velocity models are *styled after* FlatVel-A and CurveVel-A — flat layers
with velocity increasing with depth, and the same layers bent by a smooth
sinusoid. They are not drawn from the OpenFWI distribution and are not used to
score anything; they exist so the flat-versus-curved contrast is visible in one
figure.

## Running

```bash
python wave/openfwi/sim_demo/forward2d.py    # simulate  -> sim_flat.npz, sim_curve.npz
python wave/openfwi/sim_demo/render.py       # render    -> results/openfwi/simulations/
```

The `.npz` files are gitignored (`*.npz`) and regenerate in ~3 s, so only the
two scripts are tracked.

## Output

Written to `results/openfwi/simulations/`:

| file | what |
|---|---|
| `task_velocity_to_gather.png` | the learned map: velocity model → forward simulation → shot gather |
| `wavefield_snapshots.png` | six snapshots of one shot propagating through the curved model, velocity interfaces contoured |
| `five_shots.png` | all five shots from one model — a single training target |
| `flat_vs_curve.png` | flat against curved layering, and what each does to the recorded data |

## Display note

Gather panels carry a standard seismic **t-gain** (`t^1.3`). Without it the
direct arrival saturates the colour scale and the reflections — the part every
model gets wrong, and the whole point of the late-time metric — are invisible.
Wavefield snapshots are instead scaled per panel, because the field decays by
orders of magnitude between 60 ms and 650 ms.
