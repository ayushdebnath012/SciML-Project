# Multi-physics subsurface dataset

One latent geology, three forward problems, three (model, data) pairs per
sample — the seismic (v, p) pair that OpenFWI provides, plus the two
electrical pairs the project is extending to:

| pair | model map | data | physics | PDE class | solver |
|---|---|---|---|---|---|
| (v, p) | velocity `(1,70,70)` m/s | shot gathers `(5,1000,70)` | acoustic wave | hyperbolic | `seismic_solver.py` |
| (c, m) | conductivity `(1,70,70)` S/m | MT response `(8,70,4)` — ρₐ + phase, TE and TM | quasi-static Maxwell | parabolic | `mt_solver.py` |
| (r, i) | resistivity `(1,70,70)` Ω·m | DC pole-pole ρₐ `(10,70)` | Poisson | elliptic | `dc_solver.py` |

All three maps are painted from the *same* layered geology (`geology.py`):
porosity and lithology per layer, then Raymer–Hunt–Gardner for velocity,
Gardner for density, Archie for resistivity. Fluid resistivity, saturation,
cementation exponent and matrix velocity are drawn per layer and Archie gets
a log-normal scatter, so velocity and log-resistivity are correlated
(ρ ≈ 0.3) but not functions of each other. That is deliberate: a dataset in
which σ = f(v) makes joint inversion trivial.

Geometry is OpenFWI's 2-D families exactly (70 × 70 cells at 10 m, 5 shots,
receivers on every column, 1000 × 1 ms, 15 Hz Ricker), so the existing
loaders, metrics and plots carry over.

## Build

```bash
uv run --with numpy --with scipy python wave/multiphysics/build_dataset.py --test            # 6 samples, ~20 s
uv run --with numpy --with scipy python wave/multiphysics/build_dataset.py \
    --out multiphysics_data/LayeredAB --n-train 2000 --n-val 500 --jobs 8                     # ~1 h on a laptop
uv run --with numpy --with scipy python wave/multiphysics/test_multiphysics.py               # analytic checks, ~30 s
```

`--from-openfwi ROOT DATASET` instead reuses real OpenFWI velocity models and
their gathers, derives conductivity from each map with the same recipe
(`geology.properties_from_velocity`) and runs only the electrical solvers.

Output mirrors OpenFWI: `<out>/<modality>/<modality><k>.npy` in chunks of 500,
train chunks numbered first, then val; `meta.json` records geometry,
frequencies, electrode columns, chunk ids and per-modality statistics on the
training chunks. Stored values are physical; the loader normalises (log10 for
the resistive quantities).

## Solver accuracy (measured, `test_multiphysics.py`)

| solver | reference | error |
|---|---|---|
| DC | half-space (exact by calibration); two-layer image series at 1:25 and 1:500 contrasts | < 0.9 % everywhere; 2.8 % only at the two shortest offsets over a 4.5-cell layer on a 1:250 basement |
| MT | 1-D Wait recursion, TE and TM, 5 Hz – 1 kHz | ρₐ < 0.5 % typical, 1.3 % worst (band edges at extreme contrast); phase < 0.5°, 1.8° at a 22 m skin depth |
| seismic | direct and reflected arrival times; boundary residual | arrivals within 3 ms; boundary energy 2 × 10⁻⁴ of peak |

Three things had to be got right and are commented where they live:

- **DC wavenumber quadrature must be fitted far beyond the electrode spread.** A layered earth is equivalent to image sources at 2nh — tens of km for a strong contrast. Fitted to the 830 m aperture the error was 4.7 %; fitted to 30 km with 20 wavenumbers it is < 1e-4.
- **DC far boundaries carry the Dey–Morrison mixed condition and sit ~110 km out.** A conductive layer over a resistive basement leaks current over L ≈ h ρ₂/ρ₁ (~10 km), and the mixed BC is only right once the field is radial again.
- **MT bottom boundary is a second-order Robin row (`du/dz + γu = 0`) on gentle padding.** First-order cost 2–3 % on a half-space at 5 Hz; a strong Cerjan sponge in the seismic solver reflects off its own inner edge (3 × 10⁻³ at strength 1.0 vs 2 × 10⁻⁴ at 0.2).

## Frequencies and electrodes

MT: 8 log-spaced frequencies, 5 Hz – 1 kHz. Skin depth 503 √(ρ/f) m spans
~20 m (2 Ω·m, 1 kHz) to ~16 km (5000 Ω·m, 5 Hz), i.e. from the first cells to
well below the 700 m model. DC: 10 current electrodes evenly spread across
the 70 surface columns, potentials at all 70; any four-electrode array is a
linear combination of these pole responses.
