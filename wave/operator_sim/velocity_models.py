"""Real subsurface material profiles from published velocity models.

Replaces the synthetic `sample_material_profile` sampler with 1D depth columns
taken from community-standard geological models. Supersedes the earlier
marmousi-only module; `--model marmousi` reproduces it exactly.

What is real and what is not
----------------------------
The *material* is real, in the sense that these are the velocity models the
exploration-geophysics community uses as ground truth benchmarks. They are not
field measurements: Marmousi and Overthrust are both synthetic models built to
be geologically realistic (Marmousi from a North Quenguela trough section,
Overthrust as a generic thrust belt). Well logs would be genuinely *measured*
Vp(z); these are a step short of that but far beyond tanh profiles.

The *wave field* is always FD-simulated -- there is no measured full-field
u(x,t) for this geometry, so the target has to be computed either way.

Registry
--------
    marmousi    2301 traces x 751 depth, 4 m,  Vp + rho    (water cap trimmed)
    overthrust  801x801 x 187 depth,    25 m,  Vp only     (basement trimmed)
    salt        676x676 x 210 depth,    20 m,  Vp only

Provenance is in MODELS[...]["source"]. Files live under operator_data/raw/
and are not in git.

Coarsening
----------
A depth column carries structure far finer than the 64-point operator grid, and
point-sampling it down aliases interfaces into noise. `backus_coarsen` applies
the correct long-wavelength effective medium for normal-incidence 1D
propagation (Backus 1962):

    rho_eff = <rho>              (arithmetic)
    M_eff   = 1 / <1/M>          (harmonic, M = rho Vp^2)

Harmonic averaging of the modulus is the same principle the FD solver already
uses for its interface stiffness E_{i+1/2}, applied over a block not a face.

Nondimensionalisation
---------------------
Divide by fixed reference values (medians over the whole model) so the outputs
sit in the same O(1) range as the synthetic arm:

    rho~ = rho / rho_ref        E~ = M / (rho_ref Vp_ref^2)

which preserves the wave speed exactly: c~ = sqrt(E~/rho~) = Vp / Vp_ref. One
global reference per model, so genuine sample-to-sample speed variation
survives instead of being normalised away.

Models without a density file get rho == 1 throughout, matching the synthetic
arm. Gardner's relation could synthesise one, but it is an empirical fit -- it
would add fiction while claiming realism.
"""
from pathlib import Path

import numpy as np

# Terciles of the coarsened velocity contrast Vp_max/Vp_min, measured per model
# over 2000 random windows (see `measure_contrast_terciles`). Used only as a
# grouping label. Depth band was the obvious alternative but is near-useless:
# windows span most of the section, so their centres pile up mid-model.
CONTRAST_TERCILES = {
    "marmousi": (2.32, 2.96),
    "overthrust": (1.71, 1.99),
    "salt": (1.41, 1.61),
}
# NOTE these are per-model terciles, so "high_contrast" means something
# different in each model (Marmousi's high band starts at 2.96x, Salt's at
# 1.61x). They group samples *within* a model. For cross-model comparison use
# the absolute `contrast` array stored in each dataset.

MODELS = {
    "marmousi": {
        "kind": "pair2d",
        "shape": (2301, 751),          # (trace, depth) after reshape
        "dz": 4.0,
        "dx": 4.0,                     # spacing along the split axis
        "dtype": "<f4",
        "files": {"vp": "marmousi/vp.bin", "rho": "marmousi/rho.bin"},
        "trim_top": 12,                # water cap: 1500 m/s, 7-11 samples
        "trim_bottom": 0,
        "source": "https://www.geoazur.fr/WIND/pub/nfs/FWI-DATA/GEOMODELS/Marmousi/",
        "label": "Marmousi",
    },
    "overthrust": {
        "kind": "cube",
        "shape": (187, 801, 801),      # (depth, y, x), n1=x fastest
        "dz": 25.0,
        "dx": 25.0,
        "dtype": ">f4",                # 1997 SEG data, big-endian
        "files": {"vp": "Overthrust_3D_CD1/3-D_Overthrust_Model_Disk1/"
                        "3D-Velocity-Grid/overthrust.vites"},
        "trim_top": 0,
        "trim_bottom": 26,             # uniform 6000 m/s basement (.vo: z ends 4000 m)
        "stride": 8,                   # horizontal subsample -> 101x101 traces
        "source": "https://s3.amazonaws.com/open.source.geoscience/open_data/"
                  "seg_eage_models_cd/Overthrust_3D_CD1.tar.gz",
        "label": "SEG/EAGE Overthrust",
    },
    "salt": {
        "kind": "cube",
        "shape": (210, 676, 676),      # (depth, y, x), n1=x fastest
        "dz": 20.0,
        "dx": 20.0,
        "dtype": ">f4",
        "files": {"vp": "Salt_Model_3D/3-D_Salt_Model/VEL_GRIDS/Saltf@@"},
        # Top 3 slices are pure 1500 m/s water. Deeper slices still contain
        # water where the seafloor dips, which is left in -- it is part of the
        # real marine model and a fixed trim cannot follow bathymetry.
        "trim_top": 3,
        "trim_bottom": 27,             # constant basement from idx 183
        "stride": 7,                   # -> 97x97 traces
        "source": "https://s3.amazonaws.com/open.source.geoscience/open_data/"
                  "seg_eage_models_cd/Salt_Model_3D.tar.gz  (VEL_GRIDS/SALTF.ZIP)",
        "label": "SEG/EAGE Salt",
    },
}


def _resolve(root, rel):
    p = Path(root) / rel
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. See MODELS[...]['source'] for the download URL.")
    return p


def load_model(name, root="operator_data/raw"):
    """Load a model as (vp, rho, split_pos, spec).

    vp, rho    -- (n_trace, n_depth) float64, trimmed of constant caps.
                  rho is all ones for models that ship no density.
    split_pos  -- (n_trace,) position along the split axis, in metres. Used by
                  `make_trace_split` so the buffer is a physical distance and
                  means the same thing for every model.
    """
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; have {sorted(MODELS)}")
    spec = MODELS[name]
    top, bot = spec["trim_top"], spec["trim_bottom"]

    if spec["kind"] == "pair2d":
        n_trace, n_depth = spec["shape"]
        arrs = {}
        for key, rel in spec["files"].items():
            raw = np.fromfile(_resolve(root, rel), dtype=spec["dtype"])
            if raw.size != n_trace * n_depth:
                raise ValueError(f"{rel}: {raw.size} values, expected {n_trace*n_depth}")
            arrs[key] = raw.reshape(n_trace, n_depth).astype(np.float64)
        vp = arrs["vp"]
        rho = arrs.get("rho")
        split_pos = np.arange(n_trace, dtype=np.float64) * spec["dx"]

    elif spec["kind"] == "cube":
        nz, ny, nx = spec["shape"]
        cube = np.memmap(_resolve(root, spec["files"]["vp"]),
                         dtype=spec["dtype"], mode="r", shape=(nz, ny, nx))
        st = spec.get("stride", 1)
        # (depth, y, x) -> (trace, depth) with trace ordered y-major, so a run
        # of consecutive traces is a contiguous band of rows.
        sub = np.asarray(cube[:, ::st, ::st], dtype=np.float64)
        nz_, ny_, nx_ = sub.shape
        vp = sub.transpose(1, 2, 0).reshape(ny_ * nx_, nz_)
        rho = None
        ys = np.repeat(np.arange(ny_, dtype=np.float64) * st * spec["dx"], nx_)
        split_pos = ys

    else:
        raise ValueError(f"unknown model kind {spec['kind']!r}")

    hi = vp.shape[1] - bot
    vp = vp[:, top:hi]
    rho = np.ones_like(vp) if rho is None else rho[:, top:hi]
    if not (vp > 0).all() or not (rho > 0).all():
        raise ValueError(f"{name}: non-positive Vp or rho after trimming")
    return vp, rho, split_pos, spec


def reference_values(vp, rho):
    """Global nondimensionalisation constants (medians over the model)."""
    return float(np.median(vp)), float(np.median(rho))


def backus_coarsen(vp_fine, rho_fine, n_out):
    """Effective-medium downsample of a depth column onto `n_out` points."""
    n_fine = len(vp_fine)
    if n_fine < n_out:
        raise ValueError(f"cannot coarsen {n_fine} samples onto {n_out} points")
    edges = np.linspace(0, n_fine, n_out + 1).round().astype(int)
    modulus = rho_fine * vp_fine ** 2
    rho_c = np.empty(n_out)
    mod_c = np.empty(n_out)
    for i in range(n_out):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        rho_c[i] = rho_fine[lo:hi].mean()
        mod_c[i] = 1.0 / np.mean(1.0 / modulus[lo:hi])
    return np.sqrt(mod_c / rho_c), rho_c


def contrast_label(vp_coarse, terciles):
    """Group a profile by velocity contrast: low / moderate / high."""
    ratio = float(vp_coarse.max() / vp_coarse.min())
    lo, hi = terciles
    if ratio < lo:
        return "low_contrast", ratio
    if ratio < hi:
        return "moderate_contrast", ratio
    return "high_contrast", ratio


def make_trace_split(split_pos, val_fraction=0.2, n_blocks=4, buffer_m=320.0):
    """Partition traces into train / validation pools by contiguous blocks.

    Neighbouring traces in these models are metres apart and near-identical --
    in Marmousi, 4 m apart with a mean |dVp| of 6 m/s, 0.2 % of the mean.
    Splitting samples at random would put near duplicates of training profiles
    into validation and report a generalisation error that is really an
    interpolation error. So validation takes whole blocks along the survey
    line, spread over `n_blocks` regions so it still sees varied geology, with
    a `buffer_m` gap on each side of every block discarded.

    `split_pos` is in metres, so `buffer_m` means the same physical separation
    for every model regardless of its trace spacing.
    """
    split_pos = np.asarray(split_pos, dtype=np.float64)
    lo_c, hi_c = split_pos.min(), split_pos.max()
    span = hi_c - lo_c
    block_len = val_fraction * span / n_blocks
    stride = span / n_blocks

    in_val = np.zeros(len(split_pos), dtype=bool)
    in_buf = np.zeros(len(split_pos), dtype=bool)
    for b in range(n_blocks):
        centre = lo_c + (b + 0.5) * stride
        blo, bhi = centre - 0.5 * block_len, centre + 0.5 * block_len
        in_val |= (split_pos >= blo) & (split_pos < bhi)
        in_buf |= (split_pos >= blo - buffer_m) & (split_pos < bhi + buffer_m)

    role = np.zeros(len(split_pos), dtype=np.int8)
    role[in_buf & ~in_val] = -1
    role[in_val] = 1
    train = np.flatnonzero(role == 0)
    val = np.flatnonzero(role == 1)
    if len(train) == 0 or len(val) == 0:
        raise ValueError("trace split left an empty pool; reduce buffer_m or n_blocks")
    return train, val


def sample_profile(rng, vp, rho, n_out, vp_ref, rho_ref, terciles,
                   trace_pool=None, min_window=None, max_window=None,
                   min_window_frac=0.4, dz=1.0):
    """Draw one nondimensional (E, rho, kind, meta) tuple from a depth window.

    `min_window` defaults to `min_window_frac` of the available depth, floored
    at `n_out`, so the same call works for a 739-sample Marmousi column and a
    161-sample Overthrust one.
    """
    n_trace, n_depth = vp.shape
    if min_window is None:
        min_window = max(n_out, int(round(min_window_frac * n_depth)))
    if max_window is None:
        max_window = n_depth
    max_window = min(max_window, n_depth)
    if min_window > max_window:
        raise ValueError(f"min_window {min_window} > available depth {max_window}")

    if trace_pool is None:
        trace = int(rng.integers(0, n_trace))
    else:
        trace = int(trace_pool[rng.integers(0, len(trace_pool))])
    window = int(rng.integers(min_window, max_window + 1))
    top = int(rng.integers(0, n_depth - window + 1))

    vp_c, rho_c = backus_coarsen(vp[trace, top:top + window],
                                 rho[trace, top:top + window], n_out)
    rho_nd = rho_c / rho_ref
    E_nd = rho_nd * (vp_c / vp_ref) ** 2

    kind, ratio = contrast_label(vp_c, terciles)
    meta = {
        "trace": trace,
        "top": top,
        "window": window,
        "contrast": ratio,
        "depth_centre_m": (top + 0.5 * window) * dz,
    }
    return E_nd, rho_nd, kind, meta


def measure_contrast_terciles(vp, rho, n_out, n_draw=2000, seed=0, **kw):
    """Empirical terciles of the coarsened contrast, for CONTRAST_TERCILES."""
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_draw):
        _, _, _, meta = sample_profile(rng, vp, rho, n_out, 1.0, 1.0, (1e9, 1e9), **kw)
        ratios.append(meta["contrast"])
    return tuple(np.percentile(ratios, [33.3, 66.7]).round(2))
