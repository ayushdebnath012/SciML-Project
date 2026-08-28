"""OpenFWI loading, normalization, and representation oracles. Pure NumPy.

Deliberately torch-free: dataset prep and the oracle numbers are the part of
this pipeline that has to be reproducible off the GPU box, and on the Windows
client "import torch" is blocked outright. train_openfwi.py wraps a
torch.utils.data.Dataset around ChunkedArray.

Forward direction: velocity map (1, 70, 70) -> shot gathers (5, 1000, 70).
The gather's receiver axis and the velocity map's lateral axis are the *same*
70-point grid at dx = 10 m -- receivers sit on every surface grid point -- so a
model only has to turn the depth axis into a time axis, not resample laterally.
"""
import json
from pathlib import Path

import numpy as np

# Verbatim from lanl/OpenFWI dataset_config.json. data_min/data_max are the
# published amplitude bounds the official baselines normalize with; using them
# (rather than statistics refitted per subset) is what keeps a number here
# comparable to a number in the OpenFWI paper.
DATASET_CONFIG = {
    "flatvel-a":    {"data_min": -26.95, "data_max": 52.77},
    "flatvel-b":    {"data_min": -27.17, "data_max": 56.05},
    "curvevel-a":   {"data_min": -27.11, "data_max": 55.10},
    "curvevel-b":   {"data_min": -29.04, "data_max": 57.03},
    "flatfault-a":  {"data_min": -26.10, "data_max": 50.86},
    "flatfault-b":  {"data_min": -24.86, "data_max": 50.28},
    "curvefault-a": {"data_min": -26.48, "data_max": 52.32},
    "curvefault-b": {"data_min": -24.93, "data_max": 50.98},
    "style-a":      {"data_min": -24.96, "data_max": 48.93},
    "style-b":      {"data_min": -23.76, "data_max": 46.01},
}
# Shared by every 2D family above.
_COMMON = {"label_min": 1500, "label_max": 4500, "file_size": 500, "nbc": 120,
           "dx": 10, "nt": 1000, "dt": 0.001, "f": 15, "n_grid": 70,
           "ns": 5, "ng": 70, "sz": 10, "gz": 10}
for _cfg in DATASET_CONFIG.values():
    _cfg.update(_COMMON)

N_TRAIN_CHUNKS = 48
N_VAL_CHUNKS = 12


def dataset_key(name):
    """Map a directory name to a config key: FlatVel_A -> flatvel-a."""
    key = name.strip().replace("_", "-").lower()
    if key not in DATASET_CONFIG:
        raise KeyError("unknown OpenFWI dataset " + repr(name)
                       + "; known: " + str(sorted(DATASET_CONFIG)))
    return key


def chunk_indices(split, n_chunks):
    """Prefix of the official train (1..48) or val (49..60) chunk block."""
    if split == "train":
        avail = list(range(1, N_TRAIN_CHUNKS + 1))
    elif split == "val":
        avail = list(range(N_TRAIN_CHUNKS + 1, N_TRAIN_CHUNKS + N_VAL_CHUNKS + 1))
    else:
        raise ValueError(split)
    if n_chunks > len(avail):
        raise ValueError("%s has %d chunks, asked for %d"
                         % (split, len(avail), n_chunks))
    return avail[:n_chunks]


class ChunkedArray:
    """Global indexing across a list of memory-mapped .npy chunks.

    A 700 MB seismic chunk per 500 samples means a 4-chunk training split is
    2.8 GB; memory-mapping keeps that off the heap and lets the OS page cache
    do the work, which matters on a box shared with other tenants.
    """

    def __init__(self, paths, preload=False):
        self.arrays = [np.load(p, mmap_mode=None if preload else "r") for p in paths]
        self.sizes = [len(a) for a in self.arrays]
        self.offsets = np.cumsum([0] + self.sizes)
        self.paths = list(paths)

    def __len__(self):
        return int(self.offsets[-1])

    @property
    def sample_shape(self):
        return tuple(self.arrays[0].shape[1:])

    def __getitem__(self, i):
        c = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return np.asarray(self.arrays[c][i - self.offsets[c]])

    def take(self, indices):
        return np.stack([self[int(i)] for i in indices])


def split_paths(root, dataset, split, n_chunks):
    root = Path(root).expanduser()
    ids = chunk_indices(split, n_chunks)
    data = [root / dataset / "data" / ("data%d.npy" % i) for i in ids]
    model = [root / dataset / "model" / ("model%d.npy" % i) for i in ids]
    missing = [p for p in data + model if not p.exists()]
    if missing:
        raise SystemExit("missing chunks (run fetch_openfwi.py first):\n  "
                         + "\n  ".join(str(p) for p in missing[:6]))
    return data, model


def load_split(root, dataset, split, n_chunks, preload=False):
    data, model = split_paths(root, dataset, split, n_chunks)
    return ChunkedArray(data, preload), ChunkedArray(model, preload)


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------
class OpenFWINorm:
    """OpenFWI's published min/max scaling, both axes mapped to [-1, 1].

    denorm_seismic is the inverse used before scoring: every error in this
    benchmark is reported on physical amplitudes, not on the [-1, 1] surrogate,
    so a model cannot look better by predicting into a compressed range.
    """

    def __init__(self, key):
        # `key` is either an OpenFWI dataset key or a config dict, so a cache
        # built by fetch_ssgen.py (which has no published constants) can use
        # the same scorer with statistics measured off its own training split.
        cfg = DATASET_CONFIG[key] if isinstance(key, str) else key
        self.data_min = float(cfg["data_min"])
        self.data_max = float(cfg["data_max"])
        self.label_min = float(cfg["label_min"])
        self.label_max = float(cfg["label_max"])

    def norm_velocity(self, v):
        return 2.0 * (v - self.label_min) / (self.label_max - self.label_min) - 1.0

    def norm_seismic(self, d):
        return 2.0 * (d - self.data_min) / (self.data_max - self.data_min) - 1.0

    def denorm_seismic(self, d):
        return (d + 1.0) * 0.5 * (self.data_max - self.data_min) + self.data_min

    @property
    def seismic_scale(self):
        """Physical amplitude = normalized * scale + shift."""
        return 0.5 * (self.data_max - self.data_min)

    @property
    def seismic_shift(self):
        return 0.5 * (self.data_max + self.data_min)

    def asdict(self):
        return {"data_min": self.data_min, "data_max": self.data_max,
                "label_min": self.label_min, "label_max": self.label_max}


class ZScoreNorm:
    """Gathers standardized by training-split mean/std; velocity by min/max.

    OpenFWI's published min/max constants exist to keep numbers comparable
    across papers, and are worth the cost there. A cache with no published
    constants gets no such benefit, and on heavy-tailed field-scale amplitudes
    min/max is actively harmful: the SubsurfaceGen maximum is a rare
    direct-arrival spike, 99.9 % of samples sit below 1 % of it, and scaling by
    it leaves the signal occupying 0.17 % of [-1, 1]. MSE training in that
    representation is dominated by the constant offset -- measured, it left
    PFNO at 98.9 % validation error with a vanishing train MSE, which is the
    signature of a model that fitted the mean and nothing else.

    Velocity keeps min/max: it is bounded, near-uniform over 1500-4700 m/s, and
    has no tail problem.

    Scoring is unaffected either way. `seismic_scale`/`seismic_shift` describe
    the affine map back to physical amplitudes, and every error in this
    benchmark is computed there.
    """

    def __init__(self, cfg):
        self.data_mean = float(cfg["data_mean"])
        self.data_std = float(cfg["data_std"])
        self.label_min = float(cfg["label_min"])
        self.label_max = float(cfg["label_max"])

    def norm_velocity(self, v):
        return 2.0 * (v - self.label_min) / (self.label_max - self.label_min) - 1.0

    def norm_seismic(self, d):
        return (d - self.data_mean) / self.data_std

    def denorm_seismic(self, d):
        return d * self.data_std + self.data_mean

    @property
    def seismic_scale(self):
        return self.data_std

    @property
    def seismic_shift(self):
        return self.data_mean

    def asdict(self):
        return {"mode": "zscore", "data_mean": self.data_mean,
                "data_std": self.data_std, "label_min": self.label_min,
                "label_max": self.label_max}


def gather_statistics(data, max_samples=200):
    """Streaming mean/std of a gather split, over at most `max_samples`.

    Exact over the sample axis rather than a running estimate: a split is a
    few hundred samples here, and a wrong std would silently rescale every
    error in the benchmark.
    """
    n = min(max_samples, len(data))
    idx = np.linspace(0, len(data) - 1, n).astype(int)
    total = 0.0
    total_sq = 0.0
    count = 0
    for i in idx:
        g = np.asarray(data[int(i)], dtype=np.float64)
        total += g.sum()
        total_sq += np.square(g).sum()
        count += g.size
    mean = total / count
    var = max(total_sq / count - mean * mean, 1e-24)
    return float(mean), float(np.sqrt(var))


# --------------------------------------------------------------------------
# representation oracles
# --------------------------------------------------------------------------
def rel_l2_percent(pred, target):
    """Per-sample relative L2 in percent. A zero prediction scores ~100."""
    p = pred.reshape(len(pred), -1)
    t = target.reshape(len(target), -1)
    num = np.linalg.norm(p - t, axis=1)
    den = np.maximum(np.linalg.norm(t, axis=1), 1e-12)
    return 100.0 * num / den


def band_limit_oracle(gathers, n_freqs):
    """Error floor from keeping only the lowest n_freqs rFFT bins along time.

    PFNO assigns one network per temporal frequency, so it can only afford a
    truncated band. No PFNO at this n_freqs can score below this number, and
    reporting it is what separates "the architecture lost" from "the band
    limit lost".
    """
    spec = np.fft.rfft(gathers, axis=-2)
    keep = np.zeros_like(spec)
    keep[..., :n_freqs, :] = spec[..., :n_freqs, :]
    recon = np.fft.irfft(keep, n=gathers.shape[-2], axis=-2)
    return rel_l2_percent(recon, gathers)


def time_resample_oracle(gathers, t_latent):
    """Error floor from decoding on a coarse time axis and interpolating up.

    FNO and GNO here build a latent on t_latent time points and linearly
    upsample to nt; that upsample is lossy and this measures by how much.
    """
    nt = gathers.shape[-2]
    src = np.linspace(0.0, 1.0, nt)
    dst = np.linspace(0.0, 1.0, t_latent)
    lead = gathers.shape[:-2]
    ng = gathers.shape[-1]
    flat = np.moveaxis(gathers, -2, -1).reshape(-1, nt)
    coarse = np.stack([np.interp(dst, src, row) for row in flat])
    back = np.stack([np.interp(src, dst, row) for row in coarse])
    recon = np.moveaxis(back.reshape(lead + (ng, nt)), -1, -2)
    return rel_l2_percent(recon, gathers)


# --------------------------------------------------------------------------
# manifest-driven caches (SubsurfaceGen and anything else fetch_ssgen writes)
# --------------------------------------------------------------------------
def load_meta(root):
    """Read the meta json a fetch_ssgen.py cache ships beside its shards."""
    return json.loads((Path(root).expanduser() / "ssgen_meta.json").read_text())


def manifest_split(root, meta, split, n_chunks=None, preload=False):
    """Load a split by the shard range the cache's manifest declares.

    OpenFWI's fixed 1-48 / 49-60 chunk convention does not apply here: the
    fetcher writes train, val and out-of-distribution blocks consecutively and
    records where each begins, so the split boundary travels with the data
    instead of being hard-coded.
    """
    root = Path(root).expanduser()
    entry = meta["manifest"][split]
    first, count = int(entry["first_chunk"]), int(entry["n_chunks"])
    ids = list(range(first, first + count))
    if n_chunks is not None:
        ids = ids[:n_chunks]
    data = [root / "data" / ("data%d.npy" % i) for i in ids]
    model = [root / "model" / ("model%d.npy" % i) for i in ids]
    missing = [q for q in data + model if not q.exists()]
    if missing:
        raise SystemExit("missing shards (run fetch_ssgen.py first): "
                         + ", ".join(str(q) for q in missing[:6]))
    return ChunkedArray(data, preload), ChunkedArray(model, preload)


def config_from_meta(meta, gather_shape, velocity_shape):
    """Build a config dict from the cache's own shapes and measured statistics.

    `gather_shape` is (ns, nt, ng) and `velocity_shape` is (1, nz, nx). The two
    lateral axes differ for a field-scale cache -- the velocity map is coarser
    than the receiver line -- which is exactly what nx_out in the models is for.
    """
    ns, nt, ng = (int(v) for v in gather_shape)
    nz, nx = int(velocity_shape[-2]), int(velocity_shape[-1])
    stats = meta["stats"]
    return {
        "ns": ns, "nt": nt, "ng": ng, "nz": nz, "nx": nx,
        "n_grid": nz,
        "dt": float(meta.get("propagation_time_s", 8.0)) / nt,
        "dx": 10.0 * (1000.0 / ng),
        "data_min": float(stats["gather_min"]), "data_max": float(stats["gather_max"]),
        "label_min": float(stats["velocity_min"]),
        "label_max": float(stats["velocity_max"]),
        "band": meta.get("band"), "source_indices": meta.get("source_indices"),
    }


def source_receiver_grid(cfg):
    """Physical source and receiver x-positions in metres.

    OpenFWI places ns shots evenly across the ng-point receiver line, both at
    grid depth sz/gz, so the gather's receiver axis shares the velocity map's
    lateral grid exactly.
    """
    ng, ns, dx = cfg["ng"], cfg["ns"], cfg["dx"]
    receivers = np.arange(ng, dtype=np.float64) * dx
    sources = np.linspace(0.0, (ng - 1) * dx, ns)
    return sources, receivers


def _main():
    import argparse
    p = argparse.ArgumentParser(description="oracle floors for a fetched split")
    p.add_argument("--root", default="openfwi_data")
    p.add_argument("--dataset", default="FlatVel_A")
    p.add_argument("--chunks", type=int, default=1)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--freqs", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--t-latent", type=int, nargs="+", default=[125, 250, 500])
    p.add_argument("--out", default=None)
    a = p.parse_args()

    data, _model = load_split(a.root, a.dataset, "val", a.chunks)
    idx = np.linspace(0, len(data) - 1, min(a.samples, len(data))).astype(int)
    gathers = data.take(idx).astype(np.float64)
    print("%s: %d val samples, oracle on %d, gather shape %s"
          % (a.dataset, len(data), len(idx), gathers.shape[1:]))

    report = {"dataset": a.dataset, "samples": int(len(idx)),
              "band_limit": {}, "time_resample": {}}
    for k in a.freqs:
        e = band_limit_oracle(gathers, k)
        report["band_limit"][str(k)] = {"mean": float(e.mean()), "max": float(e.max())}
        print("  band limit    %4d bins  rel L2 = %7.3f%%  (worst %7.3f%%)"
              % (k, e.mean(), e.max()))
    for t in a.t_latent:
        e = time_resample_oracle(gathers, t)
        report["time_resample"][str(t)] = {"mean": float(e.mean()), "max": float(e.max())}
        print("  time resample %4d pts   rel L2 = %7.3f%%  (worst %7.3f%%)"
              % (t, e.mean(), e.max()))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print("wrote", a.out)


if __name__ == "__main__":
    _main()
