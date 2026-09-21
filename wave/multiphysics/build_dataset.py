"""Generate the multi-physics dataset in OpenFWI's chunked layout.

    python wave/multiphysics/build_dataset.py --out multiphysics_data/LayeredAB --n-train 2000 --n-val 500 --jobs 8
    python wave/multiphysics/build_dataset.py --test                    # 6 tiny samples, a minute

Every sample is one latent geology (geology.py) and its three forward
responses. Files, per modality, in chunks of --chunk samples numbered from 1
(train chunks first, then val, as OpenFWI does):

    <out>/velocity/velocity<k>.npy       (n, 1, 70, 70)   m/s
    <out>/density/density<k>.npy         (n, 1, 70, 70)   kg/m^3
    <out>/conductivity/conductivity<k>   (n, 1, 70, 70)   S/m
    <out>/resistivity/resistivity<k>     (n, 1, 70, 70)   ohm m          (= 1/conductivity)
    <out>/porosity/porosity<k>           (n, 1, 70, 70)   -              latent
    <out>/layer_id/layer_id<k>           (n, 1, 70, 70)   int8           latent
    <out>/seismic/seismic<k>             (n, 5, 1000, 70) solver units   (v, p)
    <out>/mt/mt<k>                       (n, 8, 70, 4)    ohm m / deg    (c, m)  [rho_a TE, phase TE, rho_a TM, phase TM]
    <out>/dc/dc<k>                       (n, 10, 70)      ohm m          (r, i)  pole-pole apparent resistivity
    <out>/meta.json                      geometry, frequencies, electrode columns, chunk ids, per-modality stats

Physical units are stored; normalisation (log10 for the resistive
quantities) is the loader's business, as in the OpenFWI pipeline, so every
error is reportable in physical units. Stats in meta.json are measured on the
training chunks only.

--from-openfwi ROOT DATASET  reuses real OpenFWI velocity models and their
gathers: conductivity is derived from the velocity map by the same
rock-physics recipe (geology.properties_from_velocity) and only the two
electrical solvers run.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geology                                                   # noqa: E402
from dc_solver import DCSolver                                   # noqa: E402
from mt_solver import MTSolver, default_frequencies              # noqa: E402
from seismic_solver import SeismicSolver                         # noqa: E402

MAPS = ("velocity", "density", "conductivity", "resistivity", "porosity", "layer_id")
DATA = ("seismic", "mt", "dc")

_solvers = {}


def _get_solvers(cfg):
    """Solvers are built once per worker process (they hold grids and the DC calibration)."""
    if not _solvers:
        _solvers["dc"] = DCSolver(nz=cfg["nz"], ny=cfg["ny"], dx=cfg["dx"], n_k=cfg["dc_n_k"],
                                  src_cols=cfg["dc_src_cols"])
        _solvers["mt"] = MTSolver(nz=cfg["nz"], ny=cfg["ny"], dx=cfg["dx"], refine=cfg["mt_refine"])
        _solvers["seis"] = SeismicSolver(nz=cfg["nz"], ny=cfg["ny"], dx=cfg["dx"], dt=cfg["dt"],
                                         nt=cfg["nt"], f0=cfg["f0"], ns=cfg["ns"], nbc=cfg["nbc"])
    return _solvers


def make_sample(args):
    """One synthetic sample: (seed, cfg) -> dict of arrays."""
    seed, cfg = args
    rng = np.random.default_rng(seed)
    g = geology.sample_geology(rng, cfg["nz"], cfg["ny"], cfg["dx"])
    out = {m: g[m] for m in MAPS}
    out.update(forward(out, cfg, seismic=True))
    return out


def forward(maps, cfg, seismic=True):
    S = _get_solvers(cfg)
    res = {"mt": S["mt"].response(maps["conductivity"], cfg["frequencies"]),
           "dc": S["dc"].apparent_resistivity(maps["conductivity"])}
    if seismic:
        res["seismic"] = S["seis"].shots(maps["velocity"])
    return res


def openfwi_sample(args):
    """(seed, cfg, velocity, gather) -> dict; conductivity derived, seismic copied."""
    seed, cfg, v, gather = args
    rng = np.random.default_rng(seed)
    p = geology.properties_from_velocity(v, rng)
    out = {m: p[m] for m in MAPS}
    out.update(forward(out, cfg, seismic=False))
    out["seismic"] = gather.astype(np.float32)
    return out


class ChunkWriter:
    def __init__(self, root, chunk, first_id=1):
        self.root, self.chunk, self.k = Path(root), chunk, first_id
        self.buf = {}
        self.ids = []

    def add(self, sample):
        for key, arr in sample.items():
            arr = np.asarray(arr)
            if key in MAPS:
                arr = arr[None]                              # (1, nz, ny) channel axis, like OpenFWI models
            self.buf.setdefault(key, []).append(arr)
        if len(self.buf["velocity"]) == self.chunk:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        for key, arrs in self.buf.items():
            d = self.root / key
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / ("%s%d.npy" % (key, self.k)), np.stack(arrs))
        self.ids.append(self.k)
        self.k += 1
        self.buf = {}


def stats_over_chunks(root, key, ids, log10=False):
    vals = []
    for k in ids:
        a = np.load(Path(root) / key / ("%s%d.npy" % (key, k)), mmap_mode="r")
        a = np.asarray(a, np.float64)
        if log10:
            a = np.log10(np.maximum(a, 1e-12))
        vals.append((a.min(), a.max(), a.mean(), (a ** 2).mean(), a.size))
    v = np.array(vals)
    n = v[:, 4]
    mean = (v[:, 2] * n).sum() / n.sum()
    var = (v[:, 3] * n).sum() / n.sum() - mean ** 2
    return {"min": float(v[:, 0].min()), "max": float(v[:, 1].max()),
            "mean": float(mean), "std": float(np.sqrt(max(var, 0.0))), "log10": log10}


def build_meta(cfg, train_ids, val_ids, out):
    stats = {}
    for key in ("velocity", "density", "porosity", "seismic"):
        stats[key] = stats_over_chunks(out, key, train_ids)
    for key in ("conductivity", "resistivity", "dc"):
        stats[key] = stats_over_chunks(out, key, train_ids, log10=True)
    # mt: apparent resistivity channels in log10, phase channels linear
    mt = [np.load(Path(out) / "mt" / ("mt%d.npy" % k)) for k in train_ids]
    mt = np.concatenate(mt)
    stats["mt"] = {"rho_a_log10": {"mean": float(np.log10(mt[..., [0, 2]]).mean()), "std": float(np.log10(mt[..., [0, 2]]).std())},
                   "phase_deg": {"mean": float(mt[..., [1, 3]].mean()), "std": float(mt[..., [1, 3]].std())}}
    meta = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cfg.items()}
    meta.update({"train_chunks": train_ids, "val_chunks": val_ids, "stats": stats,
                 "modalities": {
                     "velocity": "(1, nz, ny) m/s", "density": "(1, nz, ny) kg/m^3",
                     "conductivity": "(1, nz, ny) S/m", "resistivity": "(1, nz, ny) ohm m",
                     "porosity": "(1, nz, ny)", "layer_id": "(1, nz, ny) int8",
                     "seismic": "(ns, nt, ny) pressure at 10 m depth receivers, one per column",
                     "mt": "(n_freq, ny, 4) [rho_a TE ohm m, phase TE deg, rho_a TM ohm m, phase TM deg] at surface stations",
                     "dc": "(n_dc_src, ny) pole-pole apparent resistivity ohm m; entry at the source column copied from its neighbour"}})
    with open(Path(out) / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def run_generation(tasks, fn, writer, jobs, label):
    t0 = time.time()
    n = len(tasks)
    if jobs > 1:
        with Pool(jobs) as pool:
            for i, s in enumerate(pool.imap(fn, tasks, chunksize=1)):
                writer.add(s)
                if (i + 1) % max(1, n // 10) == 0 or i + 1 == n:
                    print("  %s %d/%d  %.0fs" % (label, i + 1, n, time.time() - t0), flush=True)
    else:
        for i, task in enumerate(tasks):
            writer.add(fn(task))
            if (i + 1) % max(1, n // 10) == 0 or i + 1 == n:
                print("  %s %d/%d  %.0fs" % (label, i + 1, n, time.time() - t0), flush=True)
    writer.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="multiphysics_data/LayeredAB")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--chunk", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--nbc", type=int, default=120, help="seismic sponge width in cells")
    p.add_argument("--n-freq", type=int, default=8)
    p.add_argument("--f-min", type=float, default=5.0)
    p.add_argument("--f-max", type=float, default=1000.0)
    p.add_argument("--n-dc-src", type=int, default=10)
    p.add_argument("--mt-refine", type=int, default=2)
    p.add_argument("--from-openfwi", nargs=2, metavar=("ROOT", "DATASET"),
                   help="derive conductivity from real OpenFWI models; copies their gathers")
    p.add_argument("--train-chunks", type=int, default=4, help="with --from-openfwi")
    p.add_argument("--val-chunks", type=int, default=1, help="with --from-openfwi")
    p.add_argument("--test", action="store_true", help="6 samples, tiny sponge, 1 job")
    a = p.parse_args()

    if a.test:
        a.n_train, a.n_val, a.chunk, a.nbc, a.jobs = 4, 2, 4, 40, 1
        a.out = a.out if a.out != "multiphysics_data/LayeredAB" else "multiphysics_data/_test"

    nz = ny = 70
    cfg = {"nz": nz, "ny": ny, "dx": 10.0, "dt": 1e-3, "nt": 1000, "f0": 15.0, "ns": 5,
           "nbc": a.nbc, "chunk": a.chunk, "seed": a.seed,
           "frequencies": default_frequencies(a.n_freq, a.f_min, a.f_max),
           "dc_src_cols": np.round(np.linspace(0, ny - 1, a.n_dc_src)).astype(int),
           "dc_n_k": 20, "mt_refine": a.mt_refine,
           "source": "synthetic" if a.from_openfwi is None else "openfwi:%s" % a.from_openfwi[1]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = ChunkWriter(out, a.chunk)

    if a.from_openfwi is None:
        print("synthetic geology: %d train + %d val, chunk %d, %d jobs -> %s" % (a.n_train, a.n_val, a.chunk, a.jobs, out))
        seeds = a.seed * 1_000_003 + np.arange(a.n_train + a.n_val)
        run_generation([(int(s), cfg) for s in seeds[:a.n_train]], make_sample, writer, a.jobs, "train")
        train_ids = list(writer.ids)
        run_generation([(int(s), cfg) for s in seeds[a.n_train:]], make_sample, writer, a.jobs, "val")
        val_ids = [k for k in writer.ids if k not in train_ids]
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "openfwi"))
        from openfwi_data import split_paths                                  # noqa: E402
        root, dataset = a.from_openfwi
        ids = {}
        for split, n_chunks in (("train", a.train_chunks), ("val", a.val_chunks)):
            data_paths, model_paths = split_paths(root, dataset, split, n_chunks)
            for dp, mp in zip(data_paths, model_paths):
                V = np.load(mp, mmap_mode="r")
                D = np.load(dp, mmap_mode="r")
                k = int(Path(mp).stem.replace("model", ""))
                print("  %s chunk %d: %d samples" % (split, k, len(V)), flush=True)
                w = ChunkWriter(out, len(V), first_id=k)
                base = a.seed * 1_000_003 + k * 100_000
                tasks = [(base + i, cfg, np.asarray(V[i, 0]), np.asarray(D[i])) for i in range(len(V))]
                run_generation(tasks, openfwi_sample, w, a.jobs, "%s chunk %d" % (split, k))
                ids.setdefault(split, []).extend(w.ids)
        train_ids, val_ids = ids.get("train", []), ids.get("val", [])

    meta = build_meta(cfg, train_ids, val_ids, out)
    print("wrote", out, "train chunks", train_ids, "val chunks", val_ids)
    for key, st in meta["stats"].items():
        print("  %-13s %s" % (key, json.dumps(st)))


if __name__ == "__main__":
    main()
