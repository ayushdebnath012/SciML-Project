"""Build a trainable cache from the SubsurfaceGen field-scale dataset.

SubsurfaceGen (arXiv:2605.30541) ships one ~99 MB shot-gather cube per sample:
`shot_gather_cube` is (64 sources, 572 time, 1000 receivers), paired with a
(619, 1000) velocity slice at 10 m spacing. Holding an OpenFWI-sized benchmark
at that rate would be 250 GB, so this downloads each pair, extracts the subset
the benchmark actually trains on, writes it into a shard, and deletes the cube.
Peak disk is one shard of raw files, not the whole dataset.

What is kept, and why:

* **5 of 64 sources**, evenly spaced. Matches OpenFWI's shot count so the two
  benchmarks are the same shape of problem, and cuts the target 13x.
* **All 572 time samples and all 1000 receivers.** Measured on a real cube,
  halving the time axis costs 66 % relative L2 and decimating receivers 2x
  costs 18 % -- an 8 s record at 3-25 Hz is critically sampled, unlike
  OpenFWI's 1 s record at 15 Hz where both were nearly free.
* **Velocity downsampled 2x** to (309, 500) by area-averaging. This is the
  *input*, and a velocity model is piecewise smooth where a wavefield is not;
  it makes the graph operator's node count affordable. The decoder head
  upsamples back to 1000 receivers, so no output resolution is lost.

    python wave/subsurfacegen/fetch_ssgen.py --root ~/ssgen_data \
        --train 600 --val 100 --ood 80 --jobs 8
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = "https://huggingface.co/datasets/subsurfacegen/field-scale-dataset"
MIRROR = REPO + "/resolve/main"
BAND = "3-25Hz"
PROP = "8s"
N_SOURCES_AVAILABLE = 64


def index_frame(root, split):
    """Load (downloading if needed) one split's parquet index."""
    import pandas as pd
    cache = Path(root) / "index"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (split + ".parquet")
    if not path.exists():
        url = MIRROR + "/data/" + split + ".parquet"
        print("  fetching index", url, flush=True)
        urllib.request.urlretrieve(url, path)
    return pd.read_parquet(path)


def select(frame, n, seed, stratify=True):
    """Pick n slice_ids that have a 3-25Hz gather, spread across geologies.

    The dataset's five training geologies are unevenly sized; sampling
    proportionally would let f3 and seam dominate a small subset, which is the
    opposite of what the dataset is for.
    """
    gathers = frame[(frame.data_type == "gather") &
                    (frame.frequency_band == BAND)]
    if n is None or n >= len(gathers):
        chosen = gathers
    elif not stratify:
        chosen = gathers.sample(n=n, random_state=seed)
    else:
        groups = sorted(gathers.model_type.unique())
        per = max(1, n // len(groups))
        parts = []
        for g in groups:
            sub = gathers[gathers.model_type == g]
            parts.append(sub.sample(n=min(per, len(sub)), random_state=seed))
        import pandas as pd
        chosen = pd.concat(parts)
        if len(chosen) < n:                     # top up from what is left
            rest = gathers.drop(chosen.index)
            chosen = pd.concat([chosen, rest.sample(n=min(n - len(chosen), len(rest)),
                                                    random_state=seed)])
        chosen = chosen.iloc[:n]
    return list(chosen.slice_id), list(chosen.model_type)


def download(url, dest, retries=4):
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ssgen-fetch/1"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as fh:
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries:
                raise
            print("    retry %d/%d %s: %s" % (attempt, retries, dest.name, exc),
                  flush=True)
            time.sleep(3 * attempt)


def fetch_pair(slice_id, tmp):
    slice_path = tmp / ("slice_%s.h5" % slice_id)
    cube_path = tmp / ("cube_%s.h5" % slice_id)
    if not slice_path.exists():
        download("%s/slices/slice_%s.h5" % (MIRROR, slice_id), slice_path)
    if not cube_path.exists():
        download("%s/shot_gathers/%s/%s/shot_gather_cube_%s.h5"
                 % (MIRROR, PROP, BAND, slice_id), cube_path)
    return slice_id, slice_path, cube_path


def downsample2(velocity):
    """Area-average a (619, 1000) slice to (309, 500).

    Trims the last depth row so the depth axis halves exactly; averaging rather
    than striding is what keeps a 20 m grid from aliasing the sharp interfaces
    that salt bodies and faults put in these models.
    """
    nz = (velocity.shape[0] // 2) * 2
    v = velocity[:nz].astype(np.float32)
    v = v.reshape(nz // 2, 2, v.shape[1] // 2, 2).mean(axis=(1, 3))
    return v


def extract(slice_path, cube_path, source_idx):
    import h5py
    import hdf5plugin                                   # noqa: F401  (registers filters)
    with h5py.File(slice_path, "r") as f:
        velocity = f["velocity"][:]
    with h5py.File(cube_path, "r") as f:
        cube = f["shot_gather_cube"]
        gathers = np.stack([cube[i] for i in source_idx])
    return downsample2(velocity)[None], gathers.astype(np.float32)


def build_split(slice_ids, kinds, root, prefix_start, shard_size, jobs, source_idx,
                tmp, stats):
    """Write shards of `shard_size` samples, deleting raw files as it goes."""
    data_dir = Path(root) / "data"
    model_dir = Path(root) / "model"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    shard_id = prefix_start
    written_kinds = []
    for start in range(0, len(slice_ids), shard_size):
        block = slice_ids[start:start + shard_size]
        block_kinds = kinds[start:start + shard_size]
        dpath = data_dir / ("data%d.npy" % shard_id)
        mpath = model_dir / ("model%d.npy" % shard_id)
        if dpath.exists() and mpath.exists():
            print("  shard %d cached (%d samples)" % (shard_id, len(block)), flush=True)
            arr = np.load(dpath, mmap_mode="r")
            stats["gather_min"] = min(stats["gather_min"], float(arr[:8].min()))
            stats["gather_max"] = max(stats["gather_max"], float(arr[:8].max()))
            written_kinds.extend(block_kinds)
            shard_id += 1
            continue

        t0 = time.perf_counter()
        paths = {}
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(fetch_pair, sid, tmp) for sid in block]
            for fut in as_completed(futures):
                sid, sp, cp = fut.result()
                paths[sid] = (sp, cp)
        dl = time.perf_counter() - t0

        vel, gat = [], []
        for sid in block:
            sp, cp = paths[sid]
            v, g = extract(sp, cp, source_idx)
            vel.append(v)
            gat.append(g)
            sp.unlink(missing_ok=True)
            cp.unlink(missing_ok=True)
        V = np.stack(vel).astype(np.float32)
        G = np.stack(gat).astype(np.float32)
        np.save(mpath, V)
        np.save(dpath, G)
        stats["gather_min"] = min(stats["gather_min"], float(G.min()))
        stats["gather_max"] = max(stats["gather_max"], float(G.max()))
        stats["velocity_min"] = min(stats["velocity_min"], float(V.min()))
        stats["velocity_max"] = max(stats["velocity_max"], float(V.max()))
        written_kinds.extend(block_kinds)
        print("  shard %d: %d samples  velocity%s gathers%s  dl %.0fs total %.0fs"
              % (shard_id, len(block), V.shape[1:], G.shape[1:], dl,
                 time.perf_counter() - t0), flush=True)
        shard_id += 1
    return shard_id, written_kinds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="ssgen_data")
    p.add_argument("--train", type=int, default=600)
    p.add_argument("--val", type=int, default=100)
    p.add_argument("--ood", type=int, default=80,
                   help="Penobscot slices, held out of training entirely by the "
                        "dataset's own design; 0 to skip")
    p.add_argument("--sources", type=int, default=5)
    p.add_argument("--shard-size", type=int, default=50)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    root = Path(a.root).expanduser()
    tmp = root / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    source_idx = np.linspace(0, N_SOURCES_AVAILABLE - 1, a.sources).astype(int)
    print("sources kept: %s of %d" % (list(source_idx), N_SOURCES_AVAILABLE))

    stats = {"gather_min": np.inf, "gather_max": -np.inf,
             "velocity_min": np.inf, "velocity_max": -np.inf}
    manifest = {}
    next_id = 1
    for split, n, key in (("train", a.train, "train"),
                          ("test_in_dist", a.val, "val"),
                          ("test_out_dist", a.ood, "ood")):
        if not n:
            continue
        frame = index_frame(root, split)
        ids, kinds = select(frame, n, a.seed, stratify=(key == "train"))
        print("=== %s: %d samples, geologies %s ==="
              % (key, len(ids), sorted(set(kinds))), flush=True)
        start_id = next_id
        next_id, written = build_split(ids, kinds, root, next_id, a.shard_size,
                                       a.jobs, source_idx, tmp, stats)
        manifest[key] = {"first_chunk": start_id, "n_chunks": next_id - start_id,
                         "samples": len(ids), "kinds": written}

    stats = {k: float(v) for k, v in stats.items()}
    meta = {"source_indices": [int(i) for i in source_idx],
            "band": BAND, "propagation_time_s": 8.0,
            "manifest": manifest, "stats": stats,
            "note": "velocity area-downsampled 2x to (309,500); gathers kept at "
                    "native (572,1000) because the 8 s / 3-25 Hz record is "
                    "critically sampled"}
    (root / "ssgen_meta.json").write_text(json.dumps(meta, indent=2))
    print("\nstats:", json.dumps(stats, indent=2))
    print("wrote", root / "ssgen_meta.json")


if __name__ == "__main__":
    main()
