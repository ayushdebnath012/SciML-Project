"""Download OpenFWI chunks from the Hugging Face mirror.

OpenFWI's own distribution is per-dataset Google Drive folders, which are not
scriptable from a headless box. `ashynf/OpenFWI` mirrors the same `.npy` chunks
under stable `resolve/main` URLs, so this fetches from there and then verifies
every file against the shapes the official `dataset_config.json` declares --
a mirror is only usable if it is checked, not trusted.

Chunk numbering follows OpenFWI's published split files: chunks 1..48 are
train, 49..60 are validation. `--train-chunks/--val-chunks` take a prefix of
each block, so a subset never straddles the official boundary.

    python wave/openfwi/fetch_openfwi.py --datasets FlatVel_A CurveVel_A \
        --train-chunks 4 --val-chunks 1 --root ~/openfwi_data
"""
import argparse
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openfwi_data import DATASET_CONFIG, chunk_indices, dataset_key

MIRROR = "https://huggingface.co/datasets/ashynf/OpenFWI/resolve/main"


def chunk_ids(n_train, n_val):
    """Prefixes of the official train (1..48) and val (49..60) blocks."""
    try:
        return chunk_indices("train", n_train), chunk_indices("val", n_val)
    except ValueError as exc:
        raise SystemExit(str(exc))


def npy_header_shape(path):
    """Read shape+dtype from a .npy header without loading the array."""
    with open(path, "rb") as fh:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(fh)
        else:
            shape, _, dtype = np.lib.format.read_array_header_2_0(fh)
    return shape, dtype


def download(url, dest, label="", retries=4):
    """Resumable GET. Partial files are common on a 700 MB chunk over a shared
    link, so a truncated file is resumed rather than restarted.

    The mirror rate-limits per connection, not per client -- one stream gets
    ~1.5 MB/s while six get ~19 MB/s aggregate -- so this is written to be run
    from a thread pool and keeps its progress output to one line per file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        return 0.0
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "openfwi-fetch/1"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                if have and resp.status != 206:
                    have = 0                      # server ignored Range; restart
                total = int(resp.headers.get("Content-Length", 0)) + have
                mode = "ab" if have else "wb"
                t0, last, started = time.perf_counter(), have, time.perf_counter()
                with open(part, mode) as fh:
                    while True:
                        block = resp.read(1 << 20)
                        if not block:
                            break
                        fh.write(block)
                        have += len(block)
                        now = time.perf_counter()
                        if now - t0 > 30:
                            rate = (have - last) / (now - t0) / 1e6
                            pct = 100.0 * have / total if total else 0.0
                            print(f"    {label} {pct:5.1f}%  {rate:5.1f} MB/s",
                                  flush=True)
                            t0, last = now, have
            try:
                part.replace(dest)
            except FileNotFoundError:
                # Another fetcher finished this chunk and claimed the .part
                # first. Its result is the same file, so take it rather than
                # racing it again -- but only if it really landed.
                if not dest.exists():
                    raise
            return time.perf_counter() - started
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(f"    {label} attempt {attempt}/{retries} failed: {exc}", flush=True)
            if attempt == retries:
                raise
            time.sleep(3 * attempt)


def fetch_one(dataset, kind, index, root, expected, force):
    """`kind` is 'data' (seismic) or 'model' (velocity)."""
    name = f"{kind}{index}.npy"
    label = f"{dataset}/{kind}/{name}"
    dest = Path(root) / dataset / kind / name
    url = f"{MIRROR}/{dataset}/{kind}/{name}"
    if dest.exists() and not force:
        shape, _ = npy_header_shape(dest)
        if tuple(shape) == expected:
            return label, "cached", 0.0
        print(f"  {label} has shape {shape}, expected {expected} -- refetching",
              flush=True)
    secs = download(url, dest, label=label)
    shape, dtype = npy_header_shape(dest)
    if tuple(shape) != expected:
        dest.unlink()
        raise RuntimeError(
            f"{label}: mirror returned shape {shape}, config declares {expected}. "
            "Deleted; the mirror does not match OpenFWI and must not be used.")
    if dtype != np.dtype("float32"):
        raise RuntimeError(f"{label}: dtype {dtype}, expected float32")
    mb = dest.stat().st_size / 1e6
    return label, f"{mb:.0f} MB in {secs:.0f}s ({mb/max(secs,1e-9):.1f} MB/s)", secs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["FlatVel_A", "CurveVel_A"])
    p.add_argument("--train-chunks", type=int, default=4)
    p.add_argument("--val-chunks", type=int, default=1)
    p.add_argument("--root", default="openfwi_data")
    p.add_argument("--jobs", type=int, default=6,
                   help="concurrent downloads; the mirror rate-limits per "
                        "connection, so this is what sets throughput")
    p.add_argument("--force", action="store_true", help="refetch even if cached")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    train, val = chunk_ids(a.train_chunks, a.val_chunks)
    root = Path(a.root).expanduser()
    n_files = len(a.datasets) * (len(train) + len(val)) * 2
    gb = len(a.datasets) * (len(train) + len(val)) * 0.7098
    print(f"{len(a.datasets)} datasets x {len(train)} train + {len(val)} val chunks "
          f"= {n_files} files, ~{gb:.1f} GB -> {root}")
    print(f"train chunks {train}\nval chunks   {val}")
    if a.dry_run:
        for ds in a.datasets:
            for i in train + val:
                print(f"  {MIRROR}/{ds}/data/data{i}.npy")
        return

    tasks, manifest = [], {}
    for ds in a.datasets:
        cfg = DATASET_CONFIG[dataset_key(ds)]
        n = cfg["file_size"]
        data_shape = (n, cfg["ns"], cfg["nt"], cfg["ng"])
        model_shape = (n, 1, cfg["n_grid"], cfg["n_grid"])
        print(f"=== {ds}  data{data_shape}  model{model_shape} ===")
        for split, ids in (("train", train), ("val", val)):
            manifest[f"{ds}/{split}"] = ids
            for i in ids:
                tasks.append((ds, "data", i, data_shape))
                tasks.append((ds, "model", i, model_shape))

    t0 = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = {pool.submit(fetch_one, ds, kind, i, root, shape, a.force): (ds, kind, i)
                   for ds, kind, i, shape in tasks}
        for fut in as_completed(futures):
            label, note, _ = fut.result()          # a bad chunk raises here
            done += 1
            print(f"  [{done:2d}/{len(tasks)}] {label}  {note}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nverified all {len(tasks)} headers against dataset_config.json "
          f"in {elapsed/60:.1f} min")
    for k, v in manifest.items():
        print(f"  {k}: chunks {v}")


if __name__ == "__main__":
    main()
