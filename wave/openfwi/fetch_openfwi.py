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


def npy_file_info(path):
    """Read shape, dtype and payload boundary without loading the array."""
    with open(path, "rb") as fh:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(fh)
        else:
            shape, _, dtype = np.lib.format.read_array_header_2_0(fh)
        header_bytes = fh.tell()
    expected_bytes = header_bytes + int(np.prod(shape)) * dtype.itemsize
    return shape, dtype, expected_bytes


def npy_integrity(path, expected_shape):
    """Validate header declarations *and* the complete on-disk payload.

    A truncated download can retain a perfectly valid header, so checking only
    shape and dtype accepts a file that later fails in ``numpy.memmap``.  The
    exact .npy size is header bytes plus shape product times dtype size.
    """
    try:
        shape, dtype, expected_bytes = npy_file_info(path)
    except (EOFError, OSError, ValueError) as exc:
        return False, "unreadable header: %s" % exc, None
    if tuple(shape) != tuple(expected_shape):
        return False, "shape %s, expected %s" % (shape, expected_shape), expected_bytes
    if dtype != np.dtype("float32"):
        return False, "dtype %s, expected float32" % dtype, expected_bytes
    actual_bytes = Path(path).stat().st_size
    if actual_bytes != expected_bytes:
        label = "truncated" if actual_bytes < expected_bytes else "oversized"
        return (False, "%s payload: %d of %d bytes"
                % (label, actual_bytes, expected_bytes), expected_bytes)
    return True, "complete", expected_bytes


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
    part = dest.with_suffix(dest.suffix + ".part")
    url = f"{MIRROR}/{dataset}/{kind}/{name}"
    if force:
        dest.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
    elif dest.exists():
        ok, reason, expected_bytes = npy_integrity(dest, expected)
        if ok:
            return label, "cached", 0.0
        print(f"  {label} is invalid ({reason}) -- resuming/refetching", flush=True)
        # A valid header plus a short payload is useful: turn it back into the
        # downloader's .part file so the next request resumes at that byte.
        if (expected_bytes is not None and
                dest.stat().st_size < expected_bytes):
            if not part.exists() or dest.stat().st_size > part.stat().st_size:
                part.unlink(missing_ok=True)
                dest.replace(part)
            else:
                dest.unlink()
        else:
            dest.unlink()
            part.unlink(missing_ok=True)

    total_secs = 0.0
    for integrity_attempt in range(1, 5):
        secs = download(url, dest, label=label)
        total_secs += secs
        ok, reason, expected_bytes = npy_integrity(dest, expected)
        if ok:
            mb = dest.stat().st_size / 1e6
            return (label,
                    f"{mb:.0f} MB in {total_secs:.0f}s "
                    f"({mb/max(total_secs,1e-9):.1f} MB/s)", total_secs)
        print(f"    {label} integrity attempt {integrity_attempt}/4: {reason}",
              flush=True)
        if integrity_attempt == 4:
            raise RuntimeError(f"{label}: {reason} after 4 integrity attempts")
        if (expected_bytes is not None and
                dest.stat().st_size < expected_bytes):
            part.unlink(missing_ok=True)
            dest.replace(part)
        else:
            dest.unlink(missing_ok=True)
            part.unlink(missing_ok=True)


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
    print(f"\nverified all {len(tasks)} files (shape, dtype, payload size) "
          f"against dataset_config.json "
          f"in {elapsed/60:.1f} min")
    for k, v in manifest.items():
        print(f"  {k}: chunks {v}")


if __name__ == "__main__":
    main()
