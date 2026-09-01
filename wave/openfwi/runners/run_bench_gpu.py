"""FNO / PFNO forward-operator benchmark on a single rented notebook GPU.

Two benchmark families, one runner:

* **OpenFWI B families** -- FlatVel_B, CurveVel_B.  velocity (1, 70, 70) ->
  gathers (5, 1000, 70).  The harder counterpart of the FlatVel_A / CurveVel_A
  numbers already in results/openfwi/: same geometry classes, wider velocity
  contrast, more layers.
* **SubsurfaceGen** (arXiv:2605.30541) -- field-scale, velocity (309, 500) ->
  gathers (5, 572, 1000), plus the Penobscot out-of-distribution split the
  dataset was built to produce.

The existing numbers were produced on a two-GPU H100 box.  Here the binding
constraint is a session wall-clock limit rather than memory, so this is built
to be interrupted and resumed: work is done one target at a time, and a target
whose summary already exists is skipped unless --force.

    python wave/openfwi/runners/run_bench_gpu.py --plan          # probe only
    python wave/openfwi/runners/run_bench_gpu.py                 # B families
    python wave/openfwi/runners/run_bench_gpu.py --targets SubsurfaceGen
    python wave/openfwi/runners/run_bench_gpu.py --preset smoke   # end-to-end

It always probes the real models on the real GPU before committing, and checks
the projected wall-clock and memory against the session.  The A-family run
taught that this pipeline's cost is not where you would assume -- it was I/O
bound, not compute bound, until the split was cached -- so the projection is
measured on the host rather than scaled from the H100 timings.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
OPENFWI = HERE.parent.parent                 # wave/openfwi
SSGEN = OPENFWI.parent / "subsurfacegen"     # wave/subsurfacegen

# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
# Per-architecture sizing is not invented here: the OpenFWI values are the
# defaults the A-family table was produced with, and the SubsurfaceGen values
# are the measured ones from wave/subsurfacegen/runners/run_ssgen.sh. Changing
# either breaks comparability with the runs already in results/.
OPENFWI_EXTRA = []
SSGEN_EXTRA = [
    "--t-latent", "286", "--fno-width", "24", "--fno-modes-t", "32",
    # 220 branches, not OpenFWI's 64: the band a per-frequency model must cover
    # is f_max * T, and an 8 s record at 3-25 Hz is 4x a 1 s record at 15 Hz.
    "--pfno-freqs", "220", "--pfno-width", "4", "--pfno-modes", "8",
    "--pfno-layers", "2",
]

# Geometry the probe builds models against, before any data exists locally.
# OpenFWI's comes from its published dataset_config; SubsurfaceGen's is what
# fetch_ssgen.py deterministically produces (5 of 64 sources, velocity
# area-averaged 2x), as recorded in results/ssgen/ssgen/openfwi_summary.json.
SSGEN_CFG = {"ns": 5, "nt": 572, "ng": 1000, "nz": 309, "nx": 500,
             "n_grid": 309, "dt": 0.013986013986013986, "dx": 10.0,
             "label_min": 1500.0, "label_max": 4733.806640625}

TARGETS = {
    "FlatVel_B":     {"kind": "openfwi"},
    "CurveVel_B":    {"kind": "openfwi"},
    "SubsurfaceGen": {"kind": "ssgen"},
}
DEFAULT_TARGETS = ["FlatVel_B", "CurveVel_B"]
MODELS = "FNO,PFNO"

# OpenFWI: the published A-family configuration, matched so a B number is
# comparable to an A number. SubsurfaceGen: the configuration of the run in
# results/ssgen/, for the same reason.
PRESETS = {
    "bench": {"openfwi": dict(train_chunks=4, val_chunks=1, epochs=100,
                              batch_size=8, lr=2e-3),
              "ssgen": dict(train=600, val=100, ood=80, epochs=80,
                            batch_size=2, lr=2e-3)},
    # fits a 12 h session when the probe says `bench` will not
    "short": {"openfwi": dict(train_chunks=4, val_chunks=1, epochs=40,
                              batch_size=8, lr=2e-3),
              "ssgen": dict(train=600, val=100, ood=80, epochs=30,
                            batch_size=2, lr=2e-3)},
    # proves fetch -> train -> export end to end in minutes, CPU included
    "smoke": {"openfwi": dict(train_chunks=1, val_chunks=1, epochs=2,
                              batch_size=4, lr=2e-3, max_train=32, max_val=16),
              "ssgen": dict(train=8, val=4, ood=4, epochs=2, batch_size=1,
                            lr=2e-3)},
}

GB_PER_OPENFWI_CHUNK = 0.7098
GB_PER_OPENFWI_SAMPLE = 1.4e-3       # cached (velocity, gather) pair
GB_PER_SSGEN_SAMPLE = 11.4e-3        # 5 x 572 x 1000 float32 + velocity
GB_SSGEN_DOWNLOAD_PER_SAMPLE = 0.099  # one shot-gather cube, fetched then deleted


def detect_env():
    """Where data and results go on whichever host this landed on.

    Kaggle counts /kaggle/working against a 20 GB output quota but not
    /kaggle/temp, and the OpenFWI chunks alone are ~7 GB -- so data goes to
    temp and only results to working, or the quota is spent on .npy files that
    are re-downloadable in minutes.

    Detection is by the environment variable each host sets, not by the
    directory alone: a Windows checkout resolves "/kaggle/working" against the
    current drive, so a stray C:\\kaggle\\working is enough to make a local box
    claim to be Kaggle and write its results somewhere surprising.
    """
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle", Path("/kaggle/temp"), Path("/kaggle/working/bench_results")
    studio = Path("/teamspace/studios/this_studio")
    if os.name == "posix" and studio.is_dir():
        return "lightning", studio, studio / "bench_results"
    return "local", Path.home(), Path.home() / "bench_results"


def gpu_report():
    try:
        import torch
    except ImportError:
        raise SystemExit("torch is not installed in this environment")
    if not torch.cuda.is_available():
        print("!! no CUDA device -- --preset bench would take days here. "
              "Use --preset smoke to validate the path on CPU.", flush=True)
        return None, 0.0
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print("gpu: %s (%.0f GB), torch %s" % (name, total, torch.__version__), flush=True)
    return name, total


def check_deps(targets):
    """Fail before the first download, not 40 minutes into it.

    SubsurfaceGen ships HDF5 cubes behind a parquet index, so its fetcher needs
    three packages the OpenFWI path does not. A stock Kaggle image has pandas
    and pyarrow but neither h5py filter package.
    """
    if not any(TARGETS[t]["kind"] == "ssgen" for t in targets):
        return
    missing = []
    for mod in ("pandas", "pyarrow", "h5py", "hdf5plugin"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(
            "SubsurfaceGen needs %s.\n    pip install %s"
            % (", ".join(missing), " ".join(missing)))


def target_cfg(target):
    """Grid and model geometry for a target, without needing its data."""
    sys.path.insert(0, str(OPENFWI))
    if TARGETS[target]["kind"] == "ssgen":
        return dict(SSGEN_CFG), SSGEN_CFG["nt"]
    from openfwi_data import DATASET_CONFIG, dataset_key
    cfg = dict(DATASET_CONFIG[dataset_key(target)])
    return cfg, cfg["nt"]


def probe(target, models, batch_size, n_train, extra, steps=3):
    """Time one optimizer step per model at this target's real shapes.

    Synthetic tensors, so this is compute cost with the data path removed --
    which is what --cache gpu makes the real per-epoch cost converge to. The
    peak-memory figure is what decides whether the split can be cached on the
    GPU beside the activations.
    """
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(OPENFWI))
    from openfwi_models import build_model, count_parameters_real
    from train_openfwi import parse_args as train_args

    cfg, nt = target_cfg(target)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nz, nx = cfg.get("nz", cfg["n_grid"]), cfg.get("nx", cfg["n_grid"])
    out = {}
    print("\nprobe %s (%d steps, batch %d, synthetic tensors):"
          % (target, steps, batch_size), flush=True)
    for name in models:
        a = train_args(list(extra))
        model = build_model(name, a, cfg, nt).to(dev)
        x = torch.randn(batch_size, 1, nz, nx, device=dev)
        y = torch.randn(batch_size, cfg["ns"], nt, cfg["ng"], device=dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        F.mse_loss(model(x), y).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            F.mse_loss(model(x), y).backward()
            opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / steps
        peak = (torch.cuda.max_memory_allocated() / 1e9) if dev.type == "cuda" else 0.0
        out[name] = {"s_per_step": dt, "s_per_epoch": dt * n_train / batch_size,
                     "peak_gb": peak, "params_real": count_parameters_real(model)}
        print("  %-5s %11s params  %.3f s/step  ~%5.0f s/epoch  peak %.1f GB"
              % (name, format(count_parameters_real(model), ","), dt,
                 out[name]["s_per_epoch"], peak), flush=True)
        del model, opt, x, y
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def plan_target(target, preset, models, data_root):
    """Probe one target and report its projected cost. Returns (hours, gb)."""
    kind = TARGETS[target]["kind"]
    cf = preset[kind]
    if kind == "openfwi":
        n_train = cf.get("max_train") or cf["train_chunks"] * 500
        n_val = cf.get("max_val") or cf["val_chunks"] * 500
        cache_gb = (n_train + n_val) * GB_PER_OPENFWI_SAMPLE
        dl_gb = (cf["train_chunks"] + cf["val_chunks"]) * GB_PER_OPENFWI_CHUNK
        extra = OPENFWI_EXTRA
    else:
        n_train, n_val = cf["train"], cf["val"] + cf["ood"]
        cache_gb = (n_train + n_val) * GB_PER_SSGEN_SAMPLE
        dl_gb = (n_train + n_val) * GB_SSGEN_DOWNLOAD_PER_SAMPLE
        extra = SSGEN_EXTRA
    timings = probe(target, models, cf["batch_size"], n_train, extra)
    hours = sum(t["s_per_epoch"] for t in timings.values()) * cf["epochs"] / 3600
    peak = max(t["peak_gb"] for t in timings.values())
    print("  %d train / %d val, %d epochs -> %.1f h"
          % (n_train, n_val, cf["epochs"], hours))
    print("  peak activation %.1f GB + %.1f GB split cache = %.1f GB GPU"
          % (peak, cache_gb, peak + cache_gb))
    if kind == "ssgen":
        print("  download ~%.0f GB of cubes (fetched, subset, deleted; peak disk "
              "is one shard) -> %.1f GB cache" % (dl_gb, cache_gb))
    else:
        print("  download %.1f GB -> %s" % (dl_gb, data_root))
    return hours, peak + cache_gb


def run(cmd, log=None):
    cmd = [str(c) for c in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    if log is None:
        return subprocess.call(cmd)
    with open(log, "ab") as fh:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            fh.write(line)
        return p.wait()


def fetch_cmd(target, preset, data_root, jobs):
    cf = preset[TARGETS[target]["kind"]]
    if TARGETS[target]["kind"] == "openfwi":
        return [sys.executable, OPENFWI / "fetch_openfwi.py",
                "--datasets", target,
                "--train-chunks", cf["train_chunks"], "--val-chunks", cf["val_chunks"],
                "--root", data_root / "openfwi_data", "--jobs", jobs]
    return [sys.executable, SSGEN / "fetch_ssgen.py",
            "--root", data_root / "ssgen_data",
            "--train", cf["train"], "--val", cf["val"], "--ood", cf["ood"],
            "--jobs", jobs]


def train_cmd(target, preset, data_root, run_dir, models, cache, extra):
    cf = preset[TARGETS[target]["kind"]]
    common = ["--models", ",".join(models), "--epochs", cf["epochs"],
              "--batch-size", cf["batch_size"], "--lr", cf["lr"],
              "--cache", cache, "--outdir", run_dir]
    # Split-model work units may run for hours. Checkpoint every epoch and
    # transparently resume if a previous invocation stopped before producing
    # its summary. The trainer validates the complete configuration before it
    # accepts the checkpoint, so stale state cannot be mixed into a new run.
    if len(models) == 1:
        common += ["--checkpoint-every", 1]
        checkpoint = Path(run_dir) / ("%s_train_checkpoint.pt" % models[0])
        if checkpoint.exists():
            common += ["--resume"]
    if TARGETS[target]["kind"] == "openfwi":
        cmd = [sys.executable, "-u", OPENFWI / "train_openfwi.py",
               "--root", data_root / "openfwi_data", "--dataset", target,
               "--train-chunks", cf["train_chunks"],
               "--val-chunks", cf["val_chunks"]] + common + OPENFWI_EXTRA
        for key in ("max_train", "max_val"):
            if cf.get(key) is not None:
                cmd += ["--" + key.replace("_", "-"), cf[key]]
        return cmd + extra
    # SubsurfaceGen: the cache carries its own layout and normalization, and
    # zscore is required -- min/max leaves this data in a fraction of a
    # percent of [-1, 1] because its amplitude distribution is heavy-tailed.
    return ([sys.executable, "-u", OPENFWI / "train_openfwi.py",
             "--meta", "--root", data_root / "ssgen_data",
             "--dataset", "SubsurfaceGen", "--norm", "zscore",
             "--train-chunks", 0, "--val-chunks", 0,
             "--ood-chunks", 0 if cf["ood"] else -1] + common + SSGEN_EXTRA + extra)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS,
                   choices=sorted(TARGETS), metavar="TARGET",
                   help="any of: " + ", ".join(sorted(TARGETS)))
    p.add_argument("--preset", choices=sorted(PRESETS), default="bench")
    p.add_argument("--models", default=MODELS)
    p.add_argument("--data-root", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--cache", choices=("gpu", "ram", "none"), default=None,
                   help="default: gpu when the probe says the split plus peak "
                        "activation fits in 80%% of VRAM, ram otherwise")
    p.add_argument("--jobs", type=int, default=6,
                   help="concurrent downloads; both mirrors rate-limit per "
                        "connection, so this is what sets throughput")
    p.add_argument("--session-hours", type=float, default=12.0,
                   help="wall-clock limit of this session; the projection is "
                        "checked against it")
    p.add_argument("--plan", action="store_true",
                   help="probe and estimate every target, then stop before "
                        "downloading anything")
    p.add_argument("--split-models", action="store_true",
                   help="train one model per invocation, each into its own "
                        "result directory with automatic epoch-level resume, "
                        "so an interrupted expensive model continues in place. "
                        "Recommended for SubsurfaceGen, where PFNO is ~14x "
                        "FNO's per-epoch cost")
    p.add_argument("--skip-fetch", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="retrain targets that already have a summary")
    p.add_argument("--no-zip", dest="do_zip", action="store_false", default=True)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="everything after this is forwarded to train_openfwi.py")
    return p.parse_args(argv)


def apply_overrides(preset, a):
    preset = {k: dict(v) for k, v in preset.items()}
    for kind in preset:
        if a.epochs is not None:
            preset[kind]["epochs"] = a.epochs
        if a.batch_size is not None:
            preset[kind]["batch_size"] = a.batch_size
    return preset


def work_units(target, models, outdir, split_models):
    """The (run_dir, models) invocations a target is broken into.

    One invocation per target trains every model in a single process and writes
    a single summary at the end -- fine when the whole thing fits a session.
    With --split-models each model is its own invocation and its own summary,
    so a session that dies during PFNO still leaves a finished FNO behind. That
    is the difference between losing an hour and losing a day on SubsurfaceGen,
    where PFNO is ~14x FNO's per-epoch cost.

    Caveat: the trainer seeds weight init as `init_seed + 100 * i`, where i is
    the model's position in the list it was handed, so every split invocation
    gets i = 0. A split run is therefore a different random initialization from
    a combined one, not a different configuration. Both are valid samples of
    the same experiment -- just do not mix the two in one table without saying
    so.
    """
    if not split_models:
        return [(outdir / target.lower(), list(models))]
    return [(outdir / ("%s__%s" % (target.lower(), m.lower())), [m])
            for m in models]


def summary_paths(outdir, target):
    """Every summary belonging to a target, whole-target or per-model."""
    whole = outdir / target.lower() / "openfwi_summary.json"
    found = [whole] if whole.exists() else []
    found += sorted(p / "openfwi_summary.json"
                    for p in outdir.glob(target.lower() + "__*")
                    if (p / "openfwi_summary.json").exists())
    return found


def summarise(outdir, targets):
    done = []
    print("\n" + "=" * 68)
    for t in targets:
        paths = summary_paths(outdir, t)
        if not paths:
            continue
        done.append(t)
        head = json.loads(paths[0].read_text())
        print("%-14s %d train / %d val, %d epochs"
              % (head["dataset"], head["split"]["train"], head["split"]["val"],
                 head["args"]["epochs"]))
        for path in paths:
            for r in json.loads(path.read_text())["results"]:
                line = ("   %-5s rel L2 %7.3f%%  median %7.3f%%  RMSE %.4g  %s params"
                        % (r["model"], r["rel_l2_pct"], r["rel_l2_pct_median"],
                           r["rmse"], format(r["parameters_real"], ",")))
                if "ood" in r:
                    line += "   OOD %7.3f%%" % r["ood"]["rel_l2_pct"]
                print(line)
    return done


def main(argv=None):
    a = parse_args(argv)
    preset = apply_overrides(PRESETS[a.preset], a)
    models = [m.strip() for m in a.models.split(",") if m.strip()]

    env, data_root, outdir = detect_env()
    data_root = Path(a.data_root).expanduser() if a.data_root else data_root
    outdir = Path(a.outdir).expanduser() if a.outdir else outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(exist_ok=True)
    print("host: %s\n  data -> %s\n  out  -> %s" % (env, data_root, outdir))
    print("preset %s   targets: %s   models: %s"
          % (a.preset, ", ".join(a.targets), ",".join(models)))

    check_deps(a.targets)
    _, total_vram = gpu_report()
    total_h, need_gb = 0.0, 0.0
    for t in a.targets:
        h, gb = plan_target(t, preset, models, data_root)
        total_h += h
        need_gb = max(need_gb, gb)
    print("\ntotal projected training: %.1f h across %d target(s)"
          % (total_h, len(a.targets)))
    if total_h > a.session_hours:
        print("!! %.1f h exceeds the %.1f h session limit. Add --split-models "
              "so each model checkpoints separately, run one target per "
              "session (--targets FlatVel_B), drop to --preset short, or use a "
              "faster GPU. Finished work is skipped on the next run, so "
              "resuming is just re-issuing the same command."
              % (total_h, a.session_hours))

    if a.plan:
        print("\n--plan: stopping before download.")
        return 0

    cache = a.cache
    if cache is None:
        # Headroom, because a notebook GPU is not always exclusively ours and
        # the cache is a speed convenience -- no reported number depends on it.
        cache = "gpu" if total_vram and need_gb < 0.8 * total_vram else "ram"
        print("  cache=%s (need %.1f GB of %.0f GB)" % (cache, need_gb, total_vram))

    failed = []
    for target in a.targets:
        units = work_units(target, models, outdir, a.split_models)
        pending = [(d, ms) for d, ms in units
                   if a.force or not (d / "openfwi_summary.json").exists()]
        if not pending:
            print("\n== %s already has every summary -- skipping (--force to redo)"
                  % target)
            continue
        # One fetch per target however many invocations it is split into: the
        # fetchers are cache-aware and would just re-verify, but re-walking a
        # 77 GB SubsurfaceGen manifest per model is not free.
        if not a.skip_fetch:
            rc = run(fetch_cmd(target, preset, data_root, a.jobs))
            if rc != 0:
                print("fetch failed for %s (rc=%d)" % (target, rc))
                failed.append(target)
                continue
        for run_dir, unit_models in pending:
            t0 = time.perf_counter()
            rc = run(train_cmd(target, preset, data_root, run_dir, unit_models,
                               cache, a.extra),
                     log=outdir / "logs" / (run_dir.name + ".log"))
            print("== %s [%s] finished rc=%d in %.2f h"
                  % (target, ",".join(unit_models), rc,
                     (time.perf_counter() - t0) / 3600))
            if rc != 0:
                failed.append("%s/%s" % (target, ",".join(unit_models)))

    done = summarise(outdir, a.targets)
    if failed:
        print("FAILED: " + ", ".join(failed))
    if a.do_zip and done:
        archive = shutil.make_archive(str(outdir.parent / "bench_results"),
                                      "zip", root_dir=outdir)
        print("\nzipped -> %s (%.1f MB)"
              % (archive, Path(archive).stat().st_size / 1e6))
    print("done. copy the result directory back and render with "
          "report_openfwi.py / render_openfwi.py on a CPU box.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
