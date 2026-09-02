"""Benchmark FNO / PFNO / DeepONet / GNO on the OpenFWI forward map.

    velocity (1, 70, 70)  ->  shot gathers (5, 1000, 70)

Runs on the GPU box; writes JSON + a small prediction export and no plots, so
the training host needs no matplotlib.

    python wave/openfwi/train_openfwi.py --root ~/openfwi_data \
        --dataset FlatVel_A --train-chunks 4 --val-chunks 1 \
        --epochs 120 --outdir ~/openfwi_results/flatvel_a

Errors are reported on physical amplitudes, never on the [-1, 1] normalized
surrogate: the OpenFWI scaling has a non-zero offset, so a relative L2 computed
in normalized units is not the same number and would flatter a model that
predicts near the middle of the range.
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openfwi_data import (DATASET_CONFIG, OpenFWINorm, ZScoreNorm,
                          band_limit_oracle, config_from_meta, dataset_key,
                          gather_statistics, load_meta, load_split,
                          manifest_split, time_resample_oracle)
from openfwi_models import (MODEL_NAMES, build_model, count_parameters,
                            count_parameters_real)


class OpenFWIForward(Dataset):
    """Memory-mapped (velocity -> gather) pairs, normalized to [-1, 1]."""

    def __init__(self, data, model, norm, time_stride=1, limit=None):
        self.data, self.model, self.norm = data, model, norm
        self.time_stride = int(time_stride)
        self.n = len(data) if limit is None else min(limit, len(data))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        v = self.model[i].astype(np.float32)
        d = self.data[i]
        if self.time_stride > 1:
            d = d[:, ::self.time_stride, :]
        return (torch.from_numpy(self.norm.norm_velocity(v)),
                torch.from_numpy(self.norm.norm_seismic(d.astype(np.float32))))


def cache_split(dataset, device):
    """Normalize a whole split once into contiguous tensors.

    A gather is 1.4 MB, so streaming 2000 of them per epoch means 2.8 GB of
    reads and 2.8 GB of float conversion *per epoch, per model*. Measured on
    the H100 box that made FNO take 30 s/epoch against a 3.6 s compute cost --
    seven eighths of the run was the data path, and it would have been paid
    again for every architecture. Normalizing once up front removes it.

    `device='cuda'` keeps the split in GPU memory (3.5 GB for 2500 samples at
    full time resolution); otherwise it stays in host RAM and each batch is
    copied across.
    """
    n = len(dataset)
    x0, y0 = dataset[0]
    X = torch.empty((n,) + tuple(x0.shape), dtype=torch.float32)
    Y = torch.empty((n,) + tuple(y0.shape), dtype=torch.float32)
    X[0], Y[0] = x0, y0
    for i in range(1, n):
        X[i], Y[i] = dataset[i]
    if device is not None:
        X, Y = X.to(device), Y.to(device)
    return X, Y


def iterate(X, Y, batch_size, device, shuffle=False, generator=None):
    n = len(X)
    if shuffle:
        order = torch.randperm(n, generator=generator).to(X.device)
    else:
        order = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        sel = order[i:i + batch_size]
        xb, yb = X[sel], Y[sel]
        if xb.device != device:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
        yield xb, yb


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="openfwi_data")
    p.add_argument("--dataset", default="FlatVel_A")
    p.add_argument("--meta", action="store_true",
                   help="treat --root as a fetch_ssgen.py cache: read its "
                        "ssgen_meta.json for the split layout, grid shapes and "
                        "normalization instead of the OpenFWI constants. With "
                        "this, --train-chunks/--val-chunks 0 means all shards")
    p.add_argument("--train-chunks", type=int, default=4)
    p.add_argument("--val-chunks", type=int, default=1)
    p.add_argument("--max-train", type=int, default=None,
                   help="cap training samples (chunks are 500 each)")
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--time-stride", type=int, default=1,
                   help="subsample the 1000-step time axis; 1 keeps the benchmark "
                        "resolution. The 15 Hz source is heavily oversampled at "
                        "dt = 1 ms, so 2 is close to lossless -- check with the "
                        "band-limit oracle before using it")
    p.add_argument("--preload", action="store_true",
                   help="read chunks into RAM instead of memory-mapping")
    p.add_argument("--cache", choices=("gpu", "ram", "none"), default="gpu",
                   help="where the normalized split lives. 'gpu' (3.5 GB for "
                        "2500 samples) removes the data path from the loop "
                        "entirely; 'none' falls back to a DataLoader over the "
                        "memmap, which is I/O bound and ~8x slower per epoch")
    p.add_argument("--norm", choices=("minmax", "zscore"), default="minmax",
                   help="minmax uses OpenFWI's published constants, which keep "
                        "numbers comparable across papers. zscore standardises "
                        "gathers by training-split mean/std -- the right choice "
                        "for a cache with no published constants and a "
                        "heavy-tailed amplitude distribution, where min/max "
                        "leaves the signal in a fraction of a percent of "
                        "[-1, 1]. Scoring is on physical amplitudes either way")
    p.add_argument("--models", default=",".join(MODEL_NAMES))
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-seed", type=int, default=None,
                   help="weight init and batch order; defaults to --seed")
    p.add_argument("--ood-chunks", type=int, default=0,
                   help="with --meta, also score every trained model on the "
                        "cache's out-of-distribution block (0 = all of it, "
                        "-1 = skip). SubsurfaceGen holds Penobscot out of "
                        "training entirely, so this is the generalisation "
                        "number the dataset was built to produce")
    p.add_argument("--export", type=int, default=6,
                   help="validation samples to export predictions for")
    p.add_argument("--outdir", default="openfwi_results")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="write a resumable training checkpoint every N epochs "
                        "(0 disables checkpoints)")
    p.add_argument("--resume", action="store_true",
                   help="resume the single requested model from its checkpoint "
                        "in --outdir; requires --checkpoint-every > 0")

    # shared decoder geometry (FNO and GNO)
    p.add_argument("--t-latent", type=int, default=250,
                   help="FNO time-latent length before the upsample head")
    # FNO
    p.add_argument("--fno-width", type=int, default=32)
    p.add_argument("--fno-modes-z", type=int, default=16)
    p.add_argument("--fno-modes-x", type=int, default=16)
    p.add_argument("--fno-modes-t", type=int, default=32)
    p.add_argument("--fno-enc-layers", type=int, default=3)
    p.add_argument("--fno-dec-layers", type=int, default=3)
    # PFNO
    p.add_argument("--pfno-freqs", type=int, default=64)
    p.add_argument("--pfno-width", type=int, default=10)
    p.add_argument("--pfno-modes", type=int, default=8)
    p.add_argument("--pfno-layers", type=int, default=2)
    # DeepONet
    p.add_argument("--don-latent", type=int, default=128)
    p.add_argument("--don-hidden", type=int, default=256)
    p.add_argument("--don-fourier", type=int, default=32,
                   help="random Fourier features on the trunk; 0 = plain MLP trunk")
    p.add_argument("--don-fourier-scale", type=float, default=8.0)
    # GNO
    p.add_argument("--gno-width", type=int, default=32)
    p.add_argument("--gno-kernel-hidden", type=int, default=64)
    p.add_argument("--gno-radius", type=int, default=3)
    p.add_argument("--gno-dec-radius", type=int, default=2)
    p.add_argument("--gno-enc-layers", type=int, default=3)
    p.add_argument("--gno-dec-layers", type=int, default=2)
    p.add_argument("--gno-t-latent", type=int, default=250)
    p.add_argument("--gno-checkpoint", action="store_true",
                   help="gradient-checkpoint the graph layers; slower, much less "
                        "activation memory on a shared GPU")
    a = p.parse_args(argv)
    if a.init_seed is None:
        a.init_seed = a.seed
    if a.checkpoint_every < 0:
        p.error("--checkpoint-every must be >= 0")
    if a.resume and a.checkpoint_every == 0:
        p.error("--resume requires --checkpoint-every > 0")
    if a.resume and len([n for n in a.models.split(",") if n.strip()]) != 1:
        p.error("--resume requires exactly one model; use one invocation per model")
    return a


def checkpoint_signature(args, model_name):
    """Configuration that must stay fixed for an exact epoch-level resume."""
    ignored = {"checkpoint_every", "resume", "outdir", "export"}
    return {"model": model_name,
            "args": {k: v for k, v in vars(args).items() if k not in ignored}}


def save_training_checkpoint(path, payload):
    """Atomically replace a checkpoint so interruption cannot corrupt it."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_training_checkpoint(path, device):
    """Load trusted, locally-created trainer state across torch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.6 has no weights_only keyword
        return torch.load(path, map_location=device)


def make_scorer(norm):
    """Relative L2 (%), MAE and RMSE on physical amplitudes.

    The normalized-to-physical map is affine with a non-zero shift, so the
    target norm has to be formed in physical units rather than rescaled.
    """
    scale = float(norm.seismic_scale)
    shift = float(norm.seismic_shift)

    def denorm(y):
        return y * scale + shift

    def per_sample(pred_n, target_n):
        pred, target = denorm(pred_n), denorm(target_n)
        p, t = pred.flatten(1), target.flatten(1)
        num = torch.linalg.vector_norm(p - t, dim=1)
        den = torch.linalg.vector_norm(t, dim=1).clamp_min(1e-12)
        rel = 100.0 * num / den
        mae = (p - t).abs().mean(dim=1)
        rmse = (p - t).pow(2).mean(dim=1).sqrt()
        return rel, mae, rmse

    return denorm, per_sample


@torch.no_grad()
def evaluate(model, batches, per_sample, device):
    model.eval()
    rels, maes, rmses = [], [], []
    for xb, yb in batches():
        xb, yb = xb.to(device), yb.to(device)
        rel, mae, rmse = per_sample(model(xb), yb)
        rels.append(rel.cpu()); maes.append(mae.cpu()); rmses.append(rmse.cpu())
    rels = torch.cat(rels); maes = torch.cat(maes); rmses = torch.cat(rmses)
    return {"rel_l2_pct": float(rels.mean()), "rel_l2_pct_median": float(rels.median()),
            "mae": float(maes.mean()), "rmse": float(rmses.mean()),
            "per_sample_rel_l2": rels.tolist()}


def oracle_report(val_data, cfg, args, nt, n_samples=32):
    """Representation floors: what the band limit and the time latent cost
    before any model has been trained."""
    idx = np.linspace(0, len(val_data) - 1, min(n_samples, len(val_data))).astype(int)
    g = val_data.take(idx).astype(np.float64)
    if args.time_stride > 1:
        g = g[:, :, ::args.time_stride, :]
    band = band_limit_oracle(g, min(args.pfno_freqs, nt // 2 + 1))
    fno_t = time_resample_oracle(g, min(args.t_latent, nt))
    gno_t = time_resample_oracle(g, min(args.gno_t_latent, nt))
    return {
        "samples": int(len(idx)),
        "pfno_band_limit": {"n_freqs": int(args.pfno_freqs),
                            "rel_l2_pct": float(band.mean()),
                            "worst": float(band.max())},
        "fno_time_resample": {"t_latent": int(args.t_latent),
                              "rel_l2_pct": float(fno_t.mean()),
                              "worst": float(fno_t.max())},
        "gno_time_resample": {"t_latent": int(args.gno_t_latent),
                              "rel_l2_pct": float(gno_t.mean()),
                              "worst": float(gno_t.max())},
    }


def main(argv=None):
    a = parse_args(argv)
    if a.meta:
        meta = load_meta(a.root)
        key = a.dataset
    else:
        key = dataset_key(a.dataset)
        cfg = DATASET_CONFIG[key]
        nt = cfg["nt"] // a.time_stride
        norm = OpenFWINorm(key)

    torch.manual_seed(a.init_seed)
    np.random.seed(a.init_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(a.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    print("device: %s (%s)" % (device, torch.cuda.get_device_name(0)
                               if device.type == "cuda" else "cpu"), flush=True)

    if a.meta:
        train_data, train_model = manifest_split(a.root, meta, "train",
                                                 a.train_chunks or None, a.preload)
        val_data, val_model = manifest_split(a.root, meta, "val",
                                             a.val_chunks or None, a.preload)
        cfg = config_from_meta(meta, train_data.sample_shape,
                               train_model.sample_shape)
        nt = cfg["nt"] // a.time_stride
        if a.norm == "zscore":
            mean, std = gather_statistics(train_data)
            cfg["data_mean"], cfg["data_std"] = mean, std
            norm = ZScoreNorm(cfg)
            print("gather statistics (train split): mean %.4g  std %.4g" % (mean, std),
                  flush=True)
        else:
            norm = OpenFWINorm(cfg)
        print("cache grid: velocity (%d, %d) -> gathers (%d, %d, %d)"
              % (cfg["nz"], cfg["nx"], cfg["ns"], nt, cfg["ng"]), flush=True)
    else:
        train_data, train_model = load_split(a.root, a.dataset, "train",
                                             a.train_chunks, a.preload)
        val_data, val_model = load_split(a.root, a.dataset, "val", a.val_chunks,
                                         a.preload)
    train_set = OpenFWIForward(train_data, train_model, norm, a.time_stride, a.max_train)
    val_set = OpenFWIForward(val_data, val_model, norm, a.time_stride, a.max_val)
    print("%s: train %d / val %d   velocity %s -> gather (%d, %d, %d)"
          % (a.dataset, len(train_set), len(val_set), train_model.sample_shape,
             cfg["ns"], nt, cfg["ng"]), flush=True)

    # Keep this generator reachable: its state determines the shuffled order
    # and must be restored for an exact epoch-level resume.
    batch_gen = torch.Generator().manual_seed(a.init_seed + 23)
    cache_device = device if a.cache == "gpu" else (None if a.cache == "ram" else False)
    if cache_device is False:
        train_loader = DataLoader(
            train_set, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
            pin_memory=(device.type == "cuda"), drop_last=False,
            generator=batch_gen,
            persistent_workers=a.workers > 0)
        val_loader = DataLoader(val_set, batch_size=a.batch_size, num_workers=a.workers,
                                pin_memory=(device.type == "cuda"),
                                persistent_workers=a.workers > 0)
        n_steps = len(train_loader)

        def train_batches():
            return train_loader

        def val_batches():
            return val_loader
    else:
        t_cache = time.perf_counter()
        Xtr, Ytr = cache_split(train_set, cache_device)
        Xva, Yva = cache_split(val_set, cache_device)
        gb = (Xtr.numel() + Ytr.numel() + Xva.numel() + Yva.numel()) * 4 / 1e9
        print("cached %.1f GB on %s in %.0f s"
              % (gb, a.cache, time.perf_counter() - t_cache), flush=True)
        n_steps = (len(train_set) + a.batch_size - 1) // a.batch_size

        def train_batches():
            return iterate(Xtr, Ytr, a.batch_size, device, shuffle=True,
                           generator=batch_gen)

        def val_batches():
            return iterate(Xva, Yva, a.batch_size, device)

    ood_batches = None
    if a.meta and a.ood_chunks >= 0 and "ood" in meta.get("manifest", {}):
        ood_data, ood_model = manifest_split(a.root, meta, "ood",
                                             a.ood_chunks or None, a.preload)
        ood_set = OpenFWIForward(ood_data, ood_model, norm, a.time_stride, a.max_val)
        # Kept on the host: it is scored once per model at the end, so paying a
        # copy per batch is cheaper than holding a second split in GPU memory
        # beside the training cache.
        Xoo, Yoo = cache_split(ood_set, None)
        print("out-of-distribution split: %d samples" % len(ood_set), flush=True)

        def ood_batches():
            return iterate(Xoo, Yoo, a.batch_size, device)

    oracles = oracle_report(val_data, cfg, a, nt)
    print("representation scales:")
    print("  PFNO band limit  %4d bins : %6.3f %%  <- hard floor, bins above "
          "the band are zeroed"
          % (a.pfno_freqs, oracles["pfno_band_limit"]["rel_l2_pct"]))
    print("  FNO  time latent %4d pts  : %6.3f %%  (single-channel resample; "
          "the latent is multi-channel, so this is a reference, not a bound)"
          % (a.t_latent, oracles["fno_time_resample"]["rel_l2_pct"]))
    print("  GNO  time latent %4d pts  : %6.3f %%  (same)"
          % (a.gno_t_latent, oracles["gno_time_resample"]["rel_l2_pct"]), flush=True)

    denorm, per_sample = make_scorer(norm)
    results, histories, trained = {}, {}, {}
    checkpoint_paths = []

    for mid, name in enumerate(n.strip() for n in a.models.split(",") if n.strip()):
        torch.manual_seed(a.init_seed + 100 * mid)
        model = build_model(name, a, cfg, nt).to(device)
        n_real = count_parameters_real(model)
        print("\n=== %s: %s params (%s real) ==="
              % (name, format(count_parameters(model), ","), format(n_real, ",")),
              flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=a.lr, total_steps=a.epochs * max(1, n_steps),
            pct_start=0.15)
        best, best_state, hist = float("inf"), None, []
        start_epoch, elapsed_before = 1, 0.0
        checkpoint = outdir / ("%s_train_checkpoint.pt" % name)
        if a.checkpoint_every:
            checkpoint_paths.append(checkpoint)
        if a.resume:
            if not checkpoint.exists():
                raise FileNotFoundError("resume checkpoint not found: %s" % checkpoint)
            state = load_training_checkpoint(checkpoint, device)
            expected = checkpoint_signature(a, name)
            if state.get("signature") != expected:
                raise RuntimeError(
                    "checkpoint configuration does not match this run; "
                    "use the original arguments or a new --outdir")
            model.load_state_dict(state["model_state"])
            opt.load_state_dict(state["optimizer_state"])
            sched.load_state_dict(state["scheduler_state"])
            best = state["best"]
            best_state = state["best_state"]
            hist = state["history"]
            start_epoch = int(state["epoch"]) + 1
            elapsed_before = float(state.get("elapsed_training_s", 0.0))
            batch_gen.set_state(state["batch_generator_state"])
            torch.set_rng_state(state["torch_rng_state"].cpu())
            if device.type == "cuda" and state.get("cuda_rng_states") is not None:
                torch.cuda.set_rng_state_all(
                    [rng_state.cpu() for rng_state in state["cuda_rng_states"]])
            if state.get("numpy_rng_state") is not None:
                np.random.set_state(state["numpy_rng_state"])
            print("  resumed %s after epoch %d/%d (best %.3f%%)"
                  % (name, start_epoch - 1, a.epochs, best), flush=True)
        elif checkpoint.exists() and a.checkpoint_every:
            print("  existing checkpoint ignored (pass --resume to use it): %s"
                  % checkpoint, flush=True)
        t0 = time.perf_counter()
        for epoch in range(start_epoch, a.epochs + 1):
            model.train()
            running, nb = 0.0, 0
            for xb, yb in train_batches():
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(xb), yb)
                loss.backward()
                opt.step()
                sched.step()
                running += loss.item()
                nb += 1
            metrics = evaluate(model, val_batches, per_sample, device)
            hist.append({"epoch": epoch, "train_mse_norm": running / max(1, nb),
                         "val_rel_l2_pct": metrics["rel_l2_pct"],
                         "val_rmse": metrics["rmse"]})
            if metrics["rel_l2_pct"] < best:
                best = metrics["rel_l2_pct"]
                best_state = copy.deepcopy(model.state_dict())
            if epoch == 1 or epoch % max(1, a.epochs // 12) == 0 or epoch == a.epochs:
                print("  %-9s ep %4d/%d  train_mse=%.4e  val_relL2=%7.3f%%"
                      % (name, epoch, a.epochs, running / max(1, nb),
                         metrics["rel_l2_pct"]), flush=True)
            if (a.checkpoint_every and
                    (epoch % a.checkpoint_every == 0 or epoch == a.epochs)):
                save_training_checkpoint(checkpoint, {
                    "format_version": 1,
                    "signature": checkpoint_signature(a, name),
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "scheduler_state": sched.state_dict(),
                    "best": best,
                    "best_state": best_state,
                    "history": hist,
                    "elapsed_training_s": (elapsed_before +
                                             time.perf_counter() - t0),
                    "batch_generator_state": batch_gen.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_states": (torch.cuda.get_rng_state_all()
                                        if device.type == "cuda" else None),
                    "numpy_rng_state": np.random.get_state(),
                })
                print("     checkpoint -> %s" % checkpoint, flush=True)
        secs = elapsed_before + time.perf_counter() - t0
        model.load_state_dict(best_state)
        metrics = evaluate(model, val_batches, per_sample, device)
        trained[name] = model
        histories[name] = hist
        results[name] = {
            "parameters": count_parameters(model), "parameters_real": n_real,
            "training_time_s": secs, "seconds_per_epoch": secs / max(1, a.epochs),
            **{k: v for k, v in metrics.items() if k != "per_sample_rel_l2"},
            "per_sample_rel_l2": metrics["per_sample_rel_l2"],
        }
        if ood_batches is not None:
            ood = evaluate(model, ood_batches, per_sample, device)
            results[name]["ood"] = {k: v for k, v in ood.items()
                                    if k != "per_sample_rel_l2"}
            print("     out-of-distribution rel L2 = %.3f%%" % ood["rel_l2_pct"],
                  flush=True)
        torch.save(best_state, outdir / ("%s_state.pt" % name))
        print("  -> %s: rel L2 = %.3f%% (median %.3f%%), RMSE %.4f, %.0f s"
              % (name, metrics["rel_l2_pct"], metrics["rel_l2_pct_median"],
                 metrics["rmse"], secs), flush=True)

    # ---- export a comparable handful of validation samples --------------
    if a.export > 0 and trained:
        ids = np.linspace(0, len(val_set) - 1, min(a.export, len(val_set))).astype(int)
        v = torch.stack([val_set[int(i)][0] for i in ids])
        tgt = torch.stack([val_set[int(i)][1] for i in ids])
        export = {"velocity": np.stack([val_model[int(i)] for i in ids]).astype(np.float32),
                  "target": denorm(tgt).numpy().astype(np.float32),
                  "sample_ids": ids.astype(np.int64)}
        with torch.inference_mode():
            for name, model in trained.items():
                model.eval()
                out = [denorm(model(v[i:i + 2].to(device))).cpu().numpy()
                       for i in range(0, len(ids), 2)]
                export[name] = np.concatenate(out).astype(np.float32)
        np.savez_compressed(outdir / "openfwi_predictions.npz", **export)

    summary = {
        "dataset": a.dataset, "dataset_key": key,
        "config": cfg, "normalization": norm.asdict(),
        "grid": {"nz": cfg.get("nz", cfg.get("n_grid")),
                 "nx": cfg.get("nx", cfg.get("n_grid")),
                 "ng": cfg.get("ng"), "nt": nt,
                 "n_sources": cfg["ns"], "time_stride": a.time_stride},
        "split": {"train": len(train_set), "val": len(val_set),
                  "train_chunks": a.train_chunks, "val_chunks": a.val_chunks,
                  "note": ("manifest-defined cache split: train, validation "
                           "and optional out-of-distribution boundaries come "
                           "from ssgen_meta.json" if a.meta else
                           "official OpenFWI chunk split: 1-48 train, 49-60 val")},
        "oracles": oracles,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "args": vars(a),
        "results": [{"model": k, **v} for k, v in results.items()],
    }
    (outdir / "openfwi_summary.json").write_text(json.dumps(summary, indent=2))
    (outdir / "openfwi_histories.json").write_text(json.dumps(histories))
    # A summary is the completion marker used by the outer runner. Remove the
    # larger transient checkpoint only after all final artifacts are durable.
    for checkpoint in checkpoint_paths:
        checkpoint.unlink(missing_ok=True)
    print("\n" + json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "per_sample_rel_l2"}
         for k, v in results.items()}, indent=2))
    print("wrote", outdir)


if __name__ == "__main__":
    main()
