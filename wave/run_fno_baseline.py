import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WAVE_DIR = os.path.abspath(os.path.dirname(__file__))
for path in (ROOT, WAVE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from fd_solver import solve_wave_1d
# Re-exported: the dataset generators import these off this module.
from wave_problem import (  # noqa: F401
    gaussian_derivative_ic,
    make_input_tensor,
    sample_material_profile,
    solve_case,
)
from src.operator_baselines import (
    ConvBaseline2d,
    FNO2d,
    count_parameters,
    relative_l2_percent,
)
from wave.materials import HomogeneousModel, MultiLayerModel, TwoLayerModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_dataset(args):
    rng = np.random.default_rng(args.seed)
    x_grid = np.linspace(-1.0, 1.0, args.nx)
    t_grid = np.linspace(0.0, args.t_max, args.nt)
    inputs, outputs, kinds = [], [], []

    print(f"Generating {args.num_samples} FD training samples...")
    for i in range(args.num_samples):
        E, rho, kind = sample_material_profile(rng, x_grid)
        sigma_g = args.sigma_g
        x0 = 0.0
        if args.random_ic:
            sigma_g = float(rng.uniform(0.07, 0.16))
            x0 = float(rng.uniform(-0.25, 0.25))

        u = solve_case(x_grid, t_grid, E, rho, sigma_g, x0, args.cfl)
        inputs.append(make_input_tensor(x_grid, t_grid, E, rho, sigma_g, x0))
        outputs.append(u[None, ...].astype(np.float32))
        kinds.append(kind)
        if (i + 1) % max(1, args.num_samples // 10) == 0:
            print(f"  {i + 1}/{args.num_samples} samples")

    return {
        "x": x_grid.astype(np.float32),
        "t": t_grid.astype(np.float32),
        "inputs": np.stack(inputs, axis=0).astype(np.float32),
        "outputs": np.stack(outputs, axis=0).astype(np.float32),
        "kinds": np.asarray(kinds),
    }


def dataset_cache_path(args):
    if args.dataset_cache:
        return Path(args.dataset_cache)
    ic_tag = "randomic" if args.random_ic else "fixedic"
    name = (
        f"wave_operator_{ic_tag}_n{args.num_samples}_nx{args.nx}_"
        f"nt{args.nt}_t{args.t_max:g}_seed{args.seed}.npz"
    )
    return Path("operator_data") / name


def load_or_generate_dataset(args):
    cache = dataset_cache_path(args)
    if cache.exists() and not args.rebuild_dataset:
        print(f"Loading cached dataset: {cache}")
        data = np.load(cache, allow_pickle=False)
        return {key: data[key] for key in data.files}

    data = generate_dataset(args)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    print(f"Saved dataset cache: {cache}")
    return data


def make_loaders(data, args):
    x = torch.from_numpy(data["inputs"])
    y = torch.from_numpy(data["outputs"])
    n = x.shape[0]
    generator = torch.Generator().manual_seed(args.seed + 17)
    perm = torch.randperm(n, generator=generator)
    val_count = max(1, int(round(n * args.val_fraction)))
    val_count = min(val_count, n - 1)
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    train_ds = TensorDataset(x[train_idx], y[train_idx])
    val_ds = TensorDataset(x[val_idx], y[val_idx])
    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory and use_cuda),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 23),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def save_history_plot(history, outpath):
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    train = [row["train_mse"] for row in history]
    val = [row["val_mse"] for row in history]
    rel_l2 = [row["val_relative_l2_percent"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train, label="train MSE", linewidth=2)
    axes[0].plot(epochs, val, label="val MSE", linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, rel_l2, color="#b23a48", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation relative L2 (%)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_model(name: str, args):
    if name == "fno":
        return FNO2d(
            in_channels=5,
            out_channels=1,
            width=args.width,
            modes_x=args.modes_x,
            modes_t=args.modes_t,
            n_layers=args.layers,
            padding=args.padding,
        )
    if name == "cnn":
        return ConvBaseline2d(
            in_channels=5,
            out_channels=1,
            width=args.width,
            n_layers=args.layers,
        )
    raise ValueError(f"Unknown model: {name}")


def evaluate_loader(model, loader, device):
    model.eval()
    mse_sum = 0.0
    n_values = 0
    l2_values = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            mse_sum += F.mse_loss(pred, yb, reduction="sum").item()
            n_values += yb.numel()
            l2_values.extend(relative_l2_percent(pred, yb).cpu().tolist())
    return mse_sum / max(1, n_values), float(np.mean(l2_values))


def _autocast_enabled(args, device):
    return bool(args.amp and device.type == "cuda")


def _make_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(enabled):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _load_checkpoint_if_requested(model, optimizer, scheduler, args, device):
    if not args.resume:
        return 1, float("inf"), []
    resume_path = Path(args.resume)
    if not resume_path.exists():
        raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val = float(checkpoint.get("best_val_mse", float("inf")))
    history = list(checkpoint.get("history", []))
    print(f"Resumed from {resume_path} at epoch {start_epoch}")
    return start_epoch, best_val, history


def _save_training_checkpoint(path, model_name, model, optimizer, scheduler,
                              args, epoch, best_val, history):
    payload = {
        "model_name": model_name,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "best_val_mse": best_val,
        "history": history,
        "parameter_count": count_parameters(model),
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)


def train_model(model_name, data, args, device):
    model = build_model(model_name, args).to(device)
    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    train_loader, val_loader = make_loaders(data, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.min_lr,
        )
    use_amp = _autocast_enabled(args, device)
    scaler = _make_grad_scaler(use_amp)

    outdir = Path(args.output_dir) / model_name
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / "model_best.pt"
    latest_path = outdir / "model_latest.pt"
    start_epoch, best_val, history = _load_checkpoint_if_requested(
        model, optimizer, scheduler, args, device
    )

    print(f"\nTraining {model_name.upper()} ({count_parameters(model):,} parameters)")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_values = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(use_amp):
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += F.mse_loss(pred.detach(), yb, reduction="sum").item()
            train_values += yb.numel()

        train_mse = train_loss_sum / max(1, train_values)
        val_mse, val_rel_l2 = evaluate_loader(model, val_loader, device)
        if scheduler is not None:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_relative_l2_percent": val_rel_l2,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        if val_mse < best_val:
            best_val = val_mse
            _save_training_checkpoint(
                ckpt_path, model_name, model, optimizer, scheduler,
                args, epoch, best_val, history,
            )
        if args.save_every > 0 and (epoch == args.epochs or epoch % args.save_every == 0):
            _save_training_checkpoint(
                latest_path, model_name, model, optimizer, scheduler,
                args, epoch, best_val, history,
            )

        if epoch == 1 or epoch == args.epochs or epoch % args.print_every == 0:
            print(
                f"  epoch {epoch:04d} | train_mse={train_mse:.4e} "
                f"| val_mse={val_mse:.4e} | val_l2={val_rel_l2:.3f}% "
                f"| lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    with open(outdir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    save_history_plot(history, outdir / "history.png")

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
    return model, outdir, history


def material_profile(material, x_grid):
    x_t = torch.tensor(x_grid, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        E = material.E(x_t).reshape(-1).cpu().numpy()
        rho = material.rho(x_t).reshape(-1).cpu().numpy()
    return E.astype(np.float64), rho.astype(np.float64)


def save_solution_plot(x, t, target, pred, title, outpath):
    residual = np.abs(target - pred)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("FD Reference", target, "plasma"),
        (title, pred, "plasma"),
        ("Absolute Residual", residual, "coolwarm"),
    ]
    for ax, (name, values, cmap) in zip(axes, panels):
        im = ax.imshow(
            np.flip(values.T, axis=0),
            extent=[float(x.min()), float(x.max()), float(t.min()), float(t.max())],
            cmap=cmap,
            aspect="auto",
        )
        ax.set_title(name)
        ax.set_xlabel("Space x")
        ax.set_ylabel("Time t")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=250, bbox_inches="tight")
    plt.close(fig)


def evaluate_reference_materials(model, model_name, data, args, device, outdir):
    eval_nx = args.eval_nx or int(data["x"].shape[0])
    eval_nt = args.eval_nt or int(data["t"].shape[0])
    x_grid = np.linspace(-1.0, 1.0, eval_nx, dtype=np.float64)
    t_grid = np.linspace(0.0, args.t_max, eval_nt, dtype=np.float64)
    materials = [HomogeneousModel(), TwoLayerModel(), MultiLayerModel()]
    metrics = {}
    model.eval()

    for material in materials:
        E, rho = material_profile(material, x_grid)
        target = solve_case(x_grid, t_grid, E, rho, args.sigma_g, 0.0, args.cfl)
        inp = make_input_tensor(x_grid, t_grid, E, rho, args.sigma_g, 0.0)
        xb = torch.from_numpy(inp[None, ...]).to(device)
        with torch.no_grad():
            pred = model(xb).cpu().numpy()[0, 0]

        rel_l2 = float(100.0 * np.linalg.norm(pred - target) /
                       max(np.linalg.norm(target), 1e-12))
        metrics[material.name] = {
            "relative_l2_error_percent": rel_l2,
            "material_domain": [float(material.x_min), float(material.x_max)],
            "evaluation_grid": [int(eval_nx), int(eval_nt)],
        }
        save_solution_plot(
            x_grid,
            t_grid,
            target,
            pred,
            f"{model_name.upper()} Prediction",
            outdir / f"{material.name}_solution_comparison.png",
        )
        print(f"  {material.name:<12s} benchmark L2: {rel_l2:.3f}%")

    with open(outdir / "material_benchmarks.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train supervised FNO/CNN neural-operator baselines for the 1D elastic wave equation."
    )
    parser.add_argument("--model", choices=["fno", "cnn", "all"], default="fno")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--nt", type=int, default=128)
    parser.add_argument("--eval-nx", type=int, default=None,
                        help="Evaluation grid size in x; default matches --nx.")
    parser.add_argument("--eval-nt", type=int, default=None,
                        help="Evaluation grid size in t; default matches --nt.")
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--sigma-g", type=float, default=0.1)
    parser.add_argument("--random-ic", action="store_true")
    parser.add_argument("--cfl", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes-x", type=int, default=12)
    parser.add_argument("--modes-t", type=int, default=12)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="./operator_results")
    parser.add_argument("--dataset-cache", default=None)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true",
                        help="Use CUDA autocast/GradScaler. Keep off if FFT autocast is unstable.")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile. Optional; first epoch can be slower.")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--resume", default=None,
                        help="Path to a model_latest.pt or model_best.pt checkpoint.")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Write model_latest.pt every N epochs; 0 disables.")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--test", action="store_true",
                        help="Small smoke test: tiny dataset, one epoch, small models.")
    args = parser.parse_args()

    if args.test:
        args.num_samples = 6
        args.nx = 24
        args.nt = 24
        args.eval_nx = 24
        args.eval_nt = 24
        args.epochs = 1
        args.batch_size = 2
        args.width = 8
        args.modes_x = 4
        args.modes_t = 4
        args.layers = 2
        args.output_dir = "./operator_results_test"
        args.print_every = 1
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"Using device: {device}")

    data = load_or_generate_dataset(args)
    model_names = ["fno", "cnn"] if args.model == "all" else [args.model]
    summary = {
        "problem": "1D conservative elastic wave equation",
        "operator_input_channels": ["E(x)", "rho(x)", "g(x)", "x", "t"],
        "operator_output": "u(x,t)",
        "dataset_shape": {
            "inputs": list(data["inputs"].shape),
            "outputs": list(data["outputs"].shape),
        },
        "evaluation_grid": [
            int(args.eval_nx or data["x"].shape[0]),
            int(args.eval_nt or data["t"].shape[0]),
        ],
        "training_args": vars(args),
        "models": {},
    }

    for model_name in model_names:
        model, outdir, history = train_model(model_name, data, args, device)
        _, val_loader = make_loaders(data, args)
        val_mse, val_rel_l2 = evaluate_loader(model, val_loader, device)
        benchmarks = evaluate_reference_materials(model, model_name, data, args, device, outdir)
        summary["models"][model_name] = {
            "val_mse": val_mse,
            "val_relative_l2_percent": val_rel_l2,
            "epochs": args.epochs,
            "parameter_count": count_parameters(model),
            "output_dir": str(outdir),
            "material_benchmarks": benchmarks,
            "last_history_row": history[-1] if history else None,
        }

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.output_dir) / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
