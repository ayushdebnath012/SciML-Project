"""Train FNO / DeepONet / PFNO on GPU and export predictions + FD targets.

Runs on the H100 box; plotting happens locally (no matplotlib needed here).
"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from operator_models import SimpleFNO2d, SimpleDeepONet, SimplePFNO, count_parameters


def count_parameters_real(model):
    """Trainable *real* scalars. `count_parameters` counts a complex weight as
    one entry, which undercounts every spectral layer by 2x and makes FNO/PFNO
    look smaller than they are next to the all-real DeepONet."""
    return sum(p.numel() * (2 if p.is_complex() else 1)
               for p in model.parameters() if p.requires_grad)

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--weight-decay", type=float, default=1e-4)
p.add_argument("--fno-width", type=int, default=48)
p.add_argument("--fno-modes", type=int, default=20)
p.add_argument("--fno-layers", type=int, default=4)
p.add_argument("--don-latent", type=int, default=192)
p.add_argument("--don-hidden", type=int, default=384)
p.add_argument("--pfno-width", type=int, default=24)
p.add_argument("--pfno-modes", type=int, default=20)
p.add_argument("--pfno-layers", type=int, default=3)
p.add_argument("--models", default="FNO,DeepONet,PFNO")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--split-seed", type=int, default=None,
               help="seed for the random train/val split; defaults to --seed. "
                    "Hold it fixed and vary --init-seed to measure run-to-run "
                    "variation with the held-out set held constant.")
p.add_argument("--init-seed", type=int, default=None,
               help="seed for weight init and batch order; defaults to --seed")
p.add_argument("--outdir", default="server_outputs_v2")
a = p.parse_args()
if a.split_seed is None: a.split_seed = a.seed
if a.init_seed is None: a.init_seed = a.seed

torch.manual_seed(a.init_seed); np.random.seed(a.init_seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
print(f"device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'cpu'})")

with np.load(a.data) as raw:
    x_grid = raw["x"].astype(np.float32); t_grid = raw["t"].astype(np.float32)
    inputs = raw["inputs"].astype(np.float32); outputs = raw["outputs"].astype(np.float32)
    kinds = raw["kinds"].astype(str)
    # Datasets drawn from a real model carry their own split: neighbouring
    # Marmousi traces are near-identical, so a random split would score
    # near-duplicates of the training profiles as held out. Honour it when
    # present; the synthetic sets have no `split` and fall back to random.
    split = raw["split"].astype(str) if "split" in raw.files else None
n_samples, _, nx, nt = inputs.shape

X = torch.from_numpy(inputs); Y = torch.from_numpy(outputs)
if split is not None:
    val_idx = torch.from_numpy(np.flatnonzero(split == "val")).long()
    train_idx = torch.from_numpy(np.flatnonzero(split != "val")).long()
    if len(val_idx) == 0 or len(train_idx) == 0:
        raise SystemExit(f"dataset `split` leaves an empty pool: {np.unique(split)}")
    print(f"using dataset-provided split (disjoint trace blocks)")
else:
    g = torch.Generator().manual_seed(a.split_seed + 17)
    perm = torch.randperm(n_samples, generator=g)
    n_val = max(1, min(n_samples - 1, int(round(0.2 * n_samples))))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

x_mean = X[train_idx].mean(dim=(0, 2, 3), keepdim=True)
x_std = X[train_idx].std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
y_mean = Y[train_idx].mean(); y_std = Y[train_idx].std().clamp_min(1e-6)
Xn = (X - x_mean) / x_std; Yn = (Y - y_mean) / y_std

train_loader = DataLoader(TensorDataset(Xn[train_idx], Yn[train_idx]), batch_size=a.batch_size,
                          shuffle=True, generator=torch.Generator().manual_seed(a.init_seed + 23))
val_loader = DataLoader(TensorDataset(Xn[val_idx], Yn[val_idx]), batch_size=a.batch_size)
print(f"train/val = {len(train_idx)}/{len(val_idx)}   grid {nx}x{nt}   samples {n_samples}")

coordinates = Xn[0, 3:5].permute(1, 2, 0).reshape(nx * nt, 2).clone().to(DEVICE)
builders = {
    "FNO": lambda: SimpleFNO2d(width=a.fno_width, modes_x=a.fno_modes,
                               modes_t=a.fno_modes, layers=a.fno_layers),
    "DeepONet": lambda: SimpleDeepONet(nx=nx, coordinates=coordinates,
                                       latent=a.don_latent, hidden=a.don_hidden),
    "PFNO": lambda: SimplePFNO(nt=nt, width=a.pfno_width, modes=a.pfno_modes,
                               layers=a.pfno_layers),
}

y_mean_d, y_std_d = y_mean.to(DEVICE), y_std.to(DEVICE)
def denorm(y): return y * y_std_d + y_mean_d

@torch.no_grad()
def evaluate(model):
    model.eval(); se = 0.0; n = 0; rels = []
    for xb, yb in val_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = denorm(model(xb)); tgt = denorm(yb)
        se += F.mse_loss(pred, tgt, reduction="sum").item(); n += tgt.numel()
        pf, tf = pred.flatten(1), tgt.flatten(1)
        rel = torch.linalg.vector_norm(pf - tf, dim=1) / torch.linalg.vector_norm(tf, dim=1).clamp_min(1e-12)
        rels.extend((100.0 * rel).cpu().tolist())
    return se / n, float(np.mean(rels)), rels

results = {}; histories = {}; trained = {}
for mid, name in enumerate(a.models.split(",")):
    torch.manual_seed(a.init_seed + 100 * mid)
    model = builders[name]().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(train_loader), pct_start=0.15)
    best_val, best_state, hist = float("inf"), None, []
    t0 = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train(); ls = 0.0; nb = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb), yb)
            loss.backward(); opt.step(); sched.step()
            ls += loss.item(); nb += 1
        vmse, vrel, _ = evaluate(model)
        hist.append({"epoch": epoch, "train_nmse": ls / max(1, nb),
                     "val_mse": vmse, "val_rel_l2": vrel})
        if vmse < best_val:
            best_val = vmse; best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % max(1, a.epochs // 10) == 0 or epoch == a.epochs:
            print(f"  {name:9s} ep {epoch:4d}/{a.epochs} train_nMSE={ls/max(1,nb):.3e} val_L2={vrel:7.3f}%", flush=True)
    secs = time.perf_counter() - t0
    model.load_state_dict(best_state)
    vmse, vrel, per_sample = evaluate(model)
    trained[name] = model
    histories[name] = hist
    results[name] = {"parameters": count_parameters(model),
                     "parameters_real": count_parameters_real(model),
                     "training time (s)": secs,
                     "validation MSE": vmse, "validation rel L2 (%)": vrel,
                     "per_sample_rel_l2": per_sample}
    print(f"  -> {name}: {count_parameters(model):,} params "
          f"({count_parameters_real(model):,} real), {secs:.1f}s, "
          f"rel L2 = {vrel:.3f}%\n", flush=True)
    torch.save(best_state, outdir / f"{name}_state.pt")

# ---- export predictions for every validation sample -------------------------
export_ids = val_idx.tolist()
preds = {}
with torch.inference_mode():
    xb = Xn[export_ids].to(DEVICE)
    for name, model in trained.items():
        model.eval()
        out = []
        for i in range(0, len(export_ids), 16):
            out.append(denorm(model(xb[i:i+16]))[:, 0].cpu().numpy())
        preds[name] = np.concatenate(out).astype(np.float32)

np.savez_compressed(
    outdir / "operator_wave_predictions_v2.npz",
    x=x_grid, t=t_grid,
    validation_ids=np.asarray(export_ids, dtype=np.int64),
    validation_kinds=kinds[export_ids],
    E=inputs[export_ids, 0, :, 0], rho=inputs[export_ids, 1, :, 0],
    target=outputs[export_ids, 0].astype(np.float32),
    **preds,
)

summary = {
    "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
    "torch_version": torch.__version__,
    "dataset": {"path": a.data, "samples": int(n_samples), "nx": int(nx), "nt": int(nt),
                "train": int(len(train_idx)), "val": int(len(val_idx))},
    "args": vars(a),
    "training_results": [{"model": k, **v} for k, v in results.items()],
}
(outdir / "operator_wave_summary_v2.json").write_text(json.dumps(summary, indent=2))
(outdir / "histories_v2.json").write_text(json.dumps(histories))
print(json.dumps({k: {kk: vv for kk, vv in v.items()
                      if kk != "per_sample_rel_l2"}
                  for k, v in results.items()}, indent=2))
print("wrote", outdir)
