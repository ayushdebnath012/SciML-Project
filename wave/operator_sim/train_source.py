"""Train FNO / DeepONet / PFNO on the forced wave equation.

Reports the field-wise relative L2 plus initial- and boundary-condition
residuals, each alongside the finite-difference reference's own value (the
achievable floor on this grid).
"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from operator_models import SimpleFNO2d, SimpleDeepONet, SimplePFNO, count_parameters
from physics_metrics import evaluate_set
from physics_losses_torch import ic_bc_losses, training_scales

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--epochs", type=int, default=400)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--weight-decay", type=float, default=1e-4)
p.add_argument("--fno-width", type=int, default=48)
p.add_argument("--fno-modes", type=int, default=20)
p.add_argument("--fno-layers", type=int, default=4)
p.add_argument("--don-latent", type=int, default=256)
p.add_argument("--don-hidden", type=int, default=512)
p.add_argument("--pfno-width", type=int, default=24)
p.add_argument("--pfno-modes", type=int, default=20)
p.add_argument("--pfno-layers", type=int, default=3)
p.add_argument("--models", default="FNO,DeepONet,PFNO")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--outdir", default="server_outputs_source")
# Physics-informed terms. 0.0 reproduces the plain supervised run exactly.
p.add_argument("--lambda-physics", type=float, default=0.0,
               help="weight on the combined IC+BC loss")
a = p.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
print(f"device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'cpu'})")

with np.load(a.data) as raw:
    x_grid = raw["x"].astype(np.float32); t_grid = raw["t"].astype(np.float32)
    inputs = raw["inputs"].astype(np.float32); outputs = raw["outputs"].astype(np.float32)
    kinds = raw["kinds"].astype(str)
    x_src = raw["x_src"]; f_peak = raw["f_peak"]
n_samples, n_ch, nx, nt = inputs.shape
assert n_ch == 6, f"expected 6 input channels, got {n_ch}"

X = torch.from_numpy(inputs); Y = torch.from_numpy(outputs)
g = torch.Generator().manual_seed(a.seed + 17)
perm = torch.randperm(n_samples, generator=g)
n_val = max(1, min(n_samples - 1, int(round(0.2 * n_samples))))
val_idx, train_idx = perm[:n_val], perm[n_val:]

x_mean = X[train_idx].mean(dim=(0, 2, 3), keepdim=True)
x_std = X[train_idx].std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
y_mean = Y[train_idx].mean(); y_std = Y[train_idx].std().clamp_min(1e-6)
Xn = (X - x_mean) / x_std; Yn = (Y - y_mean) / y_std

train_loader = DataLoader(TensorDataset(Xn[train_idx], Yn[train_idx], train_idx), batch_size=a.batch_size,
                          shuffle=True, generator=torch.Generator().manual_seed(a.seed + 23))
val_loader = DataLoader(TensorDataset(Xn[val_idx], Yn[val_idx]), batch_size=a.batch_size)
print(f"train/val = {len(train_idx)}/{len(val_idx)}   grid {nx}x{nt}")

# Coordinate channels are 4 and 5 in the forced layout [E,rho,s,w,x,t].
coordinates = Xn[0, 4:6].permute(1, 2, 0).reshape(nx * nt, 2).clone().to(DEVICE)
builders = {
    "FNO": lambda: SimpleFNO2d(width=a.fno_width, modes_x=a.fno_modes,
                               modes_t=a.fno_modes, layers=a.fno_layers, in_channels=6),
    "DeepONet": lambda: SimpleDeepONet(nx=nx, coordinates=coordinates,
                                       latent=a.don_latent, hidden=a.don_hidden, n_time=nt),
    "PFNO": lambda: SimplePFNO(nt=nt, width=a.pfno_width, modes=a.pfno_modes,
                               layers=a.pfno_layers, source_mode=True),
}

y_mean_d, y_std_d = y_mean.to(DEVICE), y_std.to(DEVICE)
def denorm(y): return y * y_std_d + y_mean_d

# --- physics-loss setup -----------------------------------------------------
dx_phys = float(x_grid[1] - x_grid[0]); dt_phys = float(t_grid[1] - t_grid[0])
E_all = torch.from_numpy(inputs[:, 0, :, 0]); rho_all = torch.from_numpy(inputs[:, 1, :, 0])
C_all = torch.sqrt(E_all / rho_all.clamp_min(1e-12))
if a.lambda_physics > 0:
    SCALES = training_scales(Y[train_idx][:, 0], C_all[train_idx], dx_phys, dt_phys)
    print(f"physics scales: s_u={SCALES[0]:.4f} s_ut={SCALES[1]:.4f} s_bc={SCALES[2]:.4f}")
    print(f"lambda_physics = {a.lambda_physics}")
else:
    SCALES = (1.0, 1.0, 1.0)
C_all = C_all.to(DEVICE)

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
    return se / n, float(np.mean(rels))

results, histories, trained = {}, {}, {}
for mid, name in enumerate(a.models.split(",")):
    torch.manual_seed(a.seed + 100 * mid)
    model = builders[name]().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(train_loader), pct_start=0.15)
    best_val, best_state, hist = float("inf"), None, []
    t0 = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train(); ls = 0.0; nb = 0; lp = 0.0
        for xb, yb, ib in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            if a.lambda_physics > 0:
                l_icu, l_icut, l_bc = ic_bc_losses(
                    denorm(pred)[:, 0], C_all[ib.to(DEVICE)], dx_phys, dt_phys, SCALES)
                phys = l_icu + l_icut + l_bc
                loss = loss + a.lambda_physics * phys
                lp += float(phys.detach())
            loss.backward(); opt.step(); sched.step()
            ls += loss.item(); nb += 1
        vmse, vrel = evaluate(model)
        hist.append({"epoch": epoch, "train_nmse": ls / max(1, nb),
                     "val_mse": vmse, "val_rel_l2": vrel})
        if vmse < best_val:
            best_val = vmse; best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % max(1, a.epochs // 10) == 0 or epoch == a.epochs:
            msg = f"  {name:9s} ep {epoch:4d}/{a.epochs} train_loss={ls/max(1,nb):.3e} val_L2={vrel:7.3f}%"
            if a.lambda_physics > 0:
                msg += f" phys={lp/max(1,nb):.3e}"
            print(msg, flush=True)
    secs = time.perf_counter() - t0
    model.load_state_dict(best_state)
    vmse, vrel = evaluate(model)
    trained[name] = model; histories[name] = hist
    results[name] = {"parameters": count_parameters(model), "training time (s)": secs,
                     "validation MSE": vmse, "validation rel L2 (%)": vrel}
    print(f"  -> {name}: {count_parameters(model):,} params, {secs:.1f}s, rel L2 = {vrel:.3f}%\n", flush=True)
    torch.save(best_state, outdir / f"{name}_state.pt")

# ---- predictions on the whole validation set -------------------------------
export_ids = val_idx.tolist()
preds = {}
with torch.inference_mode():
    xb_all = Xn[export_ids].to(DEVICE)
    for name, model in trained.items():
        model.eval(); out = []
        for i in range(0, len(export_ids), 16):
            out.append(denorm(model(xb_all[i:i + 16]))[:, 0].cpu().numpy())
        preds[name] = np.concatenate(out).astype(np.float32)

E_val = inputs[export_ids, 0, :, 0]
rho_val = inputs[export_ids, 1, :, 0]
target_val = outputs[export_ids, 0].astype(np.float32)

# ---- IC / BC residuals, model vs the FD reference floor --------------------
physics = {}
ref_row = None
for name in preds:
    model_res, ref_res = evaluate_set(preds[name], target_val, E_val, rho_val, x_grid, t_grid)
    physics[name] = model_res
    ref_row = ref_res
physics["FD_reference"] = ref_row

print("\n=== IC / BC residuals (validation mean) ===")
cols = ["ic_displacement_rms", "ic_velocity_rms", "bc_rms", "bc_rel"]
print(f"{'model':14s}" + "".join(f"{c:>22s}" for c in cols))
for name in list(preds) + ["FD_reference"]:
    print(f"{name:14s}" + "".join(f"{physics[name][c]:22.6e}" for c in cols))

np.savez_compressed(
    outdir / "operator_source_predictions.npz",
    x=x_grid, t=t_grid,
    validation_ids=np.asarray(export_ids, dtype=np.int64),
    validation_kinds=kinds[export_ids],
    E=E_val, rho=rho_val,
    s_x=inputs[export_ids, 2, :, 0], w_t=inputs[export_ids, 3, 0, :],
    x_src=x_src[export_ids], f_peak=f_peak[export_ids],
    target=target_val, **preds,
)
summary = {
    "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
    "torch_version": torch.__version__,
    "arm": "forced (Ricker point source, quiescent IC, absorbing BC)",
    "lambda_physics": a.lambda_physics,
    "physics_scales": list(SCALES),
    "dataset": {"path": a.data, "samples": int(n_samples), "nx": int(nx), "nt": int(nt),
                "train": int(len(train_idx)), "val": int(len(val_idx))},
    "args": vars(a),
    "training_results": [{"model": k, **v} for k, v in results.items()],
    "physics_residuals": physics,
}
(outdir / "operator_source_summary.json").write_text(json.dumps(summary, indent=2))
(outdir / "histories_source.json").write_text(json.dumps(histories))
print("\n" + json.dumps(results, indent=2))
print("wrote", outdir)
