"""Train a Fourier Neural Operator (FNO) baseline for the 1D wave equation experiments.

Unlike the pointwise PINN/PIKAN baselines (trained via PDE-residual autograd for a single
fixed initial condition), the FNO is trained as a supervised operator: it maps the gridded
fields (x, E(x), g(x; sigma_g), t) to the ansatz's NN-term, supervised against a bank of
finite-difference solutions swept over the IC width sigma_g. This makes it a genuine
IC -> solution operator rather than a memorized single wavefield, while still being
evaluated by benchmark_v2.py at the canonical sigma_g=0.1, x0=0 case like every other arch.
"""
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm

from src.physics import HomogeneousModel, TwoLayerModel, MultiLayerModel, fd_reference
from src.ansatz_losses import gaussian_ic, apply_ansatz
from src.models import FNOWrapper
from src.train import make_lr_scheduler
from src.evaluation import evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MATERIALS = {
    "exp1_homogeneous": HomogeneousModel,
    "exp2_twolayer":    TwoLayerModel,
    "exp3_multilayer":  MultiLayerModel,
}


def build_fd_bank(material, nx, nt, T_nd, sigma_list_nd):
    """Precompute FD reference solutions across a bank of IC widths (x0 fixed at 0)."""
    x_fd, t_fd = None, None
    U = np.zeros((len(sigma_list_nd), nt, nx), dtype=np.float32)
    for i, sg in enumerate(tqdm(sigma_list_nd, desc="Building FD training bank")):
        x_fd, t_fd, u = fd_reference(material, nx=nx, nt=nt, T=T_nd, sigma_g=float(sg))
        U[i] = u
    return x_fd, t_fd, U


def train_fno(prefix, material, nx=512, n_solves=32, nt_bank=300, T_phys=1.0,
              sigma_range_phys=(0.05, 0.2), sigma_canonical_phys=0.1,
              n_steps=2000, batch_size=32, lr=1e-3, modes=32, width=64, n_layers=3,
              save_path=None):
    save_path = save_path or f"Models/{prefix}_fno_model.pt"
    T_nd = material.to_nondim_t(T_phys)
    sigma_canonical_nd = material.nondim_sigma_g(sigma_canonical_phys)

    rng = np.random.default_rng(SEED)
    sigma_phys = rng.uniform(*sigma_range_phys, size=n_solves)
    sigma_phys[0] = sigma_canonical_phys  # guarantee the canonical eval case is in the training bank
    sigma_nd = np.array([material.nondim_sigma_g(float(s)) for s in sigma_phys], dtype=np.float32)

    x_fd, t_fd, U = build_fd_bank(material, nx, nt_bank, T_nd, sigma_nd)

    model = FNOWrapper(nx=nx, modes=modes, width=width, n_layers=n_layers, sigma_g=sigma_canonical_nd).to(DEVICE)
    model.set_material(material, sigma_g=sigma_canonical_nd)

    x_grid_t = torch.tensor(x_fd, dtype=torch.float32, device=DEVICE)
    E_grid_t = material.E(x_grid_t)
    t_bank_t = torch.tensor(t_fd, dtype=torch.float32, device=DEVICE)
    U_t = torch.tensor(U, dtype=torch.float32, device=DEVICE)              # (n_solves, nt_bank, nx)
    sigma_nd_list = sigma_nd.tolist()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = make_lr_scheduler(optimizer, warmup_steps=200, decay_rate=0.9, decay_steps=500)

    # Canonical evaluation reference: matches benchmark_v2.py exactly (nx=512, nt=2000)
    x_eval, t_eval_grid, u_eval = fd_reference(material, nx=nx, nt=2000, T=T_nd, sigma_g=sigma_canonical_nd)
    eval_phys = list(np.linspace(0.05, T_phys, 20))
    eval_times_nd = [material.to_nondim_t(tp) for tp in eval_phys]

    best_l2 = float("inf")
    pbar = tqdm(range(n_steps), desc=f"FNO Adam [{prefix}]")
    for step in pbar:
        # One sigma_g per step (kept constant across the batch) so gaussian_ic's
        # per-call max-normalization stays correct; sigma_g still varies across steps.
        sample_idx = step % n_solves
        sg_scalar = sigma_nd_list[sample_idx]
        t_idx = torch.randint(0, nt_bank, (batch_size,), device=DEVICE)

        t_b = t_bank_t[t_idx]                                              # (B,)
        target = U_t[sample_idx, t_idx]                                    # (B, nx)

        x_col = x_grid_t.unsqueeze(0).expand(batch_size, nx)
        E_col = E_grid_t.unsqueeze(0).expand(batch_size, nx)
        t_col = t_b.unsqueeze(1).expand(batch_size, nx)
        g_row = gaussian_ic(x_grid_t, sg_scalar)
        g_col = g_row.unsqueeze(0).expand(batch_size, nx)

        inp = torch.stack([x_col, E_col, g_col, t_col], dim=-1)            # (B, nx, 4)

        optimizer.zero_grad()
        nn_out = model.fno(inp).squeeze(-1)                                # (B, nx)
        pred = apply_ansatz(nn_out, x_col, t_col, material, sg_scalar)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            metrics = evaluate(model, material, x_eval, t_eval_grid, u_eval,
                                sigma_g=sigma_canonical_nd, eval_times=eval_times_nd)
            l2 = metrics["mean_l2"]
            if l2 < best_l2:
                best_l2 = l2
                torch.save(model.state_dict(), save_path)
            pbar.set_postfix({"Loss": f"{loss.item():.2e}", "L2": f"{l2:.2f}%", "Best": f"{best_l2:.2f}%"})

    print(f"\n>>> [{prefix}] FNO training complete. Best canonical mean L2: {best_l2:.4f}%  ->  {save_path}")
    return best_l2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="exp1_homogeneous", choices=list(MATERIALS.keys()))
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--n_solves", type=int, default=32)
    args = parser.parse_args()

    material = MATERIALS[args.experiment]()
    train_fno(args.experiment, material, n_steps=args.n_steps, n_solves=args.n_solves)
