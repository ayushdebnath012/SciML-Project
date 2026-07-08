import os
import time
import pickle as pkl
import itertools
import numpy as np
from numpy import linalg as LA
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
from scipy.interpolate import RegularGridInterpolator

from src.physics import MaterialModel, solve_wave_fd, u_sol, u_grad_sol
from src.ansatz_losses import apply_ansatz, compute_pde_residual, dy_dx, my_material
from src.models import build_kan, num_parameters_kan

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARCH_LABELS = {
    "vanilla": "Vanilla PINN",
    "fourier": "Fourier-feature PINN",
    "pirate":  "PirateNet",
    "piann_fd": "PIANN (FD)",
    "piann_ad": "PIANN (AD)",
    "pikan": "PIKAN",
    "WaveKAN_3-10": "WaveKAN (3 layers)",
    "WaveKAN_5-10": "WaveKAN (5 layers)",
    "WaveKAN_7-10": "WaveKAN (7 layers)"
}

ARCH_COLORS = {
    "vanilla": "#E67E22",
    "fourier": "#2980B9",
    "pirate":  "#27AE60",
    "piann_fd": "#C0392B",
    "piann_ad": "#8E44AD",
    "pikan": "#1ABC9C",
    "WaveKAN_3-10": "#D35400",
    "WaveKAN_5-10": "#C0392B",
    "WaveKAN_7-10": "#9B59B6"
}

# ─────────────────────────────────────────────
# 1. Numerical Evaluation Metrics
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, material: MaterialModel, x_fd: np.ndarray, t_fd: np.ndarray, 
             u_fd: np.ndarray, sigma_g: float = 0.1, eval_times: list = None) -> dict:
    if eval_times is None:
        n_snap = min(10, len(t_fd))
        eval_times = [t_fd[int(i * (len(t_fd) - 1) / (n_snap - 1))] for i in range(n_snap)]

    l2_errors, energy_diffs, t_eval_actual = [], [], []
    x_tensor = torch.tensor(x_fd, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    
    # Ensure model is in evaluation mode
    model.eval()

    for t_val in eval_times:
        t_idx = np.argmin(np.abs(t_fd - t_val))
        t_actual = t_fd[t_idx]
        u_ref    = u_fd[t_idx]

        t_tensor = torch.full_like(x_tensor, t_actual)
        u_pred   = apply_ansatz(model(x_tensor, t_tensor), x_tensor, t_tensor, material, sigma_g)
        u_pred_np = u_pred.cpu().numpy().flatten()

        denom = np.linalg.norm(u_ref)
        l2_errors.append((np.linalg.norm(u_ref - u_pred_np) / (denom + 1e-12)) * 100.0)
        energy_diffs.append(np.sum((u_pred_np - u_ref) ** 2) / (np.sum(u_ref ** 2) + 1e-12))
        t_eval_actual.append(t_actual)

    return {"t": np.array(t_eval_actual), "l2_errors": np.array(l2_errors),
            "energy_diff": np.array(energy_diffs), "mean_l2": float(np.mean(l2_errors))}


def rel_l2_error(residual, ref_sol):
    return 100 * LA.norm(residual) / LA.norm(ref_sol)


def max_abs_error(residual):
    return np.max(np.abs(residual))


def compute_error_stats(model, x_eval, ref_sol, ref_grad, dir, repetitions):
    rel_l2_errors = []
    max_abs_errors = []
    rel_l2_errors_grad = []
    max_abs_errors_grad = []
    model.to(x_eval.device)

    for rep in range(repetitions):
        model.load_state_dict(torch.load(dir + f"model{rep+1}.pkl", weights_only=True))
        model.eval()
        
        # Use ansatz for evaluation
        with torch.no_grad():
            x_eval_t = x_eval[:, 0:1]
            t_eval_t = x_eval[:, 1:2]
            pred_sol_raw = model(x_eval)
            pred_sol_torch = apply_ansatz(pred_sol_raw, x_eval_t, t_eval_t, my_material)
            
        # For gradient, we need autograd
        x_eval_grad = x_eval.clone().requires_grad_(True)
        pred_sol_grad = apply_ansatz(model(x_eval_grad), x_eval_grad[:, 0:1], x_eval_grad[:, 1:2], my_material)
        pred_grad_torch = dy_dx(pred_sol_grad, x_eval_grad)
        
        with torch.no_grad():
            pred_sol_np = pred_sol_torch.to('cpu').detach().numpy()
            residual = ref_sol - pred_sol_np
            rel_l2_errors.append(rel_l2_error(residual, ref_sol))
            max_abs_errors.append(max_abs_error(residual))

            pred_grad_np = pred_grad_torch.to('cpu').detach().numpy()
            grad_residual_norm = LA.norm(ref_grad - pred_grad_np, axis=-1, keepdims=True)
            ref_grad_norm = LA.norm(ref_grad, axis=-1, keepdims=True)
            rel_l2_errors_grad.append(rel_l2_error(grad_residual_norm, ref_grad_norm))
            max_abs_errors_grad.append(max_abs_error(grad_residual_norm))
            
    summary = {
        'relative_l2_errors': rel_l2_errors, 
        'avg_rel_l2_err': np.mean(rel_l2_errors), 
        'std_rel_l2_err': np.std(rel_l2_errors), 
        'max_abs_errors': max_abs_errors, 
        'avg_max_abs_err': np.mean(max_abs_errors), 
        'std_max_abs_err': np.std(max_abs_errors), 
        'relative_l2_grad_errors': rel_l2_errors_grad, 
        'avg_rel_l2_grad_err': np.mean(rel_l2_errors_grad), 
        'std_rel_l2_grad_err': np.std(rel_l2_errors_grad), 
        'max_abs_grad_errors': max_abs_errors_grad, 
        'avg_max_abs_grad_err': np.mean(max_abs_errors_grad), 
        'std_max_abs_grad_err': np.std(max_abs_errors_grad), 
        'lowest_relative_l2_ix': np.argmin(rel_l2_errors)
    }
    return summary


def summarize_results(network_type, n_inputs, n_outputs, n_hidden_list, hidden_width_list, x_eval, ref_sol, ref_grad, repetitions, dir, G=None, k=None):
    assert network_type in ['PINN', 'PIKAN']
    summary = {}
    for n_hidden, hidden_width in zip(n_hidden_list, hidden_width_list):
        model = build_kan(n_inputs, n_outputs, n_hidden, hidden_width, G, k)
        subdir = dir + f"PIKAN/pikan_G{G}_k{k}_hidden{n_hidden}_width{hidden_width}/"
        summary[(n_hidden, hidden_width)] = compute_error_stats(model, x_eval, ref_sol, ref_grad, subdir, repetitions)
        summary[(n_hidden, hidden_width)]["n_params"] = num_parameters_kan(n_inputs, n_outputs, n_hidden, hidden_width, G, k)
    
    os.makedirs(dir + f"{network_type}/", exist_ok=True)
    with open(dir + f"{network_type}/summary.pkl", 'wb') as file:
        pkl.dump(summary, file)


def print_summary(summary):
    for (n_hidden, hidden_width), local_summary in summary.items():
        print(f"N hidden layers: {n_hidden} | Hidden layer width: {hidden_width} | N parameters: {local_summary['n_params']}")
        print(f"    Relative L2 error (mean): {local_summary['avg_rel_l2_err']}%")
        print(f"    Relative L2 error (std): {local_summary['std_rel_l2_err']}%")
        print(f"    Maximum absolute error (mean): {local_summary['avg_max_abs_err']}")
        print(f"    Maximum absolute error (std): {local_summary['std_max_abs_err']}")
        print(f"    Relative L2 gradient error (mean): {local_summary['avg_rel_l2_grad_err']}%")
        print(f"    Relative L2 gradient error (std): {local_summary['std_rel_l2_grad_err']}%")
        print(f"    Maximum absolute gradient error (mean): {local_summary['avg_max_abs_grad_err']}")
        print(f"    Maximum absolute gradient error (std): {local_summary['std_max_abs_grad_err']}")


def smooth_losses(losses, smoothness):
    smoothed_losses = np.copy(losses)
    for i in range(1, smoothed_losses.size):
        smoothed_losses[i] = smoothness * smoothed_losses[i-1] + (1 - smoothness) * smoothed_losses[i]
    return smoothed_losses


# ─────────────────────────────────────────────
# 2. Static Plottings
# ─────────────────────────────────────────────

def plot_results(results: dict, material: MaterialModel, x_fd: np.ndarray,
                 t_fd: np.ndarray, u_fd: np.ndarray, sigma_g: float = 0.1,
                 save_prefix: str = "pinn_1d"):
    arch_names   = list(results.keys())
    arch_colors  = {"vanilla": "#E67E22", "fourier": "#2980B9", "pirate": "#27AE60"}
    arch_labels  = {"vanilla": "Vanilla PINN", "fourier": "Fourier-feature PINN", "pirate": "PirateNet"}

    snap_times_phys = [0.3, 0.6, 0.9]
    snap_times_nd   = [material.to_nondim_t(tp) for tp in snap_times_phys]
    fig, axes = plt.subplots(1, len(snap_times_nd), figsize=(14, 4), sharey=False)
    fig.suptitle(f"{material.name} Model – Displacement Field Traces", fontsize=13)
    x_tensor = torch.tensor(x_fd, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    x_fd_phys = material.to_physical_x(x_fd)

    for col, (t_snap_nd, t_snap_phys) in enumerate(zip(snap_times_nd, snap_times_phys)):
        t_idx   = np.argmin(np.abs(t_fd - t_snap_nd))
        t_actual = t_fd[t_idx]
        u_ref   = u_fd[t_idx]
        axes[col].plot(x_fd_phys, u_ref, "k-", lw=2, label="FD Reference")

        for arch in arch_names:
            model = results[arch]["model"]
            model.eval()
            with torch.no_grad():
                t_t   = torch.full_like(x_tensor, t_actual)
                nn_out = model(x_tensor, t_t)
                u_pred = apply_ansatz(nn_out, x_tensor, t_t, material, sigma_g)
            axes[col].plot(x_fd_phys, u_pred.cpu().numpy().flatten(),
                           color=arch_colors.get(arch, "grey"),
                           lw=1.5, linestyle="--", label=arch_labels.get(arch, arch))

        axes[col].set_title(f"t = {t_snap_phys:.2f} s")
        axes[col].set_xlabel("x")
        if col == 0: 
            axes[col].set_ylabel("u(x,t)")
        axes[col].legend(fontsize=7)
        axes[col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_traces.png", dpi=150, bbox_inches="tight")
    plt.show()

    if "pirate" in results:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"{material.name} – Space-Time Wavefield (PirateNet vs FD)", fontsize=13)
        t_sub_idx = np.linspace(0, len(t_fd) - 1, 200, dtype=int)
        t_sub, u_fd_sub = t_fd[t_sub_idx], u_fd[t_sub_idx]
        model = results["pirate"]["model"]
        model.eval()
        u_pinn_sub = np.zeros_like(u_fd_sub)
        with torch.no_grad():
            for i, t_val in enumerate(t_sub):
                t_t    = torch.full_like(x_tensor, t_val)
                u_pred = apply_ansatz(model(x_tensor, t_t), x_tensor, t_t, material, sigma_g)
                u_pinn_sub[i] = u_pred.cpu().numpy().flatten()

        x_ext = [material.to_physical_x(x_fd[0]), material.to_physical_x(x_fd[-1])]
        t_ext = [material.to_physical_t(t_sub[0]), material.to_physical_t(t_sub[-1])]
        vmax = max(np.abs(u_fd_sub).max(), np.abs(u_pinn_sub).max()) * 0.8
        kw   = dict(aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=[x_ext[0], x_ext[1], t_ext[0], t_ext[1]])
        
        im0 = axes[0].imshow(u_pinn_sub, **kw); axes[0].set_title("PirateNet"); axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
        im1 = axes[1].imshow(u_fd_sub, **kw); axes[1].set_title("FD Reference"); axes[1].set_xlabel("x")
        diff = u_pinn_sub - u_fd_sub
        vd   = np.abs(diff).max() * 0.8
        im2  = axes[2].imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vd, vmax=vd, extent=[x_ext[0], x_ext[1], t_ext[0], t_ext[1]])
        axes[2].set_title("Difference"); axes[2].set_xlabel("x")
        
        for ax, im in zip(axes, [im0, im1, im2]): 
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_heatmap.png", dpi=150, bbox_inches="tight")
        plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    for arch in arch_names:
        ax.semilogy(results[arch]["loss_history"], color=arch_colors.get(arch, "grey"), label=arch_labels.get(arch, arch), linewidth=1.8)
    ax.set_xlabel("Iterations (Adam + L-BFGS)")
    ax.set_ylabel("Total Loss (log scale)")
    ax.set_title(f"{material.name} – Training Loss Convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss.png", dpi=150, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    for arch in arch_names:
        metrics = results[arch]["metrics"]
        ax.plot(material.to_physical_t(metrics["t"]), metrics["energy_diff"], color=arch_colors.get(arch, "grey"), label=arch_labels.get(arch, arch), linewidth=1.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Energy Difference")
    ax.set_title(f"{material.name} – Energy Difference over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_energy.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n{'─'*50}\n  {material.name} – Performance Summary\n{'─'*50}\n  {'Architecture':<25} {'Avg. L2 Error (%)':>18}\n{'─'*50}")
    for arch in arch_names: 
        print(f"  {arch_labels.get(arch, arch):<25} {results[arch]['metrics']['mean_l2']:>18.2f}")
    print(f"{'─'*50}\n")


def plot_losses(n_hidden_pinn_list, hidden_width_pinn_list, best_pinn_ixs, n_hidden_pikan_list, hidden_width_pikan_list, best_pikan_ixs, G, k, results_dir, figsize=(14, 6), smoothness=0.98):
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#ff0000","#00b3ff","#000000","#43b813","#f700ff","#ff8f00","#0000ff","#767676"]
    color_cycle = itertools.cycle(colors)
    ax.set_yscale('log')
    ax.set_xlabel(r'Iteration')
    ax.set_ylabel(r'Total Loss')
    ax.grid(True, which='major', linestyle='solid', color='gainsboro')

    alpha = 0.2
    for n_hidden, hidden_width, best_ix in zip(n_hidden_pinn_list, hidden_width_pinn_list, best_pinn_ixs):
        losses = np.load(results_dir + f"PINN/pinn_hidden{n_hidden}_width{hidden_width}/losses{best_ix+1}.npz")
        ax.plot(losses['total'], color=next(color_cycle), alpha=alpha)

    for n_hidden, hidden_width, best_ix in zip(n_hidden_pikan_list, hidden_width_pikan_list, best_pikan_ixs):
        losses = np.load(results_dir + f"PIKAN/pikan_G{G}_k{k}_hidden{n_hidden}_width{hidden_width}/losses{best_ix+1}.npz")
        ax.plot(losses['total'], color=next(color_cycle), alpha=alpha)

    color_cycle = itertools.cycle(colors)
    for n_hidden, hidden_width, best_ix in zip(n_hidden_pinn_list, hidden_width_pinn_list, best_pinn_ixs):
        losses = np.load(results_dir + f"PINN/pinn_hidden{n_hidden}_width{hidden_width}/losses{best_ix+1}.npz")
        losses = smooth_losses(losses['total'], smoothness)
        ax.plot(losses, label=rf"PINN: $L={n_hidden}$, $W={hidden_width}$", color=next(color_cycle))

    for n_hidden, hidden_width, best_ix in zip(n_hidden_pikan_list, hidden_width_pikan_list, best_pikan_ixs):
        losses = np.load(results_dir + f"PIKAN/pikan_G{G}_k{k}_hidden{n_hidden}_width{hidden_width}/losses{best_ix+1}.npz")
        losses = smooth_losses(losses['total'], smoothness)
        ax.plot(losses, label=rf"PIKAN: $L={n_hidden}$, $W={hidden_width}$", color=next(color_cycle))

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    fig.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(results_dir + "losses_fig.png", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.show()


# ─────────────────────────────────────────────
# 3. Benchmark Specific Visualizations
# ─────────────────────────────────────────────

@torch.no_grad()
def predict_at_time(model, material, x_fd, t_nd, sigma_g_nd):
    """ Prediction at a single non-dimensional time step. """
    x_t = torch.tensor(x_fd, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    t_t = torch.full_like(x_t, float(t_nd))
    u = apply_ansatz(model(x_t, t_t), x_t, t_t, material, sigma_g_nd)
    return u.cpu().numpy().flatten()


@torch.no_grad()
def predict_spacetime(model, material, x_fd, t_arr_nd, sigma_g_nd, batch_size=32):
    """ Batched prediction over space-time grid. """
    nx     = len(x_fd)
    nt     = len(t_arr_nd)
    u_grid = np.zeros((nt, nx), dtype=np.float32)
    x_base = torch.tensor(x_fd, dtype=torch.float32, device=DEVICE)

    for start in range(0, nt, batch_size):
        end     = min(start + batch_size, nt)
        t_batch = t_arr_nd[start:end]
        B       = len(t_batch)

        x_rep = x_base.unsqueeze(0).expand(B, nx).reshape(-1, 1)
        t_rep = torch.tensor(t_batch, dtype=torch.float32, device=DEVICE).unsqueeze(1).expand(B, nx).reshape(-1, 1)

        out = model(x_rep, t_rep)
        u   = apply_ansatz(out, x_rep, t_rep, material, sigma_g_nd)
        u_grid[start:end] = u.reshape(B, nx).cpu().numpy()

    return u_grid


def plot_traces(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir):
    snap_times_nd = [0.3 * t_fd_nd[-1], 0.6 * t_fd_nd[-1], 0.9 * t_fd_nd[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{material.name} - Displacement Traces", fontsize=14)

    for col, t_nd in enumerate(snap_times_nd):
        idx = np.argmin(np.abs(t_fd_nd - t_nd))
        t_actual_nd = t_fd_nd[idx]
        t_phys = material.to_physical_t(t_actual_nd)
        
        axes[col].plot(x_fd, u_fd[idx], "k-", lw=2.5, label="FD Reference", zorder=5)
        for arch, model in models.items():
            u_p = predict_at_time(model, material, x_fd, t_actual_nd, sigma_g_nd)
            axes[col].plot(x_fd, u_p, color=ARCH_COLORS[arch], lw=1.8, ls="--", label=ARCH_LABELS[arch])
            
        axes[col].set_title(f"t = {t_phys:.2f} s")
        axes[col].set_xlabel("x (non-dim)")
        if col == 0: 
            axes[col].set_ylabel("u(x,t)")
        axes[col].legend(fontsize=8)
        axes[col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_traces.png"), dpi=150)
    plt.close()


def plot_final_comparison(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir):
    snap_times_nd = [0.4 * t_fd_nd[-1], 0.7 * t_fd_nd[-1], 1.0 * t_fd_nd[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{material.name} - Key Snapshots (Middle & End)", fontsize=14)

    for col, t_nd in enumerate(snap_times_nd):
        idx = np.argmin(np.abs(t_fd_nd - t_nd))
        t_actual_nd = t_fd_nd[idx]
        t_phys = material.to_physical_t(t_actual_nd)
        
        axes[col].plot(x_fd, u_fd[idx], "k-", lw=2.5, label="FD Reference", zorder=5)
        for arch, model in models.items():
            u_p = predict_at_time(model, material, x_fd, t_actual_nd, sigma_g_nd)
            axes[col].plot(x_fd, u_p, color=ARCH_COLORS[arch], lw=1.8, ls="--", label=ARCH_LABELS[arch])
            
        axes[col].set_title(f"t = {t_phys:.2f} s")
        axes[col].set_xlabel("x (non-dim)")
        if col == 0: 
            axes[col].set_ylabel("u(x,t)")
        axes[col].legend(fontsize=8)
        axes[col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_final_comparison.png"), dpi=150)
    plt.close()


def plot_wavefield(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir):
    t_sub_idx = np.linspace(0, len(t_fd_nd)-1, 200, dtype=int)
    t_sub_nd  = t_fd_nd[t_sub_idx]
    u_fd_sub  = u_fd[t_sub_idx]
    
    t_phys_sub = material.to_physical_t(t_sub_nd)
    ext = [x_fd[0], x_fd[-1], t_phys_sub[0], t_phys_sub[-1]]
    vmax = np.abs(u_fd_sub).max() * 0.85

    n_arch = len(models)
    fig, axes = plt.subplots(n_arch, 3, figsize=(14, 4*n_arch), squeeze=False)
    fig.suptitle(f"{material.name} - Space-Time Heatmaps", fontsize=14)

    for row, (arch, model) in enumerate(models.items()):
        u_p = predict_spacetime(model, material, x_fd, t_sub_nd, sigma_g_nd)
        diff = u_p - u_fd_sub
        
        vd = np.abs(diff).max() * 0.8
        kw  = dict(aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=ext)
        kwd = dict(aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vd, vmax=vd, extent=ext)

        for ax, data, k, title in zip(axes[row], [u_p, u_fd_sub, diff], [kw, kw, kwd], 
                                      [ARCH_LABELS[arch], "FD Reference", "Difference"]):
            im = ax.imshow(data, **k)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("x")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[row, 0].set_ylabel("t (s)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_heatmaps.png"), dpi=150)
    plt.close()


def plot_metrics(metrics_dict, material, prefix, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{material.name} - Error Metrics over Time", fontsize=14)

    for arch, m in metrics_dict.items():
        c, l = ARCH_COLORS[arch], ARCH_LABELS[arch]
        t_phys = material.to_physical_t(m["t"])
        axes[0].plot(t_phys, m["energy_diff"], color=c, label=l, lw=1.8)
        axes[1].plot(t_phys, m["l2_errors"],   color=c, label=l, lw=1.8)

    for ax, ylabel, title in zip(
            axes,
            ["Normalized Energy Difference", "Relative L2 Error (%)"],
            ["Energy Difference", "L2 Error"]):
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_metrics.png"), dpi=150)
    plt.close()


def plot_residual_map(models, material, sigma_g_nd, prefix, save_dir):
    nx_r, nt_r = 80, 80
    x_r = np.linspace(material.x_min, material.x_max, nx_r)
    
    # Self-contained time limit configurations
    t_max_config = {
        "exp1_homogeneous": 1.0,
        "exp2_twolayer":    1.0,
        "exp3_multilayer":  1.0,
    }
    t_max = t_max_config.get(prefix, 1.0)
    t_r = np.linspace(0.0, material.to_nondim_t(t_max), nt_r)

    XX, TT = np.meshgrid(x_r, t_r)

    x_t = torch.tensor(XX.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(1)
    t_t = torch.tensor(TT.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(1)

    n_arch = len(models)
    fig, axes = plt.subplots(1, n_arch, figsize=(5*n_arch, 4), squeeze=False)
    fig.suptitle(f"{material.name} - PDE Residual |R(x,t)|", fontsize=14)

    for col, (arch, model) in enumerate(models.items()):
        if arch.startswith("piann"):
            # PIANN's pointwise forward uses torch.unique which breaks autograd
            res = np.zeros((nt_r, nx_r))
        else:
            with torch.enable_grad():
                xt = x_t.detach().requires_grad_(True)
                tt = t_t.detach().requires_grad_(True)
                R  = compute_pde_residual(model, xt, tt, material, sigma_g_nd)
            res = R.abs().detach().cpu().numpy().reshape(nt_r, nx_r)
        
        vmax = np.percentile(res, 97) if res.size > 0 else 1.0
        if vmax == 0: 
            vmax = 1.0
        im = axes[0, col].imshow(res, aspect="auto", origin="lower",
                                  cmap="hot_r", vmin=0, vmax=vmax,
                                  extent=[material.x_min, material.x_max, 0, material.to_physical_t(t_r[-1])])
        axes[0, col].set_title(ARCH_LABELS[arch], fontsize=10)
        axes[0, col].set_xlabel("x")
        if col == 0: 
            axes[0, col].set_ylabel("t (s)")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_residual_map.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# 4. Benchmarking Animations (GIF compilations)
# ─────────────────────────────────────────────

def animate_results(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir, n_frames=120, fps=30):
    arch_names = list(models.keys())
    t_idx = np.linspace(0, len(t_fd_nd)-1, n_frames, dtype=int)
    t_frames_nd = t_fd_nd[t_idx]
    u_fd_frames = u_fd[t_idx]
    
    print(f"  Animating {prefix}...")
    pinn_frames = {a: predict_spacetime(m, material, x_fd, t_frames_nd, sigma_g_nd) for a, m in models.items()}
    
    n_cols = len(arch_names) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 4), sharey=True)
    
    all_u = np.concatenate([u_fd_frames] + list(pinn_frames.values()))
    ymin, ymax = all_u.min() * 1.2, all_u.max() * 1.2

    lines = []
    for i in range(n_cols):
        ax = axes[i]
        ax.set_xlim(x_fd[0], x_fd[-1])
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.2)
        
        if i < len(arch_names):
            arch = arch_names[i]
            l_p, = ax.plot([], [], color=ARCH_COLORS[arch], lw=2, label=ARCH_LABELS[arch])
            l_r, = ax.plot([], [], "k--", lw=1, alpha=0.5, label="Ref")
            ax.set_title(ARCH_LABELS[arch])
            ax.legend(loc="upper right", fontsize=8)
            lines.append((l_p, l_r))
        else:
            l_fd, = ax.plot([], [], "k-", lw=2, label="FD Reference")
            ax.set_title("FD Reference")
            ax.legend(loc="upper right", fontsize=8)
            lines.append((l_fd,))

    time_text = fig.text(0.5, 0.95, "", ha="center", fontsize=12)

    def init():
        for g in lines:
            for l in g: 
                l.set_data([], [])
        time_text.set_text("")
        return [l for g in lines for l in g] + [time_text]

    def update(fi):
        for i in range(n_cols):
            if i < len(arch_names):
                lines[i][0].set_data(x_fd, pinn_frames[arch_names[i]][fi])
                lines[i][1].set_data(x_fd, u_fd_frames[fi])
            else:
                lines[i][0].set_data(x_fd, u_fd_frames[fi])
        t_phys = material.to_physical_t(t_frames_nd[fi])
        time_text.set_text(f"t = {t_phys:.3f} s")
        return [l for g in lines for l in g] + [time_text]

    anim = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=1000/fps)
    
    out_path = os.path.join(save_dir, f"{prefix}_animation.gif")
    anim.save(out_path, writer='pillow', fps=fps)
    print(f"  [OK] Saved animation: {out_path}")
    plt.close()


def animate_error(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir, n_frames=120, fps=30):
    arch_names = list(models.keys())
    t_idx = np.linspace(0, len(t_fd_nd)-1, n_frames, dtype=int)
    t_frames_nd = t_fd_nd[t_idx]
    u_fd_frames = u_fd[t_idx]

    print(f"  Animating error for {prefix}...")
    err_frames = {}
    for arch, model in models.items():
        u_p = predict_spacetime(model, material, x_fd, t_frames_nd, sigma_g_nd)
        err_frames[arch] = np.abs(u_p - u_fd_frames)

    emax = 0
    for e in err_frames.values():
        if e.size > 0: 
            emax = max(emax, e.max())
    global_emax = emax * 0.9 if emax > 0 else 1.0

    n_arch = len(arch_names)
    fig, axes = plt.subplots(1, n_arch, figsize=(5*n_arch, 3.5), sharey=True, squeeze=False)
    fig.suptitle(f"{material.name} - Pointwise Error |u_pred - u_FD|", fontsize=12)

    images = []
    for col, arch in enumerate(arch_names):
        im = axes[0, col].imshow(
            err_frames[arch][[0], :],
            aspect="auto", cmap="hot",
            vmin=0, vmax=global_emax,
            extent=[x_fd[0], x_fd[-1], 0, material.to_physical_t(t_fd_nd[-1])])
        axes[0, col].set_title(ARCH_LABELS[arch], fontsize=9)
        axes[0, col].set_xlabel("x")
        axes[0, col].set_yticks([])
        plt.colorbar(im, ax=axes[0, col], fraction=0.05, pad=0.04)
        images.append(im)

    time_text = fig.text(0.5, 0.95, "", ha="center", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    def init():
        for im, arch in zip(images, arch_names):
            im.set_data(err_frames[arch][[0], :])
        time_text.set_text("")
        return images + [time_text]

    def update(fi):
        for im, arch in zip(images, arch_names):
            im.set_data(err_frames[arch][[fi], :])
        t_phys = material.to_physical_t(t_frames_nd[fi])
        time_text.set_text(f"t = {t_phys:.3f} s")
        return images + [time_text]

    anim = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=1000/fps)

    out_path = os.path.join(save_dir, f"{prefix}_error_animation.gif")
    anim.save(out_path, writer='pillow', fps=fps)
    print(f"  [OK] Saved error animation: {out_path}")
    plt.close()
