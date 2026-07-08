import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Prevent plot windows from popping up
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display

from training_code import (
    DEVICE, SEED,
    HomogeneousModel, TwoLayerModel, MultiLayerModel,
    VanillaPINN, FourierFeaturePINN, PirateNet,
    apply_ansatz, compute_pde_residual, fd_reference, evaluate
)
from src.models import FNOWrapper

try:
    from kan import KAN
except ImportError:
    KAN = None

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_DIR     = "Models"
BASE_SAVE_DIR  = "results"
SIGMA_G_PHYS   = 0.1
N_FRAMES       = 120
FPS            = 30

# Which architectures to benchmark
RUN_ARCHS = [#"vanilla",
             #"fourier",
             #"pirate",
             #"piann_fd",
             #"piann_ad",
             #"pikan",
             "fno",
            ]



EXPERIMENTS = {
    "exp1_homogeneous": HomogeneousModel(),
    #"exp2_twolayer":    TwoLayerModel(),
    #"exp3_multilayer":  MultiLayerModel(),
}


# ===============================================================================
# HELPERS
# ===============================================================================
ARCH_LABELS = {
    "vanilla": "Vanilla PINN",
    "fourier": "Fourier-feature PINN",
    "pirate":  "PirateNet",
    "piann_fd": "PIANN (FD)",
    "piann_ad": "PIANN (AD)",
    "pikan": "PIKAN",
    "fno": "FNO"
}
T_MAX_DICT = {
    "exp1_homogeneous": 0.75,
    "exp2_twolayer":    1.0,
    "exp3_multilayer":  1.0,
}
ARCH_COLORS = {
    "vanilla": "#E67E22",
    "fourier": "#2980B9",
    "pirate":  "#27AE60",
    "piann_fd": "#C0392B",
    "piann_ad": "#8E44AD",
    "pikan": "#1ABC9C",
    "fno": "#F1C40F"
}
def load_models(prefix):
    """
    Robust model loader that infers architecture params from checkpoint keys.
    """
    models = {}
    
    # Check for possible architectures
    for arch in RUN_ARCHS:
        # Try different extensions and suffixes
        found_path = None
        for ext in [".pt", ".pkl"]:
            for suffix in ["_model", "_best"]:
                path = os.path.join(MODEL_DIR, f"{prefix}_{arch}{suffix}{ext}")
                if os.path.exists(path):
                    found_path = path
                    break
            if found_path: break
        
        if not found_path:
            continue

        sd = torch.load(found_path, map_location=DEVICE, weights_only=False)
        
        try:
            if arch == "vanilla":
                units = sd['net.0.weight'].shape[0]
                n_layers = sum(1 for k in sd.keys() if 'net.' in k and '.weight' in k) - 1
                model = VanillaPINN(hidden_layers=n_layers, hidden_units=units).to(DEVICE)
            
            elif arch == "fourier":
                n_fourier = sd['B'].shape[0]
                units = sd['net.0.weight'].shape[0]
                n_layers = sum(1 for k in sd.keys() if 'net.' in k and '.weight' in k) - 1
                model = FourierFeaturePINN(hidden_layers=n_layers, hidden_units=units, n_fourier=n_fourier).to(DEVICE)
            
            elif arch == "pirate":
                n_fourier = sd['B'].shape[0]
                is_rwf = 'enc_U.V' in sd
                units = sd['enc_U.V'].shape[0] if is_rwf else sd['enc_U.0.weight'].shape[0]
                block_ids = [int(k.split('.')[1]) for k in sd.keys() if k.startswith('blocks.')]
                n_blocks = max(block_ids) + 1 if block_ids else 0
                model = PirateNet(n_blocks=n_blocks, units=units, n_fourier=n_fourier).to(DEVICE)
                
                # Conversion logic for legacy standard Linear weights to RWFLinear
                if not is_rwf:
                    print(f"  [PirateNet] Converting legacy weights for {prefix}...")
                    new_sd = {}
                    for k, v in sd.items():
                        if any(x in k for x in ['enc_U', 'enc_V', 'proj', 'blocks', 'out_layer']) and '.weight' in k:
                            prefix_key = k.replace('.weight', '')
                            new_sd[f"{prefix_key}.V"] = v
                            new_sd[f"{prefix_key}.s"] = torch.ones(v.shape[0], device=DEVICE)
                        else:
                            new_sd[k] = v
                    sd = new_sd
            
            elif arch == "pikan":
                if KAN is None:
                    print(f"  [Skip] pykan library not found. Cannot load {arch}")
                    continue
                
                # Infer grid and k from state dict to avoid size mismatch
                # coef shape is (in, out, grid + k)
                # grid shape is (in, grid + 2k + 1)
                # (grid + 2k + 1) - (grid + k) = k + 1
                try:
                    grid_key = next(k for k in sd.keys() if 'grid' in k)
                    coef_key = next(k for k in sd.keys() if 'coef' in k)
                    g_len = sd[grid_key].shape[-1]
                    c_len = sd[coef_key].shape[-1]
                    k_val = g_len - c_len - 1
                    grid_val = c_len - k_val
                    print(f"  [pikan] Inferred grid={grid_val}, k={k_val}")
                except Exception:
                    grid_val, k_val = 5, 3 # Fallback to defaults
                
                # Architecture specified by user: 2 5 5 5 1
                kan_model = KAN(width=[2, 5, 5, 5, 5, 1], grid=grid_val, k=k_val).to(DEVICE)
                
                # Robust loading: handles state_dict or full model object
                if isinstance(sd, dict):
                    kan_model.load_state_dict(sd)
                elif hasattr(sd, 'state_dict'):
                    kan_model.load_state_dict(sd.state_dict())
                else:
                    print(f"  [Warn] Unknown model format for {arch}")
                    kan_model = sd # Try using it directly if it's already a model
                
                # Wrapper to match the (x, t) signature used in the benchmark
                class KANWrapper(nn.Module):
                    def __init__(self, k_model):
                        super().__init__()
                        self.k_model = k_model
                    def forward(self, x, t):
                        # pykan KAN expects a single input tensor (batch, 2)
                        return self.k_model(torch.cat([x, t], dim=-1))
                
                model = KANWrapper(kan_model)
                model.eval()
                models[arch] = model
                print(f"  [OK] Loaded {arch} model (pykan 2-5-5-5-1)")
                continue
            
            elif arch == "fno":
                width = sd['fno.fc0.weight'].shape[0]
                modes = sd['fno.spectral_layers.0.weight_real'].shape[-1]
                layer_ids = [int(k.split('.')[2]) for k in sd.keys()
                             if k.startswith('fno.spectral_layers.') and k.endswith('.weight_real')]
                n_layers = max(layer_ids) + 1
                nx = sd['x_grid'].shape[0]
                model = FNOWrapper(nx=nx, modes=modes, width=width, n_layers=n_layers).to(DEVICE)

            elif arch.startswith("piann"):
                # We assume standard 128 hidden size for PIANN benchmarks unless specified
                model = PIANN(x_min=-1.0, x_max=1.0, N=128, hidden_size=128).to(DEVICE)

            model.load_state_dict(sd)
            model.eval()
            models[arch] = model
            print(f"  [OK] Loaded {arch} model")
            
        except Exception as e:
            print(f"  ! Failed to load {arch} from {path}: {e}")

    return models

@torch.no_grad()
def predict_at_time(model, material, x_fd, t_nd, sigma_g_nd):
    """
    Prediction at a single non-dimensional time step.
    """
    x_t = torch.tensor(x_fd, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    t_t = torch.full_like(x_t, float(t_nd))
    u = apply_ansatz(model(x_t, t_t), x_t, t_t, material, sigma_g_nd)
    return u.cpu().numpy().flatten()

@torch.no_grad()
def predict_spacetime(model, material, x_fd, t_arr_nd, sigma_g_nd, batch_size=32):
    """
    Batched prediction over space-time grid.
    """
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

# ===============================================================================
# VISUALIZATION
# ===============================================================================

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
        if col == 0: axes[col].set_ylabel("u(x,t)")
        axes[col].legend(fontsize=8)
        axes[col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_traces.png"), dpi=150)
    plt.close()

def plot_final_comparison(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir):
    # 2 middle and 1 end
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
        if col == 0: axes[col].set_ylabel("u(x,t)")
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
    t_r = np.linspace(0.0, material.to_nondim_t(T_MAX_DICT.get(prefix, 1.0)), nt_r)
    XX, TT = np.meshgrid(x_r, t_r)

    x_t = torch.tensor(XX.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(1)
    t_t = torch.tensor(TT.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(1)

    n_arch = len(models)
    fig, axes = plt.subplots(1, n_arch, figsize=(5*n_arch, 4), squeeze=False)
    fig.suptitle(f"{material.name} - PDE Residual |R(x,t)|", fontsize=14)

    for col, (arch, model) in enumerate(models.items()):
        if arch.startswith("piann") or arch == "fno":
            # PIANN's pointwise forward uses torch.unique which breaks autograd;
            # FNO's grid-interpolating forward is not smoothly differentiable in x either.
            res = np.zeros((nt_r, nx_r))
        else:
            with torch.enable_grad():
                xt = x_t.detach().requires_grad_(True)
                tt = t_t.detach().requires_grad_(True)
                R  = compute_pde_residual(model, xt, tt, material, sigma_g_nd)
            res = R.abs().detach().cpu().numpy().reshape(nt_r, nx_r)
        
        vmax = np.percentile(res, 97) if res.size > 0 else 1.0
        if vmax == 0: vmax = 1.0
        im = axes[0, col].imshow(res, aspect="auto", origin="lower",
                                  cmap="hot_r", vmin=0, vmax=vmax,
                                  extent=[material.x_min, material.x_max, 0, material.to_physical_t(t_r[-1])])
        axes[0, col].set_title(ARCH_LABELS[arch], fontsize=10)
        axes[0, col].set_xlabel("x")
        if col == 0: axes[0, col].set_ylabel("t (s)")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_residual_map.png"), dpi=150)
    plt.close()

def animate_results(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir):
    """
    Creates the final wavefield animation.
    """
    arch_names = list(models.keys())
    t_idx = np.linspace(0, len(t_fd_nd)-1, N_FRAMES, dtype=int)
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
            for l in g: l.set_data([], [])
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

    anim = animation.FuncAnimation(fig, update, frames=N_FRAMES, init_func=init, blit=True, interval=1000/FPS)
    
    out_path = os.path.join(save_dir, f"{prefix}_animation.gif")
    anim.save(out_path, writer='pillow', fps=FPS)
    print(f"  ✓ Saved animation: {out_path}")
    plt.close()

def animate_error(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir):
    arch_names = list(models.keys())
    t_idx = np.linspace(0, len(t_fd_nd)-1, N_FRAMES, dtype=int)
    t_frames_nd = t_fd_nd[t_idx]
    u_fd_frames = u_fd[t_idx]

    print(f"  Animating error for {prefix}...")
    err_frames = {}
    for arch, model in models.items():
        u_p = predict_spacetime(model, material, x_fd, t_frames_nd, sigma_g_nd)
        err_frames[arch] = np.abs(u_p - u_fd_frames)

    emax = 0
    for e in err_frames.values():
        if e.size > 0: emax = max(emax, e.max())
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

    anim = animation.FuncAnimation(fig, update, frames=N_FRAMES, init_func=init, blit=True, interval=1000/FPS)

    out_path = os.path.join(save_dir, f"{prefix}_error_animation.gif")
    anim.save(out_path, writer='pillow', fps=FPS)
    print(f"  ✓ Saved error animation: {out_path}")
    plt.close()

# ===============================================================================
# MAIN
# ===============================================================================

if __name__ == "__main__":
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    
    for prefix, material in EXPERIMENTS.items():
        print(f"\n{'='*60}\n  Experiment: {material.name} ({prefix})\n{'='*60}")
        
        # 1. Load models
        models = load_models(prefix)
        if not models:
            print(f"  ! No models found for {prefix}, skipping.")
            continue
            
        # 2. Configure scales
        T_phys      = T_MAX_DICT.get(prefix, 1.0)
        T_nd        = material.to_nondim_t(T_phys)
        sigma_g_nd  = material.nondim_sigma_g(SIGMA_G_PHYS)
        
        # 3. Reference solution
        print(f"  Generating reference (T_nd={T_nd:.3f}, sigma_nd={sigma_g_nd:.3f})...")
        x_fd, t_fd_nd, u_fd = fd_reference(material, nx=512, nt=2000, T=T_nd, sigma_g=sigma_g_nd)
        
        # 4. Evaluation
        eval_phys  = list(np.linspace(0.05, T_phys, 20))
        eval_nd    = [material.to_nondim_t(tp) for tp in eval_phys]
        
        results_summary = {}
        print(f"\n  {'Architecture':<25} {'L2 Error (%)':>15}")
        print(f"  {'-'*40}")
        
        for arch, model in models.items():
            m = evaluate(model, material, x_fd, t_fd_nd, u_fd, sigma_g=sigma_g_nd, eval_times=eval_nd)
            results_summary[arch] = m
            print(f"  {ARCH_LABELS[arch]:<25} {m['mean_l2']:>15.2f}%")
            
        # 5. Output
        save_dir = os.path.join(BASE_SAVE_DIR, prefix.split('_')[0])
        os.makedirs(save_dir, exist_ok=True)
        
        plot_traces(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir)
        plot_final_comparison(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir)
        plot_wavefield(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir)
        plot_metrics(results_summary, material, prefix, save_dir)
        plot_residual_map(models, material, sigma_g_nd, prefix, save_dir)
        
        animate_results(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir)
        animate_error(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir)

    print(f"\n* All experiments complete. Results in {BASE_SAVE_DIR}/")
