import os, sys

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display

from pinn_1d_elastic_wave import (
    DEVICE, SEED,
    HomogeneousModel, TwoLayerModel, MultiLayerModel,
    VanillaPINN, FourierFeaturePINN, PirateNet,
    apply_ansatz, compute_pde_residual,
    fd_reference, evaluate,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — change nothing below unless you retrained with different hyper-params
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_DIR  = "Models"
BASE_SAVE_DIR = "results"
T_MAX_DICT = {
    "exp1_homogeneous": 0.75,
    "exp2_twolayer": 1.0,
    "exp3_multilayer": 1.0,
}
SIGMA_G    = 0.1
N_FRAMES   = 120
FPS        = 30

EXPERIMENTS = {
    "exp1_homogeneous": HomogeneousModel(),
    "exp2_twolayer":    TwoLayerModel(),
    "exp3_multilayer":  MultiLayerModel(),
}

ARCH_BUILDERS = {
    "vanilla": lambda: VanillaPINN(hidden_layers=5, hidden_units=128),
    "fourier":  lambda: FourierFeaturePINN(hidden_layers=5, hidden_units=128,
                                            n_fourier=128, sigma=1.0),
    "pirate":   lambda: PirateNet(n_blocks=3, units=128, n_fourier=64, sigma=2.0),
}

ARCH_COLORS = {"vanilla": "#E67E22", "fourier": "#2980B9", "pirate": "#27AE60"}
ARCH_LABELS = {"vanilla": "Vanilla PINN", "fourier": "Fourier-feature PINN",
               "pirate":  "PirateNet"}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_models(prefix):
    models = {}
    for arch, build_fn in ARCH_BUILDERS.items():
        path = os.path.join(MODEL_DIR, f"{prefix}_{arch}_model.pt")
        model = build_fn().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        models[arch] = model
        print(f"  ✓ Loaded {path}")
    return models


@torch.no_grad()
def predict_at_time(model, x_fd, t_val):
    x_t = torch.tensor(x_fd, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    t_t = torch.full_like(x_t, float(t_val))
    return apply_ansatz(model(x_t, t_t), x_t, t_t, SIGMA_G).cpu().numpy().flatten()


@torch.no_grad()
def predict_spacetime(model, x_fd, t_arr, batch_size=32):
    """
    Batched GPU prediction over a (nt, nx) grid.
    Tiles x for each t in a batch so the GPU stays fully utilised
    instead of launching one tiny kernel per time step.
    """
    nx     = len(x_fd)
    nt     = len(t_arr)
    u_grid = np.zeros((nt, nx), dtype=np.float32)
    x_base = torch.tensor(x_fd, dtype=torch.float32, device=DEVICE)  # (nx,)

    for start in range(0, nt, batch_size):
        end     = min(start + batch_size, nt)
        t_batch = t_arr[start:end]   # (B,)
        B       = len(t_batch)

        # shape (B*nx, 1) — repeat x for every t in the batch
        x_rep = x_base.unsqueeze(0).expand(B, nx).reshape(-1, 1)
        # shape (B*nx, 1) — repeat each t value nx times
        t_rep = torch.tensor(t_batch, dtype=torch.float32, device=DEVICE) \
                     .unsqueeze(1).expand(B, nx).reshape(-1, 1)

        out = model(x_rep, t_rep)
        u   = apply_ansatz(out, x_rep, t_rep, SIGMA_G)
        u_grid[start:end] = u.reshape(B, nx).cpu().numpy()

    return u_grid


def print_table(metrics_dict, material_name):
    print(f"\n{'─'*58}")
    print(f"  {material_name} — Performance Summary")
    print(f"{'─'*58}")
    print(f"  {'Architecture':<26} {'Avg L2 Error (%)':>16}  {'Avg Energy Diff':>15}")
    print(f"{'─'*58}")
    for arch, m in metrics_dict.items():
        print(f"  {ARCH_LABELS[arch]:<26} {m['mean_l2']:>16.2f}  {m['energy_diff'].mean():>15.5f}")
    print(f"{'─'*58}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Displacement Traces
# ═══════════════════════════════════════════════════════════════════════════════

def plot_traces(models, material, x_fd, t_fd, u_fd, prefix):
    snap_times = [0.3 * T_MAX, 0.6 * T_MAX, 0.9 * T_MAX]
    fig, axes  = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"{material.name} — Displacement Traces", fontsize=13)

    for col, t_snap in enumerate(snap_times):
        idx  = np.argmin(np.abs(t_fd - t_snap))
        t_ac = t_fd[idx]
        axes[col].plot(x_fd, u_fd[idx], "k-", lw=2.5, label="FD Reference", zorder=5)
        for arch, model in models.items():
            axes[col].plot(x_fd, predict_at_time(model, x_fd, t_ac),
                           color=ARCH_COLORS[arch], lw=1.8, ls="--",
                           label=ARCH_LABELS[arch])
        axes[col].set_title(f"t = {t_ac:.2f} s")
        axes[col].set_xlabel("x")
        if col == 0: axes[col].set_ylabel("u(x,t)")
        axes[col].legend(fontsize=7.5)
        axes[col].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, f"{prefix}_traces.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Space-Time Wavefield  (Fig. 7 / 10 / 14 analogue)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_wavefield(models, material, x_fd, t_fd, u_fd, prefix):
    t_sub_idx = np.linspace(0, len(t_fd)-1, 200, dtype=int)
    t_sub     = t_fd[t_sub_idx]
    u_sub     = u_fd[t_sub_idx]
    ext       = [x_fd[0], x_fd[-1], t_sub[0], t_sub[-1]]
    vmax      = np.abs(u_sub).max() * 0.85

    n_arch = len(models)
    fig, axes = plt.subplots(n_arch, 3, figsize=(13, 4*n_arch), squeeze=False)
    fig.suptitle(f"{material.name} — Space-Time Wavefields", fontsize=13)

    for row, (arch, model) in enumerate(models.items()):
        u_p  = predict_spacetime(model, x_fd, t_sub)
        diff = u_p - u_sub
        valid_diff = diff[~np.isnan(diff) & ~np.isinf(diff)]
        vd   = np.abs(valid_diff).max() * 0.85 if len(valid_diff) > 0 else 1.0
        if vd == 0: vd = 1.0
        kw   = dict(aspect="auto", origin="lower", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, extent=ext)
        kwd  = dict(aspect="auto", origin="lower", cmap="RdBu_r",
                    vmin=-vd, vmax=vd, extent=ext)

        for ax, data, kw_, title in zip(
                axes[row],
                [u_p, u_sub, diff],
                [kw, kw, kwd],
                [ARCH_LABELS[arch], "FD Reference", "Difference"]):
            im = ax.imshow(data, **kw_)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("x")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[row, 0].set_ylabel("t")

    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, f"{prefix}_wavefield.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Energy Difference + L2 Error  (Fig. 8a analogue)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_metrics(metrics_dict, material, prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{material.name} — Error Metrics over Time", fontsize=13)

    for arch, m in metrics_dict.items():
        c, l = ARCH_COLORS[arch], ARCH_LABELS[arch]
        axes[0].plot(m["t"], m["energy_diff"], color=c, label=l, lw=1.8)
        axes[1].plot(m["t"], m["l2_errors"],   color=c, label=l, lw=1.8)

    for ax, ylabel, title in zip(
            axes,
            ["Normalized Energy Difference", "Relative L₂ Error (%)"],
            ["Energy Difference", "L₂ Error"]):
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, f"{prefix}_metrics.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — PDE Residual Map  (diagnostic)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_residual_map(models, material, prefix):
    nx_r, nt_r = 80, 80
    x_r = np.linspace(material.x_min, material.x_max, nx_r)
    t_r = np.linspace(0.0, T_MAX, nt_r)
    XX, TT = np.meshgrid(x_r, t_r)

    x_t = torch.tensor(XX.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(1)
    t_t = torch.tensor(TT.flatten(), dtype=torch.float32, device=DEVICE).unsqueeze(1)

    n_arch = len(models)
    fig, axes = plt.subplots(1, n_arch, figsize=(5*n_arch, 4), squeeze=False)
    fig.suptitle(f"{material.name} — PDE Residual |R(x,t)|", fontsize=13)

    for col, (arch, model) in enumerate(models.items()):
        with torch.enable_grad():
            xt = x_t.detach().requires_grad_(True)
            tt = t_t.detach().requires_grad_(True)
            R  = compute_pde_residual(model, xt, tt, material, SIGMA_G)
        res = R.abs().detach().cpu().numpy().reshape(nt_r, nx_r)
        valid_res = res[~np.isnan(res) & ~np.isinf(res)]
        vmax = np.percentile(valid_res, 97) if len(valid_res) > 0 else 1.0
        if vmax == 0: vmax = 1.0
        im = axes[0, col].imshow(res, aspect="auto", origin="lower",
                                  cmap="hot_r", vmin=0, vmax=vmax,
                                  extent=[material.x_min, material.x_max, 0, T_MAX])
        axes[0, col].set_title(ARCH_LABELS[arch], fontsize=10)
        axes[0, col].set_xlabel("x")
        if col == 0: axes[0, col].set_ylabel("t")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, f"{prefix}_residual_map.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANIMATION 1 — Wavefield travelling wave
# ═══════════════════════════════════════════════════════════════════════════════

def animate_wavefield(models, material, x_fd, t_fd, u_fd, prefix):
    arch_names  = list(models.keys())
    t_idx       = np.linspace(0, len(t_fd)-1, N_FRAMES, dtype=int)
    t_frames    = t_fd[t_idx]
    u_fd_frames = u_fd[t_idx]

    print("  Pre-computing wavefield frames...")
    pinn_frames = {a: predict_spacetime(m, x_fd, t_frames)
                   for a, m in models.items()}

    all_u = np.concatenate([u_fd_frames] + list(pinn_frames.values()))
    valid_u = all_u[~np.isnan(all_u) & ~np.isinf(all_u)]
    if len(valid_u) == 0:
        ymin, ymax = -1.0, 1.0
    else:
        ymin, ymax = valid_u.min() * 1.2, valid_u.max() * 1.2
        if ymin == ymax:
            ymin, ymax = ymin - 1.0, ymax + 1.0

    n_cols = len(arch_names) + 1          # PINNs + FD
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5*n_cols, 4), sharey=True)
    fig.suptitle(f"{material.name} — Wavefield Animation", fontsize=12)

    # Material background shading (compute E on GPU, pull to CPU for matplotlib)
    x_t  = torch.tensor(x_fd, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    E_np = material.E(x_t).cpu().numpy().flatten()
    E_n  = (E_np - E_np.min()) / (E_np.max() - E_np.min() + 1e-12)

    lines = []
    for col in range(n_cols):
        ax = axes[col]
        ax.fill_between(x_fd, ymin, ymin + (ymax-ymin)*E_n*0.15,
                        alpha=0.15, color="steelblue")
        ax.set_xlim(x_fd[0], x_fd[-1])
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.25)

        if col < len(arch_names):
            arch = arch_names[col]
            ln_p, = ax.plot([], [], color=ARCH_COLORS[arch], lw=2.0,
                            label=ARCH_LABELS[arch])
            ln_r, = ax.plot([], [], "k--", lw=1.2, alpha=0.5, label="FD Ref")
            ax.set_title(ARCH_LABELS[arch], fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
            lines.append((ln_p, ln_r))
        else:
            ln_fd, = ax.plot([], [], "k-", lw=2.0, label="FD Reference")
            ax.set_title("FD Reference", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
            lines.append((ln_fd,))

    axes[0].set_ylabel("u(x,t)")
    time_text = fig.text(0.5, 0.97, "", ha="center", fontsize=10, color="dimgray")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    def init():
        for grp in lines:
            for ln in grp: ln.set_data([], [])
        time_text.set_text("")
        return [ln for grp in lines for ln in grp] + [time_text]

    def update(fi):
        for col in range(n_cols):
            if col < len(arch_names):
                lines[col][0].set_data(x_fd, pinn_frames[arch_names[col]][fi])
                lines[col][1].set_data(x_fd, u_fd_frames[fi])
            else:
                lines[col][0].set_data(x_fd, u_fd_frames[fi])
        time_text.set_text(f"t = {t_frames[fi]:.3f} s")
        return [ln for grp in lines for ln in grp] + [time_text]

    anim = animation.FuncAnimation(fig, update, frames=N_FRAMES,
                                   init_func=init, blit=True,
                                   interval=1000/FPS)

    # Save — try mp4, fall back to gif
    out_mp4 = os.path.join(SAVE_DIR, f"{prefix}_wavefield_anim.mp4")
    out_gif  = os.path.join(SAVE_DIR, f"{prefix}_wavefield_anim.gif")
    try:
        anim.save(out_mp4, writer=animation.FFMpegWriter(fps=FPS, bitrate=1800))
        print(f"  Saved: {out_mp4}")
    except Exception as e:
        print(f"  ffmpeg not found ({e}), saving as gif...")
        anim.save(out_gif, writer=animation.PillowWriter(fps=FPS))
        print(f"  Saved: {out_gif}")

    plt.close()

    # Display inline in notebook
    try:
        display(HTML(anim.to_jshtml()))
    except Exception:
        pass

    return anim


# ═══════════════════════════════════════════════════════════════════════════════
# ANIMATION 2 — Pointwise error evolving in time
# ═══════════════════════════════════════════════════════════════════════════════

def animate_error(models, material, x_fd, t_fd, u_fd, prefix):
    arch_names  = list(models.keys())
    t_idx       = np.linspace(0, len(t_fd)-1, N_FRAMES, dtype=int)
    t_frames    = t_fd[t_idx]
    u_fd_frames = u_fd[t_idx]

    print("  Pre-computing error frames...")
    err_frames  = {}
    for arch, model in models.items():
        u_p = predict_spacetime(model, x_fd, t_frames)
        err_frames[arch] = np.abs(u_p - u_fd_frames)

    valid_emax = []
    for e in err_frames.values():
        valid_e = e[~np.isnan(e) & ~np.isinf(e)]
        if len(valid_e) > 0:
            valid_emax.append(valid_e.max())
    global_emax = max(valid_emax) * 0.9 if valid_emax else 1.0

    n_arch = len(arch_names)
    fig, axes = plt.subplots(1, n_arch, figsize=(5*n_arch, 3.5),
                              sharey=True, squeeze=False)
    fig.suptitle(f"{material.name} — Pointwise Error |û − u_FD|", fontsize=12)

    images = []
    for col, arch in enumerate(arch_names):
        im = axes[0, col].imshow(
            err_frames[arch][[0], :],
            aspect="auto", cmap="hot",
            vmin=0, vmax=global_emax,
            extent=[x_fd[0], x_fd[-1], 0, T_MAX])
        axes[0, col].set_title(ARCH_LABELS[arch], fontsize=9)
        axes[0, col].set_xlabel("x")
        axes[0, col].set_yticks([])
        plt.colorbar(im, ax=axes[0, col], fraction=0.05, pad=0.04)
        images.append(im)

    time_text = fig.text(0.5, 0.97, "", ha="center", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    def init():
        for im, arch in zip(images, arch_names):
            im.set_data(err_frames[arch][[0], :])
        time_text.set_text("")
        return images + [time_text]

    def update(fi):
        for im, arch in zip(images, arch_names):
            im.set_data(err_frames[arch][[fi], :])
        time_text.set_text(f"t = {t_frames[fi]:.3f} s")
        return images + [time_text]

    anim = animation.FuncAnimation(fig, update, frames=N_FRAMES,
                                   init_func=init, blit=True,
                                   interval=1000/FPS)

    out_mp4 = os.path.join(SAVE_DIR, f"{prefix}_error_anim.mp4")
    out_gif  = os.path.join(SAVE_DIR, f"{prefix}_error_anim.gif")
    try:
        anim.save(out_mp4, writer=animation.FFMpegWriter(fps=FPS, bitrate=1800))
        print(f"  Saved: {out_mp4}")
    except Exception as e:
        print(f"  ffmpeg not found ({e}), saving as gif...")
        anim.save(out_gif, writer=animation.PillowWriter(fps=FPS))
        print(f"  Saved: {out_gif}")

    plt.close()

    try:
        display(HTML(anim.to_jshtml()))
    except Exception:
        pass

    return anim


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP — runs all three experiments
# ═══════════════════════════════════════════════════════════════════════════════

os.makedirs(BASE_SAVE_DIR, exist_ok=True)

for prefix, material in EXPERIMENTS.items():
    # Update global variables for this specific experiment
    T_MAX = T_MAX_DICT.get(prefix, 1.0)
    exp_name = prefix.split('_')[0]
    SAVE_DIR = os.path.join(BASE_SAVE_DIR, exp_name)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  {material.name}  ({prefix})")
    print(f"{'═'*60}")

    # 1. Load
    models = load_models(prefix)

    # 2. FD reference
    print("\n  Building FD reference...")
    x_fd, t_fd, u_fd = fd_reference(material, nx=512, nt=2000, T=T_MAX, sigma_g=SIGMA_G)
    print(f"  FD: {len(x_fd)}×{len(t_fd)} | u ∈ [{u_fd.min():.3f}, {u_fd.max():.3f}]")

    # 3. Metrics
    eval_times   = list(np.linspace(0.05, T_MAX, 20))
    metrics_dict = {}
    for arch, model in models.items():
        m = evaluate(model, material, x_fd, t_fd, u_fd,
                     sigma_g=SIGMA_G, eval_times=eval_times)
        metrics_dict[arch] = m
        print(f"  {ARCH_LABELS[arch]:<26} L2 = {m['mean_l2']:.2f}%")

    print_table(metrics_dict, material.name)

    # 4. Static figures
    plot_traces(models, material, x_fd, t_fd, u_fd, prefix)
    plot_wavefield(models, material, x_fd, t_fd, u_fd, prefix)
    plot_metrics(metrics_dict, material, prefix)
    plot_residual_map(models, material, prefix)

    # 5. Animations
    t0_anim = time.time()
    animate_wavefield(models, material, x_fd, t_fd, u_fd, prefix)
    print(f"  [Timer] Wavefield animation took {time.time() - t0_anim:.1f}s")
    
    t0_anim = time.time()
    animate_error(models, material, x_fd, t_fd, u_fd, prefix)
    print(f"  [Timer] Error animation took {time.time() - t0_anim:.1f}s")

print("\n✓ All experiments complete.")