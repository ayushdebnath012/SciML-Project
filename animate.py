"""
animate_waves.py
================
Standalone animation script.

Loads saved .pt model files and produces one GIF per material:
  anim_homogeneous.gif
  anim_twolayer.gif
  anim_multilayer.gif

Each GIF shows three rows:
  Row 1 — displacement u(x,t): FD reference vs Vanilla vs Fourier vs PirateNet
  Row 2 — pointwise error |u_pred - u_ref| for each model
  Row 3 — material profile E(x) (static)

Run from the same folder as the .pt files:
  python animate_waves.py

The .pt files this script expects are produced by pinn_1d_elastic_wave.py:
  exp1_homogeneous_{vanilla,fourier,pirate}_model.pt
  exp2_twolayer_{vanilla,fourier,pirate}_model.pt
  exp3_multilayer_{vanilla,fourier,pirate}_model.pt

PirateNet uses sigma=2.0 (paper default) for Homogeneous/TwoLayer and
sigma=5.0 (raised to match wavefield bandwidth) for MultiLayer, matching
the training configuration in pinn_1d_elastic_wave.py.
"""

import math, os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED     = 42
N_FRAMES = 80
FPS      = 20
DPI      = 120

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Material models ──────────────────────────────────────────────────────────

class HomogeneousModel:
    name = "Homogeneous"
    x_min, x_max = -1.0, 1.0
    def E(self, x):   return torch.full_like(x, 80.0)
    def rho(self, x): return torch.full_like(x, 100.0)

class TwoLayerModel:
    name = "TwoLayer"
    x_min, x_max = -1.0, 1.0
    def E(self, x):
        a = 0.5*(1.0+torch.tanh(x/0.02))
        return 60.0*(1.0-a) + 120.0*a   # E1=60, E2=120 (matches training)
    def rho(self, x): return torch.full_like(x, 100.0)

class MultiLayerModel:
    name = "MultiLayer"
    x_min, x_max = -1.5, 1.5
    _E_vals = np.linspace(60.0, 150.0, 6)   # matches training: 60..150
    def E(self, x):
        lw  = (self.x_max - self.x_min) / 6
        out = torch.full_like(x, float(self._E_vals[0]))
        for k in range(5):
            bnd = self.x_min + (k+1)*lw
            a   = 0.5*(1.0+torch.tanh((x-bnd)/0.05))  # width=0.05 (matches training)
            out = out*(1.0-a) + float(self._E_vals[k+1])*a
        return out
    def rho(self, x): return torch.full_like(x, 100.0)

# ─── Network architectures ────────────────────────────────────────────────────

class VanillaPINN(nn.Module):
    def __init__(self, hidden_layers=5, hidden_units=128):
        super().__init__()
        layers = [nn.Linear(2, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers-1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.Tanh()]
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))

class FourierFeaturePINN(nn.Module):
    def __init__(self, hidden_layers=5, hidden_units=128, n_fourier=128, sigma=1.0):
        super().__init__()
        self.B   = nn.Parameter(torch.randn(n_fourier, 2)*sigma)
        in_feat  = 2*n_fourier
        layers   = [nn.Linear(in_feat, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers-1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.Tanh()]
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)
    def fourier_embed(self, x, t):
        proj = torch.cat([x,t], dim=-1) @ self.B.T
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
    def forward(self, x, t):
        return self.net(self.fourier_embed(x, t))


class PirateBlock(nn.Module):
    """Mirrors PirateBlock in pinn_1d_elastic_wave.py."""
    def __init__(self, units, alpha_init=0.0):
        super().__init__()
        self.W1 = nn.Linear(units, units)
        self.W2 = nn.Linear(units, units)
        self.W3 = nn.Linear(units, units)
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))
        self.act   = nn.Tanh()
    def forward(self, x_in, U, V):
        f  = self.act(self.W1(x_in));  z1 = f * U + (1.0 - f) * V
        g  = self.act(self.W2(z1));    z2 = g * U + (1.0 - g) * V
        h  = self.act(self.W3(z2))
        return self.alpha * h + (1.0 - self.alpha) * x_in


class PirateNet(nn.Module):
    """Mirrors PirateNet in pinn_1d_elastic_wave.py.
    198,404 params with proj layer + bias on out_layer."""
    def __init__(self, n_blocks=3, units=128, n_fourier=64, sigma=2.0,
                 alpha_init=0.0, out_init_scale=0.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(n_fourier, 2) * sigma)
        in_dim = 2 * n_fourier
        self.enc_U = nn.Sequential(nn.Linear(in_dim, units), nn.Tanh())
        self.enc_V = nn.Sequential(nn.Linear(in_dim, units), nn.Tanh())
        self.proj  = nn.Linear(in_dim, units)
        self.blocks = nn.ModuleList([
            PirateBlock(units, alpha_init=alpha_init) for _ in range(n_blocks)
        ])
        self.out_layer = nn.Linear(units, 1)
    def fourier_embed(self, x, t):
        proj = torch.cat([x, t], dim=-1) @ self.B.T
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
    def forward(self, x, t):
        phi = self.fourier_embed(x, t)
        U   = self.enc_U(phi);  V = self.enc_V(phi)
        h   = torch.tanh(self.proj(phi))
        for block in self.blocks:
            h = block(h, U, V)
        return self.out_layer(h)


# ─── Physics helpers ──────────────────────────────────────────────────────────

def gaussian_ic(x, sigma_g=0.1, x0=0.0):
    f    = torch.exp(-0.5*((x-x0)/sigma_g)**2)
    dfdx = -(x-x0)/sigma_g**2 * f
    return dfdx / (dfdx.abs().max() + 1e-12)

def apply_ansatz(nn_out, x, t, sigma_g=0.1):
    g = gaussian_ic(x, sigma_g)
    return g*torch.exp(-0.5*(15.0*t)**2) + torch.tanh(25.0*t)**2 * nn_out

def fd_reference(material, nx=512, nt=2000, T=1.0, sigma_g=0.1):
    x_min, x_max = material.x_min, material.x_max
    x  = np.linspace(x_min, x_max, nx)
    dx = x[1]-x[0];  dt = T/nt;  t = np.linspace(0, T, nt)
    x_t   = torch.tensor(x, dtype=torch.float32)
    E_np  = material.E(x_t).numpy()
    rho_np = material.rho(x_t).numpy()
    f    = np.exp(-0.5*(x/sigma_g)**2)
    dfdr = -x/sigma_g**2*f
    g    = dfdr/(np.abs(dfdr).max()+1e-12)
    u_prev = g.copy();  u_curr = g.copy()
    u_all  = np.zeros((nt, nx));  u_all[0]=u_prev;  u_all[1]=u_curr
    E_half = 0.5*(E_np[:-1]+E_np[1:])
    for n in range(1, nt-1):
        u_new = np.zeros(nx)
        flux   = E_half*(u_curr[1:]-u_curr[:-1])/dx
        d_flux = (flux[1:]-flux[:-1])/dx
        u_new[1:-1] = 2*u_curr[1:-1]-u_prev[1:-1]+(dt**2/rho_np[1:-1])*d_flux
        for side, ei in [(0,0),(1,-1)]:
            Vp = math.sqrt(E_np[ei]/rho_np[ei]);  r = Vp*dt/dx
            if side==0:
                u_new[0]  = u_curr[1]  + (r-1)/(r+1)*(u_new[1]  - u_curr[0])
            else:
                u_new[-1] = u_curr[-2] + (r-1)/(r+1)*(u_new[-2] - u_curr[-1])
        u_prev=u_curr;  u_curr=u_new;  u_all[n+1]=u_curr
    return x, t, u_all

# ─── Loader / predictor ───────────────────────────────────────────────────────

def load_model(arch, path, material_is_multilayer=False):
    """
    Build a fresh model of the right type and load weights.
    arch: "vanilla" | "fourier" | "pirate"
    PirateNet uses sigma=5.0 on MultiLayer (matches training), 2.0 elsewhere.
    """
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None

    if arch == "vanilla":
        model = VanillaPINN(5, 128)
    elif arch == "fourier":
        model = FourierFeaturePINN(5, 128, 128, 1.0)
    elif arch == "pirate":
        sigma = 5.0 if material_is_multilayer else 2.0
        model = PirateNet(3, 128, 64, sigma)
    else:
        print(f"  [SKIP] unknown architecture '{arch}'")
        return None

    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    except RuntimeError as e:
        # Helpful message if the .pt was trained with a different sigma /
        # different architecture variant
        print(f"  [SKIP] {path} failed to load: {e}")
        return None

    model.to(DEVICE).eval()
    print(f"  Loaded: {path}")
    return model

@torch.no_grad()
def predict(model, x_np, t_val):
    xt = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    tt = torch.full_like(xt, t_val)
    return apply_ansatz(model(xt, tt), xt, tt).cpu().numpy().flatten()

# ─── Main animation builder ───────────────────────────────────────────────────

def build_animation(material, prefix, out_gif):
    print(f"\n{'='*50}")
    print(f"  {material.name}")

    is_multi = isinstance(material, MultiLayerModel)
    van = load_model("vanilla", f"{prefix}_vanilla_model.pt")
    fou = load_model("fourier", f"{prefix}_fourier_model.pt")
    pir = load_model("pirate",  f"{prefix}_pirate_model.pt",
                      material_is_multilayer=is_multi)
    if van is None and fou is None and pir is None:
        print("  No models found — skipping.")
        return

    print("  Computing FD reference...")
    x_np, t_np, u_fd = fd_reference(material)
    nt_fd = len(t_np)

    x_torch = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)
    E_np    = material.E(x_torch).numpy().flatten()

    frame_idx = np.linspace(0, nt_fd-1, N_FRAMES, dtype=int)

    print("  Pre-computing predictions...")
    pv = np.zeros((N_FRAMES, len(x_np))) if van else None
    pf = np.zeros((N_FRAMES, len(x_np))) if fou else None
    pp = np.zeros((N_FRAMES, len(x_np))) if pir else None
    for fi, ti in enumerate(frame_idx):
        if van: pv[fi] = predict(van, x_np, t_np[ti])
        if fou: pf[fi] = predict(fou, x_np, t_np[ti])
        if pir: pp[fi] = predict(pir, x_np, t_np[ti])

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(f"{material.name}  —  Elastic Wave Propagation",
                 fontsize=13, fontweight="bold")
    gs = GridSpec(3, 1, figure=fig, height_ratios=[3, 2, 1.5],
                  hspace=0.48, left=0.10, right=0.97, top=0.91, bottom=0.07)

    ax_u   = fig.add_subplot(gs[0])
    ax_err = fig.add_subplot(gs[1])
    ax_E   = fig.add_subplot(gs[2])

    C_FD, C_VAN, C_FOU, C_PIR = "black", "#E67E22", "#2980B9", "#27AE60"

    # global y-limits — include pirate's predictions if present
    u_vals = [u_fd[frame_idx].ravel()]
    if pv is not None: u_vals.append(pv.ravel())
    if pf is not None: u_vals.append(pf.ravel())
    if pp is not None: u_vals.append(pp.ravel())
    u_lim  = max(abs(np.concatenate(u_vals).min()),
                 abs(np.concatenate(u_vals).max())) * 1.2
    u_lim  = max(u_lim, 1e-6)

    err_vals = []
    if pv is not None: err_vals.append(np.abs(pv - u_fd[frame_idx]).ravel())
    if pf is not None: err_vals.append(np.abs(pf - u_fd[frame_idx]).ravel())
    if pp is not None: err_vals.append(np.abs(pp - u_fd[frame_idx]).ravel())
    err_max = np.concatenate(err_vals).max() * 1.2 if err_vals else 1.0
    err_max = max(err_max, 1e-6)

    # E(x) row — static
    ax_E.fill_between(x_np, E_np, alpha=0.25, color="#7F8C8D")
    ax_E.plot(x_np, E_np, color="#2C3E50", lw=1.5)
    ax_E.set_xlim(material.x_min, material.x_max)
    ax_E.set_ylabel("E(x)", fontsize=9);  ax_E.set_xlabel("x", fontsize=9)
    ax_E.set_title("Material profile E(x)", fontsize=9)
    ax_E.tick_params(labelsize=8);  ax_E.grid(True, alpha=0.3)

    # Displacement row
    ax_u.set_xlim(material.x_min, material.x_max)
    ax_u.set_ylim(-u_lim, u_lim)
    ax_u.set_ylabel("u(x,t)", fontsize=9)
    ax_u.set_title("Displacement u(x,t)", fontsize=9)
    ax_u.tick_params(labelsize=8);  ax_u.grid(True, alpha=0.3)
    ax_u.axhline(0, color="grey", lw=0.5)

    ln_fd,  = ax_u.plot([], [], color=C_FD,  lw=2.0, label="FD reference")
    ln_van  = ax_u.plot([], [], color=C_VAN, lw=1.5, ls="--", label="Vanilla PINN")[0] if van else None
    ln_fou  = ax_u.plot([], [], color=C_FOU, lw=1.5, ls=":",  label="Fourier PINN")[0] if fou else None
    ln_pir  = ax_u.plot([], [], color=C_PIR, lw=1.5, ls="-.", label="PirateNet")[0]    if pir else None
    ax_u.legend(loc="upper right", fontsize=7, framealpha=0.85)
    t_txt = ax_u.text(0.02, 0.95, "", transform=ax_u.transAxes, fontsize=9, va="top")

    # Error row
    ax_err.set_xlim(material.x_min, material.x_max)
    ax_err.set_ylim(0, err_max)
    ax_err.set_ylabel("|error|", fontsize=9)
    ax_err.set_title("Absolute error  |u_pred − u_FD|", fontsize=9)
    ax_err.tick_params(labelsize=8);  ax_err.grid(True, alpha=0.3)

    er_van = ax_err.plot([], [], color=C_VAN, lw=1.5, label="Vanilla")[0] if van else None
    er_fou = ax_err.plot([], [], color=C_FOU, lw=1.5, ls=":", label="Fourier")[0] if fou else None
    er_pir = ax_err.plot([], [], color=C_PIR, lw=1.5, ls="-.", label="PirateNet")[0] if pir else None
    ax_err.legend(loc="upper right", fontsize=7, framealpha=0.85)

    # Stack the rolling-L2 readouts in the upper-left of the error panel
    y_van, y_fou, y_pir = 0.92, 0.78, 0.64
    l2_van_txt = ax_err.text(0.02, y_van, "", transform=ax_err.transAxes,
                              color=C_VAN, fontsize=8, va="top") if van else None
    l2_fou_txt = ax_err.text(0.02, y_fou, "", transform=ax_err.transAxes,
                              color=C_FOU, fontsize=8, va="top") if fou else None
    l2_pir_txt = ax_err.text(0.02, y_pir, "", transform=ax_err.transAxes,
                              color=C_PIR, fontsize=8, va="top") if pir else None

    # ── Animation functions ───────────────────────────────────────────────────
    def _init():
        ln_fd.set_data([], [])
        arts = [ln_fd]
        for ln in [ln_van, ln_fou, ln_pir, er_van, er_fou, er_pir]:
            if ln: ln.set_data([], []); arts.append(ln)
        return arts

    def _update(fi):
        ti    = frame_idx[fi]
        t_val = t_np[ti]
        u_ref = u_fd[ti]
        ln_fd.set_data(x_np, u_ref)
        t_txt.set_text(f"t = {t_val:.3f}")
        arts = [ln_fd, t_txt]

        denom = np.linalg.norm(u_ref) + 1e-12

        if van:
            ln_van.set_data(x_np, pv[fi])
            er_van.set_data(x_np, np.abs(pv[fi]-u_ref))
            l2v = np.linalg.norm(pv[fi]-u_ref)/denom*100
            l2_van_txt.set_text(f"Vanilla    L2 = {l2v:.2f}%")
            arts += [ln_van, er_van, l2_van_txt]

        if fou:
            ln_fou.set_data(x_np, pf[fi])
            er_fou.set_data(x_np, np.abs(pf[fi]-u_ref))
            l2f = np.linalg.norm(pf[fi]-u_ref)/denom*100
            l2_fou_txt.set_text(f"Fourier    L2 = {l2f:.2f}%")
            arts += [ln_fou, er_fou, l2_fou_txt]

        if pir:
            ln_pir.set_data(x_np, pp[fi])
            er_pir.set_data(x_np, np.abs(pp[fi]-u_ref))
            l2p = np.linalg.norm(pp[fi]-u_ref)/denom*100
            l2_pir_txt.set_text(f"PirateNet  L2 = {l2p:.2f}%")
            arts += [ln_pir, er_pir, l2_pir_txt]

        return arts

    print(f"  Rendering {N_FRAMES} frames → {out_gif}")
    ani = animation.FuncAnimation(
        fig, _update, frames=N_FRAMES,
        init_func=_init, blit=True, interval=1000//FPS
    )
    ani.save(out_gif, writer="pillow", fps=FPS, dpi=DPI)
    plt.close(fig)
    print(f"  Done.")

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    configs = [
        (HomogeneousModel(), "exp1_homogeneous", "anim_homogeneous.gif"),
        (TwoLayerModel(),    "exp2_twolayer",    "anim_twolayer.gif"),
        (MultiLayerModel(),  "exp3_multilayer",  "anim_multilayer.gif"),
    ]
    built = 0
    for material, prefix, out_gif in configs:
        build_animation(material, prefix, out_gif)
        if os.path.exists(out_gif): built += 1
    print(f"\n{'='*50}")
    print(f"  Done — {built}/{len(configs)} GIFs saved.")
    if built == 0:
        print("  Tip: run pinn_1d_elastic_wave.py first to generate .pt model files.")