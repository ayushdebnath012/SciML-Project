import argparse
import math
import time
from tqdm import tqdm
from IPython import display  # Only if using Jupyter/Colab. Otherwise, use plt.pause()
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp

# ─────────────────────────────────────────────
# 0.  Global config
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PHYSICS  —  Material models + reference FD solver
# ═══════════════════════════════════════════════════════════════════════════════

class MaterialModel:
    """
    Defines the 1-D material profile E(x) = λ(x) + 2μ(x) and density ρ.

    Vp(x) = sqrt(E(x)/ρ)

    All quantities are non-dimensional (matching the paper's convention).
    """

    def __init__(self, name: str, x_domain: tuple[float, float]):
        self.name = name
        self.x_min, self.x_max = x_domain

    def E(self, x: torch.Tensor) -> torch.Tensor:
        """P-wave modulus E(x) as a tensor."""
        raise NotImplementedError

    def rho(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def Vp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.E(x) / self.rho(x))


class HomogeneousModel(MaterialModel):
    """λ=20, μ=30, ρ=100  →  E=80, Vp≈0.894  (same as paper, 1D plane wave)"""

    def __init__(self):
        super().__init__("Homogeneous", (-1.0, 1.0))
        self._E   = 80.0   # λ + 2μ
        self._rho = 100.0

    def E(self, x):
        return torch.full_like(x, self._E)

    def rho(self, x):
        return torch.full_like(x, self._rho)


class TwoLayerModel(MaterialModel):
    """
    Interface at x = 0.
      x < 0 : λ=20, μ=30, ρ=100  →  E=80
      x ≥ 0 : λ=25, μ=35, ρ=100  →  E=95
    Transition smoothed with tanh (width=0.02).
    """

    def __init__(self):
        super().__init__("TwoLayer", (-1.0, 1.0))
        self._rho = 100.0
        self._E1  = 80.0
        self._E2  = 95.0

    def E(self, x):
        w = 0.02
        alpha = 0.5 * (1.0 + torch.tanh(x / w))   # 0 left, 1 right
        return self._E1 * (1.0 - alpha) + self._E2 * alpha

    def rho(self, x):
        return torch.full_like(x, self._rho)


class MultiLayerModel(MaterialModel):
    """
    Six equal-thickness layers on x ∈ [-1.5, 1.5], E values linearly spaced
    from 60 to 110 (analogous to the paper's λ range 20→70 + 2μ).
    """

    def __init__(self):
        super().__init__("MultiLayer", (-1.5, 1.5))
        self._rho    = 100.0
        self.n_layers = 6
        self.E_vals  = np.linspace(60.0, 110.0, self.n_layers)  # per layer

    def E(self, x):
        x_min, x_max = self.x_min, self.x_max
        layer_width   = (x_max - x_min) / self.n_layers
        E_val = torch.full_like(x, self.E_vals[0])
        w = 0.02
        for k in range(self.n_layers - 1):
            boundary = x_min + (k + 1) * layer_width
            alpha    = 0.5 * (1.0 + torch.tanh((x - boundary) / w))
            E_val    = E_val * (1.0 - alpha) + self.E_vals[k + 1] * alpha
        return E_val

    def rho(self, x):
        return torch.full_like(x, self._rho)

# ─────────────────────────────────────────────
# 1-D Finite Difference reference solver
# ─────────────────────────────────────────────

def fd_reference(material: MaterialModel,
                 nx: int   = 512,
                 nt: int   = 2000,
                 T: float  = 1.0,
                 sigma_g: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Explicit second-order FD solver for:
        ρ u_tt = ∂_x(E(x) u_x)

    Returns (x_grid, t_grid, u) where u has shape (nt, nx).
    Absorbing boundaries implemented via simple Mur ABC.
    """
    x_min, x_max = material.x_min, material.x_max
    x = np.linspace(x_min, x_max, nx)
    dx = x[1] - x[0]
    dt = T / nt
    t  = np.linspace(0, T, nt)

    # Stability check (CFL)
    x_t = torch.tensor(x, dtype=torch.float32)
    E_np  = material.E(x_t).numpy()
    rho_np = material.rho(x_t).numpy()
    Vp_max = np.sqrt(E_np / rho_np).max()
    cfl    = Vp_max * dt / dx
    if cfl > 1.0:
        print(f"  [FD WARNING] CFL={cfl:.3f} > 1, reduce dt or increase nx")

    # Initial conditions — same Gaussian derivative as the paper
    x0   = 0.0
    f    = np.exp(-0.5 * ((x - x0) / sigma_g) ** 2)
    dfdr = -(x - x0) / sigma_g**2 * f
    g    = dfdr / (np.abs(dfdr).max() + 1e-12)   # normalised

    u_prev = g.copy()          # u(x, 0)
    u_curr = g.copy()          # u(x, dt) ≈ u(0) + dt*0  (zero initial velocity)
    u_all  = np.zeros((nt, nx))
    u_all[0] = u_prev
    u_all[1] = u_curr

    # Half-cell E values for staggered stencil: E_{i+1/2}
    E_half = 0.5 * (E_np[:-1] + E_np[1:])  # shape (nx-1,)

    for n in range(1, nt - 1):
        u_new = np.zeros(nx)
        # Interior
        flux       = E_half * (u_curr[1:] - u_curr[:-1]) / dx   # shape (nx-1)
        d_flux     = (flux[1:] - flux[:-1]) / dx                 # shape (nx-2)
        u_new[1:-1] = (2 * u_curr[1:-1] - u_prev[1:-1]
                       + (dt**2 / rho_np[1:-1]) * d_flux)

        # Mur absorbing boundary (1st order)
        Vp_left  = math.sqrt(E_np[0]  / rho_np[0])
        Vp_right = math.sqrt(E_np[-1] / rho_np[-1])
        r_left   = Vp_left  * dt / dx
        r_right  = Vp_right * dt / dx
        u_new[0]  = u_curr[1]  + (r_left  - 1) / (r_left  + 1) * (u_new[1]  - u_curr[0])
        u_new[-1] = u_curr[-2] + (r_right - 1) / (r_right + 1) * (u_new[-2] - u_curr[-1])

        u_prev = u_curr
        u_curr = u_new
        u_all[n + 1] = u_curr

    return x, t, u_all

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  NETWORK ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

class VanillaPINN(nn.Module):
    """
    Standard MLP PINN.
    Input : (x, t)  →  2 features
    Output: û(x, t) →  scalar displacement
    """

    def __init__(self, hidden_layers: int = 5, hidden_units: int = 128):
        super().__init__()
        layers = [nn.Linear(2, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.Tanh()]
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)


class FourierFeaturePINN(nn.Module):
    """
    Random Fourier Feature embedding → MLP.
    B is treated as a trainable parameter (as in the paper).
    """

    def __init__(self,
                 hidden_layers: int = 5,
                 hidden_units: int  = 128,
                 n_fourier: int     = 128,
                 sigma: float       = 1.0):
        super().__init__()
        # Trainable Fourier frequency matrix  B ∈ R^{n_fourier × 2}
        B_init = torch.randn(n_fourier, 2) * sigma
        self.B = nn.Parameter(B_init)

        in_features = 2 * n_fourier   # cos + sin
        layers = [nn.Linear(in_features, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.Tanh()]
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def fourier_embed(self, x, t):
        inp = torch.cat([x, t], dim=-1)        # (N, 2)
        proj = inp @ self.B.T                  # (N, n_fourier)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (N, 2*n_fourier)

    def forward(self, x, t):
        phi = self.fourier_embed(x, t)
        return self.net(phi)


class PirateBlock(nn.Module):
    """
    Single PirateNet residual block (Eq. 14-19 in the paper).
    Three dense layers + dual gating + adaptive skip (scalar α per block).
    """

    def __init__(self, units: int):
        super().__init__()
        self.W1 = nn.Linear(units, units)
        self.W2 = nn.Linear(units, units)
        self.W3 = nn.Linear(units, units)
        self.alpha = nn.Parameter(torch.zeros(1))   # initialised to 0
        self.act   = nn.Tanh()

    def forward(self, x_in, U, V):
        # Eq. 14-15
        f  = self.act(self.W1(x_in))
        z1 = f * U + (1.0 - f) * V
        # Eq. 16-17
        g  = self.act(self.W2(z1))
        z2 = g * U + (1.0 - g) * V
        # Eq. 18
        h  = self.act(self.W3(z2))
        # Eq. 19  (adaptive residual)
        return self.alpha * h + (1.0 - self.alpha) * x_in


class PirateNet(nn.Module):
    """
    Full PirateNet:
      Fourier embedding → two gating encoders U,V → L residual blocks → linear out.
    """

    def __init__(self,
                 n_blocks: int  = 3,
                 units: int     = 128,
                 n_fourier: int = 64,
                 sigma: float   = 2.0):
        super().__init__()
        B_init   = torch.randn(n_fourier, 2) * sigma
        self.B   = nn.Parameter(B_init)
        in_dim   = 2 * n_fourier

        # Gating encoders (Eq. 12-13)
        self.enc_U = nn.Sequential(nn.Linear(in_dim, units), nn.Tanh())
        self.enc_V = nn.Sequential(nn.Linear(in_dim, units), nn.Tanh())

        # Input projection to 'units' dimensions
        self.proj  = nn.Linear(in_dim, units)

        # Residual blocks
        self.blocks = nn.ModuleList([PirateBlock(units) for _ in range(n_blocks)])

        # Final linear
        self.out_layer = nn.Linear(units, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.out_layer:
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def fourier_embed(self, x, t):
        inp  = torch.cat([x, t], dim=-1)
        proj = inp @ self.B.T
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    def forward(self, x, t):
        phi = self.fourier_embed(x, t)
        U   = self.enc_U(phi)
        V   = self.enc_V(phi)
        h   = torch.tanh(self.proj(phi))
        for block in self.blocks:
            h = block(h, U, V)
        return self.out_layer(h)



# ═══════════════════════════════════════════════════════════════════════════════
# 3.  ANSATZ  —  Hard enforcement of initial conditions (Eq. 11)
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_ic(x: torch.Tensor, sigma_g: float = 0.1, x0: float = 0.0) -> torch.Tensor:
    """
    Initial displacement g(x): normalised first derivative of Gaussian.
    Matches Eq. (9-10) of the paper, adapted to 1D scalar field.
    """
    f    = torch.exp(-0.5 * ((x - x0) / sigma_g) ** 2)
    dfdx = -(x - x0) / sigma_g**2 * f
    g    = dfdx / (dfdx.abs().max() + 1e-12)
    return g


def apply_ansatz(nn_out: torch.Tensor,
                 x: torch.Tensor,
                 t: torch.Tensor,
                 sigma_g: float = 0.1) -> torch.Tensor:
    """
    û(x,t) = g(x)·exp(-½(15t)²) + tanh²(25t)·NN(x,t)
    """
    g      = gaussian_ic(x, sigma_g)
    decay  = torch.exp(-0.5 * (15.0 * t) ** 2)
    growth = torch.tanh(25.0 * t) ** 2
    return g * decay + growth * nn_out


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PHYSICS LOSS  — 1D elastic wave PDE residual
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pde_residual(model: nn.Module,
                         x: torch.Tensor,
                         t: torch.Tensor,
                         material: MaterialModel,
                         sigma_g: float = 0.1) -> torch.Tensor:
    """
    Compute the 1D elastic wave PDE residual:
        R = ρ(x) ∂²û/∂t² - ∂/∂x [ E(x) ∂û/∂x ]

    Uses torch.autograd for exact derivatives (no truncation error).
    """
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)

    nn_out = model(x, t)
    u      = apply_ansatz(nn_out, x, t, sigma_g)

    # First derivatives
    grads_1 = torch.autograd.grad(u, [x, t],
                                  grad_outputs=torch.ones_like(u),
                                  create_graph=True)[:]
    u_x, u_t = grads_1

    # Second derivative in time
    u_tt = torch.autograd.grad(u_t, t,
                               grad_outputs=torch.ones_like(u_t),
                               create_graph=True)[0]

    # Stress: σ = E(x) * u_x
    E_val = material.E(x)
    sigma = E_val * u_x

    # Divergence of stress: ∂σ/∂x
    sigma_x = torch.autograd.grad(sigma, x,
                                  grad_outputs=torch.ones_like(sigma),
                                  create_graph=True)[0]

    rho_val = material.rho(x)
    residual = rho_val * u_tt - sigma_x
    return residual


def physics_loss(model: nn.Module,
                 x: torch.Tensor,
                 t: torch.Tensor,
                 material: MaterialModel,
                 sigma_g: float = 0.1) -> torch.Tensor:
    """MSE of the PDE residual over collocation points."""
    R = compute_pde_residual(model, x, t, material, sigma_g)
    return (R ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  COLLOCATION POINTS  —  Sobol-like quasi-random sampling
# ═══════════════════════════════════════════════════════════════════════════════

def sample_collocation(n_pts: int,
                       x_min: float, x_max: float,
                       t_max: float,
                       device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sobol sequence over (x, t).  Falls back to uniform random if torch.quasirandom
    is unavailable in older PyTorch versions.
    """
    try:
        sobol  = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=SEED)
        pts    = sobol.draw(n_pts)                         # (N, 2) in [0,1]²
    except Exception:
        pts    = torch.rand(n_pts, 2)

    x = pts[:, 0:1] * (x_max - x_min) + x_min
    t = pts[:, 1:2] * t_max
    return x.to(device), t.to(device)

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING  — L-BFGS full-batch
# ═══════════════════════════════════════════════════════════════════════════════

from IPython import display

class LivePlotter:
    """Helper to handle real-time loss plotting in Jupyter/Colab."""
    def __init__(self, arch_name):
        self.arch_name = arch_name
        self.losses = []
        # Create a persistent figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_title(f"Live Training Loss: {arch_name}")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True, alpha=0.3)
        self.line, = self.ax.semilogy([], [], color='#27AE60', lw=1.5)

    def update(self, loss_val):
        self.losses.append(loss_val)
        self.line.set_data(range(len(self.losses)), self.losses)
        
        # Rescale axes
        self.ax.relim()
        self.ax.autoscale_view()
        
        # KEY CHANGE: Clear the specific output area and redraw
        display.clear_output(wait=True)
        display.display(self.fig)

def train(model, material, n_collocation=10000, max_epochs=5000, lr=1.0, 
          history_size=100, T_stages=None, sigma_g=0.1, verbose=True, 
          arch_name="Model", patience=50, min_delta=1e-9):
    
    if T_stages is None: T_stages = [1.0]
    x_min, x_max = material.x_min, material.x_max
    loss_history = []
    
    # Early Stopping Variables
    best_loss = float('inf')
    wait = 0
    best_weights = None

    plotter = LivePlotter(arch_name)
    optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=20,
                            history_size=history_size, line_search_fn="strong_wolfe")

    for stage_idx, T_end in enumerate(T_stages):
        x_col, t_col = sample_collocation(n_collocation, x_min, x_max, T_end, DEVICE)
        epochs_this_stage = max_epochs // len(T_stages)
        pbar = tqdm(range(epochs_this_stage), desc=f"Stage {stage_idx+1}/{len(T_stages)}")
        
        for ep in pbar:
            def closure():
                optimizer.zero_grad()
                loss = physics_loss(model, x_col, t_col, material, sigma_g)
                loss.backward()
                return loss

            loss_val = optimizer.step(closure)
            current_loss = loss_val.item()
            loss_history.append(current_loss)
            
            # --- Early Stopping Logic ---
            if current_loss < (best_loss - min_delta):
                best_loss = current_loss
                wait = 0
                # Record the best weights in memory
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            
            if wait >= patience:
                print(f"\n[Early Stopping] No improvement for {patience} epochs. Stopping...")
                model.load_state_dict(best_weights) # Restore best weights
                plt.close(plotter.fig)
                return loss_history
            # ----------------------------

            if ep % 5 == 0:
                plotter.update(current_loss)
                pbar.set_postfix({"loss": f"{current_loss:.2e}", "best": f"{best_loss:.2e}"})
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    plt.close(plotter.fig)
    return loss_history

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  EVALUATION  —  L2 error and energy difference
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: nn.Module,
             material: MaterialModel,
             x_fd: np.ndarray,
             t_fd: np.ndarray,
             u_fd: np.ndarray,
             sigma_g: float = 0.1,
             eval_times: list = None) -> dict:
    """
    Evaluate PINN against FD reference at specified time snapshots.
    Returns dict with L2 errors and energy differences.
    """
    if eval_times is None:
        n_snap = min(10, len(t_fd))
        eval_times = [t_fd[int(i * (len(t_fd) - 1) / (n_snap - 1))] for i in range(n_snap)]

    l2_errors   = []
    energy_diffs = []
    t_eval_actual = []

    x_tensor = torch.tensor(x_fd, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    for t_val in eval_times:
        t_idx = np.argmin(np.abs(t_fd - t_val))
        t_actual = t_fd[t_idx]
        u_ref    = u_fd[t_idx]             # shape (nx,)

        t_tensor = torch.full_like(x_tensor, t_actual)
        nn_out   = model(x_tensor, t_tensor)
        u_pred   = apply_ansatz(nn_out, x_tensor, t_tensor, sigma_g)
        u_pred_np = u_pred.cpu().numpy().flatten()

        # Relative L2 error (Eq. 22)
        denom = np.linalg.norm(u_ref)
        l2    = np.linalg.norm(u_ref - u_pred_np) / (denom + 1e-12)
        l2_errors.append(l2 * 100.0)   # percent

        # Normalized energy difference (Eq. in Section 3.7.2)
        numer = np.sum((u_pred_np - u_ref) ** 2)
        denom_e = np.sum(u_ref ** 2) + 1e-12
        energy_diffs.append(numer / denom_e)

        t_eval_actual.append(t_actual)

    return {
        "t":          np.array(t_eval_actual),
        "l2_errors":  np.array(l2_errors),
        "energy_diff": np.array(energy_diffs),
        "mean_l2":    float(np.mean(l2_errors)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  PLOTTING  —  Reproduce paper-style figures
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results: dict,
                 material: MaterialModel,
                 x_fd: np.ndarray,
                 t_fd: np.ndarray,
                 u_fd: np.ndarray,
                 sigma_g: float = 0.1,
                 save_prefix: str = "pinn_1d"):
    """
    Generate four figure types:
      1. Displacement traces at multiple time snapshots (Fig. 6/11/15 analogue)
      2. Wavefield heatmap  (space-time view)
      3. Training loss curves for all three models (Fig. 8b analogue)
      4. Energy difference over time (Fig. 8a analogue)
    """
    arch_names   = list(results.keys())
    arch_colors  = {"vanilla": "#E67E22", "fourier": "#2980B9", "pirate": "#27AE60"}
    arch_labels  = {"vanilla": "Vanilla PINN", "fourier": "Fourier-feature PINN",
                    "pirate": "PirateNet"}

    # ── 1.  Displacement traces ────────────────────────────────────────────────
    snap_times = [0.3, 0.6, 0.9]
    fig, axes = plt.subplots(1, len(snap_times), figsize=(14, 4), sharey=False)
    fig.suptitle(f"{material.name} Model – Displacement Field Traces", fontsize=13)

    x_tensor = torch.tensor(x_fd, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    for col, t_snap in enumerate(snap_times):
        t_idx   = np.argmin(np.abs(t_fd - t_snap))
        t_actual = t_fd[t_idx]
        u_ref   = u_fd[t_idx]

        axes[col].plot(x_fd, u_ref, "k-", lw=2, label="FD Reference")

        for arch in arch_names:
            model = results[arch]["model"]
            model.eval()
            with torch.no_grad():
                t_t   = torch.full_like(x_tensor, t_actual)
                nn_out = model(x_tensor, t_t)
                u_pred = apply_ansatz(nn_out, x_tensor, t_t, sigma_g)
            axes[col].plot(x_fd, u_pred.cpu().numpy().flatten(),
                           color=arch_colors.get(arch, "grey"),
                           lw=1.5, linestyle="--", label=arch_labels.get(arch, arch))

        axes[col].set_title(f"t = {t_actual:.2f} s")
        axes[col].set_xlabel("x")
        if col == 0:
            axes[col].set_ylabel("u(x,t)")
        axes[col].legend(fontsize=7)
        axes[col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_traces.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_prefix}_traces.png")

    # ── 2.  Space-time heatmap (PirateNet vs FD) ─────────────────────────────
    if "pirate" in results:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"{material.name} – Space-Time Wavefield (PirateNet vs FD)", fontsize=13)

        # Subsample for speed
        t_sub_idx = np.linspace(0, len(t_fd) - 1, 200, dtype=int)
        t_sub     = t_fd[t_sub_idx]
        u_fd_sub  = u_fd[t_sub_idx]

        # Build PINN prediction on grid
        model = results["pirate"]["model"]
        model.eval()
        u_pinn_sub = np.zeros_like(u_fd_sub)
        with torch.no_grad():
            for i, t_val in enumerate(t_sub):
                t_t    = torch.full_like(x_tensor, t_val)
                nn_out = model(x_tensor, t_t)
                u_pred = apply_ansatz(nn_out, x_tensor, t_t, sigma_g)
                u_pinn_sub[i] = u_pred.cpu().numpy().flatten()

        vmax = max(np.abs(u_fd_sub).max(), np.abs(u_pinn_sub).max()) * 0.8
        kw   = dict(aspect="auto", origin="lower", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax,
                    extent=[x_fd[0], x_fd[-1], t_sub[0], t_sub[-1]])

        im0 = axes[0].imshow(u_pinn_sub, **kw)
        axes[0].set_title("PirateNet"); axes[0].set_xlabel("x"); axes[0].set_ylabel("t")

        im1 = axes[1].imshow(u_fd_sub, **kw)
        axes[1].set_title("FD Reference"); axes[1].set_xlabel("x")

        diff = u_pinn_sub - u_fd_sub
        vd   = np.abs(diff).max() * 0.8
        im2  = axes[2].imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r",
                              vmin=-vd, vmax=vd,
                              extent=[x_fd[0], x_fd[-1], t_sub[0], t_sub[-1]])
        axes[2].set_title("Difference"); axes[2].set_xlabel("x")

        for ax, im in zip(axes, [im0, im1, im2]):
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(f"{save_prefix}_heatmap.png", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Saved: {save_prefix}_heatmap.png")

    # ── 3.  Training loss curves ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for arch in arch_names:
        losses = results[arch]["loss_history"]
        ax.semilogy(losses, color=arch_colors.get(arch, "grey"),
                    label=arch_labels.get(arch, arch), linewidth=1.8)
    ax.set_xlabel("Epoch (L-BFGS outer iteration)")
    ax.set_ylabel("Residual Loss (log scale)")
    ax.set_title(f"{material.name} – Training Loss Convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_prefix}_loss.png")

    # ── 4.  Energy difference over time ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for arch in arch_names:
        metrics = results[arch]["metrics"]
        ax.plot(metrics["t"], metrics["energy_diff"],
                color=arch_colors.get(arch, "grey"),
                label=arch_labels.get(arch, arch), linewidth=1.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Energy Difference")
    ax.set_title(f"{material.name} – Energy Difference over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_energy.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_prefix}_energy.png")

    # ── 5.  Summary table ─────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  {material.name} – Performance Summary")
    print(f"{'─'*50}")
    print(f"  {'Architecture':<25} {'Avg. L2 Error (%)':>18}")
    print(f"{'─'*50}")
    for arch in arch_names:
        mean_l2 = results[arch]["metrics"]["mean_l2"]
        print(f"  {arch_labels.get(arch, arch):<25} {mean_l2:>18.2f}")
    print(f"{'─'*50}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(material: MaterialModel,
                   n_collocation: int = 10_000,
                   max_epochs: int    = 700,
                   T_stages: list     = None,
                   sigma_g: float     = 0.1,
                   save_prefix: str   = "pinn_1d"):
    """
    Run all three architectures on a given material model and produce figures.
    """
    print(f"\n{'═'*60}")
    print(f"  Experiment: {material.name}")
    print(f"  Device: {DEVICE}")
    print(f"  Collocation pts: {n_collocation:,}  |  Max epochs: {max_epochs:,}")
    print(f"{'═'*60}")

    # Build FD reference
    print("\n  Building FD reference solution...")
    t0 = time.time()
    x_fd, t_fd, u_fd = fd_reference(material, nx=512, nt=2000, T=1.0, sigma_g=sigma_g)
    print(f"  FD done in {time.time()-t0:.1f}s | u range: [{u_fd.min():.3f}, {u_fd.max():.3f}]")

    # Eval time snapshots
    n_snap    = 20
    eval_times = list(np.linspace(0.05, 1.0, n_snap))

    # Architecture definitions
    arch_defs = {
        "vanilla": lambda: VanillaPINN(hidden_layers=5, hidden_units=128),
        "fourier": lambda: FourierFeaturePINN(hidden_layers=5, hidden_units=128,
                                              n_fourier=128, sigma=1.0),
        "pirate":  lambda: PirateNet(n_blocks=3, units=128, n_fourier=64, sigma=2.0),
    }

    results = {}

    for arch_name, build_fn in arch_defs.items():
        print(f"\n  ── Training: {arch_name.upper()} ──")
        model = build_fn().to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        t0 = time.time()
        loss_history = train(
            model, material,
            n_collocation=n_collocation,
            max_epochs=max_epochs,
            T_stages=T_stages,
            sigma_g=sigma_g,
            verbose=True,
            arch_name=arch_name.upper()
        )
        elapsed = time.time() - t0
        print(f"  Training time: {elapsed:.1f}s  |  Final loss: {loss_history[-1]:.4e}")
        
        metrics = evaluate(model, material, x_fd, t_fd, u_fd,
                           sigma_g=sigma_g, eval_times=eval_times)
        print(f"  Mean L2 Error: {metrics['mean_l2']:.2f}%")


        # 1. Prepare a filename
        model_filename = f"{save_prefix}_{arch_name}_model.pt"
        
        # 2. Save the model state
        torch.save(model.state_dict(), model_filename)
        print(f"  Model weights saved to: {model_filename}")
        
        results[arch_name] = {
            "model":        model,
            "loss_history": loss_history,
            "metrics":      metrics,
        }

    plot_results(results, material, x_fd, t_fd, u_fd,
                 sigma_g=sigma_g, save_prefix=save_prefix)

    return results

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1D Elastic Wave PINNs")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["homogeneous", "layered", "multilayer", "all"],
                        help="Which experiment to run")
    parser.add_argument("--epochs", type=int, default=700,
                        help="Max epochs per experiment (use 20000+ for paper quality)")
    parser.add_argument("--n_col", type=int, default=10000,
                        help="Collocation points (use 100000 for paper quality)")
    args, unknown = parser.parse_known_args()

    # ── Experiment 1: Homogeneous ─────────────────────────────────────────────
    #if args.experiment in ("homogeneous", "all"):
    #    run_experiment(
    #        material      = HomogeneousModel(),
    #        n_collocation = args.n_col,
    #        max_epochs    = args.epochs,
    #        T_stages      = [1.0],          # no curriculum needed
    #        save_prefix   = "exp1_homogeneous",
    #    )

    # ── Experiment 2: Two-Layer ───────────────────────────────────────────────
    if args.experiment in ("layered", "all"):
        run_experiment(
            material      = TwoLayerModel(),
            n_collocation = args.n_col,
            max_epochs    = args.epochs,
            T_stages      = [1.0],
            save_prefix   = "exp2_twolayer",
        )

    # ── Experiment 3: Multi-Layer (with curriculum learning) ──────────────────
    if args.experiment in ("multilayer", "all"):
        run_experiment(
            material      = MultiLayerModel(),
            n_collocation = args.n_col,
            max_epochs    = args.epochs,
            T_stages      = [0.3, 0.5, 0.8, 1.0],  # curriculum (Section 4.3)
            save_prefix   = "exp3_multilayer",
        )

    print("\nAll experiments complete.")


