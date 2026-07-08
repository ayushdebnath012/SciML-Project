import argparse
import math
import time
from tqdm.auto import tqdm  # Auto-detects Jupyter for beautiful HTML progress bars
from IPython import display  
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

# ─────────────────────────────────────────────
# 0.  Global Configuration & GPU Setup
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
    def __init__(self, name: str, x_domain_physical: tuple[float, float],
                 E_ref: float, rho_ref: float):
        self.name = name
        self._x_min_phys, self._x_max_phys = x_domain_physical
        
        self.E_ref   = E_ref
        self.rho_ref = rho_ref
        self.L_ref   = (self._x_max_phys - self._x_min_phys) / 2.0
        self.V_ref   = math.sqrt(E_ref / rho_ref)
        self.T_ref   = self.L_ref / self.V_ref
        
        self.x_min = self._x_min_phys / self.L_ref
        self.x_max = self._x_max_phys / self.L_ref

    def E(self, x: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def rho(self, x: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def Vp(self, x: torch.Tensor) -> torch.Tensor: return torch.sqrt(self.E(x) / self.rho(x))
    
    def to_physical_x(self, x_nd): return x_nd * self.L_ref
    def to_physical_t(self, t_nd): return t_nd * self.T_ref
    def to_nondim_t(self, t_phys): return t_phys / self.T_ref
    def nondim_sigma_g(self, sigma_g_phys: float) -> float: return sigma_g_phys / self.L_ref

class HomogeneousModel(MaterialModel):
    def __init__(self):
        super().__init__("Homogeneous", (-1.0, 1.0), E_ref=80.0, rho_ref=100.0)
    def E(self, x): return torch.full_like(x, 1.0)
    def rho(self, x): return torch.full_like(x, 1.0)

class TwoLayerModel(MaterialModel):
    def __init__(self):
        super().__init__("TwoLayer", (-1.0, 1.0), E_ref=80.0, rho_ref=100.0)
        self._E1_nd = 80.0  / self.E_ref
        self._E2_nd = 120.0 / self.E_ref
    def E(self, x):
        w = 0.02 / self.L_ref
        alpha = 0.5 * (1.0 + torch.tanh(x / w))
        return self._E1_nd * (1.0 - alpha) + self._E2_nd * alpha
    def rho(self, x): return torch.full_like(x, 1.0)

class MultiLayerModel(MaterialModel):
    def __init__(self):
        super().__init__("MultiLayer", (-1.5, 1.5), E_ref=60.0, rho_ref=100.0)
        self.n_layers  = 6
        self.E_vals_nd = np.linspace(60.0, 150.0, self.n_layers) / self.E_ref
    def E(self, x):
        x_min, x_max = self.x_min, self.x_max
        layer_width  = (x_max - x_min) / self.n_layers
        E_val = torch.full_like(x, self.E_vals_nd[0])
        w = 0.02 / self.L_ref
        for k in range(self.n_layers - 1):
            boundary = x_min + (k + 1) * layer_width
            alpha    = 0.5 * (1.0 + torch.tanh((x - boundary) / w))
            E_val    = E_val * (1.0 - alpha) + self.E_vals_nd[k + 1] * alpha
        return E_val
    def rho(self, x): return torch.full_like(x, 1.0)

def fd_reference(material: MaterialModel, nx: int = 512, nt: int = 2000,
                 T: float = 1.0, sigma_g: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ CPU-based Finite Difference Reference Solver to evaluate our PINN against. """
    x_min, x_max = material.x_min, material.x_max
    x = np.linspace(x_min, x_max, nx)
    dx = x[1] - x[0]
    dt = T / nt
    t  = np.linspace(0, T, nt)

    x_t = torch.tensor(x, dtype=torch.float32)
    E_np  = material.E(x_t).numpy()
    rho_np = material.rho(x_t).numpy()

    x0   = 0.0
    f    = np.exp(-0.5 * ((x - x0) / sigma_g) ** 2)
    dfdr = -(x - x0) / sigma_g**2 * f
    g    = dfdr / (np.abs(dfdr).max() + 1e-12)

    u_prev, u_curr = g.copy(), g.copy()
    u_all  = np.zeros((nt, nx))
    u_all[0], u_all[1] = u_prev, u_curr

    E_half = 0.5 * (E_np[:-1] + E_np[1:])

    for n in range(1, nt - 1):
        u_new = np.zeros(nx)
        flux       = E_half * (u_curr[1:] - u_curr[:-1]) / dx
        d_flux     = (flux[1:] - flux[:-1]) / dx
        u_new[1:-1] = (2 * u_curr[1:-1] - u_prev[1:-1] + (dt**2 / rho_np[1:-1]) * d_flux)

        Vp_left  = math.sqrt(E_np[0]  / rho_np[0])
        Vp_right = math.sqrt(E_np[-1] / rho_np[-1])
        r_left   = Vp_left  * dt / dx
        r_right  = Vp_right * dt / dx
        
        u_new[0]  = u_curr[1]  + (r_left  - 1) / (r_left  + 1) * (u_new[1]  - u_curr[0])
        u_new[-1] = u_curr[-2] + (r_right - 1) / (r_right + 1) * (u_new[-2] - u_curr[-1])

        u_prev, u_curr = u_curr, u_new
        u_all[n + 1] = u_curr

    return x, t, u_all


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  NETWORK ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════
class VanillaPINN(nn.Module):
    def __init__(self, hidden_layers: int = 7, hidden_units: int = 128):
        super().__init__()
        layers =[nn.Linear(2, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers +=[nn.Linear(hidden_units, hidden_units), nn.Tanh()]
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x, t): return self.net(torch.cat([x, t], dim=-1))

class FourierFeaturePINN(nn.Module):
    def __init__(self, hidden_layers: int = 7, hidden_units: int  = 128, n_fourier: int = 128, sigma: float = 1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(n_fourier, 2) * sigma)
        layers =[nn.Linear(2 * n_fourier, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers +=[nn.Linear(hidden_units, hidden_units), nn.Tanh()]
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x, t):
        #x, t = x.double(), t.double()
        proj = torch.cat([x, t], dim=-1) @ self.B.T
        return self.net(torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1))

class RWFLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, rwf_mu: float = 1.0, rwf_sigma: float = 0.1):
        super().__init__()
        self.V = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.V)
        self.s = nn.Parameter(torch.normal(mean=rwf_mu, std=rwf_sigma, size=(out_features,)))
        if bias: self.bias = nn.Parameter(torch.zeros(out_features))
        else: self.register_parameter("bias", None)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.s.unsqueeze(1) * self.V, self.bias)

class PirateBlock(nn.Module):
    def __init__(self, units: int, rwf_mu: float = 1.0, rwf_sigma: float = 0.1):
        super().__init__()
        self.W1 = RWFLinear(units, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.W2 = RWFLinear(units, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.W3 = RWFLinear(units, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, x: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        f  = torch.tanh(self.W1(x))
        z1 = f * U + (1.0 - f) * V
        g  = torch.tanh(self.W2(z1))
        z2 = g * U + (1.0 - g) * V
        h  = torch.tanh(self.W3(z2))
        return self.alpha * h + (1.0 - self.alpha) * x

class PirateNet(nn.Module):
    def __init__(self, n_blocks: int = 3, units: int = 256, n_fourier: int = 128,
                 sigma: float = 2.0, in_dim: int = 2, rwf_mu: float = 1.0, rwf_sigma: float = 0.1):
        super().__init__()
        self.register_buffer("B", torch.randn(n_fourier, in_dim) * sigma)
        embed_dim = 2 * n_fourier
        self.enc_U = RWFLinear(embed_dim, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.enc_V = RWFLinear(embed_dim, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.proj  = RWFLinear(embed_dim, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.blocks = nn.ModuleList([PirateBlock(units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma) for _ in range(n_blocks)])
        self.out_layer = nn.Linear(units, 1, bias=False)
        nn.init.zeros_(self.out_layer.weight)
 
    def fourier_embed(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        proj = torch.cat([x, t], dim=-1) @ self.B.T
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
 
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        phi = self.fourier_embed(x, t)
        U, V = torch.tanh(self.enc_U(phi)), torch.tanh(self.enc_V(phi))
        h = torch.tanh(self.proj(phi))
        for block in self.blocks: h = block(h, U, V)
        return self.out_layer(h)

    @torch.no_grad()
    def physics_informed_init(self, xt_data: torch.Tensor, y_data: torch.Tensor) -> None:
        self.eval()
        phi = self.fourier_embed(xt_data[:, 0:1], xt_data[:, 1:2])
        U, V = torch.tanh(self.enc_U(phi)), torch.tanh(self.enc_V(phi))
        Phi_eff = torch.tanh(self.proj(phi))
        for block in self.blocks: Phi_eff = block(Phi_eff, U, V)
        W_star = torch.linalg.lstsq(Phi_eff, y_data).solution
        self.out_layer.weight.copy_(W_star.T)
        self.train()



# ═══════════════════════════════════════════════════════════════════════════════
# 3.  ANSATZ & LOSS FUNCTIONS 
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_ic(x: torch.Tensor, sigma_g: float = 0.1, x0: float = 0.0) -> torch.Tensor:
    f    = torch.exp(-0.5 * ((x - x0) / sigma_g) ** 2)
    dfdx = -(x - x0) / sigma_g**2 * f
    return dfdx / (dfdx.abs().max() + 1e-12)

def apply_ansatz(nn_out: torch.Tensor, x: torch.Tensor, t: torch.Tensor, material: MaterialModel, sigma_g: float = 0.1) -> torch.Tensor:
    """ Hard-Constraint Ansatz: Ensures that u(x, 0) == g(x) mathematically. """
    g      = gaussian_ic(x, sigma_g)
    t_phys = material.to_physical_t(t)
    decay  = torch.exp(-0.5 * (15.0 * t_phys) ** 2)
    growth = torch.tanh(25.0 * t_phys) ** 2
    return g * decay + growth * nn_out

def compute_pde_residual(model: nn.Module, x: torch.Tensor, t: torch.Tensor,
                         material: MaterialModel, sigma_g: float = 0.1) -> torch.Tensor:
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = apply_ansatz(model(x, t), x, t, material, sigma_g)

    u_x, u_t = torch.autograd.grad(u,[x, t], grad_outputs=torch.ones_like(u), create_graph=True)[:]
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]

    sigma = material.E(x) * u_x
    sigma_x = torch.autograd.grad(sigma, x, grad_outputs=torch.ones_like(sigma), create_graph=True)[0]

    return u_tt - (sigma_x / material.rho(x))

def causal_pde_loss(model: nn.Module, x: torch.Tensor, t: torch.Tensor,
                    material: MaterialModel, sigma_g: float, n_chunks: int, tolerance: float) -> torch.Tensor:
    R_all = compute_pde_residual(model, x, t, material, sigma_g)
    t_max = t.max().item()
    boundaries = torch.linspace(0.0, t_max, n_chunks + 1, device=t.device)

    chunk_losses =[]
    for i in range(n_chunks):
        mask = (t.squeeze() >= boundaries[i]) & (t.squeeze() < boundaries[i + 1])
        if mask.sum() == 0:
            chunk_losses.append(torch.zeros(1, device=t.device))
            continue
        chunk_losses.append((R_all[mask] ** 2).mean().unsqueeze(0))

    chunk_losses = torch.cat(chunk_losses)
    cumsum = torch.zeros_like(chunk_losses)
    cumsum[1:] = torch.cumsum(chunk_losses[:-1].detach(), dim=0)
    weights = torch.exp(-tolerance * cumsum)

    return (weights * chunk_losses).mean()

def compute_grad_norm_weights(model: nn.Module, loss_pde: torch.Tensor, loss_bc: torch.Tensor) -> tuple:
    def _mean_abs_grad(loss: torch.Tensor) -> float:
        if not loss.requires_grad or loss.item() == 0.0: return 1.0
        grads = torch.autograd.grad(loss, list(model.parameters()), retain_graph=True, allow_unused=True)
        vals  =[g.abs().mean().item() for g in grads if g is not None]
        return float(sum(vals) / max(len(vals), 1))

    n_pde, n_bc = _mean_abs_grad(loss_pde), _mean_abs_grad(loss_bc)
    ref = (n_pde + n_bc) / 2.0 + 1e-12
    return ref / (n_pde + 1e-12), ref / (n_bc + 1e-12)

def physics_loss(model:        nn.Module,
                 x_int:        torch.Tensor,
                 t_int:        torch.Tensor,
                 x_bc:         torch.Tensor,
                 t_bc:         torch.Tensor,
                 material:     MaterialModel,
                 sigma_g:      float = 0.1,
                 n_chunks:     int   = 32,
                 tolerance:    float = 0.1,
                 use_causal:   bool  = True,
                 w_pde:        float = 1.0,
                 w_bc:         float = 1.0) -> tuple:
    """ Computes total weighted loss using 1D Absorbing Boundary Conditions. """
    if use_causal:
        loss_pde = causal_pde_loss(model, x_int, t_int, material, sigma_g, n_chunks, tolerance)
    else:
        loss_pde = (compute_pde_residual(model, x_int, t_int, material, sigma_g) ** 2).mean()

    # 1D Absorbing Boundary Condition Loss
    x_bc = x_bc.clone().detach().requires_grad_(True)
    t_bc = t_bc.clone().detach().requires_grad_(True)
    
    u_bc = apply_ansatz(model(x_bc, t_bc), x_bc, t_bc, material, sigma_g)
    
    grads_bc = torch.autograd.grad(u_bc,[x_bc, t_bc], grad_outputs=torch.ones_like(u_bc), create_graph=True)
    u_x_bc, u_t_bc = grads_bc[0], grads_bc[1]
    
    Vp_bc = material.Vp(x_bc)
    x_mid = (material.x_min + material.x_max) / 2.0
    n_x   = torch.where(x_bc > x_mid, 1.0, -1.0) 
    
    abc_residual = u_t_bc + (Vp_bc * n_x * u_x_bc)
    loss_bc = (abc_residual ** 2).mean()

    total = w_pde * loss_pde + w_bc * loss_bc
    return total, loss_pde, loss_bc



# ═══════════════════════════════════════════════════════════════════════════════
# 4.  GPU-DIRECT SAMPLING & SCHEDULING
# ═══════════════════════════════════════════════════════════════════════════════

def sample_batch(n_int: int, n_bc: int, x_min: float, x_max: float, t_max: float, device: torch.device) -> tuple:
    x_int = torch.rand(n_int, 1, device=device) * (x_max - x_min) + x_min
    t_int = torch.rand(n_int, 1, device=device) * t_max
    t_bc  = torch.rand(n_bc, 1, device=device) * t_max
    x_bc  = torch.where(torch.rand(n_bc, 1, device=device) < 0.5,
                        torch.full((n_bc, 1), x_min, device=device),
                        torch.full((n_bc, 1), x_max, device=device))
    return x_int, t_int, x_bc, t_bc

def make_lr_scheduler(optimizer: optim.Optimizer, warmup_steps: int = 5000, decay_rate: float = 0.9, decay_steps: int = 5000) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps: return float(step + 1) / float(warmup_steps)
        return decay_rate ** ((step - warmup_steps) // decay_steps)
    return LambdaLR(optimizer, lr_lambda)



# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING LOOPS  —  Two Phase (Adam -> L-BFGS)
# ═══════════════════════════════════════════════════════════════════════════════

class LivePlotter:
    def __init__(self, arch_name):
        self.losses, self.pde_losses, self.bc_losses = [], [], []
        self.w_pde_hist, self.w_bc_hist = [], []

        self.fig, (self.ax_loss, self.ax_w) = plt.subplots(1, 2, figsize=(12, 4))
        self.fig.suptitle(f"Live Training: {arch_name}", fontsize=12)

        # Left subplot: losses (log scale)
        self.ax_loss.set_xlabel("Update"); self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True, alpha=0.3)
        self.line_tot, = self.ax_loss.semilogy([], [], color='#27AE60', lw=1.5, label='Total')
        self.line_pde, = self.ax_loss.semilogy([], [], color='#E67E22', lw=1.5, linestyle='--', label='PDE')
        self.line_bc,  = self.ax_loss.semilogy([], [], color='#8E44AD', lw=1.5, linestyle=':', label='ABC')
        self.ax_loss.legend(fontsize=8)

        # Right subplot: grad-norm weights (linear scale)
        self.ax_w.set_xlabel("Update"); self.ax_w.set_ylabel("Weight")
        self.ax_w.grid(True, alpha=0.3)
        self.line_wpde, = self.ax_w.plot([], [], color='#E67E22', lw=1.5, label='w_pde')
        self.line_wbc,  = self.ax_w.plot([], [], color='#8E44AD', lw=1.5, label='w_bc')
        self.ax_w.legend(fontsize=8)

        self.fig.tight_layout()

        # Isolate plot in an Output widget so clear_output doesn't wipe tqdm
        self._out = None
        try:
            import ipywidgets as _ipw
            self._out = _ipw.Output()
            display.display(self._out)
        except Exception:
            pass

    def update(self, loss_val, pde_val=None, bc_val=None, w_pde=None, w_bc=None):
        self.losses.append(loss_val)
        self.line_tot.set_data(range(len(self.losses)), self.losses)

        if pde_val is not None:
            self.pde_losses.append(pde_val)
            self.line_pde.set_data(range(len(self.pde_losses)), self.pde_losses)
        if bc_val is not None:
            self.bc_losses.append(bc_val)
            self.line_bc.set_data(range(len(self.bc_losses)), self.bc_losses)
        if w_pde is not None:
            self.w_pde_hist.append(w_pde)
            self.line_wpde.set_data(range(len(self.w_pde_hist)), self.w_pde_hist)
        if w_bc is not None:
            self.w_bc_hist.append(w_bc)
            self.line_wbc.set_data(range(len(self.w_bc_hist)), self.w_bc_hist)

        self.ax_loss.relim(); self.ax_loss.autoscale_view()
        self.ax_w.relim();    self.ax_w.autoscale_view()

        if self._out is not None:
            with self._out:
                display.clear_output(wait=True)
                display.display(self.fig)
        else:
            display.clear_output(wait=True)
            display.display(self.fig)

def train_model_two_phase(model, material, x_fd, t_fd, u_fd, n_steps=300000, lbfgs_steps=500, 
                          lr=1e-3, warmup_steps=5000, decay_rate=0.9, decay_steps=5000, 
                          n_int=8192, n_bc=256, T_stages=None, sigma_g=0.1, 
                          n_chunks=32, tolerance=0.1, use_causal=True, 
                          weight_update_freq=1000, arch_name="Model",
                          save_path="best_model.pt"): # Added save_path
    
    if T_stages is None: T_stages = [1.0]
    x_min, x_max = material.x_min, material.x_max
    loss_history = []
    plotter = LivePlotter(arch_name)
    global_step = 0
    
    # Track the best accuracy
    best_l2 = float('inf')
    current_l2 = 100.0

    # 1. Physics Initialization
    if hasattr(model, "physics_informed_init"):
        n_pi = 4096
        x_pi = torch.rand(n_pi, 1, device=DEVICE) * (x_max - x_min) + x_min
        T_max = T_stages[-1] if T_stages else 1.0
        t_pi = torch.rand(n_pi, 1, device=DEVICE) * T_max
        y_pi = gaussian_ic(x_pi, sigma_g) 
        model.physics_informed_init(torch.cat([x_pi, t_pi], dim=1), y_pi)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = make_lr_scheduler(optimizer, warmup_steps, decay_rate, decay_steps)
    w_pde, w_bc = 1.0, 1.0
    steps_per_stage = n_steps // len(T_stages)

    # ── Phase 1: Adam ──
    for stage_idx, T_end in enumerate(T_stages):
        if steps_per_stage <= 0: continue
        pbar = tqdm(range(steps_per_stage), desc=f"Adam Stage {stage_idx+1}")
        for step in pbar:
            optimizer.zero_grad()
            x_i, t_i, x_b, t_b = sample_batch(n_int, n_bc, x_min, x_max, T_end, DEVICE)
            loss, l_pde, l_bc = physics_loss(model, x_i, t_i, x_b, t_b, material, sigma_g, 
                                             n_chunks, tolerance, use_causal, w_pde, w_bc)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (step + 1) % weight_update_freq == 0:
                xw, tw, xbw, tbw = sample_batch(n_int, n_bc, x_min, x_max, T_end, DEVICE)
                _, lp_w, lb_w = physics_loss(model, xw, tw, xbw, tbw, material, sigma_g, n_chunks, tolerance, use_causal, 1.0, 1.0)
                w_pde, w_bc = compute_grad_norm_weights(model, lp_w, lb_w)

            # Evaluate L2 every 10 steps
            if global_step % 10 == 0:
                metrics = evaluate(model, material, x_fd, t_fd, u_fd, sigma_g=sigma_g)
                current_l2 = metrics['mean_l2']
                
                # SAVE BEST MODEL LOGIC
                if current_l2 < best_l2:
                    best_l2 = current_l2
                    torch.save(model.state_dict(), save_path)
                
                plotter.update(loss.item(), l_pde.item(), l_bc.item(), l2_val=current_l2, step=global_step)
                pbar.set_postfix({"Loss": f"{loss.item():.2e}", "L2": f"{current_l2:.2f}%", "Best": f"{best_l2:.2f}%"})
            
            global_step += 1
            loss_history.append(loss.item())

    # ── Phase 2: L-BFGS ──
    if lbfgs_steps > 0:
        x_int_f, t_int_f, x_bc_f, t_bc_f = sample_batch(n_int * 2, n_bc * 2, x_min, x_max, T_stages[-1], DEVICE)
        lbfgs_opt = optim.LBFGS(model.parameters(), max_iter=20, history_size=50, line_search_fn="strong_wolfe")
        
        pbar_lbfgs = tqdm(range(lbfgs_steps), desc="L-BFGS Phase")
        for lbfgs_ep in pbar_lbfgs:
            closure_logs = {}
            def closure():
                lbfgs_opt.zero_grad()
                loss, lp, lb = physics_loss(model, x_int_f, t_int_f, x_bc_f, t_bc_f, material, 
                                            sigma_g, use_causal=False, w_pde=w_pde, w_bc=w_bc)
                loss.backward()
                closure_logs.update({'tot': loss.item(), 'pde': lp.item(), 'bc': lb.item()})
                return loss
            
            lbfgs_opt.step(closure)
            
            if lbfgs_ep % 10 == 0:
                metrics = evaluate(model, material, x_fd, t_fd, u_fd, sigma_g=sigma_g)
                current_l2 = metrics['mean_l2']
                
                # SAVE BEST MODEL LOGIC (L-BFGS Phase)
                if current_l2 < best_l2:
                    best_l2 = current_l2
                    torch.save(model.state_dict(), save_path)
                
                plotter.update(closure_logs['tot'], closure_logs['pde'], closure_logs['bc'], 
                               l2_val=current_l2, step=global_step + lbfgs_ep)
                pbar_lbfgs.set_postfix({"Loss": f"{closure_logs['tot']:.2e}", "L2": f"{current_l2:.2f}%", "Best": f"{best_l2:.2f}%"})
                
            loss_history.append(closure_logs['tot'])

    print(f"\n>>> Training Complete. Best Mean L2 Error: {best_l2:.4f}%")
    # Load the best weights back into the model before returning
    model.load_state_dict(torch.load(save_path))
    
    plt.close(plotter.fig)
    return loss_history


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION 
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model: nn.Module, material: MaterialModel, x_fd: np.ndarray, t_fd: np.ndarray, 
             u_fd: np.ndarray, sigma_g: float = 0.1, eval_times: list = None) -> dict:
    if eval_times is None:
        n_snap = min(10, len(t_fd))
        eval_times = [t_fd[int(i * (len(t_fd) - 1) / (n_snap - 1))] for i in range(n_snap)]

    l2_errors, energy_diffs, t_eval_actual =[], [],[]
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


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════
def plot_results(results: dict, material: MaterialModel, x_fd: np.ndarray,
                 t_fd: np.ndarray, u_fd: np.ndarray, sigma_g: float = 0.1,
                 save_prefix: str = "pinn_1d"):
    arch_names   = list(results.keys())
    arch_colors  = {"vanilla": "#E67E22", "fourier": "#2980B9", "pirate": "#27AE60"}
    arch_labels  = {"vanilla": "Vanilla PINN", "fourier": "Fourier-feature PINN", "pirate": "PirateNet"}

    snap_times_phys =[0.3, 0.6, 0.9]
    snap_times_nd   =[material.to_nondim_t(tp) for tp in snap_times_phys]
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
        if col == 0: axes[col].set_ylabel("u(x,t)")
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

        x_ext =[material.to_physical_x(x_fd[0]), material.to_physical_x(x_fd[-1])]
        t_ext =[material.to_physical_t(t_sub[0]), material.to_physical_t(t_sub[-1])]
        vmax = max(np.abs(u_fd_sub).max(), np.abs(u_pinn_sub).max()) * 0.8
        kw   = dict(aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=[x_ext[0], x_ext[1], t_ext[0], t_ext[1]])
        
        im0 = axes[0].imshow(u_pinn_sub, **kw); axes[0].set_title("PirateNet"); axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
        im1 = axes[1].imshow(u_fd_sub, **kw); axes[1].set_title("FD Reference"); axes[1].set_xlabel("x")
        diff = u_pinn_sub - u_fd_sub
        vd   = np.abs(diff).max() * 0.8
        im2  = axes[2].imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vd, vmax=vd, extent=[x_ext[0], x_ext[1], t_ext[0], t_ext[1]])
        axes[2].set_title("Difference"); axes[2].set_xlabel("x")
        
        for ax, im in zip(axes,[im0, im1, im2]): plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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
    for arch in arch_names: print(f"  {arch_labels.get(arch, arch):<25} {results[arch]['metrics']['mean_l2']:>18.2f}")
    print(f"{'─'*50}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(material: MaterialModel, n_collocation: int = 10_000, 
                   max_epochs: int = 700, lbfgs_epochs: int = 200, T_stages: list = None, 
                   sigma_g: float = 0.1, save_prefix: str = "pinn_1d"):
    if T_stages is None: T_stages =[1.0]
    sigma_g_nd   = material.nondim_sigma_g(sigma_g)
    T_stages_nd  =[material.to_nondim_t(T) for T in T_stages]
    T_physical   = T_stages[-1]
    T_nd         = T_stages_nd[-1]

    print(f"\n{'═'*60}\n  Experiment: {material.name}\n  Device: {DEVICE}")
    print(f"  Collocation: {n_collocation:,} | Adam steps: {max_epochs:,} | LBFGS steps: {lbfgs_epochs:,}")
    print(f"  Non-dim T_nd={T_nd:.4f}\n{'═'*60}")

    print("\n  Building FD reference solution...")
    x_fd, t_fd, u_fd = fd_reference(material, nx=512, nt=2000, T=T_nd, sigma_g=sigma_g_nd)

    eval_phys  = list(np.linspace(0.05, T_physical, 20))
    eval_times =[material.to_nondim_t(tp) for tp in eval_phys]

    # Initialize all 3 baseline models
    arch_defs = {
        #"vanilla": lambda: VanillaPINN(hidden_layers=5, hidden_units=128),
        #"fourier": lambda: FourierFeaturePINN(hidden_layers=6, hidden_units=128, n_fourier=64, sigma=3.0),
        "pirate":  lambda: PirateNet(n_blocks=2, units=128, n_fourier=64, sigma=3.0),
        "pikan":   lambda: PIKAN(hidden_layers=3, hidden_units=50, grid_size=10, n_fourier=25, sigma=7.0),
    }

    results = {}

    for arch_name, build_fn in arch_defs.items():
        print(f"\n  ── Training: {arch_name.upper()} ──")
        model = build_fn().to(DEVICE)
        
        warmup = min(5000, max(10, max_epochs // 10))
        decay  = min(5000, max(10, max_epochs // 10))
        freq   = max(10, max_epochs // 20)

        t0 = time.time()
        loss_history = train_model_two_phase(
            model, material, n_steps=max_epochs, lbfgs_steps=lbfgs_epochs,
            n_int=n_collocation, T_stages=T_stages_nd, sigma_g=sigma_g_nd,
            warmup_steps=warmup, decay_steps=decay, weight_update_freq=freq,
            arch_name=f"{arch_name.upper()} (Adam -> L-BFGS)"
        )
        
        print(f"  Training time: {time.time() - t0:.1f}s  |  Final loss: {loss_history[-1]:.4e}")
        metrics = evaluate(model, material, x_fd, t_fd, u_fd, sigma_g=sigma_g_nd, eval_times=eval_times)
        print(f"  Mean L2 Error: {metrics['mean_l2']:.2f}%")

        torch.save(model.state_dict(), f"{save_prefix}_{arch_name}_model.pt")
        results[arch_name] = {"model": model, "loss_history": loss_history, "metrics": metrics}

    plot_results(results, material, x_fd, t_fd, u_fd, sigma_g=sigma_g_nd, save_prefix=save_prefix)
    return results




#Pikan

class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5, spline_order: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_basis = grid_size + spline_order

        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))

        self.spline_weight = nn.Parameter(torch.empty(out_dim, in_dim, self.n_basis))
        nn.init.xavier_uniform_(self.spline_weight)

        grid = torch.linspace(-1, 1, grid_size + 1)
        step = grid[1] - grid[0]
        pad_left = torch.linspace(grid[0] - spline_order * step, grid[0] - step, spline_order)
        pad_right = torch.linspace(grid[-1] + step, grid[-1] + spline_order * step, spline_order)
        full_grid = torch.cat([pad_left, grid, pad_right])
        self.register_buffer("grid", full_grid)

    def b_spline(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            d1 = grid[k:-1] - grid[:-k-1]
            d2 = grid[k+1:] - grid[1:-k]
            term1 = (x - grid[:-k-1]) / d1 * bases[:, :, :-1]
            term2 = (grid[k+1:] - x) / d2 * bases[:, :, 1:]
            bases = term1 + term2

        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = F.linear(F.silu(x), self.base_weight)
        basis = self.b_spline(x)
        spline_output = torch.einsum("bik,oik->bo", basis, self.spline_weight)
        return base_output + spline_output


class PIKAN(nn.Module):
    def __init__(self, hidden_layers=3, hidden_units=64, grid_size=5, spline_order=4, n_fourier=128, sigma=1.0):
        super().__init__()

        # Fourier embedding
        self.register_buffer("B", torch.randn(n_fourier, 2) * sigma)
        embed_dim = 2 * n_fourier

        # layer0
        self.layer0 = KANLayer(embed_dim, hidden_units, grid_size=grid_size, spline_order=spline_order)
        self.norm0 = nn.LayerNorm(hidden_units)

        # mid layers + norms
        self.kan_mid = nn.ModuleList([
            KANLayer(hidden_units, hidden_units, grid_size=grid_size, spline_order=spline_order)
            for _ in range(hidden_layers - 1)
        ])
        self.norms_mid = nn.ModuleList([
            nn.LayerNorm(hidden_units)
            for _ in range(hidden_layers - 1)
        ])

        self.out_layer = nn.Linear(hidden_units, 1, bias=False)
        nn.init.xavier_uniform_(self.out_layer.weight)

    def fourier_embed(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        inputs = torch.cat([x, t], dim=-1)
        proj = 2 * torch.pi * (inputs @ self.B.T)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        phi = self.fourier_embed(x, t)
        h = torch.tanh(self.norm0(self.layer0(phi)))

        for i, layer in enumerate(self.kan_mid):
            h = torch.tanh(self.norms_mid[i](layer(h)))

        return self.out_layer(h)

    def get_l1_regularization(self) -> torch.Tensor:
        reg = torch.norm(self.layer0.spline_weight, 1)
        for layer in self.kan_mid:
            reg = reg + torch.norm(layer.spline_weight, 1)
        return reg