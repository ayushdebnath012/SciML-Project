
import itertools
from kan import KAN
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
from numpy import linalg as LA
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
import os
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# For LaTeX in plots
# Uncomment if latex is installed on Kaggle
# plt.rcParams.update({'font.size': 18, 'font.family': 'serif'})
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{lmodern}'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SEED = 21
torch.manual_seed(SEED)
np.random.seed(SEED)

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

my_material = HomogeneousModel()


INTERVAL_X = (-1.0, 1.0)
INTERVAL_T = (0.0, 1.0)

# Solution Cache for Efficiency
_SOL_CACHE = {}

def solve_wave_fd(material, sigma_g=0.1, x0=0.0, nx=200, nt=600, t_max=1.0):
    """Finite Difference solver for the wave equation with ABCs."""
    x_coords = np.linspace(material.x_min, material.x_max, nx)
    t_coords = np.linspace(0, t_max + 0.1, nt)
    dx = x_coords[1] - x_coords[0]
    dt = t_coords[1] - t_coords[0]
    c = 1.0 # Wave speed in non-dim units
    
    # Stability Check (CFL condition)
    if dt > dx/c:
        dt = 0.9 * dx / c
        nt = int((t_max + 0.1) / dt) + 2
        t_coords = np.linspace(0, t_max + 0.1, nt)
        dt = t_coords[1] - t_coords[0]

    u = np.zeros((nt, nx))

    # Initial Conditions (Gaussian Derivative)
    def get_ic(x_arr):
        f = np.exp(-0.5 * ((x_arr - x0) / sigma_g) ** 2)
        dfdx = -(x_arr - x0) / sigma_g**2 * f
        return dfdx / (np.abs(dfdx).max() + 1e-12)

    u[0, :] = get_ic(x_coords)
    r2 = (dt * c / dx)**2
    for i in range(1, nx-1):
        u[1, i] = u[0, i] + 0.5 * r2 * (u[0, i+1] - 2*u[0, i] + u[0, i-1])

    # Time Stepping
    for n in range(1, nt - 1):
        u[n+1, 1:-1] = 2*u[n, 1:-1] - u[n-1, 1:-1] + r2 * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
        # Mur Absorbing Boundary Conditions
        u[n+1, 0] = u[n, 1] + (c*dt - dx)/(c*dt + dx) * (u[n+1, 1] - u[n, 0])
        u[n+1, -1] = u[n, -2] + (c*dt - dx)/(c*dt + dx) * (u[n+1, -2] - u[n, -1])

    return t_coords, x_coords, u

def u_sol(x_query, t_query, material=my_material, sigma_g=0.1, x0=0.0):
    t_max = float(t_query.max()) if torch.is_tensor(t_query) else float(np.max(t_query))
    key = (material.name, sigma_g, x0, t_max)
    if key not in _SOL_CACHE:
        _SOL_CACHE[key] = solve_wave_fd(material, sigma_g, x0, t_max=t_max)
    
    t_coords, x_coords, u = _SOL_CACHE[key]
    
    if hasattr(x_query, 'detach'):
        xq = x_query.detach().cpu().numpy().flatten()
        tq = t_query.detach().cpu().numpy().flatten()
    else:
        xq, tq = x_query.flatten(), t_query.flatten()

    interp = RegularGridInterpolator((t_coords, x_coords), u, bounds_error=False, fill_value=0)
    return interp(np.vstack((tq, xq)).T).reshape(-1, 1)

def u_grad_sol(x_query, t_query, material=my_material, sigma_g=0.1, x0=0.0):
    """Numerical gradients of the reference solution."""
    t_max = float(t_query.max()) if torch.is_tensor(t_query) else float(np.max(t_query))
    key = (material.name, sigma_g, x0, t_max)
    if key not in _SOL_CACHE:
        _SOL_CACHE[key] = solve_wave_fd(material, sigma_g, x0, t_max=t_max)
    
    t_coords, x_coords, u = _SOL_CACHE[key]
    
    # Compute numerical gradients on the grid
    dt = t_coords[1] - t_coords[0]
    dx = x_coords[1] - x_coords[0]
    ut, ux = np.gradient(u, dt, dx, axis=(0, 1))
    
    if hasattr(x_query, 'detach'):
        xq = x_query.detach().cpu().numpy().flatten()
        tq = t_query.detach().cpu().numpy().flatten()
    else:
        xq, tq = x_query.flatten(), t_query.flatten()

    interp_x = RegularGridInterpolator((t_coords, x_coords), ux, bounds_error=False, fill_value=0)
    interp_t = RegularGridInterpolator((t_coords, x_coords), ut, bounds_error=False, fill_value=0)
    
    pts = np.vstack((tq, xq)).T
    return np.concatenate((interp_x(pts).reshape(-1, 1), interp_t(pts).reshape(-1, 1)), axis=1)

NUM_INPUTS = 2
NUM_OUTPUTS = 1

def gaussian_ic(x, sigma_g=0.1, x0=0.0):
    """Initial condition: Gaussian derivative."""
    f = torch.exp(-0.5 * ((x - x0) / sigma_g) ** 2)
    dfdx = -(x - x0) / sigma_g**2 * f
    return dfdx / (dfdx.abs().max() + 1e-12)

def apply_ansatz(nn_out, x, t, material, sigma_g=0.1):
    """Hard-Constraint Ansatz: Ensures u(x, 0) = g(x) and u_t(x, 0) = 0."""
    g = gaussian_ic(x, sigma_g)
    t_phys = material.to_physical_t(t)
    # decay and growth factors to blend from IC to NN solution
    decay = torch.exp(-0.5 * (15.0 * t_phys) ** 2)
    growth = torch.tanh(25.0 * t_phys) ** 2
    return g * decay + growth * nn_out

# Losses function
def losses(model, x, t, material=my_material, sigma_g=0.1, x0=0.0):
    xt = torch.cat((x, t), 1)
    t0 = torch.zeros_like(t).requires_grad_(True)
    xt0 = torch.cat((x.detach(), t0), 1)
    x0t = torch.cat((torch.zeros_like(x), t), 1)
    x1t = torch.cat((torch.ones_like(x), t), 1)

    # Wrap model output with ansatz
    u = apply_ansatz(model(xt), x, t, material, sigma_g)
    
    u_t = dy_dx(u, t)
    u_tt = dy_dx(u_t, t)
    u_x = dy_dx(u, x)
    u_xx = dy_dx(u_x, x)

    physics_residual = u_tt - u_xx

    # BC 1: Left Absorbing Boundary (at x_min)
    x_min_val = torch.full_like(x, material.x_min).requires_grad_(True)
    x0t = torch.cat((x_min_val, t), 1) 
    u_l = apply_ansatz(model(x0t), x_min_val, t, material, sigma_g)
    
    # Get gradients with respect to the concatenated input
    grads_l = torch.autograd.grad(u_l, x0t, torch.ones_like(u_l), create_graph=True)[0]
    u_x_l = grads_l[:, 0:1] # Gradient w.r.t x (first column)
    u_t_l = grads_l[:, 1:2] # Gradient w.r.t t (second column)
    
    bc_residual_l = u_t_l - material.Vp(x_min_val) * u_x_l

    # BC 2: Right Absorbing Boundary (at x_max)
    x_max_val = torch.full_like(x, material.x_max).requires_grad_(True)
    x1t = torch.cat((x_max_val, t), 1)
    u_r = apply_ansatz(model(x1t), x_max_val, t, material, sigma_g)
    
    # Get gradients with respect to the concatenated input
    grads_r = torch.autograd.grad(u_r, x1t, torch.ones_like(u_r), create_graph=True)[0]
    u_x_r = grads_r[:, 0:1] 
    u_t_r = grads_r[:, 1:2] 
    
    bc_residual_r = u_t_r + material.Vp(x_max_val) * u_x_r

    bc_loss = (bc_residual_l**2).mean() + (bc_residual_r**2).mean()
    
    return (physics_residual**2).mean(), bc_loss



def dy_dx(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]

class FourierKAN(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden, hidden_width, G, k, n_fourier=64, sigma=10.0):
        super().__init__()
        # Random projection matrix (B) for Fourier Features
        self.register_buffer("B", torch.randn(n_fourier, n_inputs) * sigma)
        
        # KAN input is 2 * n_fourier (cos and sin components)
        width = [2 * n_fourier] + [hidden_width] * n_hidden + [n_outputs]
        self.kan = KAN(width=width, grid=G, k=k, symbolic_enabled=False)
        
    def forward(self, xt):
        proj = xt @ self.B.T
        rff = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return self.kan(rff)
    
    def speed(self):
        if hasattr(self.kan, 'speed'):
            self.kan.speed()

def build_kan(n_inputs, n_outputs, n_hidden, hidden_width, G, k):
    assert n_hidden >= 1
    return FourierKAN(n_inputs, n_outputs, n_hidden, hidden_width, G, k, n_fourier=64, sigma=10.0)

def num_parameters_kan(n_inputs, n_outputs, n_hidden, hidden_width, G, k):
    return (n_inputs * hidden_width + (n_hidden-1) * hidden_width**2 + n_outputs * hidden_width) * (2 + G + k)

def train_adam(model, inputs, losses_func, iterations, lr=None, physics_loss_weight=1.0, bc_loss_weight=1.0, print_every=500, outdir=None, reuse_optimizer=None):
    model.train(True)
    if reuse_optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    else:
        optimizer = reuse_optimizer

    total_losses = []
    physics_losses = []
    bc_losses = []
    best_loss = torch.inf

    for iter in range(iterations):
        optimizer.zero_grad()
        physics_loss, bc_loss = losses_func(model, *inputs)
        
        # --- Dynamic Reweighting every 250 steps ---
        if iter % 250 == 0:
            with torch.no_grad():
                p_val = physics_loss.item()
                b_val = bc_loss.item()
                total = p_val + b_val
                if total > 0:
                    physics_loss_weight = p_val / total
                    bc_loss_weight = b_val / total
        
        total_loss = physics_loss * physics_loss_weight + bc_loss * bc_loss_weight
        
        total_loss.backward()
        optimizer.step()

        total_loss_val = total_loss.item()
        if iter % print_every == 0:
            print(f"    ITERATION: {iter+1} | LOSS: {total_loss_val:.4e} | PDE: {physics_loss.item():.4e} | BC: {bc_loss.item():.4e} | w_pde: {physics_loss_weight:.2f} | w_bc: {bc_loss_weight:.2f}", flush=True)

        total_losses.append(total_loss_val)
        physics_losses.append(physics_loss.item())
        bc_losses.append(bc_loss.item())

        if outdir is not None and total_loss_val < best_loss:
            torch.save(model.state_dict(), outdir)
            best_loss = total_loss_val

    return total_losses, physics_losses, bc_losses, physics_loss_weight, bc_loss_weight

def train_lbfgs(model, inputs, losses_func, iterations, physics_loss_weight=1.0, bc_loss_weight=1.0, print_every=50):
    """L-BFGS optimization with frozen loss weights and training points."""
    model.train(True)
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=1,
        max_iter=iterations,
        max_eval=iterations,
        history_size=50,
        tolerance_grad=1e-7,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe"
    )

    total_losses = []
    physics_losses = []
    bc_losses = []
    count = 0

    def closure():
        nonlocal count
        optimizer.zero_grad()
        physics_loss, bc_loss = losses_func(model, *inputs)
        total_loss = physics_loss * physics_loss_weight + bc_loss * bc_loss_weight
        total_loss.backward()
        
        loss_val = total_loss.item()
        total_losses.append(loss_val)
        physics_losses.append(physics_loss.item())
        bc_losses.append(bc_loss.item())

        if count % print_every == 0:
            print(f"    LBFGS ITERATION: {count} | LOSS: {loss_val:.4e} | PDE: {physics_loss.item():.4e} | BC: {bc_loss.item():.4e}", flush=True)
        count += 1
        return total_loss

    optimizer.step(closure)
    return total_losses, physics_losses, bc_losses

def train_ic(model, inputs, material, sigma_g, iterations, lr=0.001, print_every=250):
    print(f"    Starting FAST IC Pretraining for {iterations} iterations...", flush=True)
    model.train(True)
    x, _ = inputs
    
    # Detach inputs to avoid "backward through graph a second time" error
    x_in = x.detach()
    t0 = torch.zeros_like(x_in)
    target = gaussian_ic(x_in, sigma_g).detach()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for i in range(iterations):
        optimizer.zero_grad()
        # Compute loss only at t=0
        u_pred = model(torch.cat([x_in, t0], dim=1))
        loss = torch.mean((u_pred - target)**2)
        loss.backward()
        optimizer.step()
        if i % print_every == 0:
            print(f"    IC PRETRAIN | ITER: {i} | LOSS: {loss.item():.4e}", flush=True)

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

def print_summary(summary):
    for (n_hidden, hidden_width), local_summary in summary.items():
        print(f"N� hidden layers: {n_hidden} | Hidden layer width: {hidden_width} | N� paramers: {local_summary['n_params']}")
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



ITERATIONS = 10
LR = 0.001
REPETITIONS = 1 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

NUM_POINTS_X = 100
NUM_POINTS_T = 100

x = torch.linspace(INTERVAL_X[0], INTERVAL_X[1], NUM_POINTS_X, requires_grad=True)
t = torch.linspace(INTERVAL_T[0], INTERVAL_T[1], NUM_POINTS_T, requires_grad=True)
x, t = torch.meshgrid(x, t, indexing='ij')

x = x.reshape(-1, 1).to(device)
t = t.reshape(-1, 1).to(device)

G = 5
k = 3

for hpt_config, (n_hidden, hidden_width) in enumerate([(2,7), (3,5)]):
    dir = f"./hpt_results/PIKAN/pikan_G{G}_k{k}_hidden{n_hidden}_width{hidden_width}/"
    os.makedirs(dir, exist_ok=True)

    for rep in range(REPETITIONS):
        print(f"Hyperparameter configuration: {hpt_config+1} | Repetition: {rep+1}", flush=True)

        model = build_kan(NUM_INPUTS, NUM_OUTPUTS, n_hidden, hidden_width, G, k)
        model.to(device)
        model.train(True)
        model.speed() 

        # 1. IC Pretraining (Fast version)
        train_ic(model, [x, t], my_material, 0.1, iterations=1000, lr=LR)

        # 2. Adam Training
        total_losses_adam, physics_losses_adam, bc_losses_adam, w_p, w_b = train_adam(model, [x, t], losses, ITERATIONS, LR, outdir=dir + f"model{rep+1}.pkl")
        
        print(f"    Switching to L-BFGS fine-tuning...", flush=True)
        total_losses_lbfgs, physics_losses_lbfgs, bc_losses_lbfgs = train_lbfgs(
            model, [x, t], losses, iterations=500, physics_loss_weight=w_p, bc_loss_weight=w_b
        )

        total_losses = total_losses_adam + total_losses_lbfgs
        physics_losses = physics_losses_adam + physics_losses_lbfgs
        bc_losses = bc_losses_adam + bc_losses_lbfgs
        
        # Final save after L-BFGS
        torch.save(model.state_dict(), dir + f"model{rep+1}.pkl")

        np.savez(
            file=dir + f"losses{rep+1}.npz",
            physics=np.array(physics_losses),
            bc=np.array(bc_losses),
            total=np.array(total_losses)
        )



results_dir = f"./hpt_results/"

n_hidden_pikan = [2, 3]
hidden_width_pikan = [7, 5]

NUM_X_EVAL = 500
NUM_T_EVAL = 500
x_eval = np.linspace(INTERVAL_X[0], INTERVAL_X[1], NUM_X_EVAL)
t_eval = np.linspace(INTERVAL_T[0], INTERVAL_T[1], NUM_T_EVAL)
x_eval, t_eval = np.meshgrid(x_eval, t_eval, indexing='ij')
x_eval = x_eval.reshape(-1, 1)
t_eval = t_eval.reshape(-1, 1)

u_true = u_sol(x_eval, t_eval)
u_grad_true = u_grad_sol(x_eval, t_eval)
xt_eval_torch = torch.cat(
    (
        torch.tensor(x_eval, dtype=torch.float32, requires_grad=True, device=device),
        torch.tensor(t_eval, dtype=torch.float32, requires_grad=True, device=device)
    ),
    1
)

# Compute errors
summarize_results('PIKAN', NUM_INPUTS, NUM_OUTPUTS, n_hidden_pikan, hidden_width_pikan, xt_eval_torch, u_true, u_grad_true, REPETITIONS, results_dir, G, k)

# Read and print summary
with open(results_dir + "PIKAN/summary.pkl", 'rb') as file:
    pikan_summary = pkl.load(file)
    print("PIKAN results:")
    print_summary(pikan_summary)



best_pikan_ixs = [pikan_summary[(n, w)]['lowest_relative_l2_ix'] for n, w in zip(n_hidden_pikan, hidden_width_pikan)]
plot_losses([], [], [], n_hidden_pikan, hidden_width_pikan, best_pikan_ixs, G, k, results_dir)



# ============================================================
# Animate: Best PIKAN model vs Finite Difference vs Exact
# ============================================================

# ----- Find best config (lowest mean L2 error) -----
best_config = min(
    [(n, w) for n, w in zip(n_hidden_pikan, hidden_width_pikan)],
    key=lambda nw: pikan_summary[nw]['avg_rel_l2_err']
)
best_n_hidden, best_width = best_config
best_rep_ix = pikan_summary[best_config]['lowest_relative_l2_ix']
print(f"Best config: L={best_n_hidden}, W={best_width}, rep={best_rep_ix+1}")
print(f"  Mean L2 error: {pikan_summary[best_config]['avg_rel_l2_err']:.4f}%")

# ----- Load best PIKAN model -----
best_model = build_kan(NUM_INPUTS, NUM_OUTPUTS, best_n_hidden, best_width, G, k)
best_model_path = (results_dir +
    f"PIKAN/pikan_G{G}_k{k}_hidden{best_n_hidden}_width{best_width}/model{best_rep_ix+1}.pkl")
best_model.load_state_dict(torch.load(best_model_path, weights_only=True))
best_model.to(device)
best_model.eval()

# ----- Spatial / temporal grids for animation -----
Nx = 200          # spatial points
Nt_anim = 100     # animation frames
x_anim = np.linspace(INTERVAL_X[0], INTERVAL_X[1], Nx)
t_frames = np.linspace(0.0, 1.0, Nt_anim)

# ----- Reference Solution (FD with ABCs) -----
# We compute it once here for efficiency
t_fd_grid, x_fd_grid, u_fd_full = solve_wave_fd(my_material, t_max=1.0)
fd_interp = RegularGridInterpolator((t_fd_grid, x_fd_grid), u_fd_full, bounds_error=False, fill_value=0)

# Extract frames
u_fd_frames = np.array([fd_interp(np.vstack(([tj]*Nx, x_anim)).T) for tj in t_frames])

# ----- PIKAN predictions at each frame -----
u_pikan_frames = np.zeros((Nt_anim, Nx))
with torch.no_grad():
    for j, tj in enumerate(t_frames):
        x_torch = torch.tensor(x_anim.reshape(-1, 1), dtype=torch.float32, device=device)
        t_torch = torch.full((Nx, 1), tj, dtype=torch.float32, device=device)
        xt = torch.cat([x_torch, t_torch], dim=1)
        # Apply ansatz for animation frames
        u_raw = best_model(xt)
        u_pikan_frames[j, :] = apply_ansatz(u_raw, x_torch, t_torch, my_material).cpu().numpy().flatten()

# ----- Exact solution (using our reference FD) -----
u_exact_frames = u_fd_frames

# ----- Build animation -----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Wave Equation - Best PIKAN (L={best_n_hidden}, W={best_width}) vs FD Reference", fontsize=13)

ax1, ax2 = axes

# Left: waveform comparison
line_ref1,   = ax1.plot(x_anim, u_exact_frames[0], 'k-',  lw=2,   label='FD Reference')
line_pikan1, = ax1.plot(x_anim, u_pikan_frames[0],'r:',  lw=2,   label='PIKAN')
ax1.set_xlim(INTERVAL_X[0], INTERVAL_X[1])
ax1.set_ylim(-1.2, 1.2)
ax1.set_xlabel('x')
ax1.set_ylabel('u(x, t)')
ax1.set_title('Solution')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, linestyle='dotted', alpha=0.6)
time_text1 = ax1.text(0.02, 1.08, '', transform=ax1.transAxes, fontsize=10)

# Right: pointwise absolute error vs exact
err_pikan = np.abs(u_exact_frames[0] - u_pikan_frames[0])
line_err_pikan, = ax2.plot(x_anim, err_pikan, 'r:',  lw=2,   label='|Ref - PIKAN|')
ax2.set_xlim(INTERVAL_X[0], INTERVAL_X[1])
max_err = np.max(np.abs(u_exact_frames - u_pikan_frames))
ax2.set_ylim(0, max_err * 1.15 if max_err > 0 else 0.1)
ax2.set_xlabel('x')
ax2.set_ylabel('|error|')
ax2.set_title('Absolute Error vs Reference')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, linestyle='dotted', alpha=0.6)
time_text2 = ax2.text(0.02, 1.08, '', transform=ax2.transAxes, fontsize=10)

fig.tight_layout()

def update(frame):
    tj = t_frames[frame]
    # waveform
    line_ref1.set_ydata(u_exact_frames[frame])
    line_pikan1.set_ydata(u_pikan_frames[frame])
    time_text1.set_text(f't = {tj:.3f}')
    # error
    err_pikan_f = np.abs(u_exact_frames[frame] - u_pikan_frames[frame])
    line_err_pikan.set_ydata(err_pikan_f)
    time_text2.set_text(f't = {tj:.3f}')
    return line_ref1, line_pikan1, line_err_pikan, time_text1, time_text2

anim = animation.FuncAnimation(fig, update, frames=Nt_anim, interval=60, blit=True)
plt.close()

# Display inline in Kaggle/Jupyter
# Display inline in Kaggle/Jupyter
HTML(anim.to_jshtml())


# ============================================================
# Static Snapshots: Best PIKAN model vs FD Reference
# ============================================================

# ----- Setup Snapshot Grid -----
n_snapshots = 6
t_snapshots = np.linspace(0.0, 1.0, n_snapshots)
Nx_snap = 200
x_grid_snap = np.linspace(INTERVAL_X[0], INTERVAL_X[1], Nx_snap)

# Create Plot
fig_snap, axes_snap = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
fig_snap.suptitle(f"Wave Equation: Best PIKAN (L={best_n_hidden}, W={best_width}) vs FD Reference", fontsize=16)
axes_snap = axes_snap.flatten()

with torch.no_grad():
    for i, tj in enumerate(t_snapshots):
        # 1. Get FD Reference
        u_fd_snap = fd_interp(np.vstack(([tj]*Nx_snap, x_grid_snap)).T)
        
        # 2. Get PIKAN Prediction (WITH ANSATZ)
        x_torch = torch.tensor(x_grid_snap.reshape(-1, 1), dtype=torch.float32, device=device)
        t_torch = torch.full((Nx_snap, 1), tj, dtype=torch.float32, device=device)
        xt = torch.cat([x_torch, t_torch], dim=1)
        # Apply ansatz for snapshot evaluations
        u_raw = best_model(xt)
        u_pikan_snap = apply_ansatz(u_raw, x_torch, t_torch, my_material).cpu().numpy().flatten()
        
        # 3. Plot waveform
        ax = axes_snap[i]
        ax.plot(x_grid_snap, u_fd_snap, 'k-', lw=2.5, label='FD Reference' if i == 0 else "")
        ax.plot(x_grid_snap, u_pikan_snap, 'r--', lw=2, label='PIKAN' if i == 0 else "")
        
        # 4. Plot absolute error on twin axis
        ax_err = ax.twinx()
        abs_err = np.abs(u_fd_snap - u_pikan_snap)
        ax_err.fill_between(x_grid_snap, abs_err, color='red', alpha=0.1, label='Abs Error' if i == 0 else "")
        ax_err.set_ylim(0, 0.5) # Adjust based on your expected error range
        if i % 3 != 2: ax_err.set_yticklabels([])
            
        # Metrics and styling
        rel_l2 = np.linalg.norm(u_fd_snap - u_pikan_snap) / (np.linalg.norm(u_fd_snap) + 1e-8)
        ax.set_title(f"Time t = {tj:.2f}\nRel. L2 Error: {rel_l2:.4%}", fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if i >= 3: ax.set_xlabel('x')
        if i % 3 == 0: ax.set_ylabel('u(x, t)')

# Global Legend
lines_1, labels_1 = axes_snap[0].get_legend_handles_labels()
lines_2, labels_2 = ax_err.get_legend_handles_labels()
fig_snap.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', bbox_to_anchor=(0.98, 0.95))

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.show()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
