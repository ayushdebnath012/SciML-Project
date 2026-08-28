import torch
import torch.nn as nn
from src.physics import MaterialModel, HomogeneousModel

# Instantiate default material model matching PIKAN.py behavior
my_material = HomogeneousModel()

# ─────────────────────────────────────────────
# 1. Initial Boundary Conditions & Ansatz Formulation
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# 2. PINN / PirateNet Loss Functions
# ─────────────────────────────────────────────

def compute_pde_residual(model: nn.Module, x: torch.Tensor, t: torch.Tensor,
                         material: MaterialModel, sigma_g: float = 0.1) -> torch.Tensor:
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = apply_ansatz(model(x, t), x, t, material, sigma_g)

    u_x, u_t = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u), create_graph=True)[:]
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]

    sigma = material.E(x) * u_x
    sigma_x = torch.autograd.grad(sigma, x, grad_outputs=torch.ones_like(sigma), create_graph=True)[0]

    return u_tt - (sigma_x / material.rho(x))


def causal_pde_loss(model: nn.Module, x: torch.Tensor, t: torch.Tensor,
                    material: MaterialModel, sigma_g: float, n_chunks: int, tolerance: float) -> torch.Tensor:
    R_all = compute_pde_residual(model, x, t, material, sigma_g)
    t_max = t.max().item()
    boundaries = torch.linspace(0.0, t_max, n_chunks + 1, device=t.device)

    chunk_losses = []
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
        if not loss.requires_grad or loss.item() == 0.0: 
            return 1.0
        grads = torch.autograd.grad(loss, list(model.parameters()), retain_graph=True, allow_unused=True)
        vals  = [g.abs().mean().item() for g in grads if g is not None]
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
    
    grads_bc = torch.autograd.grad(u_bc, [x_bc, t_bc], grad_outputs=torch.ones_like(u_bc), create_graph=True)
    u_x_bc, u_t_bc = grads_bc[0], grads_bc[1]
    
    Vp_bc = material.Vp(x_bc)
    x_mid = (material.x_min + material.x_max) / 2.0
    n_x   = torch.where(x_bc > x_mid, 1.0, -1.0) 
    
    abc_residual = u_t_bc + (Vp_bc * n_x * u_x_bc)
    loss_bc = (abc_residual ** 2).mean()

    total = w_pde * loss_pde + w_bc * loss_bc
    return total, loss_pde, loss_bc


# ─────────────────────────────────────────────
# 3. PIKAN / KAN Specific Loss Functions
# ─────────────────────────────────────────────

def dy_dx(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


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
