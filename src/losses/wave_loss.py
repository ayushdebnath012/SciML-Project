import math
import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# 2. Physics residuals and Causal losses
# ─────────────────────────────────────────────

def _call_model(model: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Call model robustly, handling KAN / WavKAN (single concatenated (N, 2) input)
    and standard PINN (separate x and t inputs) correctly.
    """
    from kan import KAN
    from src.models import WavKAN
    if isinstance(model, KAN) or isinstance(model, WavKAN):
        return model(torch.cat([x, t], dim=-1))
    return model(x, t)


def compute_pde_residual(model: nn.Module,
                         x: torch.Tensor,
                         t: torch.Tensor,
                         material: 'MaterialModel',
                         sigma_g: float = 0.1,
                         use_ansatz: bool = True) -> torch.Tensor:
    """
    Compute the 1D elastic wave PDE residual:
        R = ρ(x) ∂²û/∂t² - ∂/∂x [ E(x) ∂û/∂x ]

    Uses torch.autograd for exact derivatives (no truncation error).
    """
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)

    nn_out = _call_model(model, x, t)
    import problem_data as wave_problem
    u      = wave_problem.apply_ansatz(nn_out, x, t, sigma_g, use_ansatz=use_ansatz)

    # First derivatives
    grads_1 = torch.autograd.grad(u, [x, t],
                                  grad_outputs=torch.ones_like(u),
                                  create_graph=True)
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


# Module-level store updated by causal_pde_loss on every forward pass.
# Lets training loops read causal progress without changing return signatures.
_causal_state: dict = {
    "weights":      None,   # (n_chunks,) tensor, detached
    "chunk_losses": None,   # (n_chunks,) tensor, detached
    "n_chunks":     32,
}


def get_causal_frontier(threshold: float = 0.1) -> str:
    """
    Returns a one-line string summarising how far causality has propagated.

    Example outputs:
        "causal frontier: chunk 7/32  (w_min=0.0001  w_last=0.0000)  |████████░░░░...|"
        "causal frontier: FULL 32/32  (w_min=0.912   w_last=0.912 )  |████████████████|"

    A chunk is considered 'reached' when its causal weight exceeds `threshold`.
    Per the paper (Wang et al. 2022), the true convergence criterion is
    w_min → 1 (all weights converge to 1, meaning all prior residuals ≈ 0).
    """
    weights = _causal_state["weights"]
    if weights is None:
        return "causal frontier: (not yet computed)"

    w = weights.cpu().numpy()
    n = len(w)
    reached = int((w >= threshold).sum())     # how many chunks are above threshold
    w_last  = float(w[-1])
    w_min   = float(w.min())                  # paper convergence criterion: w_min → 1

    # ASCII bar: filled = above threshold, empty = below
    bar_len = min(n, 40)
    filled  = int(round(reached / n * bar_len))
    bar     = "█" * filled + "░" * (bar_len - filled)

    converged = w_min >= 0.9   # paper criterion: all weights ≈ 1
    status = "CONVERGED" if converged else ("FULL" if reached == n else f"chunk {reached}/{n}")
    return (f"causal frontier: {status} {reached}/{n}  "
            f"(w_min={w_min:.4f}  w_last={w_last:.4f})  |{bar}|")


def causal_pde_loss(model:     nn.Module,
                    x:         torch.Tensor,
                    t:         torch.Tensor,
                    material:  'MaterialModel',
                    sigma_g:   float = 0.1,
                    n_chunks:  int   = 32,
                    tolerance: float = 1.0,
                    use_ansatz: bool = True) -> torch.Tensor:
    """
    Causal PDE residual loss (paper ref [30], Wang et al. 2022).
    Splits collocation points into n_chunks time slabs and weights each
    by  w_m = exp(-tolerance * cumsum_of_previous_losses).

    Side-effect: updates _causal_state so get_causal_frontier() can be
    called from the training loop without any API change.
    """
    t_max = t.max().item()
    boundaries = torch.linspace(0.0, t_max, n_chunks + 1, device=t.device)

    chunk_losses = []
    for i in range(n_chunks):
        if i < n_chunks - 1:
            mask = (t.squeeze() >= boundaries[i]) & (t.squeeze() < boundaries[i + 1])
        else:
            mask = (t.squeeze() >= boundaries[i]) & (t.squeeze() <= boundaries[i + 1])
        if mask.sum() == 0:
            chunk_losses.append(torch.zeros(1, device=t.device))
            continue
        R = compute_pde_residual(model, x[mask], t[mask], material, sigma_g, use_ansatz=use_ansatz)
        chunk_losses.append((R ** 2).mean().unsqueeze(0))

    chunk_losses = torch.cat(chunk_losses)                      # (n_chunks,)

    # w_m = exp(-epsilon * sum_{k<m} L_k)  (Eq from ref [30])
    cumsum       = torch.zeros_like(chunk_losses)
    cumsum[1:]   = torch.cumsum(chunk_losses[:-1].detach(), dim=0)
    
    if not use_ansatz:
        # Treat the IC at t=0 as the "0-th chunk" in the causal chain.
        # If the model has not learned the IC, suppress all subsequent chunks (m >= 1).
        loss_ic_val = ic_loss(model, x, use_ansatz=False, sigma_g=sigma_g).detach()
        cumsum[1:] += 1000.0 * loss_ic_val

    weights      = torch.exp(-tolerance * cumsum)

    # ── Persist for diagnostic readout ────────────────────────────────────────
    _causal_state["weights"]      = weights.detach().cpu()
    _causal_state["chunk_losses"] = chunk_losses.detach().cpu()
    _causal_state["n_chunks"]     = n_chunks

    return (weights * chunk_losses).mean()


class GradNormWeighter:
    def __init__(self, alpha: float = 0.9, update_freq: int = 100):
        """
        Implements the Gradient-Based Loss Balancing scheme from 
        Wang et al. (2023), Algorithm 1 & Section 5.2.
        
        Args:
            alpha: Smoothing factor for EMA (paper defaults to 0.9).
            update_freq: How often to compute the expensive gradients (paper defaults to 1,000).
        """
        self.alpha = alpha
        self.update_freq = update_freq
        self.step_counter = 0
        self.ema_weights = None 

    def compute_weights(self, 
                        model: nn.Module,
                        loss_pde: torch.Tensor, 
                        loss_bc: torch.Tensor, 
                        loss_ic: torch.Tensor, 
                        use_ansatz: bool = False) -> tuple:
        
        self.step_counter += 1
        
        # 1. THE FREQUENCY TRICK: Skip the heavy math if it's not an update step
        # This saves massive amounts of compute time while maintaining stability.
        if self.ema_weights is not None and self.step_counter % self.update_freq != 0:
            return tuple(self.ema_weights.tolist())

        # 2. FULL NETWORK GRADIENT CALCULATION (Only runs every `update_freq` steps)
        def _get_l2_grad_norm(loss: torch.Tensor) -> float:
            if loss is None or not isinstance(loss, torch.Tensor) or loss.item() == 0.0:
                return 1e-12
            try:
                # Calculate gradients w.r.t ALL parameters as per the paper
                grads = torch.autograd.grad(
                    loss, list(model.parameters()),
                    retain_graph=True, allow_unused=True, create_graph=False
                )
                # The paper explicitly uses the L2 norm of the gradients (Eq 2.12 - 2.14)
                grad_norm = torch.norm(torch.stack([
                    torch.norm(g) for g in grads if g is not None
                ]))
                return grad_norm.item() + 1e-12
            except Exception:
                return 1e-12

        n_pde = _get_l2_grad_norm(loss_pde)
        n_bc  = _get_l2_grad_norm(loss_bc)

        # 3. CALCULATE RAW WEIGHTS (\hat{\lambda})
        if use_ansatz:
            # IC is satisfied by hard constraints. 
            # Numerator is the sum of remaining gradient norms.
            sum_norms = n_pde + n_bc
            hat_lambda = torch.tensor([sum_norms / n_pde, sum_norms / n_bc, 0.0])
            sum_hat = hat_lambda.sum()
            if sum_hat > 0:
                hat_lambda = 2.0 * (hat_lambda / sum_hat)
        else:
            # Unscale loss_ic by _IC_SCALE to get the raw gradient norm for balancing,
            # preventing the scaling factor from artificially deflating w_ic to 0.0.
            raw_loss_ic = loss_ic / _IC_SCALE if _IC_SCALE > 0 else loss_ic
            n_ic = _get_l2_grad_norm(raw_loss_ic)
            # Equations 2.12, 2.13, 2.14 exactly as written in the paper
            sum_norms = n_pde + n_bc + n_ic
            hat_lambda = torch.tensor([sum_norms / n_pde, sum_norms / n_bc, sum_norms / n_ic])
            sum_hat = hat_lambda.sum()
            if sum_hat > 0:
                hat_lambda = 3.0 * (hat_lambda / sum_hat)

        # 4. APPLY EXPONENTIAL MOVING AVERAGE (Eq 2.15)
        if self.ema_weights is None:
            self.ema_weights = hat_lambda
        else:
            self.ema_weights = self.alpha * self.ema_weights + (1 - self.alpha) * hat_lambda

        # Ensure the soft IC weight is never starved to 0.0 by GradNorm
        if not use_ansatz:
            self.ema_weights[2] = max(float(self.ema_weights[2]), 1.0)

        return tuple(self.ema_weights.tolist())


def physics_loss(model:        nn.Module,
                 x_int:        torch.Tensor,
                 t_int:        torch.Tensor,
                 x_bc:         torch.Tensor,
                 t_bc:         torch.Tensor,
                 x_ic:         torch.Tensor,
                 material:     'MaterialModel',
                 sigma_g:      float = 0.1,
                 n_chunks:     int   = 32,
                 tolerance:    float = 1.0,
                 use_causal:   bool  = True,
                 w_pde:        float = 1.0,
                 w_bc:         float = 1.0,
                 w_ic:         float = 1.0,
                 use_ansatz:   bool  = True
                 ) -> tuple:
    """
    Returns (total_loss, loss_pde, loss_bc, loss_ic).
    Weights w_* come from the grad-norm scheme (updated externally).
    """
    # Interior / PDE loss
    if use_causal:
        loss_pde = causal_pde_loss(model, x_int, t_int, material,
                                   sigma_g, n_chunks, tolerance, use_ansatz=use_ansatz)
    else:
        R        = compute_pde_residual(model, x_int, t_int, material, sigma_g, use_ansatz=use_ansatz)
        loss_pde = (R ** 2).mean()

    # Boundary loss: Absorbing Boundary Conditions (ABCs)
    import problem_data as wave_problem
    x_bc = x_bc.clone().detach().requires_grad_(True)
    t_bc = t_bc.clone().detach().requires_grad_(True)
    u_bc = wave_problem.apply_ansatz(_call_model(model, x_bc, t_bc), x_bc, t_bc, sigma_g, use_ansatz=use_ansatz)
    grads_bc = torch.autograd.grad(u_bc, [x_bc, t_bc], grad_outputs=torch.ones_like(u_bc), create_graph=True)
    u_bc_x, u_bc_t = grads_bc
    c_bc = material.Vp(x_bc)
    midpoint = (material.x_min + material.x_max) / 2.0
    is_left = x_bc < midpoint
    abc_residual = torch.where(is_left, u_bc_t - c_bc * u_bc_x, u_bc_t + c_bc * u_bc_x)
    loss_bc = (abc_residual ** 2).mean()

    # IC loss  (hard-constraint ansatz makes this ~0, kept for completeness)
    t_zero  = torch.zeros_like(x_ic).requires_grad_(True)
    u_ic    = wave_problem.apply_ansatz(_call_model(model, x_ic, t_zero), x_ic, t_zero, sigma_g, use_ansatz=use_ansatz)
    ic_ref  = wave_problem.gaussian_ic(x_ic, sigma_g)
    loss_ic_disp = ((u_ic - ic_ref) ** 2).mean()
    # Velocity IC: u_t(x, 0) = 0 (was missing — a key source of trivial solutions)
    u_t_ic = torch.autograd.grad(u_ic, t_zero, grad_outputs=torch.ones_like(u_ic), create_graph=True)[0]
    loss_ic_vel = (u_t_ic ** 2).mean()
    loss_ic = loss_ic_disp + loss_ic_vel

    total = w_pde * loss_pde + w_bc * loss_bc + w_ic * loss_ic
    return total, loss_pde, loss_bc, loss_ic


def sample_batch(n_int:  int,
                 n_bc:   int,
                 n_ic:   int,
                 x_min:  float,
                 x_max:  float,
                 t_max:  float,
                 device: torch.device) -> tuple:
    """Fresh random collocation points every step (paper Sec 5)."""
    # Interior points
    x_int = torch.rand(n_int, 1, device=device) * (x_max - x_min) + x_min
    t_int = torch.rand(n_int, 1, device=device) * t_max

    # Boundary points  (x=x_min and x=x_max, random t)
    t_bc  = torch.rand(n_bc, 1, device=device) * t_max
    x_bc  = torch.where(torch.rand(n_bc, 1, device=device) < 0.5,
                        torch.full((n_bc, 1), x_min, device=device),
                        torch.full((n_bc, 1), x_max, device=device))

    # Initial condition points  (random x, t=0 handled in loss)
    x_ic  = torch.rand(n_ic, 1, device=device) * (x_max - x_min) + x_min

    return x_int, t_int, x_bc, t_bc, x_ic


# ─────────────────────────────────────────────
# 3. Backwards-compatible standard losses function
# ─────────────────────────────────────────────

# Relative weight of displacement vs. velocity IC terms.
# Displacement is the primary constraint; velocity is secondary.
_IC_DISP_WEIGHT = 5.0
_IC_VEL_WEIGHT  = 1.0


def ic_loss(model: nn.Module, x: torch.Tensor, use_ansatz: bool = False, sigma_g: float = 0.1) -> torch.Tensor:
    """
    Dedicated Initial Condition (IC) loss computation:
        Displacement: u(x, 0) - g(x)   (weighted ×_IC_DISP_WEIGHT)
        Velocity:     u_t(x, 0) - 0    (weighted ×_IC_VEL_WEIGHT)
    """
    import problem_data as wave_problem

    t0 = torch.zeros_like(x).requires_grad_(True)
    nn_out = _call_model(model, x, t0)
    u_ic = wave_problem.apply_ansatz(nn_out, x, t0, sigma_g, use_ansatz=use_ansatz)
    u_t_ic = torch.autograd.grad(u_ic, t0, grad_outputs=torch.ones_like(u_ic), create_graph=True)[0]

    ic_ref = wave_problem.gaussian_ic(x, sigma_g)
    loss_disp = ((u_ic - ic_ref) ** 2).mean()
    loss_vel  = (u_t_ic ** 2).mean()
    return _IC_DISP_WEIGHT * loss_disp + _IC_VEL_WEIGHT * loss_vel


def losses(model, x, t, use_ansatz=False, material=None, x_ic=None):
    """
    Standard waves PDE loss.
    x_ic: optional dense spatial grid for IC evaluation (overrides x).
          Pass a grid with extra points near the Gaussian peak for better IC fitting.
    """
    # Create a basic Homogeneous material for default evaluation if None
    if material is None:
        from wave.materials import HomogeneousModel
        material = HomogeneousModel()
    
    # We evaluate standard PDE loss
    R = compute_pde_residual(model, x, t, material, sigma_g=0.1, use_ansatz=use_ansatz)
    pde_loss = (R**2).mean()
    
    # BC loss: Absorbing Boundary Conditions (ABCs)
    # Use unique time values to avoid repeating BC evaluations from the meshgrid.
    import problem_data as wave_problem
    t_bc = t.detach().unique().reshape(-1, 1).requires_grad_(True)
    n_bc = t_bc.shape[0]

    # Left boundary (x = x_min): u_t - c * u_x = 0
    x0t = torch.full((n_bc, 1), material.x_min, device=x.device, dtype=x.dtype).requires_grad_(True)
    u_bc1 = wave_problem.apply_ansatz(_call_model(model, x0t, t_bc), x0t, t_bc, sigma_g=0.1, use_ansatz=use_ansatz)
    grads_bc1 = torch.autograd.grad(u_bc1, [x0t, t_bc], grad_outputs=torch.ones_like(u_bc1), create_graph=True)
    u_bc1_x, u_bc1_t = grads_bc1
    c_bc1 = material.Vp(x0t)
    abc_res1 = u_bc1_t - c_bc1 * u_bc1_x

    # Right boundary (x = x_max): u_t + c * u_x = 0
    x1t = torch.full((n_bc, 1), material.x_max, device=x.device, dtype=x.dtype).requires_grad_(True)
    u_bc2 = wave_problem.apply_ansatz(_call_model(model, x1t, t_bc), x1t, t_bc, sigma_g=0.1, use_ansatz=use_ansatz)
    grads_bc2 = torch.autograd.grad(u_bc2, [x1t, t_bc], grad_outputs=torch.ones_like(u_bc2), create_graph=True)
    u_bc2_x, u_bc2_t = grads_bc2
    c_bc2 = material.Vp(x1t)
    abc_res2 = u_bc2_t + c_bc2 * u_bc2_x

    bc_loss = (abc_res1**2).mean() + (abc_res2**2).mean()
    
    if use_ansatz:
        ic_loss_val = torch.tensor(0.0, device=x.device, requires_grad=True)
    else:
        x_for_ic = x_ic if x_ic is not None else x
        ic_loss_val = ic_loss(model, x_for_ic, use_ansatz=False, sigma_g=0.1)

    conds_loss = bc_loss + ic_loss_val
    return pde_loss, conds_loss


# Scale factor applied to IC loss before returning from losses_gradnorm.
# GradNorm re-balances anyway, but the IC gradient signal is so small early
# in training that the weighter deprioritises it and the trivial solution
# (u≡0) becomes entrenched.  A fixed pre-scale of 15000.0 ensures GradNorm
# always *sees* a meaningful IC gradient to work with.
_IC_SCALE = 15000.0


def losses_gradnorm(model, x, t, use_ansatz=False, material=None, x_ic=None,
                    causal_tolerance=1.0, n_chunks=16):
    """
    Like `losses()` but returns (loss_pde, loss_bc, loss_ic) as THREE separate
    tensors for grad-norm adaptive weighting.

    causal_tolerance: ε for the causal weighting.  train_adam injects a linearly
        scheduled value each step when causal_tolerance=(start, end) is passed.
    n_chunks: number of causal time slabs (default 16; fewer = faster propagation).
    x_ic: optional dense spatial grid for IC evaluation (see losses()).
    """
    if material is None:
        from wave.materials import HomogeneousModel
        material = HomogeneousModel()

    import problem_data as wave_problem

    loss_pde = causal_pde_loss(
        model, x, t, material, sigma_g=0.1,
        n_chunks=n_chunks, tolerance=causal_tolerance, use_ansatz=use_ansatz
    )

    # ── BC loss: Absorbing Boundary Conditions ────────────────────────────────
    # Use unique time values to avoid repeating BC evaluations from the meshgrid.
    t_bc = t.detach().unique().reshape(-1, 1).requires_grad_(True)
    n_bc = t_bc.shape[0]

    x0t = torch.full((n_bc, 1), material.x_min, device=x.device, dtype=x.dtype).requires_grad_(True)
    u_bc1 = wave_problem.apply_ansatz(_call_model(model, x0t, t_bc), x0t, t_bc, sigma_g=0.1, use_ansatz=use_ansatz)
    grads_bc1 = torch.autograd.grad(u_bc1, [x0t, t_bc], grad_outputs=torch.ones_like(u_bc1), create_graph=True)
    abc_res1 = grads_bc1[1] - material.Vp(x0t) * grads_bc1[0]

    x1t = torch.full((n_bc, 1), material.x_max, device=x.device, dtype=x.dtype).requires_grad_(True)
    u_bc2 = wave_problem.apply_ansatz(_call_model(model, x1t, t_bc), x1t, t_bc, sigma_g=0.1, use_ansatz=use_ansatz)
    grads_bc2 = torch.autograd.grad(u_bc2, [x1t, t_bc], grad_outputs=torch.ones_like(u_bc2), create_graph=True)
    abc_res2 = grads_bc2[1] + material.Vp(x1t) * grads_bc2[0]

    loss_bc = (abc_res1 ** 2).mean() + (abc_res2 ** 2).mean()

    # ── IC loss ───────────────────────────────────────────────────────────────
    if use_ansatz:
        loss_ic = torch.zeros(1, device=x.device)
    else:
        x_for_ic = x_ic if x_ic is not None else x
        loss_ic = _IC_SCALE * ic_loss(model, x_for_ic, use_ansatz=False, sigma_g=0.1)

    return loss_pde, loss_bc, loss_ic
