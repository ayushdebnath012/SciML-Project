import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython import display

from src.physics import MaterialModel
from src.ansatz_losses import (
    gaussian_ic, apply_ansatz, physics_loss, 
    compute_grad_norm_weights
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# 1. Batching and Schedulers
# ─────────────────────────────────────────────

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
        if step < warmup_steps: 
            return float(step + 1) / float(warmup_steps)
        return decay_rate ** ((step - warmup_steps) // decay_steps)
    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
# 2. Interactive Live Plotter
# ─────────────────────────────────────────────

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

    def update(self, loss_val, pde_val=None, bc_val=None, w_pde=None, w_bc=None, l2_val=None, step=None):
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


# ─────────────────────────────────────────────
# 3. PINN Training Loop (Two Phase Adam -> L-BFGS)
# ─────────────────────────────────────────────

def train_model_two_phase(model, material, x_fd, t_fd, u_fd, n_steps=300000, lbfgs_steps=500, 
                          lr=1e-3, warmup_steps=5000, decay_rate=0.9, decay_steps=5000, 
                          n_int=8192, n_bc=256, T_stages=None, sigma_g=0.1, 
                          n_chunks=32, tolerance=0.1, use_causal=True, 
                          weight_update_freq=1000, arch_name="Model",
                          save_path="best_model.pt"):
    
    from src.evaluation import evaluate # Local import to prevent circularity
    
    if T_stages is None: 
        T_stages = [1.0]
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
        if steps_per_stage <= 0: 
            continue
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
                
                plotter.update(loss.item(), l_pde.item(), l_bc.item(), w_pde, w_bc, l2_val=current_l2, step=global_step)
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
                               w_pde, w_bc, l2_val=current_l2, step=global_step + lbfgs_ep)
                pbar_lbfgs.set_postfix({"Loss": f"{closure_logs['tot']:.2e}", "L2": f"{current_l2:.2f}%", "Best": f"{best_l2:.2f}%"})
                
            loss_history.append(closure_logs['tot'])

    print(f"\n>>> Training Complete. Best Mean L2 Error: {best_l2:.4f}%")
    # Load the best weights back into the model before returning
    model.load_state_dict(torch.load(save_path))
    
    plt.close(plotter.fig)
    return loss_history


# ─────────────────────────────────────────────
# 4. KAN / PIKAN Training Loops
# ─────────────────────────────────────────────

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
        line_search_fn="strong_wolf"
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
