import os
import sys
import json
import random
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure package imports from parent and wave folders resolve correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models import build_mlp, build_kan, FourierFeaturePINN, PirateNet, WavKAN
from src.train import train_adam, train_lbfgs, train_ic_warmup
from src.losses.wave_loss import losses, losses_gradnorm
import problem_data as pbd
from wave.materials import HomogeneousModel, TwoLayerModel, MultiLayerModel

# Set matplotlib plotting configurations for high-quality visuals
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})


def set_seed(seed: int):
    """Seed everything for full reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # multi-GPU safety
    torch.backends.cudnn.deterministic = True  # deterministic convolutions
    torch.backends.cudnn.benchmark     = False  # disable auto-tuner (not reproducible)

# ==============================================================================
# MULTIPLE ARCHITECTURE SWEEP CONFIGURATIONS (Change Model parameters here):
# ==============================================================================
MODEL_CONFIGS = [
    # 1. Vanilla PINN sweeps (Deeper layers)
    {"model_type": "PINN", "n_hidden": 2, "hidden_width": 32, "extra_params": {}},
    {"model_type": "PINN", "n_hidden": 3, "hidden_width": 64, "extra_params": {}},
    {"model_type": "PINN", "n_hidden": 4, "hidden_width": 128, "extra_params": {}},
    {"model_type": "PINN", "n_hidden": 5, "hidden_width": 128, "extra_params": {}},
    
    # 2. Fourier Feature PINN sweeps (Deeper layers + Sigmas 3 and 7)
    {"model_type": "FourierFeaturePINN", "n_hidden": 2, "hidden_width": 32, "extra_params": {"n_fourier": 64, "sigma": 3.0}},
    {"model_type": "FourierFeaturePINN", "n_hidden": 2, "hidden_width": 32, "extra_params": {"n_fourier": 64, "sigma": 7.0}},
    {"model_type": "FourierFeaturePINN", "n_hidden": 3, "hidden_width": 64, "extra_params": {"n_fourier": 128, "sigma": 3.0}},
    {"model_type": "FourierFeaturePINN", "n_hidden": 3, "hidden_width": 64, "extra_params": {"n_fourier": 128, "sigma": 7.0}},
    {"model_type": "FourierFeaturePINN", "n_hidden": 4, "hidden_width": 128, "extra_params": {"n_fourier": 128, "sigma": 3.0}},
    {"model_type": "FourierFeaturePINN", "n_hidden": 4, "hidden_width": 128, "extra_params": {"n_fourier": 128, "sigma": 7.0}},
    {"model_type": "FourierFeaturePINN", "n_hidden": 5, "hidden_width": 128, "extra_params": {"n_fourier": 256, "sigma": 3.0}},
    {"model_type": "FourierFeaturePINN", "n_hidden": 5, "hidden_width": 128, "extra_params": {"n_fourier": 256, "sigma": 7.0}},
    
    # 3. PirateNet sweeps (Sigmas 3 and 7)
    {"model_type": "PirateNet", "n_hidden": 2, "hidden_width": 32, "extra_params": {"n_fourier": 64, "sigma": 3.0}},
    {"model_type": "PirateNet", "n_hidden": 2, "hidden_width": 32, "extra_params": {"n_fourier": 64, "sigma": 7.0}},
    {"model_type": "PirateNet", "n_hidden": 3, "hidden_width": 64, "extra_params": {"n_fourier": 128, "sigma": 3.0}},
    {"model_type": "PirateNet", "n_hidden": 3, "hidden_width": 64, "extra_params": {"n_fourier": 128, "sigma": 7.0}},
    {"model_type": "PirateNet", "n_hidden": 4, "hidden_width": 128, "extra_params": {"n_fourier": 128, "sigma": 3.0}},
    {"model_type": "PirateNet", "n_hidden": 4, "hidden_width": 128, "extra_params": {"n_fourier": 128, "sigma": 7.0}},
    
    # 4. KAN sweeps (Deeper layers)
    {"model_type": "KAN", "n_hidden": 1, "hidden_width": 16, "extra_params": {"G": 3, "k": 2}},
    {"model_type": "KAN", "n_hidden": 2, "hidden_width": 16, "extra_params": {"G": 5, "k": 3}},
    {"model_type": "KAN", "n_hidden": 3, "hidden_width": 16, "extra_params": {"G": 5, "k": 3}},
    {"model_type": "KAN", "n_hidden": 4, "hidden_width": 16, "extra_params": {"G": 5, "k": 3}},
    
    # 5. WavKAN sweeps (Deeper layers + Wavelets: Morlet and Mexican Hat)
    {"model_type": "WavKAN", "n_hidden": 1, "hidden_width": 16, "extra_params": {"wavelet_type": "morlet"}},
    {"model_type": "WavKAN", "n_hidden": 1, "hidden_width": 16, "extra_params": {"wavelet_type": "mexican_hat"}},
    {"model_type": "WavKAN", "n_hidden": 2, "hidden_width": 16, "extra_params": {"wavelet_type": "morlet"}},
    {"model_type": "WavKAN", "n_hidden": 2, "hidden_width": 16, "extra_params": {"wavelet_type": "mexican_hat"}},
    {"model_type": "WavKAN", "n_hidden": 3, "hidden_width": 16, "extra_params": {"wavelet_type": "morlet"}},
    {"model_type": "WavKAN", "n_hidden": 3, "hidden_width": 16, "extra_params": {"wavelet_type": "mexican_hat"}},
    {"model_type": "WavKAN", "n_hidden": 4, "hidden_width": 16, "extra_params": {"wavelet_type": "morlet"}},
    {"model_type": "WavKAN", "n_hidden": 4, "hidden_width": 16, "extra_params": {"wavelet_type": "mexican_hat"}}
]

# General experiment controls
CONFIG = {
    # ── Domain intervals ──────────────────────────────────────────────────────
    # X is derived from material.x_min / material.x_max (set in materials.py)
    # Change T interval here:
    "t_min":  0.0,                   # Start of time domain
    "t_max":  1.0,                   # End of time domain

    # ── Collocation grid size ──────────────────────────────────────────────────
    "Nx_collocation": 100,           # Number of spatial collocation points
    "Nt_collocation": 100,           # Number of temporal collocation points

    # ── Optimizer settings ────────────────────────────────────────────────────
    "adam_iterations": 15000,        # Number of Adam optimization steps
    "lbfgs_iterations": 2000,        # Maximum number of L-BFGS steps (Adam+LBFGS runs)
    "lbfgs_only_iterations": 5000,   # Number of L-BFGS steps for LBFGS-only runs
    "lr": 0.001,                     # Learning rate for Adam phase
    "gradnorm_update_freq": 100,     # How often to recompute grad-norm weights (every N Adam steps)
    "ic_warmup_iterations": 2000,    # IC-only Adam steps before joint training (use_ansatz=False only)
    "ic_gradnorm_delay": 3000,       # Adam steps with fixed IC-heavy weights before GradNorm activates

    # ── Reproducibility ───────────────────────────────────────────────────────
    "seed": 42,                      # Global random seed for full reproducibility

    # ── Output ────────────────────────────────────────────────────────────────
    "output_base_dir": "./experiment_results/"  # Base directory (material subdirs added automatically)
}

def instantiate_model(cfg, device):
    mtype = cfg["model_type"]
    nh = cfg["n_hidden"]
    hw = cfg["hidden_width"]
    ep = cfg["extra_params"]
    
    if mtype == "PINN":
        model = build_mlp(pbd.NUM_INPUTS, pbd.NUM_OUTPUTS, n_hidden=nh, hidden_width=hw)
    elif mtype == "FourierFeaturePINN":
        model = FourierFeaturePINN(hidden_layers=nh, hidden_units=hw, n_fourier=ep["n_fourier"], sigma=ep["sigma"])
    elif mtype == "PirateNet":
        model = PirateNet(n_blocks=nh, units=hw, n_fourier=ep["n_fourier"], sigma=ep["sigma"])
    elif mtype == "KAN":
        model = build_kan(pbd.NUM_INPUTS, pbd.NUM_OUTPUTS, n_hidden=nh, hidden_width=hw, G=ep["G"], k=ep["k"])
        model.speed()  # Deactivate symbolic branch for efficiency
    elif mtype == "WavKAN":
        width = [pbd.NUM_INPUTS] + [hw] * nh + [pbd.NUM_OUTPUTS]
        model = WavKAN(width, wavelet_type=ep["wavelet_type"])
    else:
        raise ValueError(f"Unknown model name: {mtype}")
    
    return model.to(device)

def evaluate_ic_metrics(model, use_ansatz, material, device):
    """
    Evaluates how well the trained model satisfies the initial conditions at t=0.
    Returns:
        disp_rel_l2 (%): Relative L2 error of displacement u(x, 0) against the exact Gaussian IC.
        vel_mse: Mean squared error of initial velocity u_t(x, 0) against 0.
    """
    import problem_data as wave_problem
    
    # Grid of x-points for evaluation
    x_eval = torch.linspace(material.x_min, material.x_max, 400, device=device).reshape(-1, 1).requires_grad_(True)
    t_eval = torch.zeros_like(x_eval).requires_grad_(True)
    
    # We must enable gradients to perform autograd.grad for velocity u_t
    with torch.set_grad_enabled(True):
        from src.models import WavKAN
        from kan import KAN
        if isinstance(model, KAN) or isinstance(model, WavKAN):
            nn_out = model(torch.cat([x_eval, t_eval], dim=-1))
        else:
            nn_out = model(x_eval, t_eval)
            
        u_ic = wave_problem.apply_ansatz(nn_out, x_eval, t_eval, sigma_g=0.1, use_ansatz=use_ansatz)
        ic_ref = wave_problem.gaussian_ic(x_eval, sigma_g=0.1)
        
        # Velocity u_t(x, 0)
        u_t_ic = torch.autograd.grad(u_ic, t_eval, grad_outputs=torch.ones_like(u_ic), create_graph=False)[0]
        
    u_ic_np = u_ic.detach().cpu().numpy()
    ic_ref_np = ic_ref.detach().cpu().numpy()
    u_t_ic_np = u_t_ic.detach().cpu().numpy()
    
    # Relative L2 displacement error (%)
    ref_norm = np.linalg.norm(ic_ref_np)
    if ref_norm > 1e-12:
        disp_rel_l2 = float(100 * np.linalg.norm(u_ic_np - ic_ref_np) / ref_norm)
    else:
        disp_rel_l2 = float(100 * np.linalg.norm(u_ic_np - ic_ref_np))
        
    # Velocity MSE
    vel_mse = float(np.mean(u_t_ic_np ** 2))
    
    return disp_rel_l2, vel_mse

def run_single_experiment(cfg, use_ansatz, material, device, idx, total_runs,
                          lbfgs_only=False):
    """
    Run one experiment.

    Args:
        lbfgs_only: If True, skip Adam entirely and run L-BFGS from a fresh
                    random model for CONFIG["lbfgs_only_iterations"] steps.
                    Results go to a separate output directory with _lbfgs_only suffix.
    """
    # Re-seed at the start of every experiment so each run is independently
    # reproducible regardless of what ran before it.
    set_seed(CONFIG["seed"])

    mtype = cfg["model_type"]
    nh = cfg["n_hidden"]
    hw = cfg["hidden_width"]
    ep = cfg["extra_params"]

    # Resolve unique name based on parameters
    suffix = f"_lbfgs_only_ansatz_{'true' if use_ansatz else 'false'}" if lbfgs_only else f"_ansatz_{'true' if use_ansatz else 'false'}"
    if "sigma" in ep:
        run_name = f"{mtype}_h{nh}_w{hw}_sigma{int(ep['sigma'])}{suffix}"
    elif "wavelet_type" in ep:
        run_name = f"{mtype}_h{nh}_w{hw}_{ep['wavelet_type']}{suffix}"
    else:
        run_name = f"{mtype}_h{nh}_w{hw}{suffix}"

    output_dir = os.path.join(CONFIG["output_base_dir"], material.name, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f" [{idx}/{total_runs}] RUNNING: {mtype} (Layers={nh}, Width={hw})")
    if "sigma" in ep:
        print(f"            Sigma: {ep['sigma']}")
    if "wavelet_type" in ep:
        print(f"            Wavelet: {ep['wavelet_type']}")
    if lbfgs_only:
        print(f"            Mode: L-BFGS Only ({CONFIG['lbfgs_only_iterations']} iters) | IC: {'hardcoded' if use_ansatz else 'loss'} | Material: {material.name}")
    else:
        print(f"            IC Setup: {'IC hardcoded' if use_ansatz else 'IC loss'} | Material: {material.name}")
    print("=" * 80)

    # 1. Instantiate Model
    model = instantiate_model(cfg, device)

    # 2. Setup Collocation Training Grid
    x_coll = torch.linspace(material.x_min, material.x_max, CONFIG["Nx_collocation"], requires_grad=True).reshape(-1, 1).to(device)
    t_coll = torch.linspace(CONFIG["t_min"], CONFIG["t_max"], CONFIG["Nt_collocation"], requires_grad=True).reshape(-1, 1).to(device)
    x_mesh, t_mesh = torch.meshgrid(x_coll.flatten(), t_coll.flatten(), indexing='ij')
    x_train = x_mesh.reshape(-1, 1).to(device)
    t_train = t_mesh.reshape(-1, 1).to(device)

    # 2b. Dense IC grid: uniform coverage + extra points near the Gaussian peak
    #     (sigma=0.1 centered at x=0 → most content in [-0.3, 0.3])
    sigma_g = 0.1
    x_ic_uniform = torch.linspace(material.x_min, material.x_max, 200, device=device).reshape(-1, 1)
    peak_lo = max(material.x_min, -3 * sigma_g)
    peak_hi = min(material.x_max,  3 * sigma_g)
    x_ic_peak = torch.linspace(peak_lo, peak_hi, 300, device=device).reshape(-1, 1)
    x_ic_dense = torch.cat([x_ic_uniform, x_ic_peak], dim=0)

    # 3. IC Warm-up + PI-init (soft IC mode only)
    if not use_ansatz and not lbfgs_only:
        print(f"\n--- [0/2] IC Warm-up: {CONFIG['ic_warmup_iterations']} iterations ---")
        train_ic_warmup(
            model, x_ic_dense, sigma_g=sigma_g,
            iterations=CONFIG["ic_warmup_iterations"]
        )
        # Physics-informed output-layer init for architectures that support it
        if mtype in ("FourierFeaturePINN", "PirateNet"):
            print("    Applying physics-informed output-layer init...")
            xt_all = torch.cat([x_train.detach(), t_train.detach()], dim=-1)
            ic_ref_all = pbd.gaussian_ic(x_train.detach(), sigma_g=sigma_g)
            model.physics_informed_init(xt_all, ic_ref_all)
            print("    PI-init complete.", flush=True)

    # ── BRANCH: L-BFGS Only ──────────────────────────────────────────────────
    if lbfgs_only:
        lbfgs_outdir = os.path.join(output_dir, "model_best_lbfgs_only.pkl")
        n_iters = CONFIG["lbfgs_only_iterations"]
        print(f"\n--- [1/1] L-BFGS Only: {n_iters} Max Iterations ---")
        lbfgs_total, lbfgs_phys, lbfgs_cond = train_lbfgs(
            model=model,
            inputs=[x_train, t_train],
            losses_func=losses,
            max_iter=n_iters,
            lr=1.0,
            print_every=500,
            outdir=lbfgs_outdir,
            use_gradnorm=False,   # no Adam weights to inherit — use equal weights
            use_ansatz=use_ansatz,
            material=material
        )
        model.load_state_dict(torch.load(lbfgs_outdir, map_location=device, weights_only=True))

        # Save loss curve
        print("\nSaving loss curve...")
        steps = range(1, len(lbfgs_total) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(steps, lbfgs_total, label="L-BFGS Only", color="#9467bd", linewidth=2, alpha=0.85)
        plt.yscale("log")
        plt.xlabel("Iteration Step")
        plt.ylabel("Total Loss")
        plt.title(f"Loss Curve (L-BFGS Only) - {mtype} ({material.name})")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend(frameon=True, facecolor="white", edgecolor="gainsboro")
        plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Evaluate
        sigma_g = 0.1
        f_fn = lambda x_val: pbd.gaussian_ic(torch.tensor(x_val, dtype=torch.float32).reshape(-1, 1), sigma_g=sigma_g).flatten().cpu().numpy()
        g_fn = lambda x_val: 0.0
        c_fn = lambda x_val: material.Vp(torch.tensor(x_val, dtype=torch.float32).reshape(-1, 1)).item()
        Nx_fd = 200
        x_fd, t_fd, u_fd = pbd.solve_reference_fd(f_fn, g_fn, c_fn, Nx=Nx_fd, CFL=0.9)
        x_mesh_fd, t_mesh_fd = np.meshgrid(x_fd, t_fd, indexing='ij')
        x_eval_fd = torch.tensor(x_mesh_fd, dtype=torch.float32).reshape(-1, 1).to(device)
        t_eval_fd = torch.tensor(t_mesh_fd, dtype=torch.float32).reshape(-1, 1).to(device)

        def _call_model_local(net, x_in, t_in):
            from src.models import WavKAN
            from kan import KAN
            if isinstance(net, KAN) or isinstance(net, WavKAN):
                return net(torch.cat([x_in, t_in], dim=-1))
            return net(x_in, t_in)

        print("Evaluating best L-BFGS-only model...")
        model_eval = instantiate_model(cfg, device)
        model_eval.load_state_dict(torch.load(lbfgs_outdir, map_location=device, weights_only=True))
        model_eval.eval()
        with torch.no_grad():
            nn_out = _call_model_local(model_eval, x_eval_fd, t_eval_fd)
            u_pred = pbd.apply_ansatz(nn_out, x_eval_fd, t_eval_fd, sigma_g=sigma_g, use_ansatz=use_ansatz)
            u_pred_np = u_pred.cpu().numpy().reshape(Nx_fd + 1, -1)
        residual = u_fd - u_pred_np
        rel_l2 = float(100 * np.linalg.norm(residual) / np.linalg.norm(u_fd))
        print(f"  -> L-BFGS-Only Relative L2 Error: {rel_l2:.6f}%")

        # Evaluate IC metrics
        disp_rel_l2, vel_mse = evaluate_ic_metrics(model_eval, use_ansatz, material, device)
        print(f"  -> IC Displacement L2 Error: {disp_rel_l2:.6f}%")
        print(f"  -> IC Velocity MSE: {vel_mse:.6e}")

        metrics = {
            "model_type": mtype, "n_hidden": nh, "hidden_width": hw,
            "material_type": material.name, "use_ansatz": use_ansatz,
            "optimizer": "lbfgs_only",
            "lbfgs_only_iterations": n_iters,
            "relative_l2_error_lbfgs_only_percent": rel_l2,
            "ic_displacement_l2_error_percent": disp_rel_l2,
            "ic_velocity_mse": vel_mse
        }
        with open(os.path.join(output_dir, "l2_errors.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        # Solution comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].set_title("FD Reference Solution")
        im0 = axes[0].imshow(np.flip(np.transpose(u_fd), 0), extent=[x_fd.min(), x_fd.max(), t_fd.min(), t_fd.max()], cmap='plasma', aspect='auto')
        fig.colorbar(im0, ax=axes[0])
        axes[1].set_title(f"{mtype} Prediction (L-BFGS Only)")
        im1 = axes[1].imshow(np.flip(np.transpose(u_pred_np), 0), extent=[x_fd.min(), x_fd.max(), t_fd.min(), t_fd.max()], cmap='plasma', aspect='auto')
        fig.colorbar(im1, ax=axes[1])
        axes[2].set_title("Absolute Residuals")
        im2 = axes[2].imshow(np.flip(np.transpose(np.abs(residual)), 0), extent=[x_fd.min(), x_fd.max(), t_fd.min(), t_fd.max()], cmap='coolwarm', aspect='auto')
        fig.colorbar(im2, ax=axes[2])
        for ax in axes:
            ax.set_xlabel("Space x"); ax.set_ylabel("Time t")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "solution_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Finished Run: {run_name} | IC: {'hardcoded' if use_ansatz else 'loss'} | L-BFGS-Only L2 Error: {rel_l2:.4f}% | IC Disp L2 Error: {disp_rel_l2:.4f}% | IC Vel MSE: {vel_mse:.4e}")
        return

    # ── BRANCH: Adam + L-BFGS (original) ────────────────────────────────────
    # 3. Phase 1: Adam Training
    print(f"\n--- [1/2] Adam Optimizer: {CONFIG['adam_iterations']} Iterations ---")
    adam_outdir = os.path.join(output_dir, "model_best_adam.pkl")
    adam_total, adam_phys, adam_cond, final_w_pde, final_w_bc, final_w_ic, \
        w_pde_hist, w_bc_hist, w_ic_hist, w_min_hist, w_chunks_snaps = train_adam(
        model=model,
        inputs=[x_train, t_train],
        losses_func=losses_gradnorm,
        iterations=CONFIG["adam_iterations"],
        lr=CONFIG["lr"],
        print_every=2000,
        outdir=adam_outdir,
        use_gradnorm=True,
        gradnorm_update_freq=CONFIG.get("gradnorm_update_freq", 100),
        ic_gradnorm_delay=(CONFIG["ic_gradnorm_delay"] if not use_ansatz else 0),
        use_ansatz=use_ansatz,
        material=material,
        x_ic=x_ic_dense if not use_ansatz else None,
    )

    # Load best checkpoint from Adam phase
    model.load_state_dict(torch.load(adam_outdir, map_location=device, weights_only=True))

    # 4. Phase 2: L-BFGS Training
    print(f"\n--- [2/2] L-BFGS Optimizer: {CONFIG['lbfgs_iterations']} Max Iterations ---")
    lbfgs_outdir = os.path.join(output_dir, "model_best_lbfgs.pkl")
    lbfgs_total, lbfgs_phys, lbfgs_cond = train_lbfgs(
        model=model,
        inputs=[x_train, t_train],
        losses_func=losses_gradnorm,
        max_iter=CONFIG["lbfgs_iterations"],
        lr=1.0,
        print_every=400,
        outdir=lbfgs_outdir,
        use_gradnorm=True,
        w_pde=final_w_pde,
        w_bc=final_w_bc,
        w_ic=final_w_ic,
        use_ansatz=use_ansatz,
        material=material,
        x_ic=x_ic_dense if not use_ansatz else None,
    )

    # Load best overall checkpoint from L-BFGS phase
    model.load_state_dict(torch.load(lbfgs_outdir, map_location=device, weights_only=True))

    # 5. Save Loss Progression Curve
    print("\nSaving loss progression curve...")
    adam_len  = len(adam_total)
    lbfgs_len = len(lbfgs_total)
    adam_steps  = range(1, adam_len + 1)
    lbfgs_steps = range(adam_len + 1, adam_len + lbfgs_len + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(adam_steps,  adam_total,  label="Adam Phase",              color="#1f77b4", linewidth=2, alpha=0.85)
    plt.plot(lbfgs_steps, lbfgs_total, label="L-BFGS Phase (Fine-Tuning)", color="#2ca02c", linewidth=2, alpha=0.85)
    plt.yscale("log")
    plt.xlabel("Iteration Step")
    plt.ylabel("Total Loss")
    plt.title(f"Loss Curve - {mtype} ({material.name})")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(frameon=True, facecolor="white", edgecolor="gainsboro")
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 5b. Save Grad-Norm Weight Progression (Adam phase only)
    print("Saving grad-norm weight progression...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(adam_steps, w_pde_hist, label="w_pde (PDE residual)",  color="#d62728", linewidth=1.8, alpha=0.9)
    ax.plot(adam_steps, w_bc_hist,  label="w_bc  (Absorbing BC)",  color="#ff7f0e", linewidth=1.8, alpha=0.9)
    if not use_ansatz:
        ax.plot(adam_steps, w_ic_hist,  label="w_ic  (Initial Cond.)", color="#9467bd", linewidth=1.8, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("Adam Iteration Step")
    ax.set_ylabel("Grad-Norm Weight  (log scale)")
    ax.set_title(f"Grad-Norm Adaptive Weights - {mtype} ({material.name})")
    ax.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.5, label="Equal weight (1.0)")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, facecolor="white", edgecolor="gainsboro")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "gradnorm_weights.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 5c. Save causal convergence data + plots
    print("Saving causal convergence data...")

    # Determine final convergence status (paper criterion: w_min -> 1)
    final_w_min = w_min_hist[-1] if w_min_hist else 0.0
    causal_converged = final_w_min >= 0.99

    causal_data = {
        "converged":       causal_converged,
        "final_w_min":     final_w_min,
        "w_min_history":   w_min_hist,           # one value per Adam iteration
        "chunk_snapshots": w_chunks_snaps,        # (iter, [w_0..w_31]) at print_every steps
    }
    with open(os.path.join(output_dir, "causal_convergence.json"), "w") as f:
        json.dump(causal_data, f, indent=2)
    print(f"  -> Causal converged: {causal_converged}  (final w_min = {final_w_min:.4f})")

    # Plot 1: w_min over Adam iterations
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(adam_steps, w_min_hist, color="#e377c2", linewidth=1.8, alpha=0.9,
            label=r"$w_{\min}$ (paper convergence criterion)")
    ax.axhline(1.0, color="green",  linewidth=1.2, linestyle="--", alpha=0.7, label="Target ($w=1$)")
    ax.axhline(0.99, color="green", linewidth=0.8, linestyle=":",  alpha=0.5, label="Convergence threshold (0.99)")
    ax.set_xlabel("Adam Iteration Step")
    ax.set_ylabel(r"$w_{\min}$ (min causal weight across all chunks)")
    ax.set_title(f"Causal Convergence - {mtype} ({material.name})")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, facecolor="white", edgecolor="gainsboro")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "causal_wmin.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: heatmap of per-chunk weights at each snapshot (shows frontier advancing)
    if w_chunks_snaps:
        snap_iters  = [s[0] for s in w_chunks_snaps]
        snap_weights = np.array([s[1] for s in w_chunks_snaps])   # (n_snaps, n_chunks)
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(snap_weights.T, aspect="auto", origin="lower",
                       extent=[snap_iters[0], snap_iters[-1], 0.5, snap_weights.shape[1] + 0.5],
                       cmap="plasma", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax, label="Causal weight $w_m$")
        ax.set_xlabel("Adam Iteration Step")
        ax.set_ylabel("Time chunk index $m$")
        ax.set_title(f"Causal Weight Heatmap - {mtype} ({material.name})\n"
                     f"(yellow = weight≈1, purple = weight≈0 = not yet trained)")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "causal_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 6. Generate High-Precision Finite Difference Reference Solution
    print("Generating high-precision reference FD wave solution...")
    sigma_g = 0.1
    f_fn = lambda x_val: pbd.gaussian_ic(torch.tensor(x_val, dtype=torch.float32).reshape(-1, 1), sigma_g=sigma_g).flatten().cpu().numpy()
    g_fn = lambda x_val: 0.0
    c_fn = lambda x_val: material.Vp(torch.tensor(x_val, dtype=torch.float32).reshape(-1, 1)).item()
    Nx_fd = 200
    x_fd, t_fd, u_fd = pbd.solve_reference_fd(f_fn, g_fn, c_fn, Nx=Nx_fd, CFL=0.9)
    x_mesh_fd, t_mesh_fd = np.meshgrid(x_fd, t_fd, indexing='ij')
    x_eval_fd = torch.tensor(x_mesh_fd, dtype=torch.float32).reshape(-1, 1).to(device)
    t_eval_fd = torch.tensor(t_mesh_fd, dtype=torch.float32).reshape(-1, 1).to(device)

    def _call_model_local(net, x_in, t_in):
        from src.models import WavKAN
        from kan import KAN
        if isinstance(net, KAN) or isinstance(net, WavKAN):
            return net(torch.cat([x_in, t_in], dim=-1))
        return net(x_in, t_in)

    # 7. Evaluate best Adam model
    print("Evaluating best Adam model...")
    model_adam = instantiate_model(cfg, device)
    model_adam.load_state_dict(torch.load(adam_outdir, map_location=device, weights_only=True))
    model_adam.eval()
    with torch.no_grad():
        nn_out_adam = _call_model_local(model_adam, x_eval_fd, t_eval_fd)
        u_pred_adam = pbd.apply_ansatz(nn_out_adam, x_eval_fd, t_eval_fd, sigma_g=sigma_g, use_ansatz=use_ansatz)
        u_pred_adam_np = u_pred_adam.cpu().numpy().reshape(Nx_fd + 1, -1)
    residual_adam = u_fd - u_pred_adam_np
    rel_l2_adam = float(100 * np.linalg.norm(residual_adam) / np.linalg.norm(u_fd))
    print(f"  -> Adam Checkpoint Relative L2 Error: {rel_l2_adam:.6f}%")

    # Evaluate IC metrics for Adam model
    disp_rel_l2_adam, vel_mse_adam = evaluate_ic_metrics(model_adam, use_ansatz, material, device)
    print(f"  -> Adam Checkpoint IC Displacement L2 Error: {disp_rel_l2_adam:.6f}%")
    print(f"  -> Adam Checkpoint IC Velocity MSE: {vel_mse_adam:.6e}")

    # 8. Evaluate best L-BFGS model
    print("Evaluating best L-BFGS model...")
    model_lbfgs = instantiate_model(cfg, device)
    model_lbfgs.load_state_dict(torch.load(lbfgs_outdir, map_location=device, weights_only=True))
    model_lbfgs.eval()
    with torch.no_grad():
        nn_out_lbfgs = _call_model_local(model_lbfgs, x_eval_fd, t_eval_fd)
        u_pred_lbfgs = pbd.apply_ansatz(nn_out_lbfgs, x_eval_fd, t_eval_fd, sigma_g=sigma_g, use_ansatz=use_ansatz)
        u_pred_lbfgs_np = u_pred_lbfgs.cpu().numpy().reshape(Nx_fd + 1, -1)
    residual_lbfgs = u_fd - u_pred_lbfgs_np
    rel_l2_lbfgs = float(100 * np.linalg.norm(residual_lbfgs) / np.linalg.norm(u_fd))
    print(f"  -> L-BFGS Checkpoint Relative L2 Error: {rel_l2_lbfgs:.6f}%")

    # Evaluate IC metrics for L-BFGS model
    disp_rel_l2_lbfgs, vel_mse_lbfgs = evaluate_ic_metrics(model_lbfgs, use_ansatz, material, device)
    print(f"  -> L-BFGS Checkpoint IC Displacement L2 Error: {disp_rel_l2_lbfgs:.6f}%")
    print(f"  -> L-BFGS Checkpoint IC Velocity MSE: {vel_mse_lbfgs:.6e}")

    # 9. Save metrics to JSON file
    metrics = {
        "model_type": mtype, "n_hidden": nh, "hidden_width": hw,
        "material_type": material.name, "use_ansatz": use_ansatz,
        "relative_l2_error_best_adam_percent": rel_l2_adam,
        "relative_l2_error_best_lbfgs_percent": rel_l2_lbfgs,
        "ic_displacement_l2_error_adam_percent": disp_rel_l2_adam,
        "ic_velocity_mse_adam": vel_mse_adam,
        "ic_displacement_l2_error_lbfgs_percent": disp_rel_l2_lbfgs,
        "ic_velocity_mse_lbfgs": vel_mse_lbfgs
    }
    with open(os.path.join(output_dir, "l2_errors.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 10. Save solution comparison imshow plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].set_title("FD Reference Solution")
    im0 = axes[0].imshow(np.flip(np.transpose(u_fd), 0), extent=[x_fd.min(), x_fd.max(), t_fd.min(), t_fd.max()], cmap='plasma', aspect='auto')
    fig.colorbar(im0, ax=axes[0])
    axes[1].set_title(f"{mtype} Prediction (L-BFGS)")
    im1 = axes[1].imshow(np.flip(np.transpose(u_pred_lbfgs_np), 0), extent=[x_fd.min(), x_fd.max(), t_fd.min(), t_fd.max()], cmap='plasma', aspect='auto')
    fig.colorbar(im1, ax=axes[1])
    axes[2].set_title("Absolute Residuals")
    im2 = axes[2].imshow(np.flip(np.transpose(np.abs(residual_lbfgs)), 0), extent=[x_fd.min(), x_fd.max(), t_fd.min(), t_fd.max()], cmap='coolwarm', aspect='auto')
    fig.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.set_xlabel("Space x"); ax.set_ylabel("Time t")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "solution_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Finished Run: {run_name} | IC: {'hardcoded' if use_ansatz else 'loss'} | Best L-BFGS L2 Error: {rel_l2_lbfgs:.4f}% | IC Disp L2 Error: {disp_rel_l2_lbfgs:.4f}% | IC Vel MSE: {vel_mse_lbfgs:.4e}")

def main():
    import sys
    # Fix UnicodeEncodeError on Windows cp1252 terminals
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description="Wave Equation PINN/KAN Experiment Runner")
    parser.add_argument("--test", action="store_true",
                        help="Run a fast sanity-check test: 5 Adam + 5 L-BFGS iterations, one config per architecture")
    parser.add_argument("--run-id", type=int, default=None,
                        help="(SLURM) Run only this experiment index (0-based). "
                             "Maps to a unique (material, config, ansatz) combination. "
                             "Total range: 0 to N_configs*2*N_materials-1.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override the output base directory (e.g. $SCRATCH/wave_results on Sherlock). ")
    parser.add_argument("--adam-iterations", type=int, default=None,
                        help="Override number of Adam iterations")
    parser.add_argument("--lbfgs-iterations", type=int, default=None,
                        help="Override number of L-BFGS iterations")
    args = parser.parse_args()

    # Override iterations if supplied
    if args.adam_iterations is not None:
        CONFIG["adam_iterations"] = args.adam_iterations
    if args.lbfgs_iterations is not None:
        CONFIG["lbfgs_iterations"] = args.lbfgs_iterations

    # Override output directory if --output-dir was supplied (e.g. from SLURM $SCRATCH)
    if args.output_dir is not None:
        CONFIG["output_base_dir"] = args.output_dir

    # Seed globally before anything else
    set_seed(CONFIG["seed"])
    print(f"Global seed set to {CONFIG['seed']} for reproducibility.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 50)
    if device.type == "cuda":
        gpu_idx   = torch.cuda.current_device()
        gpu_name  = torch.cuda.get_device_name(gpu_idx)
        vram_total = torch.cuda.get_device_properties(gpu_idx).total_memory / 1024**3
        torch.cuda.set_per_process_memory_fraction(0.8, device=gpu_idx)
        print(f"  DEVICE : GPU [OK]")
        print(f"  NAME   : {gpu_name}")
        print(f"  VRAM   : {vram_total:.1f} GB total  (capped at 80% = {vram_total*0.8:.1f} GB)")
    else:
        print(f"  DEVICE : CPU  [WARNING] no GPU detected, training will be very slow!")
        print(f"  CUDA available : {torch.cuda.is_available()}")
        print(f"  Tip: check your venv is activated and PyTorch has CUDA support.")
    print("=" * 50 + "\n")

    if args.test:
        print("\n*** TEST MODE: 1 config per arch, 5 Adam + 5 L-BFGS iterations, all 3 materials ***\n")
        test_configs = [
            {"model_type": "PINN",              "n_hidden": 2, "hidden_width": 32, "extra_params": {}},
            {"model_type": "FourierFeaturePINN", "n_hidden": 2, "hidden_width": 32, "extra_params": {"n_fourier": 64, "sigma": 3.0}},
            {"model_type": "PirateNet",          "n_hidden": 2, "hidden_width": 32, "extra_params": {"n_fourier": 64, "sigma": 3.0}},
            {"model_type": "KAN",                "n_hidden": 1, "hidden_width": 16, "extra_params": {"G": 3, "k": 2}},
            {"model_type": "WavKAN",             "n_hidden": 1, "hidden_width": 16, "extra_params": {"wavelet_type": "morlet"}},
        ]
        CONFIG["adam_iterations"]       = 5
        CONFIG["lbfgs_iterations"]      = 5
        CONFIG["lbfgs_only_iterations"] = 5
        CONFIG["ic_warmup_iterations"]  = 5
        CONFIG["ic_gradnorm_delay"]     = 0
        CONFIG["output_base_dir"]       = "./experiment_results_test/"
        configs_to_run = test_configs
    else:
        configs_to_run = MODEL_CONFIGS

    # Run experiments for ALL 3 materials
    all_materials = [
        HomogeneousModel(),
        TwoLayerModel(),
        MultiLayerModel()
    ]

    # Build the full ordered list of run tuples (material, cfg, use_ansatz, lbfgs_only).
    # First half: Adam+LBFGS runs (ansatz True/False).
    # Second half: LBFGS-only runs (ansatz True/False).
    # Order is stable so --run-id indices never change.
    adam_lbfgs_runs = [
        (material, cfg, use_ansatz, False)
        for material   in all_materials
        for cfg        in configs_to_run
        for use_ansatz in [True, False]
    ]
    lbfgs_only_runs = [
        (material, cfg, use_ansatz, True)
        for material   in all_materials
        for cfg        in configs_to_run
        for use_ansatz in [True, False]
    ]
    all_runs = adam_lbfgs_runs + lbfgs_only_runs
    total_runs = len(all_runs)  # 360 for full sweep

    if args.run_id is not None:
        # ── SLURM / parallel mode: run exactly one experiment ──────────────────
        if not (0 <= args.run_id < total_runs):
            raise ValueError(f"--run-id {args.run_id} out of range [0, {total_runs - 1}]")
        material, cfg, use_ansatz, lbfgs_only = all_runs[args.run_id]
        print(f"\nSLURM mode: running experiment {args.run_id + 1}/{total_runs}")
        print(f"  Material   : {material.name}")
        print(f"  Config     : {cfg}")
        print(f"  Ansatz     : {use_ansatz}")
        print(f"  LBFGS Only : {lbfgs_only}")
        run_single_experiment(cfg, use_ansatz, material, device,
                              idx=args.run_id + 1, total_runs=total_runs,
                              lbfgs_only=lbfgs_only)
        print(f"\nExperiment {args.run_id + 1}/{total_runs} complete.")
    else:
        # ── Sequential mode: original behaviour ───────────────────────────────
        run_idx = 1
        for material, cfg, use_ansatz, lbfgs_only in all_runs:
            if run_idx == 1 or all_runs[run_idx - 2][0] != material:
                print(f"\n{'#' * 80}")
                print(f"#  MATERIAL: {material.name}  |  X: [{material.x_min:.2f}, {material.x_max:.2f}]  |  T: [{CONFIG['t_min']}, {CONFIG['t_max']}]")
                print(f"{'#' * 80}")
            run_single_experiment(cfg, use_ansatz, material, device, run_idx, total_runs,
                                  lbfgs_only=lbfgs_only)
            run_idx += 1

        print("\n" + "=" * 80)
        print(f" ALL {total_runs} EXPERIMENTS COMPLETED AND LOGGED SUCCESSFULLY!")
        print(f" Output Location: {CONFIG['output_base_dir']}")
        print("=" * 80)

if __name__ == "__main__":
    main()
