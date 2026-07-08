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

from src.models import (build_mlp, build_kan, FourierFeaturePINN, PirateNet,
                        WavKAN, ChebyshevKAN, FourierWavKAN)
from src.train import train_adam, train_lbfgs, train_ic_warmup
from src.losses.wave_loss import losses, losses_gradnorm, _call_model, reset_rba_state
import problem_data as pbd
from wave.materials import HomogeneousModel, TwoLayerModel, MultiLayerModel

# Set matplotlib plotting configurations for high-quality visuals
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})


_BACKEND: str = "pytorch"   # overridden in main() via --backend arg


def set_seed(seed: int):
    """Seed everything for full reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # multi-GPU safety
    torch.backends.cudnn.deterministic = True  # deterministic convolutions
    torch.backends.cudnn.benchmark     = False  # disable auto-tuner (not reproducible)


def _compute_fd_reference(material, cfg):
    """
    Run the FD reference solver once for a given material and return
    (x_fd, t_fd, u_fd, x_eval_fd, t_eval_fd) as a 5-tuple.

    Callers cache this per material so the solve only happens once per sweep.
    """
    sigma_g = cfg["sigma_g"]
    f_fn = lambda x_val: pbd.gaussian_ic(
        torch.tensor(x_val, dtype=torch.float32).reshape(-1, 1), sigma_g=sigma_g
    ).flatten().cpu().numpy()
    g_fn = lambda x_val: 0.0
    Nx_fd = 1000
    x_fd, t_fd, u_fd = pbd.solve_reference_fd(
        f_fn, g_fn, material, Nx=Nx_fd, CFL=0.9, t_max=cfg["t_max"]
    )
    x_mesh_fd, t_mesh_fd = np.meshgrid(x_fd, t_fd, indexing='ij')
    return x_fd, t_fd, u_fd, x_mesh_fd, t_mesh_fd


def _save_solution_plot(u_fd, u_pred, x_fd, t_fd, model_label, outpath):
    """Save a 3-panel FD reference | prediction | absolute residuals plot."""
    residual = u_fd - u_pred
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].set_title("FD Reference Solution")
    im0 = axes[0].imshow(
        np.flip(np.transpose(u_fd), 0),
        extent=[x_fd.min(), x_fd.max(), t_fd.min(), t_fd.max()],
        cmap='plasma', aspect='auto'
    )
    fig.colorbar(im0, ax=axes[0])
    axes[1].set_title(f"{model_label} Prediction")
    im1 = axes[1].imshow(
        np.flip(np.transpose(u_pred), 0),
        extent=[x_fd.min(), x_fd.max(), t_fd.min(), t_fd.max()],
        cmap='plasma', aspect='auto'
    )
    fig.colorbar(im1, ax=axes[1])
    axes[2].set_title("Absolute Residuals")
    im2 = axes[2].imshow(
        np.flip(np.transpose(np.abs(residual)), 0),
        extent=[x_fd.min(), x_fd.max(), t_fd.min(), t_fd.max()],
        cmap='coolwarm', aspect='auto'
    )
    fig.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.set_xlabel("Space x")
        ax.set_ylabel("Time t")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

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
    {"model_type": "WavKAN", "n_hidden": 4, "hidden_width": 16, "extra_params": {"wavelet_type": "mexican_hat"}},


    # 6. ChebyshevKAN sweeps
    {"model_type": "ChebyshevKAN", "n_hidden": 2, "hidden_width": 16, "extra_params": {"degree": 3}},
    {"model_type": "ChebyshevKAN", "n_hidden": 2, "hidden_width": 16, "extra_params": {"degree": 5}},
    {"model_type": "ChebyshevKAN", "n_hidden": 3, "hidden_width": 16, "extra_params": {"degree": 5}},
    {"model_type": "ChebyshevKAN", "n_hidden": 4, "hidden_width": 16, "extra_params": {"degree": 5}},

    # 7. FourierWavKAN: random Fourier embedding -> WavKAN trunk
    {"model_type": "FourierWavKAN", "n_hidden": 2, "hidden_width": 16, "extra_params": {"n_fourier": 32, "sigma": 3.0, "wavelet_type": "mexican_hat"}},
    {"model_type": "FourierWavKAN", "n_hidden": 2, "hidden_width": 16, "extra_params": {"n_fourier": 32, "sigma": 3.0, "wavelet_type": "morlet"}},
    {"model_type": "FourierWavKAN", "n_hidden": 3, "hidden_width": 16, "extra_params": {"n_fourier": 32, "sigma": 3.0, "wavelet_type": "mexican_hat"}},
    {"model_type": "FourierWavKAN", "n_hidden": 3, "hidden_width": 16, "extra_params": {"n_fourier": 32, "sigma": 3.0, "wavelet_type": "morlet"}},

    # 8. SOAP optimizer arms
    {"model_type": "PINN",               "n_hidden": 3, "hidden_width": 64, "extra_params": {},                                                          "optimizer": "soap"},
    {"model_type": "FourierFeaturePINN", "n_hidden": 3, "hidden_width": 64, "extra_params": {"n_fourier": 128, "sigma": 3.0},                            "optimizer": "soap"},
    {"model_type": "PirateNet",          "n_hidden": 3, "hidden_width": 64, "extra_params": {"n_fourier": 128, "sigma": 3.0},                            "optimizer": "soap"},
    {"model_type": "WavKAN",             "n_hidden": 2, "hidden_width": 16, "extra_params": {"wavelet_type": "mexican_hat"},                             "optimizer": "soap"},
    {"model_type": "ChebyshevKAN",       "n_hidden": 2, "hidden_width": 16, "extra_params": {"degree": 5},                                               "optimizer": "soap"},

    # 9. RBA weighting arms
    {"model_type": "PINN",               "n_hidden": 3, "hidden_width": 64, "extra_params": {},                                                          "weighting": "rba"},
    {"model_type": "FourierFeaturePINN", "n_hidden": 3, "hidden_width": 64, "extra_params": {"n_fourier": 128, "sigma": 3.0},                            "weighting": "rba"},
    {"model_type": "PirateNet",          "n_hidden": 3, "hidden_width": 64, "extra_params": {"n_fourier": 128, "sigma": 3.0},                            "weighting": "rba"},
    {"model_type": "WavKAN",             "n_hidden": 2, "hidden_width": 16, "extra_params": {"wavelet_type": "mexican_hat"},                             "weighting": "rba"},
    {"model_type": "ChebyshevKAN",       "n_hidden": 2, "hidden_width": 16, "extra_params": {"degree": 5},                                               "weighting": "rba"},

    # 10. SOAP + RBA combined
    {"model_type": "FourierFeaturePINN", "n_hidden": 3, "hidden_width": 64, "extra_params": {"n_fourier": 128, "sigma": 3.0},                            "optimizer": "soap", "weighting": "rba"},
    {"model_type": "ChebyshevKAN",       "n_hidden": 2, "hidden_width": 16, "extra_params": {"degree": 5},                                               "optimizer": "soap", "weighting": "rba"},
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

    # ── Causal training ───────────────────────────────────────────────────────
    "causal_n_chunks": 16,           # Time slabs for causal loss (16 vs 32: each slab thicker, propagates faster)
    "causal_tol_start": 0.1,         # ε at iter=0: loose weighting, all chunks see gradients
    "causal_tol_end": 1.0,           # ε at iter=adam_iterations: strict causal enforcement
    "causal_continuation_budget": 9000,  # Extra Adam steps if w_min < 0.9 after main phase (0 to disable)

    # ── Gaussian IC width ─────────────────────────────────────────────────────
    # Changing this propagates to FD reference, ansatz, IC loss, and evaluation.
    # All six call sites read from here so they stay in sync.
    "sigma_g": 0.1,

    # ── Reproducibility ───────────────────────────────────────────────────────
    "seed": 42,                      # Global random seed for full reproducibility

    # ── Output ────────────────────────────────────────────────────────────────
    "output_base_dir": "./experiment_results/"  # Base directory (material subdirs added automatically)
}

def _cfg_variant(cfg):
    """Returns (optimizer_name, weighting, tag) for a model config."""
    opt_name  = cfg.get("optimizer", "adam")
    weighting = cfg.get("weighting", "causal_gradnorm")
    tag = ""
    if opt_name == "soap":
        tag += "_soap"
    if weighting == "rba":
        tag += "_rba"
    return opt_name, weighting, tag


def _resolve_run_name(cfg, use_ansatz, lbfgs_only):
    """Unique run name from architecture + hyperparams + variant + ansatz mode."""
    mtype = cfg["model_type"]
    nh, hw, ep = cfg["n_hidden"], cfg["hidden_width"], cfg["extra_params"]
    _, _, variant = _cfg_variant(cfg)
    tag = ""
    if "sigma" in ep:
        tag += f"_sigma{int(ep['sigma'])}"
    if "wavelet_type" in ep:
        tag += f"_{ep['wavelet_type']}"
    if "degree" in ep:
        tag += f"_deg{ep['degree']}"
    suffix = (f"_lbfgs_only_ansatz_{'true' if use_ansatz else 'false'}"
              if lbfgs_only else f"_ansatz_{'true' if use_ansatz else 'false'}")
    return f"{mtype}_h{nh}_w{hw}{tag}{variant}{suffix}"


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
    elif mtype == "ChebyshevKAN":
        width = [pbd.NUM_INPUTS] + [hw] * nh + [pbd.NUM_OUTPUTS]
        model = ChebyshevKAN(width, degree=ep["degree"])
    elif mtype == "FourierWavKAN":
        model = FourierWavKAN(hidden_layers=nh, hidden_units=hw,
                              n_fourier=ep["n_fourier"], sigma=ep["sigma"],
                              wavelet_type=ep["wavelet_type"])
    else:
        raise ValueError(f"Unknown model name: {mtype}")

    return model.to(device)


def instantiate_model_jax(cfg, seed: int = 42):
    """JAX equivalent: returns an equinox model (no .to(device) needed)."""
    import jax
    from src.models_jax import (
        build_mlp_jax, build_kan_jax, FourierFeaturePINN as FFP_jax,
        PirateNet as PN_jax, WavKAN as WavKAN_jax,
        ChebyshevKAN as ChebyKAN_jax, FourierWavKAN as FWKAN_jax,
    )
    mtype = cfg["model_type"]
    nh    = cfg["n_hidden"]
    hw    = cfg["hidden_width"]
    ep    = cfg["extra_params"]
    key   = jax.random.PRNGKey(seed)

    if mtype == "PINN":
        return build_mlp_jax(pbd.NUM_INPUTS, pbd.NUM_OUTPUTS,
                              n_hidden=nh, hidden_width=hw, key=key)
    elif mtype == "FourierFeaturePINN":
        return FFP_jax(hidden_layers=nh, hidden_units=hw,
                       n_fourier=ep["n_fourier"], sigma=ep["sigma"], key=key)
    elif mtype == "PirateNet":
        return PN_jax(n_blocks=nh, units=hw,
                      n_fourier=ep["n_fourier"], sigma=ep["sigma"], key=key)
    elif mtype == "KAN":
        return build_kan_jax(pbd.NUM_INPUTS, pbd.NUM_OUTPUTS,
                              n_hidden=nh, hidden_width=hw,
                              G=ep["G"], k=ep["k"], key=key)
    elif mtype == "WavKAN":
        width = [pbd.NUM_INPUTS] + [hw] * nh + [pbd.NUM_OUTPUTS]
        return WavKAN_jax(width, wavelet_type=ep["wavelet_type"], key=key)
    elif mtype == "ChebyshevKAN":
        width = [pbd.NUM_INPUTS] + [hw] * nh + [pbd.NUM_OUTPUTS]
        return ChebyKAN_jax(width, degree=ep["degree"], key=key)
    elif mtype == "FourierWavKAN":
        return FWKAN_jax(hidden_layers=nh, hidden_units=hw,
                         n_fourier=ep["n_fourier"], sigma=ep["sigma"],
                         wavelet_type=ep["wavelet_type"], key=key)
    else:
        raise ValueError(f"Unknown model name: {mtype}")

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
    
    sigma_g = CONFIG["sigma_g"]
    # We must enable gradients to perform autograd.grad for velocity u_t
    with torch.set_grad_enabled(True):
        nn_out = _call_model(model, x_eval, t_eval)
        u_ic = wave_problem.apply_ansatz(nn_out, x_eval, t_eval, sigma_g=sigma_g, use_ansatz=use_ansatz)
        ic_ref = wave_problem.gaussian_ic(x_eval, sigma_g=sigma_g)
        
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

def predict_on_grid(model, x_eval, t_eval, use_ansatz, sigma_g=0.1, batch_size=200_000):
    """Batched no-grad evaluation of model + ansatz on a large (x, t) grid."""
    from src.losses.wave_loss import _call_model
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, x_eval.shape[0], batch_size):
            xb = x_eval[i:i + batch_size]
            tb = t_eval[i:i + batch_size]
            nn_out = _call_model(model, xb, tb)
            outs.append(pbd.apply_ansatz(nn_out, xb, tb, sigma_g=sigma_g, use_ansatz=use_ansatz))
    model.train()
    return torch.cat(outs, dim=0).cpu().numpy()


def run_single_experiment(cfg, use_ansatz, material, device, idx, total_runs,
                          lbfgs_only=False, backend="pytorch", fd_reference=None):
    """
    Run one experiment.

    Args:
        lbfgs_only:    If True, skip Adam and run L-BFGS from scratch.
        backend:       "pytorch" or "jax".
        fd_reference:  Pre-computed (x_fd, t_fd, u_fd, x_mesh_fd, t_mesh_fd) tuple.
                       Computed once per material if not supplied.
    """
    # Re-seed at the start of every experiment so each run is independently
    # reproducible regardless of what ran before it.
    set_seed(CONFIG["seed"])

    mtype = cfg["model_type"]
    nh = cfg["n_hidden"]
    hw = cfg["hidden_width"]
    ep = cfg["extra_params"]
    opt_name, weighting, _ = _cfg_variant(cfg)
    use_rba = (weighting == "rba")

    # Resolve unique name based on parameters
    run_name = _resolve_run_name(cfg, use_ansatz, lbfgs_only)

    output_dir = os.path.join(CONFIG["output_base_dir"], material.name, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f" [{idx}/{total_runs}] RUNNING: {mtype} (Layers={nh}, Width={hw})")
    if "sigma" in ep:
        print(f"            Sigma: {ep['sigma']}")
    if "wavelet_type" in ep:
        print(f"            Wavelet: {ep['wavelet_type']}")
    if "degree" in ep:
        print(f"            Chebyshev degree: {ep['degree']}")
    if opt_name != "adam" or weighting != "causal_gradnorm":
        print(f"            Variant: optimizer={opt_name} | weighting={weighting}")
    if lbfgs_only:
        print(f"            Mode: L-BFGS Only ({CONFIG['lbfgs_only_iterations']} iters) | IC: {'hardcoded' if use_ansatz else 'loss'} | Material: {material.name}")
    else:
        print(f"            IC Setup: {'IC hardcoded' if use_ansatz else 'IC loss'} | Material: {material.name}")
    print("=" * 80)

    # ── JAX backend dispatch ─────────────────────────────────────────────────
    if backend == "jax":
        _run_single_experiment_jax(cfg, use_ansatz, material, idx, total_runs, lbfgs_only,
                                   fd_reference=fd_reference)
        return

    # 1. Instantiate Model (and clear any RBA multipliers from a previous run)
    reset_rba_state()
    model = instantiate_model(cfg, device)

    # 2. Setup Collocation Training Grid
    x_coll = torch.linspace(material.x_min, material.x_max, CONFIG["Nx_collocation"], requires_grad=True).reshape(-1, 1).to(device)
    t_coll = torch.linspace(CONFIG["t_min"], CONFIG["t_max"], CONFIG["Nt_collocation"], requires_grad=True).reshape(-1, 1).to(device)
    x_mesh, t_mesh = torch.meshgrid(x_coll.flatten(), t_coll.flatten(), indexing='ij')
    x_train = x_mesh.reshape(-1, 1).to(device)
    t_train = t_mesh.reshape(-1, 1).to(device)

    # 2b. Dense IC grid: uniform coverage + extra points near the Gaussian peak
    sigma_g = CONFIG["sigma_g"]
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

        # FD reference (use cached value if provided)
        if fd_reference is None:
            fd_reference = _compute_fd_reference(material, CONFIG)
        x_fd, t_fd, u_fd, x_mesh_fd, t_mesh_fd = fd_reference
        x_eval_fd = torch.tensor(x_mesh_fd, dtype=torch.float32).reshape(-1, 1).to(device)
        t_eval_fd = torch.tensor(t_mesh_fd, dtype=torch.float32).reshape(-1, 1).to(device)
        Nx_fd = len(x_fd) - 1

        print("Evaluating best L-BFGS-only model...")
        u_pred_np = predict_on_grid(model, x_eval_fd, t_eval_fd, use_ansatz, sigma_g).reshape(Nx_fd + 1, -1)
        rel_l2 = float(100 * np.linalg.norm(u_fd - u_pred_np) / np.linalg.norm(u_fd))
        print(f"  -> L-BFGS-Only Relative L2 Error: {rel_l2:.6f}%")

        disp_rel_l2, vel_mse = evaluate_ic_metrics(model, use_ansatz, material, device)
        print(f"  -> IC Displacement L2 Error: {disp_rel_l2:.6f}%")
        print(f"  -> IC Velocity MSE: {vel_mse:.6e}")

        metrics = {
            "model_type": mtype, "n_hidden": nh, "hidden_width": hw,
            "material_type": material.name, "use_ansatz": use_ansatz,
            "extra_params": ep,
            "optimizer": "lbfgs_only",
            "lbfgs_only_iterations": n_iters,
            "relative_l2_error_lbfgs_only_percent": rel_l2,
            "ic_displacement_l2_error_percent": disp_rel_l2,
            "ic_velocity_mse": vel_mse
        }
        with open(os.path.join(output_dir, "l2_errors.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        _save_solution_plot(u_fd, u_pred_np, x_fd, t_fd,
                            f"{mtype} (L-BFGS Only)",
                            os.path.join(output_dir, "solution_comparison.png"))

        print(f"Finished Run: {run_name} | IC: {'hardcoded' if use_ansatz else 'loss'} | L-BFGS-Only L2 Error: {rel_l2:.4f}% | IC Disp L2 Error: {disp_rel_l2:.4f}% | IC Vel MSE: {vel_mse:.4e}")
        return

    # ── BRANCH: Adam + L-BFGS (original) ────────────────────────────────────
    # 3. Phase 1: Adam Training
    print(f"\n--- [1/2] Adam Optimizer: {CONFIG['adam_iterations']} Iterations ---")
    adam_outdir = os.path.join(output_dir, "model_best_adam.pkl")
    adam_total, adam_phys, adam_cond, final_w_pde, final_w_bc, final_w_ic, \
        w_pde_hist, w_bc_hist, w_ic_hist, w_min_hist, w_chunks_snaps, best_val = train_adam(
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
        causal_tolerance=(CONFIG["causal_tol_start"], CONFIG["causal_tol_end"]),
        use_ansatz=use_ansatz,
        material=material,
        x_ic=x_ic_dense if not use_ansatz else None,
        n_chunks=CONFIG["causal_n_chunks"],
        optimizer_name=opt_name,
        use_rba=use_rba,
    )

    # Load best checkpoint from Adam phase
    model.load_state_dict(torch.load(adam_outdir, map_location=device, weights_only=True))

    # 3b. Causal continuation — keep running Adam until w_min >= 0.9 or budget exhausted
    continuation_budget = CONFIG.get("causal_continuation_budget", 0)
    continuation_block  = 3000
    total_continuation  = 0
    while w_min_hist[-1] < 0.9 and total_continuation < continuation_budget:
        steps_left = min(continuation_block, continuation_budget - total_continuation)
        print(f"\n--- Causal continuation [{total_continuation}/{continuation_budget}] "
              f"w_min={w_min_hist[-1]:.4f} — running {steps_left} more Adam steps ---")
        cont = train_adam(
            model=model,
            inputs=[x_train, t_train],
            losses_func=losses_gradnorm,
            iterations=steps_left,
            lr=CONFIG["lr"] * 0.3,          # lower LR for fine-grained continuation
            print_every=1000,
            outdir=adam_outdir,              # keep updating best checkpoint
            use_gradnorm=True,
            gradnorm_update_freq=CONFIG.get("gradnorm_update_freq", 100),
            ic_gradnorm_delay=0,             # GradNorm already warmed up
            causal_tolerance=(0.5, 1.0),     # short ramp: medium → strict
            use_ansatz=use_ansatz,
            material=material,
            x_ic=x_ic_dense if not use_ansatz else None,
            n_chunks=CONFIG["causal_n_chunks"],
            initial_best=best_val,           # never regress the saved checkpoint
            optimizer_name=opt_name,
            use_rba=use_rba,
        )
        adam_offset   = len(adam_total)   # global step base for this continuation block
        adam_total   += cont[0];  adam_phys  += cont[1];  adam_cond  += cont[2]
        final_w_pde, final_w_bc, final_w_ic = cont[3], cont[4], cont[5]
        w_pde_hist   += cont[6];  w_bc_hist  += cont[7];  w_ic_hist  += cont[8]
        w_min_hist   += cont[9]
        w_chunks_snaps += [(s[0] + adam_offset, s[1]) for s in cont[10]]
        best_val      = cont[11]
        total_continuation += steps_left
        model.load_state_dict(torch.load(adam_outdir, map_location=device, weights_only=True))

    if total_continuation > 0:
        print(f"\n    Causal continuation done: {total_continuation} extra steps, "
              f"final w_min={w_min_hist[-1]:.4f}", flush=True)

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
        use_causal=False,   # plain residual: causal weights would change inside L-BFGS line searches
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

    # 6. FD reference (use cached value if provided, else compute now)
    if fd_reference is None:
        print("Generating high-precision reference FD wave solution...")
        fd_reference = _compute_fd_reference(material, CONFIG)
    x_fd, t_fd, u_fd, x_mesh_fd, t_mesh_fd = fd_reference
    x_eval_fd = torch.tensor(x_mesh_fd, dtype=torch.float32).reshape(-1, 1).to(device)
    t_eval_fd = torch.tensor(t_mesh_fd, dtype=torch.float32).reshape(-1, 1).to(device)
    Nx_fd = len(x_fd) - 1

    # 7. Evaluate best Adam checkpoint (load separately — model is already on L-BFGS weights)
    print("Evaluating best Adam model...")
    model_adam = instantiate_model(cfg, device)
    model_adam.load_state_dict(torch.load(adam_outdir, map_location=device, weights_only=True))
    u_pred_adam_np = predict_on_grid(model_adam, x_eval_fd, t_eval_fd, use_ansatz, sigma_g).reshape(Nx_fd + 1, -1)
    rel_l2_adam = float(100 * np.linalg.norm(u_fd - u_pred_adam_np) / np.linalg.norm(u_fd))
    print(f"  -> Adam Checkpoint Relative L2 Error: {rel_l2_adam:.6f}%")

    disp_rel_l2_adam, vel_mse_adam = evaluate_ic_metrics(model_adam, use_ansatz, material, device)
    print(f"  -> Adam Checkpoint IC Displacement L2 Error: {disp_rel_l2_adam:.6f}%")
    print(f"  -> Adam Checkpoint IC Velocity MSE: {vel_mse_adam:.6e}")

    # 8. Evaluate best L-BFGS checkpoint (model is already loaded with these weights)
    print("Evaluating best L-BFGS model...")
    u_pred_lbfgs_np = predict_on_grid(model, x_eval_fd, t_eval_fd, use_ansatz, sigma_g).reshape(Nx_fd + 1, -1)
    rel_l2_lbfgs = float(100 * np.linalg.norm(u_fd - u_pred_lbfgs_np) / np.linalg.norm(u_fd))
    print(f"  -> L-BFGS Checkpoint Relative L2 Error: {rel_l2_lbfgs:.6f}%")

    disp_rel_l2_lbfgs, vel_mse_lbfgs = evaluate_ic_metrics(model, use_ansatz, material, device)
    print(f"  -> L-BFGS Checkpoint IC Displacement L2 Error: {disp_rel_l2_lbfgs:.6f}%")
    print(f"  -> L-BFGS Checkpoint IC Velocity MSE: {vel_mse_lbfgs:.6e}")

    # 9. Save metrics
    metrics = {
        "model_type": mtype, "n_hidden": nh, "hidden_width": hw,
        "material_type": material.name, "use_ansatz": use_ansatz,
        "extra_params": ep,
        "optimizer": opt_name, "weighting": weighting,
        "relative_l2_error_best_adam_percent": rel_l2_adam,
        "relative_l2_error_best_lbfgs_percent": rel_l2_lbfgs,
        "ic_displacement_l2_error_adam_percent": disp_rel_l2_adam,
        "ic_velocity_mse_adam": vel_mse_adam,
        "ic_displacement_l2_error_lbfgs_percent": disp_rel_l2_lbfgs,
        "ic_velocity_mse_lbfgs": vel_mse_lbfgs
    }
    with open(os.path.join(output_dir, "l2_errors.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 10. Solution comparison plot
    _save_solution_plot(u_fd, u_pred_lbfgs_np, x_fd, t_fd,
                        f"{mtype} (L-BFGS)",
                        os.path.join(output_dir, "solution_comparison.png"))

    print(f"Finished Run: {run_name} | IC: {'hardcoded' if use_ansatz else 'loss'} | Best L-BFGS L2 Error: {rel_l2_lbfgs:.4f}% | IC Disp L2 Error: {disp_rel_l2_lbfgs:.4f}% | IC Vel MSE: {vel_mse_lbfgs:.4e}")

def _run_single_experiment_jax(cfg, use_ansatz, material, idx, total_runs, lbfgs_only=False,
                                fd_reference=None):
    """
    JAX-backend experiment runner.  Mirrors run_single_experiment() but uses:
      - equinox models (src/models_jax.py)
      - JAX loss functions (src/losses/wave_loss_jax.py)
      - optax Adam + jaxopt L-BFGS (src/train_jax.py)
    Plotting, FD evaluation, and JSON metrics are backend-agnostic (numpy).
    """
    import jax
    import jax.numpy as jnp
    from src.models_jax import physics_informed_init_jax
    from src.losses.wave_loss_jax import losses_gradnorm_jax
    from src.train_jax import (
        train_ic_warmup_jax, train_adam_jax, train_lbfgs_jax,
        save_model_jax, load_model_jax,
    )

    set_seed(CONFIG["seed"])

    mtype = cfg["model_type"]
    nh    = cfg["n_hidden"]
    hw    = cfg["hidden_width"]
    ep    = cfg["extra_params"]
    opt_name, weighting, _ = _cfg_variant(cfg)
    use_rba = (weighting == "rba")

    # Clear any RBA multipliers left over from a previous run in this process
    from src.losses.wave_loss_jax import _rba_state_jax
    _rba_state_jax["weights"] = None

    run_name = _resolve_run_name(cfg, use_ansatz, lbfgs_only) + "_jax"

    output_dir = os.path.join(CONFIG["output_base_dir"], material.name, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f" [{idx}/{total_runs}] RUNNING (JAX): {mtype} (Layers={nh}, Width={hw})")
    if "sigma" in ep:        print(f"            Sigma: {ep['sigma']}")
    if "wavelet_type" in ep: print(f"            Wavelet: {ep['wavelet_type']}")
    if "degree" in ep:       print(f"            Chebyshev degree: {ep['degree']}")
    if opt_name != "adam" or weighting != "causal_gradnorm":
        print(f"            Variant: optimizer={opt_name} | weighting={weighting}")
    print(f"            IC Setup: {'IC hardcoded' if use_ansatz else 'IC loss'}"
          f" | Material: {material.name}")
    print("=" * 80)

    # ── 1. Build collocation grids (JAX arrays) ───────────────────────────────
    sigma_g   = CONFIG["sigma_g"]
    Nx        = CONFIG["Nx_collocation"]
    Nt        = CONFIG["Nt_collocation"]
    x_coll    = jnp.linspace(material.x_min, material.x_max, Nx).reshape(-1, 1)
    t_coll    = jnp.linspace(CONFIG["t_min"], CONFIG["t_max"], Nt).reshape(-1, 1)
    xm, tm    = jnp.meshgrid(x_coll.reshape(-1), t_coll.reshape(-1), indexing='ij')
    x_train   = xm.reshape(-1, 1)
    t_train   = tm.reshape(-1, 1)

    # BC points: left/right boundary at every training t value
    x_bc_left  = jnp.full((Nt, 1), material.x_min)
    x_bc_right = jnp.full((Nt, 1), material.x_max)
    x_bc       = jnp.concatenate([x_bc_left,  x_bc_right], axis=0)   # (2*Nt, 1)
    t_bc       = jnp.concatenate([t_coll,      t_coll],     axis=0)   # (2*Nt, 1)

    # Dense IC grid
    x_ic_uniform = jnp.linspace(material.x_min, material.x_max, 200).reshape(-1, 1)
    peak_lo = max(material.x_min, -3 * sigma_g)
    peak_hi = min(material.x_max,  3 * sigma_g)
    x_ic_peak  = jnp.linspace(peak_lo, peak_hi, 300).reshape(-1, 1)
    x_ic_dense = jnp.concatenate([x_ic_uniform, x_ic_peak], axis=0)

    inputs = [x_train, t_train, x_bc, t_bc, x_ic_dense]

    # ── 2. Instantiate model ──────────────────────────────────────────────────
    model = instantiate_model_jax(cfg, seed=CONFIG["seed"])

    # ── 3. IC warm-up + PI-init ───────────────────────────────────────────────
    if not use_ansatz and not lbfgs_only:
        print(f"\n--- [0/2] IC Warm-up: {CONFIG['ic_warmup_iterations']} iterations ---")
        model = train_ic_warmup_jax(model, x_ic_dense, sigma_g=sigma_g,
                                     iterations=CONFIG["ic_warmup_iterations"])
        if mtype in ("FourierFeaturePINN", "PirateNet"):
            print("    Applying physics-informed output-layer init...")
            xt_all = jnp.concatenate([x_train, t_train], axis=-1)
            from src.losses.wave_loss_jax import gaussian_ic_jax
            ic_ref = gaussian_ic_jax(x_train, sigma_g)           # (N, 1) jnp array
            model  = physics_informed_init_jax(model, xt_all, ic_ref)
            print("    PI-init complete.", flush=True)

    adam_outdir  = os.path.join(output_dir, "model_best_adam.eqx")
    lbfgs_outdir = os.path.join(output_dir, "model_best_lbfgs.eqx")

    # ── L-BFGS only branch ────────────────────────────────────────────────────
    if lbfgs_only:
        n_iters = CONFIG["lbfgs_only_iterations"]
        print(f"\n--- [1/1] L-BFGS Only (JAX): {n_iters} Max Iterations ---")
        lbfgs_out = os.path.join(output_dir, "model_best_lbfgs_only.eqx")
        train_lbfgs_jax(model, inputs, losses_gradnorm_jax,
                         max_iter=n_iters, print_every=500,
                         outdir=lbfgs_out,
                         w_pde=1.0, w_bc=1.0, w_ic=1.0,
                         use_ansatz=use_ansatz, material=material, sigma_g=sigma_g,
                         n_chunks=CONFIG["causal_n_chunks"])
        model = load_model_jax(lbfgs_out, model)
        _save_eval_jax(model, cfg, use_ansatz, material, output_dir,
                       "lbfgs_only", {}, sigma_g, lbfgs_only=True,
                       fd_reference=fd_reference)
        return

    # ── Adam phase ────────────────────────────────────────────────────────────
    print(f"\n--- [1/2] Adam (JAX): {CONFIG['adam_iterations']} Iterations ---")
    (adam_total, adam_phys, adam_cond,
     final_w_pde, final_w_bc, final_w_ic,
     w_pde_hist, w_bc_hist, w_ic_hist,
     w_min_hist, w_chunks_snaps, best_val) = train_adam_jax(
        model, inputs, losses_gradnorm_jax,
        iterations=CONFIG["adam_iterations"],
        lr=CONFIG["lr"],
        print_every=2000,
        outdir=adam_outdir,
        use_gradnorm=True,
        gradnorm_update_freq=CONFIG.get("gradnorm_update_freq", 100),
        ic_gradnorm_delay=(CONFIG["ic_gradnorm_delay"] if not use_ansatz else 0),
        causal_tolerance=(CONFIG["causal_tol_start"], CONFIG["causal_tol_end"]),
        use_ansatz=use_ansatz,
        material=material, sigma_g=sigma_g,
        n_chunks=CONFIG["causal_n_chunks"],
        optimizer_name=opt_name,
        use_rba=use_rba,
    )
    model = load_model_jax(adam_outdir, model)

    # Causal continuation
    continuation_budget = CONFIG.get("causal_continuation_budget", 0)
    continuation_block  = 3000
    total_continuation  = 0
    while w_min_hist[-1] < 0.9 and total_continuation < continuation_budget:
        steps_left = min(continuation_block, continuation_budget - total_continuation)
        print(f"\n--- Causal continuation [{total_continuation}/{continuation_budget}] "
              f"w_min={w_min_hist[-1]:.4f} — running {steps_left} more Adam steps ---")
        cont = train_adam_jax(
            model, inputs, losses_gradnorm_jax,
            iterations=steps_left,
            lr=CONFIG["lr"] * 0.3,
            print_every=1000,
            outdir=adam_outdir,
            use_gradnorm=True,
            gradnorm_update_freq=CONFIG.get("gradnorm_update_freq", 100),
            ic_gradnorm_delay=0,
            causal_tolerance=(0.5, 1.0),
            use_ansatz=use_ansatz,
            material=material, sigma_g=sigma_g,
            n_chunks=CONFIG["causal_n_chunks"],
            initial_best=best_val,
            optimizer_name=opt_name,
            use_rba=use_rba,
        )
        adam_offset  = len(adam_total)   # global step base for this continuation block
        adam_total  += cont[0];  adam_phys  += cont[1]; adam_cond  += cont[2]
        final_w_pde, final_w_bc, final_w_ic = cont[3], cont[4], cont[5]
        w_pde_hist  += cont[6];  w_bc_hist  += cont[7]; w_ic_hist  += cont[8]
        w_min_hist  += cont[9]
        w_chunks_snaps += [(s[0] + adam_offset, s[1]) for s in cont[10]]
        best_val     = cont[11]
        total_continuation += steps_left
        model = load_model_jax(adam_outdir, model)

    # ── L-BFGS phase ─────────────────────────────────────────────────────────
    print(f"\n--- [2/2] L-BFGS (JAX): {CONFIG['lbfgs_iterations']} Max Iterations ---")
    train_lbfgs_jax(model, inputs, losses_gradnorm_jax,
                     max_iter=CONFIG["lbfgs_iterations"],
                     print_every=400, outdir=lbfgs_outdir,
                     w_pde=final_w_pde, w_bc=final_w_bc, w_ic=final_w_ic,
                     use_ansatz=use_ansatz, material=material, sigma_g=sigma_g,
                     n_chunks=CONFIG["causal_n_chunks"])
    model = load_model_jax(lbfgs_outdir, model)

    # ── Save plots and metrics ────────────────────────────────────────────────
    _save_eval_jax(model, cfg, use_ansatz, material, output_dir,
                   "lbfgs", {
                       "adam_total": adam_total,
                       "adam_phys":  adam_phys,
                       "adam_cond":  adam_cond,
                       "w_pde_hist": w_pde_hist,
                       "w_bc_hist":  w_bc_hist,
                       "w_ic_hist":  w_ic_hist,
                       "w_min_hist": w_min_hist,
                       "w_chunks_snaps": w_chunks_snaps,
                   }, sigma_g, lbfgs_only=False,
                   adam_outdir=adam_outdir, fd_reference=fd_reference)


def _save_eval_jax(model_lbfgs, cfg, use_ansatz, material, output_dir,
                    mode, history, sigma_g, lbfgs_only=False, adam_outdir=None,
                    fd_reference=None):
    """
    Evaluation + JSON + plots shared by both JAX branches.
    model_lbfgs is the final (best L-BFGS or L-BFGS-only) model.
    """
    import jax
    import jax.numpy as jnp
    from src.train_jax import load_model_jax
    from src.losses.wave_loss_jax import _causal_state_jax
    from src.models_jax import _call_model_jax

    mtype = cfg["model_type"]
    nh    = cfg["n_hidden"]
    hw    = cfg["hidden_width"]

    # ── Loss curves ───────────────────────────────────────────────────────────
    if not lbfgs_only and history:
        adam_len  = len(history["adam_total"])
        adam_steps = range(1, adam_len + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(adam_steps, history["adam_total"],
                 label="Adam Phase", color="#1f77b4", linewidth=2, alpha=0.85)
        plt.yscale("log"); plt.xlabel("Iteration"); plt.ylabel("Total Loss")
        plt.title(f"Loss Curve (JAX) - {mtype} ({material.name})")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend(frameon=True, facecolor="white", edgecolor="gainsboro")
        plt.savefig(os.path.join(output_dir, "loss_curve.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        # GradNorm weight plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(adam_steps, history["w_pde_hist"],
                label="w_pde", color="#d62728", linewidth=1.8, alpha=0.9)
        ax.plot(adam_steps, history["w_bc_hist"],
                label="w_bc",  color="#ff7f0e", linewidth=1.8, alpha=0.9)
        if not use_ansatz:
            ax.plot(adam_steps, history["w_ic_hist"],
                    label="w_ic", color="#9467bd", linewidth=1.8, alpha=0.9)
        ax.set_yscale("log"); ax.set_xlabel("Adam Iteration")
        ax.set_ylabel("Grad-Norm Weight (log)")
        ax.set_title(f"Grad-Norm Weights (JAX) - {mtype} ({material.name})")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend(frameon=True, facecolor="white", edgecolor="gainsboro")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "gradnorm_weights.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Causal w_min plot
        w_min_hist = history["w_min_hist"]
        if w_min_hist:
            final_w_min    = w_min_hist[-1]
            causal_conv    = final_w_min >= 0.99
            causal_data    = {
                "converged":       causal_conv,
                "final_w_min":     final_w_min,
                "w_min_history":   w_min_hist,
                "chunk_snapshots": history["w_chunks_snaps"],
            }
            with open(os.path.join(output_dir, "causal_convergence.json"), "w") as f:
                json.dump(causal_data, f, indent=2)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(adam_steps, w_min_hist, color="#e377c2", linewidth=1.8)
            ax.axhline(1.0,  color="green", linewidth=1.2, linestyle="--", alpha=0.7)
            ax.axhline(0.99, color="green", linewidth=0.8, linestyle=":",  alpha=0.5)
            ax.set_xlabel("Adam Iteration"); ax.set_ylabel("w_min")
            ax.set_title(f"Causal Convergence (JAX) - {mtype} ({material.name})")
            ax.set_ylim(-0.05, 1.1); ax.grid(True, linestyle="--", alpha=0.35)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "causal_wmin.png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)

    # ── FD reference ─────────────────────────────────────────────────────────
    if fd_reference is None:
        fd_reference = _compute_fd_reference(material, CONFIG)
    x_fd, t_fd, u_fd, xm_fd, tm_fd = fd_reference
    Nx_fd = len(x_fd) - 1

    def _predict_jax(model_jax):
        import jax.numpy as jnp
        x_ev = jnp.array(xm_fd.reshape(-1, 1), dtype=jnp.float32)
        t_ev = jnp.array(tm_fd.reshape(-1, 1), dtype=jnp.float32)
        # Batch through the model in chunks to avoid OOM
        chunk = 200_000
        outs  = []
        for i in range(0, x_ev.shape[0], chunk):
            xb = x_ev[i:i+chunk]; tb = t_ev[i:i+chunk]
            nn_out = _call_model_jax(model_jax, xb, tb)
            from src.losses.wave_loss_jax import apply_ansatz_jax
            outs.append(np.array(apply_ansatz_jax(nn_out, xb, tb, sigma_g, use_ansatz)))
        return np.concatenate(outs, axis=0).reshape(Nx_fd + 1, -1)

    opt_name, weighting, _ = _cfg_variant(cfg)
    metrics = {
        "model_type": mtype, "n_hidden": nh, "hidden_width": hw,
        "material_type": material.name, "use_ansatz": use_ansatz,
        "extra_params": cfg["extra_params"],
        "optimizer": ("lbfgs_only" if lbfgs_only else opt_name),
        "weighting": weighting,
        "backend": "jax",
    }

    if not lbfgs_only and adam_outdir and os.path.exists(adam_outdir):
        model_adam = load_model_jax(adam_outdir, model_lbfgs)
        u_adam     = _predict_jax(model_adam)
        rel_l2_adam = float(100 * np.linalg.norm(u_fd - u_adam) / np.linalg.norm(u_fd))
        metrics["relative_l2_error_best_adam_percent"] = rel_l2_adam
        print(f"  -> Adam L2 Error: {rel_l2_adam:.6f}%")

    u_lbfgs    = _predict_jax(model_lbfgs)
    rel_l2     = float(100 * np.linalg.norm(u_fd - u_lbfgs) / np.linalg.norm(u_fd))
    key_name   = "relative_l2_error_best_lbfgs_only_percent" if lbfgs_only \
                 else "relative_l2_error_best_lbfgs_percent"
    metrics[key_name] = rel_l2
    print(f"  -> L2 Error: {rel_l2:.6f}%")

    with open(os.path.join(output_dir, "l2_errors.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Solution comparison plot
    label = f"{mtype} (JAX {'L-BFGS Only' if lbfgs_only else 'L-BFGS'})"
    _save_solution_plot(u_fd, u_lbfgs, x_fd, t_fd, label,
                        os.path.join(output_dir, "solution_comparison.png"))
    print(f"Finished JAX Run: L2 = {rel_l2:.4f}%")


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
    parser.add_argument("--causal-continuation-budget", type=int, default=None,
                        help="Override causal continuation Adam budget (0 to disable)")
    parser.add_argument("--backend", type=str, default="jax",
                        choices=["pytorch", "jax"],
                        help="Compute backend: 'jax' (default; requires "
                             "pip install jax equinox optax jaxopt jaxkan) or 'pytorch'")
    parser.add_argument("--list-runs", action="store_true",
                        help="Print the full run-id -> experiment mapping and exit "
                             "(use to size SLURM arrays)")
    args = parser.parse_args()

    # Override iterations if supplied
    if args.adam_iterations is not None:
        CONFIG["adam_iterations"] = args.adam_iterations
    if args.lbfgs_iterations is not None:
        CONFIG["lbfgs_iterations"] = args.lbfgs_iterations
    if args.causal_continuation_budget is not None:
        CONFIG["causal_continuation_budget"] = args.causal_continuation_budget

    # Override output directory if --output-dir was supplied (e.g. from SLURM $SCRATCH)
    if args.output_dir is not None:
        CONFIG["output_base_dir"] = args.output_dir

    # Set compute backend
    global _BACKEND
    _BACKEND = args.backend
    if _BACKEND == "jax":
        import jax
        print(f"\nJAX backend selected — devices: {jax.devices()}")

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
            {"model_type": "ChebyshevKAN",       "n_hidden": 1, "hidden_width": 16, "extra_params": {"degree": 3}},
            {"model_type": "FourierWavKAN",      "n_hidden": 1, "hidden_width": 16, "extra_params": {"n_fourier": 16, "sigma": 3.0, "wavelet_type": "mexican_hat"}},
            {"model_type": "PINN",               "n_hidden": 2, "hidden_width": 32, "extra_params": {}, "optimizer": "soap"},
            {"model_type": "PINN",               "n_hidden": 2, "hidden_width": 32, "extra_params": {}, "weighting": "rba"},
        ]
        CONFIG["adam_iterations"]            = 5
        CONFIG["lbfgs_iterations"]           = 5
        CONFIG["lbfgs_only_iterations"]      = 5
        CONFIG["ic_warmup_iterations"]       = 5
        CONFIG["ic_gradnorm_delay"]          = 0
        CONFIG["causal_continuation_budget"] = 0   # disabled in test mode
        CONFIG["output_base_dir"]            = "./experiment_results_test/"
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
    # Optimizer/weighting variant configs are EXCLUDED from the LBFGS-only half:
    # those runs have no Adam phase, so the variants would be exact duplicates.
    base_configs = [
        cfg for cfg in configs_to_run
        if cfg.get("optimizer", "adam") == "adam"
        and cfg.get("weighting", "causal_gradnorm") == "causal_gradnorm"
    ]
    adam_lbfgs_runs = [
        (material, cfg, use_ansatz, False)
        for material   in all_materials
        for cfg        in configs_to_run
        for use_ansatz in [True, False]
    ]
    lbfgs_only_runs = [
        (material, cfg, use_ansatz, True)
        for material   in all_materials
        for cfg        in base_configs
        for use_ansatz in [True, False]
    ]
    all_runs = adam_lbfgs_runs + lbfgs_only_runs
    total_runs = len(all_runs)  # 528 for the full sweep (50 configs + 38 base)

    if args.list_runs:
        print(f"\nTotal runs: {total_runs}  (SLURM array: 0-{total_runs - 1})\n")
        for i, (material, cfg, use_ansatz, lbfgs_only) in enumerate(all_runs):
            name = _resolve_run_name(cfg, use_ansatz, lbfgs_only)
            print(f"  {i:4d}  {material.name:<14s} {name}")
        return

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
                              lbfgs_only=lbfgs_only, backend=_BACKEND)
        print(f"\nExperiment {args.run_id + 1}/{total_runs} complete.")
    else:
        # ── Sequential mode — compute FD reference once per material ──────────
        fd_cache: dict = {}
        run_idx = 1
        for material, cfg, use_ansatz, lbfgs_only in all_runs:
            if run_idx == 1 or all_runs[run_idx - 2][0] != material:
                print(f"\n{'#' * 80}")
                print(f"#  MATERIAL: {material.name}  |  X: [{material.x_min:.2f}, {material.x_max:.2f}]  |  T: [{CONFIG['t_min']}, {CONFIG['t_max']}]")
                print(f"{'#' * 80}")
            if material.name not in fd_cache:
                print(f"  Pre-computing FD reference for {material.name}...")
                fd_cache[material.name] = _compute_fd_reference(material, CONFIG)
            fd_ref = fd_cache.get(material.name)
            run_single_experiment(cfg, use_ansatz, material, device, run_idx, total_runs,
                                  lbfgs_only=lbfgs_only, backend=_BACKEND,
                                  fd_reference=fd_ref)
            run_idx += 1

        print("\n" + "=" * 80)
        print(f" ALL {total_runs} EXPERIMENTS COMPLETED AND LOGGED SUCCESSFULLY!")
        print(f" Output Location: {CONFIG['output_base_dir']}")
        print("=" * 80)

if __name__ == "__main__":
    main()
