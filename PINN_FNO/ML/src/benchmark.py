import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Clean modular package imports
from src.physics import (
    HomogeneousModel, TwoLayerModel, MultiLayerModel,
    fd_reference
)
from src.models import (
    VanillaPINN, FourierFeaturePINN, PirateNet
)
from src.ansatz_losses import (
    apply_ansatz, compute_pde_residual
)
from src.loader import load_models
from src.evaluation import (
    evaluate, plot_traces, plot_final_comparison, plot_wavefield,
    plot_metrics, plot_residual_map, animate_results, animate_error,
    ARCH_LABELS
)

# ─────────────────────────────────────────────
# 0. Global Configuration Setup
# ─────────────────────────────────────────────

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED          = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MODEL_DIR     = "Models"
BASE_SAVE_DIR  = "results"
SIGMA_G_PHYS   = 0.1
N_FRAMES       = 120
FPS            = 30

# Which architectures to benchmark
RUN_ARCHS = [
    #"vanilla",
    #"fourier", 
    #"pirate",
    #"pikan",
    "WaveKAN_3-10",
    "WaveKAN_5-10",
    "WaveKAN_7-10",
]

EXPERIMENTS = {
    "exp1_homogeneous": HomogeneousModel(),
    #"exp2_twolayer":    TwoLayerModel(),
    #"exp3_multilayer":  MultiLayerModel(),
}

T_MAX_DICT = {
    "exp1_homogeneous": 1.0,
    "exp2_twolayer":    1.0,
    "exp3_multilayer":  1.0,
}

# ─────────────────────────────────────────────
# Main Benchmarking Loop
# ─────────────────────────────────────────────

def run_benchmark():
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    
    for prefix, material in EXPERIMENTS.items():
        print(f"\n{'='*60}\n  Experiment: {material.name} ({prefix})\n{'='*60}")
        
        # 1. Load models using modular loader
        models = load_models(prefix, model_dir=MODEL_DIR, run_archs=RUN_ARCHS, device=DEVICE)
        if not models:
            print(f"  ! No models found for {prefix}, skipping.")
            continue
            
        # 2. Configure scales
        T_phys      = T_MAX_DICT.get(prefix, 1.0)
        T_nd        = material.to_nondim_t(T_phys)
        sigma_g_nd  = material.nondim_sigma_g(SIGMA_G_PHYS)
        
        # 3. Reference solution
        print(f"  Generating reference (T_nd={T_nd:.3f}, sigma_nd={sigma_g_nd:.3f})...")
        x_fd, t_fd_nd, u_fd = fd_reference(material, nx=512, nt=2000, T=T_nd, sigma_g=sigma_g_nd)
        
        # 4. Evaluation
        eval_phys  = list(np.linspace(0.05, T_phys, 20))
        eval_nd    = [material.to_nondim_t(tp) for tp in eval_phys]
        
        results_summary = {}
        print(f"\n  {'Architecture':<25} {'L2 Error (%)':>15} {'Parameters':>18}")
        print(f"  {'-'*62}")
        
        for arch, model in models.items():
            m = evaluate(model, material, x_fd, t_fd_nd, u_fd, sigma_g=sigma_g_nd, eval_times=eval_nd)
            results_summary[arch] = m
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  {ARCH_LABELS.get(arch, arch):<25} {m['mean_l2']:>14.2f}% {num_params:>18,}")
            
        # 5. Save and visualize results
        save_dir = os.path.join(BASE_SAVE_DIR, prefix.split('_')[0])
        os.makedirs(save_dir, exist_ok=True)
        
        # Call modular plotting and animation utilities
        plot_traces(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir)
        plot_final_comparison(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir)
        plot_wavefield(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir)
        plot_metrics(results_summary, material, prefix, save_dir)
        plot_residual_map(models, material, sigma_g_nd, prefix, save_dir)
        
        animate_results(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir, n_frames=N_FRAMES, fps=FPS)
        animate_error(models, material, x_fd, t_fd_nd, u_fd, sigma_g_nd, prefix, save_dir, n_frames=N_FRAMES, fps=FPS)

    print(f"\n* All experiments complete. Results in {BASE_SAVE_DIR}/")

if __name__ == "__main__":
    run_benchmark()
