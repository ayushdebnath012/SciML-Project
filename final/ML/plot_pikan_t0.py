import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from kan import KAN

# Import necessary parts from the local codebase
from training_code import DEVICE, HomogeneousModel, apply_ansatz, gaussian_ic
# Define KANWrapper locally since it's nested in benchmark_v2 functions
class KANWrapper(nn.Module):
    def __init__(self, k_model):
        super().__init__()
        self.k_model = k_model
    def forward(self, x, t):
        return self.k_model(torch.cat([x, t], dim=-1))

def plot_pikan_at_t0():
    # 0. Configuration
    TARGET_TIME = 0.05  # <--- CHANGE THIS TO PLOT DIFFERENT TIMES
    PREFIX = "exp1_homogeneous"
    ARCH = "pikan"
    model_path = os.path.join("Models", f"{PREFIX}_{ARCH}_model.pkl")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    # 2. Load weights
    print(f"Loading {model_path}...")
    sd = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    # 3. Infer architecture and initialize
    try:
        grid_key = next(k for k in sd.keys() if 'grid' in k)
        coef_key = next(k for k in sd.keys() if 'coef' in k)
        g_len = sd[grid_key].shape[-1]
        c_len = sd[coef_key].shape[-1]
        k_val = g_len - c_len - 1
        grid_val = c_len - k_val
        print(f"Inferred grid={grid_val}, k={k_val}")
    except Exception as e:
        print(f"Inference failed: {e}. Using defaults.")
        grid_val, k_val = 5, 3

    kan_model = KAN(width=[2, 5, 5, 5, 1], grid=grid_val, k=k_val).to(DEVICE)
    if isinstance(sd, dict):
        kan_model.load_state_dict(sd)
    elif hasattr(sd, 'state_dict'):
        kan_model.load_state_dict(sd.state_dict())
    
    model = KANWrapper(kan_model).to(DEVICE)
    model.eval()

    # 4. Generate Data
    material = HomogeneousModel()
    nx = 500
    x_np = np.linspace(material.x_min, material.x_max, nx)
    x_t = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    t_t = torch.full_like(x_t, float(TARGET_TIME))
    
    sigma_g_nd = material.nondim_sigma_g(0.1) # Default sigma in benchmark

    with torch.no_grad():
        # NN Output
        nn_out = model(x_t, t_t)
        # Prediction with Ansatz
        u_pred = apply_ansatz(nn_out, x_t, t_t, material, sigma_g_nd).cpu().numpy().flatten()
        # Reference IC
        u_ref = gaussian_ic(x_t, sigma_g_nd).cpu().numpy().flatten()

    # 5. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_np, u_ref, 'k-', lw=3, label='Reference IC (Gaussian Derivative)', alpha=0.7)
    plt.plot(x_np, u_pred, 'r--', lw=2, label=f'PIKAN Prediction (t={TARGET_TIME})')
    
    plt.title(f"PIKAN Initial Condition Match (t={TARGET_TIME})\n{PREFIX} | Architecture [2,5,5,5,1]", fontsize=14)
    plt.xlabel("x (non-dimensional)")
    plt.ylabel("u(x, 0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = "pikan_t0_plot.png"
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_pikan_at_t0()
