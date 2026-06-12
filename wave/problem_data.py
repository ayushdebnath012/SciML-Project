import numpy as np
import torch

# Relative package imports
from src import physics as utils

INTERVAL_T = (0.0, 1.0)

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
                 sigma_g: float = 0.1,
                 use_ansatz: bool = True) -> torch.Tensor:
    """
    û(x,t) = g(x)·exp(-½(15t)²) + tanh²(25t)·NN(x,t)    [Eq. 11]

    If use_ansatz is True:
        At t=0 : û = g(x),  tanh=0  ✓
        t > 0  : exponential decays, NN term grows.
    If use_ansatz is False:
        Directly returns nn_out (enables soft initial condition penalty).
    """
    if not use_ansatz:
        return nn_out
        
    g      = gaussian_ic(x, sigma_g)
    decay  = torch.exp(-0.5 * (15.0 * t) ** 2)
    growth = torch.tanh(25.0 * t) ** 2
    return g * decay + growth * nn_out


# Finite Difference reference solution generator
def solve_reference_fd(f, g, material, Nx=200, CFL=0.9, t_max=None):
    """
    Conservative-form FD reference on the material's own nondimensional domain
    [material.x_min, material.x_max] (NOT the legacy INTERVAL_X, which only
    covered [0, 1] while all materials nondimensionalize to [-1, 1]).
    """
    import fd_solver
    t_hi = INTERVAL_T[1] if t_max is None else t_max

    def E_fn(x_arr):
        return material.E(torch.tensor(x_arr, dtype=torch.float64).reshape(-1, 1)).flatten().numpy()

    def rho_fn(x_arr):
        return material.rho(torch.tensor(x_arr, dtype=torch.float64).reshape(-1, 1)).flatten().numpy()

    return fd_solver.solve_wave_1d(
        f, g, E_fn, rho_fn,
        x_limits=(material.x_min, material.x_max),
        t_limits=(INTERVAL_T[0], t_hi), Nx=Nx, CFL=CFL
    )

NUM_INPUTS = 2
NUM_OUTPUTS = 1