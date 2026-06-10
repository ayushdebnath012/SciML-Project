import numpy as np
import torch

# Relative package imports
from src import physics as utils

INTERVAL_X = (0.0, 1.0)
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
def solve_reference_fd(f, g, c_speed, Nx=200, CFL=0.9):
    import fd_solver
    return fd_solver.solve_wave_1d(
        f, g, c_speed, x_limits=INTERVAL_X, t_limits=INTERVAL_T, Nx=Nx, CFL=CFL
    )

NUM_INPUTS = 2
NUM_OUTPUTS = 1

# Losses function imported from dedicated losses package
from src.losses.wave_loss import losses