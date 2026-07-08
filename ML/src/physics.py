import math
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

# ─────────────────────────────────────────────
# 1. Physical Material Models
# ─────────────────────────────────────────────

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

    def E(self, x: torch.Tensor) -> torch.Tensor: 
        raise NotImplementedError
        
    def rho(self, x: torch.Tensor) -> torch.Tensor: 
        raise NotImplementedError
        
    def Vp(self, x: torch.Tensor) -> torch.Tensor: 
        return torch.sqrt(self.E(x) / self.rho(x))
    
    def to_physical_x(self, x_nd): 
        return x_nd * self.L_ref
        
    def to_physical_t(self, t_nd): 
        return t_nd * self.T_ref
        
    def to_nondim_t(self, t_phys): 
        return t_phys / self.T_ref
        
    def nondim_sigma_g(self, sigma_g_phys: float) -> float: 
        return sigma_g_phys / self.L_ref


class HomogeneousModel(MaterialModel):
    def __init__(self):
        super().__init__("Homogeneous", (-1.0, 1.0), E_ref=80.0, rho_ref=100.0)
        
    def E(self, x): 
        return torch.full_like(x, 1.0)
        
    def rho(self, x): 
        return torch.full_like(x, 1.0)


class TwoLayerModel(MaterialModel):
    def __init__(self):
        super().__init__("TwoLayer", (-1.0, 1.0), E_ref=80.0, rho_ref=100.0)
        self._E1_nd = 80.0  / self.E_ref
        self._E2_nd = 120.0 / self.E_ref
        
    def E(self, x):
        w = 0.02 / self.L_ref
        alpha = 0.5 * (1.0 + torch.tanh(x / w))
        return self._E1_nd * (1.0 - alpha) + self._E2_nd * alpha
        
    def rho(self, x): 
        return torch.full_like(x, 1.0)


class MultiLayerModel(MaterialModel):
    def __init__(self):
        super().__init__("MultiLayer", (-1.5, 1.5), E_ref=60.0, rho_ref=100.0)
        self.n_layers  = 6
        self.E_vals_nd = np.linspace(60.0, 150.0, self.n_layers) / self.E_ref
        
    def E(self, x):
        x_min, x_max = self.x_min, self.x_max
        layer_width  = (x_max - x_min) / self.n_layers
        E_val = torch.full_like(x, self.E_vals_nd[0])
        w = 0.02 / self.L_ref
        for k in range(self.n_layers - 1):
            boundary = x_min + (k + 1) * layer_width
            alpha    = 0.5 * (1.0 + torch.tanh((x - boundary) / w))
            E_val    = E_val * (1.0 - alpha) + self.E_vals_nd[k + 1] * alpha
        return E_val
        
    def rho(self, x): 
        return torch.full_like(x, 1.0)


# ─────────────────────────────────────────────
# 2. Finite Difference Reference Solvers
# ─────────────────────────────────────────────

def fd_reference(material: MaterialModel, nx: int = 512, nt: int = 2000,
                 T: float = 1.0, sigma_g: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ CPU-based Finite Difference Reference Solver to evaluate our PINN against. """
    x_min, x_max = material.x_min, material.x_max
    x = np.linspace(x_min, x_max, nx)
    dx = x[1] - x[0]
    dt = T / nt
    t  = np.linspace(0, T, nt)

    x_t = torch.tensor(x, dtype=torch.float32)
    E_np  = material.E(x_t).numpy()
    rho_np = material.rho(x_t).numpy()

    x0   = 0.0
    f    = np.exp(-0.5 * ((x - x0) / sigma_g) ** 2)
    dfdr = -(x - x0) / sigma_g**2 * f
    g    = dfdr / (np.abs(dfdr).max() + 1e-12)

    u_prev, u_curr = g.copy(), g.copy()
    u_all  = np.zeros((nt, nx))
    u_all[0], u_all[1] = u_prev, u_curr

    E_half = 0.5 * (E_np[:-1] + E_np[1:])

    for n in range(1, nt - 1):
        u_new = np.zeros(nx)
        flux       = E_half * (u_curr[1:] - u_curr[:-1]) / dx
        d_flux     = (flux[1:] - flux[:-1]) / dx
        u_new[1:-1] = (2 * u_curr[1:-1] - u_prev[1:-1] + (dt**2 / rho_np[1:-1]) * d_flux)

        Vp_left  = math.sqrt(E_np[0]  / rho_np[0])
        Vp_right = math.sqrt(E_np[-1] / rho_np[-1])
        r_left   = Vp_left  * dt / dx
        r_right  = Vp_right * dt / dx
        
        u_new[0]  = u_curr[1]  + (r_left  - 1) / (r_left  + 1) * (u_new[1]  - u_curr[0])
        u_new[-1] = u_curr[-2] + (r_right - 1) / (r_right + 1) * (u_new[-2] - u_curr[-1])

        u_prev, u_curr = u_curr, u_new
        u_all[n + 1] = u_curr

    return x, t, u_all


def solve_wave_fd(material, sigma_g=0.1, x0=0.0, nx=200, nt=600, t_max=1.0):
    """Finite Difference solver for the wave equation with ABCs (used in PIKAN)."""
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


# ─────────────────────────────────────────────
# 3. Reference Solutions and Gradients (Cache and Interpolation)
# ─────────────────────────────────────────────

_SOL_CACHE = {}

def u_sol(x_query, t_query, material, sigma_g=0.1, x0=0.0):
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


def u_grad_sol(x_query, t_query, material, sigma_g=0.1, x0=0.0):
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
