import math
import numpy as np
import torch

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

    def E(self, x: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def rho(self, x: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def Vp(self, x: torch.Tensor) -> torch.Tensor: return torch.sqrt(self.E(x) / self.rho(x))

    # ── JAX-compatible variants (accept scalars or jnp arrays) ───────────────
    def E_jax(self, x): raise NotImplementedError
    def rho_jax(self, x): raise NotImplementedError
    def Vp_jax(self, x):
        import jax.numpy as jnp
        return jnp.sqrt(self.E_jax(x) / self.rho_jax(x))

    def to_physical_x(self, x_nd): return x_nd * self.L_ref
    def to_physical_t(self, t_nd): return t_nd * self.T_ref
    def to_nondim_t(self, t_phys): return t_phys / self.T_ref
    def nondim_sigma_g(self, sigma_g_phys: float) -> float: return sigma_g_phys / self.L_ref

class HomogeneousModel(MaterialModel):
    def __init__(self):
        super().__init__("Homogeneous", (-1.0, 1.0), E_ref=80.0, rho_ref=100.0)
    def E(self, x): return torch.full_like(x, 1.0)
    def rho(self, x): return torch.full_like(x, 1.0)
    def E_jax(self, x): return 1.0
    def rho_jax(self, x): return 1.0

class TwoLayerModel(MaterialModel):
    def __init__(self):
        super().__init__("TwoLayer", (-1.0, 1.0), E_ref=80.0, rho_ref=100.0)
        self._E1_nd = 80.0  / self.E_ref
        self._E2_nd = 120.0 / self.E_ref
    def E(self, x):
        w = 0.02 / self.L_ref
        alpha = 0.5 * (1.0 + torch.tanh(x / w))
        return self._E1_nd * (1.0 - alpha) + self._E2_nd * alpha
    def rho(self, x): return torch.full_like(x, 1.0)
    def E_jax(self, x):
        import jax.numpy as jnp
        w = 0.02 / self.L_ref
        alpha = 0.5 * (1.0 + jnp.tanh(x / w))
        return self._E1_nd * (1.0 - alpha) + self._E2_nd * alpha
    def rho_jax(self, x): return 1.0

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
    def rho(self, x): return torch.full_like(x, 1.0)
    def E_jax(self, x):
        import jax.numpy as jnp
        x_min, x_max = self.x_min, self.x_max
        layer_width  = (x_max - x_min) / self.n_layers
        w = 0.02 / self.L_ref
        E_val = float(self.E_vals_nd[0])
        for k in range(self.n_layers - 1):
            boundary = x_min + (k + 1) * layer_width
            alpha    = 0.5 * (1.0 + jnp.tanh((x - boundary) / w))
            E_val    = E_val * (1.0 - alpha) + float(self.E_vals_nd[k + 1]) * alpha
        return E_val
    def rho_jax(self, x): return 1.0