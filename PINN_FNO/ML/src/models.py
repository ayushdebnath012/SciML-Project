import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

try:
    from kan import KAN
except ImportError as e:
    print(f"  [DEBUG] Models failed to import KAN: {e}")
    KAN = None

# ─────────────────────────────────────────────
# 1. Vanilla and Fourier Feature PINN Models
# ─────────────────────────────────────────────

class VanillaPINN(nn.Module):
    def __init__(self, hidden_layers: int = 7, hidden_units: int = 128):
        super().__init__()
        layers = [nn.Linear(2, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.Tanh()]
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x, t): 
        return self.net(torch.cat([x, t], dim=-1))


class FourierFeaturePINN(nn.Module):
    def __init__(self, hidden_layers: int = 7, hidden_units: int = 128, n_fourier: int = 128, sigma: float = 1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(n_fourier, 2) * sigma)
        layers = [nn.Linear(2 * n_fourier, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.Tanh()]
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x, t):
        proj = torch.cat([x, t], dim=-1) @ self.B.T
        return self.net(torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1))


# ─────────────────────────────────────────────
# 2. PirateNet and Random Weight Fluctuation (RWF) Components
# ─────────────────────────────────────────────

class RWFLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, rwf_mu: float = 1.0, rwf_sigma: float = 0.1):
        super().__init__()
        self.V = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.V)
        self.s = nn.Parameter(torch.normal(mean=rwf_mu, std=rwf_sigma, size=(out_features,)))
        if bias: 
            self.bias = nn.Parameter(torch.zeros(out_features))
        else: 
            self.register_parameter("bias", None)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.s.unsqueeze(1) * self.V, self.bias)


class PirateBlock(nn.Module):
    def __init__(self, units: int, rwf_mu: float = 1.0, rwf_sigma: float = 0.1):
        super().__init__()
        self.W1 = RWFLinear(units, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.W2 = RWFLinear(units, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.W3 = RWFLinear(units, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.alpha = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        f  = torch.tanh(self.W1(x))
        z1 = f * U + (1.0 - f) * V
        g  = torch.tanh(self.W2(z1))
        z2 = g * U + (1.0 - g) * V
        h  = torch.tanh(self.W3(z2))
        return self.alpha * h + (1.0 - self.alpha) * x


class PirateNet(nn.Module):
    def __init__(self, n_blocks: int = 3, units: int = 256, n_fourier: int = 128,
                 sigma: float = 2.0, in_dim: int = 2, rwf_mu: float = 1.0, rwf_sigma: float = 0.1):
        super().__init__()
        self.register_buffer("B", torch.randn(n_fourier, in_dim) * sigma)
        embed_dim = 2 * n_fourier
        self.enc_U = RWFLinear(embed_dim, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.enc_V = RWFLinear(embed_dim, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.proj  = RWFLinear(embed_dim, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.blocks = nn.ModuleList([PirateBlock(units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma) for _ in range(n_blocks)])
        self.out_layer = nn.Linear(units, 1, bias=False)
        nn.init.zeros_(self.out_layer.weight)
 
    def fourier_embed(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        proj = torch.cat([x, t], dim=-1) @ self.B.T
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
 
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        phi = self.fourier_embed(x, t)
        U, V = torch.tanh(self.enc_U(phi)), torch.tanh(self.enc_V(phi))
        h = torch.tanh(self.proj(phi))
        for block in self.blocks: 
            h = block(h, U, V)
        return self.out_layer(h)

    @torch.no_grad()
    def physics_informed_init(self, xt_data: torch.Tensor, y_data: torch.Tensor) -> None:
        self.eval()
        phi = self.fourier_embed(xt_data[:, 0:1], xt_data[:, 1:2])
        U, V = torch.tanh(self.enc_U(phi)), torch.tanh(self.enc_V(phi))
        Phi_eff = torch.tanh(self.proj(phi))
        for block in self.blocks: 
            Phi_eff = block(Phi_eff, U, V)
        W_star = torch.linalg.lstsq(Phi_eff, y_data).solution
        self.out_layer.weight.copy_(W_star.T)
        self.train()


# ─────────────────────────────────────────────
# 3. Kolmogorov-Arnold Network (KAN) Components
# ─────────────────────────────────────────────

class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5, spline_order: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_basis = grid_size + spline_order

        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))

        self.spline_weight = nn.Parameter(torch.empty(out_dim, in_dim, self.n_basis))
        nn.init.xavier_uniform_(self.spline_weight)

        grid = torch.linspace(-1, 1, grid_size + 1)
        step = grid[1] - grid[0]
        pad_left = torch.linspace(grid[0] - spline_order * step, grid[0] - step, spline_order)
        pad_right = torch.linspace(grid[-1] + step, grid[-1] + spline_order * step, spline_order)
        full_grid = torch.cat([pad_left, grid, pad_right])
        self.register_buffer("grid", full_grid)

    def b_spline(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            d1 = grid[k:-1] - grid[:-k-1]
            d2 = grid[k+1:] - grid[1:-k]
            term1 = (x - grid[:-k-1]) / d1 * bases[:, :, :-1]
            term2 = (grid[k+1:] - x) / d2 * bases[:, :, 1:]
            bases = term1 + term2

        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = F.linear(F.silu(x), self.base_weight)
        basis = self.b_spline(x)
        spline_output = torch.einsum("bik,oik->bo", basis, self.spline_weight)
        return base_output + spline_output


class PIKAN(nn.Module):
    def __init__(self, hidden_layers=3, hidden_units=64, grid_size=5, spline_order=4, n_fourier=128, sigma=1.0):
        super().__init__()

        # Fourier embedding
        self.register_buffer("B", torch.randn(n_fourier, 2) * sigma)
        embed_dim = 2 * n_fourier

        # layer0
        self.layer0 = KANLayer(embed_dim, hidden_units, grid_size=grid_size, spline_order=spline_order)
        self.norm0 = nn.LayerNorm(hidden_units)

        # mid layers + norms
        self.kan_mid = nn.ModuleList([
            KANLayer(hidden_units, hidden_units, grid_size=grid_size, spline_order=spline_order)
            for _ in range(hidden_layers - 1)
        ])
        self.norms_mid = nn.ModuleList([
            nn.LayerNorm(hidden_units)
            for _ in range(hidden_layers - 1)
        ])

        self.out_layer = nn.Linear(hidden_units, 1, bias=False)
        nn.init.xavier_uniform_(self.out_layer.weight)

    def fourier_embed(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        inputs = torch.cat([x, t], dim=-1)
        proj = 2 * torch.pi * (inputs @ self.B.T)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        phi = self.fourier_embed(x, t)
        h = torch.tanh(self.norm0(self.layer0(phi)))

        for i, layer in enumerate(self.kan_mid):
            h = torch.tanh(self.norms_mid[i](layer(h)))

        return self.out_layer(h)

    def get_l1_regularization(self) -> torch.Tensor:
        reg = torch.norm(self.layer0.spline_weight, 1)
        for layer in self.kan_mid:
            reg = reg + torch.norm(layer.spline_weight, 1)
        return reg


class FourierKAN(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden, hidden_width, G, k, n_fourier=64, sigma=10.0):
        super().__init__()
        if KAN is None:
            raise ImportError("The 'pykan' library is required to use FourierKAN. Please install it.")
        # Random projection matrix (B) for Fourier Features
        self.register_buffer("B", torch.randn(n_fourier, n_inputs) * sigma)
        
        # KAN input is 2 * n_fourier (cos and sin components)
        width = [2 * n_fourier] + [hidden_width] * n_hidden + [n_outputs]
        self.kan = KAN(width=width, grid=G, k=k, symbolic_enabled=False)
        
    def forward(self, xt):
        proj = xt @ self.B.T
        rff = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return self.kan(rff)
    
    def speed(self):
        if hasattr(self.kan, 'speed'):
            self.kan.speed()

# ─────────────────────────────────────────────
# 4. WaveKAN
# ─────────────────────────────────────────────
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        self.base_activation = nn.SiLU()

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x
        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded
        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2)-1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi**0.25)) * term1 * term2
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'dog':
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet_weighted = dog * self.wavelet_weights.unsqueeze(0).expand_as(dog)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'meyer':
            v = torch.abs(x_scaled)
            pi = math.pi
            def nu(t):
                return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)
            def meyer_aux(v):
                return torch.where(v <= 1/2, torch.ones_like(v),
                       torch.where(v >= 1,   torch.zeros_like(v),
                       torch.cos(pi / 2 * nu(2 * v - 1))))
            wavelet = torch.sin(pi * v) * meyer_aux(v)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'shannon':
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)
            window = torch.hamming_window(x_scaled.size(-1), periodic=False,
                                          dtype=x_scaled.dtype, device=x_scaled.device)
            wavelet = sinc * window
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        else:
            raise ValueError("Unsupported wavelet type")
        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        return wavelet_output


class WavKAN(nn.Module):
    def __init__(self, layers_hidden, wavelet_type='mexican_hat'):
        super(WavKAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(KANLinear(in_features, out_features, wavelet_type))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ─────────────────────────────────────────────
# 4. Helpers and Wrappers
# ─────────────────────────────────────────────

class KANWrapper(nn.Module):
    def __init__(self, k_model):
        super().__init__()
        self.k_model = k_model
        
    def forward(self, x, t):
        return self.k_model(torch.cat([x, t], dim=-1))


def build_kan(n_inputs, n_outputs, n_hidden, hidden_width, G, k):
    assert n_hidden >= 1
    return FourierKAN(n_inputs, n_outputs, n_hidden, hidden_width, G, k, n_fourier=64, sigma=10.0)


def num_parameters_kan(n_inputs, n_outputs, n_hidden, hidden_width, G, k):
    return (n_inputs * hidden_width + (n_hidden-1) * hidden_width**2 + n_outputs * hidden_width) * (2 + G + k)
