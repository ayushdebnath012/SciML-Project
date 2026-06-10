import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN

class VanillaPINN(nn.Module):
    """
    Standard MLP PINN.
    Input : (x, t)  →  in_dim features
    Output: û(x, t) →  out_dim values
    """

    def __init__(self, in_dim: int = 2, out_dim: int = 1,
                 hidden_layers: int = 7, hidden_units: int = 128):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.Tanh()]
        layers.append(nn.Linear(hidden_units, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t=None):
        if t is None:
            return self.net(x)
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)


class FourierFeaturePINN(nn.Module):
    """
    Random Fourier Feature embedding → MLP.
    B is treated as a trainable parameter (as in the paper).
    """

    def __init__(self,
                 hidden_layers: int = 7,
                 hidden_units: int  = 128,
                 n_fourier: int     = 128,
                 sigma: float       = 1.0):
        super().__init__()
        # Trainable Fourier frequency matrix  B ∈ R^{n_fourier × 2}
        B_init = torch.randn(n_fourier, 2) * sigma
        self.B = nn.Parameter(B_init)

        in_features = 2 * n_fourier   # cos + sin
        layers = [nn.Linear(in_features, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.Tanh()]
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def fourier_embed(self, x, t):
        inp = torch.cat([x, t], dim=-1)        # (N, 2)
        proj = inp @ self.B.T                  # (N, n_fourier)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (N, 2*n_fourier)

    def forward(self, x, t=None):
        if t is None:
            proj = x @ self.B.T
            return self.net(torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1))
        phi = self.fourier_embed(x, t)
        return self.net(phi)

    @torch.no_grad()
    def physics_informed_init(self,
                              xt_data: torch.Tensor,
                              y_data:  torch.Tensor) -> None:
        """
        Fit the final linear layer to the IC via least-squares (same scheme as
        PirateNet.physics_informed_init).  Passes xt_data through all layers
        except the last, then solves min_W ||Phi W^T - y||.
        """
        self.eval()
        x_pts, t_pts = xt_data[:, 0:1], xt_data[:, 1:2]
        phi = self.fourier_embed(x_pts, t_pts)
        h = phi
        for layer in self.net[:-1]:
            h = layer(h)
        W_star = torch.linalg.lstsq(h, y_data).solution  # (units, 1)
        self.net[-1].weight.copy_(W_star.T)
        if self.net[-1].bias is not None:
            self.net[-1].bias.zero_()
        self.train()


class RWFLinear(nn.Module):
    """
    Linear layer with Random Weight Factorization (paper Sec 5, ref [75]).
    W_effective = diag(s) @ V.
    """
 
    def __init__(self,
                 in_features:  int,
                 out_features: int,
                 bias:         bool  = True,
                 rwf_mu:       float = 1.0,
                 rwf_sigma:    float = 0.1):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
 
        # Base weight V — Glorot (Xavier) uniform init.
        # Paper Sec 3 / App A: "weight matrices are initialized by the Glorot
        # initialization scheme, W^(l) ~ D(0, 2/(d_l + d_{l-1}))"
        # xavier_uniform_ samples from Uniform(-a, a) with
        # a = sqrt(6 / (fan_in + fan_out)), which satisfies Var = 2/(fan_in+fan_out).
        self.V = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.V)
 
        # Per-row scale s ~ N(mu, sigma^2)  (paper Tables 4-8: mu=1.0, sigma=0.1)
        self.s = nn.Parameter(
            torch.normal(mean=rwf_mu, std=rwf_sigma, size=(out_features,))
        )
 
        if bias:
            # "biases are initialized to zero" (paper Sec 4)
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_eff = self.s.unsqueeze(1) * self.V   # diag(s) @ V, broadcast
        return nn.functional.linear(x, W_eff, self.bias)

class PirateBlock(nn.Module):
    """
    One PirateNet residual block (paper Eqs 4.1-4.6):
 
        f^(l)   = sigma( W1 x^(l) + b1 )                          (4.1)
        z1^(l)  = f^(l) * U  +  (1 - f^(l)) * V                   (4.2)
        g^(l)   = sigma( W2 z1^(l) + b2 )                         (4.3)
        z2^(l)  = g^(l) * U  +  (1 - g^(l)) * V                   (4.4)
        h^(l)   = sigma( W3 z2^(l) + b3 )                         (4.5)
        x^(l+1) = alpha^(l) * h^(l) + (1 - alpha^(l)) * x^(l)   (4.6)
 
    alpha^(l) is a trainable SCALAR initialised to ZERO.
    Paper Sec 4: "Throughout all experiments we initialize alpha^(l) to zero."
    => at init, each block is the identity map.
    => full network = W^(L+1) Phi(x)  (linear combo of Fourier basis, Eq 4.8).
 
    Activation: Tanh (paper: "Tanh activation function employed by default").
    Dense layers use RWFLinear (paper Sec 5: "all weight matrices enhanced
    using random weight factorization").
    """
 
    def __init__(self, units: int, rwf_mu: float = 1.0, rwf_sigma: float = 0.1):
        super().__init__()
        self.W1 = RWFLinear(units, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.W2 = RWFLinear(units, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.W3 = RWFLinear(units, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        # Critical: alpha initialised to 0, not 1 (paper Sec 4, ablation Figs 10/13)
        self.alpha = nn.Parameter(torch.zeros(1))
 
    def forward(self, x: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        f  = torch.tanh(self.W1(x))          # Eq (4.1)
        z1 = f * U + (1.0 - f) * V           # Eq (4.2)
        g  = torch.tanh(self.W2(z1))         # Eq (4.3)
        z2 = g * U + (1.0 - g) * V           # Eq (4.4)
        h  = torch.tanh(self.W3(z2))         # Eq (4.5)
        return self.alpha * h + (1.0 - self.alpha) * x  # Eq (4.6)


class PirateNet(nn.Module):
    """
    Full PirateNet architecture (paper Sec 4).
 
    Pipeline:
      1. Random Fourier embedding  Phi(x) = [cos(Bx), sin(Bx)], B~N(0,s^2)
      2. Two gating encoders  U = sigma(W_U Phi + b_U),  V = sigma(W_V Phi + b_V)
      3. Input projection  x^(1) = sigma(W_proj Phi + b_proj)
      4. L residual blocks  (Eqs 4.1-4.6)
      5. Final linear  u_theta = W^(L+1) x^(L)                     (Eq 4.7)
      6. Physics-informed init of final layer (Eq 4.9)
 
    At init (all alpha=0):  u_theta(x) = W^(L+1) Phi(x)            (Eq 4.8)
    -- a linear combination of Fourier basis functions.
 
    Paper defaults (Table 4, Allen-Cahn):
        n_blocks=3, units=256, n_fourier=128, sigma=2.0,
        rwf_mu=1.0, rwf_sigma=0.1
    """
 
    def __init__(self,
                 n_blocks:  int   = 3,
                 units:     int   = 256,
                 n_fourier: int   = 128,   # m; embedding dim = 2m
                 sigma:     float = 2.0,   # Fourier feature scale s
                 in_dim:    int   = 2,     # (x, t)
                 rwf_mu:    float = 1.0,
                 rwf_sigma: float = 0.1):
        super().__init__()
        self.units = units
 
        # Fourier matrix B in R^{m x d}, entries ~ N(0, s^2), FIXED (buffer).
        # Paper: "each entry in B is i.i.d. sampled from N(0, s^2)"
        self.register_buffer("B", torch.randn(n_fourier, in_dim) * sigma)
        embed_dim = 2 * n_fourier
 
        # Gating encoders U and V — dense layers with RWF (paper Sec 4 + Sec 5)
        self.enc_U = RWFLinear(embed_dim, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
        self.enc_V = RWFLinear(embed_dim, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
 
        # Input projection Phi -> units (produces x^(1))
        self.proj  = RWFLinear(embed_dim, units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
 
        # L residual blocks
        self.blocks = nn.ModuleList(
            [PirateBlock(units, rwf_mu=rwf_mu, rwf_sigma=rwf_sigma)
             for _ in range(n_blocks)]
        )
 
        # Final linear layer  u_theta = W^(L+1) x^(L)  (Eq 4.7)
        # No bias (paper Eq 4.7 has no bias term).
        # Initialised to zeros — before PI-init, output is 0 everywhere.
        self.out_layer = nn.Linear(units, 1, bias=False)
        nn.init.zeros_(self.out_layer.weight)
 
    def fourier_embed(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Phi(x,t) = [cos(B[x;t]^T), sin(B[x;t]^T)]  (paper Sec 4)."""
        inp  = torch.cat([x, t], dim=-1)
        proj = inp @ self.B.T
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
 
    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        if t is None:
            proj = x @ self.B.T
            phi = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        else:
            phi = self.fourier_embed(x, t)
        U   = torch.tanh(self.enc_U(phi))
        V   = torch.tanh(self.enc_V(phi))
        h   = torch.tanh(self.proj(phi))    # x^(1)
        for block in self.blocks:
            h = block(h, U, V)
        return self.out_layer(h)            # u_theta  (N, 1)
 
    @torch.no_grad()
    def physics_informed_init(self,
                              xt_data: torch.Tensor,
                              y_data:  torch.Tensor) -> None:
        """
        Physics-informed initialisation of the output layer (Eq 4.9):
            min_W  || W Phi_eff^T - Y ||_2^2
 
        where Phi_eff = x^(L) evaluated with current weights.
        At init (all alpha=0), each block is identity, so Phi_eff = proj(Phi(x)).
 
        Paper Sec 5: "we initialize the weights of the last layer to fit
        u_theta(t,x) to u_0(x) for ALL t."
 
        Args:
            xt_data: (N, 2) tensor of [x, t] pairs spanning the time window
            y_data:  (N, 1) tensor of u_0(x) values  (same for all t)
        """
        self.eval()
        x_pts   = xt_data[:, 0:1]
        t_pts   = xt_data[:, 1:2]
        phi     = self.fourier_embed(x_pts, t_pts)
        U       = torch.tanh(self.enc_U(phi))
        V       = torch.tanh(self.enc_V(phi))
        Phi_eff = torch.tanh(self.proj(phi))
        for block in self.blocks:
            Phi_eff = block(Phi_eff, U, V)   # identity at init (alpha=0)
 
        # lstsq: min || Phi_eff @ W^T - y ||  (paper uses jax.numpy.linalg.lstsq)
        W_star = torch.linalg.lstsq(Phi_eff, y_data).solution  # (units, 1)
        self.out_layer.weight.copy_(W_star.T)                   # (1, units)
        self.train()


class WavKANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat', use_base=True):
        super(WavKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        self.use_base = use_base

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        # Linear weights for combining outputs
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features)) 
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # Base activation function #not used for this experiment
        self.base_activation = nn.SiLU()

        # Batch normalization REMOVED

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        # Implementation of different wavelet types
        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2)-1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi**0.25)) * term1 * term2
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
            
        elif self.wavelet_type == 'dog':
            # Implementing Derivative of Gaussian Wavelet 
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'meyer':
            # Implement Meyer Wavelet here
            # Constants for the Meyer wavelet transition boundaries
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1/2,torch.ones_like(v),torch.where(v >= 1,torch.zeros_like(v),torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)
            # Meyer wavelet calculation using the auxiliary function
            wavelet = torch.sin(pi * v) * meyer_aux(v)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'shannon':
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)
            # Gaussian envelope tapers the infinite-support sinc to a compactly-
            # supported wavelet, acting pointwise on each scalar x_scaled value.
            window = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = sinc * window
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        else:
            raise ValueError("Unsupported wavelet type")

        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        if self.use_base:
            wavelet_output = wavelet_output + F.linear(x, self.weight1)
        return wavelet_output

class WavKAN(nn.Module):
    def __init__(self, layers_hidden, wavelet_type='mexican_hat', use_base=True):
        super(WavKAN, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(WavKANLinear(in_features, out_features, wavelet_type, use_base))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def build_kan(n_inputs, n_outputs, n_hidden, hidden_width, G, k):
    assert n_hidden >= 1
    width = [n_inputs] + [hidden_width] * n_hidden + [n_outputs]
    kan = KAN(width=width, grid=G, k=k, symbolic_enabled=False)
    return kan


def num_parameters_mlp(n_inputs, n_outputs, n_hidden, hidden_width):
    return (n_inputs+1) * hidden_width + (hidden_width+1) * ((n_hidden-1) * hidden_width + n_outputs)


def num_parameters_kan(n_inputs, n_outputs, n_hidden, hidden_width, G, k):
    return (n_inputs * hidden_width + (n_hidden-1) * hidden_width**2 + n_outputs * hidden_width) * (2 + G + k)


def build_mlp(n_inputs, n_outputs, n_hidden, hidden_width):
    return VanillaPINN(in_dim=n_inputs, out_dim=n_outputs, hidden_layers=n_hidden, hidden_units=hidden_width)
