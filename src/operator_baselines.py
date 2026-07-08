import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer used by the FNO baseline.

    Input/output layout is (batch, channels, x, t). The first spatial axis uses
    positive and negative Fourier modes; the time axis uses the non-negative
    modes returned by rfft2.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes_x: int, modes_t: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_t = modes_t

        scale = 1.0 / max(1, in_channels * out_channels)
        self.weights_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_t,
                                dtype=torch.cfloat)
        )
        self.weights_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_t,
                                dtype=torch.cfloat)
        )

    @staticmethod
    def _complex_mul2d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, nx, nt = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            nx,
            nt // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        mx = min(self.modes_x, nx)
        mt = min(self.modes_t, nt // 2 + 1)
        out_ft[:, :, :mx, :mt] = self._complex_mul2d(
            x_ft[:, :, :mx, :mt], self.weights_pos[:, :, :mx, :mt]
        )
        out_ft[:, :, -mx:, :mt] = self._complex_mul2d(
            x_ft[:, :, -mx:, :mt], self.weights_neg[:, :, :mx, :mt]
        )
        return torch.fft.irfft2(out_ft, s=(nx, nt), norm="ortho")


class FNO2d(nn.Module):
    """
    Supervised Fourier Neural Operator baseline for full wavefields.

    The model maps gridded coefficient/coordinate channels to u(x, t):
        [E(x), rho(x), g(x), x, t] -> u(x, t)

    This is intentionally separate from the PINN loss stack, because FNO is a
    grid-to-grid neural operator rather than a coordinate MLP differentiated by
    autograd at collocation points.
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 1,
                 width: int = 32, modes_x: int = 12, modes_t: int = 12,
                 n_layers: int = 4, padding: int = 4):
        super().__init__()
        self.padding = padding
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes_x, modes_t)
            for _ in range(n_layers)
        ])
        self.local_layers = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1)
            for _ in range(n_layers)
        ])
        self.project1 = nn.Conv2d(width, width * 2, kernel_size=1)
        self.project2 = nn.Conv2d(width * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        if self.padding > 0:
            x = F.pad(x, (0, self.padding, 0, self.padding))

        for spectral, local in zip(self.spectral_layers, self.local_layers):
            x = F.gelu(spectral(x) + local(x))

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
        x = F.gelu(self.project1(x))
        return self.project2(x)


class ConvBaseline2d(nn.Module):
    """
    Local convolutional baseline with the same input/output tensors as FNO2d.

    This gives a simple non-spectral comparator for the operator experiments.
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 1,
                 width: int = 32, n_layers: int = 6):
        super().__init__()
        layers = [nn.Conv2d(in_channels, width, kernel_size=3, padding=1), nn.GELU()]
        for _ in range(max(0, n_layers - 2)):
            layers.extend([
                nn.Conv2d(width, width, kernel_size=3, padding=1),
                nn.GELU(),
            ])
        layers.append(nn.Conv2d(width, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def relative_l2_percent(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = pred.reshape(pred.shape[0], -1)
    target_f = target.reshape(target.shape[0], -1)
    err = torch.linalg.norm(pred_f - target_f, dim=1)
    denom = torch.linalg.norm(target_f, dim=1).clamp_min(1e-12)
    return 100.0 * err / denom


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
