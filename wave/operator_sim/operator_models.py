"""Model + data helpers extracted verbatim from the comparison notebook."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_t):
        super().__init__()
        self.modes_x = modes_x
        self.modes_t = modes_t
        scale = 1.0 / max(1, in_channels * out_channels)
        shape = (in_channels, out_channels, modes_x, modes_t)
        self.weight_pos = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_neg = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def complex_mul(x, weight):
        return torch.einsum("bixy,ioxy->boxy", x, weight)

    def forward(self, x):
        batch, _, nx_local, nt_local = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            batch, self.weight_pos.shape[1], nx_local, nt_local // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        mx = min(self.modes_x, nx_local)
        mt = min(self.modes_t, nt_local // 2 + 1)
        out_ft[:, :, :mx, :mt] = self.complex_mul(
            x_ft[:, :, :mx, :mt], self.weight_pos[:, :, :mx, :mt])
        out_ft[:, :, -mx:, :mt] = self.complex_mul(
            x_ft[:, :, -mx:, :mt], self.weight_neg[:, :, :mx, :mt])
        return torch.fft.irfft2(out_ft, s=(nx_local, nt_local), norm="ortho")


class SimpleFNO2d(nn.Module):
    def __init__(self, width=8, modes_x=6, modes_t=6, layers=2, in_channels=5):
        # in_channels: 5 for the unforced arm [E,rho,g,x,t], 6 for the forced
        # arm [E,rho,s(x),w(t),x,t].
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes_x, modes_t) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.project = nn.Sequential(
            nn.Conv2d(width, 2 * width, 1), nn.GELU(), nn.Conv2d(2 * width, 1, 1))

    def forward(self, x):
        z = self.lift(x)
        for spectral, local in zip(self.spectral, self.local):
            z = F.gelu(spectral(z) + local(z))
        return self.project(z)


def make_mlp(sizes, activation=nn.Tanh):
    layers = []
    for in_size, out_size in zip(sizes[:-2], sizes[1:-1]):
        layers += [nn.Linear(in_size, out_size), activation()]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


class SimpleDeepONet(nn.Module):
    def __init__(self, nx, coordinates, latent=32, hidden=64, n_time=0):
        # n_time > 0 appends the temporal source signature w(t) (channel 3)
        # to the branch input, which the forced arm needs: the branch would
        # otherwise see only x-profiles and be blind to the wavelet.
        super().__init__()
        self.n_time = int(n_time)
        self.branch = make_mlp([3 * nx + self.n_time, hidden, hidden, latent])
        self.trunk = make_mlp([2, hidden, hidden, latent])
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("coordinates", coordinates)
        self.nx = nx
        self.nt = coordinates.shape[0] // nx

    def forward(self, x):
        branch_input = x[:, :3, :, 0].reshape(x.shape[0], -1)
        if self.n_time:
            branch_input = torch.cat([branch_input, x[:, 3, 0, :]], dim=1)
        branch_features = self.branch(branch_input)
        trunk_features = self.trunk(self.coordinates)
        values = torch.einsum("bp,np->bn", branch_features, trunk_features)
        values = values / math.sqrt(branch_features.shape[-1]) + self.bias
        return values.reshape(x.shape[0], 1, self.nx, self.nt)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x):
        nx_local = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1, norm="ortho")
        modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            x.shape[0], self.weight.shape[1], x_ft.shape[-1],
            dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :modes], self.weight[:, :, :modes])
        return torch.fft.irfft(out_ft, n=nx_local, dim=-1, norm="ortho")


class FrequencyFNO1d(nn.Module):
    def __init__(self, width=8, modes=8, layers=2, in_channels=4):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.project = nn.Sequential(
            nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 2, 1))

    def forward(self, profiles):
        z = self.lift(profiles)
        for spectral, local in zip(self.spectral, self.local):
            z = F.gelu(spectral(z) + local(z))
        return self.project(z)


class SimplePFNO(nn.Module):
    def __init__(self, nt, width=8, modes=8, layers=2, source_mode=False):
        # source_mode conditions each frequency branch on the source's own
        # coefficient at that temporal bin -- the natural input for a model
        # that solves each frequency independently.
        super().__init__()
        self.nt = nt
        self.source_mode = bool(source_mode)
        self.n_frequencies = nt // 2 + 1
        in_channels = 6 if self.source_mode else 4
        self.frequency_branches = nn.ModuleList(
            [FrequencyFNO1d(width=width, modes=modes, layers=layers,
                            in_channels=in_channels)
             for _ in range(self.n_frequencies)])

    def forward(self, x):
        if self.source_mode:
            # [E, rho, s(x), x] -- skip channel 3, which is w(t), not a profile.
            profiles = x[:, [0, 1, 2, 4], :, 0]
            wavelet = x[:, 3, 0, :]                                  # (B, nt)
            spectrum_w = torch.fft.rfft(wavelet, dim=-1, norm="ortho")
        else:
            profiles = x[:, :4, :, 0]
        coefficients = []
        for frequency_id, branch in enumerate(self.frequency_branches):
            if self.source_mode:
                nx_local = profiles.shape[-1]
                wk = spectrum_w[:, frequency_id]
                wr = wk.real[:, None, None].expand(-1, 1, nx_local)
                wi = wk.imag[:, None, None].expand(-1, 1, nx_local)
                branch_input = torch.cat([profiles, wr, wi], dim=1)
            else:
                branch_input = profiles
            real_imag = branch(branch_input)
            imag = real_imag[:, 1]
            if frequency_id == 0 or (
                    self.nt % 2 == 0 and frequency_id == self.n_frequencies - 1):
                imag = torch.zeros_like(imag)
            coefficients.append(torch.complex(real_imag[:, 0], imag))
        spectrum = torch.stack(coefficients, dim=-1).unsqueeze(1)
        return torch.fft.irfft(spectrum, n=self.nt, dim=-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
