"""FNO / PFNO / DeepONet / GNO for the OpenFWI forward map.

Every model here learns the same operator:

    velocity  (B, 1, 70, 70)      ->   shot gathers  (B, 5, 1000, 70)
              (depth, offset)                        (shot, time, receiver)

The receiver axis and the velocity map's lateral axis are the same 70-point
grid, so the only axis that has to change meaning is depth -> time. FNO and GNO
share that reparametrization (`DepthToTime`) and the same output head, and
differ only in the block they stack -- spectral convolution vs graph kernel
integration -- which is what makes their comparison an architecture comparison
rather than a decoder comparison. DeepONet and PFNO do not fit that template
and are left in their own idiom: DeepONet is a separable branch/trunk expansion
over the full (shot, time, receiver) point set, PFNO is frequency-native and
never builds a time-domain latent at all.

Conventions follow wave/operator_sim/operator_models.py: separate positive and
negative first-axis spectral modes, GELU between blocks, and a complex weight
counted as two real scalars when reporting parameters.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------
def coord_grid(n_a, n_b, device=None, dtype=torch.float32):
    """(2, n_a, n_b) normalized coordinates in [-1, 1]."""
    a = torch.linspace(-1.0, 1.0, n_a, device=device, dtype=dtype)
    b = torch.linspace(-1.0, 1.0, n_b, device=device, dtype=dtype)
    return torch.stack(torch.meshgrid(a, b, indexing="ij"))


class SpectralConv2d(nn.Module):
    """Truncated 2D spectral convolution, `wave/operator_sim` convention."""

    def __init__(self, in_channels, out_channels, modes_a, modes_b):
        super().__init__()
        self.modes_a = modes_a
        self.modes_b = modes_b
        scale = 1.0 / max(1, in_channels * out_channels)
        shape = (in_channels, out_channels, modes_a, modes_b)
        self.weight_pos = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_neg = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    def forward(self, x):
        batch, _, na, nb = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(batch, self.weight_pos.shape[1], na, nb // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        ma = min(self.modes_a, na)
        mb = min(self.modes_b, nb // 2 + 1)
        out_ft[:, :, :ma, :mb] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :ma, :mb], self.weight_pos[:, :, :ma, :mb])
        out_ft[:, :, -ma:, :mb] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -ma:, :mb], self.weight_neg[:, :, :ma, :mb])
        return torch.fft.irfft2(out_ft, s=(na, nb), norm="ortho")


class DepthToTime(nn.Module):
    """Turn the depth axis of a latent into a time axis: (B,W,nz,nx) -> (B,W,T,nx).

    A single learned (T, nz) matrix shared across channels and receivers. It is
    initialized to the linear-interpolation matrix, so the block starts life as
    a plain resampler and only departs from it if the data pays for it. Sharing
    one matrix across x encodes the assumption that the depth-to-traveltime
    warp is laterally smooth, which is what a 10 m grid over a 1500-4500 m/s
    model gives you.
    """

    def __init__(self, n_depth, t_latent):
        super().__init__()
        src = torch.linspace(0.0, 1.0, n_depth)
        dst = torch.linspace(0.0, 1.0, t_latent)
        # rows of the linear-interpolation operator
        w = torch.zeros(t_latent, n_depth)
        idx = torch.clamp(torch.searchsorted(src, dst, right=True) - 1, 0, n_depth - 2)
        left, right = src[idx], src[idx + 1]
        frac = (dst - left) / (right - left).clamp_min(1e-12)
        rows = torch.arange(t_latent)
        w[rows, idx] = 1.0 - frac
        w[rows, idx + 1] = frac
        self.map = nn.Parameter(w)

    def forward(self, z):
        return torch.einsum("bwzx,tz->bwtx", z, self.map)


class TimeDecoderHead(nn.Module):
    """(B,W,T_latent,nx) -> (B,n_sources,nt,nx_out).

    Bilinear upsampling alone would cap accuracy at the time-resample oracle in
    openfwi_data.py, so the upsample is followed by two convolutions with a
    temporal kernel that can put back detail the interpolation smoothed out.

    `nx_out` exists for the field-scale case, where the velocity map is coarser
    laterally than the receiver line. On OpenFWI the two axes are the same
    70-point grid and this is a no-op.
    """

    def __init__(self, width, n_sources, nt, kernel=5, nx_out=None):
        super().__init__()
        self.nt = nt
        self.nx_out = nx_out
        self.pre = nn.Conv2d(width, width, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(width, width, (kernel, 1), padding=(kernel // 2, 0)),
            nn.GELU(),
            nn.Conv2d(width, n_sources, 1),
        )

    def forward(self, z):
        z = self.pre(z)
        nx_out = self.nx_out or z.shape[-1]
        if z.shape[-2] != self.nt or z.shape[-1] != nx_out:
            z = F.interpolate(z, size=(self.nt, nx_out),
                              mode="bilinear", align_corners=True)
        return self.refine(z)


# ---------------------------------------------------------------------------
# FNO
# ---------------------------------------------------------------------------
class OpenFWIFNO(nn.Module):
    def __init__(self, width=32, modes_z=16, modes_x=16, modes_t=32,
                 enc_layers=3, dec_layers=3, n_sources=5, nz=70, nx=70,
                 nt=1000, t_latent=250, nx_out=None):
        super().__init__()
        self.nz, self.nx = nz, nx
        self.lift = nn.Conv2d(3, width, 1)
        self.enc_spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes_z, modes_x) for _ in range(enc_layers)])
        self.enc_local = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(enc_layers)])
        self.d2t = DepthToTime(nz, t_latent)
        self.dec_spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes_t, modes_x) for _ in range(dec_layers)])
        self.dec_local = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(dec_layers)])
        self.head = TimeDecoderHead(width, n_sources, nt, nx_out=nx_out)
        self.register_buffer("coords", coord_grid(nz, nx), persistent=False)

    def forward(self, v):
        z = self.lift(torch.cat([v, self.coords.expand(v.shape[0], -1, -1, -1)], dim=1))
        for spectral, local in zip(self.enc_spectral, self.enc_local):
            z = F.gelu(spectral(z) + local(z))
        z = self.d2t(z)
        for spectral, local in zip(self.dec_spectral, self.dec_local):
            z = F.gelu(spectral(z) + local(z))
        return self.head(z)


# ---------------------------------------------------------------------------
# GNO
# ---------------------------------------------------------------------------
def ball_offsets(radius):
    """Integer (da, db) offsets inside an L2 ball of `radius` grid points."""
    r = int(radius)
    return [(i, j) for i in range(-r, r + 1) for j in range(-r, r + 1)
            if i * i + j * j <= r * r]


class GraphKernelLayer(nn.Module):
    """One graph kernel integration (Li et al. 2020, arXiv:2003.03485):

        v_{l+1}(i) = GELU( W v_l(i) + mean_{j in N(i)} k(dz, dx, a_i, a_j) * v_l(j) )

    N(i) is the L2 ball of `radius` grid points, so the operator is a local
    kernel integral rather than a global convolution. The kernel is *diagonal*
    in the channel axis: k returns a width-vector applied elementwise instead of
    a width x width matrix. A matrix kernel costs `width` times more memory per
    edge -- at 4900 nodes x 29 neighbours that is the difference between a layer
    that fits beside another tenant on the GPU and one that does not -- and the
    channel-mixing it would provide is exactly what the W term already does.

    The kernel stays genuinely input-dependent (it reads a_i and a_j, the
    velocity at both endpoints), which is what separates a GNO from a CNN with
    the same stencil.

    Implemented by shifting the padded feature map once per offset rather than
    with scatter/gather on an explicit edge list: the grid is regular and the
    stencil is identical at every node, so an edge list would store 142k
    redundant indices and force an unsorted scatter.
    """

    def __init__(self, width, radius, kernel_hidden, a_channels=1):
        super().__init__()
        self.radius = int(radius)
        offsets = ball_offsets(radius)
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.float32), persistent=False)
        self.offset_list = offsets
        # First kernel layer split into its constant (edge geometry) and its
        # spatially varying (endpoint velocities) halves. Identical to one
        # Linear on the concatenation, but the geometry half is a per-offset
        # constant so it is evaluated once instead of broadcast over the grid.
        self.k_pos = nn.Linear(2, kernel_hidden)
        self.k_a = nn.Conv2d(2 * a_channels, kernel_hidden, 1, bias=False)
        self.k_out = nn.Conv2d(kernel_hidden, width, 1)
        self.local = nn.Conv2d(width, width, 1)

    def forward(self, v, a):
        r = self.radius
        n_a, n_b = v.shape[-2:]
        v_pad = F.pad(v, (r, r, r, r))
        a_pad = F.pad(a, (r, r, r, r))
        mask_pad = F.pad(torch.ones_like(a[:, :1]), (r, r, r, r))
        pos = self.k_pos(self.offsets / max(1.0, r))          # (O, kernel_hidden)

        total = torch.zeros_like(v)
        count = torch.zeros_like(mask_pad[:, :, :n_a, :n_b])
        for oid, (da, db) in enumerate(self.offset_list):
            sa, sb = r + da, r + db
            v_shift = v_pad[:, :, sa:sa + n_a, sb:sb + n_b]
            a_shift = a_pad[:, :, sa:sa + n_a, sb:sb + n_b]
            m_shift = mask_pad[:, :, sa:sa + n_a, sb:sb + n_b]
            hidden = self.k_a(torch.cat([a, a_shift], dim=1))
            hidden = hidden + pos[oid].view(1, -1, 1, 1)
            k = self.k_out(F.gelu(hidden))
            total = total + k * v_shift * m_shift
            count = count + m_shift
        return F.gelu(self.local(v) + total / count.clamp_min(1.0))


class OpenFWIGNO(nn.Module):
    """Graph neural operator over the velocity grid, then over the (t, x) grid.

    Same DepthToTime reparametrization and output head as OpenFWIFNO; the only
    difference is that every block is a local graph kernel integration instead
    of a global spectral convolution.

    The decoder graph carries `a` = the surface row of the velocity map,
    broadcast along time. The decoder's nodes live on (time, receiver) where no
    velocity is defined, and the near-surface velocity is what actually controls
    the gather, so it is the physically right thing to condition on -- and it
    keeps the decoder kernel input-dependent rather than collapsing to a CNN.
    """

    def __init__(self, width=32, kernel_hidden=64, radius=3, dec_radius=2,
                 enc_layers=3, dec_layers=2, n_sources=5, nz=70, nx=70,
                 nt=1000, t_latent=250, use_checkpoint=False, nx_out=None):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.lift = nn.Conv2d(3, width, 1)
        self.enc = nn.ModuleList(
            [GraphKernelLayer(width, radius, kernel_hidden, a_channels=1)
             for _ in range(enc_layers)])
        self.d2t = DepthToTime(nz, t_latent)
        self.dec = nn.ModuleList(
            [GraphKernelLayer(width, dec_radius, kernel_hidden, a_channels=1)
             for _ in range(dec_layers)])
        self.head = TimeDecoderHead(width, n_sources, nt, nx_out=nx_out)
        self.register_buffer("coords", coord_grid(nz, nx), persistent=False)
        self.t_latent = t_latent

    def _run(self, layer, z, a):
        if self.use_checkpoint and self.training:
            return checkpoint(layer, z, a, use_reentrant=False)
        return layer(z, a)

    def forward(self, v):
        z = self.lift(torch.cat([v, self.coords.expand(v.shape[0], -1, -1, -1)], dim=1))
        for layer in self.enc:
            z = self._run(layer, z, v)
        z = self.d2t(z)
        a_dec = v[:, :, :1, :].expand(-1, -1, self.t_latent, -1)
        for layer in self.dec:
            z = self._run(layer, z, a_dec)
        return self.head(z)


# ---------------------------------------------------------------------------
# DeepONet
# ---------------------------------------------------------------------------
def make_mlp(sizes, activation=nn.Tanh):
    layers = []
    for in_size, out_size in zip(sizes[:-2], sizes[1:-1]):
        layers += [nn.Linear(in_size, out_size), activation()]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


class OpenFWIDeepONet(nn.Module):
    """u(y) ~ sum_p b_p(velocity) * tau_p(y), y = (shot, time, receiver).

    The trunk gets random Fourier features. A plain tanh trunk cannot represent
    a 15 Hz wavelet across a 1 s window -- that is spectral bias at its most
    literal -- and would report a DeepONet number that measures the trunk's
    frequency ceiling rather than the branch/trunk factorization. This favours
    DeepONet against its textbook form, deliberately: the point of the
    comparison is the low-rank separable ansatz, not the MLP inside it.
    Set fourier_features=0 to recover the plain version.

    The trunk is evaluated on all ns*nt*ng points once per forward and shared
    across the batch, so its cost does not scale with batch size.
    """

    def __init__(self, nz=70, nx=70, nt=1000, n_sources=5, latent=128,
                 hidden=256, fourier_features=32, fourier_scale=8.0, seed=0,
                 nx_out=None):
        super().__init__()
        nx_out = nx_out or nx
        self.nt, self.nx, self.n_sources = nt, nx_out, n_sources
        self.branch = make_mlp([nz * nx, hidden, hidden, latent])
        trunk_in = 3 if not fourier_features else 2 * fourier_features
        self.trunk = make_mlp([trunk_in, hidden, hidden, latent])
        self.bias = nn.Parameter(torch.zeros(1))

        s = torch.linspace(-1.0, 1.0, n_sources)
        t = torch.linspace(-1.0, 1.0, nt)
        g = torch.linspace(-1.0, 1.0, nx_out)
        pts = torch.stack(torch.meshgrid(s, t, g, indexing="ij"), dim=-1)
        self.register_buffer("points", pts.reshape(-1, 3), persistent=False)

        if fourier_features:
            gen = torch.Generator().manual_seed(seed)
            b = torch.randn(3, fourier_features, generator=gen) * fourier_scale
            # Time is the axis that oscillates; give it the full bandwidth and
            # damp the two geometric axes, which vary smoothly across a gather.
            b[0] *= 0.25
            b[2] *= 0.25
            self.register_buffer("fourier_b", b, persistent=False)
        else:
            self.fourier_b = None

    def _trunk_input(self):
        if self.fourier_b is None:
            return self.points
        proj = 2.0 * math.pi * self.points @ self.fourier_b
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, v):
        b = self.branch(v.reshape(v.shape[0], -1))
        tau = self.trunk(self._trunk_input())
        out = torch.einsum("bp,np->bn", b, tau) / math.sqrt(b.shape[-1]) + self.bias
        return out.reshape(v.shape[0], self.n_sources, self.nt, self.nx)


# ---------------------------------------------------------------------------
# PFNO
# ---------------------------------------------------------------------------
class GroupedSpectralConv2d(nn.Module):
    """`groups` independent SpectralConv2d evaluated in one einsum.

    Mathematically identical to a list of SpectralConv2d modules applied to a
    stacked leading axis -- test_openfwi.py checks that against a loop
    reference -- but the loop version costs one kernel launch per frequency
    branch, which is what made the 1D PFNO in wave/operator_sim 11x slower than
    its FNO.
    """

    def __init__(self, groups, in_channels, out_channels, modes_a, modes_b):
        super().__init__()
        self.modes_a, self.modes_b = modes_a, modes_b
        scale = 1.0 / max(1, in_channels * out_channels)
        shape = (groups, in_channels, out_channels, modes_a, modes_b)
        self.weight_pos = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_neg = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    def forward(self, x):
        batch, groups, _, na, nb = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(batch, groups, self.weight_pos.shape[2], na, nb // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        ma = min(self.modes_a, na)
        mb = min(self.modes_b, nb // 2 + 1)
        out_ft[:, :, :, :ma, :mb] = torch.einsum(
            "bkixy,kioxy->bkoxy", x_ft[:, :, :, :ma, :mb],
            self.weight_pos[:, :, :, :ma, :mb])
        out_ft[:, :, :, -ma:, :mb] = torch.einsum(
            "bkixy,kioxy->bkoxy", x_ft[:, :, :, -ma:, :mb],
            self.weight_neg[:, :, :, :ma, :mb])
        return torch.fft.irfft2(out_ft, s=(na, nb), norm="ortho")


class OpenFWIPFNO(nn.Module):
    """Paralleled FNO (Li et al., arXiv:2209.12340): one operator per frequency.

    Each of `n_freqs` branches is an independent 2D FNO over the velocity map
    that predicts the real and imaginary parts of one temporal rFFT bin of the
    gather, for every shot and receiver. An inverse rFFT reassembles the time
    series. This is the time-domain analogue of the paper's per-frequency
    Helmholtz solves, applied to the 2D OpenFWI geometry that the 1D pipeline
    in wave/operator_sim explicitly did not attempt.

    `n_freqs` band-limits the prediction: bins at or above it are set to zero,
    which imposes a hard error floor no amount of training can cross. Measure
    it with openfwi_data.band_limit_oracle and report it next to the model's
    error -- at OpenFWI's 15 Hz Ricker and dt = 1 ms, 64 of the 501 bins hold
    essentially all the energy, but that is a fact about this source, not a
    free lunch.
    """

    def __init__(self, n_freqs=64, width=10, modes=8, layers=2, n_sources=5,
                 nz=70, nx=70, nt=1000, depth_reductions=3, nx_out=None):
        super().__init__()
        self.n_freqs, self.nt, self.nx = n_freqs, nt, nx
        self.nx_out = nx_out or nx
        self.n_sources, self.width = n_sources, width
        self.n_rfft = nt // 2 + 1
        if n_freqs > self.n_rfft:
            raise ValueError("n_freqs %d exceeds %d rFFT bins" % (n_freqs, self.n_rfft))
        k = n_freqs
        self.lift = nn.Conv2d(k * 3, k * width, 1, groups=k)
        self.spectral = nn.ModuleList(
            [GroupedSpectralConv2d(k, width, width, modes, modes) for _ in range(layers)])
        self.local = nn.ModuleList(
            [nn.Conv2d(k * width, k * width, 1, groups=k) for _ in range(layers)])
        # Collapse depth with strided convolutions rather than a mean: a mean
        # over depth would throw away exactly the layering that sets the
        # arrival times this branch has to predict.
        reduce_layers, n_depth = [], nz
        for _ in range(depth_reductions):
            reduce_layers.append(nn.Conv2d(k * width, k * width, (3, 1),
                                           stride=(2, 1), padding=(1, 0), groups=k))
            n_depth = (n_depth + 1) // 2
        self.reduce = nn.ModuleList(reduce_layers)
        self.n_depth_out = n_depth
        self.head = nn.Conv1d(k * width * n_depth, k * 2 * n_sources, 1, groups=k)
        self.register_buffer("coords", coord_grid(nz, nx), persistent=False)

    def forward(self, v):
        batch = v.shape[0]
        k, w = self.n_freqs, self.width
        x = torch.cat([v, self.coords.expand(batch, -1, -1, -1)], dim=1)
        z = self.lift(x.repeat(1, k, 1, 1))
        for spectral, local in zip(self.spectral, self.local):
            na, nb = z.shape[-2:]
            grouped = spectral(z.view(batch, k, w, na, nb)).reshape(batch, k * w, na, nb)
            z = F.gelu(grouped + local(z))
        for layer in self.reduce:
            z = F.gelu(layer(z))
        z = z.reshape(batch, k * w * self.n_depth_out, self.nx)
        out = self.head(z).view(batch, k, 2, self.n_sources, self.nx)

        real, imag = out[:, :, 0], out[:, :, 1]
        # A real-valued time series needs a real DC coefficient.
        imag = torch.cat([torch.zeros_like(imag[:, :1]), imag[:, 1:]], dim=1)
        spectrum = torch.complex(real, imag)                  # (B, K, ns, nx)
        spectrum = spectrum.permute(0, 2, 3, 1)               # (B, ns, nx, K)
        if k < self.n_rfft:
            spectrum = F.pad(spectrum, (0, self.n_rfft - k))
        gather = torch.fft.irfft(spectrum, n=self.nt, dim=-1)  # (B, ns, nx, nt)
        gather = gather.permute(0, 1, 3, 2)                    # (B, ns, nt, nx)
        if self.nx_out != self.nx:
            gather = F.interpolate(gather, size=(self.nt, self.nx_out),
                                   mode="bilinear", align_corners=True)
        return gather.contiguous()


# ---------------------------------------------------------------------------
# factory + accounting
# ---------------------------------------------------------------------------
MODEL_NAMES = ("FNO", "PFNO", "DeepONet", "GNO")


def build_model(name, args, cfg, nt):
    """`args` is the argparse namespace (or a dict from a saved summary)."""
    if not isinstance(args, dict):
        args = vars(args)
    # OpenFWI's velocity map is square and shares its lateral axis with the
    # receiver line; the field-scale sets do neither, so all three are read
    # separately and fall back to the square convention.
    nz = cfg.get("nz", cfg.get("n_grid"))
    nx = cfg.get("nx", cfg.get("n_grid"))
    ng = cfg.get("ng", nx)
    nx_out = None if ng == nx else ng
    ns = cfg["ns"]
    if name == "FNO":
        return OpenFWIFNO(width=args["fno_width"], modes_z=args["fno_modes_z"],
                          modes_x=args["fno_modes_x"], modes_t=args["fno_modes_t"],
                          enc_layers=args["fno_enc_layers"],
                          dec_layers=args["fno_dec_layers"],
                          n_sources=ns, nz=nz, nx=nx, nt=nt,
                          t_latent=args["t_latent"], nx_out=nx_out)
    if name == "GNO":
        return OpenFWIGNO(width=args["gno_width"], kernel_hidden=args["gno_kernel_hidden"],
                          radius=args["gno_radius"], dec_radius=args["gno_dec_radius"],
                          enc_layers=args["gno_enc_layers"],
                          dec_layers=args["gno_dec_layers"],
                          n_sources=ns, nz=nz, nx=nx, nt=nt,
                          t_latent=args["gno_t_latent"],
                          use_checkpoint=args.get("gno_checkpoint", False),
                          nx_out=nx_out)
    if name == "DeepONet":
        return OpenFWIDeepONet(nz=nz, nx=nx, nt=nt, n_sources=ns,
                               latent=args["don_latent"], hidden=args["don_hidden"],
                               fourier_features=args["don_fourier"],
                               fourier_scale=args["don_fourier_scale"],
                               seed=args.get("init_seed", 0) or 0,
                               nx_out=nx_out)
    if name == "PFNO":
        return OpenFWIPFNO(n_freqs=args["pfno_freqs"], width=args["pfno_width"],
                           modes=args["pfno_modes"], layers=args["pfno_layers"],
                           n_sources=ns, nz=nz, nx=nx, nt=nt, nx_out=nx_out)
    raise ValueError("unknown model %r; known: %s" % (name, list(MODEL_NAMES)))


def count_parameters(model):
    """Nominal count: a complex weight counts once."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_real(model):
    """Trainable real scalars. A complex spectral weight holds two, and
    counting it as one understates FNO/PFNO 2x against the all-real DeepONet
    and GNO."""
    return sum(p.numel() * (2 if p.is_complex() else 1)
               for p in model.parameters() if p.requires_grad)
