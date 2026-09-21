"""2-D acoustic finite-difference solver: the (v, p) pair.

Constant-density acoustic wave equation  p_tt = v^2 (p_yy + p_zz) + v^2 s(t) delta,
fourth-order in space, second-order in time, on the model padded by `nbc`
cells of Cerjan sponge on all four sides (no free surface, as in OpenFWI's
generator: the published gathers carry no surface multiples).

Acquisition follows OpenFWI's 2-D families exactly: 70 x 70 cells at 10 m,
`ns` = 5 sources at 10 m depth spread evenly across the surface, receivers at
10 m depth on every one of the 70 columns, nt = 1000 samples at dt = 1 ms,
15 Hz Ricker wavelet. Output (ns, nt, 70). Amplitudes are in the solver's own
units (unit-amplitude wavelet injected as a pressure source), so they are
consistent across this dataset but not numerically identical to OpenFWI's.

Stability: v_max dt / dx <= 0.61 for this stencil in 2-D; with v <= 4500 m/s,
dt = 1 ms, dx = 10 m the Courant number is 0.45. Dispersion: 15 Hz in 1500 m/s
is 10 cells per wavelength, which is the same compromise OpenFWI made.

All shots are propagated together as a batch axis, in float32, so the whole
inner loop is a handful of array expressions per time step.
"""
import numpy as np


def ricker(t, f0, t0=None):
    t0 = 1.0 / f0 if t0 is None else t0
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def cerjan_profile(n_core, nbc, strength=0.2):
    """1-D per-step damping factor: 1 in the core, exp(-(strength*d/nbc)^2) into the
    sponge (d = cells from the inner edge). Gentle is right: measured boundary
    residual on a homogeneous model with nbc=120 is 2e-4 of the peak at 0.2,
    3e-3 at 1.0 and 2e-2 at 3.0 -- a strong sponge reflects off its own inner
    edge. Cerjan et al. (1985) used the equivalent of 0.3 for nbc=20."""
    n = n_core + 2 * nbc
    g = np.ones(n)
    d = np.arange(nbc, 0, -1) / nbc                 # 1 at the edge -> ~0 at the core
    g[:nbc] = np.exp(-(strength * d) ** 2)
    g[n - nbc:] = g[:nbc][::-1]
    return g


def source_columns(ny, ns):
    return np.round(np.linspace(0, ny - 1, ns)).astype(int)


class SeismicSolver:
    def __init__(self, nz=70, ny=70, dx=10.0, dt=1e-3, nt=1000, f0=15.0, ns=5,
                 src_depth=1, rec_depth=1, nbc=120, sponge=0.2, dtype=np.float32):
        self.nz, self.ny, self.dx, self.dt, self.nt, self.f0 = nz, ny, dx, dt, nt, f0
        self.ns, self.nbc, self.dtype = ns, nbc, dtype
        self.src_cols = source_columns(ny, ns)
        self.src_row, self.rec_row = src_depth, rec_depth
        t = np.arange(nt) * dt
        self.wavelet = ricker(t, f0).astype(dtype)
        self.damp = np.outer(cerjan_profile(nz, nbc, sponge), cerjan_profile(ny, nbc, sponge)).astype(dtype)

    def courant(self, v):
        return float(np.max(v)) * self.dt / self.dx

    def shots(self, v):
        """(ns, nt, ny) pressure at the receivers for a (nz, ny) velocity map."""
        v = np.asarray(v, np.float64)
        if self.courant(v) > 0.61:
            raise ValueError("unstable: v_max dt/dx = %.3f > 0.61" % self.courant(v))
        nbc, nz, ny, ns = self.nbc, self.nz, self.ny, self.ns
        vp = np.pad(v, nbc, mode="edge")
        c2 = ((vp * self.dt) ** 2 / self.dx ** 2).astype(self.dtype)     # v^2 dt^2 / dx^2
        src_amp = (vp[nbc + self.src_row, nbc + self.src_cols] * self.dt) ** 2       # v^2 dt^2 at the sources
        Nz, Ny = vp.shape
        p0 = np.zeros((ns, Nz, Ny), self.dtype)      # previous
        p1 = np.zeros_like(p0)                        # current
        p2 = np.zeros_like(p0)                        # next
        rec = np.zeros((ns, self.nt, ny), self.dtype)
        shot = np.arange(ns)
        zs, ys = nbc + self.src_row, nbc + self.src_cols
        zr, yc = nbc + self.rec_row, slice(nbc, nbc + ny)
        c2i = c2[2:-2, 2:-2]
        damp = self.damp
        for it in range(self.nt):
            lap = (-60.0 * p1[:, 2:-2, 2:-2]
                   + 16.0 * (p1[:, 1:-3, 2:-2] + p1[:, 3:-1, 2:-2] + p1[:, 2:-2, 1:-3] + p1[:, 2:-2, 3:-1])
                   - (p1[:, :-4, 2:-2] + p1[:, 4:, 2:-2] + p1[:, 2:-2, :-4] + p1[:, 2:-2, 4:])) / 12.0
            p2[:, 2:-2, 2:-2] = 2.0 * p1[:, 2:-2, 2:-2] - p0[:, 2:-2, 2:-2] + c2i * lap
            p2[shot, zs, ys] += src_amp * self.wavelet[it]
            p2 *= damp
            p1 *= damp
            rec[:, it, :] = p2[:, zr, yc]
            p0, p1, p2 = p1, p2, p0
        return rec
