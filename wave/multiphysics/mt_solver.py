"""2-D magnetotelluric forward solver: the (c, m) pair.

Maxwell in the quasi-static limit (displacement current dropped: the earth is
a conductor and the frequencies are low), with a plane-wave source and a 2-D
conductivity structure sigma(y, z) invariant along strike x, splits into two
scalar problems (time dependence e^{+i w t}):

    TE  (E along strike)   d2E/dy2 + d2E/dz2 - i w mu0 sigma E = 0
    TM  (H along strike)   d/dy(rho dH/dy) + d/dz(rho dH/dz) - i w mu0 H = 0

Both are solved on the padded finite-volume grid of grid.py:

  * lateral padding replicates the edge columns (the structure continues),
    natural Neumann at the far sides;
  * the core is refined `refine`x (10 m -> 5 m) because the skin depth at the
    top of the band in conductive ground is only a few tens of metres;
  * below the core the bottom row is replicated and the last row carries the
    Robin condition du/dz + gamma u = 0, gamma = sqrt(i w mu0 sigma), which is
    the exact 1-D basement response -- so the grid does not have to reach
    several skin depths at the lowest frequency;
  * TE needs the air: sigma_air = 1e-8 S/m, stretched padding upward with E = 1
    at the top (in the insulating air E is linear in z, so any far Dirichlet
    value is fine); TM needs no air, H = 1 on the surface.

Surface impedance from the field and its one-sided vertical derivative:

    TE  Z = -i w mu0 E / (dE/dz)        TM  Z = rho dH/dz / H  (sign so phase ~ 45 deg)
    rho_a = |Z|^2 / (w mu0),  phase = arg Z  (degrees)

Both give rho_a = rho and phase = 45 deg on a half-space, which the tests
check, together with the 1-D layered recursion `layered_impedance`.
"""
import numpy as np
import scipy.sparse.linalg as spla

from grid import Grid2D, padded_axis, pad_edge, assemble, set_dirichlet, set_robin_bottom

MU0 = 4e-7 * np.pi
SIGMA_AIR = 1e-8


def default_frequencies(n=8, f_min=5.0, f_max=1000.0):
    """Log-spaced band. Skin depth 503 sqrt(rho/f) m: for rho in [2, 5000] ohm m
    this spans ~20 m (conductive, 1 kHz) to ~16 km (resistive, 5 Hz), so the
    band sees from the first cells to well below the 700 m model."""
    return np.logspace(np.log10(f_min), np.log10(f_max), n)


def _one_sided_dz(u0, u1, u2, h1, h2):
    """Second-order one-sided derivative at node 0 from nodes at h1, h1+h2 below."""
    return (-(2 * h1 + h2) / (h1 * (h1 + h2)) * u0
            + (h1 + h2) / (h1 * h2) * u1
            - h1 / (h2 * (h1 + h2)) * u2)


class MTSolver:
    """Fixed geometry; call `response(sigma, freqs)` per sample."""

    def __init__(self, nz=70, ny=70, dx=10.0, refine=2, n_pad_side=14,
                 n_pad_bottom=24, n_pad_air=16, growth=1.4, growth_bottom=1.25):
        # Bottom padding is gentler and deeper (24 x 1.25 from 5 m: ~5 km, last
        # cell ~1 km) because the Robin row's accuracy scales with the last
        # spacing over the skin depth; sides (~1.9 km) and air (~3.8 km) can
        # stretch faster.
        self.nz, self.ny, self.dx, self.refine = nz, ny, dx, refine
        h = dx / refine
        self.h = h
        self.n_pad_side, self.n_pad_bottom, self.n_pad_air = n_pad_side, n_pad_bottom, n_pad_air
        # ground-only grid (TM) and ground + air grid (TE) share the lateral axis
        y, self.yc = padded_axis(ny * refine, h, n_pad_side, n_pad_side, growth)
        z_g, _ = padded_axis(nz * refine, h, 0, n_pad_bottom, growth_bottom)
        z_a, _ = padded_axis(nz * refine, h, n_pad_air, n_pad_bottom, growth)
        # air padding grows at `growth`, bottom at `growth_bottom`: splice them
        z_a = np.concatenate([z_a[:n_pad_air], z_g])
        self.grid_tm = Grid2D(z_g, y)
        self.grid_te = Grid2D(z_a, y)
        self.surface_te = n_pad_air                     # row index of z = 0 in the TE grid
        # station columns: every core (10 m) column -> node index of its centre
        self.station_cols = self.yc.start + np.arange(ny) * refine + (refine // 2 if refine > 1 else 0)
        self.station_y = self.grid_tm.y[self.station_cols]

    # -- model preparation --------------------------------------------------
    def _refine(self, sigma):
        return np.repeat(np.repeat(np.asarray(sigma, float), self.refine, 0), self.refine, 1)

    def _ground(self, sigma):
        return pad_edge(self._refine(sigma), 0, self.n_pad_bottom, self.n_pad_side, self.n_pad_side)

    # -- modes ----------------------------------------------------------------
    def _solve_te(self, sig_ground, omega):
        g = self.grid_te
        sig = np.vstack([np.full((self.n_pad_air, g.ny), SIGMA_AIR), sig_ground])
        # surface node: half its control volume is air
        s = self.surface_te
        w_air, w_gnd = g.hz[s - 1] / 2, g.hz[s] / 2
        sig_cv = sig.copy()
        sig_cv[s] = (w_air * SIGMA_AIR + w_gnd * sig_ground[0]) / (w_air + w_gnd)
        M = assemble(g, 1.0, 1j * omega * MU0 * sig_cv)
        M = set_robin_bottom(M, g, np.sqrt(1j * omega * MU0 * sig[-1]))
        M, rhs = set_dirichlet(M, g.index()[0, :], 1.0)
        E = spla.splu(M).solve(rhs).reshape(g.shape)
        E0, E1, E2 = E[s], E[s + 1], E[s + 2]
        dE = _one_sided_dz(E0, E1, E2, g.hz[s], g.hz[s + 1])
        Z = -1j * omega * MU0 * E0 / dE
        return Z

    def _solve_tm(self, sig_ground, omega):
        g = self.grid_tm
        rho = 1.0 / sig_ground
        M = assemble(g, rho, 1j * omega * MU0 * np.ones(g.shape))
        M = set_robin_bottom(M, g, np.sqrt(1j * omega * MU0 * sig_ground[-1]))
        M, rhs = set_dirichlet(M, g.index()[0, :], 1.0)
        H = spla.splu(M).solve(rhs).reshape(g.shape)
        dH = _one_sided_dz(H[0], H[1], H[2], g.hz[0], g.hz[1])
        Z = -rho[0] * dH / H[0]
        return Z

    # -- public ---------------------------------------------------------------
    def impedances(self, sigma, freqs):
        """Complex surface impedances at the stations: (Z_te, Z_tm), each (n_freq, ny)."""
        sig_g = self._ground(sigma)
        Zte = np.empty((len(freqs), self.ny), complex)
        Ztm = np.empty_like(Zte)
        for i, f in enumerate(freqs):
            w = 2 * np.pi * f
            Zte[i] = self._solve_te(sig_g, w)[self.station_cols]
            Ztm[i] = self._solve_tm(sig_g, w)[self.station_cols]
        return Zte, Ztm

    def response(self, sigma, freqs):
        """(n_freq, ny, 4): [rho_a TE, phase TE, rho_a TM, phase TM]; ohm m and degrees."""
        Zte, Ztm = self.impedances(sigma, freqs)
        w = 2 * np.pi * np.asarray(freqs)[:, None]
        out = np.stack([np.abs(Zte) ** 2 / (w * MU0), np.degrees(np.angle(Zte)),
                        np.abs(Ztm) ** 2 / (w * MU0), np.degrees(np.angle(Ztm))], axis=-1)
        return out.astype(np.float32)


# --------------------------------------------------------------------------
# analytic reference used by the tests
# --------------------------------------------------------------------------
def layered_impedance(freq, resistivities, thicknesses):
    """1-D MT impedance (Wait recursion) for layers over a half-space.
    thicknesses has one entry fewer than resistivities."""
    w = 2 * np.pi * freq
    rho = np.asarray(resistivities, float)
    gam = np.sqrt(1j * w * MU0 / rho)
    Z0 = 1j * w * MU0 / gam                     # intrinsic impedances
    Z = Z0[-1]
    for j in range(len(thicknesses) - 1, -1, -1):
        t = np.tanh(gam[j] * thicknesses[j])
        Z = Z0[j] * (Z + Z0[j] * t) / (Z0[j] + Z * t)
    return Z


def layered_apparent(freq, resistivities, thicknesses):
    Z = layered_impedance(freq, resistivities, thicknesses)
    return abs(Z) ** 2 / (2 * np.pi * freq * MU0), np.degrees(np.angle(Z))
