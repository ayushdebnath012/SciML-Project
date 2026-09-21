"""2.5-D DC resistivity forward solver: the (r, i) pair.

A point current source on a 2-D conductivity structure sigma(y, z) is a 3-D
problem. The standard trick (Dey & Morrison 1979) is a cosine transform along
strike x: for each wavenumber k solve the 2-D modified-Helmholtz problem

    div( sigma grad phi_k ) - k^2 sigma phi_k = -(I/2) delta(y - y_s) delta(z - z_s)

and recover the potential in the plane of the electrodes as

    phi(y, z) = (2/pi) * integral_0^inf phi_k dk  ~  (2/pi) * sum_j w_j phi_kj.

The wavenumbers and weights are fitted (Xu et al. 2000): with the half-space
solution phi_k = I rho K0(k r) / (2 pi), the weights are chosen by non-negative
least squares so that sum_j w_j K0(k_j r) = pi / (2 r). The fit range is *not*
the electrode spread: a layered earth is equivalent to image sources at
distances 2 n h, tens of kilometres for a strong contrast, and a quadrature
that is only accurate inside the spread mis-integrates them by several
percent (measured: 4.7 % on a 20 / 500 ohm m two-layer case). Fitting out to
30 km with 20 wavenumbers takes that below 1e-4.

Those small wavenumbers see the far boundary, so the outer edges carry the
Dey & Morrison (1979) mixed condition  d(phi_k)/dn + k K1(kr)/K0(kr) cos(theta)
phi_k = 0, exact for a half-space at every k, with r measured from the centre
of the electrode spread. What discretisation error remains at short offsets is
removed by a geometric-factor calibration against the unit half-space,
computed once per grid.

Data: surface electrodes at every core column; `n_src` of them inject current
(return electrode at infinity, i.e. pole source); potentials are read at all
electrodes and reported as pole-pole apparent resistivity

    rho_a = 2 pi r V / I          (r = electrode separation, in-plane).

The entry at r = 0 is undefined and is filled with its nearest neighbour.
Any four-electrode array (Wenner, dipole-dipole, ...) is a linear combination
of these pole potentials, so nothing is lost by storing the pole response.
"""
import numpy as np
import scipy.sparse.linalg as spla
from scipy.optimize import nnls
import scipy.sparse as sp
from scipy.special import k0, k0e, k1e

from grid import Grid2D, padded_axis, pad_edge, assemble


def wavenumber_quadrature(r_min, r_max=30e3, n_k=20, n_fit=200):
    """Fitted wavenumbers k_j and weights w_j with sum_j w_j K0(k_j r) = pi/(2r)
    to relative accuracy over r in [r_min, r_max]."""
    k = np.logspace(np.log10(0.02 / r_max), np.log10(6.0 / r_min), n_k)
    r = np.logspace(np.log10(r_min), np.log10(r_max), n_fit)
    G = k0(np.outer(r, k))
    target = np.pi / (2.0 * r)
    w, _ = nnls(G / target[:, None], np.ones(n_fit), maxiter=20000)   # relative-error fit
    fit_err = np.abs(G @ w / target - 1.0).max()
    return k, w, fit_err


class DCSolver:
    """Grid, wavenumbers, factorisation-free calibration for a fixed geometry.

    Build once, call `potentials(sigma)` per sample.
    """

    def __init__(self, nz=70, ny=70, dx=10.0, src_cols=None, n_pad_side=24,
                 n_pad_bottom=24, growth=1.4, n_k=20, current=1.0):
        # 24 stretched nodes at growth 1.4 put the outer edges ~110 km out. That
        # is not paranoia: a conductive layer over a resistive basement leaks
        # current over L ~ h rho2/rho1 (~10 km for 45 m of 20 on 5000 ohm m) and
        # the mixed BC is only right once the field is radial again. Measured
        # on that case: 8 km padding 4.3 %, 110 km 2.8 % at the two shortest
        # offsets and < 1.2 % elsewhere; ordinary contrasts < 0.9 %.
        self.nz, self.ny, self.dx, self.I = nz, ny, dx, current
        self.n_pad_side, self.n_pad_bottom = n_pad_side, n_pad_bottom
        z, self.zc = padded_axis(nz, dx, 0, n_pad_bottom, growth)
        y, self.yc = padded_axis(ny, dx, n_pad_side, n_pad_side, growth)
        self.grid = Grid2D(z, y)
        if src_cols is None:
            src_cols = np.round(np.linspace(0, ny - 1, 10)).astype(int)
        self.src_cols = np.asarray(src_cols, int)
        self.rec_y = self.grid.y[self.yc]                      # surface electrodes
        self.src_y = self.rec_y[self.src_cols]
        self.k, self.w, self.quad_err = wavenumber_quadrature(dx, n_k=n_k)
        idx = self.grid.index()
        self.src_nodes = idx[0, self.yc][self.src_cols]
        self._boundary_geometry()
        self.calibration = None
        self._calibrate()

    def _boundary_geometry(self):
        """Nodes, radial distance, cos(theta) and face length of the outer edges."""
        g = self.grid
        idx = g.index()
        yc = 0.5 * (self.rec_y[0] + self.rec_y[-1])
        nodes, r, cos, L = [], [], [], []
        # bottom face
        zb = g.z[-1]
        rr = np.hypot(g.y - yc, zb)
        nodes.append(idx[-1, :]); r.append(rr); cos.append(zb / rr); L.append(g.dy)
        # side faces
        for j in (0, -1):
            rr = np.hypot(g.y[j] - yc, g.z)
            nodes.append(idx[:, j]); r.append(rr); cos.append(np.abs(g.y[j] - yc) / rr); L.append(g.dz)
        self._bnd = [np.concatenate(v) for v in (nodes, r, cos, L)]

    def _mixed_bc_diagonal(self, k, sig):
        """Dey-Morrison outward-flux term, added to the diagonal of boundary nodes."""
        nodes, r, cos, L = self._bnd
        kr = k * r
        alpha = k * k1e(kr) / k0e(kr) * cos          # scaled Bessel ratio: no overflow at large kr
        a = sig.ravel()[nodes]
        diag = np.zeros(self.grid.n)
        np.add.at(diag, nodes, -a * alpha * L)          # corners get both faces
        return sp.diags(diag)

    # -- core solve ---------------------------------------------------------
    def _raw_potentials(self, sigma):
        """Uncalibrated surface potentials (n_src, ny) for node conductivities."""
        sig = pad_edge(np.asarray(sigma, float), 0, self.n_pad_bottom, self.n_pad_side, self.n_pad_side)
        rhs = np.zeros((self.grid.n, len(self.src_nodes)))
        rhs[self.src_nodes, np.arange(len(self.src_nodes))] = -0.5 * self.I
        acc = np.zeros((self.grid.n, len(self.src_nodes)))
        for k, w in zip(self.k, self.w):
            M = assemble(self.grid, sig, (k * k) * sig) + self._mixed_bc_diagonal(k, sig)
            acc += w * spla.splu(M.tocsc()).solve(rhs)
        phi = (2.0 / np.pi) * acc                               # (n, n_src)
        surf = phi.reshape(self.grid.nz, self.grid.ny, -1)[0, self.yc, :]   # (ny, n_src)
        return surf.T

    def _calibrate(self):
        """Geometric-factor correction from a unit half-space (rho = 1)."""
        num = self._raw_potentials(np.ones((self.nz, self.ny)))
        r = np.abs(self.rec_y[None, :] - self.src_y[:, None])
        with np.errstate(divide="ignore"):
            ana = self.I / (2.0 * np.pi * r)
        cal = np.ones_like(num)
        mask = r > 0
        cal[mask] = ana[mask] / num[mask]
        self.calibration = cal
        self.calibration_range = (cal[mask].min(), cal[mask].max())

    # -- public -------------------------------------------------------------
    def potentials(self, sigma):
        """Calibrated surface potentials (n_src, ny) in volts for I = current."""
        return self._raw_potentials(sigma) * self.calibration

    def apparent_resistivity(self, sigma):
        """Pole-pole rho_a (n_src, ny) in ohm m; r = 0 filled from the neighbour."""
        V = self.potentials(sigma)
        r = np.abs(self.rec_y[None, :] - self.src_y[:, None])
        rho_a = 2.0 * np.pi * r * V / self.I
        for s, c in enumerate(self.src_cols):
            nb = c + 1 if c + 1 < self.ny else c - 1
            rho_a[s, c] = rho_a[s, nb]
        return rho_a.astype(np.float32)


# --------------------------------------------------------------------------
# analytic references used by the tests
# --------------------------------------------------------------------------
def two_layer_pole_pole(r, rho1, rho2, h, n_terms=6000):
    """Pole-pole apparent resistivity over a layer of resistivity rho1 and
    thickness h on a half-space rho2 (image series; |K|^n < 1e-6 needs ~3500
    terms for a 1:500 contrast, hence the default)."""
    r = np.asarray(r, float).ravel()
    K = (rho2 - rho1) / (rho2 + rho1)
    n = np.arange(1, n_terms + 1)[:, None]
    series = (K ** n * r[None, :] / np.sqrt(r[None, :] ** 2 + (2 * n * h) ** 2)).sum(axis=0)
    return rho1 * (1.0 + 2.0 * series)
