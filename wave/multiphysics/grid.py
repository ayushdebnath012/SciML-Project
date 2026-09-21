"""Padded tensor grids and the sparse operator shared by the DC and MT solvers.

Both electrical problems are of the form

    div( a grad u ) - m u = f        on a 2-D (z down, y lateral) grid

with piecewise-constant coefficients: a = sigma, m = k^2 sigma for DC;
a = 1, m = i w mu sigma (TE) or a = rho, m = i w mu (TM) for MT. The core
70 x 70 model sits on a uniform 10 m grid; outside it the grid stretches
geometrically so far boundaries are kilometres away for a few dozen extra
nodes.

Discretisation is node-based finite volume: every node owns a control volume
bounded by the midpoints to its neighbours, edge coefficients are harmonic
means of the two node values (series composition of the half-cells), and the
equation for a node is the *integrated* balance

    sum_edges c_e (u_nb - u) - m A u = b_int,   b_int = integral of f over the CV,

so a point source of strength q at a node is simply b_int = q. Boundaries
with no flux term are natural Neumann; Dirichlet/Robin rows are overwritten.
"""
import numpy as np
import scipy.sparse as sp


def stretched_spacings(h0, n, growth):
    """n spacings growing geometrically from h0*growth."""
    return h0 * growth ** np.arange(1, n + 1)


def control_widths(h):
    """Control-volume widths of the nodes of an axis with spacings h."""
    w = np.empty(len(h) + 1)
    w[0], w[-1] = h[0] / 2, h[-1] / 2
    w[1:-1] = (h[:-1] + h[1:]) / 2
    return w


class Grid2D:
    """Tensor grid: rows = z (depth, positive down), cols = y (lateral)."""

    def __init__(self, z, y):
        self.z, self.y = np.asarray(z, float), np.asarray(y, float)
        self.nz, self.ny = len(self.z), len(self.y)
        self.hz, self.hy = np.diff(self.z), np.diff(self.y)
        self.dz, self.dy = control_widths(self.hz), control_widths(self.hy)

    @property
    def shape(self):
        return (self.nz, self.ny)

    @property
    def n(self):
        return self.nz * self.ny

    def area(self):
        return np.outer(self.dz, self.dy)

    def index(self):
        return np.arange(self.n).reshape(self.shape)


def padded_axis(n_core, h, n_lo, n_hi, growth, origin=0.0):
    """Coordinates of n_core uniform nodes with stretched padding either side.

    Returns (coords, core_slice). The core nodes sit at origin + k*h.
    """
    lo = stretched_spacings(h, n_lo, growth)[::-1]
    hi = stretched_spacings(h, n_hi, growth)
    spacings = np.concatenate([lo, np.full(n_core - 1, h), hi])
    coords = origin - lo.sum() + np.concatenate([[0.0], np.cumsum(spacings)])
    return coords, slice(n_lo, n_lo + n_core)


def pad_edge(field, n_top, n_bottom, n_left, n_right):
    """Replicate the outermost rows/cols of a (nz, ny) field into the padding."""
    return np.pad(field, ((n_top, n_bottom), (n_left, n_right)), mode="edge")


def _harmonic(a, b):
    return 2.0 * a * b / (a + b)


def assemble(grid, a, m):
    """Sparse integrated-balance matrix for  div(a grad u) - m u.

    a, m : node arrays of shape grid.shape (m may be complex). Natural Neumann
    on every boundary. Returns a CSR matrix; convert to CSC before splu.
    """
    nz, ny = grid.shape
    idx = grid.index()
    a = np.broadcast_to(np.asarray(a, float), (nz, ny))
    m = np.broadcast_to(m, (nz, ny))
    dtype = complex if np.iscomplexobj(m) else float

    # integrated edge conductances: coefficient x transverse CV width / distance
    cy = _harmonic(a[:, :-1], a[:, 1:]) * grid.dz[:, None] / grid.hy[None, :]
    cz = _harmonic(a[:-1, :], a[1:, :]) * grid.dy[None, :] / grid.hz[:, None]

    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r.ravel()); cols.append(c.ravel()); vals.append(v.ravel())

    p, q = idx[:, :-1], idx[:, 1:]
    add(p, q, cy); add(q, p, cy); add(p, p, -cy); add(q, q, -cy)
    p, q = idx[:-1, :], idx[1:, :]
    add(p, q, cz); add(q, p, cz); add(p, p, -cz); add(q, q, -cz)
    add(idx, idx, -(m * grid.area()))

    M = sp.coo_matrix((np.concatenate(vals).astype(dtype),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(grid.n, grid.n))
    return M.tocsr()


def set_dirichlet(M, nodes, value=None):
    """Overwrite rows `nodes` with identity. Returns (M_csc, rhs_fill) where
    rhs_fill is an array of zeros with `value` at the Dirichlet nodes (or None)."""
    M = M.tolil()
    nodes = np.asarray(nodes).ravel()
    for n in nodes:
        M.rows[n] = [int(n)]
        M.data[n] = [1.0]
    rhs = None
    if value is not None:
        rhs = np.zeros(M.shape[0], dtype=M.dtype)
        rhs[nodes] = value
    return M.tocsc(), rhs


def set_robin_bottom(M, grid, gamma):
    """Bottom-row rows become  du/dz + gamma u = 0  with a second-order one-sided
    derivative (nodes N, N-1, N-2 on the non-uniform grid).

    gamma: per-column array (ny,), the local decay constant sqrt(i w mu sigma)
    of the underlying half-space. Exact for a 1-D basement, which is what the
    replicated bottom row of padding represents. First-order was measured at
    1.7-3 % on a half-space at 5 Hz; this brings it under 0.5 %.
    """
    M = M.tolil()
    idx = grid.index()
    h1, h2 = grid.hz[-1], grid.hz[-2]
    c0 = (2 * h1 + h2) / (h1 * (h1 + h2))
    c1 = -(h1 + h2) / (h1 * h2)
    c2 = h1 / (h2 * (h1 + h2))
    for j in range(grid.ny):
        n, n1, n2 = int(idx[-1, j]), int(idx[-2, j]), int(idx[-3, j])
        cols = sorted([n2, n1, n])
        vals = {n: c0 + gamma[j], n1: c1, n2: c2}
        M.rows[n] = cols
        M.data[n] = [vals[c] for c in cols]
    return M.tocsc()
