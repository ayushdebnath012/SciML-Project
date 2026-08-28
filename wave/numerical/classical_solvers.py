"""Three classical discretizations of rho u_tt = d/dx[E u_x] on a bounded
domain with first-order absorbing boundaries.

All three solve the same nondimensional problem the PINNs are trained on --
same domain, same Gaussian-derivative initial pulse, same zero initial
velocity, same radiation conditions -- so their errors and run times sit on
the same axes as the network results.

  fd2    second-order finite differences in flux form, harmonic-mean interface
         stiffness, Mur absorbing boundaries. This is the repository's
         reference solver (wave/fd_solver.py), reimplemented here in a form
         that returns only the requested output grid.
  fem_p1 linear finite elements, lumped mass, explicit central differences.
         Absorbing boundaries enter naturally as a rho*c dashpot in the weak
         form, which is why no Mur-style special case is needed.
  cheb   Chebyshev collocation in space on the velocity-stress system, RK4 in
         time, radiation boundaries imposed as a characteristic projection.
         Spectral in space, so its error is set by the time step and by how
         well the grid resolves the pulse and the tanh interface -- not by a
         polynomial order in dx.

Every solver returns u sampled on the caller's (x_out, t_out) grid, so the
error metric never depends on the solver's internal resolution.
"""
import numpy as np

from material_profiles import gaussian_derivative_ic


def _aligned_steps(nt_stability, t_out):
    """Round the step count up until the output times are hit exactly.

    `t_out` is uniform on [0, t_max], so any `nt` divisible by `len(t_out) - 1`
    lands a step on every output time. Doing this removes interpolation in time
    from the error budget entirely -- which matters, because a linear
    interpolation between stored steps is second-order accurate and would put a
    floor under the spectral solver well above its actual spatial error.

    Returns `(nt, stride)`: take `nt` steps, store every `stride`-th.
    """
    n_intervals = len(t_out) - 1
    dt_out = t_out[1] - t_out[0]
    if not np.allclose(np.diff(t_out), dt_out) or not np.isclose(t_out[0], 0.0):
        raise ValueError("t_out must be uniform and start at 0")
    stride = int(np.ceil(nt_stability / n_intervals))
    return stride * n_intervals, stride


def _bary_weights(x):
    """Barycentric weights for Chebyshev-Gauss-Lobatto nodes (ascending)."""
    n = len(x) - 1
    w = np.ones(n + 1)
    w[1::2] = -1.0
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def _bary_interp(x_nodes, w, values, x_query):
    """Barycentric interpolation -- exact for the collocation polynomial.

    Linear interpolation off a Chebyshev grid is second-order accurate and
    would hide the spectral convergence this solver exists to demonstrate.
    """
    num = np.zeros_like(x_query, dtype=float)
    den = np.zeros_like(x_query, dtype=float)
    exact = np.full(len(x_query), -1, dtype=int)
    for j, (xj, wj) in enumerate(zip(x_nodes, w)):
        diff = x_query - xj
        hit = diff == 0.0
        exact[hit] = j
        diff[hit] = 1.0                      # placeholder; overwritten below
        term = wj / diff
        num += term * values[j]
        den += term
    out = num / den
    on_node = exact >= 0
    out[on_node] = values[exact[on_node]]
    return out


def _resample_space(u_by_time, x_int, x_out, interp="linear", w=None):
    out = np.empty((len(x_out), u_by_time.shape[1]))
    for j in range(u_by_time.shape[1]):
        if interp == "bary":
            out[:, j] = _bary_interp(x_int, w, u_by_time[:, j], x_out.copy())
        else:
            out[:, j] = np.interp(x_out, x_int, u_by_time[:, j])
    return out


# ---------------------------------------------------------------- fd2 --------

def solve_fd2(material, x_out, t_out, nx, cfl=0.9, sigma_g=0.1, store_every=None):
    x_min, x_max = material.x_min, material.x_max
    dx = (x_max - x_min) / nx
    x = np.linspace(x_min, x_max, nx + 1)
    E, rho = material.E(x), material.rho(x)
    c = np.sqrt(E / rho)

    t_max = float(t_out[-1])
    nt, stride = _aligned_steps(t_max / (cfl * dx / c.max()), t_out)
    dt = t_max / nt

    # Harmonic mean at cell interfaces: the arithmetic mean gets the
    # transmission coefficient across a stiffness jump wrong.
    E_half = 2.0 * E[:-1] * E[1:] / (E[:-1] + E[1:])
    lam = dt ** 2 / (rho * dx ** 2)
    beta_l = (c[0] * dt - dx) / (c[0] * dt + dx)
    beta_r = (c[-1] * dt - dx) / (c[-1] * dt + dx)

    def flux_div(un):
        return E_half[1:] * (un[2:] - un[1:-1]) - E_half[:-1] * (un[1:-1] - un[:-2])

    u_prev = gaussian_derivative_ic(x, sigma_g)          # u^0
    u_cur = np.empty_like(u_prev)                        # u^1, zero initial velocity
    u_cur[1:-1] = u_prev[1:-1] + 0.5 * lam[1:-1] * flux_div(u_prev)
    u_cur[0] = u_prev[1] + beta_l * (u_cur[1] - u_prev[0])
    u_cur[-1] = u_prev[-2] + beta_r * (u_cur[-2] - u_prev[-1])

    snaps = [u_prev.copy()]                       # step 0 == t_out[0]
    if stride == 1:
        snaps.append(u_cur.copy())
    for n in range(1, nt):
        u_next = np.empty_like(u_cur)
        u_next[1:-1] = 2 * u_cur[1:-1] - u_prev[1:-1] + lam[1:-1] * flux_div(u_cur)
        u_next[0] = u_cur[1] + beta_l * (u_next[1] - u_cur[0])
        u_next[-1] = u_cur[-2] + beta_r * (u_next[-2] - u_cur[-1])
        u_prev, u_cur = u_cur, u_next
        if (n + 1) % stride == 0:
            snaps.append(u_cur.copy())

    return _resample_space(np.array(snaps).T, x, x_out)


# -------------------------------------------------------------- fem_p1 -------

def solve_fem_p1(material, x_out, t_out, nx, cfl=0.9, sigma_g=0.1, store_every=None):
    x_min, x_max = material.x_min, material.x_max
    h = (x_max - x_min) / nx
    x = np.linspace(x_min, x_max, nx + 1)
    E, rho = material.E(x), material.rho(x)
    c = np.sqrt(E / rho)

    # Two-point Gauss per element: exact for the linear basis, and it keeps
    # the smooth tanh interface from being aliased onto element midpoints.
    xm = 0.5 * (x[:-1] + x[1:])
    off = h / (2.0 * np.sqrt(3.0))
    E_e = 0.5 * (material.E(xm - off) + material.E(xm + off))
    rho_e = 0.5 * (material.rho(xm - off) + material.rho(xm + off))

    # Lumped mass (row sums of the consistent matrix) keeps the scheme explicit.
    m = np.zeros(nx + 1)
    np.add.at(m, np.arange(nx), rho_e * h / 2.0)
    np.add.at(m, np.arange(1, nx + 1), rho_e * h / 2.0)

    k_e = E_e / h                                   # element stiffness
    damp = np.zeros(nx + 1)                         # rho*c dashpot, boundaries only
    damp[0] = rho[0] * c[0]
    damp[-1] = rho[-1] * c[-1]

    def K_apply(u):
        flux = k_e * (u[1:] - u[:-1])               # per element
        out = np.zeros_like(u)
        out[:-1] -= flux
        out[1:] += flux
        return out

    t_max = float(t_out[-1])
    nt, stride = _aligned_steps(t_max / (cfl * h / c.max()), t_out)
    dt = t_max / nt
    a = m / dt ** 2 + damp / (2.0 * dt)             # coefficient of u^{n+1}
    b = m / dt ** 2 - damp / (2.0 * dt)             # coefficient of u^{n-1}

    u_prev = gaussian_derivative_ic(x, sigma_g)
    # First step from zero initial velocity: u^1 = u^0 + dt^2/2 * a^0, where
    # M a^0 = -K u^0 - C u_t^0 and u_t^0 = 0.
    u_cur = u_prev + 0.5 * dt ** 2 * (-K_apply(u_prev) / m)

    snaps = [u_prev.copy()]
    if stride == 1:
        snaps.append(u_cur.copy())
    for n in range(1, nt):
        u_next = (2.0 * m / dt ** 2 * u_cur - b * u_prev - K_apply(u_cur)) / a
        u_prev, u_cur = u_cur, u_next
        if (n + 1) % stride == 0:
            snaps.append(u_cur.copy())

    return _resample_space(np.array(snaps).T, x, x_out)


# ---------------------------------------------------------------- cheb -------

def _chebyshev_D(n):
    """Differentiation matrix and nodes on [-1, 1] (Trefethen's construction)."""
    if n == 0:
        return np.zeros((1, 1)), np.ones(1)
    xs = np.cos(np.pi * np.arange(n + 1) / n)
    ctmp = np.hstack([2.0, np.ones(n - 1), 2.0]) * (-1.0) ** np.arange(n + 1)
    X = np.tile(xs, (n + 1, 1)).T
    dX = X - X.T
    D = np.outer(ctmp, 1.0 / ctmp) / (dX + np.eye(n + 1))
    D -= np.diag(D.sum(axis=1))
    return D, xs


def solve_cheb(material, x_out, t_out, n, cfl=0.6, sigma_g=0.1, store_every=None):
    """Chebyshev collocation on the velocity-stress system, RK4 in time.

    The second-order form u_tt = (1/rho) D(E D u) is the obvious thing to
    collocate, but its absorbing boundary has no stable second-order update:
    advancing u at the end nodes from u^{n-1} decouples them from the interior
    and the corner modes grow without bound. Splitting into

        rho v_t = D sigma,      sigma_t = E D v,      u_t = v

    fixes that, because the radiation condition is then exactly "the incoming
    Riemann invariant is zero" and can be imposed as a projection at every
    Runge-Kutta stage. With Z = rho c,

        R- = v - sigma/Z   travels at +c   (incoming at the left  boundary)
        R+ = v + sigma/Z   travels at -c   (incoming at the right boundary)

    so at each end the outgoing invariant is advected with the interior
    derivative and the incoming one is held at zero.
    """
    x_min, x_max = material.x_min, material.x_max
    D_ref, xs = _chebyshev_D(n)
    # Trefethen's nodes run +1 -> -1; flip so x ascends and the map stays affine.
    xs = xs[::-1]
    D = D_ref[::-1, ::-1]
    half = (x_max - x_min) / 2.0
    x = x_min + (xs + 1.0) * half
    D = D / half

    E, rho = material.E(x), material.rho(x)
    c = np.sqrt(E / rho)
    Z = rho * c

    def rhs(state):
        u, v, sig = state
        du = v
        dv = (D @ sig) / rho
        dsig = E * (D @ v)

        # Characteristic projection at the two end nodes.
        Rp = v + sig / Z
        Rm = v - sig / Z
        dRp_dx = D @ Rp
        dRm_dx = D @ Rm
        for j, incoming_is_Rm in ((0, True), (-1, False)):
            if incoming_is_Rm:                       # left end: R- enters
                dRp = c[j] * dRp_dx[j]               # R+ leaves, R+_t = +c R+_x
                dRm = 0.0
            else:                                    # right end: R+ enters
                dRp = 0.0
                dRm = -c[j] * dRm_dx[j]              # R- leaves, R-_t = -c R-_x
            dv[j] = 0.5 * (dRp + dRm)
            dsig[j] = 0.5 * Z[j] * (dRp - dRm)
        return np.array([du, dv, dsig])

    # Chebyshev nodes cluster like 1/n^2 at the ends, which is what sets dt.
    dx_min = np.min(np.diff(x))
    t_max = float(t_out[-1])
    nt, stride = _aligned_steps(t_max / (cfl * dx_min / c.max()), t_out)
    dt = t_max / nt

    u0 = gaussian_derivative_ic(x, sigma_g)
    state = np.array([u0, np.zeros_like(u0), E * (D @ u0)])   # u_t = 0 at t = 0

    snaps = [u0.copy()]
    for step in range(nt):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if (step + 1) % stride == 0:
            snaps.append(state[0].copy())

    return _resample_space(np.array(snaps).T, x, x_out,
                           interp="bary", w=_bary_weights(x))


SOLVERS = {"fd2": solve_fd2, "fem_p1": solve_fem_p1, "cheb": solve_cheb}
