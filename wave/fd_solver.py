import numpy as np


def _coeff_array(val, x):
    """Resolve a coefficient (callable / scalar / array) to an array on grid x."""
    if callable(val):
        out = np.asarray(val(x), dtype=float)
        if out.shape != x.shape:
            out = np.array([float(val(xi)) for xi in x])
        return out
    if np.isscalar(val):
        return np.full(x.shape, float(val))
    return np.asarray(val, dtype=float)


def solve_wave_1d(f, g, E_val, rho_val, x_limits=(0.0, 1.0), t_limits=(0.0, 1.0),
                  Nx=200, CFL=0.9):
    """
    Solves the 1D elastic wave equation in CONSERVATIVE form:
        rho(x) u_tt = d/dx [ E(x) u_x ]
    using an explicit second-order leapfrog scheme with a flux-form stencil.

    The flux form uses harmonic-mean interface stiffness
        E_{i+1/2} = 2 E_i E_{i+1} / (E_i + E_{i+1})
    which preserves correct transmission/reflection behaviour across material
    interfaces (a plain  u_tt = c(x)^2 u_xx  stencil drops the E_x u_x term
    and gets layered media wrong).

    Boundary conditions: first-order Absorbing Boundary Conditions (ABCs),
    matching the PINN training objective (u_t - c*u_x = 0 left, u_t + c*u_x = 0 right),
    with the local sound speed c = sqrt(E/rho) at each boundary node.
    Discretised with the Mur scheme
        u_0^{n+1} = u_1^n + (c dt - dx)/(c dt + dx) * (u_1^{n+1} - u_0^n)
    (and mirrored on the right). NOTE: the seemingly natural centred-time /
    one-sided-space discretisation of the ABC is unconditionally unstable —
    round-off grows exponentially from the boundary — so Mur is required.

    Parameters:
    -----------
    f : callable
        Initial displacement function: f(x) = u(x, 0). Must accept an array.
    g : callable
        Initial velocity function: g(x) = u_t(x, 0). May return a scalar.
    E_val : float or callable or numpy array
        Stiffness profile E(x) on the (nondimensional) grid.
    rho_val : float or callable or numpy array
        Density profile rho(x) on the (nondimensional) grid.
    x_limits : tuple (float, float), optional
        Domain bounds in space (x_min, x_max).
    t_limits : tuple (float, float), optional
        Domain bounds in time (t_min, t_max).
    Nx : int, optional
        Number of spatial grid subdivisions. Default is 200.
    CFL : float, optional
        Courant-Friedrichs-Lewy stability safety factor. Default is 0.9.

    Returns:
    --------
    x : numpy array of shape (Nx + 1,)
        Spatial grid coordinates.
    t : numpy array of shape (Nt + 1,)
        Time step coordinates.
    u : numpy array of shape (Nx + 1, Nt + 1)
        High-precision numerical solution of displacement.
    """
    x_min, x_max = x_limits
    t_min, t_max = t_limits

    dx = (x_max - x_min) / Nx
    x = np.linspace(x_min, x_max, Nx + 1)

    # 1. Resolve material coefficients at each grid coordinate
    E = _coeff_array(E_val, x)
    rho = _coeff_array(rho_val, x)
    if np.any(E <= 0) or np.any(rho <= 0):
        raise ValueError("Stiffness E(x) and density rho(x) must be strictly positive.")
    c = np.sqrt(E / rho)

    # 2. Determine time step size dt satisfying the CFL condition
    dt_cfl = CFL * dx / c.max()
    T = t_max - t_min
    Nt = int(np.ceil(T / dt_cfl))
    dt = T / Nt
    t = np.linspace(t_min, t_max, Nt + 1)

    # 3. Interface stiffness (harmonic mean) and update coefficients
    E_half = 2.0 * E[:-1] * E[1:] / (E[:-1] + E[1:])     # E_{i+1/2}, shape (Nx,)
    lam = dt ** 2 / (rho * dx ** 2)                       # dt^2 / (rho_i dx^2)
    # Mur ABC coefficients at the two boundary nodes
    beta_l = (c[0]  * dt - dx) / (c[0]  * dt + dx)
    beta_r = (c[-1] * dt - dx) / (c[-1] * dt + dx)

    # Initialize solution grid
    u = np.zeros((Nx + 1, Nt + 1))

    # 4. Apply Initial Condition at t = 0 (Displacement u(x, 0) = f(x))
    u[:, 0] = f(x)

    # Evaluate initial velocity function at each coordinate
    g_arr = np.array([float(g(xi)) for xi in x])

    def flux_div(un):
        """d/dx [E u_x] at interior nodes, times dx^2:  E_{i+1/2}(u_{i+1}-u_i) - E_{i-1/2}(u_i-u_{i-1})."""
        return E_half[1:] * (un[2:] - un[1:-1]) - E_half[:-1] * (un[1:-1] - un[:-2])

    # 5. Compute first time step (n = 1) using central difference for initial velocity
    u[1:-1, 1] = u[1:-1, 0] + dt * g_arr[1:-1] + 0.5 * lam[1:-1] * flux_div(u[:, 0])

    # Mur first-order ABC at step 1 (uses the freshly computed interior values)
    u[0,  1] = u[1,    0] + beta_l * (u[1,    1] - u[0,  0])
    u[Nx, 1] = u[Nx-1, 0] + beta_r * (u[Nx-1, 1] - u[Nx, 0])

    # 6. Leapfrog time integration loop (n = 1, ..., Nt - 1), vectorized in space
    for n in range(1, Nt):
        u[1:-1, n + 1] = 2 * u[1:-1, n] - u[1:-1, n - 1] + lam[1:-1] * flux_div(u[:, n])

        # Mur first-order ABC (interior at n+1 already available)
        u[0,  n + 1] = u[1,    n] + beta_l * (u[1,    n + 1] - u[0,  n])
        u[Nx, n + 1] = u[Nx-1, n] + beta_r * (u[Nx-1, n + 1] - u[Nx, n])

    return x, t, u
