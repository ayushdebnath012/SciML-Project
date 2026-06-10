import numpy as np

def solve_wave_1d(f, g, c_val, x_limits=(0.0, 1.0), t_limits=(0.0, 1.0), Nx=200, CFL=0.9):
    """
    Solves the 1D Wave Equation: u_tt = c(x)^2 * u_xx
    using the explicit second-order central difference (Leapfrog) method.

    Boundary conditions: First-order Absorbing Boundary Conditions (ABCs),
    matching the PINN training objective (u_t - c*u_x = 0 left, u_t + c*u_x = 0 right).
    Discretised as:
        Step 1  : u[0,1]   = u[0,0]   + r[0] *(u[1,0]  -u[0,0])
                  u[Nx,1]  = u[Nx,0]  - r[Nx]*(u[Nx,0] -u[Nx-1,0])
        Step n≥1: u[0,n+1] = u[0,n-1] + 2*r[0] *(u[1,n]  -u[0,n])
                  u[Nx,n+1]= u[Nx,n-1]- 2*r[Nx]*(u[Nx,n] -u[Nx-1,n])

    Parameters:
    -----------
    f : callable
        Initial displacement function: f(x) = u(x, 0)
    g : callable
        Initial velocity function: g(x) = u_t(x, 0)
    c_val : float or callable or numpy array
        Wave speed propagation profile.
    x_limits : tuple (float, float), optional
        Domain bounds in space (x_min, x_max). Default is (0.0, 1.0).
    t_limits : tuple (float, float), optional
        Domain bounds in time (t_min, t_max). Default is (0.0, 1.0).
    Nx : int, optional
        Number of spatial grid subdivisions. Default is 200.
    CFL : float, optional
        Courant-Friedrichs-Lewy stability parameter safety factor. Default is 0.9.

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
    
    # 1. Resolve material speed c(x) at each grid coordinate
    if callable(c_val):
        c = np.array([c_val(xi) for xi in x])
    elif isinstance(c_val, (int, float)):
        c = np.full(Nx + 1, float(c_val))
    else:
        c = np.asarray(c_val)
        
    c_max = np.max(c)
    if c_max <= 0:
        raise ValueError("Wave propagation speed must be strictly positive.")
        
    # 2. Determine time step size dt satisfying CFL condition
    dt_cfl = CFL * dx / c_max
    T = t_max - t_min
    Nt = int(np.ceil(T / dt_cfl))
    dt = T / Nt
    t = np.linspace(t_min, t_max, Nt + 1)
    
    # 3. Calculate localized Courant numbers
    r = c * dt / dx
    r_sq = r ** 2

    # Initialize solution grid
    u = np.zeros((Nx + 1, Nt + 1))

    # 4. Apply Initial Condition at t = 0 (Displacement u(x, 0) = f(x))
    u[:, 0] = f(x)

    # Evaluate initial velocity function at each coordinate
    g_arr = np.array([g(xi) for xi in x])

    # 5. Compute first time step (n = 1) using central difference for initial velocity
    for i in range(1, Nx):
        u[i, 1] = u[i, 0] + dt * g_arr[i] + 0.5 * r_sq[i] * (u[i + 1, 0] - 2 * u[i, 0] + u[i - 1, 0])

    # First-order ABC at step 1 (forward-difference in time)
    u[0,  1] = u[0,  0] + r[0]  * (u[1,    0] - u[0,    0])
    u[Nx, 1] = u[Nx, 0] - r[Nx] * (u[Nx,   0] - u[Nx-1, 0])

    # 6. Leapfrog time integration loop (n = 1, ..., Nt - 1)
    for n in range(1, Nt):
        for i in range(1, Nx):
            u[i, n + 1] = 2 * u[i, n] - u[i, n - 1] + r_sq[i] * (u[i + 1, n] - 2 * u[i, n] + u[i - 1, n])

        # First-order ABC (centred-difference in time)
        u[0,  n + 1] = u[0,  n - 1] + 2 * r[0]  * (u[1,    n] - u[0,    n])
        u[Nx, n + 1] = u[Nx, n - 1] - 2 * r[Nx] * (u[Nx,   n] - u[Nx-1, n])

    return x, t, u
