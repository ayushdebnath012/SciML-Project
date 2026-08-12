"""Leapfrog FD solver for the forced 1D elastic wave equation.

Solves the CONSERVATIVE form with a separable source term

    rho(x) u_tt = d/dx [ E(x) u_x ] + s(x) w(t)

from a quiescent start (u = 0, u_t = 0), with first-order absorbing
boundaries discretised by the Mur scheme -- the same physics and the same
boundary treatment as wave/fd_solver.py, which this mirrors. The only
addition is the forcing term.

The unforced solver drops the source and is driven by the initial condition
instead; keeping the two separate leaves that arm untouched.
"""
import numpy as np


def ricker(t, f_peak, t_shift=None):
    """Ricker (Mexican-hat) wavelet, peak amplitude 1 at t = t_shift.

    t_shift defaults to 1.2 / f_peak, which puts w(0) at ~1e-5 so the start
    is quiescent to numerical precision.
    """
    t = np.asarray(t, dtype=np.float64)
    if t_shift is None:
        t_shift = 1.2 / f_peak
    a = (np.pi * f_peak * (t - t_shift)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def source_profile(x, x_src, width):
    """Narrow Gaussian standing in for a point source, peak amplitude 1."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-(((x - x_src) / width) ** 2))


def solve_wave_1d_source(E, rho, s_x, f_peak, amplitude,
                         x_limits=(-1.0, 1.0), t_limits=(0.0, 1.0),
                         nx_out=64, nt_out=64, CFL=0.35, t_shift=None):
    """Forced wave equation, quiescent start, Mur absorbing boundaries.

    E, rho, s_x are arrays on the nx_out output grid. Returns
    (x, t, u) with u of shape (nx_out, nt_out), plus the wavelet sampled on
    the output time grid.
    """
    x_min, x_max = x_limits
    t_min, t_max = t_limits
    x = np.linspace(x_min, x_max, nx_out)
    t_out = np.linspace(t_min, t_max, nt_out)
    dx = x[1] - x[0]
    Nx = nx_out - 1

    E = np.asarray(E, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    s_x = np.asarray(s_x, dtype=np.float64)
    if np.any(E <= 0) or np.any(rho <= 0):
        raise ValueError("E and rho must be strictly positive.")
    c = np.sqrt(E / rho)

    # Internal time stepping from the CFL condition, then interpolate out.
    dt_cfl = CFL * dx / c.max()
    Nt = int(np.ceil((t_max - t_min) / dt_cfl))
    dt = (t_max - t_min) / Nt
    t = np.linspace(t_min, t_max, Nt + 1)

    if t_shift is None:
        t_shift = 1.2 / f_peak

    # Harmonic-mean interface stiffness: preserves transmission/reflection
    # amplitudes across a jump in E.
    E_half = 2.0 * E[:-1] * E[1:] / (E[:-1] + E[1:])
    lam = dt ** 2 / (rho * dx ** 2)
    src_coeff = dt ** 2 / rho                      # multiplies s(x) w(t)
    beta_l = (c[0] * dt - dx) / (c[0] * dt + dx)   # Mur coefficients
    beta_r = (c[-1] * dt - dx) / (c[-1] * dt + dx)

    u = np.zeros((Nx + 1, Nt + 1))

    def flux_div(un):
        return E_half[1:] * (un[2:] - un[1:-1]) - E_half[:-1] * (un[1:-1] - un[:-2])

    # Quiescent start: u(x,0) = 0 and u_t(x,0) = 0, so the entire first step
    # comes from the source.
    w0 = amplitude * ricker(t[0], f_peak, t_shift)
    u[1:-1, 1] = 0.5 * lam[1:-1] * flux_div(u[:, 0]) + 0.5 * src_coeff[1:-1] * s_x[1:-1] * w0
    u[0, 1] = u[1, 0] + beta_l * (u[1, 1] - u[0, 0])
    u[Nx, 1] = u[Nx - 1, 0] + beta_r * (u[Nx - 1, 1] - u[Nx, 0])

    for n in range(1, Nt):
        wn = amplitude * ricker(t[n], f_peak, t_shift)
        u[1:-1, n + 1] = (2 * u[1:-1, n] - u[1:-1, n - 1]
                          + lam[1:-1] * flux_div(u[:, n])
                          + src_coeff[1:-1] * s_x[1:-1] * wn)
        u[0, n + 1] = u[1, n] + beta_l * (u[1, n + 1] - u[0, n])
        u[Nx, n + 1] = u[Nx - 1, n] + beta_r * (u[Nx - 1, n + 1] - u[Nx, n])

    # Interpolate the internal time grid onto the uniform output grid.
    u_out = np.empty((nx_out, nt_out), dtype=np.float64)
    for i in range(nx_out):
        u_out[i] = np.interp(t_out, t, u[i])

    w_out = amplitude * ricker(t_out, f_peak, t_shift)
    return x, t_out, u_out, w_out
