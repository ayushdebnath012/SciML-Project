"""Problem definition for the operator-learning arms: IC, materials, FD case setup.

Extracted verbatim from run_fno_baseline.py so that dataset generation does not
drag in torch. Generating data is pure NumPy + the FD solver; keeping it
importable without torch lets the generators run on machines that only ever
render or prepare data, and keeps one definition of the initial condition and
the input-channel layout shared by every arm.

run_fno_baseline.py re-exports these names, so existing callers that reach for
`rfb.gaussian_derivative_ic` and friends keep working unchanged.
"""
import numpy as np

from fd_solver import solve_wave_1d


def gaussian_derivative_ic(x, sigma_g: float = 0.1, x0: float = 0.0):
    x = np.asarray(x, dtype=np.float64)
    f = np.exp(-0.5 * ((x - x0) / sigma_g) ** 2)
    dfdx = -(x - x0) / sigma_g ** 2 * f
    return dfdx / (np.max(np.abs(dfdx)) + 1e-12)


def sample_material_profile(rng: np.random.Generator, x: np.ndarray):
    kind = rng.choice(["homogeneous", "two_layer", "layered", "smooth"])
    rho = np.ones_like(x, dtype=np.float64)

    if kind == "homogeneous":
        E = np.full_like(x, rng.uniform(0.7, 2.2), dtype=np.float64)
    elif kind == "two_layer":
        left = rng.uniform(0.7, 1.5)
        right = rng.uniform(1.0, 2.5)
        boundary = rng.uniform(-0.45, 0.45)
        width = rng.uniform(0.015, 0.06)
        alpha = 0.5 * (1.0 + np.tanh((x - boundary) / width))
        E = left * (1.0 - alpha) + right * alpha
    elif kind == "layered":
        n_layers = int(rng.integers(3, 8))
        values = rng.uniform(0.7, 2.5, size=n_layers)
        boundaries = np.linspace(x.min(), x.max(), n_layers + 1)[1:-1]
        boundaries += rng.normal(scale=0.04, size=boundaries.shape)
        width = rng.uniform(0.015, 0.05)
        E = np.full_like(x, values[0], dtype=np.float64)
        for value, boundary in zip(values[1:], boundaries):
            alpha = 0.5 * (1.0 + np.tanh((x - boundary) / width))
            E = E * (1.0 - alpha) + value * alpha
    else:
        E = np.full_like(x, rng.uniform(0.9, 1.4), dtype=np.float64)
        for freq in (1, 2, 3):
            amp = rng.uniform(-0.18, 0.18)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            E += amp * np.sin(freq * np.pi * (x + 1.0) + phase)
        E = np.clip(E, 0.55, 2.5)

    return E.astype(np.float64), rho, str(kind)


def solve_case(x_grid: np.ndarray, t_grid: np.ndarray, E: np.ndarray,
               rho: np.ndarray, sigma_g: float, x0: float, cfl: float):
    def f_fn(x_val):
        return gaussian_derivative_ic(x_val, sigma_g=sigma_g, x0=x0)

    def g_fn(x_val):
        return np.zeros_like(np.asarray(x_val, dtype=np.float64))

    def E_fn(x_val):
        return np.interp(np.asarray(x_val, dtype=np.float64), x_grid, E)

    def rho_fn(x_val):
        return np.interp(np.asarray(x_val, dtype=np.float64), x_grid, rho)

    x_fd, t_fd, u_fd = solve_wave_1d(
        f_fn,
        g_fn,
        E_fn,
        rho_fn,
        x_limits=(float(x_grid[0]), float(x_grid[-1])),
        t_limits=(float(t_grid[0]), float(t_grid[-1])),
        Nx=len(x_grid) - 1,
        CFL=cfl,
    )

    if len(t_fd) == len(t_grid) and np.allclose(t_fd, t_grid):
        return u_fd.astype(np.float32)

    u_interp = np.empty((len(x_grid), len(t_grid)), dtype=np.float32)
    for i in range(len(x_grid)):
        u_interp[i] = np.interp(t_grid, t_fd, u_fd[i]).astype(np.float32)
    return u_interp


def solve_case_refined(x_grid, t_grid, E, rho, sigma_g, x0, cfl, refine):
    """FD solve on a `refine`x finer grid, sampled back onto (x_grid, t_grid).

    E and rho are the coarse profiles the operator sees; they are interpolated
    up to the fine grid, so the fine solve adds accuracy without adding any
    information the network is not given. The fine grid contains every coarse
    node exactly (stride `refine`), so the spatial restriction is a subsample
    rather than another interpolation.

    `refine=1` reduces to `solve_case` and is NOT converged: measured against a
    refine=32 reference it carries 13.4 % rel L2 on Marmousi columns and 10.4 %
    on the synthetic layered/two_layer profiles. refine=8 brings that to 0.32 %.
    """
    nx = len(x_grid)
    nx_fine = (nx - 1) * refine

    def f_fn(xv):
        return gaussian_derivative_ic(xv, sigma_g=sigma_g, x0=x0)

    def g_fn(xv):
        return np.zeros_like(np.asarray(xv, dtype=np.float64))

    def E_fn(xv):
        return np.interp(np.asarray(xv, dtype=np.float64), x_grid, E)

    def rho_fn(xv):
        return np.interp(np.asarray(xv, dtype=np.float64), x_grid, rho)

    _, t_fd, u_fd = solve_wave_1d(
        f_fn, g_fn, E_fn, rho_fn,
        x_limits=(float(x_grid[0]), float(x_grid[-1])),
        t_limits=(float(t_grid[0]), float(t_grid[-1])),
        Nx=nx_fine, CFL=cfl,
    )

    u_coarse = u_fd[::refine, :]
    if u_coarse.shape[0] != nx:
        raise RuntimeError(
            f"spatial restriction gave {u_coarse.shape[0]} rows, expected {nx}")

    out = np.empty((nx, len(t_grid)), dtype=np.float32)
    for i in range(nx):
        out[i] = np.interp(t_grid, t_fd, u_coarse[i]).astype(np.float32)
    return out


def make_input_tensor(x_grid: np.ndarray, t_grid: np.ndarray, E: np.ndarray,
                      rho: np.ndarray, sigma_g: float, x0: float):
    nx, nt = len(x_grid), len(t_grid)
    x_mesh = np.repeat(x_grid[:, None], nt, axis=1)
    t_mesh = np.repeat(t_grid[None, :], nx, axis=0)
    g = gaussian_derivative_ic(x_grid, sigma_g=sigma_g, x0=x0)
    channels = np.stack(
        [
            np.repeat(E[:, None], nt, axis=1),
            np.repeat(rho[:, None], nt, axis=1),
            np.repeat(g[:, None], nt, axis=1),
            x_mesh,
            t_mesh,
        ],
        axis=0,
    )
    return channels.astype(np.float32)
