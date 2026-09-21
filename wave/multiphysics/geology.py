"""Latent layered geology and the rock-physics links that paint every map from it.

The point of the dataset is that velocity, density and conductivity are *the
same subsurface* seen through different physics. So nothing is sampled per
property: a sample is a stack of layers (OpenFWI's FlatVel / CurveVel
geometry on the 70 x 70, 10 m grid), each layer carries a lithology
(porosity, matrix velocity, pore-fluid resistivity, water saturation), and the
maps are derived:

    velocity      Raymer-Hunt-Gardner   v   = (1 - phi)^2 v_matrix + phi v_fluid
    density       Gardner               rho = 310 v^0.25                  (kg/m^3)
    resistivity   Archie                rho_e = a rho_w phi^-m S_w^-n      (ohm m)
    conductivity                        sigma = 1 / rho_e                  (S/m)

The load-bearing rule: the maps are correlated but not deterministic functions
of each other. Fluid resistivity, saturation, cementation exponent and matrix
velocity are drawn per layer, and Archie gets a log-normal scatter, so a
network cannot get sigma from v with a lookup table -- which is exactly the
situation joint inversion exists for.

`properties_from_velocity` runs the same links backwards from a given
piecewise-constant velocity map, so real OpenFWI models (with their real
seismic gathers) can be given conductivity maps by the same recipe.
"""
import numpy as np

V_FLUID = 1500.0          # m/s, water
V_MIN, V_MAX = 1500.0, 4500.0       # OpenFWI label range
RHO_E_MIN, RHO_E_MAX = 2.0, 5000.0  # ohm m, keeps skin depths inside the modelled band
PHI_MIN, PHI_MAX = 0.03, 0.50


# --------------------------------------------------------------------------
# rock physics
# --------------------------------------------------------------------------
def rhg_velocity(phi, v_matrix, v_fluid=V_FLUID):
    return (1.0 - phi) ** 2 * v_matrix + phi * v_fluid


def rhg_porosity(v, v_matrix, v_fluid=V_FLUID):
    """Invert Raymer-Hunt-Gardner for porosity (smaller root of the quadratic)."""
    b = 2.0 * v_matrix - v_fluid
    disc = np.maximum(b * b - 4.0 * v_matrix * (v_matrix - v), 0.0)
    return np.clip((b - np.sqrt(disc)) / (2.0 * v_matrix), PHI_MIN, PHI_MAX)


def gardner_density(v):
    return 310.0 * v ** 0.25


def archie_resistivity(phi, rho_w, s_w, m, n=2.0, a=1.0):
    return a * rho_w * phi ** (-m) * s_w ** (-n)


# --------------------------------------------------------------------------
# layer geometry (OpenFWI FlatVel / CurveVel style)
# --------------------------------------------------------------------------
def sample_interfaces(rng, nz, ny, n_layers, style):
    """Depths (in cells) of the n_layers-1 interfaces at every column, (n_int, ny).

    'flat'  : horizontal layers, small random tilt.
    'curve' : a shared smooth undulation with per-interface amplitude, as in
              CurveVel, plus tilt. Interfaces never cross (sorted per column).
    """
    n_int = n_layers - 1
    y = np.linspace(0.0, 1.0, ny)
    base = np.sort(rng.uniform(0.08, 0.92, n_int)) * nz
    # enforce a minimum thickness of 3 cells by spreading crowded interfaces
    for _ in range(5):
        gaps = np.diff(np.concatenate([[0.0], base, [float(nz)]]))
        if gaps.min() >= 3.0:
            break
        base = np.cumsum(np.maximum(gaps, 3.0))[:-1]
        base *= (nz - 3.0) / max(base[-1] + 3.0, nz - 3.0) if base[-1] > nz - 3.0 else 1.0
    tilt = rng.uniform(-0.08, 0.08, n_int) * nz
    depths = base[:, None] + tilt[:, None] * (y[None, :] - 0.5)
    if style == "curve":
        k = rng.uniform(0.7, 2.0)                  # periods across the section
        phase = rng.uniform(0, 2 * np.pi)
        shape = np.sin(2 * np.pi * k * y + phase)
        shape += 0.35 * np.sin(2 * np.pi * (2.3 * k) * y + rng.uniform(0, 2 * np.pi))
        shape /= np.abs(shape).max()
        amp = rng.uniform(0.03, 0.14, n_int) * nz
        depths = depths + amp[:, None] * shape[None, :]
    depths = np.sort(np.clip(depths, 1.0, nz - 1.0), axis=0)
    return depths


def layer_ids(depths, nz, ny):
    """Integer layer index per cell from interface depths (n_int, ny)."""
    z = np.arange(nz)[:, None, None] + 0.5           # cell centres
    return (z >= depths.T[None, :, :]).sum(axis=2).astype(np.int8)  # (nz, ny)


# --------------------------------------------------------------------------
# per-layer lithology
# --------------------------------------------------------------------------
def sample_lithology(rng, n_layers, layer_mid_depth_m):
    """Per-layer parameters. Porosity compacts with depth with noise so
    velocity broadly increases downward but low-velocity layers happen."""
    phi0 = rng.uniform(0.30, 0.50)
    L = rng.uniform(800.0, 2500.0)
    phi = phi0 * np.exp(-layer_mid_depth_m / L) + rng.normal(0.0, 0.05, n_layers)
    phi = np.clip(phi, PHI_MIN, PHI_MAX)
    v_matrix = rng.uniform(3200.0, 5500.0, n_layers)
    rho_w = 10.0 ** rng.uniform(np.log10(0.3), np.log10(10.0), n_layers)   # brine .. fresh
    s_w = np.where(rng.random(n_layers) < 0.2, rng.uniform(0.5, 0.9, n_layers), 1.0)
    m = rng.uniform(1.7, 2.3, n_layers)
    scatter = np.exp(rng.normal(0.0, 0.25, n_layers))
    return dict(phi=phi, v_matrix=v_matrix, rho_w=rho_w, s_w=s_w, m=m, scatter=scatter)


def maps_from_lithology(ids, lith):
    """Paint the per-layer parameters onto the grid and apply the links."""
    phi = lith["phi"][ids]
    v = np.clip(rhg_velocity(phi, lith["v_matrix"][ids]), V_MIN, V_MAX)
    rho_e = archie_resistivity(phi, lith["rho_w"][ids], lith["s_w"][ids], lith["m"][ids]) * lith["scatter"][ids]
    rho_e = np.clip(rho_e, RHO_E_MIN, RHO_E_MAX)
    return dict(velocity=v.astype(np.float32),
                density=gardner_density(v).astype(np.float32),
                resistivity=rho_e.astype(np.float32),
                conductivity=(1.0 / rho_e).astype(np.float32),
                porosity=phi.astype(np.float32),
                layer_id=ids)


def sample_geology(rng, nz=70, ny=70, dx=10.0, n_layers=None, style=None):
    """One synthetic sample: dict of (nz, ny) maps + the lithology table."""
    if style is None:
        style = "curve" if rng.random() < 0.5 else "flat"
    if n_layers is None:
        n_layers = int(rng.integers(3, 8))
    depths = sample_interfaces(rng, nz, ny, n_layers, style)
    ids = layer_ids(depths, nz, ny)
    edges = np.concatenate([[0.0], depths.mean(axis=1), [float(nz)]])
    mid_depth_m = 0.5 * (edges[:-1] + edges[1:]) * dx
    lith = sample_lithology(rng, n_layers, mid_depth_m)
    out = maps_from_lithology(ids, lith)
    out["style"] = style
    out["n_layers"] = n_layers
    out["lithology"] = lith
    return out


def properties_from_velocity(v, rng):
    """Derive density / resistivity / conductivity maps from a piecewise-constant
    velocity map (e.g. a real OpenFWI model) with the same rock-physics recipe.

    Each distinct velocity value is treated as one lithology: a matrix velocity
    is drawn above it, porosity inverted from RHG, then Archie with per-layer
    fluid parameters and scatter.
    """
    v = np.asarray(v, np.float64)
    vals, inv = np.unique(v, return_inverse=True)
    inv = inv.reshape(v.shape)
    n = len(vals)
    v_matrix = np.maximum(vals + 300.0, rng.uniform(3200.0, 5500.0, n))
    phi = rhg_porosity(vals, v_matrix)
    lith = sample_lithology(rng, n, np.zeros(n))    # only the fluid/exponent draws are used
    lith["phi"], lith["v_matrix"] = phi, v_matrix
    out = maps_from_lithology(inv.astype(np.int8), lith)
    out["velocity"] = v.astype(np.float32)          # keep the given map exactly
    out["density"] = gardner_density(v).astype(np.float32)
    out["lithology"] = lith
    return out
