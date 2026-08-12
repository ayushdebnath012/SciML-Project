"""Initial- and boundary-condition residuals for predicted wavefields.

The operators here are trained by supervised learning -- no PDE residual in
the loss -- so these are *diagnostics*, not training objectives. They answer a
question the field-wise L2 error cannot: does the prediction actually respect
the physics it was supposed to have learned?

Every quantity is also computed for the finite-difference reference. That is
essential: the reference satisfies the initial and boundary conditions only to
discretisation accuracy on the coarse 64x80 output grid, so its residual is
the achievable floor. A model result is only meaningful relative to it.

Conventions for the forced arm:
  IC   quiescent start,  u(x,0) = 0  and  u_t(x,0) = 0
  BC   first-order absorbing,  u_t - c u_x = 0 at x_min,
                               u_t + c u_x = 0 at x_max
"""
import numpy as np


def _dudt(u, dt):
    """d/dt on the (nx, nt) grid: central inside, second-order one-sided ends."""
    ut = np.empty_like(u, dtype=np.float64)
    ut[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dt)
    ut[:, 0] = (-3.0 * u[:, 0] + 4.0 * u[:, 1] - u[:, 2]) / (2.0 * dt)
    ut[:, -1] = (3.0 * u[:, -1] - 4.0 * u[:, -2] + u[:, -3]) / (2.0 * dt)
    return ut


def _dudx_edges(u, dx):
    """d/dx at the two spatial boundaries, second-order one-sided."""
    left = (-3.0 * u[0] + 4.0 * u[1] - u[2]) / (2.0 * dx)
    right = (3.0 * u[-1] - 4.0 * u[-2] + u[-3]) / (2.0 * dx)
    return left, right


def ic_residuals(u, dt):
    """Quiescent-start residuals: RMS over x of u(x,0) and of u_t(x,0)."""
    u = np.asarray(u, dtype=np.float64)
    ut = _dudt(u, dt)
    return {
        "ic_displacement_rms": float(np.sqrt(np.mean(u[:, 0] ** 2))),
        "ic_velocity_rms": float(np.sqrt(np.mean(ut[:, 0] ** 2))),
    }


def bc_residuals(u, c, dx, dt):
    """Absorbing-boundary residuals at both edges.

    Returns the absolute RMS of the residual over time, and a scale-free
    relative version: the residual divided by the typical size of the terms
    that are supposed to cancel. Relative ~0 means the condition holds;
    relative ~1 means the two terms are not cancelling at all.
    """
    u = np.asarray(u, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    ut = _dudt(u, dt)
    ux_l, ux_r = _dudx_edges(u, dx)

    r_left = ut[0] - c[0] * ux_l          # u_t - c u_x = 0
    r_right = ut[-1] + c[-1] * ux_r       # u_t + c u_x = 0

    def rel(residual, a, b):
        scale = np.sqrt(np.mean(a ** 2)) + np.sqrt(np.mean(b ** 2))
        return float(np.sqrt(np.mean(residual ** 2)) / max(scale, 1e-12))

    return {
        "bc_left_rms": float(np.sqrt(np.mean(r_left ** 2))),
        "bc_right_rms": float(np.sqrt(np.mean(r_right ** 2))),
        "bc_left_rel": rel(r_left, ut[0], c[0] * ux_l),
        "bc_right_rel": rel(r_right, ut[-1], c[-1] * ux_r),
    }


def all_residuals(u, c, x, t):
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])
    out = ic_residuals(u, dt)
    out.update(bc_residuals(u, c, dx, dt))
    out["bc_rms"] = 0.5 * (out["bc_left_rms"] + out["bc_right_rms"])
    out["bc_rel"] = 0.5 * (out["bc_left_rel"] + out["bc_right_rel"])
    return out


def evaluate_set(fields, target, E, rho, x, t):
    """Mean residuals over a set of samples, for one model plus the reference.

    fields, target: (S, nx, nt);  E, rho: (S, nx)
    """
    keys = None
    acc_model, acc_ref = [], []
    for s in range(fields.shape[0]):
        c = np.sqrt(np.asarray(E[s], dtype=np.float64)
                    / np.maximum(np.asarray(rho[s], dtype=np.float64), 1e-12))
        m = all_residuals(fields[s], c, x, t)
        r = all_residuals(target[s], c, x, t)
        if keys is None:
            keys = list(m.keys())
        acc_model.append([m[k] for k in keys])
        acc_ref.append([r[k] for k in keys])
    model_mean = np.mean(acc_model, axis=0)
    ref_mean = np.mean(acc_ref, axis=0)
    return ({k: float(v) for k, v in zip(keys, model_mean)},
            {k: float(v) for k, v in zip(keys, ref_mean)})
