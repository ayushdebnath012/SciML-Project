"""Differentiable IC / BC loss terms for physics-informed operator training.

physics_metrics.py computes the same quantities in numpy for reporting; this
module computes them in torch so they can enter the training objective and
produce gradients. The two must agree numerically -- there is a test for that
in the training script.

Each term is divided by a fixed scale precomputed from the training set, so
all three land in a comparable O(1) range and a single weight can balance them
against the data term. The scales are constants, not batch statistics: a
data-dependent denominator inside a loss makes gradients noisy.
"""
import torch


def _dudt_full(row, dt):
    """d/dt of a (B, n) row: central inside, second-order one-sided at ends."""
    left = (-3.0 * row[:, 0] + 4.0 * row[:, 1] - row[:, 2]) / (2.0 * dt)
    mid = (row[:, 2:] - row[:, :-2]) / (2.0 * dt)
    right = (3.0 * row[:, -1] - 4.0 * row[:, -2] + row[:, -3]) / (2.0 * dt)
    return torch.cat([left.unsqueeze(1), mid, right.unsqueeze(1)], dim=1)


def ic_bc_losses(u, c, dx, dt, scales):
    """u: (B, nx, nt) in physical units. c: (B, nx). scales: (s_u, s_ut, s_bc).

    Returns (loss_ic_displacement, loss_ic_velocity, loss_bc), each a scalar.
    """
    s_u, s_ut, s_bc = scales

    # --- initial condition: quiescent start, u = 0 and u_t = 0 at t = 0 ------
    u0 = u[:, :, 0]
    ut0 = (-3.0 * u[:, :, 0] + 4.0 * u[:, :, 1] - u[:, :, 2]) / (2.0 * dt)
    loss_ic_u = torch.mean((u0 / s_u) ** 2)
    loss_ic_ut = torch.mean((ut0 / s_ut) ** 2)

    # --- absorbing boundaries: u_t -/+ c u_x = 0 at the two edges -----------
    ut_left = _dudt_full(u[:, 0, :], dt)
    ut_right = _dudt_full(u[:, -1, :], dt)
    ux_left = (-3.0 * u[:, 0, :] + 4.0 * u[:, 1, :] - u[:, 2, :]) / (2.0 * dx)
    ux_right = (3.0 * u[:, -1, :] - 4.0 * u[:, -2, :] + u[:, -3, :]) / (2.0 * dx)

    c_l = c[:, 0].unsqueeze(1)
    c_r = c[:, -1].unsqueeze(1)
    r_left = ut_left - c_l * ux_left
    r_right = ut_right + c_r * ux_right
    loss_bc = 0.5 * (torch.mean((r_left / s_bc) ** 2)
                     + torch.mean((r_right / s_bc) ** 2))

    return loss_ic_u, loss_ic_ut, loss_bc


def training_scales(u, c, dx, dt):
    """Fixed normalisers from the training targets: RMS of u, of u_t, and of
    the boundary terms that are supposed to cancel."""
    s_u = torch.sqrt(torch.mean(u ** 2))
    ut = _dudt_full(u.reshape(-1, u.shape[-1]), dt)
    s_ut = torch.sqrt(torch.mean(ut ** 2))
    ut_l = _dudt_full(u[:, 0, :], dt)
    ux_l = (-3.0 * u[:, 0, :] + 4.0 * u[:, 1, :] - u[:, 2, :]) / (2.0 * dx)
    s_bc = torch.sqrt(torch.mean(ut_l ** 2)) + torch.sqrt(
        torch.mean((c[:, 0].unsqueeze(1) * ux_l) ** 2))
    return (float(s_u), float(s_ut), float(s_bc))
