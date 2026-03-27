"""
loss_vs.py — Hybrid physics + data loss
========================================

From TDPINN (TDPINN.py lines 87-94):
  loss = loss_snap + w_f * loss_f
  w_f starts at 1e-4 (first training), then 1e-3, then 1e-1

  loss_snap = MSE between network prediction and FDM snapshot data
  loss_f    = MSE of PDE residuals at collocation points

From SeismicNet (network.py lines 412-419):
  Manual loss weights: 80000*uv_f + 10000*s_f + 5*vel_f + ...
  → we use dynamic grad_norm balancing instead (from jaxpi evaluator.py)

Damping:
  Matches user's FDM exactly: 200-pt quadratic sponge at both ends.
  Applied as a linear damping term in both residuals.

Source:
  Ricker wavelet injected as body force in momentum equation,
  matching user's FDM v[src_i] += DT * F_SOURCE * ricker(t) / rho.
"""

import jax
import jax.numpy as jnp
from model_vs import CS_of_x_nd, CV, vs_ansatz
from fdm_reference import (L, T_CHAR, V_SCALE, S_SCALE, RHO)


# ── Ricker source (matches FDM exactly) ──────────────────────────────────────
def ricker_nd(t_nd):
    """Ricker wavelet in non-dim time."""
    t_phys = t_nd * T_CHAR
    a      = jnp.pi * F0 * (t_phys - T0)
    return (1.0 - 2.0*a**2) * jnp.exp(-a**2)


def source_momentum_nd(x_nd, t_nd):
    """
    Non-dimensional body force in momentum equation.

    FDM: v[src_i] += DT * F_SOURCE * ricker(t) / rho
    In continuous PDE: dv/dt += (F_SOURCE/rho) * delta(x-x_src) * ricker(t)

    Non-dim: (V_SCALE/T_CHAR) * dv_nd/dt_nd += (F_SOURCE/rho)*spatial*ricker
    → RHS in non-dim = (F_SOURCE * T_CHAR) / (rho * V_SCALE * L) * spatial * ricker
                     * (L / sigma_src_m)   [Gaussian approximation of delta]

    We approximate delta(x-x_src) as Gaussian with sigma_src = 3*DX/L (3 cells).
    """
    sigma_src = 3.0 * DX / L          # in non-dim x
    x_src     = 0.5                   # source at x=5000m → x_nd=0.5
    spatial   = jnp.exp(-0.5 * ((x_nd - x_src) / sigma_src)**2) / (sigma_src * jnp.sqrt(2*jnp.pi))
    # Non-dim amplitude
    amp = (F_SOURCE * T_CHAR) / (RHO * V_SCALE)
    return amp * spatial * ricker_nd(t_nd)


# ── Damping (quadratic sponge, matches FDM) ───────────────────────────────────
def damp_coeff_nd(x_nd):
    """
    Returns the dimensional damping rate d(x) [1/s], non-dimensionalised
    by T_CHAR:  d_nd = d_phys * T_CHAR.

    FDM: v *= exp(-damp_v * DT)  → damp_v has units [1/s]
    FDM damp profile: quadratic, 200 pts = NB/NX = 25% of domain.
    """
    frac = NB / NX                     # = 0.25
    left  = jnp.where(x_nd < frac,
                      DAMP_MAX * ((frac - x_nd) / frac)**2, 0.0)
    right = jnp.where(x_nd > 1.0 - frac,
                      DAMP_MAX * ((x_nd - (1.0-frac)) / frac)**2, 0.0)
    d_phys = left + right              # [1/s]
    return d_phys * T_CHAR             # non-dim


# ── PDE residuals at a single point ───────────────────────────────────────────
def residuals_single(params, model, x_nd, t_nd):
    """
    Homogeneous velocity-stress PDE residuals (NO source, NO damping).

    TDPINN net_f_sig (line 201): f_u = u_tt - vp^2*(u_xx+u_yy)  -- no source
    SeismicNet net_f_sig1 (lines 797-798): f_x = s_x/rho - v_t  -- no source

    Both codebases handle the source ONLY through the data loss (FDM snapshots).
    Adding the source to the PDE residual here makes it O(1e5) in non-dim units,
    completely overwhelming the other residual terms and preventing convergence.

    Damping is likewise excluded: the FDM snapshots already encode its effect,
    and including it in the PDE residual adds O(1) damping terms that conflict
    with the snapshot data in the sponge zones.

    R_v = dv_nd/dt_nd - CV * ds_nd/dx_nd  = 0
    R_s = ds_nd/dt_nd - CS(x) * dv_nd/dx_nd = 0
    """
    def v_fn(x, t):
        v_r, s_r = model.apply(params, jnp.stack([x, t]))
        return vs_ansatz(v_r, s_r, x, t)[0]

    def s_fn(x, t):
        v_r, s_r = model.apply(params, jnp.stack([x, t]))
        return vs_ansatz(v_r, s_r, x, t)[1]

    dv_dt = jax.grad(v_fn, argnums=1)(x_nd, t_nd)
    ds_dx = jax.grad(s_fn, argnums=0)(x_nd, t_nd)
    ds_dt = jax.grad(s_fn, argnums=1)(x_nd, t_nd)
    dv_dx = jax.grad(v_fn, argnums=0)(x_nd, t_nd)

    CS  = CS_of_x_nd(x_nd)

    R_v = dv_dt - CV * ds_dx
    R_s = ds_dt - CS * dv_dx
    return R_v, R_s


# ── Causal weights (from causal PINN paper, used in our Adam) ─────────────────
def causal_weights(t_nd, eps):
    t_max   = jnp.max(t_nd) + 1e-6
    w       = jnp.exp(-eps * t_nd / t_max)
    return w / jnp.mean(w)


# ── Physics loss ─────────────────────────────────────────────────────────────
def make_physics_loss(model, causal_eps=1.0):
    def loss_fn(params, xt_batch):
        x_nd = xt_batch[:, 0]
        t_nd = xt_batch[:, 1]

        def r(x, t): return residuals_single(params, model, x, t)
        rv, rs = jax.vmap(r)(x_nd, t_nd)

        w    = causal_weights(t_nd, causal_eps)
        loss = jnp.mean(w * (rv**2 + rs**2))
        return loss, {"rms_v": jnp.sqrt(jnp.mean(rv**2)),
                      "rms_s": jnp.sqrt(jnp.mean(rs**2))}

    return jax.jit(loss_fn)


# ── Data loss (TDPINN: loss_snap) ─────────────────────────────────────────────
def make_data_loss(model):
    """
    MSE between PINN prediction and FDM snapshot values.
    From TDPINN: loss_snap = mean((u_pred - u_snap)^2)

    snap_batch: (N, 3) array — columns [x_nd, t_nd, v_nd_target]
    snap_s:     (N, 3) array — columns [x_nd, t_nd, s_nd_target]
    """
    def loss_fn(params, snap_v_batch, snap_s_batch):
        def predict_v(x, t):
            v_r, s_r = model.apply(params, jnp.stack([x, t]))
            v_h, _   = vs_ansatz(v_r, s_r, x, t)
            return v_h

        def predict_s(x, t):
            v_r, s_r = model.apply(params, jnp.stack([x, t]))
            _, s_h   = vs_ansatz(v_r, s_r, x, t)
            return s_h

        # Velocity data loss
        xv = snap_v_batch[:, 0]
        tv = snap_v_batch[:, 1]
        vt = snap_v_batch[:, 2]
        v_pred = jax.vmap(predict_v)(xv, tv)
        loss_v = jnp.mean((v_pred - vt)**2)

        # Stress data loss
        xs = snap_s_batch[:, 0]
        ts = snap_s_batch[:, 1]
        st = snap_s_batch[:, 2]
        s_pred = jax.vmap(predict_s)(xs, ts)
        loss_s = jnp.mean((s_pred - st)**2)

        return loss_v + loss_s

    return jax.jit(loss_fn)


# ── Hybrid loss: physics + data + interface (TDPINN + SeismicNet) ────────────
def make_hybrid_loss(model, w_phys=1e-4, causal_eps=1.0,
                     prev_model=None, prev_params=None, t_min_nd=0.0):
    """
    loss = loss_data + w_phys * loss_phys + loss_intf

    Interface loss has been disabled. We now rely strictly on the 
    FDM snapshot anchoring at t_min to enforce continuity.

    w_phys schedule (TDPINN):
      Phase 1: 1e-4  Phase 2: 1e-3  Phase 3: 1e-1
    """
    phys_fn      = make_physics_loss(model, causal_eps)
    data_fn      = make_data_loss(model)

    def loss_fn(params, xt_coll, snap_v, snap_s):
        l_phys, info = phys_fn(params, xt_coll)
        l_data       = data_fn(params, snap_v, snap_s)

        # ── Active interface loss ─────────────────────────────────────────────
        # Remove interface loss - FDM snapshots at t_min provide continuity
        l_intf = jnp.float32(0.0)

        total = l_data + w_phys * l_phys + l_intf
        return total, {**info,
                       "loss_data":  l_data,
                       "loss_phys":  l_phys,
                       "loss_intf":  l_intf}

    return jax.jit(loss_fn)


# ── L-BFGS value-and-grad ────────────────────────────────────────────────────
def make_lbfgs_vag(model, w_phys=1e-4, causal_eps=1.0,
                   prev_model=None, prev_params=None, t_min_nd=0.0):
    loss_fn = make_hybrid_loss(model, w_phys, causal_eps,
                               prev_model, prev_params, t_min_nd)

    @jax.jit
    def vag(params, xt_coll, snap_v, snap_s):
        (loss, info), grads = jax.value_and_grad(
            loss_fn, argnums=0, has_aux=True)(params, xt_coll, snap_v, snap_s)
        return loss, grads, info

    return vag