"""
JAX equivalents of every loss in wave_loss.py.

Key differences from PyTorch version:
  - Second-order PDE derivatives use jax.grad composed with jax.grad, then
    jax.vmap over the batch dimension (avoids reverse-over-reverse overhead).
  - Causal chunk aggregation uses jax.ops.segment_sum (no Python loops).
  - jnp.unique can't run inside jit; BC points must be pre-computed and passed in.
  - gaussian_ic normalisation uses the analytical constant 1/sigma*exp(-0.5)
    instead of a data-batch max (numerically equivalent for dense grids).
  - GradNormWeighterJax recomputes each loss individually to get per-loss
    gradients (3 extra forward passes every update_freq steps).
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

# ── IC weights (mirror PyTorch constants) ─────────────────────────────────────
_IC_DISP_WEIGHT = 5.0
_IC_VEL_WEIGHT  = 1.0
_IC_SCALE       = 15000.0

# ── Causal-state sidecar (updated by training loop after each step) ───────────
_causal_state_jax: dict = {
    "weights":      None,
    "chunk_losses": None,
    "n_chunks":     32,
}


# ── IC / ansatz helpers ───────────────────────────────────────────────────────

def gaussian_ic_jax(x, sigma_g: float = 0.1, x0: float = 0.0):
    """
    Scalar-and-array-safe Gaussian IC.  Normalisation uses the analytical
    maximum of |dfdx| = (1/sigma) * exp(-0.5), identical to the PyTorch version
    for a sufficiently dense input grid.
    """
    f     = jnp.exp(-0.5 * ((x - x0) / sigma_g) ** 2)
    dfdx  = -(x - x0) / sigma_g ** 2 * f
    norm  = (1.0 / sigma_g) * math.exp(-0.5)
    return dfdx / (norm + 1e-12)


def apply_ansatz_jax(nn_out, x, t, sigma_g: float = 0.1, use_ansatz: bool = True):
    """Batched version: nn_out, x, t all (N, 1)."""
    if not use_ansatz:
        return nn_out
    g      = gaussian_ic_jax(x, sigma_g)
    decay  = jnp.exp(-0.5 * (15.0 * t) ** 2)
    growth = jnp.tanh(25.0 * t) ** 2
    return g * decay + growth * nn_out


def _ansatz_scalar(nn_scalar, xi, ti, sigma_g: float, use_ansatz: bool):
    """Scalar (pointwise) version used inside jax.grad chains."""
    if not use_ansatz:
        return nn_scalar
    g      = gaussian_ic_jax(xi, sigma_g)
    decay  = jnp.exp(-0.5 * (15.0 * ti) ** 2)
    growth = jnp.tanh(25.0 * ti) ** 2
    return g * decay + growth * nn_scalar


# ── PDE residual ──────────────────────────────────────────────────────────────

def compute_pde_residual_jax(model, x, t, material, sigma_g=0.1, use_ansatz=True):
    """
    Residual R = ρ u_tt - ∂_x[E u_x] for each collocation point.

    x, t: (N, 1) JAX arrays.
    Returns (N, 1) residuals.
    """
    x_flat = x.reshape(-1)
    t_flat = t.reshape(-1)

    def residual_fn(xi, ti):
        """Per-point residual via nested jax.grad."""
        def u_at(xi_, ti_):
            xt = jnp.stack([xi_, ti_])          # (2,)
            nn = model(xt)[0]                    # scalar NN output
            return _ansatz_scalar(nn, xi_, ti_, sigma_g, use_ansatz)

        # u_tt
        u_tt = jax.grad(jax.grad(u_at, 1), 1)(xi, ti)

        # ∂_x[E(x) ∂u/∂x]
        def stress(x_):
            return material.E_jax(x_) * jax.grad(u_at, 0)(x_, ti)

        sigma_x = jax.grad(stress)(xi)
        return material.rho_jax(xi) * u_tt - sigma_x

    residuals = jax.vmap(residual_fn)(x_flat, t_flat)
    return residuals.reshape(-1, 1)


# ── Causal PDE loss ───────────────────────────────────────────────────────────

def causal_pde_loss_jax(model, x_int, t_int, material,
                         sigma_g=0.1, n_chunks=32, tolerance=1.0,
                         use_ansatz=True):
    """
    Causal loss (Wang et al. 2022): returns (scalar loss, weights, chunk_losses).
    weights and chunk_losses are returned so the training loop can update
    _causal_state_jax without side-effects inside jit.
    """
    t_max = t_int.max()
    boundaries = jnp.linspace(0.0, t_max, n_chunks + 1)

    R  = compute_pde_residual_jax(model, x_int, t_int, material, sigma_g, use_ansatz)
    R2 = R.reshape(-1) ** 2

    # Assign each point to a time chunk
    idx = jnp.searchsorted(boundaries[1:-1], t_int.reshape(-1), side='right')

    sums   = jax.ops.segment_sum(R2,                jnp.asarray(idx, jnp.int32), n_chunks)
    counts = jax.ops.segment_sum(jnp.ones_like(R2), jnp.asarray(idx, jnp.int32), n_chunks)
    chunk_losses = sums / jnp.maximum(counts, 1.0)

    cumsum  = jnp.concatenate([jnp.zeros(1), jnp.cumsum(chunk_losses[:-1])])
    weights = jnp.exp(-tolerance * cumsum)

    loss = (weights * chunk_losses).mean()
    return loss, weights, chunk_losses


def get_causal_frontier_jax(threshold: float = 0.1) -> str:
    weights = _causal_state_jax["weights"]
    if weights is None:
        return "causal frontier: (not yet computed)"
    w       = np.asarray(weights)
    n       = len(w)
    reached = int((w >= threshold).sum())
    w_last  = float(w[-1])
    w_min   = float(w.min())
    bar_len = min(n, 40)
    filled  = int(round(reached / n * bar_len))
    bar     = "█" * filled + "░" * (bar_len - filled)
    converged = w_min >= 0.9
    status = "CONVERGED" if converged else ("FULL" if reached == n else f"chunk {reached}/{n}")
    return (f"causal frontier (jax): {status} {reached}/{n}  "
            f"(w_min={w_min:.4f}  w_last={w_last:.4f})  |{bar}|")


# ── BC loss ───────────────────────────────────────────────────────────────────

def _bc_residual_scalar(model, xi, ti, material, sigma_g, use_ansatz):
    def u_at(xi_, ti_):
        xt = jnp.stack([xi_, ti_])
        nn = model(xt)[0]
        return _ansatz_scalar(nn, xi_, ti_, sigma_g, use_ansatz)

    u_x = jax.grad(u_at, 0)(xi, ti)
    u_t = jax.grad(u_at, 1)(xi, ti)
    c   = material.Vp_jax(xi)
    mid = (material.x_min + material.x_max) / 2.0
    return jnp.where(xi < mid, u_t - c * u_x, u_t + c * u_x)


def bc_loss_jax(model, x_bc, t_bc, material, sigma_g=0.1, use_ansatz=True):
    """x_bc, t_bc: (N, 1) JAX arrays. Returns scalar BC loss."""
    residuals = jax.vmap(
        lambda xi, ti: _bc_residual_scalar(model, xi, ti, material, sigma_g, use_ansatz)
    )(x_bc.reshape(-1), t_bc.reshape(-1))
    return (residuals ** 2).mean()


# ── IC loss ───────────────────────────────────────────────────────────────────

def ic_loss_jax(model, x_ic, sigma_g=0.1, use_ansatz=False):
    """x_ic: (N, 1) JAX array. Returns scalar IC loss (unscaled)."""
    def disp_vel(xi):
        def u_at_t0(ti_):
            xt = jnp.stack([xi, ti_])
            nn = model(xt)[0]
            return _ansatz_scalar(nn, xi, ti_, sigma_g, use_ansatz)
        u_val   = u_at_t0(0.0)
        u_t_val = jax.grad(u_at_t0)(0.0)
        return u_val, u_t_val

    u_vals, u_t_vals = jax.vmap(disp_vel)(x_ic.reshape(-1))
    ic_ref    = gaussian_ic_jax(x_ic.reshape(-1), sigma_g)
    loss_disp = ((u_vals - ic_ref) ** 2).mean()
    loss_vel  = (u_t_vals ** 2).mean()
    return _IC_DISP_WEIGHT * loss_disp + _IC_VEL_WEIGHT * loss_vel


# ── Combined loss ─────────────────────────────────────────────────────────────

def losses_gradnorm_jax(model, x_int, t_int, x_bc, t_bc, x_ic,
                         material, sigma_g=0.1, use_ansatz=False,
                         causal_tolerance=1.0, n_chunks=16, use_causal=True):
    """
    Returns (loss_pde, loss_bc, loss_ic, weights, chunk_losses).

    x_int, t_int : (N_int, 1)  interior collocation points
    x_bc, t_bc   : (N_bc,  1)  boundary points (pre-sampled, fixed shape for JIT)
    x_ic         : (N_ic,  1)  IC points
    """
    if use_causal:
        loss_pde, weights, chunk_losses = causal_pde_loss_jax(
            model, x_int, t_int, material, sigma_g, n_chunks, causal_tolerance, use_ansatz
        )
    else:
        R        = compute_pde_residual_jax(model, x_int, t_int, material, sigma_g, use_ansatz)
        loss_pde = (R ** 2).mean()
        weights      = jnp.ones(n_chunks)
        chunk_losses = jnp.zeros(n_chunks)

    loss_bc = bc_loss_jax(model, x_bc, t_bc, material, sigma_g, use_ansatz)

    if use_ansatz:
        loss_ic = jnp.zeros(())
    else:
        loss_ic = _IC_SCALE * ic_loss_jax(model, x_ic, sigma_g, False)

    return loss_pde, loss_bc, loss_ic, weights, chunk_losses


# ── Residual-Based Attention (RBA) ────────────────────────────────────────────
# Pointwise multipliers λ_i on the PDE residuals (Anagnostopoulos et al. 2023,
# arXiv:2307.00379).  λ is threaded explicitly through the training step (JIT);
# the update rule lives in train_jax.  Sidecar mirrors _rba_state in PyTorch.
_RBA_GAMMA = 0.999
_RBA_ETA   = 0.01

_rba_state_jax: dict = {"weights": None}


def losses_rba_jax(model, x_int, t_int, x_bc, t_bc, x_ic,
                   material, sigma_g=0.1, use_ansatz=False,
                   rba_weights=None):
    """
    RBA variant of losses_gradnorm_jax.  Returns
        (loss_pde, loss_bc, loss_ic, R)
    where loss_pde = mean((λ r)²) and R is the raw (N, 1) residual so the
    training loop can update λ and compute the unweighted checkpoint metric.
    """
    R = compute_pde_residual_jax(model, x_int, t_int, material, sigma_g, use_ansatz)
    if rba_weights is None:
        rba_weights = jnp.ones(R.shape[0])
    loss_pde = ((rba_weights.reshape(-1, 1) * R) ** 2).mean()

    loss_bc = bc_loss_jax(model, x_bc, t_bc, material, sigma_g, use_ansatz)

    if use_ansatz:
        loss_ic = jnp.zeros(())
    else:
        loss_ic = _IC_SCALE * ic_loss_jax(model, x_ic, sigma_g, False)

    return loss_pde, loss_bc, loss_ic, R


def rba_update_jax(rba_weights, R,
                   gamma: float = _RBA_GAMMA, eta: float = _RBA_ETA):
    """λ ← γ λ + η |r| / max|r|  (detached — R must not carry gradients here)."""
    absR = jnp.abs(R.reshape(-1))
    return gamma * rba_weights + eta * absR / (absR.max() + 1e-12)


# ── GradNorm adaptive weighter ────────────────────────────────────────────────

class GradNormWeighterJax:
    """
    JAX port of GradNormWeighter.

    compute_weights() takes pre-built loss callables (model → scalar) so that
    eqx.filter_grad can compute per-loss gradient norms.  The callables are
    called 2–3 extra times every update_freq steps (cheap vs. total training).
    """
    def __init__(self, alpha: float = 0.9, update_freq: int = 100):
        self.alpha        = alpha
        self.update_freq  = update_freq
        self.step_counter = 0
        self.ema_weights  = None

    def compute_weights(self, model,
                        loss_pde_fn, loss_bc_fn, loss_ic_fn=None,
                        use_ansatz: bool = False):
        self.step_counter += 1
        if (self.ema_weights is not None
                and self.step_counter % self.update_freq != 0):
            return tuple(float(w) for w in self.ema_weights)

        def _grad_norm(fn):
            try:
                grads  = eqx.filter_grad(fn)(model)
                leaves = jax.tree_util.tree_leaves(grads)
                norm   = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves
                                      if g is not None))
                return float(norm) + 1e-12
            except Exception:
                return 1e-12

        n_pde = _grad_norm(loss_pde_fn)
        n_bc  = _grad_norm(loss_bc_fn)

        if use_ansatz:
            sum_norms  = n_pde + n_bc
            hat_lambda = jnp.array([sum_norms / n_pde, sum_norms / n_bc, 0.0])
            hat_lambda = 2.0 * hat_lambda / hat_lambda.sum()
        else:
            n_ic       = _grad_norm(loss_ic_fn) if loss_ic_fn is not None else 1e-12
            sum_norms  = n_pde + n_bc + n_ic
            hat_lambda = jnp.array([sum_norms / n_pde,
                                    sum_norms / n_bc,
                                    sum_norms / n_ic])
            hat_lambda = 3.0 * hat_lambda / hat_lambda.sum()

        if self.ema_weights is None:
            self.ema_weights = hat_lambda
        else:
            self.ema_weights = (self.alpha * self.ema_weights
                                + (1 - self.alpha) * hat_lambda)

        if not use_ansatz:
            self.ema_weights = self.ema_weights.at[2].set(
                max(float(self.ema_weights[2]), 1.0)
            )

        return tuple(float(w) for w in self.ema_weights)


# ── IC warm-up loss (plain, no causal) ────────────────────────────────────────

def ic_warmup_loss_jax(model, x_ic, sigma_g=0.1):
    return ic_loss_jax(model, x_ic, sigma_g, use_ansatz=False)
