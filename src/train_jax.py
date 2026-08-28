"""
JAX training loops: Adam (via optax) and L-BFGS (via jaxopt).

Mirrors the API of train.py so run_experiment.py can switch backends with
minimal changes.  Key differences:
  - Models are immutable pytrees; every step returns a new model.
  - Checkpoint saving uses eqx.tree_serialise_leaves (equinox 0.11+).
  - GradNorm callables are built here and passed to GradNormWeighterJax.
  - The training step is @eqx.filter_jit compiled; causal-state arrays are
    extracted from the auxiliary output after each step and written to
    _causal_state_jax from Python (side-effects are outside JIT).
"""
import os
import math
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from src.losses.wave_loss_jax import (
    losses_gradnorm_jax, ic_warmup_loss_jax,
    GradNormWeighterJax, get_causal_frontier_jax, _causal_state_jax, _IC_SCALE,
    losses_rba_jax, rba_update_jax, _rba_state_jax,
)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_model_jax(path: str, model):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    eqx.tree_serialise_leaves(path, model)


def load_model_jax(path: str, model_template, required: bool = True):
    """Deserialise a checkpoint, or return the template unchanged.

    A phase only writes a checkpoint when it beats the best validation metric so
    far. L-BFGS legitimately fails to do that -- a line search that gives up on
    its first iteration leaves no file -- and in that case the right model to
    carry forward is the one that went in, not a crash. `required=False` says
    the caller can live without it.
    """
    if not required and not os.path.exists(_with_eqx_suffix(path)):
        return model_template
    return eqx.tree_deserialise_leaves(path, model_template)


def _with_eqx_suffix(path: str) -> str:
    """equinox appends `.eqx` when the path has no suffix; mirror that so the
    existence check agrees with what it actually wrote."""
    return path if os.path.splitext(path)[1] else path + ".eqx"


# ── JIT-compiled training step ────────────────────────────────────────────────

def _make_train_step(optimizer):
    """Returns a filter_jit-compiled step function closed over the optimizer."""

    @eqx.filter_jit
    def step(model, opt_state,
             x_int, t_int, x_bc, t_bc, x_ic,
             w_pde, w_bc, w_ic, tol,
             material, sigma_g, use_ansatz, n_chunks, use_causal):

        def total_loss_fn(m):
            l_pde, l_bc, l_ic, weights, cl = losses_gradnorm_jax(
                m, x_int, t_int, x_bc, t_bc, x_ic,
                material, sigma_g,
                use_ansatz=use_ansatz,
                causal_tolerance=tol,
                n_chunks=n_chunks,
                use_causal=use_causal,
            )
            total = w_pde * l_pde + w_bc * l_bc + w_ic * l_ic
            return total, (l_pde, l_bc, l_ic, weights, cl)

        (total, aux), grads = eqx.filter_value_and_grad(
            total_loss_fn, has_aux=True
        )(model)

        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, total, aux

    return step


def _make_rba_train_step(optimizer):
    """RBA variant: λ multipliers replace causal weighting; λ updated in-step."""

    @eqx.filter_jit
    def step(model, opt_state, rba_lam,
             x_int, t_int, x_bc, t_bc, x_ic,
             w_pde, w_bc, w_ic,
             material, sigma_g, use_ansatz):

        def total_loss_fn(m):
            l_pde, l_bc, l_ic, R = losses_rba_jax(
                m, x_int, t_int, x_bc, t_bc, x_ic,
                material, sigma_g, use_ansatz=use_ansatz,
                rba_weights=rba_lam,
            )
            total = w_pde * l_pde + w_bc * l_bc + w_ic * l_ic
            return total, (l_pde, l_bc, l_ic, R)

        (total, aux), grads = eqx.filter_value_and_grad(
            total_loss_fn, has_aux=True
        )(model)

        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)

        l_pde, l_bc, l_ic, R = aux
        R = jax.lax.stop_gradient(R)
        new_lam        = rba_update_jax(rba_lam, R)
        pde_unweighted = (R ** 2).mean()
        return (new_model, new_opt_state, new_lam, total,
                (l_pde, l_bc, l_ic, pde_unweighted))

    return step


def _make_optimizer(optimizer_name: str, lr: float):
    if optimizer_name == "soap":
        from src.optimizers_jax import soap
        return soap(lr, b1=0.95, b2=0.95, precondition_frequency=10)
    return optax.adam(lr, b1=0.9, b2=0.999, eps=1e-8)


# ── IC warm-up ────────────────────────────────────────────────────────────────

def train_ic_warmup_jax(model, x_ic, sigma_g=0.1, iterations=2000, lr=1e-3):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def warmup_step(m, opt_st, x_ic_):
        (loss, grads) = eqx.filter_value_and_grad(
            lambda m_: ic_warmup_loss_jax(m_, x_ic_, sigma_g)
        )(m)
        updates, new_opt = optimizer.update(grads, opt_st, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, updates), new_opt, loss

    for i in range(iterations):
        model, opt_state, loss = warmup_step(model, opt_state, x_ic)
        if i % 500 == 0:
            print(f"    IC warmup [{i+1}/{iterations}] loss: {float(loss):.6f}", flush=True)

    print(f"    IC warmup done — final IC loss: {float(loss):.6f}", flush=True)
    return model


# ── Adam training loop ────────────────────────────────────────────────────────

def train_adam_jax(model, inputs, losses_func,
                   iterations, lr=1e-3,
                   print_every=500, outdir=None,
                   use_gradnorm=False, gradnorm_update_freq=100,
                   ic_gradnorm_delay=0, causal_tolerance=1.0,
                   initial_best=float('inf'),
                   use_ansatz=True, material=None,
                   sigma_g=0.1, n_chunks=16,
                   optimizer_name="adam", use_rba=False,
                   **kwargs):
    """
    Adam training loop. Returns the same 12 training summaries as
    train.train_adam, followed by the final model. The caller needs that final
    model when several continuation blocks are chained: the best-loss
    checkpoint may predate causal-frontier progress and must not be used as the
    starting point of every block.

    inputs = [x_int, t_int, x_bc, t_bc, x_ic]  (all JAX arrays, fixed shape)
    losses_func is ignored (always uses losses_gradnorm_jax internally).

    optimizer_name: "adam" (default) or "soap" (src/optimizers_jax.py).
    use_rba: Residual-Based Attention pointwise PDE weighting instead of
             causal weighting.  w_min is reported as 1.0 (no causal frontier),
             so the runner's causal-continuation loop is skipped naturally.
    """
    x_int, t_int, x_bc, t_bc, x_ic = inputs

    optimizer = _make_optimizer(optimizer_name, lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    if use_rba:
        rba_lam     = _rba_state_jax.get("weights")
        if rba_lam is None or rba_lam.shape[0] != x_int.shape[0]:
            rba_lam = jnp.ones(x_int.shape[0])
        rba_step_fn = _make_rba_train_step(optimizer)
    else:
        step_fn = _make_train_step(optimizer)

    if use_gradnorm:
        weighter = GradNormWeighterJax(alpha=0.9, update_freq=gradnorm_update_freq)

    total_losses   = []
    physics_losses = []
    conds_losses   = []
    w_pde, w_bc, w_ic = 1.0, 1.0, 1.0
    w_pde_hist, w_bc_hist, w_ic_hist = [], [], []
    w_min_hist     = []
    w_chunks_snaps = []
    best_loss      = initial_best

    for it in range(iterations):
        # Scheduled causal tolerance
        if isinstance(causal_tolerance, (list, tuple)):
            t_lo, t_hi = causal_tolerance
            tol = t_lo + (t_hi - t_lo) * (it / max(iterations - 1, 1))
        else:
            tol = causal_tolerance
        tol_arr = jnp.array(tol)

        if use_rba:
            model, opt_state, rba_lam, total, aux = rba_step_fn(
                model, opt_state, rba_lam,
                x_int, t_int, x_bc, t_bc, x_ic,
                jnp.array(w_pde), jnp.array(w_bc), jnp.array(w_ic),
                material, sigma_g, use_ansatz,
            )
            l_pde, l_bc, l_ic, pde_unweighted = aux
            # No causal frontier in RBA mode: report converged weights so the
            # runner's continuation loop is a no-op, and store the unweighted
            # residual for the stationary checkpoint metric.
            _causal_state_jax["weights"]      = np.ones(1)
            _causal_state_jax["chunk_losses"] = np.array([float(pde_unweighted)])
            _causal_state_jax["n_chunks"]     = 1
            _rba_state_jax["weights"]         = rba_lam
        else:
            model, opt_state, total, aux = step_fn(
                model, opt_state,
                x_int, t_int, x_bc, t_bc, x_ic,
                jnp.array(w_pde), jnp.array(w_bc), jnp.array(w_ic),
                tol_arr, material, sigma_g, use_ansatz, n_chunks, True,
            )
            l_pde, l_bc, l_ic, weights, chunk_losses = aux

            # Update causal state sidecar (outside JIT — no side-effects in traced fn)
            _causal_state_jax["weights"]      = np.array(weights)
            _causal_state_jax["chunk_losses"] = np.array(chunk_losses)
            _causal_state_jax["n_chunks"]     = n_chunks

        total_val = float(total)
        l_pde_val = float(l_pde)
        l_bc_val  = float(l_bc)
        l_ic_val  = float(l_ic)

        # GradNorm weight update
        if use_gradnorm:
            cl = _causal_state_jax["chunk_losses"]
            pde_unweighted = float(np.asarray(cl).mean()) if cl is not None else l_pde_val
            val_metric = pde_unweighted + l_bc_val + l_ic_val / _IC_SCALE

            if it >= ic_gradnorm_delay:
                # Build per-loss callables for gradient norm computation
                if use_rba:
                    _lam = jax.lax.stop_gradient(rba_lam)

                    def _pde_fn(m):
                        l, _, _, _ = losses_rba_jax(
                            m, x_int, t_int, x_bc, t_bc, x_ic,
                            material, sigma_g, use_ansatz=use_ansatz,
                            rba_weights=_lam,
                        )
                        return l

                    def _bc_fn(m):
                        _, l, _, _ = losses_rba_jax(
                            m, x_int, t_int, x_bc, t_bc, x_ic,
                            material, sigma_g, use_ansatz=use_ansatz,
                            rba_weights=_lam,
                        )
                        return l

                    def _ic_fn(m):
                        _, _, l, _ = losses_rba_jax(
                            m, x_int, t_int, x_bc, t_bc, x_ic,
                            material, sigma_g, use_ansatz=use_ansatz,
                            rba_weights=_lam,
                        )
                        return l / _IC_SCALE   # unscaled for grad norm
                else:
                    def _pde_fn(m):
                        l, _, _, _, _ = losses_gradnorm_jax(
                            m, x_int, t_int, x_bc, t_bc, x_ic,
                            material, sigma_g, use_ansatz=use_ansatz,
                            causal_tolerance=tol, n_chunks=n_chunks, use_causal=True,
                        )
                        return l

                    def _bc_fn(m):
                        _, l, _, _, _ = losses_gradnorm_jax(
                            m, x_int, t_int, x_bc, t_bc, x_ic,
                            material, sigma_g, use_ansatz=use_ansatz,
                            causal_tolerance=tol, n_chunks=n_chunks, use_causal=True,
                        )
                        return l

                    def _ic_fn(m):
                        _, _, l, _, _ = losses_gradnorm_jax(
                            m, x_int, t_int, x_bc, t_bc, x_ic,
                            material, sigma_g, use_ansatz=use_ansatz,
                            causal_tolerance=tol, n_chunks=n_chunks, use_causal=True,
                        )
                        return l / _IC_SCALE   # unscaled for grad norm

                w_pde, w_bc, w_ic = weighter.compute_weights(
                    model, _pde_fn, _bc_fn,
                    None if use_ansatz else _ic_fn,
                    use_ansatz=use_ansatz,
                )
            else:
                w_pde, w_bc, w_ic = 1.0, 0.5, 20.0
                val_metric = total_val
        else:
            val_metric = total_val

        # Causal w_min tracking
        cw = _causal_state_jax["weights"]
        w_min_val = float(np.min(cw)) if cw is not None else 0.0
        w_min_hist.append(w_min_val)

        if it % print_every == 0:
            if use_gradnorm:
                delay_tag = (f" [GradNorm delayed {it}/{ic_gradnorm_delay}]"
                             if it < ic_gradnorm_delay else "")
                print(f"    ITERATION: {it+1} | LOSS: {total_val:.6f} "
                      f"| w=(pde={w_pde:.3f}, bc={w_bc:.3f}, ic={w_ic:.3f}){delay_tag}",
                      flush=True)
                print(f"    {get_causal_frontier_jax()}", flush=True)
            else:
                print(f"    ITERATION: {it+1} | LOSS: {total_val}", flush=True)
            if cw is not None:
                w_chunks_snaps.append((it + 1, np.array(cw).tolist()))

        total_losses.append(total_val)
        physics_losses.append(l_pde_val)
        conds_losses.append(l_bc_val + l_ic_val)
        w_pde_hist.append(w_pde)
        w_bc_hist.append(w_bc)
        w_ic_hist.append(w_ic)

        if outdir is not None and val_metric < best_loss:
            save_model_jax(outdir, model)
            best_loss = val_metric

    return (total_losses, physics_losses, conds_losses,
            w_pde, w_bc, w_ic,
            w_pde_hist, w_bc_hist, w_ic_hist,
            w_min_hist, w_chunks_snaps, best_loss, model)


# ── L-BFGS training loop ──────────────────────────────────────────────────────

def train_lbfgs_jax(model, inputs, losses_func,
                     max_iter=500, lr=1.0,
                     print_every=50, outdir=None,
                     use_gradnorm=False, w_pde=1.0, w_bc=1.0, w_ic=1.0,
                     use_ansatz=True, material=None,
                     sigma_g=0.1, n_chunks=16,
                     **kwargs):
    """
    L-BFGS loop via jaxopt (use_causal=False, plain residual).
    Returns (total_losses, physics_losses, conds_losses).
    """
    try:
        import jaxopt
    except ImportError:
        raise ImportError("jaxopt not installed.  Run: pip install jaxopt")

    x_int, t_int, x_bc, t_bc, x_ic = inputs

    params, static = eqx.partition(model, eqx.is_array)

    def loss_fn(params_):
        m = eqx.combine(params_, static)
        l_pde, l_bc, l_ic, _, _ = losses_gradnorm_jax(
            m, x_int, t_int, x_bc, t_bc, x_ic,
            material, sigma_g,
            use_ansatz=use_ansatz,
            causal_tolerance=1.0,
            n_chunks=n_chunks,
            use_causal=False,    # plain residual inside L-BFGS line search
        )
        return w_pde * l_pde + w_bc * l_bc + w_ic * l_ic

    solver    = jaxopt.LBFGS(fun=loss_fn, maxiter=1, tol=0.0,
                              linesearch='zoom', history_size=50)
    state     = solver.init_state(params)

    total_losses   = []
    physics_losses = []
    conds_losses   = []
    best_loss_lbfgs = float('inf')

    for it in range(max_iter):
        params, state = solver.update(params, state)
        total_val = float(state.value)
        total_losses.append(total_val)
        physics_losses.append(total_val)   # decomposition not available per-step
        conds_losses.append(0.0)

        if it % print_every == 0:
            print(f"    L-BFGS (JAX) ITERATION: {it+1} | LOSS: {total_val}", flush=True)

        if outdir is not None and total_val < best_loss_lbfgs:
            save_model_jax(outdir, eqx.combine(params, static))
            best_loss_lbfgs = total_val

        if state.failed_linesearch:
            print(f"    L-BFGS line search failed at iter {it+1}, stopping.", flush=True)
            break

    return total_losses, physics_losses, conds_losses
