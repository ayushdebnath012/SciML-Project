import os
import torch
import torch.optim as optim
from src.losses.wave_loss import GradNormWeighter, get_causal_frontier, _causal_state, _IC_SCALE




def train_ic_warmup(model, x_ic, sigma_g=0.1, iterations=2000, lr=1e-3):
    """
    Pure IC regression phase — runs before joint PDE training to move the model
    out of the u≡0 basin (which satisfies the PDE+BCs trivially).

    Args:
        x_ic: spatial collocation points, shape (N, 1).  Use a dense grid
              around the Gaussian peak for best results.
    """
    from src.losses.wave_loss import ic_loss
    model.train(True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for i in range(iterations):
        optimizer.zero_grad()
        loss = ic_loss(model, x_ic, use_ansatz=False, sigma_g=sigma_g)
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            print(f"    IC warmup [{i+1}/{iterations}] loss: {loss.item():.6f}", flush=True)
    print(f"    IC warmup done — final IC loss: {loss.item():.6f}", flush=True)


def train_adam(model, inputs, losses_func, iterations, lr=None,
               physics_loss_weight=1, conds_loss_weight=1,
               print_every=500, outdir=None, reuse_optimizer=None,
               use_gradnorm=False, gradnorm_update_freq=100,
               ic_gradnorm_delay=0,
               causal_tolerance=1.0,
               initial_best=float('inf'),
               optimizer_name="adam",
               **kwargs):
    """
    Adam training loop (or SOAP when optimizer_name="soap").

    Args:
        optimizer_name: "adam" (default) or "soap" (rotated-Adam quasi-2nd-order,
                        arXiv:2409.11321 / arXiv:2502.00604).
        use_gradnorm: If True, `losses_func` must return (pde_loss, bc_loss, ic_loss)
                      as three separate tensors.
        gradnorm_update_freq: How often to recompute grad-norm weights (default: 100).
        ic_gradnorm_delay: Steps to use fixed IC-heavy weights (1.0/0.5/20.0) before
                           handing off to GradNorm.  Helps when use_ansatz=False.
        causal_tolerance: ε for causal weighting.  Pass a (start, end) tuple to
                          linearly ramp from start at iter=0 to end at iter=iterations-1.
                          E.g. (0.1, 1.0) starts loose (all chunks see gradients) and
                          tightens to strict causal enforcement by the end of Adam.
        initial_best: best validation metric so far (continuation runs pass the
                      previous phase's best so they cannot regress the checkpoint).

    Checkpointing uses a STATIONARY validation metric (unweighted PDE residual +
    BC + unscaled IC), not the weighted training loss: the training loss deflates
    mechanically as the causal tolerance ramps up and GradNorm weights drift, so
    it is not comparable across iterations.

    Returns a 12-tuple; the last element is the best validation metric reached.
    """
    model.train(True)

    if reuse_optimizer is None:
        if optimizer_name == "soap":
            from src.optimizers import SOAP
            optimizer = SOAP(model.parameters(), lr=lr, betas=(0.95, 0.95),
                             precondition_frequency=10)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    else:
        optimizer = reuse_optimizer

    total_losses   = []
    physics_losses = []
    conds_losses   = []
    best_loss      = initial_best

    # Grad-norm weights (start equal, updated adaptively)
    w_pde, w_bc, w_ic = 1.0, 1.0, 1.0
    w_pde_hist, w_bc_hist, w_ic_hist = [], [], []
    # Causal convergence tracking:
    #   w_min_hist       — scalar w_min at every iteration (paper convergence criterion)
    #   w_chunks_snaps   — full (n_chunks,) weight vector saved at each print_every step
    w_min_hist     = []
    w_chunks_snaps = []   # list of (iter, weights_array) tuples
    if use_gradnorm:
        weighter = GradNormWeighter(alpha=0.9, update_freq=gradnorm_update_freq)

    for iter in range(iterations):
        optimizer.zero_grad()

        # Compute scheduled causal tolerance for this step
        if isinstance(causal_tolerance, (list, tuple)):
            t_lo, t_hi = causal_tolerance
            tol = t_lo + (t_hi - t_lo) * (iter / max(iterations - 1, 1))
        else:
            tol = causal_tolerance

        if use_gradnorm:
            # losses_func returns (loss_pde, loss_bc, loss_ic) separately
            step_kwargs = {**kwargs, 'causal_tolerance': tol}
            loss_pde, loss_bc, loss_ic = losses_func(model, *inputs, **step_kwargs)

            if iter >= ic_gradnorm_delay:
                w_pde, w_bc, w_ic = weighter.compute_weights(
                    model, loss_pde, loss_bc, loss_ic,
                    use_ansatz=kwargs.get("use_ansatz", False)
                )
            else:
                # Fixed IC-heavy weights until GradNorm activates
                w_pde, w_bc, w_ic = 1.0, 0.5, 20.0

            total_loss    = w_pde * loss_pde + w_bc * loss_bc + w_ic * loss_ic
            physics_loss_val = loss_pde.item()
            conds_loss_val   = (loss_bc + loss_ic).item()
            # Stationary checkpoint metric: unweighted per-chunk PDE residual mean
            # (free — already computed inside causal_pde_loss) + BC + unscaled IC.
            cl = _causal_state["chunk_losses"]
            pde_unweighted = float(cl.mean()) if cl is not None else physics_loss_val
            val_metric = pde_unweighted + loss_bc.item() + loss_ic.item() / _IC_SCALE
        else:
            step_kwargs = {**kwargs, 'causal_tolerance': tol}
            physics_loss, conds_loss = losses_func(model, *inputs, **step_kwargs)
            total_loss = physics_loss * physics_loss_weight + conds_loss * conds_loss_weight
            physics_loss_val = physics_loss.item()
            conds_loss_val   = conds_loss.item()
            val_metric = total_loss.item()   # constant weights — already stationary

        total_loss.backward()
        optimizer.step()

        total_loss_val = total_loss.item()

        # ── Causal convergence tracking (every iteration) ─────────────────────
        cw = _causal_state["weights"]
        w_min_val = float(cw.min()) if cw is not None else 0.0
        w_min_hist.append(w_min_val)

        if iter % print_every == 0:
            if use_gradnorm:
                delay_tag = f" [GradNorm delayed {iter}/{ic_gradnorm_delay}]" if iter < ic_gradnorm_delay else ""
                print(f"    ITERATION: {iter+1} | LOSS: {total_loss_val:.6f} "
                      f"| w=(pde={w_pde:.3f}, bc={w_bc:.3f}, ic={w_ic:.3f}){delay_tag}", flush=True)
                print(f"    {get_causal_frontier()}", flush=True)
            else:
                print(f"    ITERATION: {iter+1} | LOSS: {total_loss_val}", flush=True)
            # Save full weight snapshot for post-hoc convergence plots
            if cw is not None:
                w_chunks_snaps.append((iter + 1, cw.numpy().tolist()))

        total_losses.append(total_loss_val)
        physics_losses.append(physics_loss_val)
        conds_losses.append(conds_loss_val)
        w_pde_hist.append(w_pde)
        w_bc_hist.append(w_bc)
        w_ic_hist.append(w_ic)

        if outdir is not None and val_metric < best_loss:
            os.makedirs(os.path.dirname(os.path.abspath(outdir)), exist_ok=True)
            torch.save(model.state_dict(), outdir)
            best_loss = val_metric

    return (total_losses, physics_losses, conds_losses,
            w_pde, w_bc, w_ic,
            w_pde_hist, w_bc_hist, w_ic_hist,
            w_min_hist, w_chunks_snaps, best_loss)


def train_lbfgs(model, inputs, losses_func, max_iter=500, lr=1.0,
                physics_loss_weight=1, conds_loss_weight=1,
                print_every=50, outdir=None,
                use_gradnorm=False, w_pde=1.0, w_bc=1.0, w_ic=1.0,
                **kwargs):
    """
    L-BFGS training loop.

    Args:
        use_gradnorm: If True, uses fixed weights w_pde/w_bc/w_ic passed in
                      (inherited from end of Adam phase). losses_func must return
                      (pde_loss, bc_loss, ic_loss).
    """
    model.train(True)

    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        tolerance_grad=0,
        tolerance_change=0,
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    total_losses   = []
    physics_losses = []
    conds_losses   = []

    metrics = {'best_loss': float('inf'), 'iter': 0}

    def closure():
        optimizer.zero_grad()

        if use_gradnorm:
            loss_pde, loss_bc, loss_ic = losses_func(model, *inputs, **kwargs)
            total_loss       = w_pde * loss_pde + w_bc * loss_bc + w_ic * loss_ic
            physics_loss_val = loss_pde.item()
            conds_loss_val   = (loss_bc + loss_ic).item()
            # Stationary checkpoint metric (see train_adam). With use_causal=False
            # loss_pde is already the plain residual; otherwise use chunk losses.
            if kwargs.get('use_causal', True) and _causal_state["chunk_losses"] is not None:
                pde_unweighted = float(_causal_state["chunk_losses"].mean())
            else:
                pde_unweighted = physics_loss_val
            val_metric = pde_unweighted + loss_bc.item() + loss_ic.item() / _IC_SCALE
        else:
            physics_loss, conds_loss = losses_func(model, *inputs, **kwargs)
            total_loss       = physics_loss * physics_loss_weight + conds_loss * conds_loss_weight
            physics_loss_val = physics_loss.item()
            conds_loss_val   = conds_loss.item()
            val_metric = total_loss.item()

        total_loss.backward()

        total_loss_val = total_loss.item()
        total_losses.append(total_loss_val)
        physics_losses.append(physics_loss_val)
        conds_losses.append(conds_loss_val)

        if metrics['iter'] % print_every == 0:
            print(f"    L-BFGS ITERATION: {metrics['iter']+1} | LOSS: {total_loss_val}", flush=True)

        if outdir is not None and val_metric < metrics['best_loss']:
            os.makedirs(os.path.dirname(os.path.abspath(outdir)), exist_ok=True)
            torch.save(model.state_dict(), outdir)
            metrics['best_loss'] = val_metric

        metrics['iter'] += 1
        return total_loss

    while metrics['iter'] < max_iter:
        prev_iter = metrics['iter']
        optimizer.step(closure)
        if metrics['iter'] == prev_iter:
            break

    return total_losses, physics_losses, conds_losses
