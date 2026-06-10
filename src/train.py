import os
import torch
import torch.optim as optim
from src.losses.wave_loss import GradNormWeighter, get_causal_frontier, _causal_state




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
               **kwargs):
    """
    Adam training loop.

    Args:
        use_gradnorm: If True, `losses_func` must return (pde_loss, bc_loss, ic_loss)
                      as three separate tensors.
        gradnorm_update_freq: How often to recompute grad-norm weights (default: 100).
        ic_gradnorm_delay: Steps to use fixed IC-heavy weights (1.0/0.5/20.0) before
                           handing off to GradNorm.  Helps when use_ansatz=False.
    """
    model.train(True)

    if reuse_optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    else:
        optimizer = reuse_optimizer

    total_losses   = []
    physics_losses = []
    conds_losses   = []
    best_loss      = torch.inf

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

        if use_gradnorm:
            # losses_func returns (loss_pde, loss_bc, loss_ic) separately
            loss_pde, loss_bc, loss_ic = losses_func(model, *inputs, **kwargs)

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
        else:
            physics_loss, conds_loss = losses_func(model, *inputs, **kwargs)
            total_loss = physics_loss * physics_loss_weight + conds_loss * conds_loss_weight
            physics_loss_val = physics_loss.item()
            conds_loss_val   = conds_loss.item()

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

        if outdir is not None and total_loss_val < best_loss:
            os.makedirs(os.path.dirname(os.path.abspath(outdir)), exist_ok=True)
            torch.save(model.state_dict(), outdir)
            best_loss = total_loss_val

    return (total_losses, physics_losses, conds_losses,
            w_pde, w_bc, w_ic,
            w_pde_hist, w_bc_hist, w_ic_hist,
            w_min_hist, w_chunks_snaps)


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
        else:
            physics_loss, conds_loss = losses_func(model, *inputs, **kwargs)
            total_loss       = physics_loss * physics_loss_weight + conds_loss * conds_loss_weight
            physics_loss_val = physics_loss.item()
            conds_loss_val   = conds_loss.item()

        total_loss.backward()

        total_loss_val = total_loss.item()
        total_losses.append(total_loss_val)
        physics_losses.append(physics_loss_val)
        conds_losses.append(conds_loss_val)

        if metrics['iter'] % print_every == 0:
            print(f"    L-BFGS ITERATION: {metrics['iter']+1} | LOSS: {total_loss_val}", flush=True)

        if outdir is not None and total_loss_val < metrics['best_loss']:
            os.makedirs(os.path.dirname(os.path.abspath(outdir)), exist_ok=True)
            torch.save(model.state_dict(), outdir)
            metrics['best_loss'] = total_loss_val

        metrics['iter'] += 1
        return total_loss

    while metrics['iter'] < max_iter:
        prev_iter = metrics['iter']
        optimizer.step(closure)
        if metrics['iter'] == prev_iter:
            break

    return total_losses, physics_losses, conds_losses
