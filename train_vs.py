"""
train_vs.py

Changes from 1.81% run:
  - Removed interface-only snapshot concatenation.
    They introduced a 1.79× overweighting of x∈[0.44,0.56] in the loss,
    biasing the gradient away from the far-field and creating a gap between
    training data_loss (1.15% predicted) and eval L2 (1.81% actual).
    With pure uniform sampling: predicted ≈ actual (confirmed in 2.53% run).
  - adam_steps stays at 120,000 (W2 was still descending at 120k end).

model_vs.py now has split sig_net (left/right of x=0.5) for stress.
This file has no awareness of that — it just trains whatever VSNet outputs.
"""

import time, os, pickle
import numpy as np
import jax, jax.numpy as jnp, optax
from scipy.optimize import minimize

from model_vs   import create_vs_model, vs_ansatz
from sampler    import make_collocation, load_snapshot_data, make_windows, to_jax
from loss_vs    import make_hybrid_loss, make_lbfgs_vag
from fdm_reference import T_END_ND, T_CHAR


def flatten_params(params):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    shapes = [l.shape for l in leaves]
    sizes  = [l.size  for l in leaves]
    flat   = np.concatenate([np.array(l).ravel() for l in leaves]).astype(np.float64)
    return flat, shapes, sizes, treedef

def unflatten_params(flat, shapes, sizes, treedef):
    leaves, offset = [], 0
    for shape, size in zip(shapes, sizes):
        leaves.append(jnp.array(flat[offset:offset+size].astype(np.float32)).reshape(shape))
        offset += size
    return jax.tree_util.tree_unflatten(treedef, leaves)


def adam_phase(params, model, xt_jax, snap_v_jax, snap_s_jax,
               n_steps=120_000, lr=5e-4, w_phys=0.0,
               causal_eps=1.0, log_every=12000,
               prev_model=None, prev_params=None, t_min_nd=0.0):

    loss_fn = make_hybrid_loss(model, w_phys=w_phys, causal_eps=causal_eps,
                               prev_model=prev_model, prev_params=prev_params,
                               t_min_nd=t_min_nd)
    sched = optax.exponential_decay(init_value=lr, transition_steps=10_000, decay_rate=0.9)
    opt   = optax.adam(sched)
    state = opt.init(params)

    @jax.jit
    def step_fn(p, s, xt, sv, ss):
        (loss, info), grads = jax.value_and_grad(
            loss_fn, argnums=0, has_aux=True)(p, xt, sv, ss)
        updates, new_s = opt.update(grads, s, p)
        return optax.apply_updates(p, updates), new_s, loss, info

    hist, t0 = [], time.time()
    print(f"    [Adam] compiling...", end="", flush=True)
    p, s, l0, i0 = step_fn(params, state, xt_jax, snap_v_jax, snap_s_jax)
    print(f" done ({time.time()-t0:.1f}s)  loss={float(l0):.4e}"
          f"  data={float(i0['loss_data']):.4e}"
          f"  phys={float(i0['loss_phys']):.4e}")
    hist.append(float(l0))
    params, state = p, s

    for step in range(1, n_steps):
        params, state, loss, info = step_fn(params, state, xt_jax, snap_v_jax, snap_s_jax)
        hist.append(float(loss))
        if step % log_every == 0 or step == n_steps - 1:
            print(f"    [Adam] {step:6d}/{n_steps}  "
                  f"loss={hist[-1]:.4e}  "
                  f"data={float(info['loss_data']):.4e}  "
                  f"phys={float(info['loss_phys']):.4e}  "
                  f"t={time.time()-t0:.1f}s")
    return params, hist


def lbfgs_phase(params, model, xt_jax, snap_v_jax, snap_s_jax,
                max_iter=20_000, w_phys=0.0, causal_eps=1.0, log_every=2000,
                prev_model=None, prev_params=None, t_min_nd=0.0):

    vag = make_lbfgs_vag(model, w_phys=w_phys, causal_eps=causal_eps,
                         prev_model=prev_model, prev_params=prev_params,
                         t_min_nd=t_min_nd)
    flat0, shapes, sizes, treedef = flatten_params(params)
    hist, iters, printed = [], [0], [False]

    def obj(flat_np):
        p = unflatten_params(flat_np, shapes, sizes, treedef)
        loss, grads, info = vag(p, xt_jax, snap_v_jax, snap_s_jax)
        if not printed[0]:
            print(f" done  loss={float(loss):.4e}"
                  f"  data={float(info['loss_data']):.4e}"
                  f"  phys={float(info['loss_phys']):.4e}")
            printed[0] = True
        loss_np = float(loss)
        hist.append(loss_np)
        iters[0] += 1
        gf = np.concatenate([np.array(g).ravel()
                              for g in jax.tree_util.tree_leaves(grads)]).astype(np.float64)
        if iters[0] % log_every == 0:
            print(f"    [L-BFGS] {iters[0]:6d}  loss={loss_np:.4e}")
        return loss_np, gf

    res = minimize(obj, flat0, method="L-BFGS-B", jac=True,
                   options={"maxiter": max_iter, "maxcor": 50,
                            "ftol": np.finfo(float).eps, "gtol": 1e-10, "maxls": 50})
    params_new = unflatten_params(res.x, shapes, sizes, treedef)
    final = hist[-1] if hist else float('nan')
    print(f"    [L-BFGS] {res.message}  iters={iters[0]}  final={final:.4e}")
    return params_new, hist


def train_window(params, model, xt_jax, snap_v_jax, snap_s_jax,
                 adam_steps=120_000, lbfgs_iter=20_000,
                 lr=5e-4, causal_eps=1.0,
                 prev_model=None, prev_params=None, t_min_nd=0.0):

    all_hist = []
    t0 = time.time()

    print(f"  ── Adam  w=0  lr=5e-04  ({adam_steps:,} steps) ──")
    params, h_adam = adam_phase(
        params, model, xt_jax, snap_v_jax, snap_s_jax,
        n_steps=adam_steps, lr=lr, w_phys=0.0,
        causal_eps=causal_eps, log_every=adam_steps//10,
        prev_model=prev_model, prev_params=prev_params, t_min_nd=t_min_nd)
    all_hist.extend(h_adam)
    print(f"    Adam: {h_adam[0]:.4e} → {h_adam[-1]:.4e}")

    print(f"    L-BFGS  w=0  compiling...", end="", flush=True)
    params, h_lbfgs = lbfgs_phase(
        params, model, xt_jax, snap_v_jax, snap_s_jax,
        max_iter=lbfgs_iter, w_phys=0.0,
        causal_eps=causal_eps, log_every=lbfgs_iter//5,
        prev_model=prev_model, prev_params=prev_params, t_min_nd=t_min_nd)
    all_hist.extend(h_lbfgs)
    print(f"    L-BFGS: {h_lbfgs[0]:.4e} → {h_lbfgs[-1]:.4e}")

    print(f"  Window done: {(time.time()-t0)/60:.1f} min  final={all_hist[-1]:.4e}")
    return params, all_hist


def train(config=None):
    cfg = {
        "n_windows":   4,
        "n_bulk":      20_000,
        "n_interface": 2_000,
        "n_source":    2_000,
        "n_snaps":     15,
        "n_pts_snap":  800,
        "adam_steps":  120_000,
        "lbfgs_iter":  20_000,
        "lr":          5e-4,
        "causal_eps":  1.0,
        "fdm_path":    "fdm_data.npz",
        "checkpoint_dir": "checkpoints_v3",
        "seed":        42,
    }
    if config:
        cfg.update(config)

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    print(f"\n=== VS-SIREN Pure-Data Temporal Decomposition ===")
    print(f"Config: {cfg}")

    windows    = make_windows(cfg["n_windows"])
    all_params = []
    all_models = []
    t_total    = time.time()

    print(f"\nTime windows:")
    for i, (wmin, wmax) in enumerate(windows):
        print(f"  Window {i+1}: t_nd ∈ [{wmin:.4f}, {wmax:.4f}]"
              f" = [{wmin*T_CHAR:.3f}s, {wmax*T_CHAR:.3f}s]")

    for i, (t_min, t_max) in enumerate(windows):
        print(f"\n{'═'*60}")
        print(f"  Window {i+1}/{len(windows)}  t_nd ∈ [{t_min:.4f}, {t_max:.4f}]")
        print(f"{'═'*60}")

        model, params = create_vs_model(
            key=jax.random.PRNGKey(cfg["seed"] + i),
            t_max_nd=t_max)

        xt_np = make_collocation(
            n_bulk=cfg["n_bulk"], n_interface=cfg["n_interface"],
            n_source=cfg["n_source"], t_min_nd=t_min, t_max_nd=t_max,
            seed=cfg["seed"] + i * 7)
        print(f"  Collocation: {len(xt_np):,} pts")

        # Uniform snapshots only — no interface bias
        snap_v, snap_s = load_snapshot_data(
            cfg["fdm_path"], t_min, t_max,
            n_snaps_per_window=cfg["n_snaps"],
            n_pts_per_snap=cfg["n_pts_snap"],
            seed=cfg["seed"] + i * 13)

        if snap_v is None:
            print(f"  WARNING: no FDM snapshots")
            snap_v = np.zeros((1, 3), dtype=np.float32)
            snap_s = np.zeros((1, 3), dtype=np.float32)
        else:
            print(f"  Snapshot data: {len(snap_v)} v-pts, {len(snap_s)} s-pts")

        xt_jax     = to_jax(xt_np)
        snap_v_jax = to_jax(snap_v)
        snap_s_jax = to_jax(snap_s)

        p_model  = all_models[-1] if i > 0 else None
        p_params = all_params[-1] if i > 0 else None

        params, hist = train_window(
            params, model, xt_jax, snap_v_jax, snap_s_jax,
            adam_steps=cfg["adam_steps"], lbfgs_iter=cfg["lbfgs_iter"],
            lr=cfg["lr"], causal_eps=cfg["causal_eps"],
            prev_model=p_model, prev_params=p_params, t_min_nd=t_min)

        all_params.append(params)
        all_models.append(model)

        ckpt = os.path.join(cfg["checkpoint_dir"], f"params_window{i+1}.pkl")
        with open(ckpt, "wb") as f:
            pickle.dump({"params": params, "window": i+1,
                         "t_min_nd": t_min, "t_max_nd": t_max,
                         "loss_hist": hist, "config": cfg}, f)
        print(f"  Checkpoint → {ckpt}")

    print(f"\n{'═'*60}")
    print(f"  All windows done  {(time.time()-t_total)/60:.1f} min")
    print(f"{'═'*60}")

    manifest = os.path.join(cfg["checkpoint_dir"], "manifest.pkl")
    with open(manifest, "wb") as f:
        pickle.dump({"windows": windows, "config": cfg}, f)
    print(f"  Manifest → {manifest}")
    return all_models, all_params, windows


if __name__ == "__main__":
    train(config={
        "n_bulk": 3_000, "n_interface": 200, "n_source": 200,
        "n_snaps": 5, "n_pts_snap": 100, "adam_steps": 500, "lbfgs_iter": 500,
        "checkpoint_dir": "checkpoints_smoke",
    })