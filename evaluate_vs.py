"""
evaluate_vs.py — Multi-window evaluation
=========================================

Loads each window's checkpoint, evaluates on the FDM eval grid,
stitches predictions across windows, and computes L2 errors.

Non-dim convention: x_nd = x/L, t_nd = t/T_CHAR (same as training).
Physical conversion: v = v_nd * V_SCALE, s = s_nd * S_SCALE.
"""

import os, pickle
import numpy as np
import jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fdm_reference import (XMAX, T_CHAR, T_END_ND, V_SCALE, S_SCALE, L)
from model_vs import create_vs_model, vs_ansatz


# ── Load one window's model ───────────────────────────────────────────────────
def load_window(ckpt_path):
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    params  = ckpt["params"]
    t_max   = ckpt["t_max_nd"]
    model, _ = create_vs_model(t_max_nd=t_max)
    return model, params, ckpt["t_min_nd"], ckpt["t_max_nd"]


# ── Batch inference for one window ───────────────────────────────────────────
def predict_window(model, params, x_nd, t_nd, batch=4096):
    N    = len(x_nd)
    v_nd = np.zeros(N, dtype=np.float32)
    s_nd = np.zeros(N, dtype=np.float32)

    @jax.jit
    def _batch(xb, tb):
        def f(x, t):
            v_r, s_r = model.apply(params, jnp.stack([x, t]))
            return vs_ansatz(v_r, s_r, x, t)
        return jax.vmap(f)(xb, tb)

    for start in range(0, N, batch):
        end   = min(start + batch, N)
        xb    = jnp.array(x_nd[start:end])
        tb    = jnp.array(t_nd[start:end])
        vb, sb = _batch(xb, tb)
        v_nd[start:end] = np.array(vb)
        s_nd[start:end] = np.array(sb)

    return v_nd, s_nd


# ── Route each eval point to its window ──────────────────────────────────────
def predict_all_windows(ckpt_dir, x_nd_eval, t_nd_eval):
    """
    For each eval point, find which window covers it and evaluate that network.
    Adjacent windows' predictions at boundary times are averaged (SeismicNet style).
    """
    manifest_path = os.path.join(ckpt_dir, "manifest.pkl")
    with open(manifest_path, "rb") as f:
        manifest = pickle.load(f)
    windows = manifest["windows"]  # list of (t_min_nd, t_max_nd)

    v_nd_all = np.zeros_like(x_nd_eval)
    s_nd_all = np.zeros_like(x_nd_eval)
    covered  = np.zeros(len(x_nd_eval), dtype=bool)

    for i, (t_min, t_max) in enumerate(windows):
        ckpt = os.path.join(ckpt_dir, f"params_window{i+1}.pkl")
        if not os.path.exists(ckpt):
            print(f"  WARNING: {ckpt} missing, skipping window {i+1}")
            continue

        model, params, _, _ = load_window(ckpt)

        # Points that belong to this window
        if i == len(windows) - 1:
            mask = (t_nd_eval >= t_min) & (t_nd_eval <= t_max + 1e-6)
        else:
            mask = (t_nd_eval >= t_min) & (t_nd_eval < t_max)

        if mask.sum() == 0:
            continue

        print(f"  Window {i+1}: {mask.sum()} eval points "
              f"[{t_min:.4f}, {t_max:.4f}]")

        v_nd, s_nd = predict_window(
            model, params, x_nd_eval[mask], t_nd_eval[mask])

        v_nd_all[mask] = v_nd
        s_nd_all[mask] = s_nd
        covered[mask]  = True

    if not covered.all():
        print(f"  WARNING: {(~covered).sum()} eval points not covered by any window")

    return v_nd_all, s_nd_all


# ── Main evaluator ────────────────────────────────────────────────────────────
def relative_l2(pred, ref):
    denom = np.sqrt(np.mean(ref**2))
    if denom < 1e-30:
        return float('nan')
    return float(np.sqrt(np.mean((pred - ref)**2)) / denom)


def evaluate(ckpt_dir="checkpoints_v3",
             fdm_path="fdm_data.npz",
             out_dir="eval_output_v3"):
    os.makedirs(out_dir, exist_ok=True)

    # ── Load FDM reference ────────────────────────────────────────────────────
    data       = np.load(fdm_path)
    xt_eval    = data["xt_eval"]           # physical (m, s)
    v_ref_phys = data["v_eval"].ravel()    # m/s
    s_ref_phys = data["s_eval"].ravel()    # Pa

    x_phys = xt_eval[:, 0]
    t_phys = xt_eval[:, 1]

    # Convert to PINN non-dim coords
    x_nd = (x_phys / XMAX).astype(np.float32)
    t_nd = (t_phys / T_CHAR).astype(np.float32)

    print(f"[evaluate] {len(x_nd)} eval points")
    print(f"  x_nd ∈ [{x_nd.min():.3f}, {x_nd.max():.3f}]")
    print(f"  t_nd ∈ [{t_nd.min():.4f}, {t_nd.max():.4f}]  (T_END_ND={T_END_ND:.4f})")

    # ── Evaluate PINN ─────────────────────────────────────────────────────────
    print("[evaluate] Running multi-window inference...")
    v_nd_pred, s_nd_pred = predict_all_windows(ckpt_dir, x_nd, t_nd)

    # Convert back to physical
    v_pred = v_nd_pred * V_SCALE
    s_pred = s_nd_pred * S_SCALE

    # ── L2 errors ─────────────────────────────────────────────────────────────
    l2_v = relative_l2(v_pred, v_ref_phys)
    l2_s = relative_l2(s_pred, s_ref_phys)
    print(f"\nVelocity L2 : {l2_v*100:.2f}%  ← PRIMARY")
    print(f"Stress   L2 : {l2_s*100:.2f}%")

    # ── Plots ─────────────────────────────────────────────────────────────────
    nx, nt = 256, 100
    try:
        v_pred_2d  = v_pred.reshape(nt, nx)
        s_pred_2d  = s_pred.reshape(nt, nx)
        v_ref_2d   = v_ref_phys.reshape(nt, nx)
        s_ref_2d   = s_ref_phys.reshape(nt, nx)
        t_unique   = np.unique(t_phys)[:nt]
        x_unique   = np.unique(x_phys)[:nx]

        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        snap_idx  = [0, nt//4, nt//2, nt-1]

        for k, ti in enumerate(snap_idx):
            ax = axes[0, k]
            ax.plot(x_unique/1000, v_pred_2d[ti]/V_SCALE, label="PINN", lw=1.5)
            ax.plot(x_unique/1000, v_ref_2d[ti]/V_SCALE,  label="FDM",  lw=1, ls="--")
            ax.axvline(XMAX/2/1000, color="r", lw=0.8, ls=":")
            ax.set_title(f"v  t={t_unique[ti]:.3f}s")
            ax.set_xlabel("x (km)")
            ax.legend(fontsize=7)

            ax = axes[1, k]
            ax.plot(x_unique/1000, s_pred_2d[ti]/S_SCALE, label="PINN", lw=1.5)
            ax.plot(x_unique/1000, s_ref_2d[ti]/S_SCALE,  label="FDM",  lw=1, ls="--")
            ax.axvline(XMAX/2/1000, color="r", lw=0.8, ls=":")
            ax.set_title(f"σ  t={t_unique[ti]:.3f}s")
            ax.set_xlabel("x (km)")
            ax.legend(fontsize=7)

        fig.suptitle(f"Velocity L2={l2_v*100:.1f}%   Stress L2={l2_s*100:.1f}%")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "snapshots.png"), dpi=130)
        plt.close(fig)
        print(f"[evaluate] Saved {out_dir}/snapshots.png")
    except Exception as e:
        print(f"[evaluate] Plot failed: {e}")

    return {"l2_velocity": l2_v, "l2_stress": l2_s}


if __name__ == "__main__":
    r = evaluate()
    print(f"\nFinal  Velocity L2 : {r['l2_velocity']*100:.2f}%")
    print(f"Final  Stress   L2 : {r['l2_stress']*100:.2f}%")