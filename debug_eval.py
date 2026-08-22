"""
debug_eval.py — traces exactly what evaluate.py compares
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import jax
import jax.numpy as jnp
import pickle

from model         import create_model, ic_ansatz, g_displacement
from fdm_reference import X_SCALE, T_SCALE, U_SCALE, XMAX

# ── Load checkpoint
ckpt_path = "checkpoints/params_final.pkl"
with open(ckpt_path, "rb") as f:
    ckpt = pickle.load(f)
params = ckpt["params"]
cfg    = ckpt.get("config", {})

model, _ = create_model(
    hidden_dim  = cfg.get("hidden_dim",  32),
    num_blocks  = cfg.get("num_blocks",  2),
    embed_dim   = cfg.get("embed_dim",   16),
    embed_scale = cfg.get("embed_scale", 1.0),
)

print("=" * 60)
print("A. PINN predictions at a few (x_nd, t_nd) points")
print("=" * 60)

test_pts = [
    (0.5,  0.0),
    (0.47, 0.0),
    (0.53, 0.0),
    (0.47, 0.1),
    (0.53, 0.1),
    (0.3,  0.2),
    (0.7,  0.2),
    (0.5,  0.5),
    (0.3,  0.5),
]

for xv, tv in test_pts:
    xt    = jnp.array([xv, tv], dtype=jnp.float32)
    u_raw = float(model.apply(params, xt))
    g_v   = float(g_displacement(jnp.float32(xv)))
    u_hat = float(ic_ansatz(jnp.float32(u_raw),
                             jnp.float32(xv),
                             jnp.float32(tv),
                             jnp.float32(g_v)))
    print(f"  x={xv:.2f} t={tv:.1f}: u_raw={u_raw:+.4f}  g={g_v:+.4f}  u_hat={u_hat:+.4f}")

print()
print("=" * 60)
print("B. FDM reference at same physical points")
print("=" * 60)

fdm = np.load("fdm_data.npz")
t_snaps = fdm["t_snaps"]
x_v     = fdm["x_v"]
u_snaps = fdm["u_snaps"]

print(f"  FDM shape: u_snaps {u_snaps.shape},  t range [{t_snaps[0]:.3f}, {t_snaps[-1]:.3f}]")
print(f"  FDM |u|_max = {np.max(np.abs(u_snaps)):.4e} m")
print(f"  FDM |u|_max / U_SCALE = {np.max(np.abs(u_snaps)) / U_SCALE:.4f}  (should be ~1.0)")
print()

from scipy.interpolate import RegularGridInterpolator
interp = RegularGridInterpolator(
    (t_snaps, x_v), u_snaps, method="linear",
    bounds_error=False, fill_value=0.0
)

print("  FDM u (physical) and u_nd = u/U_SCALE at test points:")
for xv, tv in test_pts:
    x_phys = xv * X_SCALE
    t_phys = tv * T_SCALE
    u_fdm  = float(interp(np.array([[t_phys, x_phys]]))[0])
    u_nd   = u_fdm / U_SCALE
    print(f"  x={xv:.2f} t={tv:.1f}: u_fdm={u_fdm:+.4e} m  u_nd={u_nd:+.4f}")

print()
print("=" * 60)
print("C. Direct comparison at t_nd=0 (must match perfectly)")
print("=" * 60)

# At t=0: PINN u_hat = g(x), FDM u = U_SCALE * g(x)
# So u_hat should equal FDM_u / U_SCALE
x_nd_arr = np.linspace(0.3, 0.7, 20)
max_err = 0.0
for xv in x_nd_arr:
    x_phys = xv * X_SCALE
    u_fdm  = float(interp(np.array([[0.0, x_phys]]))[0])
    u_nd   = u_fdm / U_SCALE

    xt    = jnp.array([xv, 0.0], dtype=jnp.float32)
    u_raw = float(model.apply(params, xt))
    g_v   = float(g_displacement(jnp.float32(xv)))
    u_hat = float(ic_ansatz(jnp.float32(u_raw),
                             jnp.float32(xv),
                             jnp.float32(0.0),
                             jnp.float32(g_v)))
    err = abs(u_hat - u_nd)
    max_err = max(max_err, err)

print(f"  Max |u_hat(t=0) - u_ref_nd(t=0)| over x in [0.3,0.7] = {max_err:.2e}")
print(f"  (Should be ~0 if ICs match)")

print()
print("=" * 60)
print("D. Grid-level comparison: what does evaluate.py actually compute")
print("=" * 60)

n_eval_x, n_eval_t = 64, 50   # small for speed
x_nd = np.linspace(0, 1, n_eval_x, dtype=np.float32)
t_nd = np.linspace(0, 1, n_eval_t, dtype=np.float32)

# PINN grid
def u_hat_single(xt):
    x, t   = xt[0], xt[1]
    u_raw  = model.apply(params, xt)
    g      = g_displacement(x)
    return ic_ansatz(u_raw, x, t, g)

XX, TT = jnp.meshgrid(jnp.array(x_nd), jnp.array(t_nd))
xt_flat = jnp.stack([XX.ravel(), TT.ravel()], axis=1)
u_pinn  = np.array(jax.vmap(u_hat_single)(xt_flat)).reshape(n_eval_t, n_eval_x)

# FDM grid
t_phys = t_nd * T_SCALE
x_phys = x_nd * X_SCALE
TT2, XX2 = np.meshgrid(t_phys, x_phys, indexing="ij")
u_ref = interp(np.stack([TT2.ravel(), XX2.ravel()], axis=1)).reshape(n_eval_t, n_eval_x)
u_ref_nd = u_ref / U_SCALE

print(f"  PINN  u_pinn  : min={u_pinn.min():.4f}  max={u_pinn.max():.4f}  "
      f"mean={u_pinn.mean():.4f}  std={u_pinn.std():.4f}")
print(f"  FDM   u_ref_nd: min={u_ref_nd.min():.4f}  max={u_ref_nd.max():.4f}  "
      f"mean={u_ref_nd.mean():.4f}  std={u_ref_nd.std():.4f}")
print()

# L2 error
num = np.sqrt(np.sum((u_pinn - u_ref_nd)**2))
den = np.sqrt(np.sum(u_ref_nd**2))
print(f"  ||u_pinn - u_ref_nd||_2 = {num:.4e}")
print(f"  ||u_ref_nd||_2          = {den:.4e}")
print(f"  Relative L2 error       = {num/den*100:.2f}%")
print()

# Per-timestep
print("  Per-timestep L2 errors:")
for i in [0, 5, 10, 25, 49]:
    n_i = np.sqrt(np.sum((u_pinn[i] - u_ref_nd[i])**2))
    d_i = np.sqrt(np.sum(u_ref_nd[i]**2)) + 1e-30
    print(f"    t_nd={t_nd[i]:.2f}: L2={n_i/d_i*100:.2f}%  "
          f"|u_pinn|={np.max(np.abs(u_pinn[i])):.4f}  "
          f"|u_ref_nd|={np.max(np.abs(u_ref_nd[i])):.4f}")