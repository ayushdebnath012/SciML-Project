"""
debug_vs.py — Full diagnostic for velocity-stress PINN

Checks every component of the pipeline to find why L2=100%.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import jax, jax.numpy as jnp
import pickle

SEP = "=" * 65

# ── Load checkpoint
ckpt_path = "checkpoints_vs/params_final.pkl"
with open(ckpt_path, "rb") as f:
    ckpt = pickle.load(f)
params    = ckpt["params"]
cfg       = ckpt.get("config", {})
loss_hist = ckpt.get("loss_hist", [])

from model_vs import (create_vs_model, vs_ansatz, sigma0_nd,
                      V_SCALE, S_SCALE, U_SCALE, CV, CS_COEFF, MU_REF)
from fdm_reference import X_SCALE, T_SCALE, g_displacement_np, XMAX, C_LEFT, C_RIGHT, RHO

model, _ = create_vs_model(
    hidden_dim=cfg.get("hidden_dim",128), num_blocks=cfg.get("num_blocks",3),
    embed_dim=cfg.get("embed_dim",64),    embed_scale=cfg.get("embed_scale",1.0),
)

print(SEP)
print("1. SCALES AND COEFFICIENTS")
print(SEP)
print(f"  V_SCALE  = {V_SCALE:.4e} m/s   (U/T = {U_SCALE:.1e}/{T_SCALE})")
print(f"  S_SCALE  = {S_SCALE:.4e} Pa    (mu_ref*U/L)")
print(f"  CV       = {CV:.6f}   (want ~0.2, encodes wave speed)")
print(f"  CS_COEFF = {CS_COEFF:.6f}   (mu ratio coefficient)")
print()
print(f"  Wave speed check:")
print(f"    c_L_nd = sqrt(CV * CS_left)  = sqrt({CV:.4f}*1.000) = {(CV*1.0)**0.5:.4f} (want 0.45)")
print(f"    c_R_nd = sqrt(CV * CS_right) = sqrt({CV:.4f}*{(C_RIGHT/C_LEFT)**2:.3f}) = {(CV*(C_RIGHT/C_LEFT)**2)**0.5:.4f} (want 0.75)")

print()
print(SEP)
print("2. NETWORK OUTPUT — RAW AND AFTER ANSATZ")
print(SEP)

test_pts = [
    (0.47, 0.0), (0.47, 0.05), (0.47, 0.1), (0.47, 0.3), (0.47, 0.5),
    (0.53, 0.0), (0.53, 0.1), (0.53, 0.5),
    (0.30, 0.1), (0.30, 0.3), (0.70, 0.1),
]

print(f"  {'x':>5} {'t':>5} | {'v_raw':>10} {'s_raw':>10} | {'v_hat':>10} {'s_hat':>10} | {'sigma0':>10}")
for xv, tv in test_pts:
    xt    = jnp.array([xv, tv], dtype=jnp.float32)
    v_r, s_r = model.apply(params, xt)
    v_h, s_h = vs_ansatz(v_r, s_r, jnp.float32(xv), jnp.float32(tv))
    s0        = float(sigma0_nd(jnp.float32(xv)))
    print(f"  {xv:5.2f} {tv:5.2f} | {float(v_r):10.4f} {float(s_r):10.4f} | "
          f"{float(v_h):10.4f} {float(s_h):10.4f} | {s0:10.4f}")

print()
print(SEP)
print("3. IC ANSATZ VERIFICATION")
print(SEP)

# At t=0: v_hat must = 0, sigma_hat must = sigma0(x)
print("  At t=0: v_hat must=0, sigma_hat must=sigma0(x)")
max_v_err = 0.0; max_s_err = 0.0
for xv in [0.40, 0.44, 0.47, 0.50, 0.53, 0.56, 0.60]:
    xt = jnp.array([xv, 0.0], dtype=jnp.float32)
    v_r, s_r = model.apply(params, xt)
    v_h, s_h = vs_ansatz(v_r, s_r, jnp.float32(xv), jnp.float32(0.0))
    s0 = float(sigma0_nd(jnp.float32(xv)))
    v_err = abs(float(v_h))
    s_err = abs(float(s_h) - s0)
    max_v_err = max(max_v_err, v_err)
    max_s_err = max(max_s_err, s_err)
    print(f"    x={xv:.2f}: v_hat={float(v_h):.6f} (want 0)  "
          f"s_hat={float(s_h):.6f}  s0={s0:.6f}  err={s_err:.2e}")
print(f"  Max v error at t=0: {max_v_err:.2e}  (want <1e-6)")
print(f"  Max s error at t=0: {max_s_err:.2e}  (want <1e-6)")

print()
print(SEP)
print("4. SIGMA0 PROFILE — IS IT CORRECT?")
print(SEP)
x_nd_arr = np.linspace(0, 1, 1000)
s0_vals  = np.array([float(sigma0_nd(jnp.float32(x))) for x in x_nd_arr])
print(f"  sigma0 max = {np.max(np.abs(s0_vals)):.6f}  (want ~1)")
print(f"  sigma0 at x=0.5 = {float(sigma0_nd(jnp.float32(0.5))):.6f}  (should be 0, antisymmetric)")
print(f"  sigma0 support: [{x_nd_arr[np.abs(s0_vals)>0.01].min():.3f}, {x_nd_arr[np.abs(s0_vals)>0.01].max():.3f}]")
print()
# Compare with FDM sigma at t=0
fdm = np.load("fdm_data.npz")
s_snaps = fdm["s_snaps"]   # (Nsnap, NX-1) Pa
x_v_fdm = fdm["x_v"]
DX = x_v_fdm[1] - x_v_fdm[0]
x_s_fdm = x_v_fdm[:-1] + 0.5*DX
s0_fdm_nd = s_snaps[0] / S_SCALE   # first snapshot (t=0)
# Interpolate to same x grid
from scipy.interpolate import interp1d
f_s0 = interp1d(x_s_fdm/XMAX, s0_fdm_nd, bounds_error=False, fill_value=0.0)
s0_fdm_at_x = f_s0(x_nd_arr)
err_s0 = np.max(np.abs(s0_vals - s0_fdm_at_x))
print(f"  FDM sigma0 max (non-dim): {np.max(np.abs(s0_fdm_nd)):.6f}  (want ~1)")
print(f"  Max |sigma0_pinn - sigma0_fdm| = {err_s0:.4f}  (want <0.01)")
print()
# Show side by side
print(f"  {'x_nd':>6} {'sigma0_pinn':>14} {'sigma0_fdm':>12} {'diff':>8}")
for i in range(0, len(x_nd_arr), 50):
    xv = x_nd_arr[i]
    sp = s0_vals[i]
    sf = float(f_s0(xv))
    print(f"  {xv:6.3f} {sp:14.6f} {sf:12.6f} {sp-sf:8.4f}")

print()
print(SEP)
print("5. FDM FIELDS — ARE THEY NON-ZERO?")
print(SEP)
t_snaps = fdm["t_snaps"]
u_snaps = fdm["u_snaps"]
v_snaps = fdm["v_snaps"]

print(f"  FDM snapshots: {len(t_snaps)}, t=[{t_snaps[0]:.3f}, {t_snaps[-1]:.3f}]s")
print(f"  |u|_max = {np.max(np.abs(u_snaps)):.4e} m")
print(f"  |v|_max = {np.max(np.abs(v_snaps)):.4e} m/s")
print(f"  |s|_max = {np.max(np.abs(s_snaps)):.4e} Pa")
print()
print(f"  Non-dim: |v|_max/V_SCALE = {np.max(np.abs(v_snaps))/V_SCALE:.4f}  (want ~1)")
print(f"  Non-dim: |s|_max/S_SCALE = {np.max(np.abs(s_snaps))/S_SCALE:.4f}  (want ~1-3)")
print()
print(f"  FDM v at x=0.47*L, various times:")
ix = int(0.47 * len(x_v_fdm))
for i in [0, 5, 10, 20, 40, 60]:
    if i < len(t_snaps):
        t_nd_val = t_snaps[i] / T_SCALE
        v_phys   = v_snaps[i, ix]
        v_nd     = v_phys / V_SCALE
        print(f"    t={t_snaps[i]:.3f}s (t_nd={t_nd_val:.3f}): v={v_phys:.4e} m/s  v_nd={v_nd:.4f}")

print()
print(SEP)
print("6. PINN vs FDM DIRECT COMPARISON AT KEY POINTS")
print(SEP)

from scipy.interpolate import RegularGridInterpolator
interp_v = RegularGridInterpolator(
    (t_snaps, x_v_fdm), v_snaps, method="linear",
    bounds_error=False, fill_value=0.0)
interp_s = RegularGridInterpolator(
    (t_snaps, x_v_fdm[:-1]+0.5*DX), s_snaps, method="linear",
    bounds_error=False, fill_value=0.0)

print(f"  {'x_nd':>5} {'t_nd':>5} | {'v_pinn':>10} {'v_fdm_nd':>10} {'ratio':>8} | "
      f"{'s_pinn':>10} {'s_fdm_nd':>10}")
compare_pts = [
    (0.47, 0.05), (0.47, 0.10), (0.47, 0.20),
    (0.53, 0.05), (0.53, 0.10),
    (0.30, 0.10), (0.30, 0.20),
    (0.70, 0.10), (0.70, 0.20),
]
for xv, tv in compare_pts:
    xt = jnp.array([xv, tv], dtype=jnp.float32)
    v_r, s_r = model.apply(params, xt)
    v_h, s_h = vs_ansatz(v_r, s_r, jnp.float32(xv), jnp.float32(tv))
    v_p = float(v_h); s_p = float(s_h)

    x_phys = xv * XMAX
    t_phys = tv * T_SCALE
    v_f = float(interp_v([[t_phys, x_phys]])) / V_SCALE
    s_f = float(interp_s([[t_phys, x_phys]])) / S_SCALE

    ratio = v_p / (v_f + 1e-10)
    print(f"  {xv:5.2f} {tv:5.2f} | {v_p:10.4f} {v_f:10.4f} {ratio:8.3f} | "
          f"{s_p:10.4f} {s_f:10.4f}")

print()
print(SEP)
print("7. PDE RESIDUALS AT TRAINED PARAMS")
print(SEP)

from loss_vs import residuals_single, CV as CV_loss, CS_COEFF as CS_loss

print(f"  CV in loss_vs = {CV_loss:.6f}")
print(f"  CS_COEFF in loss_vs = {CS_loss:.6f}")
print()
print(f"  {'x_nd':>5} {'t_nd':>5} | {'R_v':>12} {'R_s':>12} | {'dv/dt':>10} {'CV*ds/dx':>10}")

for xv, tv in [(0.47,0.05),(0.47,0.10),(0.53,0.05),(0.30,0.10),(0.70,0.10)]:
    rv, rs = residuals_single(params, model, jnp.float32(xv), jnp.float32(tv))
    print(f"  {xv:5.2f} {tv:5.2f} | {float(rv):12.6f} {float(rs):12.6f}")

print()
print(SEP)
print("8. THE TRIVIAL SOLUTION DIAGNOSIS")
print(SEP)

# If v_hat ≈ 0 and sigma_hat ≈ 0 everywhere for t>0,
# then R_v = dv/dt - CV*ds/dx = 0 - 0 = 0 ✓
# and R_s = ds/dt - CS*dv/dx = 0 - 0 = 0 ✓
# Loss = 0 but L2 = 100% (predicting zero vs non-zero FDM)
#
# The IC ansatz SHOULD prevent this:
# v_hat = tanh^2(B*t) * v_raw -> nonzero for t>0
# s_hat = s0*exp(-A*t) + tanh^2(B*t) * s_raw
#
# But if v_raw ≈ 0 and s_raw ≈ -s0/tanh^2(B*t)*exp(-A*t),
# then s_hat ≈ 0 too! This is the trivial solution.
#
# The network learned: v_raw ≈ 0, s_raw ≈ -sigma0 * decay/growth
# which makes BOTH v_hat and s_hat near zero for t > 0.05 or so.

print("  Checking if network learned trivial solution (v_raw≈0, s_hat≈0 for t>0):")
for xv in [0.47, 0.50, 0.53]:
    for tv in [0.05, 0.1, 0.2, 0.5]:
        xt = jnp.array([xv, tv], dtype=jnp.float32)
        v_r, s_r = model.apply(params, xt)
        v_h, s_h = vs_ansatz(v_r, s_r, jnp.float32(xv), jnp.float32(tv))
        if abs(float(v_h)) < 0.01 and tv > 0.02:
            print(f"    TRIVIAL: x={xv} t={tv}: v_hat={float(v_h):.4f}  s_hat={float(s_h):.4f}")

print()
print(SEP)
print("9. IC ANSATZ GROWTH/DECAY ANALYSIS")
print(SEP)
from model_vs import IC_A, IC_B
print(f"  IC_A={IC_A} (decay rate), IC_B={IC_B} (growth rate)")
for tv in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    decay  = float(jnp.exp(-IC_A * jnp.float32(tv)))
    growth = float(jnp.tanh(IC_B * jnp.float32(tv))**2)
    print(f"  t_nd={tv:.2f}: decay={decay:.4f}  growth={growth:.4f}  "
          f"decay/growth={decay/(growth+1e-8):.3f}")

print()
print("  If growth≈0 and decay≈1 for small t, the network can't learn")
print("  sigma_hat at small t (it's dominated by s0*decay, not s_raw*growth)")
print()
print("  At t_nd=0.05: growth/decay =", end=" ")
t = 0.05
print(f"{float(jnp.tanh(IC_B*t)**2):.4f}/{float(jnp.exp(-IC_A*t)):.4f} = "
      f"{float(jnp.tanh(IC_B*t)**2)/float(jnp.exp(-IC_A*t)):.4f}")
print("  This is the ratio of network contribution to IC contribution.")
print("  For training to work, both must be comparable.")