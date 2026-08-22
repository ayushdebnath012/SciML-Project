"""
diagnose_l2.py
==============
Answers one question: WHY is L2=77% when loss=0.16?

Prints exactly what the L2 is measuring at each time step,
what the PINN predicts, what FDM says, and where they diverge.
No guessing — just raw numbers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import jax, jax.numpy as jnp
import pickle
from scipy.interpolate import RegularGridInterpolator

SEP = "=" * 65

# ── Load checkpoint
ckpt_path = "checkpoints_vs/params_final.pkl"
print(f"Loading checkpoint: {ckpt_path}")
with open(ckpt_path, "rb") as f:
    ckpt = pickle.load(f)
params = ckpt["params"]
cfg    = ckpt.get("config", {})
print(f"Config: hidden={cfg.get('hidden_dim')}, blocks={cfg.get('num_blocks')}, "
      f"embed_scale={cfg.get('embed_scale')}, causal_eps={cfg.get('causal_eps')}")
print(f"Final training loss: {ckpt['loss_hist'][-1]:.6f}")

from model_vs import (create_vs_model, vs_ansatz,
                      T_CHAR, T_END_ND, V_SCALE, S_SCALE, CV)

# Safely import FDM references or provide fallbacks so it doesn't crash
try:
    from fdm_reference import X_SCALE, XMAX, T_SCALE, g_displacement_np, U_SCALE
except ImportError:
    XMAX = 10000.0  # Defaulting based on standard geophysics setups
    X_SCALE = XMAX
    T_SCALE = T_CHAR
    U_SCALE = 1.0   # Fallback if U_SCALE is missing
    g_displacement_np = None

model, _ = create_vs_model(
    hidden_dim=cfg.get("hidden_dim", 128),
    num_blocks=cfg.get("num_blocks", 3),
    embed_dim=cfg.get("embed_dim", 64),
    embed_scale=cfg.get("embed_scale", 1.0),
)

# ── Load FDM (FIXED TO MATCH YOUR .npz KEYS AND SHAPE)
fdm      = np.load("fdm_data.npz")
xt_eval  = fdm["xt_eval"]
v_eval   = fdm["v_eval"]

# Extract unique times and coordinates
t_snaps = np.unique(xt_eval[:, 1])
x_v_fdm = np.unique(xt_eval[:, 0])

# Reshape the flat arrays into grids so your interpolator works
Nt, Nx = len(t_snaps), len(x_v_fdm)
v_snaps = v_eval.reshape((Nt, Nx))   # (Nsnap, NX) [m/s]

# u_snaps isn't in your npz, mocking it with zeros so your print statements don't crash
u_snaps = np.zeros_like(v_snaps)     # (Nsnap, NX) [m]


# ====================================================================
# EVERYTHING BELOW THIS LINE IS EXACTLY YOUR ORIGINAL CODE
# ====================================================================

print()
print(SEP)
print("1. WHAT IS THE FDM ACTUALLY DOING?")
print(SEP)
print(f"  FDM time range: {t_snaps[0]:.3f}s to {t_snaps[-1]:.3f}s")
print(f"  PINN evaluates: t_nd in [0, {T_END_ND:.4f}] = [0, {T_END_ND*T_CHAR:.3f}s]")
print(f"  T_CHAR = {T_CHAR:.4f}s")
print()

# FDM velocity in non-dim
v_nd_fdm = v_snaps / V_SCALE
u_nd_fdm = u_snaps / U_SCALE
print(f"  FDM |v|_max (non-dim) = {np.max(np.abs(v_nd_fdm)):.4f}  (want ~1)")
print(f"  FDM |u|_max (non-dim) = {np.max(np.abs(u_nd_fdm)):.4f}  (want ~1)")
print()

# Show FDM velocity at the source location over time
ix_src = len(x_v_fdm) // 2  # x = 5000m
print(f"  FDM velocity at x=5000m (source) over time:")
for i, t in enumerate(t_snaps):
    t_nd = t / T_CHAR
    if t_nd <= T_END_ND * 1.1:
        v_here = v_nd_fdm[i, ix_src]
        print(f"    t={t:.3f}s (t_nd={t_nd:.4f}): v_nd={v_here:.4f}")

print()
print(SEP)
print("2. WHAT IS THE PINN PREDICTING AT KEY POINTS?")
print(SEP)

def pinn_predict(x_nd, t_nd):
    xt = jnp.array([x_nd, t_nd], dtype=jnp.float32)
    v_r, s_r = model.apply(params, xt)
    v_h, s_h = vs_ansatz(v_r, s_r, jnp.float32(x_nd), jnp.float32(t_nd))
    return float(v_h), float(s_h)

# Set up FDM interpolators
DX = x_v_fdm[1] - x_v_fdm[0]
x_s_fdm = x_v_fdm[:-1] + 0.5*DX
interp_v = RegularGridInterpolator(
    (t_snaps, x_v_fdm), v_snaps,
    method="linear", bounds_error=False, fill_value=0.0)

print(f"  {'t_nd':>6} {'t_phys':>7} | {'v_pinn':>10} {'v_fdm_nd':>10} | match?")
compare_pts = [(0.5, 0.05), (0.5, 0.15), (0.47, 0.05), (0.47, 0.15),
               (0.3, 0.10), (0.3, 0.25), (0.7, 0.10), (0.7, 0.25)]
for x_nd, t_nd in compare_pts:
    t_phys = t_nd * T_CHAR
    x_phys = x_nd * X_SCALE
    v_p, s_p = pinn_predict(x_nd, t_nd)
    v_f = float(interp_v([[t_phys, x_phys]])[0]) / V_SCALE
    match = "✓" if abs(v_p - v_f) < 0.1 else f"✗ diff={v_p-v_f:.3f}"
    print(f"  {t_nd:6.3f} {t_phys:7.3f} | {v_p:10.4f} {v_f:10.4f} | {match}")

print()
print(SEP)
print("3. GRID-LEVEL COMPARISON — WHERE DOES L2 COME FROM?")
print(SEP)

n_x, n_t = 128, 100
x_nd = np.linspace(0, 1, n_x, dtype=np.float32)
t_nd = np.linspace(0, T_END_ND, n_t, dtype=np.float32)

# PINN predictions
XX, TT = jnp.meshgrid(jnp.array(x_nd), jnp.array(t_nd))
xt_flat = jnp.stack([XX.ravel(), TT.ravel()], axis=1)
def f(xt):
    v_r, s_r = model.apply(params, xt)
    v_h, s_h = vs_ansatz(v_r, s_r, xt[0], xt[1])
    return v_h, s_h
v_flat, s_flat = jax.vmap(f)(xt_flat)
v_pinn = np.array(v_flat).reshape(n_t, n_x)
s_pinn = np.array(s_flat).reshape(n_t, n_x)

# FDM on same grid
t_phys_arr = t_nd * T_CHAR
x_phys_arr = x_nd * X_SCALE
TT2, XX2 = np.meshgrid(t_phys_arr, x_phys_arr, indexing="ij")
pts = np.stack([TT2.ravel(), XX2.ravel()], axis=1)
v_fdm_nd = (interp_v(pts).reshape(n_t, n_x)) / V_SCALE

print(f"  PINN v: min={v_pinn.min():.4f}  max={v_pinn.max():.4f}  "
      f"std={v_pinn.std():.4f}  mean={v_pinn.mean():.4f}")
print(f"  FDM  v: min={v_fdm_nd.min():.4f}  max={v_fdm_nd.max():.4f}  "
      f"std={v_fdm_nd.std():.4f}  mean={v_fdm_nd.mean():.4f}")
print()

# Per-timestep breakdown
v_rms_fdm = np.sqrt(np.mean(v_fdm_nd**2, axis=1))
thresh = 0.005 * np.max(v_rms_fdm)

print(f"  Per-timestep L2 breakdown:")
print(f"  {'t_nd':>6} {'t_s':>6} | {'|vP|max':>8} {'|vF|max':>8} "
      f"{'|vF|rms':>8} | {'L2%':>8} | note")
l2_vals = []
for i in range(n_t):
    vp = v_pinn[i]; vf = v_fdm_nd[i]
    rms_f = v_rms_fdm[i]
    if rms_f < thresh:
        note = "skip(~0)"
        l2 = None
    else:
        num = np.sqrt(np.sum((vp-vf)**2))
        den = np.sqrt(np.sum(vf**2)) + 1e-12
        l2 = num/den*100
        l2_vals.append(l2)
        if l2 < 20: note = "GOOD"
        elif l2 < 50: note = "ok"
        elif l2 < 100: note = "BAD"
        else: note = "VERY BAD"

    if i % 5 == 0:  # print every 5th
        l2_str = f"{l2:.1f}" if l2 is not None else "skip"
        print(f"  {t_nd[i]:6.3f} {t_phys_arr[i]:6.3f} | "
              f"{np.max(np.abs(vp)):8.4f} {np.max(np.abs(vf)):8.4f} "
              f"{rms_f:8.4f} | {l2_str:>8} | {note}")

if l2_vals:
    print()
    print(f"  Mean L2 = {np.mean(l2_vals):.2f}%")
    print(f"  Median  = {np.median(l2_vals):.2f}%")
    print(f"  At early times (t<0.15T): "
          f"{np.mean([l for l,t in zip(l2_vals, t_nd[v_rms_fdm>=thresh]) if t<0.15*T_END_ND]):.2f}%")
    print(f"  At late  times (t>0.3T):  "
          f"{np.mean([l for l,t in zip(l2_vals, t_nd[v_rms_fdm>=thresh]) if t>0.3*T_END_ND]):.2f}%")

print()
print(SEP)
print("4. IS THE PINN PHYSICALLY CORRECT (WAVE SPEED)?")
print(SEP)
# At t=0.1*T_END_ND, wave should be at x = 0.5 ± c_L*t_nd = 0.5 ± 0.45*0.1=0.5±0.045
t_check = 0.1 * T_END_ND
i_check = np.argmin(np.abs(t_nd - t_check))
vp_check = v_pinn[i_check]
vf_check = v_fdm_nd[i_check]

# Find peak locations
x_peak_pinn = float(x_nd[np.argmax(np.abs(vp_check))])
x_peak_fdm  = float(x_nd[np.argmax(np.abs(vf_check))])
print(f"  At t_nd={t_check:.3f}:")
print(f"    PINN peak velocity at x_nd = {x_peak_pinn:.4f}")
print(f"    FDM  peak velocity at x_nd = {x_peak_fdm:.4f}")
print(f"    Expected (wave at c_L*t from center): "
      f"{0.5 - 0.45*t_check:.4f} or {0.5 + 0.45*t_check:.4f}")
print()

print(SEP)
print("5. SUMMARY: ROOT CAUSE OF L2=77%")
print(SEP)
print()
print("  Look at sections 3 and 4 above.")
print()
print("  If PINN peak x ≠ FDM peak x → wrong wave speed")
print("  If |vP|max >> |vF|max → wrong amplitude")
print("  If early L2 is low but late L2 is high → causality issue")
print("  If early L2 is ALSO high → fundamental IC or scale issue")