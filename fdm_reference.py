"""
fdm_reference.py
================
Finite difference reference — matches the user's heterogeneous FDM exactly.

Material: left half homogeneous c=2500, right half sinusoidal heterogeneous.
Source:   Ricker wavelet at x=5000m, centred at t0=1.5/f0=0.3s.
Damping:  200-point quadratic sponge layers at both boundaries.

All scales exported for the PINN (single source of truth).
"""
import numpy as np

# ── Grid ──────────────────────────────────────────────────────────────────────
NX   = 800
XMAX = 10_000.0
DX   = XMAX / NX
x_v  = np.linspace(0, XMAX, NX)          # velocity grid
x_s  = x_v[:-1] + 0.5 * DX              # stress grid (staggered)

# ── Material ──────────────────────────────────────────────────────────────────
RHO     = 2500.0
ALPHA   = 0.4
L_HET   = 2000.0

def _wave_speed(x):
    c = np.ones_like(x) * 2500.0
    mask = x > XMAX / 2
    c[mask] = 2500.0 * (1.0 + ALPHA * np.sin(2*np.pi*(x[mask] - XMAX/2) / L_HET))
    return c

c_v  = _wave_speed(x_v)
c_s  = _wave_speed(x_s)
mu_v = RHO * c_v**2
mu_s = RHO * c_s**2

C_MAX = 2500.0 * (1.0 + ALPHA)           # = 3500 m/s

# ── Time ──────────────────────────────────────────────────────────────────────
CFL   = 0.5
DT    = CFL * DX / C_MAX
NT    = 1200
T_END = NT * DT                          # ~ 0.857 s

# ── Source ────────────────────────────────────────────────────────────────────
F0        = 5.0
T0        = 1.5 / F0                     # = 0.3 s (Ricker peak)
SRC_I     = NX // 2                      # index = 400 → x = 5000 m
EPSILON   = 3e-5
MU_REF    = RHO * 2500.0**2
F_SOURCE  = MU_REF * EPSILON / DX        # source amplitude

def ricker_np(t):
    a = np.pi * F0 * (t - T0)
    return (1.0 - 2.0*a**2) * np.exp(-a**2)

# ── Damping (200-pt quadratic sponge — matches user's FDM exactly) ─────────────
NB       = 200
DAMP_MAX = 5.0
damp_v   = np.zeros(NX)
damp_s   = np.zeros(NX - 1)
for i in range(NB):
    w = ((NB - i) / NB)**2 * DAMP_MAX
    damp_v[i]      = w
    damp_v[NX-1-i] = w
    if i < NX - 1:     damp_s[i]      = w
    if (NX-2-i) >= 0:  damp_s[NX-2-i] = w

# ── Run FDM ───────────────────────────────────────────────────────────────────
def run_fdm(save_every=10, verbose=True):
    v     = np.zeros(NX)
    v_old = np.zeros(NX)
    sigma = np.zeros(NX - 1)

    t_snaps, v_snaps, s_snaps = [], [], []

    # Store t=0
    t_snaps.append(0.0)
    v_snaps.append(v.copy())
    s_snaps.append(sigma.copy())

    for it in range(1, NT):
        t = it * DT

        # Velocity update
        v[1:-1] += (DT / (RHO * DX)) * (sigma[1:] - sigma[:-1])
        v       *= np.exp(-damp_v * DT)

        # Source injection
        v[SRC_I] += DT * F_SOURCE * ricker_np(t - T0) / RHO

        # Stress update
        sigma += (DT / DX) * mu_s * (v[1:] - v[:-1])
        sigma *= np.exp(-damp_s * DT)

        v_old[:] = v[:]

        if it % save_every == 0:
            t_snaps.append(t)
            v_snaps.append(v.copy())
            s_snaps.append(sigma.copy())
            if verbose and it % (save_every * 100) == 0:
                print(f"  FDM t={t:.3f}s  |v|={np.max(np.abs(v)):.3e}")

    return np.array(t_snaps), x_v, np.array(v_snaps), np.array(s_snaps)


def save_reference(path="fdm_data.npz", save_every=10):
    print("Running FDM reference simulation...")
    t_snaps, x_v_out, v_snaps, s_snaps = run_fdm(save_every=save_every)

    # Coarsened eval grid (physical coords — PINN normalises itself)
    x_idx = np.linspace(0, NX-1,           256, dtype=int)
    t_idx = np.linspace(0, len(t_snaps)-1, 100, dtype=int)

    XX, TT = np.meshgrid(x_v_out[x_idx], t_snaps[t_idx])
    xt_eval = np.stack([XX.ravel(), TT.ravel()], axis=1).astype(np.float32)

    vc = v_snaps[np.ix_(t_idx, x_idx)]

    # Interpolate stress from staggered grid to velocity grid
    s_on_v          = np.zeros_like(v_snaps)
    s_on_v[:, 1:-1] = 0.5 * (s_snaps[:, :-1] + s_snaps[:, 1:])
    s_on_v[:, 0]    = s_snaps[:, 0]
    s_on_v[:, -1]   = s_snaps[:, -1]
    sc = s_on_v[np.ix_(t_idx, x_idx)]

    # Also save full snapshot arrays (used for data-loss training)
    np.savez(path,
             t_snaps = t_snaps,
             x_v     = x_v_out,
             v_snaps = v_snaps,       # (N_snap, NX)
             s_snaps = s_on_v,        # (N_snap, NX) interpolated to v-grid
             xt_eval = xt_eval,
             v_eval  = vc.ravel()[:, None].astype(np.float32),
             s_eval  = sc.ravel()[:, None].astype(np.float32))

    print(f"Saved {path}")
    print(f"  |v|_max = {np.max(np.abs(v_snaps)):.4e} m/s")
    print(f"  |s|_max = {np.max(np.abs(s_snaps)):.4e} Pa")
    print(f"  T_END   = {T_END:.4f} s  ({len(t_snaps)} snapshots)")
    return path


# ── PINN non-dimensionalisation (single source of truth) ─────────────────────
#
#   x̃ = x / L
#   t̃ = t / T_CHAR   where T_CHAR = L / C_REF
#   ṽ = v / V_SCALE
#   σ̃ = σ / S_SCALE
#
#   Choosing V_SCALE and S_SCALE so BOTH PDE residuals are O(1):
#
#   From FDM: |v|_max ≈ 0.08 m/s, |σ|_max ≈ 1e6 Pa
#   We set scales = observed max values so network outputs are ≈ ±1.
#
L       = XMAX
C_REF   = 2500.0                         # reference wave speed [m/s]
T_CHAR  = L / C_REF                      # = 4.0 s
T_END_ND = T_END / T_CHAR               # ≈ 0.214

V_SCALE = 0.08                           # m/s  (from FDM plot limits)
S_SCALE = RHO * L * V_SCALE / T_CHAR    # = 5e5 Pa  (makes CV=1 exactly)

# CV check:  dv/dt = (1/rho)*dsigma/dx
#   (V_SCALE/T_CHAR)*dv_nd/dt_nd = (1/rho)*(S_SCALE/L)*dsigma_nd/dx_nd
#   CV = S_SCALE * T_CHAR / (rho * L * V_SCALE)
CV_CHECK = S_SCALE * T_CHAR / (RHO * L * V_SCALE)
print(f"[fdm_reference] CV_CHECK = {CV_CHECK:.4f}  (want ≈1.0)")
print(f"[fdm_reference] T_CHAR={T_CHAR:.3f}s  T_END_ND={T_END_ND:.4f}")
print(f"[fdm_reference] V_SCALE={V_SCALE}  S_SCALE={S_SCALE:.2e}")


if __name__ == "__main__":
    save_reference()