"""2D acoustic forward modelling reproducing the OpenFWI acquisition exactly.

    u_tt = c(z,x)^2 * laplacian(u) + s(t) delta(z_s, x_s)

Config from OpenFWI dataset_config.json (identical across all 2D families):
70x70 grid at dx = 10 m, 5 shots, 70 receivers, 1000 steps at dt = 1 ms,
15 Hz Ricker source, sources and receivers at grid depth 10, nbc = 120.

CFL = v_max*dt/dx = 4500*0.001/10 = 0.45 < 1/sqrt(2) -> stable.
"""
import numpy as np

NZ = NX = 70
DX = 10.0
NT = 1000
DT = 0.001
FREQ = 15.0
NS = 5
NG = 70
SZ = 10          # source / receiver depth index
NBC = 120        # absorbing pad
VMIN, VMAX = 1500.0, 4500.0


def ricker(nt, dt, f, shift=None):
    t = np.arange(nt) * dt
    if shift is None:
        shift = 1.2 / f
    a = (np.pi * f * (t - shift)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def vel_flat(seed=0, nlayer=4):
    """FlatVel-A style: horizontal layers, velocity increasing with depth."""
    rng = np.random.default_rng(seed)
    edges = np.sort(rng.integers(8, NZ - 6, nlayer - 1))
    vals = np.sort(rng.uniform(VMIN, VMAX, nlayer))
    v = np.empty((NZ, NX))
    bounds = [0, *edges, NZ]
    for i in range(nlayer):
        v[bounds[i]:bounds[i + 1], :] = vals[i]
    return v


def vel_curve(seed=0, nlayer=4, amp=7.0):
    """CurveVel-A style: the same layers, interfaces bent by a smooth curve."""
    rng = np.random.default_rng(seed)
    edges = np.sort(rng.integers(12, NZ - 12, nlayer - 1))
    vals = np.sort(rng.uniform(VMIN, VMAX, nlayer))
    x = np.arange(NX)
    phase, k = rng.uniform(0, 2 * np.pi), rng.uniform(0.8, 1.8)
    bend = amp * np.sin(2 * np.pi * k * x / NX + phase)
    v = np.empty((NZ, NX))
    v[:] = vals[-1]
    for i in range(nlayer - 1, -1, -1):
        top = 0 if i == 0 else edges[i - 1] + bend
        for j in range(NX):
            t0 = 0 if i == 0 else int(np.clip(round(top[j]), 0, NZ - 1))
            b0 = NZ if i == nlayer - 1 else int(np.clip(round(edges[i] + bend[j]), 0, NZ - 1))
            v[t0:b0, j] = vals[i]
    return v


def _sponge(n, nbc, strength=0.0015):
    """Multiplicative damping profile over the padded region."""
    d = np.ones(n)
    ramp = np.arange(nbc, 0, -1) / nbc
    taper = np.exp(-(strength * nbc * ramp) ** 2)
    d[:nbc] = taper
    d[-nbc:] = taper[::-1]
    return d


def forward(vel, snap_at=(), shots=None, nt=NT):
    """Run all shots. Returns gathers (ns, nt, ng) and {t: wavefield} snapshots."""
    nz, nx = vel.shape
    pz, px = nz + 2 * NBC, nx + 2 * NBC
    c = np.pad(vel, NBC, mode="edge")
    coef = (c * DT / DX) ** 2

    dz = _sponge(pz, NBC)[:, None]
    dxx = _sponge(px, NBC)[None, :]
    damp = dz * dxx

    src = ricker(nt, DT, FREQ)
    sx = np.linspace(0, nx - 1, NS).round().astype(int) if shots is None else np.asarray(shots)
    rx = np.arange(NG) + NBC
    rz = SZ + NBC

    gathers = np.zeros((len(sx), nt, NG))
    snaps = {}
    for si, s in enumerate(sx):
        u_prev = np.zeros((pz, px))
        u_cur = np.zeros((pz, px))
        for it in range(nt):
            lap = (u_cur[:-2, 1:-1] + u_cur[2:, 1:-1] +
                   u_cur[1:-1, :-2] + u_cur[1:-1, 2:] - 4.0 * u_cur[1:-1, 1:-1])
            u_next = np.zeros_like(u_cur)
            u_next[1:-1, 1:-1] = (2.0 * u_cur[1:-1, 1:-1] - u_prev[1:-1, 1:-1]
                                  + coef[1:-1, 1:-1] * lap)
            u_next[SZ + NBC, s + NBC] += coef[SZ + NBC, s + NBC] * src[it]
            u_next *= damp
            u_cur_damped = u_cur * damp
            u_prev, u_cur = u_cur_damped, u_next
            gathers[si, it, :] = u_cur[rz, rx]
            if si == len(sx) // 2 and it in snap_at:
                snaps[it] = u_cur[NBC:NBC + nz, NBC:NBC + nx].copy()
    return gathers, snaps, sx


if __name__ == "__main__":
    import time, os
    out = os.path.dirname(os.path.abspath(__file__))
    SNAPS = (60, 120, 200, 300, 450, 650)
    for name, fn, seed in [("flat", vel_flat, 3), ("curve", vel_curve, 11)]:
        v = fn(seed)
        t0 = time.time()
        g, sn, sx = forward(v, snap_at=SNAPS)
        print(f"{name}: vel {v.min():.0f}-{v.max():.0f} m/s | gathers {g.shape} "
              f"| |g|max {np.abs(g).max():.3e} | {time.time()-t0:.1f}s")
        np.savez_compressed(f"{out}/sim_{name}.npz", vel=v, gathers=g,
                            shots=sx, **{f"snap_{k}": val for k, val in sn.items()})
