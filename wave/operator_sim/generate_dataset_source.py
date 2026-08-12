"""Generate the forced-wave FD dataset (source term + absorbing boundaries).

Material profiles reuse the repo's sampler; the wave is driven by a Ricker
point source from a quiescent start rather than by an initial pulse.

Input channels (6):  [E(x), rho(x), s(x), w(t), x, t]
  s(x)  spatial source profile (narrow Gaussian at x_src)
  w(t)  Ricker wavelet in time
The source is separable, S(x,t) = s(x) w(t), so these two channels carry it
exactly.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wave"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fd_source import solve_wave_1d_source, source_profile  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "rfb", ROOT / "wave" / "run_fno_baseline.py")
rfb = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(rfb)
except SystemExit:
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--nt", type=int, default=80)
    p.add_argument("--t-max", type=float, default=2.0)
    p.add_argument("--cfl", type=float, default=0.35)
    p.add_argument("--src-width", type=float, default=0.04)
    p.add_argument("--src-amplitude", type=float, default=750.0)
    p.add_argument("--f-min", type=float, default=3.0)
    p.add_argument("--f-max", type=float, default=6.0)
    p.add_argument("--xs-range", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    x_grid = np.linspace(-1.0, 1.0, a.nx)

    inputs, outputs, kinds = [], [], []
    x_srcs, f_peaks = [], []
    t0 = time.time()
    print(f"Generating {a.num_samples} forced FD samples...")
    for i in range(a.num_samples):
        E, rho, kind = rfb.sample_material_profile(rng, x_grid)
        x_src = float(rng.uniform(-a.xs_range, a.xs_range))
        f_peak = float(rng.uniform(a.f_min, a.f_max))
        s_x = source_profile(x_grid, x_src, a.src_width)

        _, t_grid, u, w_t = solve_wave_1d_source(
            E, rho, s_x, f_peak, a.src_amplitude,
            x_limits=(-1.0, 1.0), t_limits=(0.0, a.t_max),
            nx_out=a.nx, nt_out=a.nt, CFL=a.cfl,
        )

        nx, nt = a.nx, a.nt
        channels = np.stack([
            np.repeat(E[:, None], nt, axis=1),
            np.repeat(rho[:, None], nt, axis=1),
            np.repeat(s_x[:, None], nt, axis=1),
            np.repeat(w_t[None, :], nx, axis=0),
            np.repeat(x_grid[:, None], nt, axis=1),
            np.repeat(t_grid[None, :], nx, axis=0),
        ], axis=0)

        inputs.append(channels.astype(np.float32))
        outputs.append(u[None, ...].astype(np.float32))
        kinds.append(kind)
        x_srcs.append(x_src)
        f_peaks.append(f_peak)
        if (i + 1) % max(1, a.num_samples // 8) == 0:
            print(f"  {i + 1}/{a.num_samples}  ({time.time() - t0:.1f}s)")

    inputs = np.stack(inputs).astype(np.float32)
    outputs = np.stack(outputs).astype(np.float32)
    np.savez_compressed(
        a.out,
        x=x_grid.astype(np.float32), t=t_grid.astype(np.float32),
        inputs=inputs, outputs=outputs, kinds=np.asarray(kinds),
        x_src=np.asarray(x_srcs, dtype=np.float32),
        f_peak=np.asarray(f_peaks, dtype=np.float32),
        src_width=np.float32(a.src_width),
        src_amplitude=np.float32(a.src_amplitude),
    )
    u_, c_ = np.unique(kinds, return_counts=True)
    print("saved", a.out, inputs.shape, outputs.shape,
          dict(zip(u_.tolist(), c_.tolist())))
    print(f"max|u| = {np.abs(outputs).max():.3f}   total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
