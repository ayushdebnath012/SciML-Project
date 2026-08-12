"""Generate the FD operator dataset from real published velocity models.

Same problem, same solver, same 5-channel layout as generate_dataset.py -- the
only change is where E(x) and rho(x) come from. `--model` selects one of the
models registered in velocity_models.MODELS; see that module for provenance
and for how a depth column is reduced to the operator grid.

Two differences from the synthetic generator, both forced by the real data:

  * rho may vary (Marmousi ships a density model; Overthrust does not, and
    gets rho == 1 like the synthetic arm).
  * the FD solve runs on a refined grid (`--refine`, default 8) and is sampled
    back onto the output grid, rather than being solved at output resolution.
    Real columns hold much sharper contrasts than the synthetic tanh profiles,
    and at 64 points the coarse solve is not converged. Space is now
    refined-then-sampled the same way time already was.

Output npz matches the synthetic one (x, t, inputs, outputs, kinds) plus a
`split` array and per-sample provenance for analysis.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wave"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wave_problem import make_input_tensor, solve_case_refined  # noqa: E402
from velocity_models import (  # noqa: E402
    CONTRAST_TERCILES, MODELS, load_model, make_trace_split, reference_values,
    sample_profile,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="marmousi", choices=sorted(MODELS),
                   help="which published velocity model to draw profiles from")
    p.add_argument("--raw-dir", default=str(ROOT / "operator_data" / "raw"))
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--nt", type=int, default=64)
    p.add_argument("--t-max", type=float, default=1.0)
    p.add_argument("--sigma-g", type=float, default=0.1)
    p.add_argument("--cfl", type=float, default=0.35)
    p.add_argument("--refine", type=int, default=8,
                   help="FD spatial refinement over the output grid. Measured "
                        "rel L2 against a refine=32 reference: 1 -> 13.4 %%, "
                        "2 -> 3.9 %%, 4 -> 1.1 %%, 8 -> 0.32 %%, i.e. well under "
                        "the ~2 %% the best operator achieves.")
    p.add_argument("--min-window", type=int, default=0,
                   help="shortest depth window in samples; 0 = 40 %% of the model depth")
    p.add_argument("--max-window", type=int, default=0,
                   help="longest depth window; 0 = full usable depth")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--val-blocks", type=int, default=4,
                   help="number of contiguous trace blocks reserved for validation")
    p.add_argument("--buffer-m", type=float, default=320.0,
                   help="metres discarded either side of each validation block")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    vp, rho_model, split_pos, spec = load_model(a.model, a.raw_dir)
    vp_ref, rho_ref = reference_values(vp, rho_model)
    terciles = CONTRAST_TERCILES[a.model]
    dz = spec["dz"]
    print(f"{spec['label']}: {vp.shape[0]} traces x {vp.shape[1]} depth samples "
          f"({vp.shape[1] * dz:.0f} m at {dz:.0f} m)")
    print(f"  Vp  {vp.min():.0f}-{vp.max():.0f} m/s   ref {vp_ref:.0f}")
    print(f"  rho {rho_model.min():.0f}-{rho_model.max():.0f}  ref {rho_ref:.0f}"
          + ("   (no density model; rho = 1)" if "rho" not in spec["files"] else ""))

    train_traces, val_traces = make_trace_split(
        split_pos, a.val_fraction, a.val_blocks, a.buffer_m)
    print(f"  trace split: {len(train_traces)} train / {len(val_traces)} val "
          f"({a.val_blocks} blocks, {a.buffer_m:.0f} m buffers, "
          f"{len(split_pos) - len(train_traces) - len(val_traces)} discarded)")

    rng = np.random.default_rng(a.seed)
    x_grid = np.linspace(-1.0, 1.0, a.nx)
    t_grid = np.linspace(0.0, a.t_max, a.nt)

    n_val = max(1, int(round(a.val_fraction * a.num_samples)))
    pools = ["train"] * (a.num_samples - n_val) + ["val"] * n_val
    win_kw = {}
    if a.min_window > 0:
        win_kw["min_window"] = a.min_window
    if a.max_window > 0:
        win_kw["max_window"] = a.max_window

    inputs, outputs, kinds, splits = [], [], [], []
    traces, tops, windows, contrasts, depths = [], [], [], [], []
    t0 = time.time()
    print(f"Generating {a.num_samples} samples "
          f"(FD at nx={(a.nx - 1) * a.refine + 1}, output {a.nx}x{a.nt})...")
    for i, split in enumerate(pools):
        pool = train_traces if split == "train" else val_traces
        E, rho, kind, meta = sample_profile(
            rng, vp, rho_model, a.nx, vp_ref, rho_ref, terciles,
            trace_pool=pool, dz=dz, **win_kw)
        u = solve_case_refined(x_grid, t_grid, E, rho,
                               a.sigma_g, 0.0, a.cfl, a.refine)
        inputs.append(make_input_tensor(x_grid, t_grid, E, rho, a.sigma_g, 0.0))
        outputs.append(u[None, ...].astype(np.float32))
        kinds.append(kind)
        splits.append(split)
        traces.append(meta["trace"])
        tops.append(meta["top"])
        windows.append(meta["window"])
        contrasts.append(meta["contrast"])
        depths.append(meta["depth_centre_m"])
        if (i + 1) % max(1, a.num_samples // 8) == 0:
            print(f"  {i + 1}/{a.num_samples}  ({time.time() - t0:.1f}s)", flush=True)

    inputs = np.stack(inputs).astype(np.float32)
    outputs = np.stack(outputs).astype(np.float32)
    np.savez_compressed(
        a.out,
        x=x_grid.astype(np.float32), t=t_grid.astype(np.float32),
        inputs=inputs, outputs=outputs, kinds=np.asarray(kinds),
        split=np.asarray(splits),
        model=np.asarray(a.model),
        trace=np.asarray(traces, dtype=np.int32),
        top=np.asarray(tops, dtype=np.int32),
        window=np.asarray(windows, dtype=np.int32),
        contrast=np.asarray(contrasts, dtype=np.float32),
        depth_centre_m=np.asarray(depths, dtype=np.float32),
        val_traces=val_traces.astype(np.int32),
        vp_ref=np.float32(vp_ref), rho_ref=np.float32(rho_ref),
        refine=np.int32(a.refine),
    )
    u_, c_ = np.unique(kinds, return_counts=True)
    print("saved", a.out, inputs.shape, outputs.shape,
          dict(zip(u_.tolist(), c_.tolist())))
    s_, sc_ = np.unique(splits, return_counts=True)
    print("  split", dict(zip(s_.tolist(), sc_.tolist())))
    print(f"  E~   {inputs[:, 0].min():.3f}-{inputs[:, 0].max():.3f}")
    print(f"  rho~ {inputs[:, 1].min():.3f}-{inputs[:, 1].max():.3f}")
    print(f"  max|u| = {np.abs(outputs).max():.3f}   total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
