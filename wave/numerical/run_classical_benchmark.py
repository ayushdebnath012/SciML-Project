"""Accuracy-vs-cost study for the classical solvers on the PINN problem.

This produces the baseline the coordinate networks are measured against: for
each material and each method, the relative L2 error on the shared evaluation
grid and the wall-clock time to reach it.

Ground truth is a **converged Chebyshev collocation solve**, not a fine
finite-difference solve. That choice is forced by what the study found: the
finite-difference scheme's first-order Mur absorbing boundary puts a floor
under its error at roughly 4e-4 relative, and refining the grid past ~1600
cells does not move it. Scoring the solvers against a fine FD solve would
therefore measure their agreement with that floor rather than with the
solution. The spectral solver imposes the radiation condition exactly on the
characteristic variables, has no such floor, and self-converges to 1e-12 on
the homogeneous case -- so `--reference-n` and `--reference-check-n` are run
as a pair and their difference is reported alongside every number.

Usage:
    python wave/numerical/run_classical_benchmark.py --out results.json
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from material_profiles import Material                       # noqa: E402
from classical_solvers import SOLVERS                        # noqa: E402

MATERIALS = ["Homogeneous", "TwoLayer", "MultiLayer"]

# Grid resolutions per method. `cheb` counts polynomial degree, not cells, so
# its ladder covers the same accuracy range with far fewer entries.
LADDERS = {
    "fd2": [50, 100, 200, 400, 800, 1600, 3200, 6400],
    "fem_p1": [50, 100, 200, 400, 800, 1600, 3200, 6400],
    "cheb": [32, 48, 64, 96, 128, 192, 256],
}

# Resolutions the PINN runs' own reference solver uses, plus its neighbours.
PINN_REFERENCE_LADDER = [250, 500, 1000, 2000, 4000, 8000]


def rel_l2(pred, ref):
    return 100.0 * float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def timed(fn, *args, repeats=1, **kwargs):
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return out, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nx-eval", type=int, default=201)
    p.add_argument("--nt-eval", type=int, default=101)
    p.add_argument("--t-max", type=float, default=1.0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--reference-n", type=int, default=512)
    p.add_argument("--reference-check-n", type=int, default=384)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    results = {
        "meta": {
            "machine": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "reference": {"method": "cheb", "n": a.reference_n,
                          "self_check_n": a.reference_check_n},
            "eval_grid": {"nx": a.nx_eval, "nt": a.nt_eval, "t_max": a.t_max},
        },
        "reference_self_convergence": [],
        "pinn_reference_error": [],
        "convergence": [],
    }

    for name in MATERIALS:
        material = Material(name)
        x_out = np.linspace(material.x_min, material.x_max, a.nx_eval)
        t_out = np.linspace(0.0, a.t_max, a.nt_eval)
        print(f"\n=== {name} ===", flush=True)

        reference, ref_secs = timed(SOLVERS["cheb"], material, x_out, t_out,
                                    a.reference_n)
        check, _ = timed(SOLVERS["cheb"], material, x_out, t_out,
                         a.reference_check_n)
        self_err = rel_l2(check, reference)
        results["reference_self_convergence"].append(
            {"material": name, "n": a.reference_n, "check_n": a.reference_check_n,
             "rel_l2_percent": self_err, "seconds": ref_secs})
        print(f"  reference cheb n={a.reference_n} ({ref_secs:.1f}s); "
              f"n={a.reference_check_n} differs by {self_err:.3e}%", flush=True)

        # What the PINN runs actually score against: fd2 at Nx=1000.
        for nx in PINN_REFERENCE_LADDER:
            u, secs = timed(SOLVERS["fd2"], material, x_out, t_out, nx, cfl=0.9)
            row = {"material": name, "nx": nx,
                   "rel_l2_percent": rel_l2(u, reference), "seconds": secs}
            results["pinn_reference_error"].append(row)
            print(f"    fd2 nx={nx:<6d} {row['rel_l2_percent']:9.5f}%  {secs:6.3f}s",
                  flush=True)

        for method, ladder in LADDERS.items():
            for n in ladder:
                u, secs = timed(SOLVERS[method], material, x_out, t_out, n,
                                repeats=a.repeats)
                row = {"material": name, "method": method, "n": n,
                       "rel_l2_percent": rel_l2(u, reference), "seconds": secs}
                results["convergence"].append(row)
                print(f"  {method:7s} n={n:<6d} {row['rel_l2_percent']:10.5f}%  "
                      f"{secs:8.4f}s", flush=True)

    Path(a.out).write_text(json.dumps(results, indent=2))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
