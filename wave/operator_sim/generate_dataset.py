"""Generate the synthetic FD operator dataset using the repo's own sampler/solver.

`--refine` controls the FD grid the target is solved on, relative to the 64-point
output grid. It defaults to 8: at refine=1 (what this script did originally) the
target carries 10.4 % mean rel L2 error on its own layered/two_layer profiles,
which is larger than the error the best operator achieves against it. See the
`--refine` table in README.md.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wave"))

from wave_problem import (  # noqa: E402
    make_input_tensor, sample_material_profile, solve_case_refined,
)

p = argparse.ArgumentParser()
p.add_argument("--num-samples", type=int, default=512)
p.add_argument("--nx", type=int, default=64)
p.add_argument("--nt", type=int, default=64)
p.add_argument("--t-max", type=float, default=1.0)
p.add_argument("--sigma-g", type=float, default=0.1)
p.add_argument("--cfl", type=float, default=0.35)
p.add_argument("--refine", type=int, default=8,
               help="FD spatial refinement over the output grid; 1 reproduces "
                    "the original un-converged target")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", type=str, required=True)
a = p.parse_args()

rng = np.random.default_rng(a.seed)
x_grid = np.linspace(-1.0, 1.0, a.nx)
t_grid = np.linspace(0.0, a.t_max, a.nt)

inputs, outputs, kinds = [], [], []
t0 = time.time()
print(f"Generating {a.num_samples} samples "
      f"(FD at nx={(a.nx - 1) * a.refine + 1}, output {a.nx}x{a.nt})...")
for i in range(a.num_samples):
    E, rho, kind = sample_material_profile(rng, x_grid)
    u = solve_case_refined(x_grid, t_grid, E, rho, a.sigma_g, 0.0, a.cfl, a.refine)
    inputs.append(make_input_tensor(x_grid, t_grid, E, rho, a.sigma_g, 0.0))
    outputs.append(u[None, ...].astype(np.float32))
    kinds.append(kind)
    if (i + 1) % max(1, a.num_samples // 8) == 0:
        print(f"  {i+1}/{a.num_samples}  ({time.time()-t0:.1f}s)", flush=True)

inputs = np.stack(inputs).astype(np.float32)
outputs = np.stack(outputs).astype(np.float32)
kinds = np.asarray(kinds)
np.savez_compressed(a.out, x=x_grid.astype(np.float32), t=t_grid.astype(np.float32),
                    inputs=inputs, outputs=outputs, kinds=kinds,
                    refine=np.int32(a.refine))
u, c = np.unique(kinds, return_counts=True)
print("saved", a.out, inputs.shape, outputs.shape, dict(zip(u.tolist(), c.tolist())))
print(f"total {time.time()-t0:.1f}s")
