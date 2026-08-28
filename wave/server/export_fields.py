"""Export predicted wave fields from saved checkpoints, for the paper figures.

The per-run `solution_comparison.png` the runner writes is a 300-dpi three-panel
plot in the runner's own styling. That is fine for inspecting a run and wrong
for a paper figure. This exports the underlying arrays instead -- the FD
reference and each selected model's prediction on the same grid -- so the figure
can be drawn locally with the paper's typography.

Prediction goes through the same code path the runner uses to compute the
reported error (`_call_model_jax` then `apply_ansatz_jax`), so a field exported
here and the number in `l2_errors.json` cannot disagree.

    python wave/server/export_fields.py --runs <dir> [<dir> ...] --out fields.npz
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wave"))


def build_model(cfg, seed=42):
    """Rebuild an architecture from the config recorded in l2_errors.json."""
    import run_experiment as rx
    return rx.instantiate_model_jax(cfg, seed=seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="run directories, each holding l2_errors.json and a checkpoint")
    p.add_argument("--nx", type=int, default=400, help="output grid in x")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    import jax.numpy as jnp
    import run_experiment as rx
    from src.train_jax import load_model_jax
    from src.models_jax import _call_model_jax
    from src.losses.wave_loss_jax import apply_ansatz_jax
    from wave.materials import HomogeneousModel, TwoLayerModel, MultiLayerModel

    materials = {"Homogeneous": HomogeneousModel, "TwoLayer": TwoLayerModel,
                 "MultiLayer": MultiLayerModel}
    rx._BACKEND = "jax"
    sigma_g = rx.CONFIG["sigma_g"]

    out = {}
    fd_cache = {}
    for run_dir in a.runs:
        run_dir = Path(run_dir)
        meta_path = run_dir / "l2_errors.json"
        if not meta_path.exists():
            print(f"skip {run_dir}: no l2_errors.json"); continue
        meta = json.loads(meta_path.read_text())
        name = meta["material_type"]
        material = materials[name]()

        if name not in fd_cache:
            fd_cache[name] = rx._compute_fd_reference(material, rx.CONFIG)
        x_fd, t_fd, u_fd, xm_fd, tm_fd = fd_cache[name]

        cfg = {"model_type": meta["model_type"], "n_hidden": meta["n_hidden"],
               "hidden_width": meta["hidden_width"],
               "extra_params": meta.get("extra_params", {})}
        template = build_model(cfg, seed=rx.CONFIG["seed"])

        ckpt = next((run_dir / n for n in ("model_best_lbfgs.eqx",
                                           "model_best_lbfgs_only.eqx",
                                           "model_best_adam.eqx")
                     if (run_dir / n).exists()), None)
        if ckpt is None:
            print(f"skip {run_dir}: no checkpoint"); continue
        model = load_model_jax(str(ckpt), template)

        x_ev = jnp.array(xm_fd.reshape(-1, 1), dtype=jnp.float32)
        t_ev = jnp.array(tm_fd.reshape(-1, 1), dtype=jnp.float32)
        chunk, outs = 200_000, []
        for i in range(0, x_ev.shape[0], chunk):
            xb, tb = x_ev[i:i + chunk], t_ev[i:i + chunk]
            nn_out = _call_model_jax(model, xb, tb)
            outs.append(np.array(apply_ansatz_jax(nn_out, xb, tb, sigma_g,
                                                  bool(meta["use_ansatz"]))))
        pred = np.concatenate(outs, axis=0).reshape(len(x_fd), -1)
        rel = 100 * np.linalg.norm(u_fd - pred) / np.linalg.norm(u_fd)
        print(f"{run_dir.name}: rel L2 {rel:.3f}%  (json says "
              f"{meta.get('relative_l2_error_best_lbfgs_percent', float('nan')):.3f}%)")

        # Subsample in x; the FD grid is 1001 points and the figure needs ~400.
        step = max(1, len(x_fd) // a.nx)
        key = f"{name}|{run_dir.name}"
        out[f"pred|{key}"] = pred[::step].astype(np.float32)
        out[f"rel|{key}"] = np.float32(rel)
        out[f"x|{name}"] = x_fd[::step].astype(np.float32)
        out[f"t|{name}"] = t_fd.astype(np.float32)
        out[f"ref|{name}"] = u_fd[::step].astype(np.float32)

    np.savez_compressed(a.out, **out)
    print("wrote", a.out, f"({len(out)} arrays)")


if __name__ == "__main__":
    main()
