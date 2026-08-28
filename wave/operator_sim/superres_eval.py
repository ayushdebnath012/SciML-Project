"""Zero-shot super-resolution: train at 64x64, evaluate on a different grid.

FNO and PFNO are usually sold as discretization-invariant -- their weights live
on Fourier modes rather than grid points, so the same trained operator should
apply at any resolution. That claim has never been tested in this project.

What each architecture can do here follows from its construction:

  FNO       both axes free -- rfft2 over (x, t), modes truncated the same way
            at any grid size.
  PFNO      x free, t pinned -- one branch per temporal frequency, so the
            branch count nt//2+1 is fixed at construction.
  DeepONet  neither -- the branch input is 3*nx wide and the trunk coordinate
            buffer is baked in.

The control that makes the numbers mean something is `interp`: take the model's
own 64x64 prediction and bicubically resample it to the target grid. If zero-shot
evaluation does not beat that, the operator is not resolving anything new -- it
is only being read out on a finer mesh.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import (Normalizer, build_model, load_dataset, load_summary,
                        predict, rel_l2_per_sample, split_indices)

p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True, help="a completed train_operators.py outdir")
p.add_argument("--base-data", required=True, help="the dataset that run trained on")
p.add_argument("--targets", nargs="+", required=True,
               help="nx,nt=path triples for the evaluation grids")
p.add_argument("--models", default="FNO,PFNO")
p.add_argument("--out", required=True)
a = p.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base = load_dataset(a.base_data)
summary = load_summary(a.run_dir)
args = summary["args"]
split_seed = args.get("split_seed") or args["seed"]
train_idx, val_idx = split_indices(base, split_seed)
normalizer = Normalizer(base["inputs"], base["outputs"], train_idx).to(DEVICE)
nx0, nt0 = base["inputs"].shape[2:]
print(f"trained at {nx0}x{nt0}; {len(train_idx)} train / {len(val_idx)} val")

results = []
for model_name in a.models.split(","):
    state_path = Path(a.run_dir) / f"{model_name}_state.pt"
    if not state_path.exists():
        print(f"skip {model_name}: no checkpoint"); continue
    state = torch.load(state_path, map_location=DEVICE)

    # the model's own prediction on the grid it was trained on, for the control
    m0 = build_model(model_name, args, nx0, nt0, DEVICE)
    m0.load_state_dict(state)
    base_pred = predict(m0, base["inputs"][val_idx], normalizer, DEVICE)

    for spec in a.targets:
        grid, path = spec.split("=")
        nx, nt = (int(v) for v in grid.split(","))
        if model_name == "PFNO" and nt != nt0:
            print(f"  {model_name:5s} {nx}x{nt}: not applicable "
                  f"(branch count is pinned to nt={nt0})")
            results.append({"model": model_name, "nx": nx, "nt": nt,
                            "status": "not_applicable"})
            continue
        fine = load_dataset(path)
        assert fine["inputs"].shape[2:] == (nx, nt), fine["inputs"].shape
        # Same seed and sampler, so sample i is the same material on both grids;
        # the split therefore transfers by index.
        model = build_model(model_name, args, nx, nt, DEVICE)
        model.load_state_dict(state)
        pred = predict(model, fine["inputs"][val_idx], normalizer, DEVICE)
        tgt = torch.from_numpy(fine["outputs"][val_idx])
        zero_shot = rel_l2_per_sample(pred, tgt)

        upsampled = F.interpolate(base_pred, size=(nx, nt), mode="bicubic",
                                  align_corners=True)
        control = rel_l2_per_sample(upsampled, tgt)

        row = {"model": model_name, "nx": nx, "nt": nt, "status": "ok",
               "zero_shot_rel_l2": float(np.mean(zero_shot)),
               "interp_control_rel_l2": float(np.mean(control)),
               "trained_grid_rel_l2": float(np.mean(
                   rel_l2_per_sample(base_pred,
                                     torch.from_numpy(base["outputs"][val_idx])))),
               "n_val": int(len(val_idx))}
        results.append(row)
        print(f"  {model_name:5s} {nx:>4d}x{nt:<4d} zero-shot {row['zero_shot_rel_l2']:7.2f}%   "
              f"interp control {row['interp_control_rel_l2']:7.2f}%   "
              f"(at {nx0}x{nt0}: {row['trained_grid_rel_l2']:.2f}%)", flush=True)

Path(a.out).write_text(json.dumps(
    {"run_dir": a.run_dir, "trained_nx": int(nx0), "trained_nt": int(nt0),
     "results": results}, indent=2))
print("wrote", a.out)
