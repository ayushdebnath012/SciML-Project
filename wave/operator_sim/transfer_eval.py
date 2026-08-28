"""Cross-arm transfer: train on one velocity-model family, test on another.

Every number published for this project is in-distribution -- each arm's
validation materials are drawn from the same source as its training materials.
This measures the question a practitioner actually asks: does an operator fitted
to one geology work on another?

Protocol. A model trained on arm A keeps A's input/output normalizer and is
evaluated on arm B's *validation* split, so the transfer number and B's own
in-distribution number are computed on identical samples.
"""
import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import (Normalizer, build_model, load_dataset, load_summary,
                        predict, rel_l2_per_sample, split_indices)

p = argparse.ArgumentParser()
p.add_argument("--arms", nargs="+", required=True,
               help="name=run_dir=dataset triples, e.g. marmousi=out_marmousi=data.npz")
p.add_argument("--models", default="FNO,DeepONet,PFNO")
p.add_argument("--out", required=True)
a = p.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arms = {}
for spec in a.arms:
    name, run_dir, data_path = spec.split("=")
    data = load_dataset(data_path)
    summary = load_summary(run_dir)
    args = summary["args"]
    split_seed = args.get("split_seed") or args["seed"]
    train_idx, val_idx = split_indices(data, split_seed)
    arms[name] = {
        "run_dir": Path(run_dir), "data": data, "args": args,
        "train_idx": train_idx, "val_idx": val_idx,
        "normalizer": Normalizer(data["inputs"], data["outputs"], train_idx).to(DEVICE),
        "in_domain": {r["model"]: r["validation rel L2 (%)"]
                      for r in summary["training_results"]},
    }
    print(f"{name}: {len(train_idx)} train / {len(val_idx)} val   from {run_dir}")

results = []
for model_name in a.models.split(","):
    for src, dst in itertools.product(arms, arms):
        source, target = arms[src], arms[dst]
        state_path = source["run_dir"] / f"{model_name}_state.pt"
        if not state_path.exists():
            print(f"  skip {model_name} {src}->{dst}: no {state_path.name}")
            continue
        nx, nt = target["data"]["inputs"].shape[2:]
        coords = None
        if model_name == "DeepONet":
            # The trunk grid is a normalized (x, t) mesh; it must be built with
            # the *source* normalizer, since that is what the trunk was fitted
            # against.
            xn = source["normalizer"]
            sample = torch.from_numpy(target["data"]["inputs"][:1]).to(DEVICE)
            coords = xn.norm_x(sample)[0, 3:5].permute(1, 2, 0).reshape(nx * nt, 2).clone()
        model = build_model(model_name, source["args"], nx, nt, DEVICE, coords)
        model.load_state_dict(torch.load(state_path, map_location=DEVICE))

        val = target["val_idx"]
        pred = predict(model, target["data"]["inputs"][val], source["normalizer"], DEVICE)
        tgt = torch.from_numpy(target["data"]["outputs"][val])
        per_sample = rel_l2_per_sample(pred, tgt)
        row = {
            "model": model_name, "source": src, "target": dst,
            "rel_l2_mean": float(np.mean(per_sample)),
            "rel_l2_median": float(np.median(per_sample)),
            "n_val": int(len(val)),
            "in_domain_target": target["in_domain"].get(model_name),
        }
        row["degradation"] = (row["rel_l2_mean"] / row["in_domain_target"]
                              if row["in_domain_target"] else None)
        results.append(row)
        tag = "  (in-domain)" if src == dst else ""
        print(f"  {model_name:9s} {src:12s} -> {dst:12s} "
              f"{row['rel_l2_mean']:8.2f}%{tag}", flush=True)

Path(a.out).write_text(json.dumps(results, indent=2))
print("wrote", a.out)
