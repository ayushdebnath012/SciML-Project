"""Shared loading/scoring helpers for the eval-only experiments.

`train_operators.py` saves a state dict per model and an
`operator_wave_summary_v2.json` that records the exact `args` it ran with, but
it does not save the input/output normalizer -- those statistics are a
deterministic function of the dataset and the training split, so they are
recomputed here from the same dataset the run was trained on.

That distinction matters for transfer: a model trained on arm A and evaluated
on arm B must keep **A's** normalizer. Renormalizing with B's statistics would
silently hand the model information about the target distribution and turn a
transfer measurement into a partial refit.
"""
import json
from pathlib import Path

import numpy as np
import torch

from operator_models import SimpleFNO2d, SimpleDeepONet, SimplePFNO


def load_dataset(path):
    with np.load(path) as raw:
        data = {
            "x": raw["x"].astype(np.float32),
            "t": raw["t"].astype(np.float32),
            "inputs": raw["inputs"].astype(np.float32),
            "outputs": raw["outputs"].astype(np.float32),
            "kinds": raw["kinds"].astype(str),
            "split": raw["split"].astype(str) if "split" in raw.files else None,
        }
    return data


def split_indices(data, split_seed):
    """Reproduce train_operators.py's split exactly."""
    n = data["inputs"].shape[0]
    if data["split"] is not None:
        val = np.flatnonzero(data["split"] == "val")
        train = np.flatnonzero(data["split"] != "val")
    else:
        g = torch.Generator().manual_seed(split_seed + 17)
        perm = torch.randperm(n, generator=g).numpy()
        n_val = max(1, min(n - 1, int(round(0.2 * n))))
        val, train = perm[:n_val], perm[n_val:]
    return np.asarray(train), np.asarray(val)


class Normalizer:
    """Per-channel input statistics and scalar output statistics, both fitted
    on the training split only."""

    def __init__(self, inputs, outputs, train_idx):
        X = torch.from_numpy(inputs[train_idx])
        Y = torch.from_numpy(outputs[train_idx])
        self.x_mean = X.mean(dim=(0, 2, 3), keepdim=True)
        self.x_std = X.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
        self.y_mean = Y.mean()
        self.y_std = Y.std().clamp_min(1e-6)

    def to(self, device):
        self.x_mean = self.x_mean.to(device); self.x_std = self.x_std.to(device)
        self.y_mean = self.y_mean.to(device); self.y_std = self.y_std.to(device)
        return self

    def norm_x(self, x): return (x - self.x_mean) / self.x_std
    def denorm_y(self, y): return y * self.y_std + self.y_mean


def load_summary(run_dir):
    return json.loads((Path(run_dir) / "operator_wave_summary_v2.json").read_text())


def build_model(name, args, nx, nt, device, coordinates=None):
    """Rebuild an architecture from a run's recorded `args`.

    `nx`/`nt` are the grid the model is being *evaluated* on, which is not
    necessarily the grid it trained on -- that is the whole point of the
    super-resolution arm. FNO carries over unchanged (its weights live on
    Fourier modes, not grid points). PFNO owns one branch per temporal
    frequency, so it transfers across `nx` but is pinned to its training `nt`.
    DeepONet's branch input is `3 * nx` wide and its trunk buffer is baked at
    construction, so it transfers to neither.
    """
    in_channels = 5
    if name == "FNO":
        model = SimpleFNO2d(width=args["fno_width"], modes_x=args["fno_modes"],
                            modes_t=args["fno_modes"], layers=args["fno_layers"],
                            in_channels=in_channels)
    elif name == "PFNO":
        model = SimplePFNO(nt=nt, width=args["pfno_width"], modes=args["pfno_modes"],
                           layers=args["pfno_layers"])
    elif name == "DeepONet":
        if coordinates is None:
            raise ValueError("DeepONet needs the (nx*nt, 2) coordinate grid")
        model = SimpleDeepONet(nx=nx, coordinates=coordinates,
                               latent=args["don_latent"], hidden=args["don_hidden"])
    else:
        raise ValueError(name)
    return model.to(device)


def rel_l2_per_sample(pred, target):
    """Relative L2 in percent, per sample. A zero prediction scores ~100."""
    p = pred.flatten(1); t = target.flatten(1)
    rel = (torch.linalg.vector_norm(p - t, dim=1)
           / torch.linalg.vector_norm(t, dim=1).clamp_min(1e-12))
    return (100.0 * rel).cpu().numpy()


@torch.no_grad()
def predict(model, inputs, normalizer, device, batch=16):
    model.eval()
    out = []
    for i in range(0, len(inputs), batch):
        xb = torch.from_numpy(inputs[i:i + batch]).to(device)
        out.append(normalizer.denorm_y(model(normalizer.norm_x(xb))).cpu())
    return torch.cat(out)
