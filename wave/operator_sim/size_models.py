"""Size FNO / DeepONet / PFNO to a common parameter budget.

The headline comparison in this project runs the three architectures at the
sizes they were tuned at -- FNO 7.4 M, PFNO 1.2 M, DeepONet 0.89 M by the
`count_parameters` convention -- so FNO's win is confounded with capacity.
This picks, for each architecture, the single width/hidden that lands closest
to a target budget, and prints the exact counts and the CLI flags that produce
them.

Budgets are in *real* trainable scalars: a complex spectral weight holds two.
Counting it as one (which `count_parameters` does) understates every spectral
layer by 2x and would make a "matched" comparison quietly favour FNO/PFNO.
"""
import argparse, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from operator_models import SimpleFNO2d, SimpleDeepONet, SimplePFNO


def count_real(model):
    return sum(p.numel() * (2 if p.is_complex() else 1)
               for p in model.parameters() if p.requires_grad)


def count_nominal(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build(name, knob, nx, nt, modes, fno_layers, pfno_layers, in_channels):
    """`knob` is width for FNO/PFNO and hidden for DeepONet (latent = hidden/2)."""
    if name == "FNO":
        return SimpleFNO2d(width=knob, modes_x=modes, modes_t=modes,
                           layers=fno_layers, in_channels=in_channels)
    if name == "PFNO":
        return SimplePFNO(nt=nt, width=knob, modes=modes, layers=pfno_layers)
    coords = torch.zeros(nx * nt, 2)
    return SimpleDeepONet(nx=nx, coordinates=coords, latent=max(8, knob // 2),
                          hidden=knob)


def search(name, target, lo, hi, **kw):
    """Smallest-error knob over [lo, hi]; the count is monotone in the knob."""
    best = None
    for knob in range(lo, hi + 1):
        model = build(name, knob, **kw)
        real = count_real(model)
        err = abs(real - target)
        if best is None or err < best[1]:
            best = (knob, err, real, count_nominal(model))
        if real > target and best[1] < err:
            break                      # past the crossing, error only grows
    knob, _, real, nominal = best
    return {"knob": knob, "parameters_real": real, "parameters": nominal,
            "rel_error_vs_budget": real / target - 1.0}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", type=float, nargs="+",
                   default=[0.9e6, 2.4e6, 14.8e6])
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--nt", type=int, default=64)
    p.add_argument("--modes", type=int, default=20)
    p.add_argument("--fno-layers", type=int, default=4)
    p.add_argument("--pfno-layers", type=int, default=3)
    p.add_argument("--in-channels", type=int, default=5)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    kw = dict(nx=a.nx, nt=a.nt, modes=a.modes, fno_layers=a.fno_layers,
              pfno_layers=a.pfno_layers, in_channels=a.in_channels)
    table = {}
    for budget in a.budgets:
        row = {}
        for name, (lo, hi) in {"FNO": (4, 96), "PFNO": (4, 96),
                               "DeepONet": (32, 4096)}.items():
            row[name] = search(name, budget, lo, hi, **kw)
        table[f"{budget:.3g}"] = row
        print(f"\n=== budget {budget:,.0f} real parameters ===")
        for name, r in row.items():
            flag = {"FNO": "--fno-width", "PFNO": "--pfno-width",
                    "DeepONet": "--don-hidden"}[name]
            extra = f" --don-latent {max(8, r['knob'] // 2)}" if name == "DeepONet" else ""
            print(f"  {name:9s} {flag} {r['knob']:<5d}{extra:<20s} "
                  f"real={r['parameters_real']:>10,d}  nominal={r['parameters']:>10,d}  "
                  f"({r['rel_error_vs_budget']:+.1%})")
    if a.out:
        Path(a.out).write_text(json.dumps(table, indent=2))
        print("\nwrote", a.out)
