"""Size the four OpenFWI operators to a common parameter budget.

The headline sweep runs each architecture at a sensible config for itself, and
those configs differ in size by two orders of magnitude -- FNO lands near 9.5 M
real scalars while GNO, whose parameters live in a small kernel MLP rather than
in a spectral tensor, tops out near 200 k however wide you make it. That is a
real property of the architecture, not an oversight, but it does mean the
headline table confounds architecture with capacity.

This finds, per architecture, the knob value landing closest to a target budget,
and prints the flags that produce it. Budgets are in *real* trainable scalars:
a complex spectral weight holds two, and counting it as one would quietly
favour FNO and PFNO in a comparison that is supposed to be fair.

    python wave/openfwi/size_openfwi.py --budgets 2e5 1.5e6
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openfwi_data import DATASET_CONFIG
from openfwi_models import (OpenFWIDeepONet, OpenFWIFNO, OpenFWIGNO, OpenFWIPFNO,
                            count_parameters, count_parameters_real)

# knob -> (flags it sets, search range)
KNOBS = {
    "FNO": ("--fno-width", (4, 96)),
    "PFNO": ("--pfno-width", (2, 64)),
    "DeepONet": ("--don-hidden", (16, 2048)),
    "GNO": ("--gno-width", (8, 192)),
}


def build(name, knob, cfg, nt, a):
    nz = nx = cfg["n_grid"]
    ns = cfg["ns"]
    if name == "FNO":
        return OpenFWIFNO(width=knob, modes_z=a.fno_modes_z, modes_x=a.fno_modes_x,
                          modes_t=a.fno_modes_t, enc_layers=a.fno_enc_layers,
                          dec_layers=a.fno_dec_layers, n_sources=ns, nz=nz, nx=nx,
                          nt=nt, t_latent=a.t_latent)
    if name == "PFNO":
        return OpenFWIPFNO(n_freqs=a.pfno_freqs, width=knob, modes=a.pfno_modes,
                           layers=a.pfno_layers, n_sources=ns, nz=nz, nx=nx, nt=nt)
    if name == "DeepONet":
        return OpenFWIDeepONet(nz=nz, nx=nx, nt=nt, n_sources=ns,
                               latent=max(8, knob // 2), hidden=knob,
                               fourier_features=a.don_fourier)
    if name == "GNO":
        # kernel_hidden tracks width: the kernel MLP is where a GNO's capacity
        # actually lives, so growing width alone would barely move the count.
        return OpenFWIGNO(width=knob, kernel_hidden=a.gno_kernel_ratio * knob,
                          radius=a.gno_radius, dec_radius=a.gno_dec_radius,
                          enc_layers=a.gno_enc_layers, dec_layers=a.gno_dec_layers,
                          n_sources=ns, nz=nz, nx=nx, nt=nt, t_latent=a.gno_t_latent)
    raise ValueError(name)


def search(name, target, cfg, nt, a):
    lo, hi = KNOBS[name][1]
    best = None
    for knob in range(lo, hi + 1):
        model = build(name, knob, cfg, nt, a)
        real = count_parameters_real(model)
        err = abs(real - target)
        if best is None or err < best["err"]:
            best = {"knob": knob, "err": err, "parameters_real": real,
                    "parameters": count_parameters(model)}
        if real > target and err > best["err"]:
            break                       # past the crossing, error only grows
    best["rel_error_vs_budget"] = best["parameters_real"] / target - 1.0
    best.pop("err")
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", type=float, nargs="+", default=[2.0e5, 1.5e6])
    p.add_argument("--dataset-key", default="flatvel-a")
    p.add_argument("--nt", type=int, default=1000)
    p.add_argument("--models", default="FNO,PFNO,DeepONet,GNO")
    p.add_argument("--t-latent", type=int, default=250)
    p.add_argument("--fno-modes-z", type=int, default=16)
    p.add_argument("--fno-modes-x", type=int, default=16)
    p.add_argument("--fno-modes-t", type=int, default=32)
    p.add_argument("--fno-enc-layers", type=int, default=3)
    p.add_argument("--fno-dec-layers", type=int, default=3)
    p.add_argument("--pfno-freqs", type=int, default=64)
    p.add_argument("--pfno-modes", type=int, default=8)
    p.add_argument("--pfno-layers", type=int, default=2)
    p.add_argument("--don-fourier", type=int, default=32)
    p.add_argument("--gno-kernel-ratio", type=int, default=4)
    p.add_argument("--gno-radius", type=int, default=3)
    p.add_argument("--gno-dec-radius", type=int, default=2)
    p.add_argument("--gno-enc-layers", type=int, default=3)
    p.add_argument("--gno-dec-layers", type=int, default=2)
    p.add_argument("--gno-t-latent", type=int, default=250)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    cfg = DATASET_CONFIG[a.dataset_key]
    names = [n.strip() for n in a.models.split(",") if n.strip()]
    table = {}
    for budget in a.budgets:
        row = {}
        print("\n=== budget %s real parameters ===" % format(int(budget), ","))
        for name in names:
            r = search(name, budget, cfg, a.nt, a)
            row[name] = r
            flag = KNOBS[name][0]
            extra = ""
            if name == "DeepONet":
                extra = " --don-latent %d" % max(8, r["knob"] // 2)
            elif name == "GNO":
                extra = " --gno-kernel-hidden %d" % (a.gno_kernel_ratio * r["knob"])
            print("  %-9s %s %-4d%-28s real=%10s  nominal=%10s  (%+.1f%%)"
                  % (name, flag, r["knob"], extra,
                     format(r["parameters_real"], ","),
                     format(r["parameters"], ","),
                     100.0 * r["rel_error_vs_budget"]))
        table[format(int(budget))] = row
    if a.out:
        Path(a.out).write_text(json.dumps(table, indent=2))
        print("\nwrote", a.out)


if __name__ == "__main__":
    main()
