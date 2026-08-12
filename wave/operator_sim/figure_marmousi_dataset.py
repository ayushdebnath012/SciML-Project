"""Regenerate docs/figures/marmousi_dataset.png.

The original figure was produced ad hoc and only the PNG survived, so its
caption drifted from the text (it claimed a 332 m minimum train/validation
separation; `make_trace_split` on the real model gives 324 m). This script
exists so the figure is reproducible and the caption is computed rather than
typed.

Every number in the figure is derived here, nothing is hardcoded:

  - the three profile panels are the samples the original showed, located by
    trace id (1237 / 105 / 28), with their contrast and depth read from the
    dataset's own provenance arrays;
  - heterogeneity is the median within-sample std of c(x), the same definition
    as docs/operator_simulations.md section 12.3;
  - the separation in the title comes from make_trace_split on the loaded
    velocity model, so it cannot disagree with the text again;
  - the shared colour scale follows the section 13 convention -- 98th percentile
    of |u|, not the global max, so the t=0 pulse does not eat the colormap.

Usage:
    python wave/operator_sim/figure_marmousi_dataset.py
    python wave/operator_sim/figure_marmousi_dataset.py --no-separation  # skip
        the trace-split recompute if operator_data/raw/ is not available
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Series palette from section 13, chosen for deuteranopia/protanopia separation.
BLUE = "#3573B9"
RED = "#B03A5B"
FILL = "#D6E4F0"

# The three Marmousi traces the original figure showed, one per contrast band.
TRACES = (1237, 105, 28)
BANDS = ("low contrast", "moderate contrast", "high contrast")


def heterogeneity(data):
    """Within-sample standard deviation of c(x), per sample."""
    E = data["inputs"][:, 0, :, 0]
    rho = data["inputs"][:, 1, :, 0]
    return np.sqrt(E / rho).std(axis=1)


def find_by_trace(data, trace_id):
    hits = np.flatnonzero(data["trace"] == trace_id)
    if not len(hits):
        raise SystemExit(f"trace {trace_id} not in dataset")
    return int(hits[0])


def realized_separation(raw_dir):
    """Minimum train-to-validation distance, recomputed from the real model."""
    from velocity_models import load_model, make_trace_split

    _, _, split_pos, _ = load_model("marmousi", str(raw_dir))
    train, val = make_trace_split(split_pos, 0.2, 4, 320.0)
    return float(np.abs(split_pos[train][:, None] - split_pos[val][None, :]).min())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--marmousi", default=str(
        ROOT / "operator_data" / "wave_operator_marmousi_n512_nx64_nt64_t1_seed42.npz"))
    p.add_argument("--synthetic", default=str(
        ROOT / "operator_data" / "wave_operator_fixedic_r8_n512_nx64_nt64_t1_seed42.npz"))
    p.add_argument("--raw-dir", default=str(ROOT / "operator_data" / "raw"))
    p.add_argument("--no-separation", action="store_true",
                   help="skip the trace-split recompute (needs operator_data/raw/)")
    p.add_argument("--out", default=str(ROOT / "docs" / "figures" / "marmousi_dataset.png"))
    p.add_argument("--dpi", type=int, default=140)
    a = p.parse_args()

    mar = np.load(a.marmousi, allow_pickle=True)
    syn = np.load(a.synthetic, allow_pickle=True)
    x, t = mar["x"], mar["t"]

    het_mar, het_syn = heterogeneity(mar), heterogeneity(syn)
    med_mar, med_syn = np.median(het_mar), np.median(het_syn)

    idx = [find_by_trace(mar, tid) for tid in TRACES]
    # Synthetic comparison panel: the sample sitting at the synthetic median
    # heterogeneity, so the contrast with Marmousi is representative not cherry-picked.
    syn_idx = int(np.argmin(np.abs(het_syn - med_syn)))

    fields = [mar["outputs"][i, 0] for i in idx] + [syn["outputs"][syn_idx, 0]]
    amplitude = float(np.percentile(np.abs(np.concatenate([f.ravel() for f in fields])), 98))

    n_train = int((np.array([str(v) for v in mar["split"]]) != "val").sum())
    n_val = len(mar["split"]) - n_train

    sep = None
    if not a.no_separation:
        try:
            sep = realized_separation(a.raw_dir)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  [warn] separation recompute skipped: {exc}")

    fig = plt.figure(figsize=(16.5, 7.0))
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.045],
                  hspace=0.42, wspace=0.28,
                  left=0.045, right=0.955, top=0.855, bottom=0.085)

    # ---- top row: three velocity profiles -------------------------------
    speed_max = 0.0
    for col, (i, band) in enumerate(zip(idx, BANDS)):
        E = mar["inputs"][i, 0, :, 0]
        rho = mar["inputs"][i, 1, :, 0]
        c = np.sqrt(E / rho)
        speed_max = max(speed_max, c.max())
        ax = fig.add_subplot(gs[0, col])
        ax.plot(x, c, color="0.1", lw=1.7, solid_joinstyle="round")
        ax.fill_between(x, 0, c, color=FILL, zorder=0)
        ax.set_title(f"{band}    $V_p^{{max}}/V_p^{{min}}$ = {mar['contrast'][i]:.2f}",
                     fontsize=11, fontweight="bold")
        ax.text(0.025, 0.955,
                f"trace {int(mar['trace'][i])} · {mar['depth_centre_m'][i]:.0f} m",
                transform=ax.transAxes, va="top", fontsize=9, color="0.45")
        ax.set_xlabel("x")
        ax.set_xlim(x.min(), x.max())
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel("wave speed  $c(x)$")

    for col in range(3):
        fig.axes[col].set_ylim(0, speed_max * 1.3)

    # ---- top right: heterogeneity histogram ------------------------------
    ax = fig.add_subplot(gs[0, 3])
    bins = np.linspace(0, max(het_mar.max(), het_syn.max()) * 1.02, 46)
    ax.hist(het_syn, bins=bins, color=RED, alpha=0.85, label="synthetic")
    ax.hist(het_mar, bins=bins, color=BLUE, alpha=0.85, label="Marmousi")
    ax.set_title(f"heterogeneity: {med_mar / med_syn:.1f}x the synthetic set",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("within-sample std of $c(x)$")
    ax.set_ylabel("samples")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    top = ax.get_ylim()[1]
    ax.text(med_syn, top * 0.62, f"{med_syn:.3f}", color=RED,
            ha="center", fontsize=10, fontweight="bold")
    ax.text(med_mar, top * 0.72, f"{med_mar:.3f}", color=BLUE,
            ha="center", fontsize=10, fontweight="bold")

    # ---- bottom row: FD reference fields ---------------------------------
    extent = [x.min(), x.max(), t.min(), t.max()]
    for col, field in enumerate(fields):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(field.T, origin="lower", extent=extent, aspect="auto",
                       cmap="RdBu_r", vmin=-amplitude, vmax=amplitude,
                       interpolation="nearest")
        synthetic_panel = col == 3
        ax.set_title("synthetic sample, for comparison" if synthetic_panel
                     else "FD reference  $u(x,t)$",
                     fontsize=11, color="0.45" if synthetic_panel else "0.1")
        ax.set_xlabel("x")
        if col == 0:
            ax.set_ylabel("t")

    cax = fig.add_subplot(gs[1, 4])
    fig.colorbar(im, cax=cax).set_label("displacement $u(x,t)$")

    separation = "" if sep is None else f" (min separation {sep:.0f} m)"
    fig.suptitle(
        f"Marmousi operator dataset — {len(mar['split'])} samples, "
        "real subsurface profiles, FD wave fields\n"
        f"{n_train} train / {n_val} validation drawn from disjoint trace "
        f"blocks{separation}",
        fontsize=13.5, y=0.975)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi)
    print(f"wrote {a.out}")
    print(f"  samples          {[int(mar['trace'][i]) for i in idx]} "
          f"(contrast {[round(float(mar['contrast'][i]), 2) for i in idx]})")
    print(f"  synthetic panel  idx {syn_idx} ({syn['kinds'][syn_idx]})")
    print(f"  heterogeneity    Marmousi {med_mar:.3f} / synthetic {med_syn:.3f} "
          f"= {med_mar / med_syn:.1f}x")
    print(f"  separation       {'not recomputed' if sep is None else f'{sep:.0f} m'}")
    print(f"  colour scale     +/-{amplitude:.3f} (98th pct of |u|)")


if __name__ == "__main__":
    main()
