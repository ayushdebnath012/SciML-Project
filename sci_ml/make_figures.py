"""Figures for the paper, generated from the result files.

    python paper/neurips2026_workshop/make_figures.py

Two figures:

  fig_cost_accuracy  what each family costs to reach a given accuracy. Both
                     axes are logarithmic because the two families are four
                     orders of magnitude apart in time and two in error, which
                     is the finding.
  fig_causal         the causal weight of each time slab over training, for one
                     run. Shows the frontier stalling.

Series colours are the three-hue categorical set validated for colour-vision
deficiency (worst adjacent pair dE 18.8 under deuteranopia). Line style and
marker vary with the series too, so the figures survive greyscale printing --
and `fd2` and `fem_p1` land on top of each other, so without distinct markers
one would simply hide the other.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
MATERIALS = ["Homogeneous", "TwoLayer", "MultiLayer"]

SERIES = {                       # colour, linestyle, marker
    "fd2":    ("#3573B9", "-",  "o"),
    "fem_p1": ("#C98A00", "--", "s"),
    "cheb":   ("#B03A5B", "-.", "^"),
}
LABEL = {"fd2": "FD2", "fem_p1": "P1 FEM", "cheb": "spectral"}
INK, MUTED = "#1a1a1a", "#6b6b6b"
# Log-axis floor for the cost-accuracy plot. Points below it are the
# spectral solver agreeing with the reference to its own precision, not a
# measured error, and are drawn hollow.
FLOOR = 1e-4

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def load_pinn(root):
    out = []
    root = Path(root)
    if not root.exists():
        return out
    for path in sorted(root.rglob("l2_errors.json")):
        d = json.loads(path.read_text())
        d["_material"] = d.get("material_type") or path.parent.parent.name
        best = None
        for key in ("relative_l2_error_best_lbfgs_percent",
                    "relative_l2_error_best_lbfgs_only_percent",
                    "relative_l2_error_best_adam_percent"):
            if d.get(key) is not None:
                best = d[key] if best is None else min(best, d[key])
        d["_best"] = best
        out.append(d)
    return out


def fig_cost_accuracy(classical, pinn, pinn_seconds):
    by = defaultdict(list)
    for r in classical["convergence"]:
        by[(r["material"], r["method"])].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95), sharey=True)
    for ax, mat in zip(axes, MATERIALS):
        for method, (colour, ls, marker) in SERIES.items():
            rows = sorted(by[(mat, method)], key=lambda r: r["seconds"])
            if not rows:
                continue
            xs = [r["seconds"] for r in rows]
            raw = [r["rel_l2_percent"] for r in rows]
            # The spectral ladder bottoms out at the reference's own precision.
            # Clip so a log axis does not run to 1e-12 and squash everything,
            # but draw the clipped points hollow so the floor is not read as a
            # measured error.
            ys = [max(v, FLOOR) for v in raw]
            clipped = [v < FLOOR for v in raw]
            ax.plot(xs, ys, ls, color=colour, linewidth=1.0,
                    label=LABEL[method], zorder=3)
            solid = [(x, y) for x, y, c in zip(xs, ys, clipped) if not c]
            hollow = [(x, y) for x, y, c in zip(xs, ys, clipped) if c]
            if solid:
                ax.plot(*zip(*solid), linestyle="none", marker=marker,
                        markersize=2.6, color=colour, zorder=4)
            if hollow:
                ax.plot(*zip(*hollow), linestyle="none", marker=marker,
                        markersize=2.6, markerfacecolor="white",
                        markeredgecolor=colour, markeredgewidth=0.7, zorder=4)

        pts = [r for r in pinn if r["_material"] == mat and r["_best"] is not None]
        if pts:
            ax.scatter([pinn_seconds] * len(pts), [p["_best"] for p in pts],
                       s=13, facecolor="none", edgecolor=INK, linewidth=0.8,
                       marker="D", zorder=4, label="PINN runs")
        ax.axhline(100, color=MUTED, linewidth=0.6, linestyle=":", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(mat)
        ax.set_xlabel("wall-clock (s)")
        ax.grid(True, which="major", linewidth=0.35, color="#dddddd", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(r"relative $L^2$ (\%)".replace("\\%", "%"))
    axes[0].text(1.4e-3, 130, "zero solution", fontsize=6, color=MUTED, va="bottom")
    axes[0].annotate("at reference precision", xy=(0.13, 1.2e-4),
                     fontsize=5.5, color=MUTED, va="bottom", ha="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12),
               ncol=4, frameon=False)
    fig.savefig(OUT / "fig_cost_accuracy.pdf")
    plt.close(fig)
    print("wrote fig_cost_accuracy.pdf")


def _frontier(run_dir):
    """(steps, weights) for one run, or None if it has no causal frontier.

    RBA runs replace the schedule entirely and record a single slab, so they
    have nothing to plot.
    """
    path = Path(run_dir) / "causal_convergence.json"
    if not path.exists():
        return None
    snaps = json.loads(path.read_text()).get("chunk_snapshots") or []
    if not snaps or len(snaps[0][1]) < 2:
        return None
    return (np.array([s[0] for s in snaps]),
            np.array([s[1] for s in snaps]).T)          # (chunks, snapshots)


def fig_causal(panels):
    """One panel per run in `panels`, a list of (title, run_dir, error)."""
    drawn = [(t, _frontier(d), e) for t, d, e in panels]
    drawn = [(t, f, e) for t, f, e in drawn if f is not None]
    if not drawn:
        print("skip fig_causal: no run with a causal frontier")
        return

    # Single-hue sequential ramp -- never a rainbow for a magnitude.
    ramp = LinearSegmentedColormap.from_list(
        "seq", ["#f2f5fa", "#9dbadc", "#3573B9", "#173a63"])

    fig, axes = plt.subplots(1, len(drawn), figsize=(5.5, 1.9), sharey=True)
    axes = np.atleast_1d(axes)
    mesh = None
    for ax, (title, (steps, weights), err) in zip(axes, drawn):
        mesh = ax.pcolormesh(steps, np.arange(1, weights.shape[0] + 1), weights,
                             cmap=ramp, vmin=0.0, vmax=1.0, shading="nearest")
        ax.set_xlabel("Adam step")
        ax.set_title(f"{title} ({err:.2f}\\%)".replace("\\%", "%"))
        ax.set_yticks([1, 4, 8, 12, 16])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        never = int((weights.max(axis=1) == 0).sum())
        print(f"  {title}: {never} of {weights.shape[0]} slabs never "
              f"received any gradient")
    axes[0].set_ylabel("time slab $m$")
    cb = fig.colorbar(mesh, ax=axes.tolist(), pad=0.02, fraction=0.03)
    cb.set_label(r"causal weight $\omega_m$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)
    fig.savefig(OUT / "fig_causal.pdf")
    plt.close(fig)
    print("wrote fig_causal.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cpath = ROOT / "results/classical/classical_benchmark.json"
    if not cpath.exists():
        print(f"missing {cpath}")
        return
    classical = json.loads(cpath.read_text())
    pinn = load_pinn(ROOT / "results/pinn")

    # Median measured training time from the sweep log, in seconds. Kept as a
    # single scalar because every run uses the same step budget; the spread is
    # reported in the text rather than smeared across the figure.
    secs_path = ROOT / "results/pinn/run_seconds.json"
    pinn_seconds = 800.0
    if secs_path.exists():
        pinn_seconds = json.loads(secs_path.read_text())["median_seconds"]

    fig_cost_accuracy(classical, pinn, pinn_seconds)

    # Contrast a strong stalled run with the run that advances furthest. This
    # remains truthful whether or not any corrected run opens all slabs.
    entries = []
    for path in (ROOT / "results/pinn").rglob("l2_errors.json"):
        rec = json.loads(path.read_text())
        if not rec.get("use_ansatz") or rec.get("weighting") == "rba":
            continue
        frontier = _frontier(path.parent)
        if frontier is None:
            continue
        weights = frontier[1]
        opened = int((weights[:, -1] > 0).sum())
        vals = [rec[k] for k in ("relative_l2_error_best_lbfgs_percent",
                                 "relative_l2_error_best_adam_percent")
                if rec.get(k) is not None]
        if not vals:
            continue
        entries.append((min(vals), path.parent, opened, weights.shape[0]))

    panels = []
    if entries:
        furthest = max(entries, key=lambda e: (e[2], -e[0]))
        stalled = [e for e in entries
                   if e[2] < e[3] and e[1] != furthest[1]]
        if stalled:
            err, run_dir, _, _ = min(stalled, key=lambda e: e[0])
            panels.append(("frontier stalls", run_dir, err))
        err, run_dir, opened, total = furthest
        label = ("frontier completes" if opened == total
                 else "frontier advances furthest")
        panels.append((label, run_dir, err))
    if panels:
        print("causal figure from " + ", ".join(d.name for _, d, _ in panels))
        fig_causal(panels)

    fig_fields(ROOT / "results/fields.npz")




# ------------------------------------------------------------------ fields ---

FIELD_PANELS = [
    ("FD reference", "ref"),
    ("recipe", "FourierFeaturePINN_h3_w64_sigma3_ansatz_true_jax"),
    ("both schemes off", "FourierFeaturePINN_h3_w64_sigma3_ansatz_true_plain_jax"),
    ("recipe, worst run", "PINN_h3_w64_ansatz_true_jax"),
]


def fig_fields(path):
    """The solution itself: reference against three networks, per material.

    Error tables say a run scored 129%; they do not say what that looks like.
    This does. The diverging map is the right choice because u has a sign and
    zero is meaningful -- a sequential ramp would hide the polarity of the two
    counter-propagating waves.
    """
    if not Path(path).exists():
        print(f"skip fig_fields: no {path}")
        return
    d = np.load(path)
    fig, axes = plt.subplots(len(MATERIALS), len(FIELD_PANELS),
                             figsize=(5.5, 4.5))
    for row, mat in enumerate(MATERIALS):
        ref = d[f"ref|{mat}"]
        x, t = d[f"x|{mat}"], d[f"t|{mat}"]
        # Scale from the propagating wave, not the t=0 pulse, which is ~2x
        # taller and would spend half the colormap on the first few frames.
        amp = max(float(np.percentile(np.abs(ref), 99.0)), 1e-6)
        for col, (label, key) in enumerate(FIELD_PANELS):
            ax = axes[row, col]
            if key == "ref":
                field, title = ref, label
            else:
                name = f"pred|{mat}|{key}"
                if name not in d.files:
                    ax.axis("off"); continue
                field = d[name]
                title = f"{label}\n{float(d[f'rel|{mat}|{key}']):.2f}\%".replace(
                    "\%", "%")
            ax.imshow(field.T, origin="lower", aspect="auto", cmap="RdBu_r",
                      vmin=-amp, vmax=amp,
                      extent=[x[0], x[-1], t[0], t[-1]])
            if row == 0:
                ax.set_title(title, fontsize=7, pad=3)
            elif key != "ref":
                ax.set_title(f"{float(d[f'rel|{mat}|{key}']):.2f}%",
                             fontsize=7, pad=3)
            if col == 0:
                ax.set_ylabel(f"{mat}\n$t$", fontsize=7)
            else:
                ax.set_yticklabels([])
            # Three ticks only: adjacent panels are 2mm apart and a four-tick
            # axis collides with its neighbour's first label.
            ax.set_xticks([x[0], 0.0, x[-1]])
            if row == len(MATERIALS) - 1:
                ax.set_xlabel("$x$", fontsize=7)
                ax.set_xticklabels([f"{x[0]:.0f}", "0", f"{x[-1]:.0f}"])
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=6)
    fig.subplots_adjust(hspace=0.35, wspace=0.08)
    fig.savefig(OUT / "fig_fields.pdf")
    plt.close(fig)
    print("wrote fig_fields.pdf")


if __name__ == "__main__":
    main()
