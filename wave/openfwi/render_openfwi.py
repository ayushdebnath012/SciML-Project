"""Animate the predicted shot gathers against the reference. CPU + matplotlib.

The static panels in report_openfwi.py show *where* a model is wrong; these show
*when*. Each frame is one time sample: the 2D panels carry a cursor at the
current time and the wide panel underneath plots amplitude across the receiver
line at that instant, reference against every model. Watching the direct
arrival sweep out and the reflections come back is what separates a model that
has the kinematics right and the amplitudes wrong from one that has neither.

    python wave/openfwi/render_openfwi.py \
        --run results/openfwi/curvevel_a --outdir results/openfwi/simulations

Works unchanged on a SubsurfaceGen run -- the export has the same layout, only
larger.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

from report_openfwi import (DIVERGING, GRID, INK, INK_2, MUTED, SEQUENTIAL,
                            SERIES, SURFACE, style_axes)


def load(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "openfwi_summary.json").read_text())
    preds = np.load(run_dir / "openfwi_predictions.npz")
    return summary, preds


def rel_l2(pred, ref):
    return 100.0 * np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-12)


def pick_sample(preds, names, target, how):
    """Median error under the first available model, so the GIF is typical."""
    if how.isdigit():
        return int(how)
    ref = np.asarray(preds[names[0]])
    errs = [rel_l2(ref[i], target[i]) for i in range(len(target))]
    order = np.argsort(errs)
    if how == "best":
        return int(order[0])
    if how == "worst":
        return int(order[-1])
    return int(order[len(order) // 2])


def render(run_dir, outdir, shot, pick, fps, dpi, stride, tag):
    summary, preds = load(run_dir)
    cfg = summary["config"]
    names = [m["model"] for m in sorted(summary["results"],
                                        key=lambda r: r["rel_l2_pct"])
             if m["model"] in preds.files]
    target = preds["target"]
    velocity = preds["velocity"]
    idx = pick_sample(preds, names, target, pick)

    ref = target[idx, shot]                       # (nt, ng)
    nt, ng = ref.shape
    dt = float(cfg.get("dt", 1e-3))
    dx = float(cfg.get("dx", 10.0))
    nz = velocity.shape[-2]
    nx = velocity.shape[-1]

    # 98th percentile, not the max: the direct arrival is several times taller
    # than every reflection and scaling to it washes the rest out.
    vmax = float(np.percentile(np.abs(ref), 98))
    # The direct arrival is several times taller than every reflection, so
    # scaling the trace panel to the global maximum leaves ~90 % of the frames
    # a flat line near zero. Scale to the 99.5th percentile instead and let the
    # direct arrival clip for the handful of frames it occupies -- the point of
    # the animation is the reflections, which are what the models get wrong.
    trace_lim = 1.15 * float(np.percentile(np.abs(ref), 99.5))
    clipped = float(np.abs(ref).max()) > trace_lim
    extent_g = [0, ng * dx, nt * dt, 0]
    # The velocity map spans the same physical width as the receiver line, but
    # not necessarily at the same sampling -- on the field-scale cache it is
    # downsampled 2x laterally. Derive its cell size from the shared width so
    # the depth axis is right in both cases.
    width_m = ng * dx
    cell_m = width_m / nx
    extent_v = [0, width_m, nz * cell_m, 0]

    ncol = 2 + len(names)
    fig = plt.figure(figsize=(2.4 * ncol, 7.6), facecolor=SURFACE)
    gs = GridSpec(2, ncol, figure=fig, height_ratios=[2.0, 1.15],
                  hspace=0.34, wspace=0.34)

    ax_v = fig.add_subplot(gs[0, 0])
    im = ax_v.imshow(velocity[idx, 0], cmap=SEQUENTIAL, aspect="auto", extent=extent_v)
    vel = velocity[idx, 0]
    # The range goes in the title rather than a colorbar: a colorbar here is
    # squeezed between two panels and lands on the next one's axis label, and
    # the exact value at a pixel is not what this animation is for.
    ax_v.set_title("velocity model  %.0f-%.0f m/s" % (vel.min(), vel.max()),
                   color=INK, fontsize=9)
    ax_v.set_ylabel("depth (m)", color=INK_2, fontsize=8)
    ax_v.set_xlabel("offset (m)", color=INK_2, fontsize=8)
    ax_v.tick_params(colors=INK_2, labelsize=7)

    # Materialise every slice once. Indexing an NpzFile decompresses the whole
    # array each time, and update() touches one per model per frame -- on the
    # field-scale export that is 4 x 57 MB inflated 71 times over, which turned
    # a 2-minute render into more than ten.
    series = {n: np.asarray(preds[n][idx, shot]) for n in names}
    full = {n: np.asarray(preds[n][idx]) for n in names}
    panels = [("reference", ref)] + [(n, series[n]) for n in names]
    cursors = []
    for col, (name, field) in enumerate(panels, start=1):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(field, cmap=DIVERGING, aspect="auto", vmin=-vmax, vmax=vmax,
                  extent=extent_g)
        if name == "reference":
            title = "FD reference (shot %d)" % (shot + 1)
            ax.set_ylabel("time (s)", color=INK_2, fontsize=8)
        else:
            title = "%s   %.2f%%" % (name, rel_l2(full[name], target[idx]))
            ax.set_yticklabels([])
        ax.set_title(title, color=INK if name == "reference" else
                     SERIES.get(name, INK), fontsize=9)
        ax.set_xlabel("receiver (m)", color=INK_2, fontsize=8)
        ax.tick_params(colors=INK_2, labelsize=7)
        cursors.append(ax.axhline(0.0, color=INK, linewidth=1.1, alpha=0.85))

    ax_t = fig.add_subplot(gs[1, :])
    style_axes(ax_t, grid_axis="both")
    receivers = np.arange(ng) * dx
    line_ref, = ax_t.plot(receivers, ref[0], color=INK, linewidth=1.6,
                          label="FD reference", zorder=5)
    lines = {}
    for name in names:
        lines[name], = ax_t.plot(receivers, series[name][0],
                                 color=SERIES.get(name, MUTED), linewidth=1.3,
                                 alpha=0.9, label=name)
    ax_t.set_xlim(0, receivers[-1])
    ax_t.set_ylim(-trace_lim, trace_lim)
    ax_t.set_xlabel("receiver (m)", color=INK_2, fontsize=9)
    ax_t.set_ylabel("amplitude", color=INK_2, fontsize=9)
    if clipped:
        ax_t.annotate("direct arrival clips this scale", xy=(0.99, 0.06),
                      xycoords="axes fraction", ha="right", color=MUTED,
                      fontsize=8)
    ax_t.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=len(names) + 1,
                loc="upper right")
    clock = ax_t.text(0.01, 0.93, "", transform=ax_t.transAxes, color=INK,
                      fontsize=10, va="top", family="monospace")

    fig.suptitle("%s - sample %d, amplitude along the receiver line"
                 % (summary["dataset"], idx), color=INK, fontsize=12,
                 x=0.02, ha="left")

    frames = list(range(0, nt, stride))

    def update(frame):
        t_now = frame * dt
        for cur in cursors:
            cur.set_ydata([t_now, t_now])
        line_ref.set_ydata(ref[frame])
        for name in names:
            lines[name].set_ydata(series[name][frame])
        clock.set_text("t = %6.3f s" % t_now)
        return cursors + [line_ref, clock] + list(lines.values())

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / ("%s_shot%d_sample%d.gif" % (tag or summary["dataset"].lower(),
                                                shot + 1, idx))
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // fps,
                         blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi,
              savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)
    print("wrote %s  (%d frames, %.1f MB)"
          % (out, len(frames), out.stat().st_size / 1e6))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="a directory holding "
                                                "openfwi_summary.json + predictions")
    p.add_argument("--outdir", default="results/openfwi/simulations")
    p.add_argument("--shot", type=int, default=2)
    p.add_argument("--pick", default="median",
                   help="median | best | worst | an integer export index")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument("--stride", type=int, default=8,
                   help="time samples per frame; 1000 steps at stride 8 is a "
                        "125-frame, ~10 s animation")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    render(a.run, a.outdir, a.shot, a.pick, a.fps, a.dpi, a.stride, a.tag)


if __name__ == "__main__":
    main()
