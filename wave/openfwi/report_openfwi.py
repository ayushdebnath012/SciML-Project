"""Turn OpenFWI benchmark runs into a table and figures. CPU + matplotlib only.

Runs off the GPU box: `train_openfwi.py` writes JSON and a small prediction
export precisely so that plotting happens here.

    python wave/openfwi/report_openfwi.py \
        --runs results/openfwi/flatvel_a results/openfwi/curvevel_a \
        --outdir results/openfwi/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots 1-4 of the validated default palette, in their fixed order.
# The order is the colourblind-safety mechanism, not a preference, so it is not
# reshuffled to taste. Aqua and yellow fall below 3:1 on this surface, which
# obliges the relief rule -- every series carries a direct label and the run
# also emits results_table.md.
SERIES = {"FNO": "#2a78d6", "PFNO": "#eb6834",
          "DeepONet": "#1baf7a", "GNO": "#eda100"}
ORDER = list(SERIES)
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8983"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

# Signed seismic amplitude is a polarity quantity: diverging blue<->red with a
# neutral grey midpoint. A rainbow here would invent structure that is not in
# the wavefield.
DIVERGING = LinearSegmentedColormap.from_list(
    "bwr_dataviz", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec",
                    "#f3a09f", "#e34948", "#8d2020"])
# Velocity is unsigned magnitude: one hue, light -> dark.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])


def style_axes(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, color=GRID)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)


def load_run(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "openfwi_summary.json").read_text())
    hist_path = run_dir / "openfwi_histories.json"
    histories = json.loads(hist_path.read_text()) if hist_path.exists() else {}
    pred_path = run_dir / "openfwi_predictions.npz"
    preds = np.load(pred_path) if pred_path.exists() else None
    # Label by directory, not by dataset: a matched-capacity run and a tuned
    # run share a dataset name, and keying on that silently overwrote one of
    # their figures.
    return {"dir": run_dir, "summary": summary, "histories": histories,
            "preds": preds, "name": summary["dataset"], "label": run_dir.name}


def present(run):
    """Model names actually in this run, in the canonical order."""
    got = {r["model"] for r in run["summary"]["results"]}
    return [m for m in ORDER if m in got] + sorted(got - set(ORDER))


# ---------------------------------------------------------------------------
def figure_curves(runs, outdir):
    fig, axes = plt.subplots(1, len(runs), figsize=(6.2 * len(runs), 4.2),
                             squeeze=False, facecolor=SURFACE)
    for ax, run in zip(axes[0], runs):
        style_axes(ax)
        lo = []
        for name in present(run):
            hist = run["histories"].get(name)
            if not hist:
                continue
            ep = [h["epoch"] for h in hist]
            val = [h["val_rel_l2_pct"] for h in hist]
            lo.append(min(val))
            ax.plot(ep, val, color=SERIES.get(name, MUTED), linewidth=2,
                    solid_capstyle="round", label=name)
            ax.annotate(" %s  %.1f%%" % (name, val[-1]), xy=(ep[-1], val[-1]),
                        xytext=(4, 0), textcoords="offset points",
                        color=INK_2, fontsize=8, va="center")
        if not lo:
            # Merged result bundles can contain a complete summary and
            # prediction export without duplicating the source histories.
            # Keep the rest of the report usable and make the omission
            # explicit instead of calling min([]) below.
            ax.text(0.5, 0.5, "training history\nnot included in this bundle",
                    transform=ax.transAxes, ha="center", va="center",
                    color=MUTED, fontsize=10)
            ax.set_title(run["label"], color=INK, fontsize=11, loc="left", pad=10)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue
        floor = run["summary"]["oracles"]["pfno_band_limit"]
        ax.axhline(100, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate("predicting zero", xy=(0.02, 100), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points", color=MUTED, fontsize=8)
        # Draw the band floor only when it is near the curves. On OpenFWI it is
        # 0.01 %, three decades below anything measured, and plotting it would
        # stretch the axis until every curve collapsed into the top decade.
        best = min(lo)
        if floor["rel_l2_pct"] > best / 5.0:
            ax.axhline(floor["rel_l2_pct"], color=MUTED, linewidth=1,
                       linestyle=(0, (1, 2)))
            ax.annotate("PFNO band floor %.2f%%" % floor["rel_l2_pct"],
                        xy=(0.02, floor["rel_l2_pct"]),
                        xycoords=("axes fraction", "data"), xytext=(0, 4),
                        textcoords="offset points", color=MUTED, fontsize=8)
        else:
            ax.annotate("PFNO band floor %.3f %% (below axis)" % floor["rel_l2_pct"],
                        xy=(0.99, 0.02), xycoords="axes fraction", ha="right",
                        color=MUTED, fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(best / 2.0, 2000)
        ax.set_xlabel("epoch", color=INK_2, fontsize=9)
        ax.set_ylabel("validation relative L2 (%)", color=INK_2, fontsize=9)
        ax.set_title(run["label"], color=INK, fontsize=11, loc="left", pad=10)
        ax.set_xlim(0, max(ep) * 1.30)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left",
                  ncol=2)
    fig.suptitle("Forward operator learning on OpenFWI: velocity -> shot gathers",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = Path(outdir) / "training_curves.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def figure_gathers(run, outdir, shot=2, pick="median"):
    """Velocity, target gather and every model's prediction + residual.

    The sample is chosen by the median FNO error so the panel is representative
    rather than cherry-picked, matching render_simulations.py's convention.
    """
    preds = run["preds"]
    if preds is None:
        return None
    names = [n for n in present(run) if n in preds.files]
    target = preds["target"]
    velocity = preds["velocity"]
    ref = next((n for n in names if n in preds.files), None)
    if pick == "median" and ref:
        err = [np.linalg.norm(preds[ref][i] - target[i]) / np.linalg.norm(target[i])
               for i in range(len(target))]
        idx = int(np.argsort(err)[len(err) // 2])
    else:
        idx = 0

    # 98th percentile, not the max: the direct arrival is several times taller
    # than the reflections, and scaling to it washes everything else out.
    vmax = float(np.percentile(np.abs(target[idx, shot]), 98))
    cfg = run["summary"]["config"]
    extent_g = [0, cfg["ng"] * cfg["dx"], cfg["nt"] * cfg["dt"], 0]
    extent_v = [0, cfg["n_grid"] * cfg["dx"], cfg["n_grid"] * cfg["dx"], 0]

    ncol = 2 + len(names)
    fig, axes = plt.subplots(2, ncol, figsize=(2.35 * ncol, 7.4), facecolor=SURFACE)

    im_v = axes[0, 0].imshow(velocity[idx, 0], cmap=SEQUENTIAL, aspect="auto",
                             extent=extent_v)
    axes[0, 0].set_title("velocity model", color=INK, fontsize=9)
    axes[0, 0].set_ylabel("depth (m)", color=INK_2, fontsize=8)
    axes[0, 0].set_xlabel("offset (m)", color=INK_2, fontsize=8)
    fig.colorbar(im_v, ax=axes[0, 0], fraction=0.046).set_label("m/s", size=7)

    axes[0, 1].imshow(target[idx, shot], cmap=DIVERGING, aspect="auto",
                      vmin=-vmax, vmax=vmax, extent=extent_g)
    axes[0, 1].set_title("FD reference (shot %d)" % (shot + 1), color=INK, fontsize=9)
    axes[0, 1].set_ylabel("time (s)", color=INK_2, fontsize=8)
    axes[0, 1].set_xlabel("receiver (m)", color=INK_2, fontsize=8)

    scores = {r["model"]: r for r in run["summary"]["results"]}
    for j, name in enumerate(names):
        col = 2 + j
        p = preds[name][idx, shot]
        t = target[idx, shot]
        axes[0, col].imshow(p, cmap=DIVERGING, aspect="auto", vmin=-vmax, vmax=vmax,
                            extent=extent_g)
        rel = 100 * np.linalg.norm(preds[name][idx] - target[idx]) / np.linalg.norm(target[idx])
        axes[0, col].set_title("%s\n%.2f%% on this sample" % (name, rel),
                               color=INK, fontsize=9)
        im_r = axes[1, col].imshow(p - t, cmap=DIVERGING, aspect="auto",
                                   vmin=-vmax, vmax=vmax, extent=extent_g)
        axes[1, col].set_title("residual", color=INK_2, fontsize=8)
        for r in (0, 1):
            axes[r, col].set_xlabel("receiver (m)", color=INK_2, fontsize=8)
            axes[r, col].set_yticklabels([])
    fig.colorbar(im_r, ax=axes[1, ncol - 1], fraction=0.046).set_label(
        "amplitude", size=7)

    # A single trace tells you about phase and arrival time, which a 2D panel
    # at 98th-percentile clipping hides. Offset it from the shot: a trace
    # directly under the source is all direct arrival, several times taller
    # than every reflection, and the reflections are the interesting part.
    ng = target.shape[-1]
    n_shots = target.shape[1]
    src_rec = int(round(shot * (ng - 1) / max(1, n_shots - 1)))
    rec = min(ng - 1, src_rec + ng // 3) if src_rec < ng // 2 else max(0, src_rec - ng // 3)
    tax = np.arange(cfg["nt"]) * cfg["dt"]
    ax = axes[1, 0]
    style_axes(ax, grid_axis="both")
    ax.plot(tax, target[idx, shot, :, rec], color=INK, linewidth=1.4, label="reference")
    for name in names:
        ax.plot(tax, preds[name][idx, shot, :, rec], color=SERIES.get(name, MUTED),
                linewidth=1.2, alpha=0.9, label=name)
    ax.set_xlabel("time (s)", color=INK_2, fontsize=8)
    ax.set_ylabel("amplitude", color=INK_2, fontsize=8)
    ax.set_title("trace at receiver %d (shot at %d)" % (rec, src_rec),
                 color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, labelcolor=INK_2, ncol=2,
              loc="upper right", borderaxespad=0.2)
    axes[1, 1].axis("off")
    for a in axes.ravel():
        if a is not ax:
            a.tick_params(colors=INK_2, labelsize=7)

    fig.suptitle("%s - validation sample %d (median error)" % (run["label"], idx),
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = Path(outdir) / ("gathers_%s.png" % run["label"].lower())
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def figure_error_spread(runs, outdir):
    """Mean + interquartile range of per-sample error, per model."""
    fig, axes = plt.subplots(1, len(runs), figsize=(5.6 * len(runs), 3.4),
                             squeeze=False, facecolor=SURFACE)
    for ax, run in zip(axes[0], runs):
        style_axes(ax, grid_axis="x")
        names = present(run)
        scores = {r["model"]: r for r in run["summary"]["results"]}
        ys = np.arange(len(names))[::-1]
        for y, name in zip(ys, names):
            e = np.asarray(scores[name]["per_sample_rel_l2"])
            q1, q3 = np.percentile(e, [25, 75])
            color = SERIES.get(name, MUTED)
            ax.plot([q1, q3], [y, y], color=color, linewidth=2.5,
                    solid_capstyle="round", zorder=2)
            ax.plot([e.mean()], [y], marker="o", markersize=8, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            ax.annotate("  %.2f%%" % e.mean(), xy=(q3, y), xytext=(6, 0),
                        textcoords="offset points", color=INK_2, fontsize=8,
                        va="center")
        ax.set_yticks(ys)
        ax.set_yticklabels(names, color=INK, fontsize=9)
        ax.set_xscale("log")
        # A log axis defaults to labelling every minor decade step, which at
        # this width overprints itself into an unreadable smear.
        lo_all = min(np.percentile(np.asarray(scores[n]["per_sample_rel_l2"]), 25)
                     for n in names)
        hi_all = max(np.percentile(np.asarray(scores[n]["per_sample_rel_l2"]), 75)
                     for n in names)
        ticks = [t for t in (0.1, 0.3, 1, 3, 10, 30, 100, 300)
                 if lo_all / 1.6 <= t <= hi_all * 1.6]
        if len(ticks) >= 2:
            ax.set_xticks(ticks)
            ax.set_xticklabels([("%g" % t) for t in ticks])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("per-sample relative L2 (%), dot = mean, bar = IQR",
                      color=INK_2, fontsize=8.5)
        ax.set_title(run["label"], color=INK, fontsize=11, loc="left", pad=8)
    fig.tight_layout()
    path = Path(outdir) / "error_spread.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def late_time_error(run, early_fraction=0.2):
    """Relative L2 restricted to the late part of the record, per model.

    The direct arrival dominates the energy budget -- 83 % of a SubsurfaceGen
    record and a large share of an OpenFWI one -- so a model can post a
    respectable overall score while emitting nothing at all where the
    reflections are. Since a zero prediction scores ~100 % by construction, a
    late-time column at or above 100 % means exactly that: the model learned
    the direct arrival and no propagation. Without this column the headline
    number silently rewards fitting one loud event.

    The cut is the first `early_fraction` of the record, which lands just after
    the direct arrival crosses the array on both benchmarks.
    """
    preds = run["preds"]
    if preds is None:
        return {}
    target = preds["target"]
    cut = max(1, int(early_fraction * target.shape[2]))
    out = {}
    for name in present(run):
        if name not in preds.files:
            continue
        p_late = preds[name][:, :, cut:].reshape(len(target), -1)
        t_late = target[:, :, cut:].reshape(len(target), -1)
        num = np.linalg.norm(p_late - t_late, axis=1)
        den = np.maximum(np.linalg.norm(t_late, axis=1), 1e-12)
        out[name] = float(np.mean(100.0 * num / den))
    return out


def results_table(runs):
    lines = ["# Neural-operator forward benchmark", "",
             "Task: velocity model -> shot gathers, per run. Errors on physical "
             "amplitudes; a zero prediction scores ~100 %.", ""]
    for run in runs:
        s = run["summary"]
        g, sp, o = s["grid"], s["split"], s["oracles"]
        has_ood = any("ood" in r for r in s["results"])
        late = late_time_error(run)
        lines += ["## %s" % run["label"], "",
                  "%d train / %d val%s, velocity (%d x %d) -> "
                  "%d shots x %d steps x %d receivers, %d epochs on %s. "
                  "Normalization: %s."
                  % (sp["train"], sp["val"],
                     ", %d out-of-distribution" % s["results"][0]["ood"]["n"]
                     if has_ood and "n" in s["results"][0].get("ood", {}) else "",
                     g["nz"], g["nx"], g["n_sources"], g["nt"],
                     g.get("ng") or s["config"]["ng"], s["args"]["epochs"], s["gpu"],
                     s.get("normalization", {}).get("mode", "minmax")), "",
                  "| model | real params | rel L2 % |"
                  + (" out-of-dist % |" if has_ood else "")
                  + (" late-time % |" if late else "")
                  + " median % | MAE | RMSE | s/epoch |",
                  "|---|---:|---:|" + ("---:|" if has_ood else "")
                  + ("---:|" if late else "")
                  + "---:|---:|---:|---:|"]
        rows = sorted(s["results"], key=lambda r: r["rel_l2_pct"])
        for r in rows:
            ood = (" %.2f |" % r["ood"]["rel_l2_pct"]) if has_ood and "ood" in r else (
                " - |" if has_ood else "")
            lt = (" %.1f |" % late[r["model"]]) if r["model"] in late else (
                " - |" if late else "")
            lines.append("| %s | %s | **%.2f** |%s%s %.2f | %.4f | %.4f | %.1f |"
                         % (r["model"], format(r["parameters_real"], ","),
                            r["rel_l2_pct"], ood, lt, r["rel_l2_pct_median"],
                            r["mae"], r["rmse"], r["seconds_per_epoch"]))
        if late:
            lines += ["",
                      "**late-time %** is relative L2 after the first 20 % of "
                      "the record, where the reflections live. A zero "
                      "prediction scores ~100 %, so a value at or above 100 "
                      "means the model reproduced the direct arrival and "
                      "nothing else."]
        lines += ["",
                  "Representation scales: PFNO band limit (%d bins) **%.3f %%** "
                  "- a hard floor, bins above the band are zeroed. "
                  "Time latent (%d pts) %.3f %% for a single-channel resample - "
                  "a reference, not a bound: the latent is multi-channel."
                  % (o["pfno_band_limit"]["n_freqs"], o["pfno_band_limit"]["rel_l2_pct"],
                     o["fno_time_resample"]["t_latent"],
                     o["fno_time_resample"]["rel_l2_pct"]), ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--outdir", default="results/openfwi/figures")
    p.add_argument("--shot", type=int, default=2)
    a = p.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = [load_run(r) for r in a.runs]
    table = results_table(runs)
    (outdir / "results_table.md").write_text(table, encoding="utf-8")
    print(table)

    made = [figure_curves(runs, outdir), figure_error_spread(runs, outdir)]
    for run in runs:
        made.append(figure_gathers(run, outdir, shot=a.shot))
    for m in made:
        if m:
            print("wrote", m)


if __name__ == "__main__":
    main()
