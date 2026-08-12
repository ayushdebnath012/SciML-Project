"""Render the learned-wave comparison GIFs against the FD reference."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

MODELS = ("FNO", "DeepONet", "PFNO")
# Validated categorical palette: worst adjacent pair is dE 16.0 (protan),
# 22.8 (normal vision). The previous blue/teal pair sat at dE 14.0 normal,
# below the legibility floor -- hard to separate even with full colour vision.
COLORS = {"FNO": "#3573B9", "DeepONet": "#B03A5B", "PFNO": "#C98A00"}
INK = "#222222"

p = argparse.ArgumentParser()
p.add_argument("--pred", required=True)
p.add_argument("--summary", default="")
p.add_argument("--outdir", required=True)
p.add_argument("--positions", default="")
p.add_argument("--fps", type=int, default=12)
p.add_argument("--dpi", type=int, default=110)
a = p.parse_args()

d = np.load(a.pred, allow_pickle=False)
x, t = d["x"], d["t"]
ids, kinds = d["validation_ids"], d["validation_kinds"]
target, E, rho = d["target"], d["E"], d["rho"]
preds = {m: d[m] for m in MODELS}
nt = len(t)
# Forced arm only: the source position and peak frequency vary per sample,
# so show them alongside the velocity model.
x_src = d["x_src"] if "x_src" in d.files else None
f_peak = d["f_peak"] if "f_peak" in d.files else None

overall = {}
if a.summary and Path(a.summary).exists():
    s = json.loads(Path(a.summary).read_text())
    overall = {r["model"]: r["validation rel L2 (%)"] for r in s["training_results"]}


def rel_l2(pred, ref):
    return 100.0 * np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-12)


def pick_positions():
    """One representative sample per material kind, prefer typical (median) error."""
    if a.positions:
        return [int(v) for v in a.positions.split(",")]
    # Synthetic arms use the four sampler categories; other arms (e.g. the
    # Marmousi set, grouped by velocity contrast) bring their own labels.
    known = ("homogeneous", "two_layer", "layered", "smooth")
    present = [k for k in known if k in set(kinds)] or sorted(set(map(str, kinds)))
    chosen = []
    for kind in present:
        cand = [i for i in range(len(ids)) if kinds[i] == kind]
        if not cand:
            continue
        errs = [rel_l2(preds["FNO"][i], target[i]) for i in cand]
        chosen.append(cand[int(np.argsort(errs)[len(errs) // 2])])
    return chosen


def render(pos):
    ref = target[pos]
    fields = {m: preds[m][pos] for m in MODELS}
    c_profile = np.sqrt(E[pos] / np.maximum(rho[pos], 1e-12))

    # Scale colour to the propagating wave, not the t=0 pulse, which is ~1.8x
    # larger and otherwise washes out every later frame. The pulse saturates
    # for the first few frames, which is the intended trade.
    amp = float(np.percentile(np.abs(ref), 98.0))
    amp = max(amp, 1e-6)
    # Line plots keep the full range so the initial pulse is not clipped.
    line_amp = float(max(np.abs(ref).max(), max(np.abs(f).max() for f in fields.values())))
    line_amp = max(line_amp, 1e-6)

    fig, axes = plt.subplots(
        2, 4, figsize=(17.5, 7.6), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )

    # --- (0,0) velocity model -------------------------------------------------
    ax = axes[0, 0]
    ax.plot(x, c_profile, color="#444", lw=2.2)
    ax.fill_between(x, 0, c_profile, color="#999", alpha=0.25)
    ax.set_title(r"velocity model  $c(x)=\sqrt{E/\rho}$", fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("c(x)")
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(0, 1.25 * c_profile.max())
    ax.grid(alpha=0.25)
    if x_src is not None:
        ax.axvline(float(x_src[pos]), color="#c0392b", lw=1.6, ls="--")
        label = "source"
        if f_peak is not None:
            label += f"  f={float(f_peak[pos]):.1f}"
        ax.text(float(x_src[pos]), 1.16 * c_profile.max(), " " + label,
                color="#c0392b", fontsize=8.5)

    # --- (1,0) FD reference field --------------------------------------------
    ax = axes[1, 0]
    im0 = ax.imshow(ref.T, origin="lower", extent=[x[0], x[-1], t[0], t[-1]],
                    aspect="auto", cmap="RdBu_r", vmin=-amp, vmax=amp)
    cur0 = ax.axhline(t[0], color="k", lw=1.8)
    ax.set_title("FD reference  $u(x,t)$", fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("t")

    lines, cursors = [], [cur0]
    for col, name in enumerate(MODELS, start=1):
        f = fields[name]
        err = rel_l2(f, ref)

        axw = axes[0, col]
        # Reference drawn thick and pale underneath so overlap stays legible.
        axw.plot(x, ref[:, 0], color="#555", lw=5.0, alpha=0.35,
                 solid_capstyle="round", label="FD reference")
        ln, = axw.plot(x, f[:, 0], color=COLORS[name], lw=2.2, label=name)
        axw.axhline(0.0, color="0.5", lw=0.8)
        axw.set_xlim(x[0], x[-1]); axw.set_ylim(-1.1 * line_amp, 1.1 * line_amp)
        axw.set_xlabel("x")
        if col == 1:
            axw.set_ylabel("displacement u")
        axw.grid(alpha=0.25)
        axw.legend(loc="upper right", fontsize=8, framealpha=0.9)
        # Title in ink, not the series colour: identity is carried by the
        # coloured line in the legend, so the text stays legible on its own.
        axw.set_title(f"{name} — rel $L_2$ = {err:.2f}%", fontsize=11,
                      color=INK, fontweight="bold")

        axf = axes[1, col]
        axf.imshow(f.T, origin="lower", extent=[x[0], x[-1], t[0], t[-1]],
                   aspect="auto", cmap="RdBu_r", vmin=-amp, vmax=amp)
        cur = axf.axhline(t[0], color="k", lw=1.8)
        axf.set_title(f"{name} prediction", fontsize=11)
        axf.set_xlabel("x")
        axf.set_yticklabels([])

        lines.append((ln, axw, name, err))
        cursors.append(cur)

    fig.colorbar(im0, ax=axes[1, :], shrink=0.9, pad=0.01,
                 label="displacement u(x,t)")

    kind = str(kinds[pos]); sid = int(ids[pos])
    sub = "  |  ".join(f"{m} {overall[m]:.2f}%" for m in MODELS if m in overall)
    fig.suptitle(
        f"Learned wave propagation vs finite-difference reference — "
        f"held-out sample {sid} ({kind})"
        + (f"\nvalidation-set mean rel $L_2$:  {sub}" if sub else ""),
        fontsize=14,
    )

    ref_lines = [axes[0, c].lines[0] for c in range(1, 4)]

    def update(frame):
        tv = float(t[frame])
        artists = []
        for k, rl in enumerate(ref_lines):
            rl.set_ydata(ref[:, frame]); artists.append(rl)
        for (ln, axw, name, err) in lines:
            ln.set_ydata(fields[name][:, frame]); artists.append(ln)
        for cur in cursors:
            cur.set_ydata([tv, tv]); artists.append(cur)
        axes[0, 0].set_title(
            rf"velocity model  $c(x)=\sqrt{{E/\rho}}$        t = {tv:.3f}",
            fontsize=11)
        return artists

    anim = FuncAnimation(fig, update, frames=range(nt), interval=1000 // a.fps,
                         blit=False, repeat=True)
    out = Path(a.outdir) / f"FNO_DeepONet_PFNO_wave_simulation_sample_{sid}_{kind}.gif"
    anim.save(out, writer=PillowWriter(fps=a.fps), dpi=a.dpi)
    plt.close(fig)

    # Quantise to a 128-colour palette: these frames are mostly flat regions,
    # so this cuts file size several-fold with no visible change.
    from PIL import Image
    im = Image.open(out)
    frames = []
    try:
        while True:
            frames.append(im.convert("RGB").quantize(colors=128, method=Image.MEDIANCUT))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / a.fps), loop=0, optimize=True)
    print(f"  wrote {out.name}   ({', '.join(f'{n}={e:.2f}%' for _, _, n, e in lines)})")
    return out


Path(a.outdir).mkdir(parents=True, exist_ok=True)
positions = pick_positions()
print(f"rendering {len(positions)} samples: "
      f"{[(int(ids[p]), str(kinds[p])) for p in positions]}")
for pos in positions:
    render(pos)
print("done")
