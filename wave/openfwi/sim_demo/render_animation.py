"""Render the OpenFWI forward simulation as an animated GIF.

Follows the visual conventions of `wave/operator_sim/render_simulations.py`:
2 x N panel grid, RdBu_r on a symmetric scale, a black cursor sweeping the
gathers in step with the field above, shared colourbar, 128-colour quantise.

The difference is what is being compared. That script compares three trained
operators against the FD reference on a 1D problem; this one has no trained
model to show -- it animates the forward problem itself, so the wavefield and
the gather it writes are visible side by side.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forward2d import (vel_flat, vel_curve, ricker, _sponge,
                       NZ, NX, NT, DT, DX, FREQ, NS, NG, SZ, NBC)

INK = "#222222"
# Sequential single-hue ramp for velocity (magnitude), matching the static
# figures in render.py. A perceptually-uniform multi-hue map would still read
# as a rainbow against the diverging RdBu_r used for signed amplitude.
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
KINDS = {"flatvel": (vel_flat, 3), "curvevel": (vel_curve, 11)}

p = argparse.ArgumentParser()
p.add_argument("--kind", default="all", choices=["all", *KINDS])
p.add_argument("--outdir", default="simulations/openfwi")
p.add_argument("--every", type=int, default=15, help="timesteps between frames")
p.add_argument("--fps", type=int, default=12)
p.add_argument("--dpi", type=int, default=110)
a = p.parse_args()


def simulate(vel, every):
    """All five shots; wavefield snapshots kept for the centre shot only."""
    nz, nx = vel.shape
    pz, px = nz + 2 * NBC, nx + 2 * NBC
    c = np.pad(vel, NBC, mode="edge")
    coef = (c * DT / DX) ** 2
    damp = _sponge(pz, NBC)[:, None] * _sponge(px, NBC)[None, :]
    src = ricker(NT, DT, FREQ)
    sx = np.linspace(0, nx - 1, NS).round().astype(int)
    rx = np.arange(NG) + NBC
    rz = SZ + NBC

    gathers = np.zeros((NS, NT, NG))
    frames = []
    for si, s in enumerate(sx):
        up = np.zeros((pz, px)); uc = np.zeros((pz, px))
        for it in range(NT):
            lap = (uc[:-2, 1:-1] + uc[2:, 1:-1] +
                   uc[1:-1, :-2] + uc[1:-1, 2:] - 4.0 * uc[1:-1, 1:-1])
            un = np.zeros_like(uc)
            un[1:-1, 1:-1] = (2.0 * uc[1:-1, 1:-1] - up[1:-1, 1:-1]
                              + coef[1:-1, 1:-1] * lap)
            un[SZ + NBC, s + NBC] += coef[SZ + NBC, s + NBC] * src[it]
            un *= damp
            up, uc = uc * damp, un
            gathers[si, it, :] = uc[rz, rx]
            if si == NS // 2 and it % every == 0:
                frames.append(uc[NBC:NBC + nz, NBC:NBC + nx].copy())
    return gathers, np.array(frames), sx


def spreading_gain(n, p=1.0):
    """Compensate 2D geometric spreading so late arrivals stay on one colour scale."""
    t = np.arange(n) * DT
    return (t / max(t.max(), 1e-12)) ** p


def render(kind):
    fn, seed = KINDS[kind]
    vel = fn(seed)
    print(f"{kind}: simulating {NS} shots x {NT} steps ...", flush=True)
    gathers, frames, sx = simulate(vel, a.every)
    nf = len(frames)
    steps = np.arange(nf) * a.every

    # Fixed colour scale for both field and gathers, after gain. A single scale
    # keeps relative amplitude readable across the record; without the gain the
    # 2D spreading buries everything after the first 100 ms.
    g_t = spreading_gain(NT, 1.3)[:, None]
    gshow = gathers * g_t
    gamp = max(float(np.percentile(np.abs(gshow), 98.0)), 1e-12)
    f_gain = spreading_gain(NT, 1.0)[steps]
    fshow = frames * f_gain[:, None, None]
    famp = max(float(np.percentile(np.abs(fshow), 98.0)), 1e-12)
    # Line plots keep the full range: clipping the direct arrival would
    # misrepresent how much larger it is than everything after it.
    lamp = max(float(np.abs(gshow).max()), 1e-12)

    fig, axes = plt.subplots(
        2, 5, figsize=(19.0, 7.6), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )
    EXT = [0, NX * DX, NZ * DX, 0]
    src_x = float(sx[NS // 2] * DX)

    # --- (0,0) velocity model -------------------------------------------------
    ax = axes[0, 0]
    imv = ax.imshow(vel, cmap=SEQ, extent=EXT, aspect="equal", origin="upper")
    ax.plot([src_x], [SZ * DX], marker="*", ms=13, color="#eda100",
            mec="k", mew=0.8, ls="")
    ax.set_title(f"velocity model  $c(z,x)$        t = 0.000 s", fontsize=11)
    ax.set_xlabel("offset (m)"); ax.set_ylabel("depth (m)")
    fig.colorbar(imv, ax=ax, shrink=0.82, pad=0.02, label="Vp (m/s)")

    # --- (0,1) wavefield ------------------------------------------------------
    ax = axes[0, 1]
    imw = ax.imshow(fshow[0], cmap="RdBu_r", vmin=-famp, vmax=famp,
                    extent=EXT, aspect="equal", origin="upper")
    ax.contour(np.linspace(0, NX * DX, NX), np.linspace(0, NZ * DX, NZ), vel,
               levels=6, colors="k", linewidths=0.55, alpha=0.3)
    ax.plot([src_x], [SZ * DX], marker="*", ms=13, color="#eda100",
            mec="k", mew=0.8, ls="")
    ax.set_title("wavefield  $u(z,x,t)$", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel("offset (m)"); ax.set_yticklabels([])

    # --- (0,2) surface amplitude at t ----------------------------------------
    ax = axes[0, 2]
    off = np.arange(NG) * DX
    ln_surf, = ax.plot(off, gshow[NS // 2, 0, :], color="#3573B9", lw=2.2)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlim(off[0], off[-1]); ax.set_ylim(-1.08 * lamp, 1.08 * lamp)
    ax.set_title("surface amplitude at $t$", fontsize=11)
    ax.set_xlabel("receiver offset (m)"); ax.set_ylabel("amplitude (gained)")
    ax.grid(alpha=0.25)

    # --- (0,3) centre-receiver trace -----------------------------------------
    ax = axes[0, 3]
    tt = np.arange(NT) * DT
    ax.plot(tt, gshow[NS // 2, :, NG // 2], color="#555", lw=1.4)
    cur_tr = ax.axvline(0.0, color="k", lw=1.8)
    ax.set_xlim(0, tt[-1]); ax.set_ylim(-1.08 * lamp, 1.08 * lamp)
    ax.set_title("trace at centre receiver", fontsize=11)
    ax.set_xlabel("t (s)"); ax.grid(alpha=0.25)

    # --- (0,4) acquisition card ----------------------------------------------
    ax = axes[0, 4]; ax.axis("off")
    ax.text(0.0, 0.97,
            "OpenFWI acquisition\n\n"
            f"grid          {NZ} x {NX} @ dx = {DX:.0f} m\n"
            f"record        {NT} steps @ dt = {DT*1000:.0f} ms\n"
            f"source        {FREQ:.0f} Hz Ricker, {NS} shots\n"
            f"receivers     {NG} @ depth {SZ*DX:.0f} m\n"
            f"boundary      nbc = {NBC} sponge\n"
            f"CFL           {vel.max()*DT/DX:.2f}  (limit 0.707)\n\n"
            "Display carries a geometric-\nspreading gain; colour scale is\n"
            "fixed across the whole record.",
            va="top", ha="left", fontsize=9, family="monospace", color=INK,
            transform=ax.transAxes)

    # --- (1,*) the five gathers ----------------------------------------------
    cursors = []
    for j in range(NS):
        axg = axes[1, j]
        im = axg.imshow(gshow[j], cmap="RdBu_r", vmin=-gamp, vmax=gamp,
                        extent=[0, NG * DX, tt[-1], 0], aspect="auto")
        cursors.append(axg.axhline(0.0, color="k", lw=1.8))
        axg.plot([sx[j] * DX], [0.0], marker="*", ms=11, color="#eda100",
                 mec="k", mew=0.8, ls="", clip_on=False)
        axg.set_title(f"shot {j+1} gather", fontsize=11)
        axg.set_xlabel("receiver offset (m)")
        if j == 0: axg.set_ylabel("t (s)")
        else: axg.set_yticklabels([])
        if j == NS - 1:
            fig.colorbar(im, ax=axes[1, :], shrink=0.9, pad=0.01,
                         label="amplitude (gained)")

    fig.suptitle(
        f"Forward simulation on the OpenFWI acquisition — {kind} velocity model\n"
        f"velocity model → shot gathers, the map the neural operators are trained to approximate "
        f"(no network involved)", fontsize=14)

    def update(i):
        tv = float(steps[i] * DT)
        imw.set_data(fshow[i])
        axes[0, 0].set_title(
            f"velocity model  $c(z,x)$        t = {tv:.3f} s", fontsize=11)
        row = min(int(steps[i]), NT - 1)
        ln_surf.set_ydata(gshow[NS // 2, row, :])
        cur_tr.set_xdata([tv, tv])
        for cur in cursors:
            cur.set_ydata([tv, tv])
        return [imw, ln_surf, cur_tr, *cursors]

    anim = FuncAnimation(fig, update, frames=range(nf),
                         interval=1000 // a.fps, blit=False, repeat=True)
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    out = Path(a.outdir) / f"forward_simulation_{kind}.gif"
    anim.save(out, writer=PillowWriter(fps=a.fps), dpi=a.dpi)
    plt.close(fig)

    from PIL import Image
    im = Image.open(out); qframes = []
    try:
        while True:
            q = im.convert("RGB").quantize(colors=128, method=Image.MEDIANCUT)
            # convert() carries the source GIF's transparency key into the P-mode
            # frame, where it is invalid and breaks the multi-frame writer.
            q.info.pop("transparency", None)
            qframes.append(q)
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    qframes[0].save(out, save_all=True, append_images=qframes[1:],
                    duration=int(1000 / a.fps), loop=0, optimize=True)
    print(f"  wrote {out}   ({nf} frames, {out.stat().st_size/1e6:.1f} MB)")
    return out


for k in (list(KINDS) if a.kind == "all" else [a.kind]):
    render(k)
print("done")
