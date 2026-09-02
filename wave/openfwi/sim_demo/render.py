import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                    "results", "openfwi", "simulations"))
os.makedirs(OUT, exist_ok=True)
S2 = "#eb6834"
SURF,TP,TS,TS3,GRID = "#fcfcfb","#0b0b0b","#52514e","#868580","#e3e2de"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.facecolor":SURF,
 "figure.facecolor":SURF,"axes.edgecolor":GRID,"axes.labelcolor":TS,
 "xtick.color":TS,"ytick.color":TS})

SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb","#86b6ef","#3987e5","#256abf","#184f95","#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95","#2a78d6","#86b6ef","#f0efec","#f0a6a5","#e34948","#8f2020"])

flat = np.load(os.path.join(HERE, "sim_flat.npz"))
curve = np.load(os.path.join(HERE, "sim_curve.npz"))
SNAPS=(60,120,200,300,450,650); EXT=[0,700,700,0]; DT=0.001

def gain(g, p=1.3):
    """Standard seismic display t-gain: reflections are weaker than the direct arrival."""
    t = (np.arange(g.shape[0]) * DT)
    w = (t / t.max()) ** p
    return g * w[:, None]

def vpanel(ax, v, title):
    im = ax.imshow(v, cmap=SEQ, extent=EXT, aspect="equal", origin="upper")
    ax.set_title(title, fontsize=10.5, color=TP, fontweight="bold", pad=7)
    ax.set_xlabel("offset (m)", fontsize=9); ax.set_ylabel("depth (m)", fontsize=9)
    return im

def gpanel(ax, g, title, clip=98.5, do_gain=True):
    gg = gain(g) if do_gain else g
    m = max(np.percentile(np.abs(gg), clip), 1e-14)
    ax.imshow(gg, cmap=DIV, vmin=-m, vmax=m, aspect="auto",
              extent=[0,700,1.0,0], interpolation="nearest")
    ax.set_title(title, fontsize=10.5, color=TP, fontweight="bold", pad=7)
    ax.set_xlabel("receiver offset (m)", fontsize=9); ax.set_ylabel("time (s)", fontsize=9)

# ---------- V1: the task ----------
fig = plt.figure(figsize=(11.4,4.7))
axL = fig.add_axes([0.055,0.175,0.285,0.645]); vpanel(axL, flat["vel"], "INPUT   velocity model  (70 x 70)")
cax = fig.add_axes([0.352,0.175,0.012,0.645])
cb = fig.colorbar(axL.images[0], cax=cax); cb.set_label("Vp (m/s)", fontsize=8.5, color=TS)
cb.ax.tick_params(labelsize=8); cb.outline.set_edgecolor(GRID)
fig.text(0.475,0.545,"forward\nsimulation", ha="center", va="center", fontsize=10.5,
         color=TS, fontweight="bold")
fig.patches.append(plt.Arrow(0.425,0.475,0.10,0, width=0.045, transform=fig.transFigure,
                             color=S2, figure=fig))
axR = fig.add_axes([0.585,0.175,0.345,0.645])
gpanel(axR, flat["gathers"][2], "OUTPUT   shot gather  (1000 x 70), shot 3 of 5")
fig.text(0.055,0.925,"The map the neural operators are asked to learn",
         fontsize=13.5, color=TP, fontweight="bold")
fig.text(0.055,0.035,"Simulated with a 2D acoustic finite-difference solver on OpenFWI's exact acquisition: dx = 10 m, 15 Hz Ricker,\n"
         "source and receivers at 100 m depth, 1000 steps at dt = 1 ms. Display carries a standard t-gain.",
         fontsize=8.6, color=TS3, linespacing=1.45)
fig.savefig(os.path.join(OUT, "task_velocity_to_gather.png"), dpi=200); plt.close(fig)

# ---------- V2: wavefield propagation ----------
fig, axes = plt.subplots(2,3, figsize=(11.4,6.7))
v = curve["vel"]
for ax, it in zip(axes.ravel(), SNAPS):
    w = curve[f"snap_{it}"]
    m = max(np.percentile(np.abs(w), 99.7), 1e-12)
    ax.imshow(w, cmap=DIV, vmin=-m, vmax=m, extent=EXT, aspect="equal", origin="upper")
    ax.contour(np.linspace(0,700,70), np.linspace(0,700,70), v, levels=6,
               colors="#0b0b0b", linewidths=0.55, alpha=0.34)
    ax.set_title(f"t = {it} ms", fontsize=10.5, color=TP, fontweight="bold", pad=5)
    ax.tick_params(labelsize=7.5)
    ax.plot([350],[100], marker="*", ms=11, color="#eda100", mec="#0b0b0b", mew=0.7)
for ax in axes[:,0]: ax.set_ylabel("depth (m)", fontsize=8.5)
for ax in axes[1,:]: ax.set_xlabel("offset (m)", fontsize=8.5)
fig.text(0.045,0.955,"Why the task is hard: one shot propagating through the curved model",
         fontsize=13.5, color=TP, fontweight="bold")
fig.text(0.045,0.022,"Thin contours are the velocity interfaces; the star is the source. Each panel is scaled to its own amplitude, so late and\n"
         "weak reflections stay visible. The operator never sees this field - only the trace it leaves on the surface.",
         fontsize=8.6, color=TS3, linespacing=1.45)
fig.subplots_adjust(left=0.065,right=0.98,top=0.90,bottom=0.115,wspace=0.28,hspace=0.30)
fig.savefig(os.path.join(OUT, "wavefield_snapshots.png"), dpi=200); plt.close(fig)

# ---------- V3: five shots ----------
fig, axes = plt.subplots(1,5, figsize=(12.6,4.0), sharey=True)
for i, ax in enumerate(axes):
    gpanel(ax, curve["gathers"][i], f"shot {i+1}")
    if i: ax.set_ylabel("")
    ax.set_xlabel("offset (m)", fontsize=8.5); ax.tick_params(labelsize=8)
fig.text(0.035,0.945,"All five shots from one velocity model - this whole block is a single training target",
         fontsize=13, color=TP, fontweight="bold")
fig.text(0.035,0.028,"Curved model. The direct arrival is the steep V at the top of every panel; everything beneath it is reflected energy,\n"
         "and that is the part the models get wrong.", fontsize=8.6, color=TS3, linespacing=1.45)
fig.subplots_adjust(left=0.055,right=0.985,top=0.865,bottom=0.185,wspace=0.13)
fig.savefig(os.path.join(OUT, "five_shots.png"), dpi=200); plt.close(fig)

# ---------- V4: flat vs curve ----------
fig, axes = plt.subplots(2,2, figsize=(9.8,7.1))
vpanel(axes[0,0], flat["vel"], "FlatVel-A style   velocity")
vpanel(axes[1,0], curve["vel"], "CurveVel-A style   velocity")
gpanel(axes[0,1], flat["gathers"][2], "resulting gather (centre shot)")
gpanel(axes[1,1], curve["gathers"][2], "resulting gather (centre shot)")
fig.text(0.045,0.955,"Flat against curved layering, and what each does to the recorded data",
         fontsize=13, color=TP, fontweight="bold")
fig.text(0.045,0.022,"Flat interfaces give clean, symmetric reflections. Curvature makes arrivals bend, cross and interfere - which is why\n"
         "every model scores worse on CurveVel than on FlatVel.", fontsize=8.6, color=TS3, linespacing=1.45)
fig.subplots_adjust(left=0.075,right=0.975,top=0.90,bottom=0.115,wspace=0.30,hspace=0.42)
fig.savefig(os.path.join(OUT, "flat_vs_curve.png"), dpi=200); plt.close(fig)
print("ok")
