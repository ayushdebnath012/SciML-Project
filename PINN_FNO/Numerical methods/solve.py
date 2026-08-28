import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from tqdm import tqdm  # Progress bar
import time

from physics import *
from solvers import solve_fd, solve_ps, solve_fem, solve_fvm, solve_dg, solve_sem

# ============================================================
# 1. SETUP & CONFIGURATION (COSMETIC)
# ============================================================
print("-" * 30)
print("INITIALIZING NUMERICAL LAB")
print("-" * 30)

# Aesthetics
plt.style.use('dark_background')
plt.rcParams['axes.facecolor'] = '#121212'
plt.rcParams['figure.facecolor'] = '#121212'
NEON_COLORS = ['#00FFFF', '#FF00FF', '#FFFF00', '#FF3333', '#33FF33', '#3333FF'] # Cyan, Magenta, Yellow, Red, Green, Blue

NX_VISIBLE = 800  
NX_BASE    = int(NX_VISIBLE * (L_DOMAIN / L_VISIBLE)) 
DX_BASE    = L_DOMAIN / NX_BASE
X_AXIS_FULL = np.linspace(0, L_DOMAIN, NX_BASE)

# Use global time steps defined in physics.py (dynamic based on material)
# dt_global and nt_global are imported from physics.py

# Shared Sponge
nb = int(PAD_WIDTH / DX_BASE)
damp_mask = np.zeros(NX_BASE)
for i in range(nb):
    w = ((nb - i) / nb)**3
    damp_mask[i] = 30.0 * w; damp_mask[-i-1] = 30.0 * w

# ============================================================
# 2. EXECUTION WITH PROGRESS BARS
# ============================================================
solvers = [
    ("Finite Difference", solve_fd, (NX_BASE, nt_global, dt_global, DX_BASE, damp_mask, X_AXIS_FULL, X_AXIS_FULL)),
    ("Pseudo-Spectral",   solve_ps, (NX_BASE, nt_global, dt_global, DX_BASE, X_AXIS_FULL)),
    ("Finite Element",    solve_fem, (NX_BASE, nt_global, dt_global, DX_BASE, damp_mask, X_AXIS_FULL, X_AXIS_FULL)),
    ("Finite Volume",     solve_fvm, (NX_BASE,)),
    ("Disc. Galerkin",    solve_dg, ()),
    ("Spectral Element",  solve_sem, ())
]

results = {}

for name, func, args in solvers:
    print(f"\n[Process] Running {name}...")
    start_time = time.time()
    results[name] = func(*args)
    end_time = time.time()
    print(f"[Success] {name}: {end_time - start_time:.2f}s")

# Unpack results
x_fd,  v_fd,  t_fd  = results["Finite Difference"]
x_ps,  v_ps,  t_ps  = results["Pseudo-Spectral"]
x_fem, v_fem, t_fem = results["Finite Element"]
x_fvm, v_fvm, t_fvm = results["Finite Volume"]
x_dg,  v_dg,  t_dg  = results["Disc. Galerkin"]
x_sem, v_sem, t_sem = results["Spectral Element"]

# ============================================================
# 3. SAVE DATA & PREP ANIMATION
# ============================================================
print("\n[File] Saving data to 'numerical_lab_results.npz'...")
np.savez_compressed("numerical_lab_results.npz", 
                    fd_v=v_fd, ps_v=v_ps, fem_v=v_fem, 
                    fvm_v=v_fvm, dg_v=v_dg, sem_v=v_sem,
                    fd_t=t_fd, ps_t=t_ps, fem_t=t_fem,
                    fvm_t=t_fvm, dg_t=t_dg, sem_t=t_sem,
                    fvm_x=x_fvm, dg_x=x_dg, sem_x=x_sem) # Saving grids too

# ============================================================
# 4. ANIMATION & SYNCHRONIZATION
# ============================================================

# Interpolation Helper for Sync
def get_interpolated_val(t_target, v_arr, t_arr):
    """Interpolates visualization array to t_target."""
    idx = np.searchsorted(t_arr, t_target)
    if idx == 0: return v_arr[0]
    if idx >= len(t_arr): return v_arr[-1]
    
    t0, t1 = t_arr[idx-1], t_arr[idx]
    v0, v1 = v_arr[idx-1], v_arr[idx]
    
    # Avoid division by zero
    if t1 == t0: return v0
    
    alpha = (t_target - t0) / (t1 - t0)
    return v0 * (1 - alpha) + v1 * alpha

print("[Anim] Synchronizing timelines...")
FPS = 60
t_anim = np.linspace(0.0, T_MAX, int(T_MAX * FPS))

fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
fig.suptitle("Wave Propagation: Numerical Methods Comparison", fontsize=16, color='white')
axs = axs.flatten()
titles = ["FD", "PS", "FEM", "FVM", "DG", "SEM"]

lines = []
# Move time text to bottom to avoid title overlap
time_text = fig.text(0.5, 0.02, "", ha='center', fontsize=12, color='white')

for i, ax in enumerate(axs):
    ax.set_xlim(0, L_VISIBLE); ax.set_ylim(-0.25, 0.25)
    ax.set_title(titles[i], color=NEON_COLORS[i], fontweight='bold')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values(): spine.set_edgecolor('gray')
    ax.grid(alpha=0.1)
    
    ln, = ax.plot([], [], color=NEON_COLORS[i], lw=2)
    lines.append(ln)

def update_grid(frame):
    t_curr = t_anim[frame]
    
    # Interpolated Updates
    lines[0].set_data(X_AXIS_FULL - PAD_WIDTH, get_interpolated_val(t_curr, v_fd, t_fd))
    lines[1].set_data(X_AXIS_FULL - PAD_WIDTH, get_interpolated_val(t_curr, v_ps, t_ps))
    lines[2].set_data(X_AXIS_FULL - PAD_WIDTH, get_interpolated_val(t_curr, v_fem, t_fem))
    lines[3].set_data(x_fvm - PAD_WIDTH,       get_interpolated_val(t_curr, v_fvm, t_fvm))
    lines[4].set_data(x_dg - PAD_WIDTH,        get_interpolated_val(t_curr, v_dg, t_dg))
    lines[5].set_data(x_sem - PAD_WIDTH,       get_interpolated_val(t_curr, v_sem, t_sem))
    
    time_text.set_text(f"Simulation Time: {t_curr:.3f} s")
    return lines + [time_text]

ani = FuncAnimation(fig, update_grid, frames=len(t_anim), interval=1000/FPS, blit=False)

# ============================================================
# 5. VIDEO EXPORT
# ============================================================

writer = PillowWriter(fps=30)
ani.save("wave_simulation.gif", writer=writer)
print("Saved 'wave_simulation.gif'")


plt.show()