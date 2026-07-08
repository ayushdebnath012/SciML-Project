import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from physics import * 

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
METHODS_TO_PLOT = [
    #"Analytical", # Benchmark
    "FD", "PS", "FEM", "FVM", "DG", "SEM" 
]

print("[File] Loading saved data...")
try:
    data = np.load("numerical_lab_results.npz")
except FileNotFoundError:
    print("'numerical_lab_results.npz' not found.")
    print("Please run 'python animate.py' first to generate the data.")
    exit()

# Aesthetics
plt.style.use('dark_background')
plt.rcParams['axes.facecolor'] = '#121212'
plt.rcParams['figure.facecolor'] = '#121212'

COLOR_MAP = {
    "Analytical": 'white', 
    "FD": '#00FFFF', "PS": '#FF00FF', "FEM": '#FFFF00', 
    "FVM": '#FF3333', "DG": '#33FF33', "SEM": '#3333FF'
}

def get_data(key):
    if key in data: return data[key]
    if key.endswith('_t'):
        v_key = key.replace('_t', '_v')
        if v_key in data: return np.linspace(0.0, T_MAX, data[v_key].shape[0])
    raise KeyError(f"Missing '{key}' in NPZ file.") 

NX_BASE = get_data('fd_v').shape[1]
X_AXIS_FULL_GRID = np.linspace(0, L_DOMAIN, NX_BASE)

result_map = {
    "FD":  {"v": get_data('fd_v'),  "t": get_data('fd_t'),  "x_full": X_AXIS_FULL_GRID},
    "PS":  {"v": get_data('ps_v'),  "t": get_data('ps_t'),  "x_full": X_AXIS_FULL_GRID},
    "FEM": {"v": get_data('fem_v'), "t": get_data('fem_t'), "x_full": X_AXIS_FULL_GRID},
    "FVM": {"v": get_data('fvm_v'), "t": get_data('fvm_t'), "x_full": data.get('fvm_x', X_AXIS_FULL_GRID)},
    "DG":  {"v": get_data('dg_v'),  "t": get_data('dg_t'),  "x_full": data.get('dg_x', X_AXIS_FULL_GRID)},
    "SEM": {"v": get_data('sem_v'), "t": get_data('sem_t'), "x_full": data.get('sem_x', X_AXIS_FULL_GRID)},
}

# ==========================================
# 2. ANALYTICAL SOLUTION (SIMPLIFIED / REVERTED)
# ==========================================
def get_analytical_wave(x_grid, t):
    grid_v = np.zeros_like(x_grid)
    x0 = SOURCE_X
    
    # Check if we are on a boundary
    boundaries = np.array(MAT_BOUNDARIES)
    velocities = np.array(MAT_VELOCITIES)
    
    # Determine Source Properties
    # If source is on a boundary, we split left/right properties.
    # If inside a zone, we assume homogeneous medium for that side.
    
    dist_to_boundary = np.min(np.abs(boundaries - x0)) if len(boundaries) > 0 else 1.0
    
    if len(boundaries) > 0 and dist_to_boundary < 1e-5:
        # Interface Source
        b_idx = np.argmin(np.abs(boundaries - x0))
        c_L = velocities[b_idx]
        c_R = velocities[b_idx + 1]
        Z_L = RHO_CONST * c_L
        Z_R = RHO_CONST * c_R
        factor = AMP_SCALE / (Z_L + Z_R)
    else:
        # Homogeneous Source (inside a zone)
        idx = np.searchsorted(boundaries, x0)
        c_L = velocities[idx]
        c_R = velocities[idx]
        Z = RHO_CONST * c_L
        factor = AMP_SCALE / (2 * Z)

    # Calculate Wave
    # Note: This does NOT account for layers further away (no Ray Tracing).
    # It assumes the medium is c_L everywhere to the left, and c_R everywhere to the right.
    
    # Left Side
    mask_left = x_grid <= x0
    if np.any(mask_left):
        dist = np.abs(x_grid[mask_left] - x0)
        grid_v[mask_left] = factor * ricker_source(t - dist / c_L)
    
    # Right Side
    mask_right = x_grid > x0
    if np.any(mask_right):
        dist = np.abs(x_grid[mask_right] - x0)
        grid_v[mask_right] = factor * ricker_source(t - dist / c_R)

    return grid_v

result_map["Analytical"] = {"v": None, "t": None, "x_full": X_AXIS_FULL_GRID, "is_analytical": True}

# ==========================================
# 3. METRICS CALCULATION
# ==========================================
error_metrics = {}

if "Analytical" in METHODS_TO_PLOT:
    print("\n[Metrics] Calculating L2 Relative Error vs Analytical Solution...")
    print("-" * 60)
    print(f"{'METHOD':<15} | {'L2 RELATIVE ERROR':<20}")
    print("-" * 60)

    FPS = 60
    t_eval = np.linspace(0, T_MAX, int(T_MAX * FPS))

    def get_interp(t_target, v_arr, t_arr):
        idx = np.searchsorted(t_arr, t_target)
        if idx == 0: return v_arr[0]
        if idx >= len(t_arr): return v_arr[-1]
        t0, t1 = t_arr[idx-1], t_arr[idx]
        v0, v1 = v_arr[idx-1], v_arr[idx]
        if t1 == t0: return v0
        alpha = (t_target - t0) / (t1 - t0)
        return v0 * (1 - alpha) + v1 * alpha

    for method in METHODS_TO_PLOT:
        if method == "Analytical": continue
        if method not in result_map: continue

        dat = result_map[method]
        x_grid = dat["x_full"]
        sum_diff_sq, sum_ana_sq = 0.0, 0.0

        for t in t_eval:
            u_num = get_interp(t, dat["v"], dat["t"])
            u_ana = get_analytical_wave(x_grid, t)
            sum_diff_sq += np.sum((u_num - u_ana)**2)
            sum_ana_sq  += np.sum(u_ana**2)

        l2_error = np.sqrt(sum_diff_sq / sum_ana_sq) if sum_ana_sq > 0 else 0.0
        error_metrics[method] = l2_error

        print(f"{method:<15} | {l2_error:.4f}")

    print("-" * 60 + "\n")

# ==========================================
# 4. ANIMATION SETUP
# ==========================================
t_anim = np.linspace(0, T_MAX, int(T_MAX * 60))
fig, ax = plt.subplots(figsize=(15, 8))
fig.suptitle("Wave Propagation: Numerical vs Analytical Benchmark", fontsize=16, color='white')

for b in MAT_BOUNDARIES:
    boundary_loc = b - PAD_WIDTH
    if 0 <= boundary_loc <= L_VISIBLE:
        ax.axvline(x=boundary_loc, color='#777777', linestyle=':', linewidth=1.5, label='Interface')

ax.set_xlim(0, L_VISIBLE); ax.set_ylim(-0.25, 0.25)
ax.set_xlabel("Position (m)", color='white'); ax.set_ylabel("Amplitude", color='white')
ax.tick_params(colors='white'); ax.grid(alpha=0.1)
for spine in ax.spines.values(): spine.set_edgecolor('gray')

lines = {}
for method in METHODS_TO_PLOT:
    if method not in result_map: continue
    
    is_ana = (method == "Analytical")
    style, width, alpha = ('--', 2.5, 1.0) if is_ana else ('-', 2.0, 0.8)
    color = COLOR_MAP.get(method, 'white')
    
    label = method
    if not is_ana and method in error_metrics:
        err_val = error_metrics[method]
        label = f"{method} (Err: {err_val:.4f})"
    
    ln, = ax.plot([], [], color=color, lw=width, ls=style, label=label, alpha=alpha)
    lines[method] = ln

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right', facecolor='#222222', edgecolor='white', labelcolor='white')

time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=12, color='white', ha='left')

# Quick fix for interp function scope for animation
def get_interp_anim(t_target, v_arr, t_arr):
    idx = np.searchsorted(t_arr, t_target)
    if idx == 0: return v_arr[0]
    if idx >= len(t_arr): return v_arr[-1]
    t0, t1 = t_arr[idx-1], t_arr[idx]
    v0, v1 = v_arr[idx-1], v_arr[idx]
    if t1 == t0: return v0
    alpha = (t_target - t0) / (t1 - t0)
    return v0 * (1 - alpha) + v1 * alpha

def update(frame):
    t_curr = t_anim[frame]
    artists = []
    
    for method, ln in lines.items():
        dat = result_map[method]
        x_view = dat["x_full"] - PAD_WIDTH
        
        if method == "Analytical":
            v_vals = get_analytical_wave(dat["x_full"], t_curr)
            ln.set_data(x_view, v_vals)
        else:
            v_interp = get_interp_anim(t_curr, dat["v"], dat["t"])
            ln.set_data(x_view, v_interp)
        artists.append(ln)
    
    time_text.set_text(f"Time: {t_curr:.3f} s")
    artists.append(time_text)
    return artists

ani = FuncAnimation(fig, update, frames=len(t_anim), interval=1000/60, blit=False)
plt.show()