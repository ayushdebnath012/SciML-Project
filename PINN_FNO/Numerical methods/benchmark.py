import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from physics import * 

# 1. Setup parameters (Matching physics.py)
nx = 800
x = np.linspace(0, L_DOMAIN, nx) # Full Domain
t_max = 2.4
nt = 200
t_axis = np.linspace(0, t_max, nt)

# 2. Analytical Solution (Ray Tracing)
def calculate_arrival_times_and_amplitudes(x_grid):
    """
    Pre-calculates arrival times and amplitude factors for the entire grid
    based on Ray Tracing through MAT_BOUNDARIES.
    """
    arrival_times = np.zeros_like(x_grid)
    amp_factors   = np.zeros_like(x_grid)
    
    x0 = SOURCE_X
    boundaries = np.sort(MAT_BOUNDARIES)
    velocities = np.array(MAT_VELOCITIES)
    
    # Identify Source Zone
    src_idx = np.searchsorted(boundaries, x0)
    c_source = velocities[src_idx]
    Z_source = RHO_CONST * c_source
    
    base_amp = AMP_SCALE / (2 * Z_source)
    
    # --- Propagate RIGHT from Source ---
    mask_right = x_grid >= x0
    x_r = x_grid[mask_right]
    t_r = np.zeros_like(x_r)
    a_r = np.ones_like(x_r) * base_amp
    
    bounds_right = boundaries[boundaries > x0]
    
    last_x, last_time, last_amp, last_v = x0, 0.0, base_amp, c_source
    
    params = []
    
    for b in bounds_right:
        # Segment from last_x to b
        params.append( (last_x, b, last_v, last_time, last_amp) )
        
        dist = b - last_x
        last_time += dist / last_v
        last_x = b
        
        # New Zone
        next_zone_idx = np.searchsorted(boundaries, b + 1e-5)
        next_v = velocities[next_zone_idx]
        next_Z = RHO_CONST * next_v
        last_Z = RHO_CONST * last_v
        
        T = 2 * last_Z / (last_Z + next_Z)
        last_amp *= T
        last_v = next_v
        
    params.append( (last_x, L_DOMAIN, last_v, last_time, last_amp) )
    
    for px_start, px_end, p_v, p_t, p_a in params:
        seg_mask = (x_r >= px_start) & (x_r < px_end)
        if np.any(seg_mask):
            dists = x_r[seg_mask] - px_start
            t_r[seg_mask] = p_t + dists / p_v
            a_r[seg_mask] = p_a
            
    arrival_times[mask_right] = t_r
    amp_factors[mask_right]   = a_r
    
    # --- Propagate LEFT from Source ---
    mask_left = x_grid < x0
    x_l = x_grid[mask_left]
    t_l = np.zeros_like(x_l)
    a_l = np.ones_like(x_l) * base_amp
    
    bounds_left = boundaries[boundaries < x0][::-1] 
    
    last_x, last_time, last_amp, last_v = x0, 0.0, base_amp, c_source
    
    params_l = []
    
    for b in bounds_left:
        params_l.append( (b, last_x, last_v, last_time, last_amp) )
        
        dist = last_x - b
        last_time += dist / last_v
        last_x = b
        
        prev_zone_idx = np.searchsorted(boundaries, b - 1e-5)
        next_v = velocities[prev_zone_idx]
        next_Z = RHO_CONST * next_v
        last_Z = RHO_CONST * last_v
        
        T = 2 * last_Z / (last_Z + next_Z)
        last_amp *= T
        last_v = next_v
        
    params_l.append( (0.0, last_x, last_v, last_time, last_amp) )
    
    for px_start, px_end, p_v, p_t, p_a in params_l:
        seg_mask = (x_l >= px_start) & (x_l < px_end)
        if np.any(seg_mask):
            dists = px_end - x_l[seg_mask]
            t_l[seg_mask] = p_t + dists / p_v
            a_l[seg_mask] = p_a
            
    arrival_times[mask_left] = t_l
    amp_factors[mask_left] = a_l

    return arrival_times, amp_factors

print("[Benchmark] Computing Ray Tracing paths...")
grid_arrival_times, grid_amp_factors = calculate_arrival_times_and_amplitudes(x)

def get_analytical_vals(x_grid, t):
    # Vectorized Ray Tracing
    # ricker_source(t) = ... (t-T0)
    # arrival_time is equivalent to dist/c
    # So we call ricker_source(t - arrival_time)
    return grid_amp_factors * ricker_source(t - grid_arrival_times)

# 3. Visualization
plt.style.use('dark_background')
plt.rcParams['axes.facecolor'] = '#121212'
plt.rcParams['figure.facecolor'] = '#121212'

fig, ax = plt.subplots(figsize=(10, 5))
line, = ax.plot(x, np.zeros(nx), color='cyan', lw=2, label='Analytical Velocity')

# Plot Source Position
ax.axvline(x=SOURCE_X, color='#777777', linestyle='-.', label='Source')

# Plot ALL Material Boundaries
for b in MAT_BOUNDARIES:
    ax.axvline(x=b, color='white', linestyle=':', label='Interface', alpha=0.5)

ax.set_ylim(-0.15, 0.15) 
ax.set_title("Benchmark: Generic Material Propagation")
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Velocity (m/s)")

# Deduplicate legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())
ax.grid(True, alpha=0.3)

def update(frame):
    t = t_axis[frame]
    v = get_analytical_vals(x, t)
    line.set_ydata(v)
    ax.set_title(f"Benchmark: t={t:.3f}s")
    return line,

ani = FuncAnimation(fig, update, frames=nt, interval=30, blit=False)
plt.show()