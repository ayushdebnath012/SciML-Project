import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Grid
# -----------------------------
L  = 10000.0
nx = 800
x  = np.linspace(0, L, nx)
h  = x[1] - x[0]

# -----------------------------
# Physical parameters (HETEROGENEOUS)
# -----------------------------
rho = np.ones(nx) * 2500.0
vs  = np.ones(nx) * 3000.0

# Right half: heterogeneous material
vs[x > L/2] = 5000.0  # Increased shear wave velocity

mu = rho * vs**2  # Now mu is spatially varying

# -----------------------------
# Time stepping
# -----------------------------
CFL = 0.5
vs_max = np.max(vs)
dt  = CFL * h / vs_max
nt  = 1200

# -----------------------------
# Source
# -----------------------------
f0 = 5.0
t0 = 1.5 / f0
epsilon = 3e-5
f_amp = mu[nx//2] * epsilon  # Use mu at source location

def ricker(t):
    a = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * a**2) * np.exp(-a**2)

isrc = np.argmin(np.abs(x - L / 2))

# -----------------------------
# FEM assembly (lumped mass) - HETEROGENEOUS
# -----------------------------
M_lumped = np.zeros(nx)
K = np.zeros((nx, nx))

for e in range(nx - 1):
    i, j = e, e + 1
    
    # Use average mu for the element
    mu_elem = 0.5 * (mu[i] + mu[j])
    ke = mu_elem / h
    
    K[i, i] += ke
    K[i, j] -= ke
    K[j, i] -= ke
    K[j, j] += ke

    # Use average rho for the element
    rho_elem = 0.5 * (rho[i] + rho[j])
    me = rho_elem * h / 2.0
    M_lumped[i] += me
    M_lumped[j] += me

Minv = 1.0 / M_lumped

# -----------------------------
# Fields
# -----------------------------
u     = np.zeros(nx)
u_old = np.zeros(nx)

snap_u = []
snap_v = []
snap_s = []

# -----------------------------
# Time loop
# -----------------------------
for it in range(nt):
    t = it * dt

    f_internal = -K @ u
    f_source = np.zeros(nx)
    f_source[isrc] = f_amp * ricker(t)

    acc = (f_internal + f_source) * Minv
    u_new = dt**2 * acc + 2*u - u_old

    # Boundary condition
    u_new[0] = 0.0

    # Velocity (time-centered)
    v = (u_new - u_old) / (2 * dt)

    # Stress (central difference) - spatially varying mu
    du_dx = np.zeros(nx)
    du_dx[1:-1] = (u[2:] - u[:-2]) / (2 * h)
    sigma = mu * du_dx  # mu is now an array

    u_old, u = u, u_new

    if it % 10 == 0:
        snap_u.append(u.copy())
        snap_v.append(v.copy())
        snap_s.append(sigma.copy())

print("Max displacement:", np.max(np.abs(snap_u)))
print("Max velocity:", np.max(np.abs(snap_v)))
print("Max stress (Pa):", np.max(np.abs(snap_s)))

# -----------------------------
# Plot / Animation
# -----------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

line_u, = ax1.plot(x, snap_u[0], lw=2)
line_v, = ax2.plot(x, snap_v[0], lw=2)
line_s, = ax3.plot(x, snap_s[0] / 1e6, lw=2)  # MPa

# Add interface line to all subplots
for ax in (ax1, ax2, ax3):
    ax.axvline(L/2, color='k', linestyle=':', alpha=0.5, label='Interface')
    ax.grid(alpha=0.3)

ax1.set_ylabel("Displacement (m)")
ax2.set_ylabel("Velocity (m/s)")
ax3.set_ylabel("Stress σ (MPa)")
ax3.set_xlabel("x (m)")

ax1.set_title("1D Elastic Wave — FEM (Heterogeneous)")
ax1.legend(loc='upper right')

ax1.set_ylim(-1.2*np.max(np.abs(snap_u)), 1.2*np.max(np.abs(snap_u)))
ax2.set_ylim(-1.2*np.max(np.abs(snap_v)), 1.2*np.max(np.abs(snap_v)))
ax3.set_ylim(-1.2*np.max(np.abs(snap_s))/1e6,
              1.2*np.max(np.abs(snap_s))/1e6)

def update(frame):
    line_u.set_ydata(snap_u[frame])
    line_v.set_ydata(snap_v[frame])
    line_s.set_ydata(snap_s[frame] / 1e6)
    ax1.set_title(f"1D Elastic Wave — FEM (Heterogeneous) | Time = {frame * 10 * dt:.3f} s")
    return line_u, line_v, line_s

ani = FuncAnimation(fig, update,
                    frames=len(snap_u),
                    interval=30,
                    blit=False)

plt.tight_layout()
plt.show()