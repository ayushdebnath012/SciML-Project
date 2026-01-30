import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------- GRID ----------------
nx = 800
xmax = 10000.0
dx = xmax / nx
x = np.linspace(0, xmax, nx)

# ---------------- MATERIAL (DONE PROPERLY) ----------------
rho = np.ones(nx) * 2500.0
c   = np.ones(nx) * 2500.0

# Right half: heterogeneous wave speed
alpha = .4        # ±15% variation
L_hetero = 2000.0

mask = x > xmax/2
c[mask] = 2500.0 * (
    1.0 + alpha * np.sin(2*np.pi*(x[mask] - xmax/2) / L_hetero)
)

# Derived material properties
mu = rho * c**2

# ---------------- TIME (CFL-consistent) ----------------
CFL = 0.5
dt = CFL * dx / np.max(c)
nt = 1200

# ---------------- FIELDS ----------------
v = np.zeros(nx)
v_old = np.zeros(nx)
sigma = np.zeros(nx - 1)

# ---------------- SOURCE ----------------
f0 = 5.0
t0 = 1.5 / f0
src_i = nx // 2   # source at center (homogeneous region)

def ricker(t, f0):
    return (1 - 2*(np.pi*f0*t)**2) * np.exp(-(np.pi*f0*t)**2)

# Reference source scaling (homogeneous)
epsilon = 3e-5
mu_ref = 2500.0 * 2500.0**2
f_source = mu_ref * epsilon / dx

# ---------------- DAMPING ----------------
nb = 200
damping_max = 5.0
damping = np.zeros(nx)

for i in range(nb):
    w = ((nb - i) / nb)**2
    damping[i] = damping_max * w
    damping[-i - 1] = damping_max * w

# ---------------- PLOT ----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

line_sigma, = ax1.plot(x[:-1], sigma / 1e6)
line_v, = ax2.plot(x, v)

ax1.set_ylabel("Stress σ (MPa)")
ax2.set_ylabel("Velocity v (m/s)")
ax2.set_xlabel("x (m)")

ax1.set_ylim(-1.0, 1.0)
ax2.set_ylim(-0.08, 0.08)

ax1.axvline(xmax/2, color='k', linestyle='--')
ax2.axvline(xmax/2, color='k', linestyle='--')

# ---------------- UPDATE ----------------
def update(frame):
    global v, v_old, sigma
    t = frame * dt

    # Velocity update (Eq. 4.67)
    for i in range(1, nx-1):
        v[i] = v_old[i] + (dt / rho[i]) * (sigma[i] - sigma[i-1]) / dx
        v[i] *= np.exp(-damping[i] * dt)

    # Source (clean, homogeneous)
    v[src_i] += dt * f_source * ricker(t - t0, f0) / rho[src_i]

    # Stress update (Eq. 4.68)
    for i in range(nx-1):
        sigma[i] += dt * mu[i] * (v[i+1] - v[i]) / dx
        sigma[i] *= np.exp(-damping[i] * dt)

    v_old[:] = v[:]

    line_sigma.set_ydata(sigma / 1e6)
    line_v.set_ydata(v)
    ax1.set_title(f"Time = {t:.3f} s")

    return line_sigma, line_v

ani = FuncAnimation(fig, update, frames=nt, interval=20)
plt.tight_layout()
plt.show()
