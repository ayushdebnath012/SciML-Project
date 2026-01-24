import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


nx = 800
xmax = 10000.0
dx = xmax / nx

rho = 2500.0
c = 2500.0
mu = rho * c**2

dt = 0.002
nt = 1200

x = np.linspace(0, xmax, nx)

v = np.zeros(nx)          # velocity v_i^(n+1/2)
v_old = np.zeros(nx)      # velocity v_i^(n-1/2)
sigma = np.zeros(nx - 1)  # stress σ_(i+1/2)^n

# Ricker Wavelet Source (Stress Source)
f0 = 5.0
t0 = 1.5 / f0
src_x = xmax / 2
src_i = int(src_x / dx)

def ricker(t, f0):
    return (1 - 2 * (np.pi * f0 * t)**2) * np.exp(-(np.pi * f0 * t)**2)

# Absorbing Boundary (Sponge Layer)
nb = 200
damping_max = 5.0
damping = np.zeros(nx)

for i in range(nb):
    w = ((nb - i) / nb)**2
    damping[i] = damping_max * w
    damping[-i - 1] = damping_max * w


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

line_sigma, = ax1.plot(x[:-1], sigma)
line_v, = ax2.plot(x, v)

ax1.set_ylabel("Stress σ")
ax2.set_ylabel("Velocity v")
ax2.set_xlabel("x (m)")

ax1.set_ylim(-1.2, 1.2)
ax2.set_ylim(-2e-7, 2e-7)

def update(frame):
    global v, v_old, sigma

    t = frame * dt

    # Velocity update
    for i in range(1, nx - 1):
        v[i] = (
            v_old[i]
            + (dt / rho) * (sigma[i] - sigma[i - 1]) / dx
        )
        v[i] *= np.exp(-damping[i] * dt)

    # Stress update
    for i in range(nx - 1):
        sigma[i] += dt * mu * (v[i + 1] - v[i]) / dx
        sigma[i] *= np.exp(-damping[i] * dt)

    # Ricker stress source
    sigma[src_i] = ricker(t - t0, f0)

    v_old[:] = v[:]

    line_sigma.set_ydata(sigma)
    line_v.set_ydata(v)
    ax1.set_title(f"Time = {t:.3f} s")

    return line_sigma, line_v

ani = FuncAnimation(fig, update, frames=nt, interval=20)
plt.tight_layout()
plt.show()

plt.savefig("1D_elastic_wave_ricker_absorbing.png")
