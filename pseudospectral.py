import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Grid
# -----------------------------
nx = 1000
dx = 1.0
L  = nx * dx
x  = np.linspace(0, L, nx, endpoint=False)

# -----------------------------
# Physical parameters (HETEROGENEOUS)
# -----------------------------
rho = np.ones(nx) * 2500.0       # Density (kg/m^3)
c   = np.ones(nx) * 3000.0       # Wave speed (m/s)

# Right half: heterogeneous material
c[x > L/2] = 5000.0              # Increased wave speed

mu  = rho * c**2                 # Shear modulus (Pa) - now spatially varying

# -----------------------------
# Time stepping
# -----------------------------
CFL = 0.14
c_max = np.max(c)
dt  = CFL * dx / c_max
nt  = 4500
time = np.arange(nt) * dt

# -----------------------------
# Ricker source
# -----------------------------
f0 = 25.0
t0 = 1.5 / f0

def ricker(t):
    a = np.pi * f0 * (t - t0)
    return (1 - 2*a**2) * np.exp(-a**2)

src_time = 1.7e7 * ricker(time)

# -----------------------------
# Spatial source (Gaussian)
# -----------------------------
xsrc = L / 2
sigma = 4.0 * dx
sg = np.exp(-(x - xsrc)**2 / sigma**2)
sg /= np.sum(sg) * dx

# -----------------------------
# Spectral operators
# -----------------------------
k  = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)
k2 = k**2

# -----------------------------
# Gaussian sponge
# -----------------------------
eta = np.zeros(nx)
nb = int(0.25 * nx)
sponge_width_m = nb * dx
eta_max = 1.8

for i in range(nx):
    dist = min(i, nx - 1 - i)
    if dist < nb:
        eta[i] = eta_max * np.exp(-((dist / nb) * 2.5)**2)

# -----------------------------
# Fields
# -----------------------------
u_nm1 = np.zeros(nx)
u_n   = np.zeros(nx)
u_np1 = np.zeros(nx)

snapshots     = []
vel_snapshots = []

envelope = np.zeros(nx)
global_max_strain = 0.0

print("Simulating...")

# -----------------------------
# Time loop
# -----------------------------
for it in range(nt):

    # Spectral second derivative
    u_hat = np.fft.rfft(u_n)
    uxx   = np.fft.irfft(-k2 * u_hat)

    # Heterogeneous elastic force: mu(x) * u_xx
    elastic_force = mu * uxx

    # Leapfrog update with spatially varying density
    u_np1 = (
        2*u_n - u_nm1
        + (dt**2 / rho) * (elastic_force + sg * src_time[it])
    )

    # Sponge
    u_np1 *= (1 - eta * dt)

    # Velocity (time-centered)
    velocity = (u_np1 - u_nm1) / (2 * dt)

    # Shift buffers
    u_nm1[:] = u_n
    u_n[:]   = u_np1

    # Store every 15 steps
    if it % 15 == 0:
        snapshots.append(u_n.copy())
        vel_snapshots.append(velocity.copy())

        # Strain
        strain = np.fft.irfft(1j * k * np.fft.rfft(u_n))

        envelope = np.maximum(envelope, np.abs(strain))
        global_max_strain = max(global_max_strain, np.max(np.abs(strain)))

print("Done.")

# -----------------------------
# Prepare envelope for animation
# -----------------------------
frame_envelopes = []
temp_env = np.zeros(nx)

for snap in snapshots:
    strain = np.fft.irfft(1j * k * np.fft.rfft(snap))
    temp_env = np.maximum(temp_env, np.abs(strain))
    frame_envelopes.append(temp_env.copy())

# -----------------------------
# Plot setup
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Strain
line_strain, = ax1.plot([], [], lw=2, color='blue', label="Strain")
line_env,    = ax1.plot([], [], lw=1.5, color='gray', linestyle='--',
                        alpha=0.6, label="Strain Envelope")

# Velocity
line_vel, = ax2.plot([], [], lw=2, color='green', label="Velocity")

# Material interface
for ax in (ax1, ax2):
    ax.axvline(L/2, color='black', linestyle=':', linewidth=2, alpha=0.7, label='Interface')
    
# Sponge visualization
for ax in (ax1, ax2):
    ax.axvline(sponge_width_m, color='red', linestyle=':', alpha=0.5)
    ax.axvline(L - sponge_width_m, color='red', linestyle=':', alpha=0.5)
    ax.axvspan(0, sponge_width_m, color='red', alpha=0.05)
    ax.axvspan(L - sponge_width_m, L, color='red', alpha=0.05)

# Limits and labels
limit = global_max_strain * 1.1
ax1.set_ylim(-limit, limit)
ax1.set_ylabel("Strain")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_title("Spectral Method - Heterogeneous (c=3000 -> c=5000)")

vel_max = max(np.max(np.abs(v)) for v in vel_snapshots)
ax2.set_ylim(-1.1 * vel_max, 1.1 * vel_max)
ax2.set_xlabel("Position (m)")
ax2.set_ylabel("Velocity (m/s)")
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

ax2.set_xlim(0, L)

# -----------------------------
# Animation update
# -----------------------------
def update(frame):
    u = snapshots[frame]
    v = vel_snapshots[frame]

    strain = np.fft.irfft(1j * k * np.fft.rfft(u))

    line_strain.set_data(x, strain)
    line_env.set_data(x, frame_envelopes[frame])
    line_vel.set_data(x, v)

    ax1.set_title(f"Spectral Method - Heterogeneous | Time = {frame * 15 * dt:.3f} s")

    return line_strain, line_env, line_vel

ani = FuncAnimation(fig, update,
                    frames=len(snapshots),
                    interval=20,
                    blit=False)

plt.tight_layout()
plt.show()