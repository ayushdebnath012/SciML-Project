import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

rho = 2500.0                  
vs  = 3000.0                 
mu  = rho * vs**2            

L  = 10000.0                  
nx = 401                     
x  = np.linspace(0, L, nx)
h  = x[1] - x[0]

dt = 0.001                    
nt = 3000                     

f0 = 20.0                     # dominant frequency (Hz)
t0 = 1.5 / f0

def ricker(t):
    a = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * a**2) * np.exp(-a**2)

# source location
xs   = L / 2
isrc = np.argmin(np.abs(x - xs))

# FEM matrices (linear elements)
M = np.zeros((nx, nx))
K = np.zeros((nx, nx))

for i in range(1, nx - 1):
    # Mass matrix
    M[i, i]     = 2.0 * rho * h / 3.0
    M[i, i - 1] = rho * h / 6.0
    M[i, i + 1] = rho * h / 6.0

    # Stiffness matrix
    K[i, i]     = 2.0 * mu / h
    K[i, i - 1] = -mu / h
    K[i, i + 1] = -mu / h

# boundary nodes
M[0, 0]   = rho * h / 3.0
M[-1, -1] = rho * h / 3.0

# Initial conditions (rest)
u     = np.zeros(nx)
u_old = np.zeros(nx)

# 7. Time integration 
snapshots = []

for it in range(nt):
    t = it * dt

    # Force vector 
    f = np.zeros(nx)
    f[isrc] = ricker(t) / h

    # Acceleration solve
    acc = np.linalg.solve(M, f - K @ u)

    # Central difference update
    u_new = dt**2 * acc + 2.0 * u - u_old

    # REMOVE RIGID-BODY MODE 
    u_new[0] = 0.0

    u_old, u = u, u_new

    if it % 5 == 0:
        snapshots.append(u.copy())

print("Max displacement:", np.max(np.abs(snapshots)))

fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot(x, snapshots[0], lw=2)

amp = np.max(np.abs(snapshots))
ax.set_xlim(0, L)
ax.set_ylim(-1.2 * amp, 1.2 * amp)

ax.set_xlabel("x (m)")
ax.set_ylabel("Displacement")
ax.set_title("1D Elastic Wave Propagation (FEM)")

def update(frame):
    line.set_ydata(snapshots[frame])
    ax.set_title(f"Time = {frame * 5 * dt:.3f} s")
    return line,

ani = FuncAnimation(fig, update,
                    frames=len(snapshots),
                    interval=30)

ani.save("elastic_wave_1D_FEM.gif",
         writer=PillowWriter(fps=30))

plt.show()
print("Saved: elastic_wave_1D_FEM.gif")
