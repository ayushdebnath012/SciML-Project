import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

rho  = 2500.0
vs   = 2500.0
mu   = rho * vs**2


xmax = 10_000.0
ne   = 200        
N    = 4          # Higher order for better accuracy

x0    = 5000.0      # Pulse center 
sigma = 20.0 

def gll(N):
    if N == 1:
        xi = np.array([-1.0, 1.0])
        w  = np.array([1.0, 1.0])
    elif N == 2:
        xi = np.array([-1.0, 0.0, 1.0])
        w  = np.array([1/3, 4/3, 1/3])
    elif N == 3:
        xi = np.array([-1.0, -np.sqrt(1/5), np.sqrt(1/5), 1.0])
        w  = np.array([1/6, 5/6, 5/6, 1/6])
    elif N == 4:
        xi = np.array([-1.0, -np.sqrt(3/7), 0.0, np.sqrt(3/7), 1.0])
        w  = np.array([1/10, 49/90, 32/45, 49/90, 1/10])
    return xi, w

xi, w = gll(N)

def lagrange_derivative_matrix(xi):
    N = len(xi) - 1
    D = np.zeros((N+1, N+1))

    w_bary = np.ones(N+1)
    for j in range(N+1):
        for k in range(N+1):
            if j != k:
                w_bary[j] *= (xi[j] - xi[k])
        w_bary[j] = 1.0 / w_bary[j]

    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                D[i,j] = (w_bary[j]/w_bary[i]) / (xi[i]-xi[j])
        D[i,i] = -np.sum(D[i,:])
        
    return D

l1d = lagrange_derivative_matrix(xi)

h  = xmax / ne
J  = h / 2
Ji = 2 / h

dx_min = np.min(np.diff(xi)) * J
dt = 0.5 * dx_min / vs # Stability
nt = 3000

ng = ne * N + 1
x  = np.linspace(0, xmax, ng)

x_nodes = np.zeros(ng)
for e in range(ne):
    x_local = (xi + 1) * h/2 + e*h
    idx_start = e * N
    x_nodes[idx_start : idx_start + N + 1] = x_local
x = x_nodes

Me = rho * w * J #Element mass matrix
Ke = np.zeros((N+1, N+1))
for i in range(N+1):
    for j in range(N+1):
        # Integrate \phi'_i \phi'_j
        val = 0.0
        for k in range(N+1):
            val += w[k] * l1d[k, i] * l1d[k, j] # D_ki is dphi_i/dxi at k
        Ke[i, j] = val * (mu * Ji)

# Global assembly

Mg = np.zeros(ng)
K  = np.zeros((ng, ng))

for e in range(ne):
    for i in range(N+1):
        idx_i = e*N + i
        Mg[idx_i] += Me[i]
        
        for j in range(N+1):
            idx_j = e*N + j
            K[idx_i, idx_j] += Ke[i, j]

Minv = np.zeros_like(Mg)
mask = Mg > 1e-12
Minv[mask] = 1.0 / Mg[mask]

#Initial Conditions
u    = np.zeros(ng)
u[:] = np.exp(-(x - x0)**2 / sigma**2)
uold = u.copy()

frames_u = []

print(f"Simulating {nt} steps...")
for n in range(nt):
    # Central difference
    # M a = -K u
    accel = -Minv * (K @ u)
    unew  = 2*u - uold + dt**2 * accel
    
    # Revert to Displacement storage
    uold = u[:]
    u    = unew[:]

    if n % 20 == 0:
        frames_u.append(u.copy())

# Animation
fig, ax = plt.subplots(figsize=(10,4))
line, = ax.plot(x, frames_u[0], lw=1.5)

ax.set_xlim(0, xmax)
umax = np.max(np.abs(frames_u)) 
if umax == 0: umax = 1
ax.set_ylim(-umax*1.2, umax*1.2)

ax.set_xlabel("x (m)")
ax.set_ylabel("Displacement")
ax.set_title(f"SEM")

txt = ax.text(0.02, 0.95, "", transform=ax.transAxes)

def update(frame_idx):
    line.set_ydata(frames_u[frame_idx])
    txt.set_text(f"Step {frame_idx*20}")
    return line, txt

ani = FuncAnimation(fig, update, frames=len(frames_u), interval=30, blit=True)
plt.show()
