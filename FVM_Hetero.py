import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

NX   = 800
XMAX = 10000.0
DX   = XMAX / NX
x    = np.linspace(0, XMAX, NX)

# Heterogeneous Material
rho = np.ones(NX) * 2500.0
c   = np.ones(NX) * 2500.0
# Right half
c[x > XMAX/2] = 2500.0

mu = rho * c**2
Z  = rho * c

CFL = 0.5
c_max = np.max(c)
DT  = CFL * DX / c_max

A_plus  = np.zeros((NX, 2, 2))
A_minus = np.zeros((NX, 2, 2))
A_full  = np.zeros((NX, 2, 2))
A2_full = np.zeros((NX, 2, 2))

for i in range(NX):
    Ri = np.array([[Z[i], -Z[i]],
                   [1.0,   1.0]])
    Rinv = 1.0/(2*Z[i]) * np.array([[1.0, Z[i]],
                                    [-1.0, Z[i]]])

    Lambda_p = np.array([[0.0, 0.0],
                          [0.0, c[i]]])
    Lambda_m = np.array([[-c[i], 0.0],
                          [0.0,  0.0]])
    
    # Upwind Flux Matrices
    A_plus[i]  = Ri @ Lambda_p @ Rinv
    A_minus[i] = Ri @ Lambda_m @ Rinv
    
    # Full System Matrix A for Lax-Wendroff
    # A = [[0, -mu], [-1/rho, 0]]
    Ai = np.array([[0.0, -mu[i]], 
                   [-1.0/rho[i], 0.0]])
    A_full[i]  = Ai
    A2_full[i] = Ai @ Ai

def solve_hetero(scheme):
    X0    = 5000.0
    SIGMA = 20.0
    
    t_end = (XMAX / np.min(c)) * 1.5
    nt    = int(t_end / DT)
    
    print(f"[{scheme}] Simulating {nt} steps (dt={DT:.4e})...")

    # Initial Condition
    Q = np.zeros((2, NX))
    Q[0, :] = np.exp(-(x - X0)**2 / SIGMA**2) # Stress
    Q[1, :] = 0.0                             # Velocity
    
    Qnew = np.zeros_like(Q)
    
    # Storage
    save_skip = 5
    vel_hist = []
    time_hist = []
    
    for n in range(nt):
        Qnew[:] = Q[:]
        
        # Interior Update
        if scheme == "upwind":
            dQl = Q[:, 1:-1] - Q[:, 0:-2]               # ΔQ_i- = Q_i − Q_{i−1}
            dQr = Q[:, 2:]   - Q[:, 1:-1]               # ΔQ_i+ = Q_{i+1} − Q_i

            dQl_v = dQl.T.reshape(-1, 2, 1)             
            dQr_v = dQr.T.reshape(-1, 2, 1)             

            Ap = A_plus[1:-1]                           # A_i+  (lambda > 0)
            Am = A_minus[1:-1]                          # A_i-  (lambda < 0)

            res_p = Ap @ dQl_v                          # A_i+ (Q_i − Q_{i−1})
            res_m = Am @ dQr_v                          # A_i- (Q_{i+1} − Q_i)

            flux = res_p + res_m                        # F_i = A_i+deltaQ_i- + A_i-deltaQ_i+

            Qnew[:, 1:-1] = (
                Q[:, 1:-1]
                - (DT/DX) * flux.reshape(-1, 2).T       # Q_i^{n+1} = Q_i^n − (delta_t/delta_x)F_i
            )

        elif scheme == "lax-wendroff":
            dQ1 = Q[:, 2:] - Q[:, 0:-2]          # Central diff
            dQ2 = Q[:, 2:] - 2*Q[:, 1:-1] + Q[:, 0:-2] # Second order diff
            
            # Reshape
            dQ1_v = dQ1.T.reshape(-1, 2, 1)
            dQ2_v = dQ2.T.reshape(-1, 2, 1)
            
            A_local  = A_full[1:-1]
            A2_local = A2_full[1:-1]
            
            term1 = A_local @ dQ1_v
            term2 = A2_local @ dQ2_v
            
            Qnew[:, 1:-1] = Q[:, 1:-1] - (DT/(2*DX)) * term1.reshape(-1, 2).T \
                                       + 0.5 * (DT/DX)**2 * term2.reshape(-1, 2).T

        # Boundaries
        Qnew[:, 0]  = Qnew[:, 1]
        Qnew[:, -1] = Qnew[:, -2]
        
        Q[:] = Qnew[:]
        
        if n % save_skip == 0:
            vel_hist.append(Q[1].copy())
            time_hist.append(n * DT)
            
    return np.array(time_hist), np.array(vel_hist)


# Run Upwind
t, v_up = solve_hetero("upwind")

# Run Lax-Wendroff
_, v_lw = solve_hetero("lax-wendroff")

# Animation
fig, ax = plt.subplots(figsize=(10,6))
l_up, = ax.plot(x, v_up[0], 'b-', lw=1.5, label='Upwind (Diffusive)', alpha=0.8)
l_lw, = ax.plot(x, v_lw[0], 'r--', lw=1.5, label='Lax-Wendroff (Dispersive)', alpha=0.8)

# Material interface line
ax.axvline(XMAX/2, color='k', linestyle=':', label='Interface')

max_val = max(np.max(np.abs(v_up)), np.max(np.abs(v_lw)))
if max_val == 0: max_val = 1.0

ax.set_ylim(-max_val*1.2, max_val*1.2)
ax.set_xlim(0, XMAX)
ax.set_xlabel("x (m)")
ax.set_ylabel("Velocity")
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_title("FVM Heterogeneous Media: c=2500 -> c=6500")

txt = ax.text(0.02, 0.95, "", transform=ax.transAxes)

def update(k):
    l_up.set_ydata(v_up[k])
    l_lw.set_ydata(v_lw[k])
    txt.set_text(f"t = {t[k]:.3f} s")
    return l_up, l_lw, txt

anim = FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)
plt.show()