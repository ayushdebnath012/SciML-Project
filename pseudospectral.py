import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

rho = 2500.0                 # Density (kg/m^3)
c   = 3000.0                 # Wave speed (m/s)
mu  = rho * c**2             # Shear modulus (Pa)

nx = 1000
dx = 1.0
L  = nx * dx
x  = np.linspace(0, L, nx, endpoint=False)

CFL = 0.14
dt  = CFL * dx / c
nt  = 4500                   
time = np.arange(nt) * dt

f0 = 25.0
t0 = 1.5 / f0

def ricker(t):
    a = np.pi * f0 * (t - t0)
    return (1 - 2*a**2) * np.exp(-a**2)

# High amplitude to be visible
src_time = 1e10 * ricker(time)

# Wide Gaussian injection
xsrc = L / 2
sigma = 4.0 * dx             
sg = np.exp(-(x - xsrc)**2 / sigma**2)
sg /= np.sum(sg) * dx

# (Real-FFT)

k = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)
k2 = k**2

# Gaussian Sponge
eta = np.zeros(nx)
nb = int(0.25 * nx)          
sponge_width_m = nb * dx
eta_max = 1.8                

for i in range(nx):
    dist = min(i, nx - 1 - i)
    if dist < nb:
        # Gaussian curve for smooth entry
        val = eta_max * np.exp(-((dist / nb) * 2.5)**2)
        eta[i] = val

# Simulation Loop with Envelope Tracking
u_nm1 = np.zeros(nx)
u_n   = np.zeros(nx)
u_np1 = np.zeros(nx)

snapshots = []
# This array will store the MAX amplitude seen at each point x
envelope = np.zeros(nx) 

# Auto-scale helper
global_max_strain = 0.0

print("Simulating...")
for it in range(nt):
    
    #  Pseudospectral Step 
    u_hat = np.fft.rfft(u_n)
    uxx_hat = -k2 * u_hat
    uxx = np.fft.irfft(uxx_hat)
    
    elastic_force = mu * uxx
    
    u_np1 = (2*u_n - u_nm1 + (dt**2 / rho) * (elastic_force + sg * src_time[it]))
    
    #  Apply Sponge 
    u_np1 *= (1 - eta * dt)
    
    # Shift buffers
    u_nm1[:] = u_n
    u_n[:]   = u_np1
    
    if it % 15 == 0:
        snapshots.append(u_n.copy())
        
        # Calculate Strain for Envelope
        u_hat_curr = np.fft.rfft(u_n)
        strain_curr = np.fft.irfft(1j * k * u_hat_curr)
        
        envelope = np.maximum(envelope, np.abs(strain_curr))
        
        # Track global max for Y-axis scaling
        current_max = np.max(np.abs(strain_curr))
        if current_max > global_max_strain:
            global_max_strain = current_max

print("Done.")


fig, ax = plt.subplots(figsize=(10, 5))

line_wave, = ax.plot([], [], lw=2, color='blue', label="Elastic Wave")


# Axis Limits
limit = global_max_strain * 1.1
ax.set_ylim(-limit, limit)
ax.set_xlim(0, L)
ax.set_xlabel("Position (m)")
ax.set_ylabel("Strain Amplitude")
ax.legend(loc='upper center')
ax.grid(True, alpha=0.3)

# Re-calculate envelopes for animation playback
# We need to rebuild the envelope progressively for the video
frame_envelopes = []
temp_env = np.zeros(nx)

for snap in snapshots:
    u_hat = np.fft.rfft(snap)
    strain = np.fft.irfft(1j * k * u_hat)
    temp_env = np.maximum(temp_env, np.abs(strain))
    frame_envelopes.append(temp_env.copy())

def update(frame):
    u = snapshots[frame]
    
    # Wave
    u_hat = np.fft.rfft(u)
    strain = np.fft.irfft(1j * k * u_hat)
    line_wave.set_data(x, strain)
    

ani = FuncAnimation(fig, update, frames=len(snapshots), interval=20, blit=False)
plt.show()