import numpy as np

# ==========================================
# 1. DOMAIN & GRID CONFIGURATION
# ==========================================
L_VISIBLE = 10000.0  
L_DOMAIN  = 15000.0  
PAD_WIDTH = (L_DOMAIN - L_VISIBLE) / 2.0
T_MAX     = 2.4  
SOURCE_X  = L_DOMAIN / 2.0 

# Grid for FD, PS, FEM, FVM
NX_VISIBLE = 800  
NX_BASE    = int(NX_VISIBLE * (L_DOMAIN / L_VISIBLE)) 
DX_BASE    = L_DOMAIN / NX_BASE
X_AXIS_FULL = np.linspace(0, L_DOMAIN, NX_BASE)
X_AXIS_VIS  = np.linspace(0, L_VISIBLE, NX_VISIBLE) 

# ==========================================
# 2. PHYSICS & SOURCE
# ==========================================
RHO_CONST = 2500.0
F0        = 5.0   
T0        = 1.5 / F0
AMP_SCALE = 1e6

# Source Spatial Width (Sigma for Gaussian)
SOURCE_WIDTH = 1.5 * DX_BASE

# ==========================================
# 3. GENERIC MATERIAL ZONES
# ==========================================
# Define interfaces (x-coordinates). 
MAT_BOUNDARIES = [3000] 

# Velocities for each zone [Region0, Region1, ...]. 
MAT_VELOCITIES = [3000.0, 4000.0] 

# --- Helper to get material properties at x ---
def get_material(x):
    """
    Returns (rho, c, mu) for input array x based on MAT_ZONES.
    """
    rho = np.ones_like(x) * RHO_CONST
    
    indices = np.searchsorted(MAT_BOUNDARIES, x)
    
    # Map indices to velocities
    vel_map = np.array(MAT_VELOCITIES)
    c = vel_map[indices]
    
    mu = rho * c**2
    return rho, c, mu

def ricker_source(t):
    """
    The canonical Ricker Wavelet function.
    t: Time (scalar or array).
    """
    a = np.pi * F0 * (t - T0)
    return (1 - 2*a**2) * np.exp(-a**2)

def get_source_spatial(x, dx):
    """
    Calculates the normalized spatial distribution of the source.
    Currently a Gaussian centered at SOURCE_X with width SOURCE_WIDTH.
    """
    sigma = SOURCE_WIDTH
    g = np.exp(-(x - SOURCE_X)**2 / (2 * sigma**2))
    # Normalize so integral (sum * dx) is 1.0
    return g / (np.sum(g) * dx)

# ==========================================
# 4. SOLVER SETTINGS (CENTRALIZED)
# ==========================================
# Polynomial Order for DG/SEM
POLY_ORDER = 4 

# Grid/Element Resolution
# DG Elements
N_ELEMENTS_DG = 200 
# SEM Elements
N_ELEMENTS_SEM = 200

# Sponge / Boundary Damping Strength
SPONGE_STRENGTH_FD = 30.0
SPONGE_STRENGTH_SEM = 60.0

# Pseudo-Spectral Padding
PAD_PS = 400

# Stability: Master DT for non-element methods
# Using max velocity reference for CFL safety
dt_global = 0.5 * DX_BASE / max(MAT_VELOCITIES)
nt_global = int(T_MAX / dt_global)