import numpy as np
from physics import *
from tqdm import tqdm

def solve_fd(NX_BASE, nt_global, dt_global, DX_BASE, damp_mask, X_AXIS_FULL, X_AXIS_VIS):
    rho_fd, c_fd, mu_fd = get_material(X_AXIS_FULL)
    v_fd = np.zeros(NX_BASE)
    sigma_fd = np.zeros(NX_BASE - 1)
    
    src_idx_fd = int(SOURCE_X / DX_BASE)
    if src_idx_fd >= NX_BASE: src_idx_fd = NX_BASE - 1
    
    snap_fd, t_snaps = [], []
    for it in tqdm(range(nt_global), desc="FD"):
        t = it * dt_global
        v_fd[1:-1] += (dt_global / rho_fd[1:-1]) * (sigma_fd[1:] - sigma_fd[:-1]) / DX_BASE
        v_fd *= np.exp(-damp_mask * dt_global) 
        v_fd[src_idx_fd] += (dt_global / rho_fd[src_idx_fd]) * (AMP_SCALE / DX_BASE) * ricker_source(t)
        mu_half = 0.5 * (mu_fd[1:] + mu_fd[:-1])
        sigma_fd += mu_half * (dt_global / DX_BASE) * (v_fd[1:] - v_fd[:-1])
        if it % 10 == 0: 
            snap_fd.append(v_fd.copy())
            t_snaps.append(t)

    return X_AXIS_VIS, np.array(snap_fd), np.array(t_snaps)

def solve_ps(NX_BASE, nt_global, dt_global, DX_BASE, X_AXIS_VIS):
    pad_ps = PAD_PS
    nx_ps = NX_BASE + 2 * pad_ps
    x_ps_full = (np.arange(nx_ps) - pad_ps) * DX_BASE
    rho_ps, c_ps, _ = get_material(x_ps_full)
    k_vec = 2 * np.pi * np.fft.rfftfreq(nx_ps, d=DX_BASE)
    k2_vec = k_vec**2
    u_ps_nm1, u_ps_n = np.zeros(nx_ps), np.zeros(nx_ps)
    snap_ps, t_snaps = [], []

    # Get Spatial Source Distribution from Physics
    sg = get_source_spatial(x_ps_full, DX_BASE)

    for it in tqdm(range(nt_global), desc="PS"):
        t = it * dt_global
        uxx = np.fft.irfft(-k2_vec * np.fft.rfft(u_ps_n))
        force = sg * ricker_source(t) * AMP_SCALE
        u_ps_np1 = 2*u_ps_n - u_ps_nm1 + (dt_global**2) * (c_ps**2 * uxx + force / rho_ps)
        v_ps = (u_ps_np1 - u_ps_nm1) / (2 * dt_global)
        u_ps_nm1, u_ps_n = u_ps_n.copy(), u_ps_np1.copy()
        if it % 10 == 0: 
            snap_ps.append(v_ps[pad_ps : pad_ps+NX_BASE].copy())
            t_snaps.append(t)

    return X_AXIS_VIS, np.array(snap_ps), np.array(t_snaps)

def solve_fem(NX_BASE, nt_global, dt_global, DX_BASE, damp_mask, X_AXIS_FULL, X_AXIS_VIS):
    rho_fe, _, mu_fe = get_material(X_AXIS_FULL)
    M_fe, K_fe = np.zeros(NX_BASE), np.zeros((NX_BASE, NX_BASE))
    for e in range(NX_BASE - 1):
        ke = (0.5 * (mu_fe[e] + mu_fe[e+1])) / DX_BASE
        me = (0.5 * (rho_fe[e] + rho_fe[e+1])) * DX_BASE / 2.0
        K_fe[e, e]+=ke; K_fe[e, e+1]-=ke; K_fe[e+1, e]-=ke; K_fe[e+1, e+1]+=ke
        M_fe[e]+=me; M_fe[e+1]+=me
    
    src_idx = int(SOURCE_X / DX_BASE)
    if src_idx >= NX_BASE: src_idx = NX_BASE - 1

    u_fe, u_fe_old = np.zeros(NX_BASE), np.zeros(NX_BASE)
    snap_fem, t_snaps = [], []
    for it in tqdm(range(nt_global), desc="FEM"):
        t = it * dt_global
        f_v = np.zeros(NX_BASE); f_v[src_idx] = AMP_SCALE * ricker_source(t)
        acc = (-K_fe @ u_fe + f_v) / M_fe
        u_fe_new = u_fe + (u_fe - u_fe_old) * np.exp(-damp_mask * dt_global) + (dt_global**2 * acc)
        v_fe = (u_fe_new - u_fe_old) / (2 * dt_global)
        u_fe_old, u_fe = u_fe.copy(), u_fe_new.copy()
        if it % 10 == 0: 
            snap_fem.append(v_fe.copy())
            t_snaps.append(t)

    return X_AXIS_VIS, np.array(snap_fem), np.array(t_snaps)

def solve_fvm(NX_BASE):
    dx = L_DOMAIN / NX_BASE
    x = np.linspace(dx/2, L_DOMAIN - dx/2, NX_BASE)
    rho, c, mu = get_material(x)
    dt = 0.5 * dx / np.max(c)
    nt = int(T_MAX / dt)
    
    src_idx = int(SOURCE_X / dx)
    if src_idx >= NX_BASE: src_idx = NX_BASE - 1

    Q = np.zeros((2, NX_BASE)); v_snaps, t_snaps = [], []
    stride = max(1, int(0.02 / dt))
    for n in tqdm(range(nt), desc="FVM"):
        t = n * dt
        if n % stride == 0: v_snaps.append(Q[1].copy()); t_snaps.append(t)
        Q[1, src_idx] += (dt / rho[src_idx]) * (AMP_SCALE * ricker_source(t) / dx)
        dQ = (Q[:, 2:] - Q[:, :-2]) / (2*dx); d2Q = (Q[:, 2:] - 2*Q[:, 1:-1] + Q[:, :-2]) / (dx**2)
        for i in range(1, NX_BASE-1):
            A = np.array([[0.0, -mu[i]], [-1.0/rho[i], 0.0]])
            Q[:, i] -= dt * (A @ dQ[:, i-1]) - 0.5 * dt**2 * (A @ A @ d2Q[:, i-1])
        Q[:, 0], Q[:, -1] = 0, 0
    return x, np.array(v_snaps), np.array(t_snaps)

def solve_dg():
    ne = N_ELEMENTS_DG
    N = POLY_ORDER; Np = N + 1; h = L_DOMAIN / ne; J = h / 2.0
    xi, _ = np.polynomial.legendre.leggauss(Np)
    for _ in range(10):
        P = np.polynomial.legendre.Legendre.basis(N)(xi); Pd = np.polynomial.legendre.Legendre.basis(N).deriv()(xi)
        xi -= ((1-xi**2)*Pd) / (-2*xi*Pd + (1-xi**2)*np.polynomial.legendre.Legendre.basis(N).deriv(2)(xi))
    w = 2 / (N * (N + 1) * np.polynomial.legendre.Legendre.basis(N)(xi)**2)
    D = np.zeros((Np, Np)); Pb = np.polynomial.legendre.Legendre.basis(N)
    for i in range(Np):
        for j in range(Np):
            if i != j: D[i,j] = (Pb(xi[i])/Pb(xi[j]))/(xi[i]-xi[j])
            elif i==j==0: D[i,j] = -N*(N+1)/4.0
            elif i==j==N: D[i,j] =  N*(N+1)/4.0
    x_glob = np.zeros((ne, Np)); rho_v, mu_v, Z_v = np.zeros(ne), np.zeros(ne), np.zeros(ne)
    for k in tqdm(range(ne), desc="DG Setup"):
        x_glob[k] = h*(k+0.5) + xi*(h/2); rv, cv, mv = get_material(np.array([h*(k+0.5)]))
        rho_v[k], mu_v[k], Z_v[k] = rv[0], mv[0], rv[0]*cv[0]
    
    src_element = int(SOURCE_X / h)
    if src_element >= ne: src_element = ne - 1
    
    def get_rhs(Q_in, t):
        S, V = Q_in[:, :, 0], Q_in[:, :, 1]; flux = np.zeros((ne, Np, 2))
        k_l, k_r = np.arange(ne) - 1, np.arange(ne) + 1
        sL, vL = S[k_l, N].copy(), V[k_l, N].copy(); sR, vR = S[:, 0].copy(), V[:, 0].copy()
        sL[0], vL[0] = sR[0], vR[0]; ZL, ZR = Z_v[k_l], Z_v
        vs = (ZL*vL + ZR*vR + (sL-sR))/(ZL+ZR); ss = (ZR*sL + ZL*sR + ZL*ZR*(vL-vR))/(ZL+ZR)
        flux[:, 0, 0], flux[:, 0, 1] = mu_v*vs, (1/rho_v)*ss
        sL_r, vL_r = S[:, N].copy(), V[:, N].copy(); sR_r, vR_r = S[k_r%ne, 0].copy(), V[k_r%ne, 0].copy()
        sR_r[-1], vR_r[-1] = sL_r[-1], vL_r[-1]; ZL_r, ZR_r = Z_v, Z_v[k_r%ne]
        vsr = (ZL_r*vL_r + ZR_r*vR_r + (sL_r-sR_r))/(ZL_r+ZR_r); ssr = (ZR_r*sL_r+ZL_r*sR_r+ZL_r*ZR_r*(vL_r-vR_r))/(ZL_r+ZR_r)
        flux[:, N, 0], flux[:, N, 1] = -mu_v*vsr, (-1/rho_v)*ssr
        rhs = np.zeros_like(Q_in)
        for k in range(ne):
            rhs[k,:,0] = (mu_v[k]*(D.T @ (w*V[k])) + flux[k,:,0])/(w*J)
            rhs[k,:,1] = ((1/rho_v[k])*(D.T @ (w*S[k])) + flux[k,:,1])/(w*J)
            if k == src_element: rhs[k, 0, 1] += (AMP_SCALE * ricker_source(t)) / (rho_v[k] * w[0] * J)
        return rhs
    
    dt_dg = 0.1 * (h/2 * np.min(np.diff(xi))) / max(MAT_VELOCITIES); nt = int(T_MAX / dt_dg)
    Q = np.zeros((ne, Np, 2)); v_snaps, t_snaps = [], []
    stride = max(1, int(0.02/dt_dg))
    for n in tqdm(range(nt), desc="DG Solver"):
        if n % stride == 0: v_snaps.append(Q[:, :, 1].flatten().copy()); t_snaps.append(n*dt_dg)
        k1 = get_rhs(Q, n*dt_dg); k2 = get_rhs(Q+0.5*dt_dg*k1, (n+0.5)*dt_dg)
        k3 = get_rhs(Q+0.5*dt_dg*k2, (n+0.5)*dt_dg); k4 = get_rhs(Q+dt_dg*k3, (n+1)*dt_dg)
        Q += (dt_dg/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return x_glob.flatten(), np.array(v_snaps), np.array(t_snaps)

def solve_sem():
    ne, N = N_ELEMENTS_SEM, POLY_ORDER
    ng = ne * N + 1
    xi = -np.cos(np.pi * np.arange(N + 1) / N)
    for _ in range(100):
        def lp(n, x):
            if n == 0: return np.ones_like(x)
            if n == 1: return x
            return ((2*n-1)*x*lp(n-1, x)-(n-1)*lp(n-2, x))/n
        def ld(n, x): return n*(x*lp(n, x)-lp(n-1, x))/(x**2-1+1e-15)
        L_N, L_N_p = lp(N, xi), ld(N, xi)
        L_N_pp = (2*xi*L_N_p - N*(N+1)*L_N)/(1-xi**2+1e-15)
        xi[1:-1] -= L_N_p[1:-1] / L_N_pp[1:-1]
    w = 2 / (N*(N+1)*lp(N, xi)**2)
    D_mat = np.zeros((N+1, N+1)); LN = lp(N, xi)
    for i in range(N+1):
        for j in range(N+1):
            if i != j: D_mat[i, j] = (LN[i] / LN[j]) / (xi[i] - xi[j])
            elif i == 0: D_mat[i, j] = -N * (N + 1) / 4.0
            elif i == N: D_mat[i, j] =  N * (N + 1) / 4.0
    x_glob = np.zeros(ng); el_w = L_DOMAIN / ne
    for e in range(ne): x_glob[e*N : (e+1)*N+1] = (xi + 1)*(el_w/2) + e*el_w
    rho_v, _, mu_v = get_material(x_glob)
    M_global, K_global = np.zeros(ng), np.zeros((ng, ng))
    for e in range(ne):
        idx = slice(e*N, (e+1)*N+1); J = el_w / 2.0
        M_global[idx]+=rho_v[idx]*w*J; K_global[idx, idx]+=D_mat.T@np.diag(w*mu_v[idx])@D_mat*(1.0/J)
    gamma = np.zeros(ng); pad = (L_DOMAIN - L_VISIBLE) / 2.0
    idx_l = x_glob <= pad; gamma[idx_l] = SPONGE_STRENGTH_SEM * ((pad - x_glob[idx_l]) / pad)**2
    idx_r = x_glob >= (L_DOMAIN-pad); gamma[idx_r] = SPONGE_STRENGTH_SEM * ((x_glob[idx_r] - (L_DOMAIN-pad)) / pad)**2
    
    dt_sem = 0.5 * np.min(np.diff(x_glob)) / max(MAT_VELOCITIES); nt = int(T_MAX / dt_sem)
    
    src_idx = np.abs(x_glob - SOURCE_X).argmin()

    u, u_old = np.zeros(ng), np.zeros(ng)
    v_snaps, t_snaps = [], []; stride = max(1, int(0.02 / dt_sem))
    for it in tqdm(range(nt), desc="SEM"):
        if it % stride == 0: v_snaps.append((u - u_old)/dt_sem); t_snaps.append(it*dt_sem)
        f_v = np.zeros(ng); f_v[src_idx] = AMP_SCALE * ricker_source(it*dt_sem)
        accel = (1.0/M_global) * (f_v - K_global @ u)
        u_new = (2*u - u_old + (dt_sem**2)*accel + gamma*dt_sem*u_old) / (1 + gamma*dt_sem)
        u_old, u = u.copy(), u_new.copy()
    return x_glob, np.array(v_snaps), np.array(t_snaps)