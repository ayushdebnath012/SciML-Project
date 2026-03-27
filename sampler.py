"""
sampler.py — Collocation + snapshot samplers

Back to clean uniform sampling (interface_only flag removed from external call).
The interface_only capability is kept internally for future use but train_vs.py
no longer calls it — the interface bias it introduced hurt velocity by 0.66%.
"""

import numpy as np
import jax.numpy as jnp

from fdm_reference import (XMAX, T_CHAR, V_SCALE, S_SCALE, T_END_ND, L)

_T_SRC_ND = 0.3 / T_CHAR
_T_SRC_HALF_WIDTH = 0.04 / T_CHAR * 3


def lhs_samples(n_points, t_min_nd, t_max_nd, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.permutation(n_points) + rng.random(n_points)) / n_points
    t = (rng.permutation(n_points) + rng.random(n_points)) / n_points
    t = t_min_nd + t * (t_max_nd - t_min_nd)
    return np.stack([x, t], axis=1).astype(np.float32)


def interface_samples(n, t_min_nd, t_max_nd, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.45, 0.55, size=n)
    t = rng.uniform(t_min_nd, t_max_nd, size=n)
    return np.stack([x, t], axis=1).astype(np.float32)


def source_samples(n, t_min_nd, t_max_nd, seed=2):
    rng   = np.random.default_rng(seed)
    x     = rng.normal(0.5,       0.01, size=n)
    t     = rng.normal(_T_SRC_ND, 0.04, size=n)
    x     = np.clip(x, 0.0, 1.0)
    t     = np.clip(t, t_min_nd, t_max_nd)
    return np.stack([x, t], axis=1).astype(np.float32)


def _source_overlaps_window(t_min_nd, t_max_nd):
    src_end = _T_SRC_ND + _T_SRC_HALF_WIDTH
    return src_end >= t_min_nd


def make_collocation(n_bulk=20_000, n_interface=2_000, n_source=2_000,
                     t_min_nd=0.0, t_max_nd=1.0, seed=42):
    bulk = lhs_samples(n_bulk,      t_min_nd, t_max_nd, seed=seed)
    intf = interface_samples(n_interface, t_min_nd, t_max_nd, seed=seed+1)

    if _source_overlaps_window(t_min_nd, t_max_nd):
        src = source_samples(n_source, t_min_nd, t_max_nd, seed=seed+2)
    else:
        src = lhs_samples(n_source, t_min_nd, t_max_nd, seed=seed+3)

    xt  = np.concatenate([bulk, intf, src], axis=0)
    rng = np.random.default_rng(seed)
    rng.shuffle(xt)
    return xt


def load_snapshot_data(fdm_path, t_min_nd, t_max_nd,
                       n_snaps_per_window=15, n_pts_per_snap=800,
                       seed=99):
    """
    Load FDM snapshots — uniform spatial sampling across full domain.
    Always includes snapshot closest to t_min as IC anchor.
    """
    data    = np.load(fdm_path)
    t_snaps = data["t_snaps"]
    x_v     = data["x_v"]
    v_snaps = data["v_snaps"]
    s_snaps = data["s_snaps"]

    t_min_phys = t_min_nd * T_CHAR
    t_max_phys = t_max_nd * T_CHAR

    mask    = (t_snaps >= t_min_phys - 1e-6) & (t_snaps <= t_max_phys + 1e-6)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None, None

    t_min_idx_in_mask = np.argmin(np.abs(t_snaps[indices] - t_min_phys))
    exact_t_min_idx   = indices[t_min_idx_in_mask]
    remaining_idx     = np.linspace(0, len(indices)-1, n_snaps_per_window, dtype=int)
    remaining         = indices[remaining_idx]
    chosen            = np.unique(np.concatenate([[exact_t_min_idx], remaining]))[:n_snaps_per_window]

    rng = np.random.default_rng(seed)
    rows_v, rows_s = [], []

    nx = len(x_v)
    for idx in chosen:
        t_phys = t_snaps[idx]
        t_nd   = t_phys / T_CHAR

        n_draw = min(n_pts_per_snap, nx)
        pts    = rng.choice(nx, size=n_draw, replace=False)

        x_nd  = (x_v[pts] / XMAX).astype(np.float32)
        v_nd  = (v_snaps[idx, pts] / V_SCALE).astype(np.float32)
        s_nd  = (s_snaps[idx, pts] / S_SCALE).astype(np.float32)
        t_arr = np.full(n_draw, t_nd, dtype=np.float32)

        rows_v.append(np.stack([x_nd, t_arr, v_nd], axis=1))
        rows_s.append(np.stack([x_nd, t_arr, s_nd], axis=1))

    snap_v = np.concatenate(rows_v, axis=0)
    snap_s = np.concatenate(rows_s, axis=0)
    return snap_v, snap_s


def make_windows(n_windows=4):
    edges = np.linspace(0, T_END_ND, n_windows + 1)
    return [(float(edges[i]), float(edges[i+1])) for i in range(n_windows)]


def to_jax(arr):
    return jnp.array(arr)