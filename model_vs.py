"""
model_vs.py — SIREN-based velocity-stress network

Change from 1.81% run:
  Spatial domain decomposition for sig_net (stress only).

Why:
  Stress L2 is 7.05% vs velocity 1.81%. The gap is structural, not a training issue.
  sigma = mu(x) * dv/dx where mu(x) jumps at x=0.5 (material interface).
  This creates a kink (slope discontinuity) in sigma(x) at every time step.
  A single smooth SIREN network cannot represent this without systematic error —
  the interface-dense snapshot experiment confirmed this (7.24% -> 7.05%, negligible).

Fix:
  Two separate sig_nets split at x_nd = 0.5:
    sig_net_left:  trained on x_nd in [0, 0.5]  — homogeneous medium, smooth stress
    sig_net_right: trained on x_nd in [0.5, 1]  — heterogeneous medium, smooth stress
  Each network only represents a smooth function in its own domain.
  The material discontinuity becomes a natural domain boundary.
  vel_net stays as a single network — velocity IS smooth across the interface.

vs_ansatz routes each (x, t) point to the correct sub-network based on x_nd < 0.5.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np

from fdm_reference import (RHO, XMAX, C_REF, T_CHAR, T_END_ND,
                            V_SCALE, S_SCALE, L, C_MAX, ALPHA, L_HET,
                            T_END, CV_CHECK)

CV = CV_CHECK   # ≈ 1.0 by construction

def CS_of_x_nd(x_nd):
    x_phys = x_nd * L
    c = jnp.where(x_phys > L/2,
                  2500.0 * (1.0 + ALPHA * jnp.sin(2*jnp.pi*(x_phys - L/2) / L_HET)),
                  2500.0)
    mu = RHO * c**2
    return mu * V_SCALE * T_CHAR / (S_SCALE * L)


def siren_init(key, shape, dtype=jnp.float32):
    n_in = shape[0]
    limit = np.sqrt(6.0 / n_in)
    return jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)


class SirenMLP(nn.Module):
    features: tuple
    lb: tuple = (0.0, 0.0)
    ub: tuple = (1.0, 1.0)

    @nn.compact
    def __call__(self, x):
        lb = jnp.array(self.lb, dtype=x.dtype)
        ub = jnp.array(self.ub, dtype=x.dtype)
        H = 2.0 * (x - lb) / (ub - lb) - 1.0

        for feat in self.features[:-1]:
            H = nn.Dense(feat, kernel_init=siren_init,
                         bias_init=nn.initializers.zeros)(H)
            H = jnp.sin(H)

        out = nn.Dense(self.features[-1], kernel_init=siren_init,
                       bias_init=nn.initializers.zeros)(H)
        return out


class VSNet(nn.Module):
    """
    vel_net:       single network, x_nd in [0, 1]   — velocity is smooth everywhere
    sig_net_left:  x_nd in [0, 0.5]  — homogeneous medium, smooth stress
    sig_net_right: x_nd in [0.5, 1]  — heterogeneous medium, smooth stress

    At inference: x_nd < 0.5 → sig_net_left, else → sig_net_right.
    Transition at x_nd = 0.5 is handled via jnp.where (smooth in autodiff).
    """
    t_max_nd: float = 1.0

    @nn.compact
    def __call__(self, x):
        ub_t = self.t_max_nd

        # Velocity: single smooth SIREN over full domain
        v_raw = SirenMLP(features=(16, 80, 80, 80, 1),
                         lb=(0.0, 0.0),
                         ub=(1.0, float(ub_t)),
                         name="vel_net")(x)[0]

        # Stress left: x_nd in [0, 0.5] — homogeneous
        s_left = SirenMLP(features=(16, 80, 80, 80, 1),
                          lb=(0.0, 0.0),
                          ub=(0.5, float(ub_t)),
                          name="sig_net_left")(x)[0]

        # Stress right: x_nd in [0.5, 1] — heterogeneous
        s_right = SirenMLP(features=(16, 80, 80, 80, 1),
                           lb=(0.5, 0.0),
                           ub=(1.0, float(ub_t)),
                           name="sig_net_right")(x)[0]

        # Route by spatial position — x is [x_nd, t_nd]
        x_nd = x[0]
        s_raw = jnp.where(x_nd < 0.5, s_left, s_right)

        return v_raw, s_raw


def vs_ansatz(v_raw, s_raw, x_nd, t_nd):
    return v_raw, s_raw


def create_vs_model(key=None, t_max_nd=None):
    if key is None:
        key = jax.random.PRNGKey(42)
    if t_max_nd is None:
        t_max_nd = T_END_ND

    model = VSNet(t_max_nd=float(t_max_nd))
    params = model.init(key, jnp.ones((2,)))
    n = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    print(f"VSNet (SIREN, split-sig) | params={n:,}  t_max_nd={t_max_nd:.4f}")
    return model, params


if __name__ == "__main__":
    model, params = create_vs_model()
    print(f"\n-- CV/CS checks --")
    print(f"  CV = {CV:.4f}")
    for xv in [0.1, 0.3, 0.49, 0.5, 0.51, 0.7, 0.9]:
        cs = float(CS_of_x_nd(jnp.float32(xv)))
        print(f"  CS(x={xv}) = {cs:.4f}")

    print(f"\n-- Split-stress inference check --")
    for xv in [0.3, 0.49, 0.5, 0.51, 0.7]:
        xt = jnp.array([xv, 0.1])
        vr, sr = model.apply(params, xt)
        print(f"  x={xv}: v={float(vr):.4f}  s={float(sr):.4f}")