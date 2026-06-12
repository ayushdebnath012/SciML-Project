"""
SOAP optimizer (JAX / optax GradientTransformation).

Mirrors src/optimizers.py — see that file for the algorithm description
(Vyas et al. 2024, arXiv:2409.11321; PINN results in arXiv:2502.00604).

Implementation notes:
  - Works on any pytree of arrays (e.g. eqx.filter(model, eqx.is_array)).
  - 2D leaves get the full rotated-Adam update; other leaves get plain Adam.
  - Eigenbasis refresh happens every `precondition_frequency` steps via
    jax.lax.cond, so the eigh only runs when scheduled (JIT-safe).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


class SoapState(NamedTuple):
    count: jax.Array   # scalar int32
    m:     object      # pytree like params — first moment (original space)
    v:     object      # pytree like params — second moment (rotated space for 2D)
    L:     object      # pytree — left Shampoo stats  (2D leaves) / dummy zeros
    R:     object      # pytree — right Shampoo stats (2D leaves) / dummy zeros
    QL:    object      # pytree — left eigenbasis
    QR:    object      # pytree — right eigenbasis


class _Leaf(NamedTuple):
    update: jax.Array
    m:      jax.Array
    v:      jax.Array
    L:      jax.Array
    R:      jax.Array
    QL:     jax.Array
    QR:     jax.Array


def _is_leaf_result(x):
    return isinstance(x, _Leaf)


def soap(learning_rate: float = 1e-3, b1: float = 0.95, b2: float = 0.95,
         shampoo_beta: float = 0.95, eps: float = 1e-8,
         precondition_frequency: int = 10) -> optax.GradientTransformation:

    def init_fn(params):
        def _zeros_mat(p):
            if p.ndim == 2:
                return jnp.zeros((p.shape[0], p.shape[0]), p.dtype)
            return jnp.zeros((), p.dtype)

        def _zeros_mat_r(p):
            if p.ndim == 2:
                return jnp.zeros((p.shape[1], p.shape[1]), p.dtype)
            return jnp.zeros((), p.dtype)

        return SoapState(
            count=jnp.zeros([], jnp.int32),
            m=jax.tree_util.tree_map(jnp.zeros_like, params),
            v=jax.tree_util.tree_map(jnp.zeros_like, params),
            L=jax.tree_util.tree_map(_zeros_mat, params),
            R=jax.tree_util.tree_map(_zeros_mat_r, params),
            QL=jax.tree_util.tree_map(_zeros_mat, params),
            QR=jax.tree_util.tree_map(_zeros_mat_r, params),
        )

    def update_fn(grads, state, params=None):
        del params
        count   = state.count + 1
        t       = count.astype(jnp.float32)
        bc1     = 1.0 - b1 ** t
        bc2     = 1.0 - b2 ** t
        refresh = (state.count % precondition_frequency) == 0   # incl. step 1

        def leaf_update(g, m, v, L, R, QL, QR):
            if g is None:
                return None

            new_m = b1 * m + (1 - b1) * g

            if g.ndim != 2:
                # Plain Adam
                new_v = b2 * v + (1 - b2) * g ** 2
                upd   = -learning_rate * (new_m / bc1) / (jnp.sqrt(new_v / bc2) + eps)
                return _Leaf(upd, new_m, new_v, L, R, QL, QR)

            # SOAP path for matrices
            new_L = shampoo_beta * L + (1 - shampoo_beta) * (g @ g.T)
            new_R = shampoo_beta * R + (1 - shampoo_beta) * (g.T @ g)

            def _refresh(_):
                eyeL = jnp.eye(new_L.shape[0], dtype=g.dtype)
                eyeR = jnp.eye(new_R.shape[0], dtype=g.dtype)
                qL = jnp.linalg.eigh(new_L + eps * eyeL)[1]
                qR = jnp.linalg.eigh(new_R + eps * eyeR)[1]
                return qL, qR

            def _keep(_):
                return QL, QR

            new_QL, new_QR = jax.lax.cond(refresh, _refresh, _keep, operand=None)

            g_rot = new_QL.T @ g @ new_QR
            m_rot = new_QL.T @ new_m @ new_QR
            new_v = b2 * v + (1 - b2) * g_rot ** 2

            n_rot = (m_rot / bc1) / (jnp.sqrt(new_v / bc2) + eps)
            upd   = -learning_rate * (new_QL @ n_rot @ new_QR.T)
            return _Leaf(upd, new_m, new_v, new_L, new_R, new_QL, new_QR)

        results = jax.tree_util.tree_map(
            leaf_update, grads, state.m, state.v, state.L, state.R,
            state.QL, state.QR,
        )

        def _pick(field):
            return jax.tree_util.tree_map(
                lambda r: getattr(r, field), results, is_leaf=_is_leaf_result
            )

        updates   = _pick("update")
        new_state = SoapState(
            count=count,
            m=_pick("m"), v=_pick("v"),
            L=_pick("L"), R=_pick("R"),
            QL=_pick("QL"), QR=_pick("QR"),
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
