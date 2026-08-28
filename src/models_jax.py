"""
JAX/Equinox equivalents of every architecture in models.py.

All models follow a single-sample interface: __call__(xt) where xt: (2,) → (1,).
Use jax.vmap(model)(xt_batch) for batched evaluation, or _call_model_jax() which
does this automatically from (N,1)/(N,1) x,t tensors.

Backend switch controlled by run_experiment.py --backend jax flag.
"""
import math
import jax
import jax.numpy as jnp
import equinox as eqx


# ── Init helpers ─────────────────────────────────────────────────────────────

def _xavier_normal(key, shape):
    fan_in, fan_out = shape[1], shape[0]
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return jax.random.normal(key, shape) * std

def _glorot_uniform(key, shape):
    fan_in, fan_out = shape[1], shape[0]
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-bound, maxval=bound)

def _kaiming_uniform(key, shape, a=math.sqrt(5)):
    fan_in = shape[1]
    gain  = math.sqrt(2.0 / (1 + a ** 2))
    bound = math.sqrt(3.0) * gain / math.sqrt(fan_in)
    return jax.random.uniform(key, shape, minval=-bound, maxval=bound)


# ── Utility ───────────────────────────────────────────────────────────────────

def _call_model_jax(model, x, t):
    """
    x, t: (N, 1) JAX arrays.
    All JAX models take (2,) single-sample; vmap gives (N, 1) output.
    """
    xt = jnp.concatenate([x, t], axis=-1)          # (N, 2)
    return jax.vmap(model)(xt)                       # (N, 1)


# ── 1. VanillaPINN ────────────────────────────────────────────────────────────

class VanillaPINN(eqx.Module):
    layers: list

    def __init__(self, in_dim: int = 2, out_dim: int = 1,
                 hidden_layers: int = 7, hidden_units: int = 128, *, key):
        dims = [in_dim] + [hidden_units] * hidden_layers + [out_dim]
        keys = jax.random.split(key, len(dims) - 1)
        layers = []
        for i, k in enumerate(keys):
            lin = eqx.nn.Linear(dims[i], dims[i + 1], key=k)
            w   = _xavier_normal(k, (dims[i + 1], dims[i]))
            b   = jnp.zeros(dims[i + 1])
            lin = eqx.tree_at(lambda l: l.weight, lin, w)
            lin = eqx.tree_at(lambda l: l.bias,   lin, b)
            layers.append(lin)
        self.layers = layers

    def __call__(self, xt):          # xt: (2,)  →  (1,)
        x = xt
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


# ── 2. FourierFeaturePINN ─────────────────────────────────────────────────────

class FourierFeaturePINN(eqx.Module):
    B:      jax.Array          # (n_fourier, 2), trainable
    hidden: list
    out:    eqx.nn.Linear

    def __init__(self, hidden_layers: int = 7, hidden_units: int = 128,
                 n_fourier: int = 128, sigma: float = 1.0, *, key):
        k_B, k_out, *ks = jax.random.split(key, hidden_layers + 2)
        self.B = jax.random.normal(k_B, (n_fourier, 2)) * sigma

        in_f = 2 * n_fourier
        dims = [in_f] + [hidden_units] * hidden_layers
        hidden = []
        for i, k in enumerate(ks):
            lin = eqx.nn.Linear(dims[i], dims[i + 1], key=k)
            w   = _xavier_normal(k, (dims[i + 1], dims[i]))
            b   = jnp.zeros(dims[i + 1])
            lin = eqx.tree_at(lambda l: l.weight, lin, w)
            lin = eqx.tree_at(lambda l: l.bias,   lin, b)
            hidden.append(lin)
        self.hidden = hidden

        out = eqx.nn.Linear(hidden_units, 1, key=k_out)
        self.out = eqx.tree_at(lambda l: l.weight, out,
                               _xavier_normal(k_out, (1, hidden_units)))

    def _embed(self, xt):             # xt: (2,)  → (2*n_fourier,)
        proj = self.B @ xt
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)])

    def features(self, xt):           # → (hidden_units,)
        x = self._embed(xt)
        for layer in self.hidden:
            x = jnp.tanh(layer(x))
        return x

    def __call__(self, xt):           # xt: (2,)  → (1,)
        return self.out(self.features(xt))


# ── 3. PirateNet ──────────────────────────────────────────────────────────────

class RWFLinear(eqx.Module):
    V:    jax.Array   # (out, in)
    s:    jax.Array   # (out,)
    bias: jax.Array   # (out,)

    def __init__(self, in_f: int, out_f: int,
                 rwf_mu: float = 1.0, rwf_sigma: float = 0.1, *, key):
        k1, k2 = jax.random.split(key)
        self.V    = _glorot_uniform(k1, (out_f, in_f))
        self.s    = rwf_mu + rwf_sigma * jax.random.normal(k2, (out_f,))
        self.bias = jnp.zeros(out_f)

    def __call__(self, x):            # x: (in,)  → (out,)
        W_eff = self.s[:, None] * self.V
        return W_eff @ x + self.bias


class PirateBlock(eqx.Module):
    W1:    RWFLinear
    W2:    RWFLinear
    W3:    RWFLinear
    alpha: jax.Array  # (1,), init 0

    def __init__(self, units: int, rwf_mu: float = 1.0,
                 rwf_sigma: float = 0.1, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.W1    = RWFLinear(units, units, rwf_mu, rwf_sigma, key=k1)
        self.W2    = RWFLinear(units, units, rwf_mu, rwf_sigma, key=k2)
        self.W3    = RWFLinear(units, units, rwf_mu, rwf_sigma, key=k3)
        self.alpha = jnp.zeros(1)

    def __call__(self, x, U, V):      # x, U, V: (units,)
        f  = jnp.tanh(self.W1(x))
        z1 = f * U + (1.0 - f) * V
        g  = jnp.tanh(self.W2(z1))
        z2 = g * U + (1.0 - g) * V
        h  = jnp.tanh(self.W3(z2))
        return self.alpha[0] * h + (1.0 - self.alpha[0]) * x


class PirateNet(eqx.Module):
    B:         jax.Array        # (n_fourier, in_dim)
    enc_U:     RWFLinear
    enc_V:     RWFLinear
    proj:      RWFLinear
    blocks:    list
    out_layer: eqx.nn.Linear    # (1, units), no bias

    def __init__(self, n_blocks: int = 3, units: int = 256,
                 n_fourier: int = 128, sigma: float = 2.0, in_dim: int = 2,
                 rwf_mu: float = 1.0, rwf_sigma: float = 0.1, *, key):
        k0, k_U, k_V, k_p, k_out, *kb = jax.random.split(key, n_blocks + 5)
        embed_dim = 2 * n_fourier

        self.B     = jax.random.normal(k0, (n_fourier, in_dim)) * sigma
        self.enc_U = RWFLinear(embed_dim, units, rwf_mu, rwf_sigma, key=k_U)
        self.enc_V = RWFLinear(embed_dim, units, rwf_mu, rwf_sigma, key=k_V)
        self.proj  = RWFLinear(embed_dim, units, rwf_mu, rwf_sigma, key=k_p)
        self.blocks = [PirateBlock(units, rwf_mu, rwf_sigma, key=kb[i])
                       for i in range(n_blocks)]

        out = eqx.nn.Linear(units, 1, use_bias=False, key=k_out)
        self.out_layer = eqx.tree_at(lambda l: l.weight, out, jnp.zeros((1, units)))

    def _embed(self, xt):             # xt: (in_dim,)  → (2*n_fourier,)
        proj = self.B @ xt
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)])

    def features(self, xt):           # → (units,)
        phi = self._embed(xt)
        U   = jnp.tanh(self.enc_U(phi))
        V   = jnp.tanh(self.enc_V(phi))
        h   = jnp.tanh(self.proj(phi))
        for block in self.blocks:
            h = block(h, U, V)
        return h

    def __call__(self, xt):           # xt: (2,)  → (1,)
        return self.out_layer(self.features(xt))


# ── 4. WavKAN ────────────────────────────────────────────────────────────────

class WavKANLinear(eqx.Module):
    scale:           jax.Array   # (out, in)
    translation:     jax.Array   # (out, in)
    weight1:         jax.Array   # (out, in)
    wavelet_weights: jax.Array   # (out, in)
    wavelet_type:    str  = eqx.field(static=True)
    use_base:        bool = eqx.field(static=True)

    def __init__(self, in_f: int, out_f: int,
                 wavelet_type: str = 'mexican_hat', use_base: bool = True, *, key):
        k1, k2 = jax.random.split(key)
        bound = 1.0 / math.sqrt(in_f)
        self.scale           = jnp.ones((out_f, in_f))
        self.translation     = jnp.zeros((out_f, in_f))
        self.weight1         = _kaiming_uniform(k1, (out_f, in_f))
        self.wavelet_weights = _kaiming_uniform(k2, (out_f, in_f))
        self.wavelet_type    = wavelet_type
        self.use_base        = use_base

    def __call__(self, x):            # x: (in,)  → (out,)
        # x_scaled: (out, in)
        xs = (x[None, :] - self.translation) / self.scale

        if self.wavelet_type == 'mexican_hat':
            t2 = jnp.exp(-0.5 * xs ** 2)
            w  = (2.0 / (math.sqrt(3) * math.pi ** 0.25)) * (xs ** 2 - 1.0) * t2
        elif self.wavelet_type == 'morlet':
            w = jnp.exp(-0.5 * xs ** 2) * jnp.cos(5.0 * xs)
        elif self.wavelet_type == 'dog':
            w = -xs * jnp.exp(-0.5 * xs ** 2)
        elif self.wavelet_type == 'meyer':
            v = jnp.abs(xs)
            def nu(t):
                return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)
            aux = jnp.where(v <= 0.5, 1.0,
                  jnp.where(v >= 1.0, 0.0,
                  jnp.cos(math.pi / 2.0 * nu(2 * v - 1))))
            w = jnp.sin(math.pi * v) * aux
        elif self.wavelet_type == 'shannon':
            w = jnp.sinc(xs / math.pi) * jnp.exp(-0.5 * xs ** 2)
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

        out = (w * self.wavelet_weights).sum(axis=1)   # (out,)
        if self.use_base:
            out = out + self.weight1 @ x
        return out


class WavKAN(eqx.Module):
    layers: list

    def __init__(self, layers_hidden, wavelet_type='mexican_hat',
                 use_base=True, *, key):
        pairs = list(zip(layers_hidden[:-1], layers_hidden[1:]))
        keys  = jax.random.split(key, len(pairs))
        self.layers = [WavKANLinear(in_f, out_f, wavelet_type, use_base, key=k)
                       for (in_f, out_f), k in zip(pairs, keys)]

    def __call__(self, xt):           # xt: (2,)  → (1,)
        x = xt
        for layer in self.layers:
            x = layer(x)
        return x


# ── 4b. ChebyshevKAN (cPIKAN basis) ──────────────────────────────────────────

class ChebyshevKANLinear(eqx.Module):
    coeffs: jax.Array            # (out, in, degree+1)
    degree: int = eqx.field(static=True)

    def __init__(self, in_f: int, out_f: int, degree: int = 5, *, key):
        self.degree = degree
        std = 1.0 / math.sqrt(in_f * (degree + 1))
        self.coeffs = jax.random.normal(key, (out_f, in_f, degree + 1)) * std

    def __call__(self, x):       # x: (in,)  → (out,)
        x = jnp.tanh(x)          # Chebyshev domain [-1, 1]
        T = [jnp.ones_like(x), x]
        for _ in range(2, self.degree + 1):
            T.append(2.0 * x * T[-1] - T[-2])
        T = jnp.stack(T[: self.degree + 1], axis=-1)      # (in, D+1)
        return jnp.einsum('id,oid->o', T, self.coeffs)


class ChebyshevKAN(eqx.Module):
    layers: list

    def __init__(self, layers_hidden, degree: int = 5, *, key):
        pairs = list(zip(layers_hidden[:-1], layers_hidden[1:]))
        keys  = jax.random.split(key, len(pairs))
        self.layers = [ChebyshevKANLinear(i, o, degree, key=k)
                       for (i, o), k in zip(pairs, keys)]

    def __call__(self, xt):      # xt: (2,)  → (1,)
        x = xt
        for layer in self.layers:
            x = layer(x)
        return x


# ── 4c. FourierWavKAN: random Fourier embedding → WavKAN trunk ───────────────

class FourierWavKAN(eqx.Module):
    B:   jax.Array               # (n_fourier, 2), trainable
    kan: WavKAN

    def __init__(self, hidden_layers: int = 2, hidden_units: int = 16,
                 n_fourier: int = 32, sigma: float = 3.0,
                 wavelet_type: str = 'mexican_hat', use_base: bool = True, *, key):
        k_B, k_kan = jax.random.split(key)
        self.B = jax.random.normal(k_B, (n_fourier, 2)) * sigma
        width = [2 * n_fourier] + [hidden_units] * hidden_layers + [1]
        self.kan = WavKAN(width, wavelet_type=wavelet_type,
                          use_base=use_base, key=k_kan)

    def __call__(self, xt):      # xt: (2,)  → (1,)
        proj = self.B @ xt
        phi  = jnp.concatenate([jnp.cos(proj), jnp.sin(proj)])
        return self.kan(phi)


# ── 5. KAN: B-spline Kolmogorov-Arnold network ─────────────────────────────────
# Same single-sample (2,) → (1,) interface as every other model here.

class SplineKANLinear(eqx.Module):
    """One B-spline KAN edge layer: a SiLU base branch plus a learned spline.

    Mirrors pykan's `KANLayer` (grid size `G`, spline order `k`) closely enough
    that the JAX and PyTorch backends run the same architecture. The grid is a
    fixed uniform extension of [-1, 1]; pykan's adaptive grid refinement is not
    reproduced, because it would make the parameter set change shape mid-run and
    neither the causal schedule nor L-BFGS tolerates that.
    """
    base_weight:   jax.Array     # (out, in)
    spline_weight: jax.Array     # (out, in, G + k)
    grid:          jax.Array = eqx.field(static=False)   # (G + 2k + 1,)
    k:             int = eqx.field(static=True)
    n_basis:       int = eqx.field(static=True)

    def __init__(self, in_f: int, out_f: int, G: int = 5, k: int = 3,
                 grid_range=(-1.0, 1.0), *, key):
        self.k = k
        self.n_basis = G + k
        h = (grid_range[1] - grid_range[0]) / G
        # k knots either side so every basis function on [-1, 1] is complete.
        self.grid = jnp.arange(-k, G + k + 1, dtype=jnp.float32) * h + grid_range[0]

        k_base, k_spline = jax.random.split(key)
        self.base_weight = (jax.random.normal(k_base, (out_f, in_f))
                            / math.sqrt(in_f))
        self.spline_weight = (jax.random.normal(k_spline, (out_f, in_f, self.n_basis))
                              * 0.1 / math.sqrt(in_f))

    def _bases(self, x):
        """Cox--de Boor recursion. x: (in,) -> (in, n_basis)."""
        g = self.grid
        # order 0: indicator of each knot interval
        b = ((x[:, None] >= g[None, :-1]) & (x[:, None] < g[None, 1:])).astype(x.dtype)
        for order in range(1, self.k + 1):
            left_den = g[order:-1] - g[: -order - 1]
            right_den = g[order + 1:] - g[1:-order]
            left = (x[:, None] - g[None, : -order - 1]) / left_den[None, :] * b[:, :-1]
            right = (g[None, order + 1:] - x[:, None]) / right_den[None, :] * b[:, 1:]
            b = left + right
        return b

    def __call__(self, x):            # x: (in,)  -> (out,)
        # The spline grid is finite, so squash first: an unbounded coordinate
        # would fall outside every basis function and contribute nothing.
        xs = jnp.tanh(x)
        bases = self._bases(xs)                                   # (in, n_basis)
        spline = jnp.einsum('ib,oib->o', bases, self.spline_weight)
        base = self.base_weight @ (xs * jax.nn.sigmoid(xs))        # SiLU branch
        return base + spline


class SplineKAN(eqx.Module):
    layers: list

    def __init__(self, layers_hidden, G: int = 5, k: int = 3, *, key):
        pairs = list(zip(layers_hidden[:-1], layers_hidden[1:]))
        keys = jax.random.split(key, len(pairs))
        self.layers = [SplineKANLinear(i, o, G=G, k=k, key=kk)
                       for (i, o), kk in zip(pairs, keys)]

    def __call__(self, xt):           # xt: (2,)  -> (1,)
        x = xt
        for layer in self.layers:
            x = layer(x)
        return x


def build_kan_jax(n_inputs, n_outputs, n_hidden, hidden_width, G, k, *, key):
    """Spline KAN for the JAX backend.

    This used to wrap the `jaxkan` package. That package's `KAN` is a
    `flax.nnx` module whose constructor and parameter handling do not fit the
    equinox/optax loop the rest of this backend uses, and its signature has
    changed across releases. The layer is ~60 lines, so it is implemented here
    instead: one fewer dependency, and the architecture no longer silently
    changes when the library does.
    """
    width = [n_inputs] + [hidden_width] * n_hidden + [n_outputs]
    return SplineKAN(width, G=G, k=k, key=key)


# ── Builder helpers ───────────────────────────────────────────────────────────

def build_mlp_jax(n_inputs, n_outputs, n_hidden, hidden_width, *, key):
    return VanillaPINN(in_dim=n_inputs, out_dim=n_outputs,
                       hidden_layers=n_hidden, hidden_units=hidden_width, key=key)


# ── Physics-informed output-layer initialisation ──────────────────────────────

def physics_informed_init_jax(model, xt_data, y_data):
    """
    Fit the output layer to the IC via least-squares (mirrors PyTorch PI-init).

    Args:
        model:   FourierFeaturePINN or PirateNet (must have a .features() method)
        xt_data: (N, 2) JAX array of [x, t] pairs
        y_data:  (N, 1) JAX array of u_0(x) target values
    Returns:
        New model with output layer weights overwritten.
    """
    xt_data = jnp.asarray(xt_data)
    y_data  = jnp.asarray(y_data)

    Phi = jax.vmap(model.features)(xt_data)          # (N, units)
    W_star = jnp.linalg.lstsq(Phi, y_data, rcond=None)[0]   # (units, 1)
    new_w  = W_star.T                                 # (1, units)

    if hasattr(model, 'out_layer'):
        return eqx.tree_at(lambda m: m.out_layer.weight, model, new_w)
    elif hasattr(model, 'out'):
        new_b = jnp.zeros(1)
        m2 = eqx.tree_at(lambda m: m.out.weight, model, new_w)
        return eqx.tree_at(lambda m: m.out.bias, m2, new_b)
    else:
        raise AttributeError("Model has no recognised output-layer attribute")
