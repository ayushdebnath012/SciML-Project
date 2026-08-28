"""Pure-NumPy E(x), rho(x) for the three canonical materials.

`wave/materials.py` imports torch at module scope, so it cannot be used by a
solver that is meant to run anywhere NumPy runs. These are transcriptions of
the same profiles, and `verify_against_materials()` checks them against the
originals to machine precision -- run it on any machine that has torch before
trusting a number produced here.
"""
import numpy as np

# (name, physical x-domain, E_ref, rho_ref) -- from materials.py
_SPECS = {
    "Homogeneous": ((-1.0, 1.0), 80.0, 100.0),
    "TwoLayer":    ((-1.0, 1.0), 80.0, 100.0),
    "MultiLayer":  ((-1.5, 1.5), 60.0, 100.0),
}


class Material:
    """Nondimensionalized material: x in [x_min, x_max], rho == 1, E order 1."""

    def __init__(self, name):
        (x_lo, x_hi), self.E_ref, self.rho_ref = _SPECS[name]
        self.name = name
        self.L_ref = (x_hi - x_lo) / 2.0
        self.V_ref = np.sqrt(self.E_ref / self.rho_ref)
        self.T_ref = self.L_ref / self.V_ref
        self.x_min = x_lo / self.L_ref
        self.x_max = x_hi / self.L_ref
        # Interface half-width, nondimensional. 0.02 physical in materials.py.
        self._w = 0.02 / self.L_ref
        if name == "TwoLayer":
            self._E1 = 80.0 / self.E_ref
            self._E2 = 120.0 / self.E_ref
        if name == "MultiLayer":
            self.n_layers = 6
            self._E_vals = np.linspace(60.0, 150.0, self.n_layers) / self.E_ref

    def E(self, x):
        x = np.asarray(x, dtype=float)
        if self.name == "Homogeneous":
            return np.ones_like(x)
        if self.name == "TwoLayer":
            alpha = 0.5 * (1.0 + np.tanh(x / self._w))
            return self._E1 * (1.0 - alpha) + self._E2 * alpha
        layer_width = (self.x_max - self.x_min) / self.n_layers
        out = np.full_like(x, self._E_vals[0])
        for k in range(self.n_layers - 1):
            boundary = self.x_min + (k + 1) * layer_width
            alpha = 0.5 * (1.0 + np.tanh((x - boundary) / self._w))
            out = out * (1.0 - alpha) + self._E_vals[k + 1] * alpha
        return out

    def rho(self, x):
        return np.ones_like(np.asarray(x, dtype=float))

    def c(self, x):
        return np.sqrt(self.E(x) / self.rho(x))


def gaussian_derivative_ic(x, sigma_g=0.1):
    """Initial displacement: derivative of a Gaussian, normalized to unit peak.

    Matches `wave/problem_data.py`'s `gaussian_ic`; sigma_g is nondimensional
    here, as it is everywhere the PINN runs use it.
    """
    x = np.asarray(x, dtype=float)
    f = np.exp(-0.5 * (x / sigma_g) ** 2)
    dfdx = -(x / sigma_g ** 2) * f
    return dfdx / (np.max(np.abs(dfdx)) + 1e-12)


def verify_against_materials(tol=1e-12):
    """Assert these profiles equal wave/materials.py's. Needs torch."""
    import torch
    from wave.materials import HomogeneousModel, TwoLayerModel, MultiLayerModel
    pairs = [("Homogeneous", HomogeneousModel()), ("TwoLayer", TwoLayerModel()),
             ("MultiLayer", MultiLayerModel())]
    report = {}
    for name, ref in pairs:
        mine = Material(name)
        assert abs(mine.x_min - ref.x_min) < tol and abs(mine.x_max - ref.x_max) < tol
        x = np.linspace(mine.x_min, mine.x_max, 4001)
        e_ref = ref.E(torch.tensor(x, dtype=torch.float64)).numpy()
        r_ref = ref.rho(torch.tensor(x, dtype=torch.float64)).numpy()
        de = float(np.max(np.abs(mine.E(x) - e_ref)))
        dr = float(np.max(np.abs(mine.rho(x) - r_ref)))
        assert de < tol and dr < tol, (name, de, dr)
        report[name] = {"max_abs_dE": de, "max_abs_drho": dr}
    return report


if __name__ == "__main__":
    import json, sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    print(json.dumps(verify_against_materials(), indent=2))
    print("material profiles match wave/materials.py")
