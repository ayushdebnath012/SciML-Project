"""Analytic checks for every solver plus an end-to-end build.

    python wave/multiphysics/test_multiphysics.py          # ~1 min

Each solver is compared with a closed-form solution, not with itself:
DC against the half-space and the two-layer image series, MT against the 1-D
Wait recursion (TE and TM), seismic against direct- and reflected-arrival
times. Tolerances are the measured accuracies with margin, so a regression in
the boundary conditions or the wavenumber quadrature fails here.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geology                                                   # noqa: E402
from dc_solver import DCSolver, two_layer_pole_pole              # noqa: E402
from mt_solver import MTSolver, default_frequencies, layered_apparent   # noqa: E402
from seismic_solver import SeismicSolver                         # noqa: E402


def test_geology():
    rng = np.random.default_rng(0)
    v_all, r_all = [], []
    for _ in range(100):
        g = geology.sample_geology(rng)
        assert g["velocity"].shape == (70, 70)
        assert g["layer_id"].dtype == np.int8 and g["layer_id"].max() == g["n_layers"] - 1
        assert np.allclose(g["conductivity"] * g["resistivity"], 1.0, rtol=1e-5)
        v_all.append(g["velocity"].ravel()); r_all.append(np.log10(g["resistivity"]).ravel())
    v, r = np.concatenate(v_all), np.concatenate(r_all)
    assert geology.V_MIN <= v.min() and v.max() <= geology.V_MAX
    corr = np.corrcoef(v, r)[0, 1]
    assert 0.1 < corr < 0.8, "v and log-resistivity should be correlated but not deterministic: %.2f" % corr
    p = geology.properties_from_velocity(g["velocity"], rng)
    assert np.array_equal(p["velocity"], g["velocity"])
    print("geology ok  corr(v, log rho_e) = %.2f" % corr)


def test_dc():
    S = DCSolver()
    r = np.abs(S.rec_y[None, :] - S.src_y[:, None])
    mask = r >= 20
    rho = 73.0
    ra = S.apparent_resistivity(np.full((70, 70), 1 / rho))
    assert np.abs(ra / rho - 1).max() < 1e-6, "half-space must be exact by calibration"
    for rho1, rho2, h_cells, tol in [(20., 500., 20, 0.012), (500., 20., 15, 0.008), (1000., 2., 30, 0.008)]:
        sig = np.full((70, 70), 1 / rho1); sig[h_cells:, :] = 1 / rho2
        ra = S.apparent_resistivity(sig)
        ana = two_layer_pole_pole(r[mask], rho1, rho2, (h_cells - 0.5) * S.dx)   # interface sits between nodes
        err = np.abs(ra[mask] / ana - 1).max()
        assert err < tol, "two-layer %g/%g: %.4f" % (rho1, rho2, err)
    assert np.isfinite(ra).all() and ra.shape == (10, 70)
    print("dc ok  two-layer max err %.4f" % err)


def test_mt():
    S = MTSolver()
    freqs = default_frequencies()
    cases = [("half-space", np.full((70, 70), 0.01), [100.], [], 0.005, 0.2),
             ("conductive half-space", np.full((70, 70), 0.5), [2.], [], 0.015, 2.5)]
    sig = np.full((70, 70), 1 / 500.); sig[15:, :] = 1 / 20.
    cases.append(("500/20", sig, [500., 20.], [150.], 0.005, 0.2))
    sig = np.full((70, 70), 1 / 50.); sig[10:30, :] = 1 / 2.; sig[30:, :] = 1 / 1000.
    cases.append(("50/2/1000", sig, [50., 2., 1000.], [100., 200.], 0.02, 0.8))
    for name, sigma, rhos, ths, tol_r, tol_p in cases:
        R = S.response(sigma, freqs)
        assert R.shape == (len(freqs), 70, 4) and np.isfinite(R).all()
        ths_eff = [t - S.h / 2 for t in ths]
        e_r = max(np.abs(R[i, :, [0, 2]] / layered_apparent(f, rhos, ths_eff)[0] - 1).max() for i, f in enumerate(freqs))
        e_p = max(np.abs(R[i, :, [1, 3]] - layered_apparent(f, rhos, ths_eff)[1]).max() for i, f in enumerate(freqs))
        assert e_r < tol_r and e_p < tol_p, "%s: rho_a err %.4f phase err %.2f" % (name, e_r, e_p)
    print("mt ok  last case rho_a err %.4f phase err %.2f deg" % (e_r, e_p))


def test_seismic():
    S = SeismicSolver(nbc=120)
    v = np.full((70, 70), 2500.0)
    d = S.shots(v)
    assert d.shape == (5, 1000, 70) and np.isfinite(d).all()
    s, rcol = 2, 60
    r = abs(S.src_cols[s] - rcol) * S.dx
    t_peak = np.argmax(np.abs(d[s, :, rcol])) * S.dt
    assert abs(t_peak - (1 / S.f0 + r / 2500.0)) < 6e-3, "direct arrival %.3f s" % t_peak
    tail = np.abs(d[:, -300:, :]).max() / np.abs(d).max()
    assert tail < 1e-3, "absorbing boundary residual %.1e" % tail
    v[35:, :] = 4000.0                       # interface at 350 m under a 2500 m/s layer
    d = S.shots(v)
    t = np.arange(S.nt) * S.dt
    tr = d[2, :, 60]                         # source column 34, receiver column 60: half-offset 130 m
    # source and receiver are 10 m deep, so the reflector is 340 m below them
    t_exp = 1 / S.f0 + 2 * np.hypot(130.0, 340.0) / 2500.0
    win = (t > t_exp - 0.05) & (t < t_exp + 0.05)
    t_ref = t[win][np.abs(tr[win]).argmax()]
    assert abs(t_ref - t_exp) < 6e-3, "reflection %.3f vs %.3f" % (t_ref, t_exp)
    print("seismic ok  direct %.0f ms, reflection %.0f ms, boundary residual %.1e" % (t_peak * 1e3, t_ref * 1e3, tail))


def test_build():
    out = Path(tempfile.mkdtemp()) / "mp"
    try:
        cmd = [sys.executable, str(Path(__file__).with_name("build_dataset.py")), "--test", "--out", str(out)]
        subprocess.run(cmd, check=True, capture_output=True)
        shapes = {"velocity": (4, 1, 70, 70), "conductivity": (4, 1, 70, 70), "seismic": (4, 5, 1000, 70),
                  "mt": (4, 8, 70, 4), "dc": (4, 10, 70), "layer_id": (4, 1, 70, 70)}
        for key, shape in shapes.items():
            a = np.load(out / key / (key + "1.npy"))
            assert a.shape == shape, (key, a.shape)
            assert np.isfinite(a).all()
        assert np.load(out / "velocity" / "velocity2.npy").shape[0] == 2
        assert (out / "meta.json").exists()
        print("build ok")
    finally:
        shutil.rmtree(out.parent, ignore_errors=True)


if __name__ == "__main__":
    test_geology(); test_dc(); test_mt(); test_seismic(); test_build()
    print("all multiphysics tests passed")
