"""Correctness checks for the OpenFWI operator models. Needs torch; CPU is fine.

    python wave/openfwi/test_openfwi.py

These are the claims the benchmark rests on that are not visible in a loss
curve: that the grouped PFNO really is a stack of independent branches, that
the graph kernel aggregates the neighbourhood it says it does, and that every
model emits the gather shape the scorer expects.
"""
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openfwi_data import DATASET_CONFIG, band_limit_oracle
from fetch_openfwi import npy_integrity
from openfwi_models import (DepthToTime, GraphKernelLayer, GroupedSpectralConv2d,
                            OpenFWIDeepONet, OpenFWIFNO, OpenFWIGNO, OpenFWIPFNO,
                            SpectralConv2d, ball_offsets, count_parameters,
                            count_parameters_real)
from train_openfwi import load_training_checkpoint, save_training_checkpoint

FAILURES = []


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + ("   " + detail if detail else ""))
    if not ok:
        FAILURES.append(name)


def test_grouped_spectral_matches_loop():
    """The whole point of the grouped implementation is that it changes speed,
    not semantics."""
    torch.manual_seed(0)
    groups, cin, cout, ma, mb = 3, 4, 5, 6, 6
    grouped = GroupedSpectralConv2d(groups, cin, cout, ma, mb)
    singles = []
    for k in range(groups):
        s = SpectralConv2d(cin, cout, ma, mb)
        with torch.no_grad():
            s.weight_pos.copy_(grouped.weight_pos[k])
            s.weight_neg.copy_(grouped.weight_neg[k])
        singles.append(s)
    x = torch.randn(2, groups, cin, 16, 16)
    got = grouped(x)
    want = torch.stack([singles[k](x[:, k]) for k in range(groups)], dim=1)
    err = (got - want).abs().max().item()
    check("grouped spectral conv == loop of independent convs",
          err < 1e-5, "max abs diff %.2e" % err)


def test_depth_to_time_starts_as_interpolation():
    d2t = DepthToTime(n_depth=70, t_latent=250)
    row_sums = d2t.map.sum(dim=1)
    ok_partition = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    nonneg = bool((d2t.map >= -1e-6).all())
    two_per_row = int((d2t.map.abs() > 1e-6).sum(dim=1).max())
    # A ramp resampled by a linear-interpolation operator is still that ramp.
    ramp = torch.linspace(-1.0, 1.0, 70).view(1, 1, 70, 1).expand(1, 2, 70, 3)
    got = d2t(ramp)[0, 0, :, 0]
    want = torch.linspace(-1.0, 1.0, 250)
    err = (got - want).abs().max().item()
    check("DepthToTime initialises to linear interpolation",
          ok_partition and nonneg and two_per_row <= 2 and err < 1e-5,
          "rows sum to 1: %s, <=2 nonzero: %d, ramp err %.2e"
          % (ok_partition, two_per_row, err))


def test_graph_kernel_neighbourhood():
    """Neighbour counts must match the stencil, including at the boundary where
    the ball is clipped."""
    r = 2
    offs = ball_offsets(r)
    expected_interior = len(offs)
    layer = GraphKernelLayer(width=3, radius=r, kernel_hidden=8, a_channels=1)
    # Force the kernel to output 1 and the local term to 0, so the layer
    # computes exactly mean_{j in N(i)} v(j) before the GELU.
    with torch.no_grad():
        layer.k_pos.weight.zero_(); layer.k_pos.bias.zero_()
        layer.k_a.weight.zero_()
        layer.k_out.weight.zero_(); layer.k_out.bias.fill_(1.0)
        layer.local.weight.zero_(); layer.local.bias.zero_()
    n = 9
    v = torch.ones(1, 3, n, n)
    a = torch.zeros(1, 1, n, n)
    out = layer(v, a)
    # mean of ones over any non-empty neighbourhood is 1 -> GELU(1)
    want = torch.nn.functional.gelu(torch.ones(1))
    uniform = (out - want).abs().max().item()

    # Count directly: put a 1 at a single node and read how many nodes see it.
    v2 = torch.zeros(1, 1, n, n)
    v2[0, 0, 4, 4] = 1.0
    layer2 = GraphKernelLayer(width=1, radius=r, kernel_hidden=8, a_channels=1)
    with torch.no_grad():
        layer2.k_pos.weight.zero_(); layer2.k_pos.bias.zero_()
        layer2.k_a.weight.zero_()
        layer2.k_out.weight.zero_(); layer2.k_out.bias.fill_(1.0)
        layer2.local.weight.zero_(); layer2.local.bias.zero_()
    raw = layer2(v2, torch.zeros(1, 1, n, n))
    seen = int(((raw - torch.nn.functional.gelu(torch.zeros(1))).abs() > 1e-6).sum())
    check("graph kernel aggregates the L2 ball, boundary included",
          uniform < 1e-5 and seen == expected_interior,
          "stencil %d offsets, %d nodes see a lone spike, uniform err %.2e"
          % (expected_interior, seen, uniform))


def test_graph_kernel_depends_on_velocity():
    """If the kernel ignored `a`, the GNO would be a CNN with extra steps."""
    torch.manual_seed(0)
    layer = GraphKernelLayer(width=4, radius=2, kernel_hidden=16, a_channels=1)
    v = torch.randn(1, 4, 8, 8)
    out1 = layer(v, torch.zeros(1, 1, 8, 8))
    out2 = layer(v, torch.ones(1, 1, 8, 8))
    diff = (out1 - out2).abs().max().item()
    check("graph kernel is input-dependent (varies with velocity)",
          diff > 1e-4, "max abs diff %.2e" % diff)


def test_model_shapes_and_gradients(nt=200, t_latent=50, batch=2):
    cfg = DATASET_CONFIG["flatvel-a"]
    nz = nx = cfg["n_grid"]
    ns = cfg["ns"]
    want = (batch, ns, nt, nx)
    v = torch.randn(batch, 1, nz, nx)
    models = {
        "FNO": OpenFWIFNO(width=8, modes_z=6, modes_x=6, modes_t=8, enc_layers=2,
                          dec_layers=2, n_sources=ns, nz=nz, nx=nx, nt=nt,
                          t_latent=t_latent),
        "GNO": OpenFWIGNO(width=8, kernel_hidden=16, radius=2, dec_radius=1,
                          enc_layers=2, dec_layers=1, n_sources=ns, nz=nz, nx=nx,
                          nt=nt, t_latent=t_latent),
        "DeepONet": OpenFWIDeepONet(nz=nz, nx=nx, nt=nt, n_sources=ns, latent=16,
                                    hidden=32, fourier_features=8),
        "PFNO": OpenFWIPFNO(n_freqs=8, width=4, modes=4, layers=2, n_sources=ns,
                            nz=nz, nx=nx, nt=nt),
    }
    for name, model in models.items():
        out = model(v)
        shape_ok = tuple(out.shape) == want
        finite = bool(torch.isfinite(out).all())
        out.pow(2).mean().backward()
        missing = [n for n, p in model.named_parameters()
                   if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        check("%-9s shape %s, finite, all params get gradient" % (name, want),
              shape_ok and finite and not missing,
              "got %s%s" % (tuple(out.shape),
                            "" if not missing else "  no-grad: %s" % missing[:3]))


def test_pfno_is_band_limited(nt=200, n_freqs=8):
    """PFNO must emit exactly zero above its band, or the reported oracle floor
    is not the floor it actually operates under."""
    cfg = DATASET_CONFIG["flatvel-a"]
    torch.manual_seed(0)
    model = OpenFWIPFNO(n_freqs=n_freqs, width=4, modes=4, layers=2,
                        n_sources=cfg["ns"], nz=70, nx=70, nt=nt)
    with torch.no_grad():
        out = model(torch.randn(2, 1, 70, 70))
    spec = torch.fft.rfft(out, dim=-2)
    above = spec[:, :, n_freqs:, :].abs().max().item()
    inside = spec[:, :, :n_freqs, :].abs().max().item()
    dc_imag = spec[:, :, 0, :].imag.abs().max().item()
    real = bool(torch.isreal(out).all())
    check("PFNO output is real, band-limited, real-valued DC",
          above < 1e-4 and inside > 1e-6 and dc_imag < 1e-4 and real,
          "|spec| above band %.2e, inside %.2e, DC imag %.2e" % (above, inside, dc_imag))


def test_band_limit_oracle_is_exact_at_full_band():
    import numpy as np
    g = np.random.default_rng(0).standard_normal((2, 5, 64, 7))
    e_full = band_limit_oracle(g, 64 // 2 + 1)
    e_half = band_limit_oracle(g, 8)
    check("band-limit oracle is lossless at the full band and lossy below it",
          e_full.max() < 1e-8 and e_half.mean() > 1.0,
          "full %.2e%%, 8 bins %.2f%%" % (e_full.max(), e_half.mean()))


def test_parameter_accounting():
    m = OpenFWIFNO(width=8, modes_z=4, modes_x=4, modes_t=4, enc_layers=1,
                   dec_layers=1, nz=70, nx=70, nt=100, t_latent=25)
    nominal, real = count_parameters(m), count_parameters_real(m)
    n_complex = sum(p.numel() for p in m.parameters() if p.is_complex())
    check("real parameter count charges complex weights twice",
          real == nominal + n_complex and n_complex > 0,
          "nominal %d, real %d, complex %d" % (nominal, real, n_complex))


def test_checkpoint_roundtrip_is_atomic():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "state.pt"
        payload = {"epoch": 7, "weights": torch.arange(5)}
        save_training_checkpoint(path, payload)
        restored = load_training_checkpoint(path, torch.device("cpu"))
        ok = (restored["epoch"] == 7 and
              torch.equal(restored["weights"], payload["weights"]) and
              not path.with_name(path.name + ".tmp").exists())
        check("training checkpoint round-trips via atomic replacement", ok)


def test_npy_integrity_rejects_truncated_payload():
    import numpy as np
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "chunk.npy"
        np.save(path, np.zeros((2, 3), dtype=np.float32))
        complete, _, _ = npy_integrity(path, (2, 3))
        with path.open("r+b") as fh:
            fh.truncate(path.stat().st_size - 4)
        truncated, reason, _ = npy_integrity(path, (2, 3))
        check("OpenFWI fetch rejects a valid header with truncated payload",
              complete and not truncated and "truncated payload" in reason,
              reason)


if __name__ == "__main__":
    print("torch", torch.__version__)
    print("\n-- kernels --")
    test_grouped_spectral_matches_loop()
    test_depth_to_time_starts_as_interpolation()
    test_graph_kernel_neighbourhood()
    test_graph_kernel_depends_on_velocity()
    print("\n-- models --")
    test_model_shapes_and_gradients()
    test_pfno_is_band_limited()
    print("\n-- accounting --")
    test_band_limit_oracle_is_exact_at_full_band()
    test_parameter_accounting()
    test_checkpoint_roundtrip_is_atomic()
    test_npy_integrity_rejects_truncated_payload()
    print("\n%d failure(s)" % len(FAILURES))
    sys.exit(1 if FAILURES else 0)
