# FNO and Neural-Operator Papers for This Project

This project studies the 1D conservative elastic wave equation,

```text
rho(x) u_tt = d_x(E(x) u_x)
```

The files below are the most relevant FNO/neural-operator references downloaded
for the baseline branch.

| Local PDF | Citation target | Why it matters |
| --- | --- | --- |
| `Fourier_Neural_Operator_for_Parametric_PDEs.pdf` | Li et al., 2020, arXiv:2010.08895 | Original FNO formulation for learning PDE solution operators by parameterizing the integral kernel in Fourier space. |
| `Physics_Informed_Neural_Operator_for_PDEs_PINO.pdf` | Li et al., 2021, arXiv:2111.03794 | PINO: combines FNO-style operator learning with PDE residual constraints, useful after the supervised baseline is stable. |
| `Solving_Seismic_Wave_Equations_with_FNO.pdf` | Li et al., 2022, arXiv:2209.12340 | FNO/PFNO for variable-velocity seismic wavefields; motivates training across material/velocity families, not only one fixed medium. |
| `Rapid_Seismic_Waveform_Modeling_with_Neural_Operators.pdf` | Yang et al., 2022/2023, arXiv:2209.11955 | Neural operators for 2D elastic waveform modeling and inversion; supports varying velocity/source datasets and differentiable inversion. |
| `FNO_Surrogate_3D_Seismic_Waves.pdf` | Lehmann et al., 2023, arXiv:2304.10242 | FNO surrogate for 3D seismic wave propagation from geological descriptions; useful for framing the future 2D/3D extension. |
| `Learning_the_elastic_wave_equation_with_FNOs_CREWES.pdf` | Zhang, Trad, and Innanen, 2023, Geophysics / CREWES | Directly targets elastic wave-equation learning with FNOs on synthetic elastic datasets. |
| `Are_FNOs_Really_Faster_for_Time_Domain_Wave_Propagation.pdf` | Benchmark/caution paper, arXiv:2508.11119 | Useful cautionary benchmark: report training cost, inference cost, FD baseline cost, and accuracy instead of claiming speedup blindly. |

The implemented code baseline is in `src/operator_baselines.py` and
`wave/run_fno_baseline.py`. It trains supervised grid-to-grid models with input
channels `[E(x), rho(x), g(x), x, t]` and output `u(x,t)`.

## How To Proceed

1. Establish a supervised FNO baseline.
   - Generate many FD solutions over a family of 1D material profiles.
   - Use channels `[E(x), rho(x), g(x), x, t]` and predict the full wavefield `u(x,t)`.
   - Compare against a local CNN baseline with the same input/output tensors.

2. Benchmark on the project materials.
   - Hold out the named Homogeneous, TwoLayer, and MultiLayer profiles as fixed tests.
   - Report relative L2 error, runtime, parameter count, training data size, and grid size.

3. Test the neural-operator claim, not just interpolation.
   - Train on one grid and evaluate on a finer grid with `--eval-nx` / `--eval-nt`.
   - Vary material interfaces and optionally source location/width using `--random-ic`.

4. Add physics only after the supervised baseline is credible.
   - Follow PINO by adding a finite-difference PDE residual penalty on predicted grids.
   - Use the conservative residual `rho u_tt - d_x(E u_x)`, not `u_tt - c^2 u_xx`.

5. Be careful with speed claims.
   - Compare amortized inference cost against FD only after accounting for training-data generation and model training.
   - Present FNO as a surrogate/operator baseline unless it beats FD in the target repeated-query regime.
