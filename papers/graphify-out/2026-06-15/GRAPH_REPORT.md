# Graph Report - papers  (2026-06-15)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 36 nodes · 47 edges · 6 communities
- Extraction: 49% EXTRACTED · 45% INFERRED · 6% AMBIGUOUS · INFERRED: 21 edges (avg confidence: 0.69)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `cc9bd361`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_PINNs for Elastic Waves|PINNs for Elastic Waves]]
- [[_COMMUNITY_PINN Training Techniques|PINN Training Techniques]]
- [[_COMMUNITY_Physics-Informed KANs|Physics-Informed KANs]]
- [[_COMMUNITY_Wavelet KAN Foundations|Wavelet KAN Foundations]]
- [[_COMMUNITY_Neural Operators & Wave Inversion|Neural Operators & Wave Inversion]]
- [[_COMMUNITY_PIKAN Applications|PIKAN Applications]]

## God Nodes (most connected - your core abstractions)
1. `An Expert's Guide to Training PINNs` - 8 edges
2. `Wav-KAN: Wavelet Kolmogorov-Arnold Networks` - 6 edges
3. `A Practitioner's Guide to Kolmogorov-Arnold Networks` - 5 edges
4. `KAN: Kolmogorov-Arnold Networks` - 5 edges
5. `PINNs for Learning High-Frequency Elastic Waves in Complex Layered Media` - 5 edges
6. `Adaptive Training of Grid-Dependent PIKANs (jaxKAN)` - 3 edges
7. `DeepONet: Learning Nonlinear Operators` - 3 edges
8. `Graph Neural Networks for Full Waveform Inversion` - 3 edges
9. `HWF-PIKAN: Hybrid Wavelet-Fourier PIKAN` - 3 edges
10. `PIKAN for Multi-Material Elasticity in Electronic Packaging` - 3 edges

## Surprising Connections (you probably didn't know these)
- `Learning the Elastic Wave Equation with Fourier Neural Operators` --semantically_similar_to--> `DeepONet: Learning Nonlinear Operators`  [INFERRED] [semantically similar]
  Learning the elastic wave equation with FNOs.pdf → DeepONet.pdf
- `Physics-Informed Attention-Based Neural Network (PIANN)` --conceptually_related_to--> `DeepONet: Learning Nonlinear Operators`  [AMBIGUOUS]
  PIANN.pdf → DeepONet.pdf
- `Graph Neural Networks for Full Waveform Inversion` --conceptually_related_to--> `Physics-Informed Attention-Based Neural Network (PIANN)`  [AMBIGUOUS]
  GNNs for FWI.pdf → PIANN.pdf
- `PIKAN for Explainable UAV Channel Modelling` --conceptually_related_to--> `A Practitioner's Guide to Kolmogorov-Arnold Networks`  [INFERRED]
  PIKAN for Explainable UAV Channel Modelling.pdf → Guide_to_KANs.pdf
- `Frequency Control and Spectral Bias in Wavelet-Based KANs` --conceptually_related_to--> `Physics-informed KANs: Architectures and Hyperparameters for Navier-Stokes`  [INFERRED]
  Spectral Bias in PIKANs.pdf → PIKAN_Hyperparam_NS.pdf

## Import Cycles
- None detected.

## Communities (6 total, 0 thin omitted)

### Community 0 - "PINNs for Elastic Waves"
Cohesion: 0.39
Nodes (8): Physics-Informed Neural Networks for Quantum Eigenvalue Problems, PINNs for Learning High-Frequency Elastic Waves in Complex Layered Media, Stability in Training PINNs for Stiff PDEs: Why Initial Conditions Matter, PINNs for Wave Propagation and Full Waveform Inversions, PirateNets: Physics-Informed Deep Learning with Residual Adaptive Networks, Scientific Machine Learning for Guided Wave and SAW Propagation, SeismicNet: PINNs for Seismic Wave Modeling in Semi-Infinite Domain, Frequency Control and Spectral Bias in Wavelet-Based KANs

### Community 1 - "PINN Training Techniques"
Cohesion: 0.29
Nodes (8): Causal Training (Temporal Weighting), Random Fourier Feature Embeddings, Self-Adaptive Gradient-Norm Loss Weighting, PDE Non-Dimensionalization, An Expert's Guide to Training PINNs, Raissi et al. Original PINNs Formulation, Random Weight Factorization, Tancik et al. Fourier Features

### Community 2 - "Physics-Informed KANs"
Cohesion: 0.67
Nodes (6): Adaptive Training of Grid-Dependent PIKANs (jaxKAN), A Practitioner's Guide to Kolmogorov-Arnold Networks, HWF-PIKAN: Hybrid Wavelet-Fourier PIKAN, KAN: Kolmogorov-Arnold Networks, PIKAN for Multi-Material Elasticity in Electronic Packaging, PIKAN for Explainable UAV Channel Modelling

### Community 3 - "Wavelet KAN Foundations"
Cohesion: 0.47
Nodes (6): Continuous Wavelet Transform, Discrete Wavelet Transform, Gabor Wavelet Implicit Neural Representation, Kolmogorov-Arnold Representation Theorem, Wav-KAN: Wavelet Kolmogorov-Arnold Networks, Spl-KAN (Liu et al. B-Spline KAN)

### Community 4 - "Neural Operators & Wave Inversion"
Cohesion: 0.60
Nodes (5): DeepONet: Learning Nonlinear Operators, Elastic Wave Lecture Notes (ECE471), Graph Neural Networks for Full Waveform Inversion, Learning the Elastic Wave Equation with Fourier Neural Operators, Physics-Informed Attention-Based Neural Network (PIANN)

### Community 5 - "PIKAN Applications"
Cohesion: 0.67
Nodes (3): Physics-informed KANs: Architectures and Hyperparameters for Navier-Stokes, Physics-Informed Kolmogorov-Arnold Networks for Power System Dynamics, PIKANs for Landslide Time-to-Failure Prediction

## Ambiguous Edges - Review These
- `DeepONet: Learning Nonlinear Operators` → `Physics-Informed Attention-Based Neural Network (PIANN)`  [AMBIGUOUS]
  PIANN.pdf · relation: conceptually_related_to
- `Graph Neural Networks for Full Waveform Inversion` → `Physics-Informed Attention-Based Neural Network (PIANN)`  [AMBIGUOUS]
  GNNs for FWI.pdf · relation: conceptually_related_to
- `Stability in Training PINNs for Stiff PDEs: Why Initial Conditions Matter` → `PirateNets: Physics-Informed Deep Learning with Residual Adaptive Networks`  [AMBIGUOUS]
  PINNs for Stiff PDEs Why Initial Conditions Matter.pdf · relation: references

## Knowledge Gaps
- **8 isolated node(s):** `Physics-Informed Neural Networks for Quantum Eigenvalue Problems`, `PIKANs for Landslide Time-to-Failure Prediction`, `Causal Training (Temporal Weighting)`, `Self-Adaptive Gradient-Norm Loss Weighting`, `Random Weight Factorization` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `DeepONet: Learning Nonlinear Operators` and `Physics-Informed Attention-Based Neural Network (PIANN)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Graph Neural Networks for Full Waveform Inversion` and `Physics-Informed Attention-Based Neural Network (PIANN)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Stability in Training PINNs for Stiff PDEs: Why Initial Conditions Matter` and `PirateNets: Physics-Informed Deep Learning with Residual Adaptive Networks`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `An Expert's Guide to Training PINNs` connect `PINN Training Techniques` to `Wavelet KAN Foundations`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Why does `Wav-KAN: Wavelet Kolmogorov-Arnold Networks` connect `Wavelet KAN Foundations` to `PINN Training Techniques`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `A Practitioner's Guide to Kolmogorov-Arnold Networks` connect `Physics-Informed KANs` to `Neural Operators & Wave Inversion`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `A Practitioner's Guide to Kolmogorov-Arnold Networks` (e.g. with `PIKAN for Multi-Material Elasticity in Electronic Packaging` and `PIKAN for Explainable UAV Channel Modelling`) actually correct?**
  _`A Practitioner's Guide to Kolmogorov-Arnold Networks` has 2 INFERRED edges - model-reasoned connections that need verification._