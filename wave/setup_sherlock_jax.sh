#!/bin/bash
# =============================================================================
# setup_sherlock_jax.sh  –  One-time JAX environment setup for Sherlock HPC
# =============================================================================
# Run this ONCE on a Sherlock login node before submitting wave/sherlock_jax.sh:
#
#   bash wave/setup_sherlock_jax.sh
#
# Builds a virtualenv at $SCRATCH/wave_jax_env with:
#   - JAX (CUDA 12, pip-bundled CUDA libraries — no cuda module needed)
#   - equinox / optax / jaxopt / jaxkan
#   - CPU-only PyTorch (the runner uses torch only for seeding + the FD
#     reference solver, so the small CPU wheel is enough)
#   - everything else from requirements.txt (pykan, matplotlib, ...)
#
# The array job (wave/sherlock_jax.sh) will also build this env automatically
# on first run (flock-guarded), but running setup ahead of time keeps the
# first array task from spending its time budget on pip.
# =============================================================================

set -euo pipefail

echo "=================================================="
echo "  Wave Experiment – Sherlock JAX Environment Setup"
echo "=================================================="

# ── 1. Resolve Scratch Directory ─────────────────────────────────────────────
if [ -z "${SCRATCH:-}" ]; then
    SCRATCH="/scratch/users/${USER:?Neither \$SCRATCH nor \$USER is set}"
    echo "[WARNING] \$SCRATCH not set. Defaulting to $SCRATCH"
fi
echo "SCRATCH : $SCRATCH"

# ── 2. Load a Python module ──────────────────────────────────────────────────
LOADED_PYTHON=""
for py_mod in "python/3.12.1" "python/3" "python"; do
    if module load "$py_mod" 2>/dev/null; then
        LOADED_PYTHON="$py_mod"
        break
    fi
done
echo "[OK]   Python module: ${LOADED_PYTHON:-'(none loaded — using system python3)'}"
echo "[OK]   Using: $(python3 --version) at $(command -v python3)"

# ── 3. Create the virtualenv on $SCRATCH ─────────────────────────────────────
VENV_DIR="$SCRATCH/wave_jax_env"
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "[OK]   Activated venv: $(which python)"

pip install --upgrade pip --quiet

# ── 4. CPU-only PyTorch (seeding + FD reference solver only) ─────────────────
echo "[INFO] Installing CPU-only PyTorch ..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu --quiet
echo "[OK]   PyTorch (CPU) installed."

# ── 5. JAX with CUDA 12 (pip-bundled CUDA — works without a cuda module) ─────
echo "[INFO] Installing JAX (CUDA 12) + equinox/optax/jaxopt/jaxkan ..."
pip install --upgrade "jax[cuda12]" --quiet
pip install equinox optax jaxopt jaxkan --quiet
echo "[OK]   JAX stack installed."

# ── 6. Remaining project requirements (pykan, matplotlib, ...) ───────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[INFO] Installing requirements from $REPO_ROOT/requirements.txt ..."
pip install -r "$REPO_ROOT/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu --quiet
echo "[OK]   Requirements installed."

# ── 7. Sanity check + output dir ─────────────────────────────────────────────
python - <<'EOF'
import jax, equinox, optax, jaxopt, torch
print("  jax     :", jax.__version__, "| devices:", jax.devices())
print("  equinox :", equinox.__version__)
print("  optax   :", optax.__version__)
print("  torch   :", torch.__version__, "(CPU build is expected)")
EOF

mkdir -p "$SCRATCH/wave_results_jax"
touch "$VENV_DIR/.ready"
echo ""
echo "=================================================="
echo "  Setup complete. Submit the sweep with:"
echo "    sbatch wave/sherlock_jax.sh"
echo "  (run from the repository root)"
echo "=================================================="
