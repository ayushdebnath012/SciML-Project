#!/bin/bash
# =============================================================================
# setup_sherlock.sh  –  One-time environment setup for Sherlock HPC
# =============================================================================
# Run this ONCE on a Sherlock login node (or interactive node) before
# submitting any SLURM jobs:
#
#   bash wave/setup_sherlock.sh
#
# What it does:
#   1. Resolves $SCRATCH directory safely.
#   2. Loads available Python module (python/3.12.1 or python/3).
#   3. Creates a Python virtualenv in $SCRATCH/wave_env (fast parallel storage).
#   4. Upgrades pip and installs PyTorch with CUDA 12.1.
#   5. Installs project requirements.txt.
#   6. Sets up the results output directory and convenience symlink.
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "  Wave Experiment – Sherlock Environment Setup"
echo "=============================================="

# ── 1. Resolve Scratch Directory ─────────────────────────────────────────────
if [ -z "${SCRATCH:-}" ]; then
    if [ -n "${USER:-}" ]; then
        SCRATCH="/scratch/users/$USER"
        echo "[WARNING] \$SCRATCH environment variable not set. Defaulting to $SCRATCH"
    else
        SCRATCH="./scratch_fallback"
        echo "[WARNING] Neither \$SCRATCH nor \$USER set. Using local fallback: $SCRATCH"
    fi
fi
echo "SCRATCH : $SCRATCH"
echo "User    : ${USER:-unknown}"
echo ""

# ── 2. Load the Python Module ────────────────────────────────────────────────
echo "[INFO] Loading Python module..."
LOADED_PYTHON=""
# Try to load Lmod modules sequentially
for py_mod in "python/3.12.1" "python/3" "python"; do
    if module load "$py_mod" 2>/dev/null; then
        LOADED_PYTHON="$py_mod"
        break
    fi
done

if [ -n "$LOADED_PYTHON" ]; then
    echo "[OK]   Loaded module: $LOADED_PYTHON"
else
    echo "[WARNING] Could not load any Python module via Lmod."
    echo "          Attempting to proceed with system python3..."
fi

if ! command -v python3 &>/dev/null; then
    echo "[ERROR] 'python3' executable not found. Please load a Python 3 module or install Python."
    exit 1
fi
echo "[OK]   Using: $(python3 --version) at $(command -v python3)"

# ── 3. Create the virtual environment on $SCRATCH ────────────────────────────
# $SCRATCH is a per-user parallel filesystem on Sherlock (high IOPS, no NFS lag).
VENV_DIR="$SCRATCH/wave_env"

if [ -d "$VENV_DIR" ]; then
    echo "[INFO] venv already exists at $VENV_DIR – skipping creation."
    echo "       Delete/rename it and re-run this script to rebuild from scratch."
else
    echo "[INFO] Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "[OK]   venv created."
fi

source "$VENV_DIR/bin/activate"
echo "[OK]   Activated virtualenv: $(which python)"

# ── 4. Upgrade pip ───────────────────────────────────────────────────────────
echo "[INFO] Upgrading pip..."
pip install --upgrade pip --quiet
echo "[OK]   pip upgraded."

# ── 5. Install PyTorch with CUDA 12.1 ────────────────────────────────────────
# CUDA 12.1 wheels are compatible with Sherlock GPU nodes (V100, A100, H100).
echo "[INFO] Installing PyTorch (CUDA 12.1) ..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121 --quiet
echo "[OK]   PyTorch installed."

# ── 6. Install remaining dependencies ────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo ""
echo "[INFO] Installing requirements from $REPO_ROOT/requirements.txt ..."
if [ -f "$REPO_ROOT/requirements.txt" ]; then
    pip install -r "$REPO_ROOT/requirements.txt" \
        --extra-index-url https://download.pytorch.org/whl/cu121 --quiet
    echo "[OK]   Requirements installed."
else
    echo "[WARNING] requirements.txt not found at $REPO_ROOT/requirements.txt"
fi

# ── 7. Create output directory on $SCRATCH ───────────────────────────────────
mkdir -p "$SCRATCH/wave_results"
echo "[OK]   Output directory ready: $SCRATCH/wave_results"

# ── 8. Optionally create a local symlink for easy inspection ─────────────────
RESULTS_LINK="$REPO_ROOT/experiment_results_sherlock"
if [ ! -L "$RESULTS_LINK" ] && [ ! -e "$RESULTS_LINK" ]; then
    ln -s "$SCRATCH/wave_results" "$RESULTS_LINK"
    echo "[OK]   Symlink created: $RESULTS_LINK -> $SCRATCH/wave_results"
fi

echo ""
echo "=============================================="
echo "  Setup complete! You can now submit jobs:"
echo ""
echo "  1. Array job (parallel parameter sweep):"
# Array limit: set concurrently. e.g. %30 uses 30 GPUs concurrently.
echo "     Adjust GPU concurrency limit inside wave/slurm_array.sh"
echo "     Submit with: sbatch wave/slurm_array.sh"
echo ""
echo "  2. Single GPU sweep (sequential):"
echo "     Submit with: sbatch wave/slurm_h100.sh"
echo "=============================================="
