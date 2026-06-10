#!/bin/bash
# =============================================================================
# slurm_h100.sh  –  Single GPU Sweeper on Sherlock
# =============================================================================
# Submit this script from the repository root:
#
#   sbatch wave/slurm_h100.sh
#
# This script runs the entire parameter sweep (360 configurations) sequentially
# on a single high-performance GPU. By default, it requests 1 H100 GPU, but
# you can easily configure this to request other GPUs if needed.
#
# ── CONFIGURING GPU TYPE ──────────────────────────────────────────────────────
# You can change which GPU type to request by editing the '--gres' line:
#
#   - To request any available GPU : #SBATCH --gres=gpu:1
#   - To request an H100 GPU       : #SBATCH --gres=gpu:h100:1  (Default)
#   - To request an A100 GPU       : #SBATCH --gres=gpu:a100:1
#   - To request a V100 GPU        : #SBATCH --gres=gpu:v100:1
# =============================================================================
#SBATCH --job-name=wave_h100
#SBATCH --output=slurm_logs/wave_h100_%j.out
#SBATCH --error=slurm_logs/wave_h100_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

# 1. Create logs directory if it doesn't exist
mkdir -p slurm_logs

echo "=============================================================="
echo "  Starting Single GPU Sweep Job (H100 Node)"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node  : $SLURM_NODELIST"
echo "  Time  : $(date)"
echo "=============================================================="

# 2. Resolve Scratch Directory Safely
if [ -z "${SCRATCH:-}" ]; then
    if [ -n "${USER:-}" ]; then
        SCRATCH="/scratch/users/$USER"
    else
        echo "[ERROR] \$SCRATCH and \$USER environment variables are undefined."
        exit 1
    fi
fi

# 3. Load Python and CUDA modules
echo "[INFO] Loading modules..."
LOADED_PYTHON=""
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
fi

# Load CUDA if available, to ensure driver library resolution
module load cuda/12.1.1 2>/dev/null || module load cuda 2>/dev/null || echo "[WARNING] Could not load system cuda module. PyTorch might still run."

# 4. Activate the virtual environment from $SCRATCH
VENV_DIR="$SCRATCH/wave_env"
if [ ! -d "$VENV_DIR" ]; then
    echo "[ERROR] Virtual environment not found at $VENV_DIR"
    echo "        Please run 'bash wave/setup_sherlock.sh' first to build the environment."
    exit 1
fi

source "$VENV_DIR/bin/activate"
echo "[OK]   Activated venv: $(which python)"

# 5. Verify GPU is detected by PyTorch
echo ""
echo "[INFO] Checking PyTorch GPU availability..."
python3 -c "
import torch
print('  PyTorch version:', torch.__version__)
print('  CUDA available :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  Device Name    :', torch.cuda.get_device_name(0))
else:
    print('  [WARNING] CUDA is NOT available to PyTorch! Running on CPU will be extremely slow.')
"
echo ""

# 6. Execute the entire sweep sequentially
OUTPUT_DIR="$SCRATCH/wave_results"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Executing full sweep sequentially..."
python3 wave/run_experiment.py --output-dir "$OUTPUT_DIR"

echo "=============================================================="
echo "  Sweep Job finished at $(date)"
echo "=============================================================="
