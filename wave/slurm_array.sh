#!/bin/bash
# =============================================================================
# slurm_array.sh  –  Slurm Array Job for Parameter Sweep on Sherlock
# =============================================================================
# Submit this script from the repository root:
#
#   sbatch wave/slurm_array.sh
#
# This script runs a parameter sweep of 360 configurations across the GPU cluster.
# Each array task runs a single (material, model_config, ansatz, lbfgs_only) run.
#
# ── CONFIGURING GPU CONCURRENCY (HOW MANY GPUS TO USE) ────────────────────────
# By default, this script uses at most 30 GPUs concurrently to avoid queue hogging.
# You can change the limit in two ways:
#
#   1. Edit the limit in the '#SBATCH --array=0-359%30' line below (change %30 to %10, etc.)
#   2. Override it directly at submission time without editing the script:
#
#         sbatch --array=0-359%10 wave/slurm_array.sh
#
#      (The above command will run at most 10 tasks in parallel, using 10 GPUs).
# =============================================================================
#SBATCH --job-name=wave_sweep
#SBATCH --output=slurm_logs/wave_%A_%a.out
#SBATCH --error=slurm_logs/wave_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-359%30

set -euo pipefail

# 1. Create logs directory if it doesn't exist
mkdir -p slurm_logs

echo "=============================================================="
echo "  Starting Slurm Array Job: Task $SLURM_ARRAY_TASK_ID"
echo "  Job ID: $SLURM_JOB_ID | Array Job ID: $SLURM_ARRAY_JOB_ID"
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

# 6. Execute the specific run
OUTPUT_DIR="$SCRATCH/wave_results"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Executing experiment configuration $SLURM_ARRAY_TASK_ID..."
python3 wave/run_experiment.py --run-id "$SLURM_ARRAY_TASK_ID" --output-dir "$OUTPUT_DIR"

echo "=============================================================="
echo "  Task $SLURM_ARRAY_TASK_ID finished at $(date)"
echo "=============================================================="
