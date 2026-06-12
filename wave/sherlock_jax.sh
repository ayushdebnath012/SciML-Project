#!/bin/bash
# =============================================================================
# sherlock_jax.sh  –  JAX backend SLURM array sweep on Sherlock
# =============================================================================
# Submit from the repository root:
#
#   sbatch wave/sherlock_jax.sh
#
# Runs the full 528-run sweep (all 7 architectures x 3 materials x hard/soft IC
# x {Adam+L-BFGS, L-BFGS-only}, plus the SOAP-optimizer and RBA-weighting arms)
# on the JAX backend, one run per array task.
#
# Concurrency: %30 caps the sweep at 30 GPUs at a time. Lower it at submission
# time without editing the file:
#
#   sbatch --array=0-527%10 wave/sherlock_jax.sh
#
# To see what each run-id maps to:
#
#   python wave/run_experiment.py --list-runs
#
# The environment ($SCRATCH/wave_jax_env) is built automatically on first use
# (flock-guarded so concurrent tasks don't race), but running
#   bash wave/setup_sherlock_jax.sh
# once beforehand is recommended.
# =============================================================================
#SBATCH --job-name=wave_jax
#SBATCH --output=slurm_logs/wave_jax_%A_%a.out
#SBATCH --error=slurm_logs/wave_jax_%A_%a.err
#SBATCH -p serc,gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16GB
#SBATCH --time=08:00:00
#SBATCH --array=0-527%30

set -euo pipefail

mkdir -p slurm_logs

echo "=============================================================="
echo "  Wave JAX sweep – array task $SLURM_ARRAY_TASK_ID"
echo "  Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST | $(date)"
echo "=============================================================="

# ── Scratch + modules ────────────────────────────────────────────────────────
if [ -z "${SCRATCH:-}" ]; then
    SCRATCH="/scratch/users/${USER:?Neither \$SCRATCH nor \$USER is set}"
fi

for py_mod in "python/3.12.1" "python/3" "python"; do
    module load "$py_mod" 2>/dev/null && break
done

# ── Environment (build once, flock-guarded against concurrent array tasks) ──
VENV_DIR="$SCRATCH/wave_jax_env"
LOCKFILE="$SCRATCH/.wave_jax_env.lock"
REPO_ROOT="$(pwd)"

(
    flock -x 9
    if [ ! -f "$VENV_DIR/.ready" ]; then
        echo "[INFO] Env not found – building it now (one-time, ~10 min) ..."
        bash "$REPO_ROOT/wave/setup_sherlock_jax.sh"
    fi
) 9>"$LOCKFILE"

source "$VENV_DIR/bin/activate"
echo "[OK]   venv: $(which python)"

# ── Verify JAX sees the GPU ──────────────────────────────────────────────────
python3 - <<'EOF'
import jax
devs = jax.devices()
print("  JAX devices:", devs)
if not any(d.platform == "gpu" for d in devs):
    print("  [WARNING] JAX does NOT see a GPU — this task will run on CPU and be slow.")
EOF

# ── Run the experiment for this array index ─────────────────────────────────
OUTPUT_DIR="$SCRATCH/wave_results_jax"
mkdir -p "$OUTPUT_DIR"

python3 wave/run_experiment.py \
    --backend jax \
    --run-id "$SLURM_ARRAY_TASK_ID" \
    --output-dir "$OUTPUT_DIR"

echo "=============================================================="
echo "  Task $SLURM_ARRAY_TASK_ID finished at $(date)"
echo "=============================================================="
