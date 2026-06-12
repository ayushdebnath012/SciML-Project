#!/bin/bash
# =============================================================================
# sherlock.sh  –  Wave PINN/KAN sweep on Sherlock (JAX backend, single script)
# =============================================================================
# USAGE (from the repository root, on a Sherlock login node):
#
#   bash wave/sherlock.sh [NUM_GPUS] [WORKERS_PER_GPU]
#
#   bash wave/sherlock.sh              # 30 GPUs, 1 experiment per GPU (default)
#   bash wave/sherlock.sh 10           # at most 10 GPUs concurrently
#   bash wave/sherlock.sh 10 2         # 10 GPUs, 2 experiments sharing each GPU
#
# The script submits itself as a SLURM array job sized for the full 528-run
# sweep (7 architectures x 3 materials x hard/soft IC x {Adam+L-BFGS,
# L-BFGS-only}, plus the SOAP-optimizer and RBA-weighting arms).
#
# WORKERS_PER_GPU > 1 packs several runs onto one GPU (JAX preallocation is
# disabled and GPU memory is split evenly). 2-3 workers is a good fit for the
# small models in this sweep on a 40/80 GB card; CPU cores and RAM per task
# are scaled automatically.
#
# No separate setup step: the first task to start builds the Python env at
# $SCRATCH/wave_jax_env (flock-guarded, ~10 min one-time); later tasks wait.
#
# Other useful commands:
#   python wave/run_experiment.py --list-runs    # run-id -> experiment mapping
#   sbatch --array=42 wave/sherlock.sh           # re-run a single task id
#
# Results land in $SCRATCH/wave_results_jax/<Material>/<RunName>_jax/
# (l2_errors.json, loss curves, solution_comparison.png, .eqx checkpoints).
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

TOTAL_RUNS=528   # keep in sync with: python wave/run_experiment.py --list-runs

# ── Launcher mode: not inside a SLURM job yet -> submit ourselves ────────────
if [ -z "${SLURM_JOB_ID:-}" ]; then
    NUM_GPUS="${1:-30}"
    WORKERS="${2:-1}"
    NTASKS=$(( (TOTAL_RUNS + WORKERS - 1) / WORKERS ))
    CPUS=$(( 2 * WORKERS + 2 ))
    MEM=$(( 12 * WORKERS + 4 ))

    mkdir -p slurm_logs
    echo "Submitting sweep: $TOTAL_RUNS runs -> $NTASKS array tasks"
    echo "  GPUs (concurrent) : $NUM_GPUS"
    echo "  Workers per GPU   : $WORKERS"
    sbatch --array="0-$((NTASKS - 1))%${NUM_GPUS}" \
           --cpus-per-task="$CPUS" \
           --mem="${MEM}GB" \
           --export=ALL,WORKERS="$WORKERS" \
           "$0"
    exit 0
fi

# ── Array-task mode ──────────────────────────────────────────────────────────
WORKERS="${WORKERS:-1}"

mkdir -p slurm_logs

echo "=============================================================="
echo "  Wave JAX sweep – array task $SLURM_ARRAY_TASK_ID (workers=$WORKERS)"
echo "  Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST | $(date)"
echo "=============================================================="

# ── Scratch + Python module ──────────────────────────────────────────────────
if [ -z "${SCRATCH:-}" ]; then
    SCRATCH="/scratch/users/${USER:?Neither \$SCRATCH nor \$USER is set}"
fi

for py_mod in "python/3.12.1" "python/3" "python"; do
    module load "$py_mod" 2>/dev/null && break
done

# ── Environment: built once by whichever task grabs the lock first ──────────
# Contents: JAX (CUDA 12, pip-bundled CUDA libs — no cuda module needed),
# equinox/optax/jaxopt/jaxkan, CPU-only PyTorch (used only for seeding and
# the FD reference solver), and requirements.txt (pykan, matplotlib, ...).
VENV_DIR="$SCRATCH/wave_jax_env"
LOCKFILE="$SCRATCH/.wave_jax_env.lock"

(
    flock -x 9
    if [ ! -f "$VENV_DIR/.ready" ]; then
        echo "[INFO] Building env at $VENV_DIR (one-time, ~10 min) ..."
        python3 -m venv "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip --quiet
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
            --index-url https://download.pytorch.org/whl/cpu --quiet
        pip install --upgrade "jax[cuda12]" --quiet
        pip install equinox optax jaxopt jaxkan --quiet
        pip install -r requirements.txt \
            --extra-index-url https://download.pytorch.org/whl/cpu --quiet
        touch "$VENV_DIR/.ready"
        deactivate
        echo "[OK]   Environment ready."
    fi
) 9>"$LOCKFILE"

source "$VENV_DIR/bin/activate"
echo "[OK]   venv: $(which python)"

# ── GPU sharing: disable JAX preallocation, split memory across workers ─────
if [ "$WORKERS" -gt 1 ]; then
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=$(awk -v w="$WORKERS" 'BEGIN {printf "%.2f", 0.85 / w}')
    echo "[INFO] GPU shared by $WORKERS workers (mem fraction $XLA_PYTHON_CLIENT_MEM_FRACTION each)"
fi

# ── Verify JAX sees the GPU ──────────────────────────────────────────────────
python3 - <<'EOF'
import jax
devs = jax.devices()
print("  JAX devices:", devs)
if not any(d.platform == "gpu" for d in devs):
    print("  [WARNING] JAX does NOT see a GPU — this task will run on CPU and be slow.")
EOF

# ── Run this task's slice of run-ids (WORKERS runs in parallel on the GPU) ──
OUTPUT_DIR="$SCRATCH/wave_results_jax"
mkdir -p "$OUTPUT_DIR"

START=$(( SLURM_ARRAY_TASK_ID * WORKERS ))
PIDS=()
RIDS=()
for (( k = 0; k < WORKERS; k++ )); do
    RID=$(( START + k ))
    [ "$RID" -ge "$TOTAL_RUNS" ] && break
    echo "[INFO] Launching run-id $RID (log: slurm_logs/run_${RID}.log)"
    python3 wave/run_experiment.py \
        --backend jax \
        --run-id "$RID" \
        --output-dir "$OUTPUT_DIR" \
        > "slurm_logs/run_${RID}.log" 2>&1 &
    PIDS+=($!)
    RIDS+=("$RID")
done

FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[OK]   run-id ${RIDS[$i]} finished"
    else
        echo "[FAIL] run-id ${RIDS[$i]} exited nonzero — see slurm_logs/run_${RIDS[$i]}.log"
        FAIL=1
    fi
done

echo "=============================================================="
echo "  Task $SLURM_ARRAY_TASK_ID finished at $(date) (status=$FAIL)"
echo "=============================================================="
exit "$FAIL"
