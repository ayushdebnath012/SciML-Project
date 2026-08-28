#!/bin/bash
# Loss-weighting ablation.
#
# The baseline recipe combines two adaptive schemes: causal time-slab weighting
# and GradNorm balancing of w_pde / w_bc / w_ic. In the first baseline runs the
# causal frontier never leaves slab 1 of 16 and GradNorm settles at
# w = (pde 0.001, bc 1.999, ic 0), i.e. the PDE residual is effectively switched
# off and the network is fitting the boundary condition alone. This turns each
# scheme off in isolation to find out which one is responsible.
#
#   base    both on (this is the main sweep; listed here only for reference)
#   nogn    GradNorm off, weights pinned at 1
#   nocaus  causal off (one slab, every time weighted equally)
#   plain   both off
set -u
ROOT=/home/trishita/sciml_pinn_neurips2026
OUT=$ROOT/results_ablation
LOGS=$ROOT/logs
PY=$ROOT/conda_env/bin/python
mkdir -p "$OUT" "$LOGS"

run () {   # $1 run-id  $2 variant  $3 gpu
  local extra=""
  case $2 in
    nogn)   extra="--no-gradnorm" ;;
    nocaus) extra="--causal-chunks 1" ;;
    plain)  extra="--no-gradnorm --causal-chunks 1" ;;
  esac
  CUDA_VISIBLE_DEVICES=$3 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg \
    "$PY" "$ROOT/wave/run_experiment.py" --run-id "$1" --output-dir "$OUT" \
    --run-tag "_$2" $extra > "$LOGS/abl_${1}_$2.log" 2>&1
  echo "$1 $2 exit=$?" >> "$OUT/progress.txt"
}

cd "$ROOT" || exit 1
: > "$OUT/progress.txt"
# run 2  = Homogeneous PINN_h3_w64 ansatz_true
# run 12 = Homogeneous FourierFeaturePINN_h3_w64_sigma3 ansatz_true
( run 2  nogn   0 ; run 2  nocaus 0 ; run 2  plain  0 ) &
( run 12 nogn   1 ; run 12 nocaus 1 ; run 12 plain  1 ) &
wait
echo "ABLATION DONE" >> "$OUT/progress.txt"
