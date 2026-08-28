#!/bin/bash
# Re-run every causal result used by the paper after fixing continuation-state
# handoff. Corrected artefacts go to a separate tree; existing sweep results are
# preserved for comparison.
#
#   bash wave/server/run_contfix.sh [WORKERS_PER_GPU]
set -u
ROOT=/home/trishita/sciml_pinn_neurips2026
WPG=${1:-3}
RUNLIST=$ROOT/wave/server/priority_contfix_runs.txt
OUT=$ROOT/results_corrected
STATE=$ROOT/contfix_state
LOGS=$ROOT/logs
PY=$ROOT/conda_env/bin/python
mkdir -p "$OUT" "$STATE" "$LOGS"

QUEUE=$STATE/queue.txt
CURSOR=$STATE/cursor
LOCK=$STATE/lock
DONE=$STATE/done.txt

if [ ! -f "$QUEUE" ]; then
  grep -v '^#' "$RUNLIST" | grep -E '^[0-9]+$' > "$QUEUE"
  echo 0 > "$CURSOR"
  : > "$DONE"
fi
TOTAL=$(wc -l < "$QUEUE")
echo "corrected causal queue: $TOTAL runs, $WPG workers per GPU on 2 GPUs"

next_id () {
  exec 9>"$LOCK"
  flock 9
  local i; i=$(cat "$CURSOR")
  if [ "$i" -ge "$TOTAL" ]; then echo ""; else
    sed -n "$((i + 1))p" "$QUEUE"
    echo $((i + 1)) > "$CURSOR"
  fi
  flock -u 9
}

worker () {
  local gpu=$1 slot=$2
  while true; do
    local id; id=$(next_id)
    [ -z "$id" ] && break
    local t0=$SECONDS
    CUDA_VISIBLE_DEVICES=$gpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=.22 \
    MPLBACKEND=Agg \
      "$PY" "$ROOT/wave/run_experiment.py" --run-id "$id" --output-dir "$OUT" \
      > "$LOGS/contfix_run_$id.log" 2>&1
    local rc=$?
    echo "$id rc=$rc secs=$((SECONDS - t0)) gpu=$gpu slot=$slot" >> "$DONE"
  done
}

cd "$ROOT" || exit 1
for gpu in 0 1; do
  for slot in $(seq 1 "$WPG"); do
    worker "$gpu" "$slot" &
  done
done
wait
echo "CORRECTED CAUSAL POOL DONE ($(wc -l < "$DONE") runs)" >> "$DONE"
