#!/bin/bash
# Worker pool for the PINN/KAN sweep on the 2-GPU box.
#
#   bash wave/server/run_pool.sh [WORKERS_PER_GPU] [RUNLIST]
#
# Workers pull the next run-id from a shared queue under flock rather than
# taking a fixed shard: run times vary by more than an order of magnitude
# across architectures (a 2-layer MLP against a 4-layer spline KAN), so static
# sharding would leave one worker running long after the others finished.
set -u
ROOT=/home/trishita/sciml_pinn_neurips2026
WPG=${1:-3}
RUNLIST=${2:-$ROOT/wave/server/priority_runs.txt}
OUT=$ROOT/results
STATE=$ROOT/sweep_state
LOGS=$ROOT/logs
PY=$ROOT/conda_env/bin/python
mkdir -p "$OUT" "$STATE" "$LOGS"

QUEUE=$STATE/queue.txt
CURSOR=$STATE/cursor
LOCK=$STATE/lock
DONE=$STATE/done.txt

if [ ! -f "$QUEUE" ]; then
  grep -v '^#' "$RUNLIST" | tr ' ' '\n' | grep -E '^[0-9]+$' > "$QUEUE"
  echo 0 > "$CURSOR"
  : > "$DONE"
fi
TOTAL=$(wc -l < "$QUEUE")
echo "queue: $TOTAL runs, $WPG workers per GPU on 2 GPUs"

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

worker () {   # $1 = gpu id, $2 = worker slot
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
      > "$LOGS/run_$id.log" 2>&1
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
echo "POOL DONE ($(wc -l < "$DONE") runs)" >> "$DONE"
