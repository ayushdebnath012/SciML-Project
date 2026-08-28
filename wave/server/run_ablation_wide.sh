#!/bin/bash
# Wide loss-weighting ablation: every architecture, every material.
#
# The two-config pilot found that switching BOTH adaptive schemes off beats the
# published recipe by 13-25x (FourierFeature 3.93 -> 0.30, VanillaPINN
# 29.55 -> 1.16). That is the paper's central claim now, so it needs the full
# architecture x material grid behind it rather than one config on one material.
#
# Order matters: `plain` (both off) runs first across the whole grid, because it
# is the headline. `nocaus` follows to separate which of the two schemes is
# responsible -- the pilot says causal weighting is the bigger culprit, but on
# two configs only.
#
#   bash wave/server/run_ablation_wide.sh [WORKERS]
set -u
ROOT=/home/trishita/sciml_pinn_neurips2026
W=${1:-4}
OUT=$ROOT/results_ablation
STATE=$ROOT/ablation_state
LOGS=$ROOT/logs
PY=$ROOT/conda_env/bin/python
mkdir -p "$OUT" "$STATE" "$LOGS"

QUEUE=$STATE/queue.txt
CURSOR=$STATE/cursor
LOCK=$STATE/lock
DONE=$STATE/done.txt

if [ ! -f "$QUEUE" ]; then
  # One representative config per architecture, hard-constraint IC:
  #   2 PINN | 12 FourierFeature | 28 PirateNet | 38 KAN
  #  50 WavKAN | 62 ChebyshevKAN | 68 FourierWavKAN
  # Material offsets: Homogeneous +0, TwoLayer +100, MultiLayer +200.
  : > "$QUEUE"
  for variant in plain nocaus; do
    for base in 2 12 28 38 50 62 68; do
      for off in 0 100 200; do
        # the pilot already covers ids 2 and 12 on Homogeneous
        if [ "$off" = "0" ] && { [ "$base" = "2" ] || [ "$base" = "12" ]; }; then
          continue
        fi
        echo "$((base + off)) $variant" >> "$QUEUE"
      done
    done
  done
  echo 0 > "$CURSOR"
  : > "$DONE"
fi
TOTAL=$(wc -l < "$QUEUE")
echo "ablation queue: $TOTAL runs, $W workers"

next_job () {
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
  local gpu=$1
  while true; do
    local job; job=$(next_job)
    [ -z "$job" ] && break
    local id=${job%% *} variant=${job##* }
    local extra=""
    case $variant in
      nogn)   extra="--no-gradnorm" ;;
      nocaus) extra="--causal-chunks 1" ;;
      plain)  extra="--no-gradnorm --causal-chunks 1" ;;
    esac
    local t0=$SECONDS
    CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg \
      "$PY" "$ROOT/wave/run_experiment.py" --run-id "$id" --output-dir "$OUT" \
      --run-tag "_$variant" $extra > "$LOGS/abl_${id}_$variant.log" 2>&1
    echo "$id $variant rc=$? secs=$((SECONDS - t0))" >> "$DONE"
  done
}

cd "$ROOT" || exit 1
for i in $(seq 1 "$W"); do
  worker $(( (i - 1) % 2 )) &
done
wait
echo "WIDE ABLATION DONE ($(wc -l < "$DONE") runs)" >> "$DONE"
