#!/bin/bash
# OpenFWI forward benchmark on the 2-GPU box.
#
#   bash wave/openfwi/runners/run_openfwi.sh [EPOCHS] [TRAIN_CHUNKS] [MODELS]
#
# One dataset per GPU, run concurrently. The box is shared with other tenants,
# so this deliberately does not try to pack more than one job per GPU: PFNO's
# 64 grouped branches and GNO's stencil activations are both memory-hungry, and
# a second tenant arriving mid-run would OOM the pair rather than slow them.
set -u

ROOT=${ROOT:-$HOME/sciml_wave_sim}
DATA=${DATA:-$HOME/openfwi_data}
OUT=${OUT:-$HOME/openfwi_results}
PY=${PY:-$HOME/miniconda3/bin/python}

EPOCHS=${1:-100}
TRAIN_CHUNKS=${2:-4}
MODELS=${3:-FNO,PFNO,DeepONet,GNO}
DATASETS=${DATASETS:-"FlatVel_A CurveVel_A"}
# GNO at its practical ceiling: parameters live in the kernel MLP, so width and
# kernel_hidden are what move the count, and gradient checkpointing is what
# keeps the stencil activations small enough to share a GPU with another tenant.
EXTRA=${EXTRA:-"--gno-width 64 --gno-kernel-hidden 256 --gno-enc-layers 4 --gno-dec-layers 2 --gno-checkpoint"}
SUFFIX=${SUFFIX:-}

mkdir -p "$OUT" "$OUT/logs"
echo "epochs=$EPOCHS train_chunks=$TRAIN_CHUNKS models=$MODELS"
echo "datasets: $DATASETS"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

gpu=0
pids=""
for ds in $DATASETS; do
  tag=$(echo "$ds" | tr 'A-Z' 'a-z')$SUFFIX
  echo "launching $ds on GPU $gpu -> $OUT/$tag"
  CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" "$ROOT/wave/openfwi/train_openfwi.py" \
      --root "$DATA" --dataset "$ds" \
      --train-chunks "$TRAIN_CHUNKS" --val-chunks 1 \
      --models "$MODELS" --epochs "$EPOCHS" \
      --outdir "$OUT/$tag" $EXTRA \
    > "$OUT/logs/$tag.log" 2>&1 &
  pids="$pids $!"
  gpu=$(( (gpu + 1) % 2 ))
done

fail=0
for pid in $pids; do
  wait "$pid" || fail=1
done
echo "OPENFWI SWEEP DONE (fail=$fail)"
exit $fail
