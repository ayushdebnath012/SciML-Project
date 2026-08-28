#!/bin/bash
# SubsurfaceGen field-scale forward benchmark.
#
#   bash wave/subsurfacegen/runners/run_ssgen.sh [EPOCHS] [MODELS] [GPU]
#
# One job, all four models in sequence. Unlike the OpenFWI sweep this does not
# split across GPUs: each model here peaks at 8-13 GB and the on-GPU split
# cache is another 8 GB, so two concurrent jobs would be a poor neighbour on a
# box that is already carrying other tenants.
set -u

ROOT=${ROOT:-$HOME/sciml_wave_sim}
DATA=${DATA:-$HOME/ssgen_data}
OUT=${OUT:-$HOME/ssgen_results}
PY=${PY:-$HOME/miniconda3/bin/python}

EPOCHS=${1:-80}
MODELS=${2:-FNO,PFNO,DeepONet,GNO}
GPU=${3:-1}
TAG=${TAG:-ssgen}

# Per-architecture sizing, measured at this grid (velocity 309x500 ->
# gathers 5x572x1000, batch 2) rather than carried over from OpenFWI:
#
#   FNO       width 24            5.40 M real   0.2 min/epoch   1.2 GB
#   PFNO      220 of 287 bins     2.19 M real   2.1 min/epoch  13.0 GB
#   DeepONet  hidden 256         39.77 M real   0.5 min/epoch  13.1 GB
#   GNO       width 24            0.11 M real   3.2 min/epoch   8.5 GB
#
# PFNO needs 220 branches here, not OpenFWI's 64: the band a per-frequency
# model must cover is f_max * T, and an 8 s record at 25 Hz is 4x the 1 s
# record at 15 Hz. Keeping 64 bins would cost 98 % relative L2 on this data.
EXTRA=${EXTRA:-"--t-latent 286 --fno-width 24 --fno-modes-t 32 \
  --pfno-freqs 220 --pfno-width 4 --pfno-modes 8 --pfno-layers 2 \
  --don-hidden 256 --don-latent 128 \
  --gno-width 24 --gno-kernel-hidden 96 --gno-enc-layers 3 --gno-dec-layers 2 \
  --gno-t-latent 286 --gno-checkpoint"}

mkdir -p "$OUT/logs"
echo "epochs=$EPOCHS models=$MODELS gpu=$GPU"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

CUDA_VISIBLE_DEVICES=$GPU \
  "$PY" -u "$ROOT/wave/openfwi/train_openfwi.py" \
    --meta --root "$DATA" --dataset SubsurfaceGen --norm zscore \
    --train-chunks 0 --val-chunks 0 --ood-chunks 0 \
    --models "$MODELS" --epochs "$EPOCHS" --batch-size 2 \
    --outdir "$OUT/$TAG" $EXTRA \
  > "$OUT/logs/$TAG.log" 2>&1
rc=$?
echo "SSGEN RUN DONE rc=$rc"
exit $rc
