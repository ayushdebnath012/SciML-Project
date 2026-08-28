#!/bin/bash
# Run-to-run variation at the published configuration.
# --split-seed is pinned at 42 so the held-out set never moves; only weight
# init and batch order vary. Without that, a "seed" would change which samples
# are held out and the spread would mix two different things.
set -u
cd ~/sciml_neurips2026_exp/wave/operator_sim || exit 1
PY=~/miniconda3/bin/python
DATA=~/sciml_neurips2026_exp/operator_data
OUT=~/sciml_neurips2026_exp/exp/seeds
LOG=~/sciml_neurips2026_exp/logs
mkdir -p "$OUT" "$LOG"

run () {  # $1 arm  $2 init-seed  $3 gpu
  case $1 in
    synthetic_r8) ds=wave_operator_fixedic_r8_n512_nx64_nt64_t1_seed42 ;;
    *)            ds=wave_operator_$1_n512_nx64_nt64_t1_seed42 ;;
  esac
  CUDA_VISIBLE_DEVICES=$3 $PY train_operators.py \
    --data "$DATA/$ds.npz" --epochs 400 \
    --split-seed 42 --init-seed "$2" \
    --don-latent 256 --don-hidden 512 \
    --outdir "$OUT/$1_s$2" > "$LOG/seeds_$1_s$2.log" 2>&1
  echo "$1 seed=$2 exit=$?" >> "$OUT/progress.txt"
}

: > "$OUT/progress.txt"
( for s in 42 1 2 3; do run synthetic_r8 "$s" 0; done ) &
( for s in 42 1 2 3; do run marmousi     "$s" 1; done ) &
wait
echo "SEEDS DONE" >> "$OUT/progress.txt"
