#!/bin/bash
# Parameter-matched grid: each architecture sized to the same real-parameter
# budget. Widths come from size_models.py; see exp/param_budgets.json.
#
#   0.9M  FNO width 12   PFNO width 15   DeepONet 512/256  (DeepONet's own size)
#   2.4M  FNO width 19   PFNO width 24   DeepONet 862/431  (PFNO's own size)
#  14.8M  FNO width 48   PFNO width 60   DeepONet 2188/1094 (FNO's own size)
set -u
cd ~/sciml_neurips2026_exp/wave/operator_sim || exit 1
PY=~/miniconda3/bin/python
DATA=~/sciml_neurips2026_exp/operator_data
OUT=~/sciml_neurips2026_exp/exp/matched
LOG=~/sciml_neurips2026_exp/logs
mkdir -p "$OUT" "$LOG"

run () {  # $1 arm  $2 budget-tag  $3 fno-width  $4 pfno-width  $5 don-hidden  $6 don-latent  $7 gpu
  case $1 in
    synthetic_r8) ds=wave_operator_fixedic_r8_n512_nx64_nt64_t1_seed42 ;;
    *)            ds=wave_operator_$1_n512_nx64_nt64_t1_seed42 ;;
  esac
  CUDA_VISIBLE_DEVICES=$7 $PY train_operators.py \
    --data "$DATA/$ds.npz" --epochs 400 \
    --fno-width "$3" --pfno-width "$4" --don-hidden "$5" --don-latent "$6" \
    --outdir "$OUT/$1_$2" > "$LOG/matched_$1_$2.log" 2>&1
  echo "$1 $2 exit=$?" >> "$OUT/progress.txt"
}

: > "$OUT/progress.txt"
(
  run synthetic_r8 b0p9  12 15 512  256  0
  run synthetic_r8 b2p4  19 24 862  431  0
  run synthetic_r8 b14p8 48 60 2188 1094 0
) &
(
  run marmousi     b0p9  12 15 512  256  1
  run marmousi     b2p4  19 24 862  431  1
  run marmousi     b14p8 48 60 2188 1094 1
) &
wait
echo "MATCHED DONE" >> "$OUT/progress.txt"
