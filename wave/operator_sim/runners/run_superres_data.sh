#!/bin/bash
# Datasets for the zero-shot super-resolution arm: identical materials to the
# 64x64 set (same seed, same sampler -- the rng draws are scalars and do not
# depend on nx), solved to convergence and written out on other grids.
set -u
cd ~/sciml_neurips2026_exp/wave/operator_sim || exit 1
PY=~/miniconda3/bin/python
DATA=~/sciml_neurips2026_exp/exp/superres_data
LOG=~/sciml_neurips2026_exp/logs
mkdir -p "$DATA" "$LOG"

gen () {  # $1 nx  $2 nt
  $PY generate_dataset.py --num-samples 512 --nx "$1" --nt "$2" --refine 8 \
      --seed 42 --out "$DATA/sr_nx$1_nt$2.npz" >> "$LOG/superres_data.log" 2>&1
  echo "nx=$1 nt=$2 exit=$?" >> "$DATA/progress.txt"
}

: > "$DATA/progress.txt"
: > "$LOG/superres_data.log"
gen 32 64
gen 48 64
gen 96 64
gen 128 64
gen 128 128
gen 64 64        # control: must reproduce the training set bit-for-bit
echo "SR DATA DONE" >> "$DATA/progress.txt"
