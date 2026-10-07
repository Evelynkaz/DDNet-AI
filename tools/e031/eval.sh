#!/usr/bin/env bash
# E-031 (task 8.7): the paired closed-loop evaluation of E-029 (fresh seeds 9.8e9, 700 starts = all 695 holdout + 208 train-val starts of bank-v2, 400 games per set):
# `train es eval` (V held, own freeze, first freeze vs scripted) and `train es hook-eval` (start rate by hook state, opening, first press, aim at the throw).
#   usage: eval.sh <binary> <out dir> name=bundle ...
set -u
BIN=$1; OUT=$2; shift 2
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
for spec in "$@"; do
  name=${spec%%=*}; b=${spec#*=}
  [ -f "$OUT/$name.json" ] || nice -n 5 "$BIN" train es eval --config configs/train/e029-eval.toml --brain "fly:$b" --starts 700 --games 400 --seed-base 9800000000 --out "$OUT/$name.json" --threads 3 > "$OUT/$name.log" 2>&1
  [ -f "$OUT/hook-$name.json" ] || nice -n 5 "$BIN" train es hook-eval --config configs/train/e029-eval.toml --brain "fly:$b" --starts 700 --out "$OUT/hook-$name.json" --threads 3 > "$OUT/hook-$name.log" 2>&1
  echo "$name done"
done
