#!/usr/bin/env bash
# E-029 (task 8.6): the paired final evaluation on fresh seeds (eval seed base 9.8e9; 700 starts = all 695 holdout + 208 train-val starts of bank-v2, 400 games per set).
#   usage: final_eval.sh <binary> <out dir> name=bundle ...    (name "planner" = the planner)
# `train es eval` (held by class, own freeze, first freeze vs scripted) and `train es hook-eval` (start rate by hook state, opening, first press, aim at the throw).
set -u
BIN=$1; OUT=$2; shift 2
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
for spec in "$@"; do
  name=${spec%%=*}; b=${spec#*=}
  if [ "$name" = planner ]; then brain=planner; else brain="fly:$b"; fi
  [ -f "$OUT/$name.json" ] || nice -n 5 "$BIN" train es eval --config configs/train/e029-eval.toml --brain "$brain" --starts 700 --games 400 --seed-base 9800000000 --out "$OUT/$name.json" --threads 3 > "$OUT/$name.log" 2>&1
  [ -f "$OUT/hook-$name.json" ] || nice -n 5 "$BIN" train es hook-eval --config configs/train/e029-eval.toml --brain "$brain" --starts 700 --out "$OUT/hook-$name.json" --threads 3 > "$OUT/hook-$name.log" 2>&1
  # Agreement with the planner at the same states (shadow run only; the fly plays, the planner labels): V starts of the holdout and of the train-val part.
  if [ "$name" != planner ]; then
    for set in holdout train-val; do
      [ -f "$OUT/agree-$name-$set.json" ] || nice -n 5 "$BIN" train es critical --config configs/train/e029-eval.toml --fly "fly:$b" --direction forward --class V --set "$set" \
        --max-starts 250 --single-first 0 --single-late-stride 100000 --component-decisions 0 --no-windows --out "$OUT/agree-$name-$set.json" --threads 3 > "$OUT/agree-$name-$set.log" 2>&1
    done
  fi
  echo "$name done"
done
