#!/usr/bin/env bash
# E-035 (task 8.8): one pilot arm end to end: write its config, BC from scratch, then the paired closed-loop evaluation (`eval.sh`) of its final checkpoint.
#   usage: pilot.sh <binary> <arm> ...      (BC_STEPS, default 1500; runs the arms one after the other on 3 threads)
# Logs and results: ~/aiddnet/data/runs/E-035/{<arm>.log,<arm>/,final/}.
set -u
BIN=$1; shift
cd "$(dirname "$0")/../.."
R=~/aiddnet/data/runs/E-035
mkdir -p "$R/final" "$R/data"
for arm in "$@"; do
  BC_STEPS=${BC_STEPS:-1500} tools/e035/gen_configs.sh "$arm" > /dev/null
  if [ ! -f "$R/$arm.done" ]; then
    nice -n 5 "$BIN" train run --config configs/train/e035-$arm.toml > "$R/$arm.log" 2>&1 && touch "$R/$arm.done"
  fi
  [ -f "$R/$arm.done" ] && tools/e035/eval.sh "$BIN" "$R/final" "$arm=$R/$arm/checkpoints/final.bundle"
  echo "$arm finished"
done
