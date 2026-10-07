#!/usr/bin/env bash
# E-029 (task 8.6), step 3: the two BC arms (control = legacy hook head, intent = two hazards by the latch), one after the other, 3 threads.
#   usage: bc_arms.sh <binary>
set -u
BIN=$1
cd "$(dirname "$0")/../.."
for arm in legacy intent; do
  [ -f ~/aiddnet/data/runs/E-029/e029-bc-$arm/checkpoints/final.bundle ] && continue
  "$BIN" train run --config configs/train/e029-bc-$arm.toml > ~/aiddnet/data/runs/E-029/bc-$arm.log 2>&1
done
echo bc done
