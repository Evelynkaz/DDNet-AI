#!/usr/bin/env bash
# E-031 (task 8.7): the readout BC arms one after the other (3 threads). usage: bc_arms.sh <binary> <arm> ...
set -u
BIN=$1; shift
cd "$(dirname "$0")/../.."
for arm in "$@"; do
  [ -f ~/aiddnet/data/runs/E-031/ro-$arm/checkpoints/final.bundle ] && [ -f ~/aiddnet/data/runs/E-031/ro-$arm.done ] && continue
  "$BIN" train run --config configs/train/e031-ro-$arm.toml > ~/aiddnet/data/runs/E-031/ro-$arm.log 2>&1 && touch ~/aiddnet/data/runs/E-031/ro-$arm.done
done
echo bc done
