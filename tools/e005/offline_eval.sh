#!/usr/bin/env bash
# Offline per-head metrics (with top-2 direction and joint accuracy) of the final bundle of each run.
#   tools/e005/offline_eval.sh e005-fly e005-mlp-w ...
set -euo pipefail
cd ~/aiddnet/wt/task-8.2
out=~/aiddnet/data/runs/E-005/offline
for n in "$@"; do
  ./target/release/ddnet-ai train eval --config configs/train/$n.toml \
    --bundle ~/aiddnet/data/runs/E-005/$n/checkpoints/final.bundle > "$out/$n.json" 2> "$out/$n.log"
  echo "$n done"
done
