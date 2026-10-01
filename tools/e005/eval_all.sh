#!/usr/bin/env bash
# E-005 evaluation of one brain: the 14-condition arena matrix (1000 games each) and the T1-T18
# technique scenarios (100 trials each).
#   tools/e005/eval_all.sh <label> <brain-spec> [threads]
# e.g. tools/e005/eval_all.sh fly fly:~/aiddnet/data/runs/E-005/e005-fly/checkpoints/final.bundle
set -euo pipefail
label=$1
spec=$2
threads=${3:-6}
out=~/aiddnet/data/runs/E-005/eval/$label
mkdir -p "$out"
cd ~/aiddnet/wt/task-8.2
bin=./target/release/ddnet-ai
$bin arena run --config configs/arena/e005-eval.toml --brain "$spec" --out "$out/arena" --threads "$threads" \
  > "$out/arena.stdout" 2> "$out/arena.log"
$bin arena scenarios --brain "$spec" --trials 100 --out "$out/scenarios" > "$out/scenarios.stdout" 2> "$out/scenarios.log"
echo "$label done"
