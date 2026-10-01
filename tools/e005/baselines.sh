#!/usr/bin/env bash
# E-005 baselines (D-059): `idle` (a brain that does nothing) and `scripted` (scripted vs scripted) on the same
# evaluation matrix, 1000 games per condition. Both are needed in every arena table: the scripted bot freezes
# itself so often that doing nothing wins most games on some maps (W/(W+L+D) 74% on chillblock5-ruler).
#   tools/e005/baselines.sh [threads] [filter]
set -euo pipefail
threads=${1:-3}
filter=${2:-"vs scripted"}
cd ~/aiddnet/wt/task-8.2
bin=${DDNET_AI:-./target/release/ddnet-ai}
for b in idle scripted; do
  out=~/aiddnet/data/runs/E-005/eval/$b
  mkdir -p "$out"
  $bin arena run --config configs/arena/e005-eval.toml --brain $b --filter "$filter" --out "$out/arena" --threads "$threads" \
    > "$out/arena.stdout" 2> "$out/arena.log"
  echo "$b done"
done
