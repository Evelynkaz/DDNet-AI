#!/usr/bin/env bash
# F2 (PLAN §0: the fly >= 40% against the planner): the standalone brain of a run's selected checkpoint against the
# planner on the three halls (E-005 conditions and seeds, 1000 games each). Resumable.
#   DDNET_AI=<binary> tools/e008/eval_vs_planner.sh <run-name> [threads]
set -euo pipefail
name=$1; threads=${2:-2}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
bin=${DDNET_AI:-./target/release/ddnet-ai}
out=~/aiddnet/data/runs/E-008/eval/$name
case $name in *fly*) kind=fly ;; *mlpw*) kind=mlp ;; *) exit 1 ;; esac
[ -f "$out/arena-planner/summary.json" ] || $bin arena run --config configs/arena/e005-eval.toml --filter "vs planner" \
  --brain "$kind:$HOME/aiddnet/data/runs/E-008/$name/checkpoints/selected.bundle" --out "$out/arena-planner" --threads "$threads" \
  > "$out/arena-planner.stdout" 2> "$out/arena-planner.log"
echo "$name vs planner done"
