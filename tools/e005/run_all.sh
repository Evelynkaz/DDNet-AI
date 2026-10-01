#!/usr/bin/env bash
# E-005: trains and evaluates the listed models (default: all seven), one after another. THREADS (default 6) and QUEUE (log suffix) can be set to run two queues side by side.
# Each `train run` is resumable: rerun this script after an interruption.
set -uo pipefail
cd ~/aiddnet/wt/task-8.2
log=~/aiddnet/data/logs/8.2
threads=${THREADS:-6}
q=${QUEUE:-all}
for name in ${@:-e005-fly e005-mlp-s e005-gru-s e005-mlp-w e005-gru-w e005-fly-noclb e005-mlp-w-noclb}; do
  ./target/release/ddnet-ai train run --config configs/train/$name.toml --threads "$threads" >> "$log/$name.log" 2>&1 || { echo "$name: train failed" >> "$log/run_all-$q.log"; continue; }
  case $name in
    e005-fly*) spec=fly ;; e005-mlp*) spec=mlp ;; e005-gru*) spec=gru ;;
  esac
  tools/e005/eval_all.sh "${name#e005-}" "$spec:$HOME/aiddnet/data/runs/E-005/$name/checkpoints/final.bundle" "$threads" >> "$log/run_all-$q.log" 2>&1
  echo "$name: done $(date)" >> "$log/run_all-$q.log"
done
