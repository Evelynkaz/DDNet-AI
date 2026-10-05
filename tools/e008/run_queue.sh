#!/usr/bin/env bash
# Runs E-008 configs one after another (2 threads, resumable), evaluating each when it finishes. The queue is a
# file (one config name per line, without .toml, from configs/train/); the first line is taken when a job starts, so
# the queue can be edited while it runs.
#   tools/e008/run_queue.sh [queue-file]     (default ~/aiddnet/data/logs/8.2b/queue.txt; THREADS=2 by default,
#   LOGTAG=queue by default: the second queue of the run uses THREADS=1 LOGTAG=queue2)
# A finished run (status.json phase "done") is not trained again; an interrupted one resumes from state.bin.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
bin=${DDNET_AI:-./target/release/ddnet-ai}
log=~/aiddnet/data/logs/8.2b
queue=${1:-$log/queue.txt}
threads=${THREADS:-2}
tag=${LOGTAG:-queue}
mkdir -p "$log"
while true; do
  name=$(grep -v '^\s*#' "$queue" 2>/dev/null | grep -v '^\s*$' | head -1 || true)
  [ -n "$name" ] || { echo "$(date +%T) queue empty" >> "$log/$tag.log"; break; }
  python3 tools/e008/pop_queue.py "$queue" "$name"
  run=~/aiddnet/data/runs/E-008/$name
  if ! grep -q '"phase":"done"' "$run/status.json" 2>/dev/null; then
    echo "$(date +%T) train $name" >> "$log/$tag.log"
    ( ulimit -v 14000000; $bin train run --config configs/train/$name.toml --threads $threads ) >> "$log/$name.log" 2>&1 \
      || { echo "$(date +%T) FAILED $name" >> "$log/$tag.log"; continue; }
  fi
  echo "$(date +%T) eval $name" >> "$log/$tag.log"
  tools/e008/eval_run.sh "$name" $threads >> "$log/$name-eval.log" 2>&1 || echo "$(date +%T) EVAL FAILED $name" >> "$log/$tag.log"
  echo "$(date +%T) done $name" >> "$log/$tag.log"
done
