#!/usr/bin/env bash
# Task 3.19 (D-116, E-034): the pre-registered confirmation, cell by cell (3 threads). Output: ~/aiddnet/data/runs/E-034/confirm-<cell>.{txt,jsonl}
set -u
cd "$(dirname "$0")/../.."
out=~/aiddnet/data/runs/E-034
for c in 21 20 22; do
  target/release/examples/duel_stats --config configs/arena/d115-confirm-$c.toml --threads 3 --jsonl $out/confirm-$c.jsonl > $out/confirm-$c.txt 2>&1
done
echo finished > $out/confirm.done
