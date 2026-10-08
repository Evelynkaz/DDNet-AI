#!/usr/bin/env bash
# Task 3.19 (D-116, E-034): the exploratory replication of `rope` and `finish` on fresh seeds 25001..25200 (3 threads). Output: ~/aiddnet/data/runs/E-034/fresh-<cell>.{txt,jsonl}
set -u
cd "$(dirname "$0")/../.."
out=~/aiddnet/data/runs/E-034
for c in 21 20 22; do
  target/release/examples/duel_stats --config configs/arena/d115-fresh-$c.toml --threads 3 --jsonl $out/fresh-$c.jsonl > $out/fresh-$c.txt 2>&1
done
echo finished > $out/fresh.done
