#!/usr/bin/env bash
# Task 3.19 (D-116, E-034), review F2: base and finish at 2.1 and at 1.25 us, seeds 27001.., 3 cells. Output: ~/aiddnet/data/runs/E-034/quiet-<cell>.{txt,jsonl}
set -u
cd "$(dirname "$0")/../.."
out=~/aiddnet/data/runs/E-034
for c in 21 20 22; do
  target/release/examples/duel_stats --config configs/arena/d115-quiet-$c.toml --threads 3 --jsonl $out/quiet-$c.jsonl > $out/quiet-$c.txt 2>&1
done
echo finished > $out/quiet.done
