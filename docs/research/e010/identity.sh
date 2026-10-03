#!/bin/bash
# bit-identity of base vs new binary on the E-007 final matrix (24 games per condition, WALL conditions removed)
cd ~/aiddnet/wt/task-3.6
NEW=${1:-v8}
for c in strength threat crowd; do
  for b in base $NEW; do
    echo "== $c $b $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
    ~/aiddnet/data/scratch/e010/ddnet-ai-$b arena run --config ~/aiddnet/data/runs/E-010/cfg/identity-$c.toml --games 24 --out ~/aiddnet/data/runs/E-010/identity/$b-$c --threads 3 --arenas-dir configs/arenas >/dev/null 2>&1
  done
  python3 ~/aiddnet/data/runs/E-010/cmp.py ~/aiddnet/data/runs/E-010/identity/base-$c ~/aiddnet/data/runs/E-010/identity/$NEW-$c
done
