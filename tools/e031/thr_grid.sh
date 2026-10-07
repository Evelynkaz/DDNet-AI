#!/usr/bin/env bash
# E-031 (task 8.7): the hook threshold of a readout arm tuned in the CLOSED loop on the TRAIN-VAL starts and games only (configs/train/e029-tune.toml has no holdout hall):
# the rate-matched threshold of the BC (matched to the teacher's start rate on all validation decisions) did not carry over to the post-freeze windows in E-029.
# Same weights every cell, only `thresholds.hook` changes. usage: thr_grid.sh <binary> <base bundle> <out dir>   (thresholds on stdin, one per line)
set -u
BIN=$1; BASE=$2; OUT=$3
cd "$(dirname "$0")/../.."
mkdir -p "$OUT/bundles"
while read -r t; do
  [ -z "$t" ] && continue
  name="t${t}"
  [ -f "$OUT/$name.json" ] && continue
  "$BIN" train set-thresholds --kind fly --bundle "$BASE" --out "$OUT/bundles/$name.bundle" --hook "$t" 2> /dev/null
  nice -n 5 "$BIN" train es eval --config configs/train/e029-tune.toml --brain "fly:$OUT/bundles/$name.bundle" --starts 700 --games 400 \
    --seed-base 9300000000 --out "$OUT/$name.json" --threads 3 > "$OUT/$name.log" 2>&1
  echo "$name done"
done
