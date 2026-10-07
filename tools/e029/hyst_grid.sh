#!/usr/bin/env bash
# E-029 (task 8.6), step 1: the tuning grid of the hysteresis decode of the EXISTING hook head, on the TRAIN-VAL starts and games only
# (configs/train/e029-tune.toml has no holdout arena). Same weights every cell: only `hook_decode` changes (`train set-thresholds --hook-hi --hook-lo`).
#   usage: hyst_grid.sh <ddnet-ai binary> <base bundle> <out dir> [threads]  (cells on stdin, "hi lo" per line)
set -u
BIN=$1; BASE=$2; OUT=$3; T=${4:-3}
cd "$(dirname "$0")/../.."
mkdir -p "$OUT/bundles"
while read -r hi lo; do
  [ -z "$hi" ] && continue
  name="h${hi}-l${lo}"
  [ -f "$OUT/$name.json" ] && continue
  "$BIN" train set-thresholds --kind fly --bundle "$BASE" --out "$OUT/bundles/$name.bundle" --hook-hi "$hi" --hook-lo "$lo" 2> /dev/null
  nice -n 5 "$BIN" train es eval --config configs/train/e029-tune.toml --brain "fly:$OUT/bundles/$name.bundle" --starts 700 --games 400 \
    --seed-base 9300000000 --out "$OUT/$name.json" --threads "$T" > "$OUT/$name.log" 2>&1
  echo "$name done"
done
