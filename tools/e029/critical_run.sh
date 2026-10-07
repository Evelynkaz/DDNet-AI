#!/usr/bin/env bash
# E-029 (task 8.6): the critical-decision analysis on V starts of the training halls, both directions (`train es critical`).
#   usage: critical_run.sh <binary> <out dir> <fly bundle>
set -u
BIN=$1; OUT=$2; FLY=$3
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
[ -f "$OUT/forward.json" ] || "$BIN" train es critical --config configs/train/e029-tune.toml --fly "fly:$FLY" --direction forward --max-starts 150 \
  --out "$OUT/forward.json" --threads 3 > "$OUT/forward.log" 2>&1
[ -f "$OUT/reverse.json" ] || "$BIN" train es critical --config configs/train/e029-tune.toml --fly "fly:$FLY" --direction reverse --max-starts 60 \
  --single-first 24 --single-late-stride 8 --component-decisions 4 --no-windows --out "$OUT/reverse.json" --threads 3 > "$OUT/reverse.log" 2>&1
echo critical done
