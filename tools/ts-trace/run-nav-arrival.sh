#!/usr/bin/env bash
# Task 4.2 acceptance criterion 3: the arrival-rate comparison with the TS navigator on three maps.
# Generates the TS side (`gen-nav-dump.mjs --section arrival|follow`) and runs the Rust side
# (`ddai-nav` tests/arrival_vs_ts.rs) on the same pairs. Needs node >= 24 and, for the follow section,
# `node_modules` linked into the checkout (the TS `bot.ts` imports the `teeworlds` package).
#   tools/ts-trace/run-nav-arrival.sh [pairs-per-mode]   (default 200)
# `DDAI_MAPS="clb"` picks maps; the TS dumps are reused unless `DDAI_REGEN=1`.
set -euo pipefail
cd "$(dirname "$0")/../.."
source "$HOME/.cargo/env"
PAIRS="${1:-200}"
M="$HOME/aiddnet/data/maps"
OUT="$HOME/aiddnet/data/traces/nav"
mkdir -p "$OUT"
declare -A MAPS=(
  [clb]="$M/copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map"
  [blmapchill]="$M/cache/BlmapChill_c902b2da07291266ab201054e6b6c28abd31e10b099fa5b1066b2f5a88f98240.map"
  [chillblock5]="$M/chillblock5/ChillBlock5.map"
)
# `DDAI_ARRIVAL_NOFRONT=1 tools/ts-trace/run-nav-arrival.sh` strips the front layer from the live side: the TS
# world ignores front-layer freeze, so BlmapChill and ChillBlock5 are compared both ways.
for k in ${DDAI_MAPS:-clb blmapchill chillblock5}; do
  echo "=== $k"
  [ -s "$OUT/$k-arrival.jsonl" ] && [ -z "${DDAI_REGEN:-}" ] || node tools/ts-trace/gen-nav-dump.mjs --map "${MAPS[$k]}" --seed 7 --section arrival --pairs "$PAIRS" --out "$OUT/$k-arrival.jsonl"
  [ -s "$OUT/$k-follow.jsonl" ] && [ -z "${DDAI_REGEN:-}" ] || node tools/ts-trace/gen-nav-dump.mjs --map "${MAPS[$k]}" --seed 9 --section follow --pairs "$PAIRS" --out "$OUT/$k-follow.jsonl"
  DDAI_ARRIVAL_DUMP="$OUT/$k-arrival.jsonl" cargo test -p ddai-nav --release --test arrival_vs_ts the_rust -- --ignored --nocapture 2>&1 | grep -E "^freeze|^mode|^/home|McNemar|panicked|test result" || true
  DDAI_ARRIVAL_DUMP="$OUT/$k-follow.jsonl" cargo test -p ddai-nav --release --test arrival_vs_ts following -- --ignored --nocapture 2>&1 | grep -E "^follow|McNemar|panicked|test result" || true
done
