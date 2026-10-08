#!/usr/bin/env bash
# Task 3.16 (D-115): the input-lag matrix on a PRIVATE DDNet server through the delay relay of the 3.11 harness
# (crates/ddai-bot/tests/e2e_live_timing.rs): fixed prediction margins x hybrid search budgets, the slot statistics of each cell.
#
#   T_TEST_BIN=target/release/deps/e2e_live_timing-<hash> tools/e2e/lag_shave.sh "4 6 8 10" "2 3 4" 2 60
#                                                         margins (ms; "a" = adaptive)  budgets (ms)  rounds  seconds
#
# Needs a private server (never the shared dev server on 8303): `T_SERVER` (default 127.0.0.1:8433), map "Copy Love Box". Cells run interleaved
# (round 1 of every cell, then round 2, ...). Output: $OUT/<label>.json per cell (label m<margin>-b<budget>-r<round>), then `lag_shave_table.py`.
# Environment: T_SCRATCH (map cache etc., default ~/aiddnet/data/scratch/task-3.16), OUT (default $T_SCRATCH/runs), T_RTT_US (one-way delay, default
# 12500 = RTT 25 ms), T_JITTER_US (default 2500), T_BRAIN (hybrid, or scripted for a decision that costs ~nothing: the pure slot phase), T_FIX (default precise,kind = the production setting of D-101), T_OPP (scripted opponents, default 1).
set -euo pipefail
BIN="${T_TEST_BIN:?set T_TEST_BIN to the e2e_live_timing test binary}"
MARGINS="${1:-4 6 8 10}"; BUDGETS="${2:-2 3 4}"; ROUNDS="${3:-2}"; SECS="${4:-60}"
S="${T_SCRATCH:-$HOME/aiddnet/data/scratch/task-3.16}"; OUT="${OUT:-$S/runs}"; SERVER="${T_SERVER:-127.0.0.1:8433}"
if [[ "$SERVER" == *":8303" ]]; then echo "refusing: 8303 is the shared dev server" >&2; exit 2; fi
mkdir -p "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  for m in $MARGINS; do
    for b in $BUDGETS; do
      label="m${m}-b${b}-r${r}"
      [[ -f "$OUT/$label.json" ]] && continue
      env_m=(); [[ "$m" != "a" ]] && env_m=(DDAI_T_MARGIN="$m")
      echo "== $label $(date +%T) load $(cut -d' ' -f1 /proc/loadavg)"
      env DDAI_E2E=1 DDAI_T_SERVER="$SERVER" DDAI_T_SCRATCH="$S" DDAI_T_OUT="$OUT" DDAI_T_LABEL="$label" DDAI_T_SECS="$SECS" \
        DDAI_T_SEED=$((20 + r)) DDAI_T_FIX="${T_FIX:-precise,kind}" DDAI_T_BUDGET="$b" DDAI_T_BRAIN="${T_BRAIN:-hybrid}" DDAI_T_OPPONENTS="${T_OPP:-1}" \
        DDAI_T_DELAY_US="${T_RTT_US:-12500}" DDAI_T_JITTER_US="${T_JITTER_US:-2500}" "${env_m[@]}" \
        "$BIN" --ignored --nocapture --test-threads=1 live_timing_run >"$OUT/$label.log" 2>&1 || { echo "run failed: $label"; tail -5 "$OUT/$label.log"; }
      sleep 22   # the server's connection throttle and the previous bot's ghost
    done
  done
done
