#!/usr/bin/env bash
# Task 3.11: one live-timing run on a PRIVATE DDNet server, with the server-side truth.
#
#   DDAI_T_LABEL=base DDAI_T_SECS=90 tools/e2e/live_timing.sh
#
# Needs a private server (never the shared dev server on 8303): `sv_tee_historian 1`, econ on loopback. Defaults match the
# scratch server of task 3.11 (game port 8453, econ 8454, password in $T_ECON_PW_FILE). The map is reloaded around the run
# (`reload` over econ) so the server closes the teehistorian that covers exactly this run; `live_timing_analyze.py` then joins it
# with the bot's input trace.
#
# Environment: DDAI_T_* are read by the harness (crates/ddai-bot/tests/e2e_live_timing.rs); T_SRV (server directory containing
# teehistorian/), T_ECON_PORT, T_ECON_PW_FILE, DDAI_T_OUT (default ~/aiddnet/data/scratch/task-3.11/runs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
S="${T_SCRATCH:-$HOME/aiddnet/data/scratch/task-3.11}"
SRV="${T_SRV:-$S/srv}"
PORT="${T_ECON_PORT:-8454}"
PW_FILE="${T_ECON_PW_FILE:-$S/econ.pw}"
OUT="${DDAI_T_OUT:-$S/runs}"
LABEL="${DDAI_T_LABEL:-run}"
export DDAI_T_OUT="$OUT" DDAI_T_LABEL="$LABEL" DDAI_E2E=1
export DDAI_T_SERVER="${DDAI_T_SERVER:-127.0.0.1:8453}"
if [[ "$DDAI_T_SERVER" == *":8303" ]]; then echo "refusing: 8303 is the shared dev server" >&2; exit 2; fi
mkdir -p "$OUT"
if [[ -n "${T_RUN_PREFIX:-}" ]]; then
  # A transient unit does not inherit this shell's environment: pass what the harness reads.
  T_RUN_PREFIX="$T_RUN_PREFIX --setenv=DDAI_E2E=1 --setenv=DDAI_T_OUT=$OUT --setenv=DDAI_T_LABEL=$LABEL --setenv=DDAI_T_SERVER=$DDAI_T_SERVER --setenv=DDAI_T_FIX=${DDAI_T_FIX:-}"
fi
econ() { python3 "$ROOT/tools/ddnet-server/econ.py" --port "$PORT" --password "$(cat "$PW_FILE")" "$@" >/dev/null; }

newest() { ls -t "$SRV/teehistorian/"*.teehistorian | head -1; }

econ reload
sleep 3
BEFORE="$(newest)"
set +e
if [[ -n "${T_TEST_BIN:-}" ]]; then
  # A saved test binary (A/B runs of different builds): `T_TEST_BIN=.../live_timing_base`.
  # `T_RUN_PREFIX` (e.g. `sudo systemd-run --wait --collect --pipe -p User=ubuntu -p Nice=-10 --setenv=...`): the run as a transient system unit.
  # shellcheck disable=SC2086
  ${T_RUN_PREFIX:-} "$T_TEST_BIN" --ignored --nocapture --test-threads=1 live_timing_run >"$OUT/$LABEL.log" 2>&1
else
  ( cd "$ROOT" && source ~/.cargo/env && cargo test --release --locked -p ddai-bot --test e2e_live_timing -- --ignored --nocapture --test-threads=1 ) \
    >"$OUT/$LABEL.log" 2>&1
fi
RC=$?
set -e
econ reload
sleep 3
echo "run exit: $RC (log $OUT/$LABEL.log)"
[[ $RC -eq 0 ]] || { tail -30 "$OUT/$LABEL.log"; exit $RC; }
python3 "$ROOT/tools/e2e/live_timing_analyze.py" "$OUT/$LABEL.trace.jsonl" "$BEFORE" --json "$OUT/$LABEL.server.json" >"$OUT/$LABEL.server.txt"
cat "$OUT/$LABEL.server.txt"
