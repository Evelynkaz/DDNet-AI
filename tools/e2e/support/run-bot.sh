#!/usr/bin/env bash
# What ddnet-ai-bot.service does, for the local e2e (tasks 5.12, 5.13, 5.15, 3.20b, 5.17; the same `--finish`, `--wb-smart`, `--no-selfkill=`, `--preinput` and `--search-threads` words
# as the unit): the bot with the validated environment the helper wrote, then the unit's ExecStopPost hook (`ddnet-ai launch exited`) with systemd's own EXIT_CODE / EXIT_STATUS variables. SIGTERM goes on to the bot.
set -u
: "${E2E_DIR:?}" "${E2E_BIN:?}"
DATA="$E2E_DIR/data"
# shellcheck source=/dev/null
source "$E2E_DIR/etc/bot-launch.env"
# `--wb-smart` and `--no-selfkill=` take the env file's values with no default, exactly like the unit (the helper always writes both lines;
# an empty value would fail here as it does in the unit).
# `$BOT_FLY_ARGS` is split at spaces and vanishes when empty, exactly as in the unit.
# shellcheck disable=SC2086
"$E2E_BIN" play --server "$BOT_SERVER" --name "$BOT_NAME" --brain "$BOT_BRAIN" --duration "$BOT_DURATION" \
  --hybrid-mirror "$BOT_HYBRID_MIRROR" --finish "${BOT_FINISH:-off}" \
  --wb-smart "$BOT_WB_SMART" --no-selfkill="$BOT_NO_SELFKILL" --window-model="${BOT_WINDOW_MODEL:-}" --preinput "${BOT_PREINPUT:-off}" --search-threads "${BOT_SEARCH_THREADS:-1}" \
  $BOT_FLY_ARGS --no-console --web-names --data-dir "$DATA" \
  --live-servers "$DATA/live-servers.toml" --report "$DATA/bot/last-report.json" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null' TERM INT
wait "$child"
code=$?
# A trapped signal interrupts the first `wait`: collect the real exit status.
wait "$child" 2>/dev/null
code2=$?
[[ $code2 -ne 127 ]] && code=$code2
EXIT_CODE=exited EXIT_STATUS=$code "$E2E_BIN" launch exited --data-dir "$DATA" --status-dir "$E2E_DIR/status" \
  --config "$E2E_DIR/none.toml" --env-file "$E2E_DIR/etc/bot-launch.env" --dropin "$E2E_DIR/etc/50-launch.conf" \
  --state "$E2E_DIR/var/state.json"
