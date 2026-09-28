#!/usr/bin/env bash
# tools/e2e/session.sh — task 2.3 acceptance criterion 6: end-to-end scenarios a-h for the DDNet
# client session (`ddai-client`/`ddnet-ai play`) against the LOCAL DDNet 20.1 server
# (`ddnet-local.service`, 127.0.0.1:8303, econ 127.0.0.1:8304) — see CLAUDE.md's live-play policy:
# this script must never touch any address other than 127.0.0.1.
#
# Usage: tools/e2e/session.sh [scenario...]   (default: all of a b c d d2 e f g h)
#
# (d) and (d2) both exercise "the server goes away and comes back", via the two distinct
# detection paths review finding F2 separated: (d) is a graceful `systemctl restart`
# (`NETMSG_CLOSE("Server shutdown")`, a peer-close the driver's policy reconnects from); (d2) is
# an unclean SIGKILL (no close message at all — detected purely via `Connection`'s own silence
# timeout, since an ICMP-driven fast-fail is deliberately ignored, see that finding).
#
# Each scenario prints exactly one "PASS <letter>: ..." or "FAIL <letter>: ..." line to stdout (in
# addition to progress output) and writes its own log(s) under
# ~/aiddnet/data/logs/e2e-2.3/<timestamp>/. Restores the server to map "Copy Love Box" on exit,
# always (even on failure/Ctrl-C), per this task's constraints.
#
# Scenario (e) (redirect) does not touch the real server at all — DDNet 20.1 has no admin command
# that triggers `NETMSG_REDIRECT` (`RedirectClient` is only ever called internally — see
# `server.cpp:544-566`); per the task spec it is instead proven by a small local UDP test double
# written in Rust (`crates/ddai-client/tests/redirect_double.rs`), run here via `cargo test`.
# Scenario (g) (chat never sent) is likewise proven by the Rust test suite, not this script's own
# actions — run here the same way, so this script's PASS/FAIL g reflects real, current test
# results rather than being hard-coded.

set -uo pipefail

# shellcheck disable=SC1090
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DDAI_DATA_DIR:-$HOME/aiddnet/data}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$DATA_DIR/logs/e2e-2.3/$STAMP"
mkdir -p "$RUN_DIR"

ECON=(python3 "$REPO_ROOT/tools/ddnet-server/econ.py")
BOT="$REPO_ROOT/target/debug/ddnet-ai"
ORIGINAL_MAP="Copy Love Box"

# DDNet 0.6's `MAX_NAME_LENGTH` is 16 bytes (`protocol.h:109`) — the server truncates anything
# longer *before* `econ status <filter>`'s substring match ever runs, so a filter longer than
# what actually survives truncation can never match (this bit a first draft of this script: a
# full ISO-timestamp-based name like `e2e-a-20260928T000920Z` is 22 bytes, silently truncated
# server-side to 16, so searching for the full 22-byte string always came back empty even though
# the bot really had joined — see the crate's BUILD REPORT for the live repro). Kept short and
# unique enough for one script run: `e2e<letter><3 digits of the current second-of-day>`.
short_name() {
    printf 'e2e%s%03d' "$1" "$(($(date +%s) % 1000))"
}

declare -a RESULTS=()
PASS_COUNT=0
FAIL_COUNT=0

pass() {
    echo "PASS $1: $2"
    RESULTS+=("PASS $1: $2")
    PASS_COUNT=$((PASS_COUNT + 1))
}
fail() {
    echo "FAIL $1: $2"
    RESULTS+=("FAIL $1: $2")
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

restore_map() {
    echo "restoring map to '$ORIGINAL_MAP' ..."
    "${ECON[@]}" change_map "$ORIGINAL_MAP" >"$RUN_DIR/restore-map.log" 2>&1
    sleep 1
    systemctl is-active --quiet ddnet-local.service || sudo systemctl start ddnet-local.service
}
trap restore_map EXIT

status_with_name() {
    "${ECON[@]}" status "$1" 2>&1
}

wait_for_port_free_of_our_bot() {
    # best-effort: give the server a moment to notice a disconnect before the next scenario reuses
    # the same client-count-sensitive checks.
    sleep 1
}

# `tracing_subscriber`'s default formatter wraps each individual field name/punctuation in its own
# ANSI color/style escape — a plain-text search for a phrase that crosses from the log message
# into a field name (e.g. "reconnecting attempt=") will not find it in the raw bytes, since escape
# codes are interspersed *inside* what looks like one contiguous phrase. Strip them first for any
# check spanning more than one tracing field.
strip_ansi() {
    sed -E 's/\x1b\[[0-9;]*m//g' "$1"
}

echo "building ddnet-ai (debug) ..."
(cd "$REPO_ROOT" && cargo build -p ddnet-ai --bin ddnet-ai) >"$RUN_DIR/build.log" 2>&1 || {
    fail all "cargo build failed, see $RUN_DIR/build.log"
    exit 1
}

scenario_a() {
    local name="$(short_name a)"
    local log="$RUN_DIR/a-join.log"
    "$BOT" play --server 127.0.0.1:8303 --name "$name" --brain idle --duration 12 --data-dir "$DATA_DIR" >"$log" 2>&1 &
    local pid=$!
    sleep 5
    local status_out
    status_out="$(status_with_name "$name")"
    echo "$status_out" >"$RUN_DIR/a-status.log"
    if [[ "$status_out" == *"name='$name'"* ]]; then
        pass a "joined within 5s; server status shows name='$name' (see a-status.log)"
    else
        fail a "server status did not show name='$name' within 5s (see a-status.log, $log)"
    fi
    wait "$pid" 2>/dev/null
    wait_for_port_free_of_our_bot
}

scenario_b() {
    local name="$(short_name b)"
    local log="$RUN_DIR/b-circle.log"
    RUST_LOG=info "$BOT" play --server 127.0.0.1:8303 --name "$name" --brain circle --duration 8 --data-dir "$DATA_DIR" >"$log" 2>&1
    local analysis
    analysis="$(python3 "$REPO_ROOT/tools/e2e/analyze_positions.py" "$log")"
    echo "$analysis" >"$RUN_DIR/b-analysis.log"
    if [[ "$analysis" == PASS* ]] || [[ "$analysis" == *$'\n'PASS* ]]; then
        pass b "$(echo "$analysis" | tail -1)"
    else
        fail b "$(echo "$analysis" | tail -1) (see $log, b-analysis.log)"
    fi
    wait_for_port_free_of_our_bot
}

scenario_c() {
    local name="$(short_name c)"
    local log="$RUN_DIR/c-mapchange.log"
    "$BOT" play --server 127.0.0.1:8303 --name "$name" --brain idle --duration 24 --data-dir "$DATA_DIR" >"$log" 2>&1 &
    local pid=$!
    sleep 5
    "${ECON[@]}" change_map "BlmapChill" >"$RUN_DIR/c-econ-blmapchill.log" 2>&1
    sleep 7
    "${ECON[@]}" change_map "$ORIGINAL_MAP" >"$RUN_DIR/c-econ-restore.log" 2>&1
    sleep 7
    wait "$pid" 2>/dev/null
    # Captured into a variable, then matched with bash's own `[[ ... == *pattern* ]]` — deliberately
    # not `content | grep -q ...`: with `pipefail` set, `grep -q` closing its stdin on the *first*
    # match makes the upstream command's `write()` see `SIGPIPE`, and `pipefail` then reports the
    # whole pipeline as failed even though `grep` itself found the match (empirically confirmed:
    # `seq 1 100000 | grep -q '^5$'` reports `$? = 141` under `pipefail`) — this bit scenarios (c),
    # (d) and (f) in earlier drafts of this script.
    local content
    content="$(strip_ansi "$log")"
    local loaded_count
    loaded_count=$(grep -c "map loaded" <<<"$content")
    if [ "$loaded_count" -ge 3 ] && [[ "$content" != *isconnected* ]]; then
        pass c "map changed twice via econ, client kept playing ($loaded_count map loads, see $log)"
    else
        fail c "expected >=3 map loads and no disconnect, got $loaded_count loads (see $log)"
    fi
    wait_for_port_free_of_our_bot
}

scenario_d() {
    local name="$(short_name d)"
    local log="$RUN_DIR/d-restart.log"
    "$BOT" play --server 127.0.0.1:8303 --name "$name" --brain idle --duration 30 --data-dir "$DATA_DIR" >"$log" 2>&1 &
    local pid=$!
    sleep 5
    # Review finding F2: a graceful `systemctl restart` (the actual, original intent of this
    # scenario — a server-initiated restart/maintenance, not a crash) sends SIGTERM first: DDNet's
    # own graceful-shutdown handler notifies every connected client with an explicit
    # `NETMSG_CLOSE("Server shutdown")` before exiting — a real, peer-initiated close
    # (`by_peer: true`). `should_reconnect_after_peer_close` classifies this exact reason string as
    # reconnect-worthy (a bot-specific policy choice, not literal real-client parity — see that
    # function's own doc comment), so the driver reconnects from it, same as any other transient
    # loss. `Restart=on-failure` also brings the unit back up if it ever crashes outright, but this
    # is not that path — this is a deliberate, on-purpose restart.
    echo "restarting ddnet-local.service (graceful, sends NETMSG_CLOSE(\"Server shutdown\") to every client) ..."
    sudo systemctl restart ddnet-local.service
    sleep 12
    wait "$pid" 2>/dev/null
    local content
    content="$(strip_ansi "$log")"
    local in_game_count
    in_game_count=$(grep -c "in game" <<<"$content")
    if [[ "$content" == *"reconnecting attempt="* ]] && [ "$in_game_count" -ge 2 ]; then
        pass d "detected the graceful restart (Server shutdown), reconnected with backoff, re-entered the game ($in_game_count times; see $log)"
    else
        fail d "expected a reconnect and a second 'in game' after the restart (in_game_count=$in_game_count; see $log)"
    fi
    wait_for_port_free_of_our_bot
}

scenario_d2() {
    local name="$(short_name d2)"
    local log="$RUN_DIR/d2-sigkill.log"
    "$BOT" play --server 127.0.0.1:8303 --name "$name" --brain idle --duration 30 --timeout-secs 6 --data-dir "$DATA_DIR" >"$log" 2>&1 &
    local pid=$!
    sleep 5
    # Review finding F2: unlike (d)'s graceful restart, SIGKILL produces *no* `NETMSG_CLOSE` at
    # all — the server simply stops responding. Since a `connect()`-ed UDP socket surfaces the
    # OS's own ICMP Port-Unreachable as `ECONNREFUSED` once nothing is listening on that port any
    # more, and the driver now deliberately ignores that (review finding F2 — it is not the
    # application-level "peer is gone" signal), this scenario exercises the *other* detection
    # path entirely: `Connection`'s own silence timeout (`--timeout-secs 6`, well under
    # `Restart=on-failure`'s few-seconds restart time, so the timeout fires before the server is
    # back — proving the timeout path itself, not a lucky race with the restart). Confirmed live
    # (BUILD REPORT): `systemctl kill --signal=SIGKILL` itself fails with "Invalid argument"
    # against this server's cgroup on this host, so this sends SIGKILL to the actual PID instead;
    # `Restart=on-failure` in the unit still brings it back up on its own afterwards (SIGKILL
    # counts as a failure exit).
    local server_pid
    server_pid="$(systemctl show ddnet-local.service --property=MainPID --value)"
    echo "killing ddnet-local.service's main process (pid $server_pid) uncleanly (SIGKILL) to simulate a crash ..."
    sudo kill -9 "$server_pid"
    sleep 20
    wait "$pid" 2>/dev/null
    local content
    content="$(strip_ansi "$log")"
    local in_game_count
    in_game_count=$(grep -c "in game" <<<"$content")
    if [[ "$content" == *"reconnecting attempt="* ]] && [ "$in_game_count" -ge 2 ]; then
        pass d2 "detected the lost connection via the silence timeout (SIGKILL, no NETMSG_CLOSE), reconnected with backoff, re-entered the game ($in_game_count times; see $log)"
    else
        fail d2 "expected a reconnect and a second 'in game' after the SIGKILL (in_game_count=$in_game_count; see $log)"
    fi
    wait_for_port_free_of_our_bot
}

scenario_e() {
    local log="$RUN_DIR/e-redirect.log"
    if (cd "$REPO_ROOT" && cargo test -p ddai-client --test redirect_double -- --nocapture) >"$log" 2>&1; then
        pass e "redirect followed once, a second redirect refused (synthetic local UDP test double — see e-redirect.log)"
    else
        fail e "redirect_double test failed (see $log)"
    fi
}

scenario_f() {
    local name="$(short_name f)"
    local log="$RUN_DIR/f-kick.log"
    "$BOT" play --server 127.0.0.1:8303 --name "$name" --brain idle --duration 20 --data-dir "$DATA_DIR" >"$log" 2>&1 &
    local pid=$!
    sleep 5
    local status_out
    status_out="$(status_with_name "$name")"
    echo "$status_out" >"$RUN_DIR/f-status.log"
    local cid
    cid="$(echo "$status_out" | grep -oP "id=\K[0-9]+(?=[^\n]*name='$name')" | head -1)"
    if [ -z "$cid" ]; then
        fail f "could not find client id for '$name' in econ status (see f-status.log)"
        wait "$pid" 2>/dev/null
        wait_for_port_free_of_our_bot
        return
    fi
    "${ECON[@]}" kick "$cid" "test-reason" >"$RUN_DIR/f-econ-kick.log" 2>&1
    sleep 3
    wait "$pid" 2>/dev/null
    local content
    content="$(strip_ansi "$log")"
    if [[ "$content" == *"test-reason"* ]] && [[ "$content" != *"reconnecting attempt="* ]]; then
        pass f "kicked with the exact reason reported ('test-reason'), no auto-reconnect (see $log)"
    else
        fail f "expected the exact kick reason and no reconnect attempt (see $log)"
    fi
    wait_for_port_free_of_our_bot
}

scenario_g() {
    local log="$RUN_DIR/g-no-chat.log"
    if (cd "$REPO_ROOT" && cargo test -p ddai-client --lib session::tests -- --nocapture) >"$log" 2>&1; then
        pass g "Cl_Say never reaches the wire (full join sequence + direct guard test — see g-no-chat.log)"
    else
        fail g "a session::tests test failed (see $log)"
    fi
}

scenario_h() {
    local log="$RUN_DIR/h-margin.log"
    if [ -f "$RUN_DIR/a-join.log" ]; then
        cp "$RUN_DIR/a-join.log" "$log"
    else
        local name="$(short_name h)"
        "$BOT" play --server 127.0.0.1:8303 --name "$name" --brain idle --duration 10 --data-dir "$DATA_DIR" >"$log" 2>&1
    fi
    local summary_line
    summary_line="$(strip_ansi "$log" | grep "margin summary" | tail -1)"
    if [ -z "$summary_line" ]; then
        fail h "no margin summary found in the log (see $log)"
        return
    fi
    echo "$summary_line"
    local late_fraction
    late_fraction="$(echo "$summary_line" | grep -oP 'late_fraction=\K[0-9.]+')"
    pass h "margin distribution recorded (late_fraction=$late_fraction; see $log for full percentiles)"
}

SCENARIOS=("$@")
if [ ${#SCENARIOS[@]} -eq 0 ]; then
    SCENARIOS=(a b c d d2 e f g h)
fi

for s in "${SCENARIOS[@]}"; do
    echo "=== scenario $s ==="
    "scenario_$s"
done

echo
echo "=== summary ==="
for r in "${RESULTS[@]}"; do
    echo "$r"
done
echo "$PASS_COUNT passed, $FAIL_COUNT failed"
echo "logs: $RUN_DIR"

[ "$FAIL_COUNT" -eq 0 ]
