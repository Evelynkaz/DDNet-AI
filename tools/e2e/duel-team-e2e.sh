#!/usr/bin/env bash
# Local end-to-end of the automatic duel detection (task 4.12, D-108): three of our bots on a PRIVATE DDNet 20.1 server; two are put into a
# DDRace team with the server's own console command (`set_team_ddr`), which makes the server send `Sv_TeamsState` exactly as F-DDrace's
# `/1vs1` does (`SetForceCharacterTeam`). On plain DDNet a two-player team is NOT a duel (no F-DDrace evidence): nobody logs anything. Then
# the server says F-DDrace's invitation line (econ `say`, a system chat line): each of the two must log "duel: ... by=team" within a second,
# the third must log nothing; then the third joins the team (three in it: not a duel) and the two must log "the 1vs1 is over". Nothing here touches a public server, a
# production unit, /etc or ~/aiddnet/data/bot:
#
#   * a PRIVATE DDNet server on 127.0.0.1:8443 (econ on 127.0.0.1:8444), its own scratch directory, `sv_register 0`, its own random econ
#     password. Never 8303, never a public server;
#   * one data directory per bot (each bot owns a bridge socket there).
#
# Usage: tools/e2e/duel-team-e2e.sh            (needs target/debug/ddnet-ai built: `cargo build -p ddnet-ai`)
#        DDAI_BIN=/path/ddnet-ai tools/e2e/duel-team-e2e.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
E2E="${DDAI_E2E_DIR:-$HOME/aiddnet/data/scratch/task-4.12-duel-e2e}"
SERVER_BIN="${DDNET_SERVER_BIN:-$HOME/aiddnet/build/ddnet-20.1/build/DDNet-Server}"
DDAI_BIN="${DDAI_BIN:-${CARGO_TARGET_DIR:-$ROOT/target}/debug/ddnet-ai}"
GAME_PORT=8443
ECON_PORT=8444

log() { printf '[duel-team-e2e] %s\n' "$*" >&2; }
die() { printf '[duel-team-e2e] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -x "$SERVER_BIN" ]] || die "no DDNet server at $SERVER_BIN"
[[ -x "$DDAI_BIN" ]] || die "no $DDAI_BIN (cargo build -p ddnet-ai)"
[[ -f "$HOME/aiddnet/data/ddnet-server/maps/Copy Love Box.map" ]] || die "no Copy Love Box.map in ~/aiddnet/data/ddnet-server/maps"
case "$E2E" in
  "$HOME"/aiddnet/data/scratch/*) ;;
  *) die "refusing to use $E2E: the scratch directory must be under ~/aiddnet/data/scratch/" ;;
esac
if [[ -f "$E2E/ddnet.pid" ]] && kill -0 "$(cat "$E2E/ddnet.pid")" 2>/dev/null; then die "a server from an earlier run is still up ($E2E)"; fi
if ss -ltnu 2>/dev/null | grep -qE ":($GAME_PORT|$ECON_PORT)\b"; then die "port $GAME_PORT or $ECON_PORT is in use"; fi
rm -rf "$E2E"
mkdir -p "$E2E/ddnet/maps"

# Stops what this script started (the pids recorded in $E2E/*.pid; nothing is found by name).
# shellcheck disable=SC2317  # run through the EXIT trap
cleanup() {
  local f
  for f in "$E2E"/bot-*.pid "$E2E"/ddnet.pid; do
    [[ -f "$f" ]] && kill -TERM "$(cat "$f")" 2>/dev/null || true
  done
  # The server leaves a moment after the TERM: wait for it (its pid is the one this script recorded).
  local i
  for i in $(seq 1 20); do
    [[ -f "$E2E/ddnet.pid" ]] && kill -0 "$(cat "$E2E/ddnet.pid")" 2>/dev/null || return 0
    sleep 0.25
  done
}
trap cleanup EXIT

cp "$HOME/aiddnet/data/ddnet-server/maps/Copy Love Box.map" "$E2E/ddnet/maps/"
cat >"$E2E/ddnet/storage.cfg" <<CFG
add_path $E2E/ddnet
add_path $HOME/aiddnet/build/ddnet-20.1/src/data
CFG
ECON_PASSWORD="$(head -c 18 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 20)"
cat >"$E2E/ddnet/server.cfg" <<CFG
bindaddr 127.0.0.1
sv_port $GAME_PORT
sv_register 0
sv_ipv4only 1
sv_name "aiddnet task-4.12 e2e (private, 127.0.0.1 only)"
sv_map "Copy Love Box"
sv_max_clients 12
sv_max_clients_per_ip 12
sv_connlimit_time 0
ec_bindaddr 127.0.0.1
ec_port $ECON_PORT
ec_password "$ECON_PASSWORD"
loglevel 0
logfile "$E2E/ddnet/server.log"
CFG
chmod 0600 "$E2E/ddnet/server.cfg"
(cd "$E2E/ddnet" && exec setsid "$SERVER_BIN" -f server.cfg >"$E2E/ddnet/stdout.log" 2>&1 </dev/null) &
echo $! >"$E2E/ddnet.pid"
econ() { python3 "$ROOT/tools/ddnet-server/econ.py" --host 127.0.0.1 --port "$ECON_PORT" --password-file "$E2E/ddnet/server.cfg" "$@" 2>&1; }
up=0
for _ in $(seq 1 60); do
  if econ status >/dev/null; then up=1; break; fi
  sleep 0.5
done
[[ "$up" == 1 ]] || { tail -20 "$E2E/ddnet/stdout.log" >&2; die "the private DDNet server did not come up"; }
log "private DDNet server up on 127.0.0.1:$GAME_PORT (econ $ECON_PORT)"

for n in dA dB dC; do
  mkdir -p "$E2E/data-$n"
  (exec setsid "$DDAI_BIN" play --server "127.0.0.1:$GAME_PORT" --name "$n" --bot --brain idle --wb off --no-console --no-control \
     --no-timeout-code --duration 120 --data-dir "$E2E/data-$n" --clips-dir "$E2E/clips-$n" --no-autoclip \
     >"$E2E/bot-$n.log" 2>&1 </dev/null) &
  echo $! >"$E2E/bot-$n.pid"
  sleep 2
done
# The ids the server gave them (in join order).
ids=""
for _ in $(seq 1 40); do
  ids="$(econ status | sed -n "s/.*id=\([0-9]*\) .*name='d\([ABC]\)'.*/\2=\1/p" | sort | tr '\n' ' ')"
  [[ "$ids" == "A=0 B=1 C=2 " ]] && break
  sleep 0.5
done
[[ "$ids" == "A=0 B=1 C=2 " ]] || die "the bots did not all join (saw: $ids)"
log "three bots in: $ids"

lines() { sed 's/\x1b\[[0-9;]*m//g' "$E2E/bot-d$1.log" | grep -a "duel:" || true; }
count() { lines "$1" | grep -c "$2" || true; }
fail=0
check() { # check <description> <got> <want>
  if [[ "$2" == "$3" ]]; then log "ok: $1 ($2)"; else log "FAIL: $1 (got $2, want $3)"; fail=1; fi
}

# 1. Plain DDNet two-player team (what `/team` or an admin's `set_team_ddr` makes): NOT a duel (review 4.12, F1): no F-DDrace evidence.
econ set_team_ddr 0 5 >/dev/null
econ set_team_ddr 1 5 >/dev/null
sleep 3
check "A: a two-player team with no F-DDrace chat is not a duel" "$(lines A | wc -l)" 0
check "B: a two-player team with no F-DDrace chat is not a duel" "$(lines B | wc -l)" 0

# 2. The server says what F-DDrace's `/1vs1` says (a system chat line, `client_id` -1: econ `say`): now the team counts.
econ say "You have been invited to a fight by 'dB', type '/1vs1 dB' to join" >/dev/null
sleep 3
check "A logs the duel (by team)" "$(count A 'is on.*by="team"')" 1
check "B logs the duel (by team)" "$(count B 'is on.*by="team"')" 1
check "C (team 0) logs nothing" "$(lines C | wc -l)" 0

# 3. A third player joins the team: not a duel any more.
econ set_team_ddr 2 5 >/dev/null
sleep 4
check "A: three in the team, the duel is over" "$(count A 'is over')" 1
check "B: three in the team, the duel is over" "$(count B 'is over')" 1
check "C (three in the team) never logs a duel" "$(lines C | wc -l)" 0

econ set_team_ddr 2 0 >/dev/null
econ set_team_ddr 1 0 >/dev/null
econ set_team_ddr 0 0 >/dev/null
sleep 1
log "private server and bots are stopped by the exit trap"
exit "$fail"
