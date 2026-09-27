#!/usr/bin/env bash
# Checks whether each given .map file loads in the built, vanilla DDNet-Server
# 20.1: starts the server briefly with sv_map = that map's basename, in an
# isolated scratch storage dir (never touches the real runtime layout or
# binds anywhere but 127.0.0.1 on a private test port), and reports
# LOAD_OK/LOAD_FAIL based on whether the server is still running afterwards
# (a failed map load makes CServer::Run() return -1 immediately, see
# server.cpp "failed to load map").
#
# Usage:
#   tools/ddnet-server/check-maps.sh <map1.map> [map2.map ...]
#   tools/ddnet-server/check-maps.sh   # no args: checks the maps this repo
#                                        cares about (copy-love-box + the
#                                        physics-scratch block maps)
set -uo pipefail

SERVER_BIN="${DDNET_SERVER_BIN:-$HOME/aiddnet/build/ddnet-20.1/build/DDNet-Server}"
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/ddnet-map-check.XXXXXX")"
TEST_PORT="${DDNET_MAP_CHECK_PORT:-38304}"

cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

mkdir -p "$SCRATCH/maps"
cat > "$SCRATCH/storage.cfg" <<EOF
add_path $SCRATCH
EOF

if [[ -x "$SERVER_BIN" ]]; then :; else
  echo "check-maps.sh: server binary not found/executable at $SERVER_BIN (run build.sh first)" >&2
  exit 1
fi

declare -a MAPS
if [[ $# -gt 0 ]]; then
  MAPS=("$@")
else
  DATA_ROOT="${DDNET_DATA_ROOT:-$HOME/aiddnet/data}"
  MAPS=(
    "$DATA_ROOT/maps/copy-love-box/Copy Love Box_6e79ef4319e553f904777e56c2a66ac243ea155c331d8665ed58919b11bdfd25.map"
  )
  for f in "$DATA_ROOT/research/physics-scratch/maps/"*.map; do
    MAPS+=("$f")
  done
fi

OK_COUNT=0
FAIL_COUNT=0

for MAP_SRC in "${MAPS[@]}"; do
  if [[ ! -f "$MAP_SRC" ]]; then
    echo "SKIP (not found): $MAP_SRC"
    continue
  fi
  MAP_BASENAME="$(basename "$MAP_SRC" .map)"
  rm -f "$SCRATCH/maps/"*.map
  cp "$MAP_SRC" "$SCRATCH/maps/$MAP_BASENAME.map"

  CFG="$SCRATCH/test.cfg"
  cat > "$CFG" <<EOF
bindaddr 127.0.0.1
sv_port $TEST_PORT
sv_register 0
sv_rcon_password "check-maps-test"
ec_password "check-maps-test"
ec_port 0
sv_max_clients 8
sv_map "$MAP_BASENAME"
logfile ""
EOF

  LOG="$SCRATCH/$MAP_BASENAME.out"
  # `exec` replaces the subshell with the server itself (after cd), so $!
  # below is the real DDNet-Server PID and a direct `kill` reaches it - no
  # `timeout` wrapper, so the port is guaranteed free before the next
  # iteration starts (otherwise a leftover process still holding the port
  # for its last few seconds causes a spurious "Address already in use").
  ( cd "$SCRATCH" && exec "$SERVER_BIN" -f "$CFG" > "$LOG" 2>&1 ) &
  PID=$!
  sleep 3
  if kill -0 "$PID" 2>/dev/null; then
    RESULT="LOAD_OK"
    OK_COUNT=$((OK_COUNT + 1))
  else
    RESULT="LOAD_FAIL"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
  kill "$PID" 2>/dev/null
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.3
  done
  kill -9 "$PID" 2>/dev/null
  wait "$PID" 2>/dev/null

  printf '%-8s %s\n' "$RESULT" "$MAP_BASENAME"
  if [[ "$RESULT" == "LOAD_FAIL" ]]; then
    grep -iE "error|fail|assert|signal" "$LOG" | sed 's/^/    /'
  fi
done

echo
echo "checked ${#MAPS[@]} map(s): $OK_COUNT load, $FAIL_COUNT fail"
[[ "$FAIL_COUNT" -eq 0 ]]
