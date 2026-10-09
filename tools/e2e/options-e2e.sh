#!/usr/bin/env bash
# Local end-to-end of the «Умный ВБ» (`--wb-smart`) and «Без самоубийств (дуэль)» (`--no-selfkill`) controls on the «Запуск» card (task 5.15,
# D-103/D-104 and D-102), built like finish-e2e.sh (task 5.13). Everything runs on this machine, in a
# scratch directory, and never touches a production unit, /etc, Caddy, ~/aiddnet/data/bot, the owner's allow-list or the real proxy secrets:
#
#   * a PRIVATE DDNet server on 127.0.0.1:8463 (econ on 127.0.0.1:8464), its own scratch directory, `sv_register 0`, its own random econ
#     password. Never 8303, never a public server;
#   * a test web instance on 127.0.0.1:7793 with its own data directory (a ddnet-ai built with the `loopback-favourites` feature, which is
#     what lets a favourite be a loopback address: a production build refuses that). The private server is one favourite;
#   * the helper and the bot through the TEST path: `launcher-sim.mjs` stands in for the path units (it runs the REAL `ddnet-ai launch
#     apply`), `fake-systemctl.sh` for systemctl, `run-bot.sh` for the bot unit and its stop hook (the same `--finish`, `--wb-smart` and
#     `--no-selfkill=` words the unit passes).
#
# `launch-options.spec.ts` then clicks through the card and checks that both flags reached the bot: the env file, the bot's own log lines
# («wb smart: on», «self-kill: off (flag)»), its STATUS (`/api/bot/status`) and the «Бот» card.
#
# Since task 5.16 the same stack also drives the «Дуэль» preset and the «Машина» rows of the «Бот» card (scripted STATUS answers; the real run checks
# the bot's `search_window` and the host's load).
# Since tasks 3.17 and 3.20b the same stack also drives the toggles «Предсказатель соперника», «Настоящие ходы соперника от сервера» and the
# «ВБ (эксперимент)» value of «Дожим»; the unit's words `--window-model=`, `--preinput` and `--finish` are run-bot.sh's.
# Since task 5.17 (D-125) it also drives «Потоки поиска» (1–4) and the preset's 3; the unit's `--search-threads ${BOT_SEARCH_THREADS}` is run-bot.sh's too.
#
# Usage: tools/e2e/options-e2e.sh            (builds target/debug/ddnet-ai with the feature, runs Playwright, stops everything)
#        DDAI_BIN=/path/ddnet-ai tools/e2e/options-e2e.sh       (use an existing build; it MUST have the loopback-favourites feature)
#        CARGO_TARGET_DIR=/path ...                            (the build and the binary it finds follow it)
# Screenshots of the run: $E2E/screenshots (the scratch directory is recreated by the next run; copy what you want to keep).
#        DDAI_E2E_KEEP=1 ...                                   (leave the stack running after the run, for a look)
#        DDAI_E2E_GAME_PORT / DDAI_E2E_ECON_PORT / DDAI_E2E_WEB_PORT / DDAI_E2E_DIR   (other free loopback ports and scratch directory, defaults 8463 / 8464 / 7793)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
E2E="${DDAI_E2E_DIR:-$HOME/aiddnet/data/scratch/task-5.15-e2e}"
SERVER_BIN="${DDNET_SERVER_BIN:-$HOME/aiddnet/build/ddnet-20.1/build/DDNet-Server}"
# The private server's and the test web instance's ports (task 5.16: overridable, so that two worktrees' runs do not meet on the same ones).
GAME_PORT="${DDAI_E2E_GAME_PORT:-8463}"
ECON_PORT="${DDAI_E2E_ECON_PORT:-8464}"
WEB_PORT="${DDAI_E2E_WEB_PORT:-7793}"

log() { printf '[options-e2e] %s\n' "$*" >&2; }
die() { printf '[options-e2e] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -x "$SERVER_BIN" ]] || die "no DDNet server at $SERVER_BIN"
[[ -f "$HOME/aiddnet/data/ddnet-server/maps/Copy Love Box.map" ]] || die "no Copy Love Box.map in ~/aiddnet/data/ddnet-server/maps"

if [[ -z "${DDAI_BIN:-}" ]]; then
  log "building ddnet-ai with --features loopback-favourites (test binary, never installed)"
  # shellcheck source=/dev/null
  (cd "$ROOT" && source "$HOME/.cargo/env" && export RUSTC_WRAPPER=sccache SCCACHE_DIR="$HOME/aiddnet/data/cache/sccache" CARGO_BUILD_JOBS=3 \
    && cargo build -p ddnet-ai --features loopback-favourites)
  DDAI_BIN="${CARGO_TARGET_DIR:-$ROOT/target}/debug/ddnet-ai"
fi
[[ -x "$DDAI_BIN" ]] || die "$DDAI_BIN is not executable"

# ---- the scratch directory (only this one is ever removed) ----
case "$E2E" in
  "$HOME"/aiddnet/data/scratch/*) ;;
  *) die "refusing to use $E2E: the scratch directory must be under ~/aiddnet/data/scratch/" ;;
esac
if [[ -f "$E2E/ddnet.pid" ]] && kill -0 "$(cat "$E2E/ddnet.pid")" 2>/dev/null; then die "a stack from an earlier run is still up ($E2E): stop its pids first"; fi
rm -rf "$E2E"
mkdir -p "$E2E"/{data/{secrets,launch,servers,bot,logs,maps/cache,run},status,etc,var,shim,ddnet/maps}
chmod 0700 "$E2E/data/secrets"
# A placeholder where the helper looks for the fly bundle, so that the card offers the fly brains (the test only needs the choice and the
# helper's refusal; a fly brain is never started here, and this file is not a bundle).
mkdir -p "$E2E/data/runs/E-005/e005-fly/checkpoints"
echo "placeholder, not a bundle" >"$E2E/data/runs/E-005/e005-fly/checkpoints/final.bundle"
# A placeholder where the helper looks for the opponent-input model (task 3.17), so that the card offers the toggle. It is not a model: the
# test only clicks the toggle with the POST answered by the test, and every real start here has the toggle off (`--window-model=` empty).
mkdir -p "$E2E/data/bot/models"
echo "placeholder, not a model" >"$E2E/data/bot/models/opp-m1.oppnet"
: >"$E2E/none.toml"
chmod 0644 "$E2E/none.toml"
install -m 0755 "$HERE/support/fake-systemctl.sh" "$E2E/shim/systemctl"
install -m 0755 "$HERE/support/run-bot.sh" "$E2E/run-bot.sh"
export E2E_DIR="$E2E" E2E_BIN="$DDAI_BIN" NO_COLOR=1

# Stops what this script started (pids recorded in $E2E/*.pid, nothing found by name): the bot first, through the fake systemctl, so its stop
# hook runs; then the web instance, the stand-in and the private DDNet server.
# shellcheck disable=SC2317  # run through the EXIT trap
cleanup() {
  if [[ "${DDAI_E2E_KEEP:-0}" == 1 ]]; then
    log "DDAI_E2E_KEEP=1: leaving the stack up (pids in $E2E/*.pid)"
    return
  fi
  PATH="$E2E/shim:$PATH" systemctl stop ddnet-ai-bot.service >/dev/null 2>&1 || true
  local f
  for f in web sim ddnet; do
    if [[ -f "$E2E/$f.pid" ]]; then kill -TERM "$(cat "$E2E/$f.pid")" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT

# ---- the web password (a fresh one for this scratch instance only) ----
"$DDAI_BIN" web-passwd --data-dir "$E2E/data" >/dev/null
PASSWORD_FILE="$E2E/data/secrets/web-password.txt"
[[ -s "$PASSWORD_FILE" ]] || die "no password file"

# ---- the private DDNet server ----
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
sv_name "aiddnet task-5.15 e2e (private, 127.0.0.1 only)"
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
ECON=("$ROOT/tools/ddnet-server/econ.py" --host 127.0.0.1 --port "$ECON_PORT" --password-file "$E2E/ddnet/server.cfg")
up=0
for _ in $(seq 1 60); do
  if python3 "${ECON[@]}" status >/dev/null 2>&1; then up=1; break; fi
  sleep 0.5
done
[[ "$up" == 1 ]] || { tail -20 "$E2E/ddnet/stdout.log" >&2; die "the private DDNet server did not come up"; }
log "private DDNet server up on 127.0.0.1:$GAME_PORT (econ $ECON_PORT)"

# ---- the one favourite: the private server (the file the site writes; consent_at is the owner's, here set by the test) ----
NOW="$(date +%s)"
cat >"$E2E/data/launch/favourites.json" <<JSON
{"v":1,"favourites":[{"address":"127.0.0.1:$GAME_PORT","name":"Private e2e","nick":"Muha","connection":"direct","consent_at":$NOW,"notes":"","added_at":$NOW,"reopened_at":0}]}
JSON
chmod 0644 "$E2E/data/launch/favourites.json"

# ---- the stand-in for the path units and the test web instance ----
node "$HERE/support/launcher-sim.mjs" "$E2E" "$DDAI_BIN" >"$E2E/sim.log" 2>&1 &
echo $! >"$E2E/sim.pid"
"$DDAI_BIN" web --listen "127.0.0.1:$WEB_PORT" --data-dir "$E2E/data" --launch-status-dir "$E2E/status" \
  --launch-config "$E2E/none.toml" --bot-socket "$E2E/data/bot/live.sock" --maps-dir "$E2E/data/maps/cache" >"$E2E/web.log" 2>&1 &
echo $! >"$E2E/web.pid"
for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:$WEB_PORT/" >/dev/null 2>&1 && break
  sleep 0.25
done
curl -fsS "http://127.0.0.1:$WEB_PORT/" >/dev/null || { tail -20 "$E2E/web.log" >&2; die "the test web instance did not come up"; }
log "test web instance up on http://127.0.0.1:$WEB_PORT"

# ---- Playwright ----
cd "$HERE"
[[ -d node_modules ]] || die "no node_modules in tools/e2e: run \`npm install\` and \`npx playwright install --only-shell chromium\` there first"
set +e
DDAI_OPTIONS_E2E_URL="http://127.0.0.1:$WEB_PORT" \
DDAI_OPTIONS_E2E_PASSWORD_FILE="$PASSWORD_FILE" \
DDAI_OPTIONS_E2E_DIR="$E2E" \
DDAI_OPTIONS_E2E_GAME_ADDR="127.0.0.1:$GAME_PORT" \
DDAI_OPTIONS_E2E_ECON_PORT="$ECON_PORT" \
DDAI_OPTIONS_E2E_ECON_CFG="$E2E/ddnet/server.cfg" \
  npx playwright test launch-options.spec.ts "$@"
rc=$?
set -e
log "playwright exit code $rc"
exit "$rc"
