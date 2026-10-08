#!/usr/bin/env bash
# Task 4.14 (D-124): one run of the CPU-priority measurement on a PRIVATE DDNet server (never 8303): the hybrid bot as a transient SYSTEM unit
# (in system.slice, like deploy/systemd/ddnet-ai-bot.service) against one scripted opponent, with or without a parallel `cargo build --release`.
#
#   tools/e2e/cpu_priority.sh <label> <seconds> <unit-variant> <load> [search-threads]
#     unit-variant  old     no CPU settings (the unit of before 4.14)
#                   new     the settings of deploy/systemd/ddnet-ai-bot.service (CPU_NEW below: Nice, CPUWeight, IOWeight)
#                   newsrv  `new` for the bot and also for the game server
#                   slicesrv  `slice` with the game server in the bot's slice too
#                   slice   `new` in an own top-level slice with the weight on the slice (CPU_SLICE_WEIGHT)
#     load          none    nothing else of ours runs
#                   normal  `cargo build --release --workspace -j 6` at normal priority, in a loop (clean target dir) until the run ends
#                   low     the same through tools/lowprio.sh (the agents' recipe)
#                   raw:<prefix>  the same behind any command prefix, e.g. raw:"nice -n 15" (to compare recipes)
#
# Output: $OUT/<label>.json (the bot's report), <label>.status.jsonl (the poller: STATUS search_window every 5 s, load average, the unit's
# cpu.pressure and cpu.stat), <label>.meta (variant, load, load average before/after). Then tools/e2e/cpu_priority_table.py.
# The server, the opponent and the bot are transient units t414-*, stopped at the end (sudo systemd-run: units live in /run, /etc is never touched).
# Environment: T_SCRATCH (default ~/aiddnet/data/scratch/task-4.14: srv/ with priv.cfg + storage.cfg, bin/ddnet-ai), T_BIN, T_LOAD_JOBS (default 6),
# T_LOAD_TARGET (default $T_SCRATCH/loadtarget, deleted at the end), T_NEW_PROPS (override the `new` settings), T_PROFILE.
set -euo pipefail
LABEL="${1:?label}"; SECS="${2:?seconds}"; VARIANT="${3:?old|new|slice}"; LOAD="${4:?none|normal|low}"; THREADS="${5:-1}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
S="${T_SCRATCH:-$HOME/aiddnet/data/scratch/task-4.14}"; OUT="${OUT:-$S/runs}"; BIN="${T_BIN:-$S/bin/ddnet-ai}"
PORT=8479   # (econ 8480 is in priv.cfg)
JOBS="${T_LOAD_JOBS:-6}"; LT="${T_LOAD_TARGET:-$S/loadtarget}"
[[ "$PORT" != 8303 ]] || exit 2
mkdir -p "$OUT" "$S/botdata" "$S/spardata"

CPU_NEW=(-p Nice=-5 -p CPUWeight=1000 -p IOWeight=1000)
[[ -n "${T_NEW_PROPS:-}" ]] && read -r -a CPU_NEW <<<"$T_NEW_PROPS"
CPU_SLICE_WEIGHT="${CPU_SLICE_WEIGHT:-1000}"
SRV_PROPS=()
case "$VARIANT" in
  old) CPU_PROPS=() ;;
  new) CPU_PROPS=("${CPU_NEW[@]}") ;;
  slice) CPU_PROPS=(-p Nice=-5 -p Slice=t414bot.slice) ;;
  slicesrv) CPU_PROPS=(-p Nice=-5 -p Slice=t414bot.slice); SRV_PROPS=(-p Nice=-5 -p Slice=t414bot.slice) ;;   # bot AND game server in the one slice
  newsrv) CPU_PROPS=("${CPU_NEW[@]}"); SRV_PROPS=("${CPU_NEW[@]}") ;;   # `new` for the bot AND for the game server (ddnet-local.service's twin)
  *) echo "unknown variant $VARIANT" >&2; exit 2 ;;
esac

# The hardening of the production unit that can matter for scheduling or capabilities (Nice=-5 must still apply with an empty bounding set).
HARDEN=(-p User=ubuntu -p Group=ubuntu -p NoNewPrivileges=true -p CapabilityBoundingSet= -p AmbientCapabilities= -p ProtectSystem=strict
        -p ProtectHome=read-only -p "ReadWritePaths=$S $HOME/aiddnet/data/run" -p PrivateTmp=true -p PrivateDevices=true -p MemoryDenyWriteExecute=true
        -p RestrictRealtime=true -p MemoryMax=2G -p TasksMax=256 -p IPAddressAllow="127.0.0.0/8 ::1" -p IPAddressDeny=any
        -p StandardInput=null -p KillSignal=SIGTERM -p TimeoutStopSec=30)
ENVS=(-E NO_COLOR=1 -E RUST_LOG=info -E MALLOC_MMAP_THRESHOLD_=131072)

units=(t414-srv t414-opp t414-bot)
stop_all() { for u in "${units[@]}"; do sudo systemctl stop "$u.service" 2>/dev/null || true; sudo systemctl reset-failed "$u.service" 2>/dev/null || true; done; }
LOADPID=""
cleanup() {
  [[ -n "$LOADPID" ]] && kill -TERM -- "-$LOADPID" 2>/dev/null || true
  stop_all
  [[ "$VARIANT" == slice* ]] && sudo systemctl stop t414bot.slice 2>/dev/null || true
}
trap cleanup EXIT
stop_all

# --- server and opponent: plain units in system.slice, like ddnet-local.service and the sparring unit (no priority) ---
sudo systemd-run -q --unit=t414-srv --collect "${HARDEN[@]}" "${SRV_PROPS[@]}" -p WorkingDirectory="$S/srv" -p "StandardOutput=file:$OUT/$LABEL.srv.log" -p "StandardError=file:$OUT/$LABEL.srv.log" \
  "${ENVS[@]}" ~/aiddnet/build/ddnet-20.1/build/DDNet-Server -f priv.cfg >/dev/null
sleep 3
sudo systemd-run -q --unit=t414-opp --collect "${HARDEN[@]}" -p WorkingDirectory="$S/spardata" -p "StandardOutput=file:$OUT/$LABEL.opp.log" -p "StandardError=file:$OUT/$LABEL.opp.log" \
  "${ENVS[@]}" "$BIN" play --server "127.0.0.1:$PORT" --name Spar1 --brain scripted --duration 0 --clan Spar --wb off --no-console --no-bridge --no-control --no-memory --no-settings --no-autoclip --data-dir "$S/spardata" >/dev/null
sleep 3

# --- the load ---
snap() { { echo "== $1 $(date -u +%T) loadavg $(cut -d' ' -f1-4 /proc/loadavg)"; ps -eo pid,ni,pcpu,nlwp,args --sort=-pcpu | head -9 | cut -c1-150; } >>"$OUT/$LABEL.ps"; }
: >"$OUT/$LABEL.ps"
printf 'variant=%s\nload=%s\nthreads=%s\nsecs=%s\nstart=%s\nloadavg_before=%s\n' "$VARIANT" "$LOAD" "$THREADS" "$SECS" "$(date -u +%FT%TZ)" "$(cut -d' ' -f1-3 /proc/loadavg)" >"$OUT/$LABEL.meta"
if [[ "$LOAD" != none ]]; then
  PFX=""; [[ "$LOAD" == low ]] && PFX="$ROOT/tools/lowprio.sh"
  [[ "$LOAD" == raw:* ]] && PFX="${LOAD#raw:}"   # e.g. raw:"nice -n 15" or raw:"chrt -i 0", to compare recipes
  # A full clean build again and again until we kill the group; sccache is off on purpose (a cache hit would be no load).
  setsid bash -c "cd '$ROOT'; source ~/.cargo/env; unset RUSTC_WRAPPER SCCACHE_DIR; export CARGO_TARGET_DIR='$LT';
    while true; do rm -rf '$LT'; $PFX cargo build --release --workspace --offline --locked -j $JOBS; done" >"$OUT/$LABEL.load.log" 2>&1 &
  LOADPID=$!
  sleep 15
fi

# --- the bot ---
snap start
BOT_ARGS=(play --server "127.0.0.1:$PORT" --name Muha --brain hybrid --wb off --duration "$SECS" --hybrid-mirror on --finish off --wb-smart off --no-selfkill=false
          "--window-model=" --preinput off --search-threads "$THREADS" --no-console --data-dir "$S/botdata" --report "$OUT/$LABEL.json")
rm -f "$OUT/$LABEL.json" "$S/botdata/bot/live.sock"
sudo systemd-run -q --wait --collect --unit=t414-bot "${HARDEN[@]}" "${CPU_PROPS[@]}" -p WorkingDirectory="$S/botdata" \
  -p "StandardOutput=file:$OUT/$LABEL.bot.log" -p "StandardError=file:$OUT/$LABEL.bot.log" "${ENVS[@]}" "$BIN" "${BOT_ARGS[@]}" &
BOTRUN=$!
if [[ "$VARIANT" == slice* ]]; then
  for _ in $(seq 1 20); do sudo systemctl set-property --runtime t414bot.slice CPUWeight="$CPU_SLICE_WEIGHT" 2>/dev/null && break; sleep 0.5; done
fi
python3 "$ROOT/tools/e2e/cpu_priority_poll.py" --sock "$S/botdata/bot/live.sock" --unit t414-bot.service --secs $((SECS + 5)) --out "$OUT/$LABEL.status.jsonl" || true
snap end
wait "$BOTRUN" || echo "bot unit exit: $?" >&2
printf 'loadavg_after=%s\nend=%s\n' "$(cut -d' ' -f1-3 /proc/loadavg)" "$(date -u +%FT%TZ)" >>"$OUT/$LABEL.meta"
[[ -n "$LOADPID" ]] && { kill -TERM -- "-$LOADPID" 2>/dev/null || true; LOADPID=""; sleep 3; rm -rf "$LT"; }
echo "done $LABEL: report $OUT/$LABEL.json"
