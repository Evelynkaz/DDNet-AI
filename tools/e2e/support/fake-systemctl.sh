#!/usr/bin/env bash
# A stand-in for `systemctl` for the local e2e of the server browser (task 5.12): the REAL `ddnet-ai launch apply` runs against it, so the
# helper's own validation, memory and files are exercised, but nothing touches the machine's systemd or any production unit. Only the bot
# unit does anything: `start` runs the wrapper `run-bot.sh` (the bot, then the unit's ExecStopPost hook `launch exited`), `stop` sends it
# SIGTERM. The wrapper's pid is kept in $E2E_DIR/bot.pid; only that pid is ever signalled (never a process this script did not start).
set -u
: "${E2E_DIR:?}"
PIDFILE="$E2E_DIR/bot.pid"
echo "$*" >>"$E2E_DIR/systemctl.log"
alive() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
case "${1:-}" in
  show)
    prop="${2:-}"
    unit="${*: -1}"
    case "$prop" in
      --property=ActiveState)
        if [[ "$unit" == ddnet-ai-bot.service ]] && alive; then echo active
        elif [[ "$unit" == ddnet-local.service ]]; then echo active
        else echo inactive; fi ;;
      --property=DropInPaths) : ;; # no foreign drop-in
    esac ;;
  start)
    if [[ "${2:-}" == ddnet-ai-bot.service ]]; then
      setsid "$E2E_DIR/run-bot.sh" >>"$E2E_DIR/bot.log" 2>&1 </dev/null &
      echo $! >"$PIDFILE"
    fi ;;
  stop)
    if [[ "${2:-}" == ddnet-ai-bot.service ]] && alive; then
      kill -TERM "$(cat "$PIDFILE")" 2>/dev/null
      # Give the wrapper (bot + the exit hook) a few seconds, bounded.
      for _ in $(seq 1 100); do alive || break; sleep 0.1; done
    fi ;;
  *) : ;; # daemon-reload, reset-failed, ...
esac
exit 0
