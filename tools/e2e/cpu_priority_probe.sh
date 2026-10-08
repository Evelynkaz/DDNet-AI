#!/usr/bin/env bash
# Task 4.14 (D-124): the probes of cpu_priority_probe.py in different scheduling settings, all at once (so all see the same ambient load).
#   tools/e2e/cpu_priority_probe.sh <seconds> <label> <variant>...    variants: user old new slice rr(diagnostic only) idle
#     user   as an agent's shell does it: in the session, no settings
#     old    transient system unit (system.slice), no settings (the bot unit of before 4.14)
#     new    system unit with Nice=-5 CPUWeight=1000 IOWeight=1000 (the bot unit of 4.14)
#     slice  `new` Nice, in an own top-level slice with CPUWeight=1000 on the slice
#     nice   only Nice=-5          weight  only CPUWeight=1000       new10k  Nice=-5 CPUWeight=10000 (the maximum)
#     slice10k  `slice` with CPUWeight=10000 on the slice
#     rr     SCHED_RR priority 1: a diagnostic ceiling, never a setting for the bot
# Results: $OUT/<label>-<variant>.json. Units: t414p-<variant> (sudo systemd-run, /run only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
S="${T_SCRATCH:-$HOME/aiddnet/data/scratch/task-4.14}"; OUT="${OUT:-$S/probes}"; mkdir -p "$OUT"
SECS="${1:?seconds}"; LABEL="${2:?label}"; shift 2
PY="$ROOT/tools/e2e/cpu_priority_probe.py"
pids=()
cleanup() { for v in "$@"; do sudo systemctl stop "t414p-$v.service" 2>/dev/null || true; done; }
trap 'cleanup "$@"; sudo systemctl stop t414p.slice t414p10k.slice 2>/dev/null || true' EXIT
{ echo "== $LABEL start $(date -u +%T) loadavg $(cut -d' ' -f1-4 /proc/loadavg)"; ps -eo pid,ni,pcpu,nlwp,args --sort=-pcpu | head -8 | cut -c1-140; } >"$OUT/$LABEL.ps"
for v in "$@"; do
  f="$OUT/$LABEL-$v.json"
  case "$v" in
    user) python3 "$PY" "$SECS" "$v" >"$f" & pids+=($!) ;;
    old|new|slice|rr|nice|weight|new10k|slice10k)
      props=(-p User=ubuntu -p Group=ubuntu)
      case "$v" in
        new) props+=(-p Nice=-5 -p CPUWeight=1000 -p IOWeight=1000) ;;
        slice) props+=(-p Nice=-5 -p Slice=t414p.slice) ;;
        slice10k) props+=(-p Nice=-5 -p Slice=t414p10k.slice) ;;
        nice) props+=(-p Nice=-5) ;;
        weight) props+=(-p CPUWeight=1000) ;;
        new10k) props+=(-p Nice=-5 -p CPUWeight=10000) ;;
        rr) props+=(-p CPUSchedulingPolicy=rr -p CPUSchedulingPriority=1) ;;
      esac
      sudo systemd-run -q --wait --collect --pipe --unit="t414p-$v" "${props[@]}" python3 "$PY" "$SECS" "$v" >"$f" & pids+=($!)
      ;;
    *) echo "unknown variant $v" >&2; exit 2 ;;
  esac
done
for v in "$@"; do
  if [[ "$v" == slice ]]; then
    for _ in $(seq 1 20); do sudo systemctl set-property --runtime t414p.slice CPUWeight=1000 2>/dev/null && break; sleep 0.5; done
  fi
  if [[ "$v" == slice10k ]]; then
    for _ in $(seq 1 20); do sudo systemctl set-property --runtime t414p10k.slice CPUWeight=10000 2>/dev/null && break; sleep 0.5; done
  fi
done
wait "${pids[@]}"
{ echo "== $LABEL end $(date -u +%T) loadavg $(cut -d' ' -f1-4 /proc/loadavg)"; } >>"$OUT/$LABEL.ps"
for v in "$@"; do cat "$OUT/$LABEL-$v.json"; done
