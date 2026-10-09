#!/usr/bin/env bash
# Task 4.14 (D-124): run a command at LOW priority among the agents' own work: nice 15 (and, if asked, SCHED_IDLE), so that a build or an arena
# run yields to the other agents' work in the same session. For anything heavy that is not the bot: cargo build / test / clippy, duel_stats, training.
#
#   tools/lowprio.sh <command> [args...]        e.g.  tools/lowprio.sh cargo test --workspace
#
# WHAT THIS DOES NOT DO: it does not protect the live bot. Everything the agents run lives in user.slice, and the bot competes with user.slice as a
# whole; no setting INSIDE user.slice changes that share. The bot's protection is its own slice (deploy/systemd/ddnetaibot.slice, D-124) or a lever on
# user.slice itself. The wrapper is fairness among agents plus the habit of fewer threads (cargo -j 3).
# It deliberately does NOT use `systemd-run --user --scope`: that moves the command out of the session scope into user@1000.service, which competes
# with the whole session scope as an equal, so a "low priority" build there got MORE CPU than a plain nice-15 process in the session (review 4.14 F1).
# The command stays in the caller's cgroup.
#
# What it does, outermost first (each step is skipped when the tool is missing or refused; the command always runs):
#   1. chrt --idle 0     only with LOWPRIO_IDLE=1: SCHED_IDLE, below every nice level in the same cgroup (no privilege needed; the command and its
#                        children cannot leave it). Off by default: two idle-class jobs starve each other's neighbours' fairness as well.
#   2. ionice -c2 -n7    best-effort I/O, the lowest level. A no-cost default: it takes effect only with the BFQ I/O scheduler, which this VPS
#                        does not use (its disk has scheduler `none`), so today it changes nothing.
#   3. nice -n 15        CPU niceness 15 (children inherit it).
# The command is exec'd at the end, so its exit status, signals, stdin/stdout/stderr, environment and cgroup are exactly its own.
#
# Environment: LOWPRIO_NICE (default 15, 0..19), LOWPRIO_IDLE=1 (add SCHED_IDLE), LOWPRIO_DRYRUN=1 (print the command line that would run, one word per
# line, and exit 0). Exit status 2: no command given or a bad value.
# See deploy/README.md, section "Приоритет CPU" and docs/DECISIONS.md D-124.
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "usage: tools/lowprio.sh <command> [args...]   (runs the command at low CPU priority: nice 15; it does not protect the live bot, see the header)" >&2
  exit 2
fi

NICE="${LOWPRIO_NICE:-15}"
[[ "$NICE" =~ ^([0-9]|1[0-9])$ ]] || { echo "lowprio.sh: LOWPRIO_NICE must be 0..19, got '$NICE'" >&2; exit 2; }

prefix=()
if [[ -n "${LOWPRIO_IDLE:-}" ]] && command -v chrt >/dev/null 2>&1 && chrt --idle 0 true >/dev/null 2>&1; then
  prefix+=(chrt --idle 0)
fi
if command -v ionice >/dev/null 2>&1 && ionice -c2 -n7 true >/dev/null 2>&1; then
  prefix+=(ionice -c2 -n7)
fi
prefix+=(nice -n "$NICE")

if [[ -n "${LOWPRIO_DRYRUN:-}" ]]; then
  printf '%s\n' "${prefix[@]}" "$@"
  exit 0
fi
exec "${prefix[@]}" "$@"
