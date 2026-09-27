#!/usr/bin/env bash
# Install and (by default) start the ddnet-local.service systemd unit for the
# local DDNet server. System unit running as the current user (see
# README.md "Установка и запуск" for the justification vs. a `--user` unit
# with lingering).
#
# What it does:
#   1. Runs setup-runtime.sh (idempotent) so storage.cfg, maps/, teehistorian/
#      and the secrets file all exist before the unit ever starts.
#   2. Copies this checkout's tools/ddnet-server/local.cfg to a STABLE path
#      under $WORKDIR (~/aiddnet/data/ddnet-server/local.cfg) and points the
#      unit's `-f` at that copy, not at the repo - so the running service
#      does not depend on this worktree/checkout still existing on disk.
#      Re-run this script after editing local.cfg to refresh the copy.
#   3. Renders /etc/systemd/system/ddnet-local.service:
#      - ExecStartPre fails the unit closed (won't start at all) if the
#        config or secrets file is missing/unreadable, instead of DDNet
#        silently falling back to the bundled data/autoexec_server.cfg
#        defaults (no bindaddr override -> listens on 0.0.0.0/::).
#      - ExecStart repeats bindaddr/sv_register/sv_ipv4only/logfile directly
#        as extra command-line arguments, on top of whatever local.cfg says,
#        as defense in depth.
#      - IPAddressAllow=127.0.0.0/8 ::1 + IPAddressDeny=any: kernel-level
#        (cgroup eBPF) traffic filtering, independent of any DDNet config -
#        a packet from/to a non-loopback address is dropped for this unit's
#        processes even if the application somehow ended up bound wider.
#   4. systemctl daemon-reload && enable, and (unless --no-start) start/restart it.
#
# Usage: tools/ddnet-server/install-service.sh [--no-start] [--no-enable]
#
# Env overrides:
#   DDNET_SERVER_BIN   default: $HOME/aiddnet/build/ddnet-20.1/build/DDNet-Server
#   DDNET_RUN_USER     default: current user ($(whoami))
#   DDNET_DATA_ROOT    default: $HOME/aiddnet/data
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DDNET_SERVER_BIN="${DDNET_SERVER_BIN:-$HOME/aiddnet/build/ddnet-20.1/build/DDNet-Server}"
DDNET_RUN_USER="${DDNET_RUN_USER:-$(whoami)}"
DDNET_DATA_ROOT="${DDNET_DATA_ROOT:-$HOME/aiddnet/data}"
WORKDIR="$DDNET_DATA_ROOT/ddnet-server"
LOG_DIR="$DDNET_DATA_ROOT/logs/ddnet-server"
SECRETS_FILE="$DDNET_DATA_ROOT/secrets/ddnet-server-secrets.cfg"
STABLE_CFG="$WORKDIR/local.cfg"
UNIT_NAME="ddnet-local.service"
UNIT_PATH="/etc/systemd/system/$UNIT_NAME"

DO_START=1
DO_ENABLE=1
for arg in "$@"; do
  case "$arg" in
    --no-start) DO_START=0 ;;
    --no-enable) DO_ENABLE=0 ;;
    *) echo "install-service.sh: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '[install-service.sh] %s\n' "$*" >&2; }
die() { printf '[install-service.sh] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -x "$DDNET_SERVER_BIN" ]] || die "server binary not found/executable at $DDNET_SERVER_BIN (run tools/ddnet-server/build.sh first)"

log "ensuring runtime layout (storage.cfg, maps/, secrets) via setup-runtime.sh"
"$SCRIPT_DIR/setup-runtime.sh"

log "copying local.cfg -> $STABLE_CFG (stable path, independent of this checkout)"
install -m 0644 "$SCRIPT_DIR/local.cfg" "$STABLE_CFG"

[[ -r "$SECRETS_FILE" ]] || die "secrets file missing/unreadable at $SECRETS_FILE (setup-runtime.sh should have created it)"

mkdir -p "$LOG_DIR"
# Pre-create so they're owned by DDNET_RUN_USER: systemd (root) would otherwise
# create+own them itself on first write via `append:`, which is harmless (the
# service can still write through the inherited fd) but leaves root-owned
# files under $HOME, which is confusing to find later.
sudo touch "$LOG_DIR/stdout.log" "$LOG_DIR/stderr.log"
sudo chown "$DDNET_RUN_USER:$DDNET_RUN_USER" "$LOG_DIR/stdout.log" "$LOG_DIR/stderr.log"

TMP_UNIT="$(mktemp)"
trap 'rm -f "$TMP_UNIT"' EXIT
cat > "$TMP_UNIT" <<EOF
[Unit]
Description=Local DDNet dedicated server 20.1 (loopback only, sv_register 0 - never public)
Documentation=file://$REPO_ROOT/tools/ddnet-server/README.md
After=network.target

[Service]
Type=simple
User=$DDNET_RUN_USER
Group=$DDNET_RUN_USER
WorkingDirectory=$WORKDIR
# Fail closed: if either file is missing/unreadable, the unit does not start
# at all, instead of DDNet silently running with the bundled
# data/autoexec_server.cfg defaults (which has no bindaddr override).
ExecStartPre=/usr/bin/test -r $STABLE_CFG
ExecStartPre=/usr/bin/test -r $SECRETS_FILE
# Two -f files (config, then secrets - both support absolute paths), plus
# bindaddr/sv_register/sv_ipv4only/logfile repeated directly as defense in
# depth on top of whatever local.cfg says.
ExecStart=$DDNET_SERVER_BIN -f $STABLE_CFG -f $SECRETS_FILE "bindaddr 127.0.0.1" "sv_register 0" "sv_ipv4only 1" "logfile $LOG_DIR/ddnet-server.log"
Restart=on-failure
RestartSec=2
StandardOutput=append:$LOG_DIR/stdout.log
StandardError=append:$LOG_DIR/stderr.log
NoNewPrivileges=true
# Kernel-level (cgroup eBPF) backstop: even if the application ended up bound
# to a wider address than intended, traffic to/from anything but loopback is
# dropped for this unit's processes.
IPAddressDeny=any
IPAddressAllow=127.0.0.0/8 ::1

[Install]
WantedBy=multi-user.target
EOF

log "installing $UNIT_PATH (cfg: $STABLE_CFG, binary: $DDNET_SERVER_BIN, user: $DDNET_RUN_USER)"
sudo install -m 0644 "$TMP_UNIT" "$UNIT_PATH"
sudo systemctl daemon-reload

if [[ "$DO_ENABLE" -eq 1 ]]; then
  sudo systemctl enable "$UNIT_NAME"
fi

if [[ "$DO_START" -eq 1 ]]; then
  if systemctl is-active --quiet "$UNIT_NAME"; then
    log "restarting $UNIT_NAME (already active)"
    sudo systemctl restart "$UNIT_NAME"
  else
    log "starting $UNIT_NAME"
    sudo systemctl start "$UNIT_NAME"
  fi
  sleep 1
  sudo systemctl status --no-pager "$UNIT_NAME" || true
else
  log "--no-start given, not starting. Start with: sudo systemctl start $UNIT_NAME"
fi
