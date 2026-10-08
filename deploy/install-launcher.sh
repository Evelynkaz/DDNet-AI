#!/usr/bin/env bash
# Installs the web launcher (task 5.9, D-089): the units that let the owner start and stop the bot from the site.
#
#   ddnet-ai-launch.path / ddnet-ai-launch.service   root: consume the web's request file, validate it, start/stop the bot
#   ddnet-ai-sparring@.service                      1-3 scripted local opponents (instances 1..3)
#   ddnet-ai-bot.service                            the bot unit, now reading its server/brain/duration from an environment file
#   ddnet-ai-servers.path / .service                task 5.12 (D-099): fetch the DDNet master list into the cache the site's «Серверы» tab reads
#   ddnet-ai-proxycheck.path / .service             task 5.12: the site's «Проверить» for a proxy profile (unprivileged, sandboxed)
#
# and the places they use: ~/aiddnet/data/launch (the web's request directory, the favourites and the proxy-check files; the status and the
# list of closed servers are in /run/ddnet-ai), ~/aiddnet/data/servers (the master-list cache, task 5.12), ~/aiddnet/data/sparring (the sparring
# opponents' own data), /etc/ddnet-ai, /var/lib/ddnet-ai and the ROOT-OWNED copy of the binary the root units run
# (/usr/local/libexec/ddnet-ai/ddnet-ai: a file the owner's user could rewrite must never be run as root).
#
# Idempotent: re-run it after every new `ddnet-ai` binary (install.sh puts it in ~/aiddnet/bin; this copies it to the root-owned
# place) and whenever the unit files change. Every unit and config file it replaces is first copied to
# /var/backups/ddnet-ai-launcher/<timestamp>/ (only when the content differs; the binary is not backed up; the last 3 backup
# directories are kept).
#
# It never starts or stops the bot and refuses to run while the bot unit is active. It does NOT install or restart the web unit
# (deploy/install.sh does, with the new ReadWritePaths for data/launch; task 5.12 changes nothing in the web unit: it gets no network
# right, only the new routes of the new binary) and never touches Caddy, ufw, the allow-list, live-servers.toml or the secrets.
#
# A hand-made drop-in on the bot unit (for example the old swarfey.conf with its own ExecStart= and IPAddressAllow=) would make the
# unit ignore the validated environment, so the helper refuses to start while one exists. This script therefore moves such drop-ins
# to the backup directory, but only with --take-over (it names them first and asks for nothing else).
#
# Usage: deploy/install-launcher.sh [--take-over] [--uninstall]
#   --take-over   move foreign drop-ins of ddnet-ai-bot.service to the backup directory (needed once, when one exists)
#   --uninstall   stop and remove the launcher (the bot unit file is restored from the newest backup that has one, if any)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SRC="$SCRIPT_DIR/systemd"
DDAI_USER="${DDAI_USER:-ubuntu}"
DATA_DIR="${DDAI_DATA_DIR:-$HOME/aiddnet/data}"
BIN_SRC="${DDAI_BIN:-$HOME/aiddnet/bin/ddnet-ai}"
LIBEXEC_DIR="/usr/local/libexec/ddnet-ai"
LIBEXEC_BIN="$LIBEXEC_DIR/ddnet-ai"
UNIT_DIR="/etc/systemd/system"
BOT_UNIT="ddnet-ai-bot.service"
DROPIN_DIR="$UNIT_DIR/$BOT_UNIT.d"
OWN_DROPIN="$DROPIN_DIR/50-launch.conf"
BACKUP_ROOT="/var/backups/ddnet-ai-launcher"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="$BACKUP_ROOT/$STAMP"
UNITS=(ddnet-ai-bot.service ddnet-ai-sparring@.service ddnet-ai-launch.service ddnet-ai-launch.path
       ddnet-ai-servers.service ddnet-ai-servers.path ddnet-ai-proxycheck.service ddnet-ai-proxycheck.path)

TAKE_OVER=0
UNINSTALL=0
for arg in "$@"; do
  case "$arg" in
    --take-over) TAKE_OVER=1 ;;
    --uninstall) UNINSTALL=1 ;;
    *) echo "install-launcher.sh: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '[deploy/install-launcher.sh] %s\n' "$*" >&2; }
die() { printf '[deploy/install-launcher.sh] ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(whoami)" == "$DDAI_USER" ]] || die "expected to run as $DDAI_USER, got $(whoami)"

# Copies $1 (an existing root-owned file) into the backup directory, keeping its path below it.
backup_file() {
  local file="$1"
  sudo install -d -m 0700 "$BACKUP_DIR$(dirname "$file")"
  sudo cp -a "$file" "$BACKUP_DIR$file"
  log "backed up $file -> $BACKUP_DIR$file"
}

# Installs $1 (source) as $2 (destination, mode $3, owner root) and backs the old one up when its content differs ($4 = nobackup
# skips the backup: the binary is hundreds of MB and can be rebuilt or copied again from ~/aiddnet/bin).
install_file() {
  local src="$1" dst="$2" mode="$3" backup="${4:-backup}"
  if [[ "$backup" == backup ]] && sudo test -e "$dst" && ! sudo cmp -s "$src" "$dst"; then
    backup_file "$dst"
  fi
  sudo install -o root -g root -m "$mode" "$src" "$dst"
}

bot_is_active() { systemctl is-active --quiet "$BOT_UNIT" || systemctl is-active --quiet 'ddnet-ai-sparring@*.service'; }

if [[ "$UNINSTALL" -eq 1 ]]; then
  bot_is_active && die "the bot or a sparring unit is active: stop it first (the launcher never stops a run by itself here)"
  log "uninstalling the launcher"
  sudo systemctl disable --now ddnet-ai-launch.path ddnet-ai-servers.path ddnet-ai-proxycheck.path 2>/dev/null || true
  for f in ddnet-ai-launch.path ddnet-ai-launch.service ddnet-ai-sparring@.service ddnet-ai-servers.path ddnet-ai-servers.service \
           ddnet-ai-proxycheck.path ddnet-ai-proxycheck.service; do
    if sudo test -e "$UNIT_DIR/$f"; then backup_file "$UNIT_DIR/$f"; sudo rm -f "$UNIT_DIR/$f"; fi
  done
  for f in "$OWN_DROPIN" /etc/ddnet-ai/bot-launch.env; do
    if sudo test -e "$f"; then backup_file "$f"; sudo rm -f "$f"; fi
  done
  # The bot unit file: the version from before the launcher, if one was kept.
  old="$BACKUP_ROOT/original$UNIT_DIR/$BOT_UNIT"
  sudo test -f "$old" || old=""
  if [[ -n "$old" ]]; then
    sudo install -o root -g root -m 0644 "$old" "$UNIT_DIR/$BOT_UNIT"
    log "restored $UNIT_DIR/$BOT_UNIT from $old"
  else
    log "no earlier copy of $BOT_UNIT in $BACKUP_ROOT: it was left as it is (it still works by hand, with its defaults)"
  fi
  sudo systemctl daemon-reload
  log "done. Left in place: $LIBEXEC_BIN, /var/lib/ddnet-ai, $DATA_DIR/launch (favourites.json included), $DATA_DIR/servers, $DATA_DIR/sparring (remove by hand if wanted)."
  exit 0
fi

# ---------------------------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------------------------
[[ -x "$BIN_SRC" ]] || die "$BIN_SRC is not an executable (build and install the binary first: deploy/install.sh)"
for u in "${UNITS[@]}"; do [[ -f "$UNIT_SRC/$u" ]] || die "missing $UNIT_SRC/$u"; done
# A test build (cargo test, the e2e) accepts loopback favourites: it must never become the root helper (task 5.12).
if "$BIN_SRC" --version 2>/dev/null | grep -q 'loopback-favourites'; then
  die "$BIN_SRC is a test build (+loopback-favourites): build and install a release binary with deploy/install.sh"
fi
"$BIN_SRC" launch --help >/dev/null 2>&1 || die "$BIN_SRC has no 'launch' subcommand: it is an older build"
# Task 5.15: the bot unit passes `--no-selfkill=${BOT_NO_SELFKILL}`, which a binary from before 5.15 rejects (every bot start would fail).
# The new binary's help prints `--no-selfkill[=<...>]`. (The help is captured first: `grep -q` closing a pipe early fails under pipefail.)
play_help="$("$BIN_SRC" play --help 2>&1 || true)"
grep -q -- '--no-selfkill\[=' <<<"$play_help" || die "$BIN_SRC is an older build (no '--no-selfkill=true|false' form, task 5.15): run deploy/install.sh first, so the binary is newer than the unit"
# Task 3.17: the unit also passes `--window-model=${BOT_WINDOW_MODEL}` (empty = off), which a binary from before 3.17 rejects as an unknown flag.
grep -q -- '--window-model' <<<"$play_help" || die "$BIN_SRC is an older build (no '--window-model', task 3.17): run deploy/install.sh first, so the binary is newer than the unit"
# Task 3.20b: the unit also passes `--preinput ${BOT_PREINPUT}` (on|off), which a binary from before 3.20 rejects as an unknown flag.
grep -q -- '--preinput' <<<"$play_help" || die "$BIN_SRC is an older build (no '--preinput', task 3.20): run deploy/install.sh first, so the binary is newer than the unit"
if bot_is_active; then
  die "ddnet-ai-bot.service or a sparring unit is active: not touching a running bot. Let it finish (or stop it) and run this again."
fi

# Foreign drop-ins on the bot unit (everything but our own 50-launch.conf).
foreign=()
if sudo test -d "$DROPIN_DIR"; then
  while IFS= read -r f; do
    [[ -n "$f" && "$f" != "$OWN_DROPIN" ]] && foreign+=("$f")
  done < <(sudo find "$DROPIN_DIR" -maxdepth 1 -type f -name '*.conf' | sort)
fi
if [[ "${#foreign[@]}" -gt 0 ]]; then
  log "drop-ins on $BOT_UNIT that the launcher does not own:"
  for f in "${foreign[@]}"; do log "  $f"; done
  [[ "$TAKE_OVER" -eq 1 ]] || die "they would override the launcher's environment (the helper refuses to start while they exist). Re-run with --take-over to move them to $BACKUP_DIR."
fi

# ---------------------------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------------------------
# The web writes request.json here (the status is in /run/ddnet-ai, root-owned): owned by the owner's user (the web unit's user), readable by root.
mkdir -p "$DATA_DIR/launch"
chmod 0755 "$DATA_DIR/launch"
# Task 5.12: the master-list cache. Written by ddnet-ai-servers.service (the owner's user, sandboxed), only read by the web. The refresh
# trigger the web rewrites exists from the start so the path unit has something to watch.
mkdir -p "$DATA_DIR/servers"
chmod 0755 "$DATA_DIR/servers"
[[ -e "$DATA_DIR/launch/servers-refresh" ]] || : >"$DATA_DIR/launch/servers-refresh"
# The proxy profiles the site writes live in $DATA_DIR/secrets (0700, 0600 files), which exists since `web-passwd`.
[[ -d "$DATA_DIR/secrets" ]] || die "$DATA_DIR/secrets is missing: run \`ddnet-ai web-passwd\` first (deploy/install.sh)"
# The sparring opponents' private data (logs, map cache): nothing of the real bot's.
mkdir -p "$DATA_DIR/sparring"
chmod 0700 "$DATA_DIR/sparring"
mkdir -p "$DATA_DIR/run" "$DATA_DIR/logs"
sudo install -d -o root -g root -m 0755 /etc/ddnet-ai "$LIBEXEC_DIR"
sudo install -d -o root -g root -m 0700 /var/lib/ddnet-ai
sudo install -d -o root -g root -m 0755 "$DROPIN_DIR"

# ---------------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------------
if [[ "${#foreign[@]}" -gt 0 ]]; then
  for f in "${foreign[@]}"; do
    backup_file "$f"
    sudo rm -f "$f"
    log "moved away the foreign drop-in $f"
  done
fi

# The root-owned copy of the binary (the root units never run the user-writable one).
tmp_bin="$(mktemp)"
trap 'rm -f "$tmp_bin"' EXIT
cp "$BIN_SRC" "$tmp_bin"
install_file "$tmp_bin" "$LIBEXEC_BIN" 0755 nobackup
log "installed $LIBEXEC_BIN ($("$LIBEXEC_BIN" --version 2>/dev/null || echo ddnet-ai))"

# The bot unit as it was before the launcher, kept for good in $BACKUP_ROOT/original (never pruned; --uninstall restores it).
ORIGINAL="$BACKUP_ROOT/original$UNIT_DIR/$BOT_UNIT"
if sudo test -e "$UNIT_DIR/$BOT_UNIT" && ! sudo grep -q 'bot-launch.env' "$UNIT_DIR/$BOT_UNIT" && ! sudo test -e "$ORIGINAL"; then
  sudo install -d -m 0700 "$(dirname "$ORIGINAL")"
  sudo cp -a "$UNIT_DIR/$BOT_UNIT" "$ORIGINAL"
  log "kept the original $BOT_UNIT in $ORIGINAL"
fi
for u in "${UNITS[@]}"; do
  install_file "$UNIT_SRC/$u" "$UNIT_DIR/$u" 0644
done

# The launcher's own cgroup drop-in, as the helper writes it for the local server (loopback only): present from the start, so the
# unit's drop-in list is exactly this one file.
own_tmp="$(mktemp)"
cat >"$own_tmp" <<'DROPIN'
# Written by `ddnet-ai launch apply` (root): the addresses the bot unit may talk to for the current launch. Do not edit.
[Service]
IPAddressAllow=
IPAddressAllow=127.0.0.0/8 ::1
DROPIN
if ! sudo test -e "$OWN_DROPIN"; then
  sudo install -o root -g root -m 0644 "$own_tmp" "$OWN_DROPIN"
fi
rm -f "$own_tmp"

# Keep the last 3 backup directories only.
if sudo test -d "$BACKUP_ROOT"; then
  sudo find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d -name '2???????T??????Z' | sort | head -n -3 | while IFS= read -r old; do
    sudo rm -rf -- "$old"
    log "pruned the old backup $old"
  done
fi

sudo systemctl daemon-reload
sudo systemctl enable ddnet-ai-launch.path ddnet-ai-servers.path ddnet-ai-proxycheck.path >/dev/null
sudo systemctl restart ddnet-ai-launch.path ddnet-ai-servers.path ddnet-ai-proxycheck.path
sleep 1
for p in ddnet-ai-launch.path ddnet-ai-servers.path ddnet-ai-proxycheck.path; do
  log "--- $p ---"
  systemctl status --no-pager "$p" || true
done
log "done. Backups (if anything was replaced): $BACKUP_ROOT. The site's «Запуск» card and «Серверы» tab need the new binary (installed above) and the web restarted: sudo systemctl restart ddnet-ai-web."
log "rollback: deploy/install-launcher.sh --uninstall (stop the bot first)."
