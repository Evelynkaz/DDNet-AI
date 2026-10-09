#!/usr/bin/env bash
# Deploys the DDNet-AI web UI behind Caddy HTTPS (task 5.3): installs Caddy from its official
# apt repository, opens 80/443 in ufw, builds+installs the `ddnet-ai` binary to a stable path,
# and installs/(re)starts both systemd units. See deploy/README.md for the full picture,
# docs/SETUP.md for exactly what got installed, and CLAUDE.md for the folder layout this assumes
# (~/aiddnet/data, ~/aiddnet/DDNet-AI).
#
# Idempotent: safe to re-run after a `git pull` to rebuild+redeploy (an "update"). Never touches
# ufw's default policy or existing rules besides adding the two it needs, and never runs
# `ufw reset`/`ufw disable`.
#
# What it does NOT do (deliberately, see deploy/README.md and the task's constraints):
#   - Does not run `ddnet-ai web-passwd` — that is a separate, explicit, one-time (or
#     "regenerate the password") operation the operator runs by hand.
#   - Does not print or touch any secret.
#
# Usage: deploy/install.sh [--no-ufw] [--skip-build] [--skip-apt]
#   --no-ufw      Don't touch ufw at all (e.g. it's already configured exactly as needed).
#   --skip-build  Reuse whatever is already at $BIN_PATH instead of rebuilding it (faster
#                 redeploy of just the Caddy/systemd-unit side after a config-only change).
#   --skip-apt    Don't run apt at all (task 5.19, D-130): Caddy must already be installed. A redeploy
#                 does this by itself when Caddy is installed and its keyring (fingerprint verified)
#                 and sources list are in place; the flag is for forcing it, e.g. when an apt
#                 repository is down or apt is locked and the automatic check still wants apt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DDAI_USER="${DDAI_USER:-ubuntu}"
DATA_DIR="${DDAI_DATA_DIR:-$HOME/aiddnet/data}"
BIN_DIR="$HOME/aiddnet/bin"
BIN_PATH="$BIN_DIR/ddnet-ai"
CADDYFILE_SRC="$SCRIPT_DIR/caddy/Caddyfile"
CADDYFILE_DST="/etc/caddy/Caddyfile"
UNIT_SRC="$SCRIPT_DIR/systemd/ddnet-ai-web.service"
UNIT_DST="/etc/systemd/system/ddnet-ai-web.service"
CADDY_KEYRING="/etc/apt/keyrings/caddy-stable-archive-keyring.gpg"
CADDY_SOURCES_LIST="/etc/apt/sources.list.d/caddy-stable.list"
CADDY_REPO_URL="https://dl.cloudsmith.io/public/caddy/stable/deb/debian"
# How long (seconds) a first install waits for the apt/dpkg lock before giving up: unattended-upgrades
# can hold it for minutes (task 5.19). `DPkg::Lock::Timeout` covers the dpkg frontend lock only; the
# lists lock of `apt-get update` is not covered (apt 2.8.3 fails at once), so `apt_run` also retries
# on lock errors every $APT_LOCK_POLL seconds until the same deadline. The DDAI_* variables exist so
# the harness test (crates/ddnet-ai/tests/deploy_units.rs) can run with a short deadline.
APT_LOCK_TIMEOUT="${DDAI_APT_LOCK_TIMEOUT:-600}"
APT_LOCK_POLL="${DDAI_APT_LOCK_POLL:-10}"
# The Caddy signing key's well-known fingerprint (Caddy Web Server <contact@caddyserver.com>,
# rsa4096/155B6D79CA56EA34) — checked (review finding F1: strictly, before the file is ever
# installed anywhere apt would read it from) so a compromised/wrong download is a hard failure
# here, not a silently-trusted repo. Compared with spaces stripped from both sides (see
# `normalize_fpr` below), so its exact spacing here is only for human readability. Recorded in
# docs/SETUP.md alongside how/when it was verified.
CADDY_KEY_FINGERPRINT="6576 0C51 EDEA 2017 CEA2 CA15 155B 6D79 CA56 EA34"

# Strips spaces and uppercases, so a pinned fingerprint written with GnuPG's traditional
# human-readable grouping compares equal to one extracted from `--with-colons` output (which has
# no spaces at all) without needing both call sites to agree on formatting.
normalize_fpr() { tr -d ' ' <<<"$1" | tr '[:lower:]' '[:upper:]'; }

DO_UFW=1
DO_BUILD=1
DO_APT=1
for arg in "$@"; do
  case "$arg" in
    --no-ufw) DO_UFW=0 ;;
    --skip-build) DO_BUILD=0 ;;
    --skip-apt) DO_APT=0 ;;
    *) echo "install.sh: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '[deploy/install.sh] %s\n' "$*" >&2; }
die() { printf '[deploy/install.sh] ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(whoami)" == "$DDAI_USER" ]] || die "expected to run as $DDAI_USER, got $(whoami)"

# ---------------------------------------------------------------------------------------------
# 1. ufw: open 80/443, never touch anything else. Keep SSH working at every step (never disable
#    or reset ufw) — acceptance criterion 3 / the task's hard constraint.
# ---------------------------------------------------------------------------------------------
if [[ "$DO_UFW" -eq 1 ]]; then
  log "ufw: ensuring 22/tcp stays allowed, and allowing 80/tcp + 443/tcp (v4 and v6)"
  sudo ufw status verbose | grep -q '^22/tcp .*ALLOW IN' || die "22/tcp is not already allowed in ufw; refusing to proceed (would risk locking out SSH)"
  # `ufw allow <port>/tcp` covers both IPv4 and IPv6 in one rule when IPv6 is enabled in
  # /etc/default/ufw (the default on Ubuntu, and what 22/tcp above already proves is in effect
  # here: `ufw status verbose` shows a "(v6)" line for it) — no separate v6-specific syntax
  # needed. Re-running `allow` for a rule that already exists is a no-op ("Skipping adding
  # existing rule"), so this is safe to repeat.
  sudo ufw allow 80/tcp comment 'caddy http (task 5.3)'
  sudo ufw allow 443/tcp comment 'caddy https (task 5.3)'
else
  log "ufw: --no-ufw given, not touching it"
fi

# ---------------------------------------------------------------------------------------------
# 2. Caddy: official apt repository (Cloudsmith), not the old Ubuntu universe package (2.6.2,
#    missing ~2 years of security fixes vs the 2.11.x this pulls — see docs/SETUP.md).
# ---------------------------------------------------------------------------------------------
# Review finding F1: verifies a keyring file holds EXACTLY ONE public key, and that its
# fingerprint matches the pin — not just "the first key's fingerprint matches" (apt trusts EVERY
# `pub` entry a keyring file contains, so a file holding the real key plus a second,
# attacker-controlled one previously passed this check as long as the real one happened to come
# first). Dies (does not echo a fingerprint at all) if the file is missing/unreadable.
verify_caddy_keyring() {
  local keyfile="$1" colons pub_count fpr
  colons="$(gpg --show-keys --with-colons "$keyfile" 2>/dev/null)" || die "could not read key material from $keyfile"
  pub_count="$(grep -c '^pub:' <<<"$colons")"
  [[ "$pub_count" == "1" ]] || die "expected exactly 1 public key in the Caddy signing keyring ($keyfile), found $pub_count — refusing (apt trusts every key a keyring file contains)"
  # The colon-format `fpr` record's 10th field is the fingerprint (no spaces); with exactly one
  # `pub` line just confirmed above, the FIRST `fpr:` line in the whole file is unambiguously that
  # one key's own primary fingerprint (any subkeys' `fpr:` lines follow after it).
  fpr="$(awk -F: '/^fpr:/{print $10; exit}' <<<"$colons")"
  [[ -n "$fpr" ]] || die "could not extract a fingerprint from $keyfile"
  [[ "$(normalize_fpr "$fpr")" == "$(normalize_fpr "$CADDY_KEY_FINGERPRINT")" ]] \
    || die "Caddy signing key fingerprint mismatch: got [$fpr], expected [$CADDY_KEY_FINGERPRINT]"
  log "Caddy signing key fingerprint verified: $fpr (exactly 1 public key in the file)"
}

# Runs `sudo apt-get`, waiting up to $APT_LOCK_TIMEOUT s for a held lock (task 5.19): the dpkg lock is
# waited for by apt itself (`DPkg::Lock::Timeout`), the lists lock of `update` by the retry loop here.
# apt's output is streamed and also kept; `--force-confold` answers dpkg's conffile questions (a
# prompt would look like a hang under the captured output). On failure it dies with a message that
# names the repository apt complained about (a down third-party repo, e.g. Caddy's answering 402,
# must be recognisable at a glance) and says how to proceed when Caddy is already installed.
apt_run() {
  local out_file rc errlines repos start="$SECONDS"
  out_file="$(mktemp)"
  log "apt-get $1: may wait up to ${APT_LOCK_TIMEOUT}s for the apt/dpkg lock"
  while :; do
    if sudo apt-get -o "DPkg::Lock::Timeout=$APT_LOCK_TIMEOUT" -o "Dpkg::Options::=--force-confold" "$@" 2>&1 | tee "$out_file" >&2; then
      rc=0
    else
      rc="${PIPESTATUS[0]}"
    fi
    if [[ "$rc" -eq 0 ]]; then
      rm -f "$out_file"
      return 0
    fi
    # Only apt's own error lines (Err:/E:/W:) say what failed: progress lines such as "Waiting for
    # cache lock: Could not get lock ..." (printed during a wait that then succeeded) must not count.
    errlines="$(grep -E '^(Err:|E:|W:)' "$out_file" || true)"
    if grep -qE '^E: (Could not get lock|Unable to lock directory|Unable to acquire the dpkg frontend lock)' <<<"$errlines"; then
      if (( SECONDS - start < APT_LOCK_TIMEOUT )); then
        log "apt-get $1: the apt/dpkg lock is held, retrying in ${APT_LOCK_POLL}s"
        sleep "$APT_LOCK_POLL"
        continue
      fi
      rm -f "$out_file"
      die "apt-get $1 failed: the apt/dpkg lock is still held after waiting $((SECONDS - start))s (unattended-upgrades or another apt running?). Retry later, or pass --skip-apt if Caddy is already installed"
    fi
    break
  done
  rm -f "$out_file"
  repos="$(grep -oE 'https?://[^ ]+' <<<"$errlines" | sort -u | tr '\n' ' ' || true)"
  if grep -q 'dl.cloudsmith.io/public/caddy' <<<"$errlines"; then
    die "apt-get $1 failed: the Caddy apt repository ($CADDY_REPO_URL) is failing (see the apt output above; e.g. 402 Payment Required / no longer signed). If Caddy is already installed, re-run with --skip-apt"
  fi
  die "apt-get $1 failed (exit $rc); repository named in the apt errors: ${repos:-none, see the apt output above}. If Caddy is already installed, re-run with --skip-apt"
}

caddy_package_installed() {
  # Status is "<want> <flag> <state>": any want (install, hold, ...) counts as long as it is installed.
  [[ "$(dpkg-query -W -f='${Status}' caddy 2>/dev/null || true)" == *" ok installed" ]]
}

if [[ "$DO_APT" -eq 0 ]]; then
  # Explicit opt-out: no apt, no key download, no sources list. Caddy itself must be there, since
  # everything below needs it (`caddy validate`, `caddy adapt`, the service).
  log "--skip-apt given, not touching apt"
  command -v caddy >/dev/null 2>&1 || die "--skip-apt given but caddy is not installed: run deploy/install.sh without --skip-apt for the first install"
  # The unattended-upgrades drop-in below trusts this repo's origin, so a keyring that is there is
  # still checked, as on every run before 5.19.
  if [[ -f "$CADDY_KEYRING" ]]; then
    verify_caddy_keyring "$CADDY_KEYRING"
  fi
elif caddy_package_installed && [[ -f "$CADDY_KEYRING" && -f "$CADDY_SOURCES_LIST" ]]; then
  # A redeploy of a working host needs no apt (task 5.19): an unreachable Caddy repo or a held
  # apt lock must not block it. The keyring is still checked (exactly one key, pinned fingerprint).
  log "Caddy package, keyring and apt source are already in place: skipping apt (no update, no install)"
  verify_caddy_keyring "$CADDY_KEYRING"
else
  log "using apt: caddy installed=$(caddy_package_installed && echo yes || echo no), keyring present=$([[ -f "$CADDY_KEYRING" ]] && echo yes || echo no), apt source present=$([[ -f "$CADDY_SOURCES_LIST" ]] && echo yes || echo no) (first install, or one of them is missing)"
  log "installing Caddy's apt prerequisites (debian-keyring, debian-archive-keyring, apt-transport-https)"
  apt_run update -qq
  apt_run install -y -qq debian-keyring debian-archive-keyring apt-transport-https

  if [[ -f "$CADDY_KEYRING" ]]; then
    log "Caddy signing key already present at $CADDY_KEYRING, checking its fingerprint"
    verify_caddy_keyring "$CADDY_KEYRING"
  else
    log "fetching Caddy's signing key"
    # A fresh `mktemp -d` every run (review finding F1): fixed /tmp filenames meant a leftover file
    # from an interrupted previous run broke `gpg --dearmor`'s refusal to overwrite an existing
    # output file. Verified BEFORE `sudo install` ever copies anything into /etc/apt/keyrings — a
    # failing check here must leave nothing behind for apt to pick up.
    KEY_TMPDIR="$(mktemp -d)"
    trap 'rm -rf "$KEY_TMPDIR"' EXIT
    curl -fsSL 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' -o "$KEY_TMPDIR/gpg.key"
    gpg --batch --yes --dearmor -o "$KEY_TMPDIR/keyring.gpg" "$KEY_TMPDIR/gpg.key"
    verify_caddy_keyring "$KEY_TMPDIR/keyring.gpg"
    sudo install -o root -g root -m 0644 "$KEY_TMPDIR/keyring.gpg" "$CADDY_KEYRING"
    rm -rf "$KEY_TMPDIR"
    trap - EXIT
  fi

  if [[ -f "$CADDY_SOURCES_LIST" ]]; then
    log "Caddy apt source already present at $CADDY_SOURCES_LIST"
  else
    log "adding Caddy apt source at $CADDY_SOURCES_LIST (signed-by $CADDY_KEYRING, per /etc/apt/keyrings convention)"
    cat <<EOF | sudo tee "$CADDY_SOURCES_LIST" >/dev/null
# Source: Caddy (official, task 5.3)
# Site: https://github.com/caddyserver/caddy
# Repository: Caddy / stable (Cloudsmith)
# Key fingerprint: $CADDY_KEY_FINGERPRINT (Caddy Web Server <contact@caddyserver.com>)
deb [signed-by=$CADDY_KEYRING] $CADDY_REPO_URL any-version main
EOF
  fi

  log "apt-get update + install caddy"
  apt_run update -qq
  apt_run install -y caddy
fi
caddy version

# Review finding F5: Ubuntu's own 50unattended-upgrades doesn't cover this repo's origin, so an
# internet-facing Caddy would otherwise never get automatic security updates. Safe to install
# unconditionally: apt.conf.d list options accumulate across files, this only adds one pattern
# on top of Ubuntu's own defaults, never replaces them (see the file's own header comment).
log "installing $UNATTENDED_UPGRADES_SRC -> $UNATTENDED_UPGRADES_DST (Caddy origin for unattended-upgrades)"
sudo install -m 0644 "$UNATTENDED_UPGRADES_SRC" "$UNATTENDED_UPGRADES_DST"

# ---------------------------------------------------------------------------------------------
# 3. Build and install the bot binary to a stable path (updates are this script re-run
#    explicitly; no auto-update).
# ---------------------------------------------------------------------------------------------
if [[ "$DO_BUILD" -eq 1 ]]; then
  log "building ddnet-ai (release) — CARGO_BUILD_JOBS=4 (other builds may be running on this host)"
  if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
  fi
  command -v cargo >/dev/null 2>&1 || die "cargo not found on PATH even after sourcing ~/.cargo/env"
  ( cd "$REPO_ROOT" && CARGO_BUILD_JOBS=4 cargo build --release --locked -p ddnet-ai )
  mkdir -p "$BIN_DIR"
  # A test build accepts loopback favourites (task 5.12): never install one.
  if "$REPO_ROOT/target/release/ddnet-ai" --version 2>/dev/null | grep -q 'loopback-favourites'; then
    die "target/release/ddnet-ai is a test build (+loopback-favourites); rebuild without that feature"
  fi
  install -m 0755 "$REPO_ROOT/target/release/ddnet-ai" "$BIN_PATH"
  log "installed $("$BIN_PATH" --version 2>/dev/null || echo ddnet-ai) -> $BIN_PATH"
else
  log "--skip-build given, reusing existing $BIN_PATH"
  [[ -x "$BIN_PATH" ]] || die "--skip-build given but $BIN_PATH does not exist/is not executable"
  if "$BIN_PATH" --version 2>/dev/null | grep -q 'loopback-favourites'; then
    die "$BIN_PATH is a test build (+loopback-favourites): rebuild without that feature"
  fi
fi

# ---------------------------------------------------------------------------------------------
# 4. Pre-create the two subtrees the web service writes to, before the hardened unit's
#    ProtectSystem=strict + ReadWritePaths ever apply to them (see the unit file's comments).
#    `ddnet-ai web-passwd` (run separately, by hand) already creates secrets/ — this only
#    guarantees logs/web/ also exists up front.
# ---------------------------------------------------------------------------------------------
mkdir -p "$DATA_DIR/logs/web"
mkdir -p "$DATA_DIR/secrets"
# Task 5.6: the unit has ReadWritePaths on bot/ (the friends editor writes relations.json there); the bot (ddnet-ai play)
# uses the same directory for its sockets. 0700: it holds nicknames and the control socket.
mkdir -p "$DATA_DIR/bot"
chmod 700 "$DATA_DIR/bot"
chmod 0700 "$DATA_DIR/secrets"
# Task 5.9 (D-089): the web unit has ReadWritePaths on launch/ (the launcher's request file is written there; a root path unit does
# the rest, deploy/install-launcher.sh). It must exist before the unit starts, or the unit fails with status 226/NAMESPACE.
mkdir -p "$DATA_DIR/launch"
chmod 0755 "$DATA_DIR/launch"

if [[ ! -f "$DATA_DIR/secrets/web-auth.toml" ]]; then
  log "NOTE: no password has been set up yet ($DATA_DIR/secrets/web-auth.toml is missing)."
  log "      Run '$BIN_PATH web-passwd' once, by hand, before relying on this deployment."
fi

# ---------------------------------------------------------------------------------------------
# 5. Install the systemd unit and the Caddyfile, then (re)start both services.
# ---------------------------------------------------------------------------------------------
log "installing $UNIT_DST"
sudo install -m 0644 "$UNIT_SRC" "$UNIT_DST"
sudo systemctl daemon-reload
sudo systemctl enable ddnet-ai-web.service >/dev/null

# Review finding F11: `caddy adapt` alone only checks Caddyfile SYNTAX — a config that parses
# fine but fails to actually LOAD (a bad log-file path/permission, a bad matcher, a TLS policy
# error, ...) would previously only be caught by the real `systemctl reload`/`start` below,
# potentially taking the live site down on a bad candidate. `caddy validate` catches that class of
# error up front instead, by actually provisioning the config (opening the log writer, etc.) —
# which is exactly why it must run as user `caddy` (round 1's own F1/F2-adjacent lesson: running
# THIS under sudo/root, as an earlier version of this script did, previously left
# /var/log/caddy/access.log root-owned and 0600 — unreadable/unwritable by the `caddy` user the
# real service runs as ever after). So /var/log/caddy must already exist (owned by `caddy`)
# before this runs, and the candidate file itself must be somewhere `caddy` can read: staged as a
# root-owned-but-world-readable temp file directly under /etc/caddy/, sidestepping any question of
# whether `caddy` can traverse this repo checkout's own directory permissions at all.
log "creating /var/log/caddy (owned by the caddy user, per the Caddyfile's access-log path)"
sudo install -d -o caddy -g caddy -m 0750 /var/log/caddy

CADDYFILE_CANDIDATE="/etc/caddy/.Caddyfile.candidate"
log "staging + really validating (provisioning, not just parsing) the candidate Caddyfile as user caddy"
sudo install -m 0644 "$CADDYFILE_SRC" "$CADDYFILE_CANDIDATE"
if ! sudo -u caddy caddy validate --config "$CADDYFILE_CANDIDATE" --adapter caddyfile; then
  sudo rm -f "$CADDYFILE_CANDIDATE"
  die "the candidate Caddyfile failed real validation (see the error above) — refusing to install it; the currently-installed $CADDYFILE_DST is untouched"
fi
sudo rm -f "$CADDYFILE_CANDIDATE"

# Whether a plain reload can even reach the CURRENTLY RUNNING instance depends on whether the
# admin API address is changing (finding F2's TCP->Unix-socket migration was exactly this case):
# `caddy reload` resolves the address to POST the new config to from the config it's given, not
# from wherever the running instance actually listens, so it can only work when those match.
# Decided here, once, by literally comparing the two — not by "try reload, guess why it failed".
get_admin_listen() {
  # Prints a Caddyfile's configured `admin.listen` address, falling back to Caddy's own built-in
  # default (localhost:2019) if adapting/parsing it doesn't cleanly yield one at all.
  caddy adapt --config "$1" --adapter caddyfile 2>/dev/null \
    | python3 -c "import json,sys; print(json.load(sys.stdin).get('admin',{}).get('listen','localhost:2019'))" 2>/dev/null \
    || echo "localhost:2019"
}
OLD_ADMIN_LISTEN=""
if [[ -f "$CADDYFILE_DST" ]]; then
  OLD_ADMIN_LISTEN="$(get_admin_listen "$CADDYFILE_DST")"
fi
NEW_ADMIN_LISTEN="$(get_admin_listen "$CADDYFILE_SRC")"

log "installing $CADDYFILE_DST"
sudo install -m 0644 "$CADDYFILE_SRC" "$CADDYFILE_DST"

if systemctl is-active --quiet ddnet-ai-web.service; then
  log "restarting ddnet-ai-web.service (already active)"
  sudo systemctl restart ddnet-ai-web.service
else
  log "starting ddnet-ai-web.service"
  sudo systemctl start ddnet-ai-web.service
fi

sudo systemctl enable caddy >/dev/null
if systemctl is-active --quiet caddy.service; then
  if [[ -n "$OLD_ADMIN_LISTEN" && "$OLD_ADMIN_LISTEN" != "$NEW_ADMIN_LISTEN" ]]; then
    # The admin address is changing: a plain reload cannot reach the OLD running instance to even
    # deliver the new config (see the comment above `get_admin_listen`), so go straight to a full
    # restart rather than let a reload fail first for a reason we already know about.
    log "admin API address is changing ($OLD_ADMIN_LISTEN -> $NEW_ADMIN_LISTEN): reload can't reach the old one, restarting instead"
    sudo systemctl restart caddy.service
  else
    log "reloading caddy.service (graceful; already active)"
    # Review finding F11: no fallback-to-restart here. The admin address (the ONE known,
    # legitimate reason a reload can fail purely because of a config CHANGE, handled above) is
    # unchanged, and the candidate already passed a REAL `caddy validate` moments ago — so a
    # reload failing now points at something this script did not anticipate. Restarting anyway
    # would risk tearing down a working listener to install a config that just proved it can't
    # even reload cleanly; failing loudly and leaving the OLD config serving is the safer default.
    sudo systemctl reload caddy.service \
      || die "caddy reload failed even though the admin API address did not change and the candidate passed 'caddy validate' — the OLD config is still serving; investigate with 'journalctl -u caddy.service' before retrying (do not loop: ACME rate limits)"
  fi
else
  log "starting caddy.service"
  sudo systemctl start caddy.service
fi

sleep 1
log "--- ddnet-ai-web.service ---"
sudo systemctl status --no-pager ddnet-ai-web.service || true
log "--- caddy.service ---"
sudo systemctl status --no-pager caddy.service || true
log "--- ufw status ---"
sudo ufw status verbose
log "--- listening sockets ---"
sudo ss -tulpn
log "done. See deploy/README.md for operations (logs, password, rollback, closing the ports again)."
