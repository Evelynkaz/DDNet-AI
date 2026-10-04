#!/usr/bin/env bash
# Probe of systemd's cgroup IP filter (task 2.6b, D-088 amendment): shows on THIS machine's systemd what the bot unit's
# `IPAddressAllow=` / `IPAddressDeny=` lines really do, so the drop-in the launcher writes for a proxied server with a relay
# on another host ("relay = public") is backed by a measurement and not by memory of the man page.
#
# What it does (needs `sudo`, python3; touches nothing but its own scratch unit `ddai-ipprobe.service`):
#   - starts two UDP echo servers OUTSIDE the unit, on 127.0.0.2 ("the game server") and 127.0.0.3 ("a relay anywhere"),
#     plus one on ::1 ("a game server over IPv6");
#   - writes a scratch unit with the SAME two filter lines as deploy/systemd/ddnet-ai-bot.service and, per scenario,
#     a drop-in, runs it once (Type=oneshot) and has it probe each destination;
#   - prints one line per scenario and destination: OPEN (an echo came back), BLOCKED (sendto failed with EPERM: the
#     egress filter dropped it) or SENT (sent without error, nobody answers: 192.0.2.1 is a documentation address);
#   - checks each against the expectation and exits 1 on the first surprise.
# Nothing here talks to a game server. The one real-network probe (a DNS query to 1.1.1.1:53) is the same neutral target
# `proxy-check` uses. Not part of CI (needs root and a systemd manager). Run: tools/e2e/ipfilter_probe.sh
set -euo pipefail

UNIT=ddai-ipprobe
DIR=/run/systemd/system
WORK=$(mktemp -d)
PORT=$((20000 + RANDOM % 20000))
trap 'sudo systemctl stop "$UNIT.service" 2>/dev/null || true; sudo rm -rf "$DIR/$UNIT.service" "$DIR/$UNIT.service.d"; sudo systemctl daemon-reload; kill $(jobs -p) 2>/dev/null || true; rm -rf "$WORK"' EXIT

systemctl --version | head -1

# Echo servers outside the unit.
python3 - "$PORT" <<'PY' &
import socket, sys, threading
port = int(sys.argv[1])
def serve(family, addr):
    s = socket.socket(family, socket.SOCK_DGRAM)
    s.bind((addr, port))
    while True:
        data, peer = s.recvfrom(64)
        s.sendto(data, peer)
for fam, addr in ((socket.AF_INET, "127.0.0.2"), (socket.AF_INET, "127.0.0.3"), (socket.AF_INET6, "::1")):
    threading.Thread(target=serve, args=(fam, addr), daemon=True).start()
threading.Event().wait()
PY
sleep 0.5

cat >"$WORK/probe.py" <<'PY'
import socket, struct, sys
port = int(sys.argv[1])
def echo(family, dst):
    s = socket.socket(family, socket.SOCK_DGRAM)
    s.settimeout(0.7)
    try:
        s.sendto(b"x", (dst, port))
    except PermissionError:
        return "BLOCKED"
    try:
        s.recvfrom(16)
        return "OPEN"
    except socket.timeout:
        return "SENT"
def dns():
    # A neutral DNS query for example.com (A) to 1.1.1.1:53, the target `proxy-check` uses.
    q = struct.pack(">HHHHHH", 0x4444, 0x0100, 1, 0, 0, 0) + b"\x07example\x03com\x00" + struct.pack(">HH", 1, 1)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(1.5)
    try:
        s.sendto(q, ("1.1.1.1", 53))
    except PermissionError:
        return "BLOCKED"
    try:
        s.recvfrom(512)
        return "OPEN"
    except socket.timeout:
        return "SENT"
print("127.0.0.2   (game server v4)   ", echo(socket.AF_INET, "127.0.0.2"))
print("127.0.0.3   (relay on lo)      ", echo(socket.AF_INET, "127.0.0.3"))
print("::ffff:127.0.0.2 (mapped form) ", echo(socket.AF_INET6, "::ffff:127.0.0.2"))
print("::1         (game server v6)   ", echo(socket.AF_INET6, "::1"))
print("::ffff:127.0.0.3 (mapped relay)", echo(socket.AF_INET6, "::ffff:127.0.0.3"))
print("192.0.2.1   (anywhere else)    ", echo(socket.AF_INET, "192.0.2.1"))
for dst in ("10.0.0.1", "172.16.0.1", "192.168.0.1", "169.254.169.254", "100.64.0.1"):
    print(dst.ljust(11), "(private range)       ", echo(socket.AF_INET, dst))
print("fd00::1     (v6 unique-local)  ", echo(socket.AF_INET6, "fd00::1"))
print("2606:4700:4700::1111 (v6 public)", echo(socket.AF_INET6, "2606:4700:4700::1111"))
print("1.1.1.1:53  (neutral DNS)      ", dns())
PY
chmod 644 "$WORK/probe.py"
chmod 755 "$WORK"

BASE="[Unit]
Description=scratch probe of the cgroup IP filter (task 2.6b)
[Service]
Type=oneshot
User=$(id -un)
# The two lines of deploy/systemd/ddnet-ai-bot.service:
IPAddressAllow=127.0.0.0/8 ::1
IPAddressDeny=any
ExecStart=/usr/bin/python3 $WORK/probe.py $PORT
StandardOutput=file:$WORK/out.txt"

run_scenario() { # name, drop-in body ("" = none)
  local name=$1 dropin=$2
  sudo rm -rf "$DIR/$UNIT.service.d"
  printf '%s\n' "$BASE" | sudo tee "$DIR/$UNIT.service" >/dev/null
  if [[ -n $dropin ]]; then
    sudo mkdir -p "$DIR/$UNIT.service.d"
    printf '[Service]\n%s\n' "$dropin" | sudo tee "$DIR/$UNIT.service.d/50.conf" >/dev/null
  fi
  sudo systemctl daemon-reload
  : >"$WORK/out.txt"; chmod 666 "$WORK/out.txt"
  sudo systemctl start "$UNIT.service"
  echo "=== $name"
  sed 's/^/    /' "$WORK/out.txt"
  RESULT=$(cat "$WORK/out.txt")
}

expect() { # destination-prefix, state
  if ! grep -q "^$1 .*$2\$" <<<"$RESULT"; then echo "UNEXPECTED: wanted '$1' = $2"; exit 1; fi
}

run_scenario "S1 shipped unit, no drop-in (allow loopback, deny any)" ""
expect "127.0.0.2" OPEN; expect "127.0.0.3" OPEN; expect "192.0.2.1" BLOCKED; expect "1.1.1.1:53" BLOCKED

run_scenario "S2 only 'IPAddressDeny=127.0.0.2' added (no reset): ALLOW WINS, the deny is dead" "IPAddressDeny=127.0.0.2"
expect "127.0.0.2" OPEN; expect "192.0.2.1" BLOCKED

run_scenario "S3 'IPAddressAllow=' reset only: Deny=any still holds, nothing leaves" "IPAddressAllow="
expect "127.0.0.2" BLOCKED; expect "192.0.2.1" BLOCKED; expect "1.1.1.1:53" BLOCKED

run_scenario "S4 'IPAddressDeny=' reset only: no filter at all, the game server is reachable" "IPAddressDeny="
expect "127.0.0.2" OPEN; expect "192.0.2.1" SENT; expect "1.1.1.1:53" OPEN

run_scenario "S5 THE public-relay drop-in: reset both, deny exactly the game server's IPs" \
"IPAddressAllow=
IPAddressDeny=
IPAddressDeny=127.0.0.2
IPAddressDeny=::1"
expect "127.0.0.2" BLOCKED; expect "::ffff:127.0.0.2" BLOCKED; expect "::1" BLOCKED
expect "127.0.0.3" OPEN; expect "192.0.2.1" SENT; expect "1.1.1.1:53" OPEN

run_scenario "S6 same, written on ONE line (space-separated list)" \
"IPAddressAllow=
IPAddressDeny=
IPAddressDeny=127.0.0.2 ::1"
expect "127.0.0.2" BLOCKED; expect "::1" BLOCKED; expect "127.0.0.3" OPEN

run_scenario "S7 the mistake to avoid: 'IPAddressAllow=any' next to the deny list: allow wins, the server is reachable" \
"IPAddressAllow=any
IPAddressDeny=
IPAddressDeny=127.0.0.2"
expect "127.0.0.2" OPEN

run_scenario "S8 THE FINAL launcher drop-in for an IPv4 server (2.6b review F3): server + ::/0 + the private ranges" \
"IPAddressAllow=
IPAddressDeny=
IPAddressDeny=127.0.0.2
IPAddressDeny=::/0
IPAddressDeny=10.0.0.0/8
IPAddressDeny=172.16.0.0/12
IPAddressDeny=192.168.0.0/16
IPAddressDeny=169.254.0.0/16
IPAddressDeny=100.64.0.0/10
IPAddressDeny=fc00::/7
IPAddressDeny=fe80::/10"
expect "127.0.0.2" BLOCKED; expect "::ffff:127.0.0.2" BLOCKED; expect "::1" BLOCKED
expect "127.0.0.3" OPEN; expect "::ffff:127.0.0.3" OPEN; expect "192.0.2.1" SENT; expect "1.1.1.1:53" OPEN
for d in 10.0.0.1 172.16.0.1 192.168.0.1 169.254.169.254 100.64.0.1 fd00::1 2606:4700:4700::1111; do expect "$d" BLOCKED; done

echo "ALL EXPECTATIONS HELD"
