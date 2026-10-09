#!/usr/bin/env python3
"""Task 4.14 (D-124): the poller of tools/e2e/cpu_priority.sh.

Connects to the bot's live bridge socket (read-only, docs/formats.md 21), and every 5 s writes one JSON line: the latest STATUS `search_window`
(candidates per decision and the brain p90 of the last 30 s of game time), the 1-minute load average, and the unit's cgroup counters
(`cpu.pressure` "some" total = microseconds in which some task of the unit was runnable but not running; `cpu.stat` usage_usec).
Stdlib only."""
import argparse, json, os, socket, struct, subprocess, sys, time


def read_exact(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf


def cgroup_of(unit):
    out = subprocess.run(["systemctl", "show", "-p", "ControlGroup", "--value", unit], capture_output=True, text=True).stdout.strip()
    return "/sys/fs/cgroup" + out if out else None


def cg_numbers(cg):
    r = {}
    if not cg:
        return r
    try:
        for line in open(cg + "/cpu.pressure"):
            parts = line.split()
            if parts and parts[0] == "some":
                r["psi_some_us"] = int(dict(p.split("=") for p in parts[1:])["total"])
        for line in open(cg + "/cpu.stat"):
            k, v = line.split()
            if k in ("usage_usec", "nr_throttled"):
                r[k] = int(v)
    except OSError:
        pass
    return r


def cpu_times():
    f = open("/proc/stat").readline().split()[1:]
    v = [int(x) for x in f]
    idle = v[3] + v[4]
    return sum(v), idle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--secs", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    deadline = t0 + a.secs
    while not os.path.exists(a.sock):
        if time.time() > t0 + 90:
            print("no bridge socket", file=sys.stderr)
            return 1
        time.sleep(0.5)
    s = socket.socket(socket.AF_UNIX)
    s.connect(a.sock)
    s.settimeout(1.0)
    cg = cgroup_of(a.unit)
    status = None
    last = 0.0
    prev_total, prev_idle = cpu_times()
    with open(a.out, "w") as out:
        while time.time() < deadline:
            try:
                (n,) = struct.unpack("<I", read_exact(s, 4))
                body = read_exact(s, n)
                if body[0] == 5:
                    status = json.loads(body[1:])
            except socket.timeout:
                pass
            except (EOFError, OSError):
                break
            now = time.time()
            if now - last >= 5.0:
                last = now
                total, idle = cpu_times()
                busy = 100.0 * (1 - (idle - prev_idle) / max(1, total - prev_total))
                prev_total, prev_idle = total, idle
                la = open("/proc/loadavg").read().split()
                row = {"t": round(now - t0, 1), "load1": float(la[0]), "runnable": int(la[3].split("/")[0]), "machine_busy_pct": round(busy, 1)}
                if status:
                    row["search_window"] = status.get("search_window")
                    row["tick"] = status.get("tick")
                row.update(cg_numbers(cg or cgroup_of(a.unit)))
                out.write(json.dumps(row) + "\n")
                out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
