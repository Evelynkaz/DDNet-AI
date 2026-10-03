#!/usr/bin/env python3
"""Task 4.4: the soak harness. python3 stdlib only. Use tools/e2e/soak.sh (it builds and checks the environment).

What one run does (everything on 127.0.0.1, never any other address):
  * our bot (`ddnet-ai play --brain hybrid`, default config) on the local DDNet server, on Copy Love Box, either as a
    plain child process (console fed through its stdin) or through the systemd unit `deploy/systemd/ddnet-ai-bot.service`
    (--unit; the console is off there, as in production);
  * three scripted bots (`--brain scripted`): `soak-s1` stays for the whole run, `soak-s2` and `soak-s3` come and go;
  * a web unit on another loopback port (never the production 127.0.0.1:7788) reading the bot's bridge, with a WebSocket
    viewer logged in for the whole run;
  * console commands (!stats !where !clip !brain planner and back, and lines that must go nowhere);
  * one graceful restart of the local server (at 20% of the run) and one map change to another block map and back (63-70%);
  * a journal sample every --sample seconds: RSS, CPU, threads and fds of every process, the bot's latency report,
    input timing, kills / deaths / blocks / freezes, clip files written and pruned, the freeze-memory file, reconnects,
    the web viewer's counters and the load average.
Then soak_analyze.analyze() judges the run. The local server's map is put back to "Copy Love Box" and read back at the end.
Output: ~/aiddnet/data/logs/4.4/<stamp>-<label>/ (never in git). The bot gets a PRIVATE data dir <run>/botdata in both modes; the
production ~/aiddnet/data/bot is never used, moved or removed, and the run ends by checking that it is unchanged (listing, modes, sizes, mtimes).
The memory gate needs a baseline after the warm-up (minute 25) and 10 more minutes: a run shorter than 35 minutes does not judge it.
"""
import argparse
import base64
import datetime
import http.client
import json
import os
import random
import re
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soak_analyze as an  # noqa: E402

HOME = Path.home()
DATA = HOME / "aiddnet" / "data"
REPO = Path(__file__).resolve().parents[2]
ECON = REPO / "tools" / "ddnet-server" / "econ.py"
SERVER_ADDR = "127.0.0.1:8303"
SERVER_UNIT = "ddnet-local.service"
HOME_MAP = "Copy Love Box"
BOT_UNIT = "ddnet-ai-bot.service"
WEB_UNIT = "ddai-soak-web"
PROD_WEB_PORT = 7788
# The memory baseline: after the latency rings fill (RING = 32 768 decisions / 25 Hz = 22 min), with margin. No scenario event (restart, map
# change) falls into the 60 s window after it in a 60 minute run.
WARMUP_S = 1500.0
CLK_TCK = os.sysconf("SC_CLK_TCK")


def sh(cmd, check=False, timeout=60, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd[:3]} failed ({r.returncode}): {r.stderr.strip()[:300]}")
    return r


# glibc's dynamic mmap threshold keeps map-sized buffers in the heap after a map change (a first visit to a 1244x667 map
# left +160 MiB resident for good); a fixed threshold gives them back at once. The unit sets the same (E-009).
BOT_ENV = {"NO_COLOR": "1", "RUST_LOG": "info", "MALLOC_MMAP_THRESHOLD_": "131072"}


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------------------------ /proc sampling


def read_proc(pid):
    """RSS (kB), threads, open fds and cumulative CPU ticks of a process; None if it is gone or unreadable."""
    try:
        rss = thr = None
        swap = anon = file = 0
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("VmSwap:"):
                    swap = int(line.split()[1])
                elif line.startswith("RssAnon:"):
                    anon = int(line.split()[1])
                elif line.startswith("RssFile:"):
                    file = int(line.split()[1])
                elif line.startswith("Threads:"):
                    thr = int(line.split()[1])
        with open(f"/proc/{pid}/stat") as fh:
            stat = fh.read()
        rest = stat[stat.rindex(")") + 2 :].split()
        ticks = int(rest[11]) + int(rest[12])  # utime + stime
        fds = len(os.listdir(f"/proc/{pid}/fd"))
        if rss is None or thr is None:
            return None
        return {"pid": pid, "rss_kb": rss, "anon_kb": anon, "file_kb": file, "swap_kb": swap, "thr": thr, "fds": fds, "ticks": ticks}
    except (OSError, ValueError, IndexError):
        return None


def meminfo():
    """MemAvailable and swap in use, MiB: the VM is shared and swaps idle pages of every process."""
    try:
        kv = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, v = line.split(":", 1)
                kv[k] = int(v.split()[0])
        return {"avail_mb": kv["MemAvailable"] // 1024, "swap_used_mb": (kv["SwapTotal"] - kv["SwapFree"]) // 1024}
    except (OSError, ValueError, KeyError):
        return {}


def unit_pid(unit):
    r = sh(["systemctl", "show", "-p", "MainPID", "--value", unit])
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


# ------------------------------------------------------------------------------------------------ the bridge reader


class BridgeReader(threading.Thread):
    """A second, read-only reader of the bot's bridge socket (docs/formats.md 21.2): counts frames, keeps the latest
    STATUS, and counts freeze / death edges (a 5 Hz status stream sees every freeze: DDNet's freeze lasts >= 3 s)."""

    def __init__(self, path, stop):
        super().__init__(daemon=True, name="soak-bridge")
        self.path, self.stop_ev = path, stop
        self.lock = threading.Lock()
        self.connects = 0
        self.frames = 0
        self.statuses = 0
        self.status = {}
        self.map = None
        self.players = 0
        self.frozen_edges = 0
        self.death_edges = 0
        self.last_frame = 0.0
        self.max_gap = 0.0
        self._frozen = None
        self._alive = None

    def run(self):
        while not self.stop_ev.is_set():
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.connect(str(self.path))
            except OSError:
                s.close()
                time.sleep(1)
                continue
            with self.lock:
                self.connects += 1
            s.settimeout(2)
            buf = b""
            try:
                while not self.stop_ev.is_set():
                    try:
                        data = s.recv(1 << 16)
                    except socket.timeout:
                        continue
                    if not data:
                        break
                    buf += data
                    while len(buf) >= 4:
                        (ln,) = struct.unpack("<I", buf[:4])
                        if ln < 1 or ln > (1 << 20):
                            raise ValueError("bad bridge message length")
                        if len(buf) < 4 + ln:
                            break
                        kind, payload = buf[4], buf[5 : 4 + ln]
                        buf = buf[4 + ln :]
                        self._message(kind, payload)
            except (OSError, ValueError):
                pass
            finally:
                s.close()
            time.sleep(1)

    def _message(self, kind, payload):
        with self.lock:
            if kind == 2:
                self.map = json.loads(payload)
            elif kind == 3:
                self.players = len(json.loads(payload).get("list", []))
            elif kind == 4:
                t = time.monotonic()
                if self.last_frame and t - self.last_frame > self.max_gap:
                    self.max_gap = t - self.last_frame
                self.last_frame = t
                self.frames += 1
            elif kind == 5:
                st = json.loads(payload)
                self.statuses += 1
                self.status = st
                frozen, alive = bool(st.get("frozen")), bool(st.get("alive"))
                if self._frozen is False and frozen:
                    self.frozen_edges += 1
                if self._alive is True and not alive:
                    self.death_edges += 1
                self._frozen, self._alive = frozen, alive

    def snapshot(self):
        with self.lock:
            return {
                "frames": self.frames,
                "statuses": self.statuses,
                "reconnects": max(0, self.connects - 1),
                "frozen_edges": self.frozen_edges,
                "death_edges": self.death_edges,
                "players": self.players,
                "map": self.map,
                "status": dict(self.status),
            }


# ------------------------------------------------------------------------------------------------ the WebSocket viewer


class Viewer(threading.Thread):
    """Logs in to the web unit and keeps a WebSocket open for the whole run, like a browser on the Game tab."""

    def __init__(self, port, password, stop):
        super().__init__(daemon=True, name="soak-viewer")
        self.port, self.password, self.stop_ev = port, password, stop
        self.lock = threading.Lock()
        self.frames = 0
        self.bot_msgs = 0
        self.text_msgs = {}
        self.connects = 0
        self.errors = []
        self.last_frame = 0.0
        self.max_gap = 0.0

    def _login(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        body = json.dumps({"password": self.password})
        c.request(
            "POST",
            "/api/login",
            body,
            {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{self.port}"},
        )
        r = c.getresponse()
        r.read()
        if r.status != 200:
            raise RuntimeError(f"login refused: HTTP {r.status}")
        cookies = [h.split(";")[0] for k, h in r.getheaders() if k.lower() == "set-cookie"]
        c.close()
        if not cookies:
            raise RuntimeError("login gave no cookie")
        return "; ".join(cookies)

    def _recv_exact(self, s, n):
        out = b""
        while len(out) < n:
            chunk = s.recv(n - len(out))
            if not chunk:
                raise ConnectionError("closed")
            out += chunk
        return out

    def _frame(self, s):
        b0, b1 = self._recv_exact(s, 2)
        op, ln = b0 & 0x0F, b1 & 0x7F
        if ln == 126:
            (ln,) = struct.unpack(">H", self._recv_exact(s, 2))
        elif ln == 127:
            (ln,) = struct.unpack(">Q", self._recv_exact(s, 8))
        if b1 & 0x80:
            self._recv_exact(s, 4)  # a server never masks; tolerate and ignore
        return op, self._recv_exact(s, ln)

    @staticmethod
    def _send(s, op, payload=b""):
        mask = os.urandom(4)
        n = len(payload)
        head = bytes([0x80 | op])
        head += bytes([0x80 | n]) if n < 126 else b"\xfe" + struct.pack(">H", n)
        s.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def run(self):
        while not self.stop_ev.is_set():
            try:
                cookie = self._login()
                s = socket.create_connection(("127.0.0.1", self.port), timeout=10)
                key = base64.b64encode(os.urandom(16)).decode()
                s.sendall(
                    (
                        f"GET /ws HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nOrigin: http://127.0.0.1:{self.port}\r\n"
                        f"Cookie: {cookie}\r\n\r\n"
                    ).encode()
                )
                head = b""
                while b"\r\n\r\n" not in head:
                    chunk = s.recv(4096)
                    if not chunk:
                        raise ConnectionError("closed during the handshake")
                    head += chunk
                if b" 101 " not in head.split(b"\r\n", 1)[0]:
                    raise RuntimeError("upgrade refused: " + head.split(b"\r\n", 1)[0].decode(errors="replace"))
                with self.lock:
                    self.connects += 1
                self._send(s, 1, b'{"type":"sub","live":25}')
                s.settimeout(1)
                next_ping = time.monotonic() + 15
                while not self.stop_ev.is_set():
                    if time.monotonic() >= next_ping:
                        self._send(s, 1, b'{"type":"ping"}')
                        next_ping = time.monotonic() + 15
                    try:
                        op, payload = self._frame(s)
                    except socket.timeout:
                        continue
                    if op == 2:
                        t = time.monotonic()
                        with self.lock:
                            if self.last_frame and t - self.last_frame > self.max_gap:
                                self.max_gap = t - self.last_frame
                            self.last_frame = t
                            self.frames += 1
                    elif op == 1:
                        try:
                            kind = json.loads(payload).get("type", "?")
                        except ValueError:
                            kind = "?"
                        with self.lock:
                            self.text_msgs[kind] = self.text_msgs.get(kind, 0) + 1
                            if kind == "bot":
                                self.bot_msgs += 1
                    elif op == 9:
                        self._send(s, 10, payload)
                    elif op == 8:
                        raise ConnectionError("closed by the web unit")
            except (OSError, RuntimeError, ConnectionError) as e:
                with self.lock:
                    if len(self.errors) < 20:
                        self.errors.append(str(e)[:120])
                time.sleep(2)

    def snapshot(self):
        with self.lock:
            return {
                "frames": self.frames,
                "bot_msgs": self.bot_msgs,
                "msgs": dict(self.text_msgs),
                "reconnects": max(0, self.connects - 1),
                "max_gap_s": round(self.max_gap, 2),
                "errors": list(self.errors),
            }


# ------------------------------------------------------------------------------------------------ the run


class Soak:
    def __init__(self, a):
        self.a = a
        self.stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run = Path(a.out).expanduser() / f"{self.stamp}-{a.label}"
        self.stop = threading.Event()  # the end of the measurement
        self.abort = threading.Event()
        self.churn_pause = threading.Event()
        self.lock = threading.Lock()
        self.econ_lock = threading.Lock()
        self.botlog = an.BotLog()
        self.botlog_lock = threading.Lock()
        self.procs = {}  # name -> subprocess.Popen (scripted bots, web, bot in process mode)
        self.threads = []
        self.t0 = None
        self.t0_epoch = None
        self.events_fh = None
        self.cpu_prev = {}
        self.clip_names = set()
        self.clips_written = 0
        self.clips_pruned = 0
        self.players = None
        self.bot_proc = None
        self.web_password = None
        self.unit_mode = a.unit
        self.meta = {}
        self.others = ["soak-s1", "soak-s2", "soak-s3"]
        # The soak's bot always gets a PRIVATE data dir (both modes): ~/aiddnet/data/bot is the owner's production state (relations.json,
        # live.sock, control.sock; in the production web unit's ReadWritePaths) and is never used, moved or removed here.
        self.botdata = self.run / "botdata"
        self.bot_state = self.botdata / "bot"
        self.bridge_path = self.bot_state / "live.sock"
        self.control_path = self.bot_state / "control.sock"
        self.clips_dir = self.bot_state / "clips"
        self.memory_dir = self.bot_state / "memory"
        self.prod_bot_before = None

    # ---- small helpers
    def t(self):
        return round(time.monotonic() - self.t0, 2) if self.t0 else -1

    def ev(self, kind, **kw):
        rec = {"t": self.t(), "wall": now_iso(), "kind": kind, **kw}
        with self.lock:
            self.events_fh.write(json.dumps(rec) + "\n")
            self.events_fh.flush()
        print(f"[{rec['t']:8.1f}s] {kind} " + " ".join(f"{k}={v}" for k, v in kw.items() if k != "reply"), flush=True)

    def econ(self, *args, timeout=60):
        with self.econ_lock:
            r = sh([sys.executable, str(ECON), *args], timeout=timeout)
        return r.returncode, r.stdout

    def server_map(self):
        rc, out = self.econ("sv_map")
        m = re.search(r"Value: (.+)", out)
        return m.group(1).strip() if m else None

    # ---- start / stop of the parts
    def bin(self):
        return str(Path(self.a.bin).expanduser())

    def start_web(self):
        web = self.run / "web-data"
        (web / "secrets").mkdir(parents=True, exist_ok=True)
        sh([self.bin(), "web-passwd", "--data-dir", str(web)], check=True)
        self.web_password = (web / "secrets" / "web-password.txt").read_text().strip()
        port = self.a.web_port
        if port == PROD_WEB_PORT:
            raise SystemExit("refusing: 7788 is the production web")
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise SystemExit(f"port {port} is already in use")
        cmd = [self.bin(), "web", "--listen", f"127.0.0.1:{port}", "--data-dir", str(web), "--bot-socket", str(self.bridge_path),
               "--maps-dir", str(self.botdata / "maps" / "cache")]
        if self.unit_mode:
            # The web side as a transient systemd unit with the production web unit's own sandbox (the bot's socket must
            # stay reachable from under ProtectHome=read-only; the sandbox properties are read from the real unit file).
            props = web_sandbox_props(REPO / "deploy" / "systemd" / "ddnet-ai-web.service", web)
            sh(["sudo", "systemctl", "stop", WEB_UNIT], timeout=30)
            sh(["sudo", "systemd-run", "--unit", WEB_UNIT, "--collect", "--quiet", "-p", "User=ubuntu", "-p", "Group=ubuntu",
                "-p", f"WorkingDirectory={self.run}", "--setenv=NO_COLOR=1", *props, *cmd], check=True)
        else:
            out = open(self.run / "web.log", "wb")
            self.procs["web"] = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=dict(os.environ, NO_COLOR="1"),
                                                 start_new_session=True)
        for _ in range(60):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.5)
        else:
            raise SystemExit("the web unit did not come up")
        self.ev("web_started", port=port, unit=self.unit_mode)

    def start_bot(self):
        a = self.a
        self.bot_state.mkdir(parents=True, exist_ok=True)
        os.chmod(self.bot_state, 0o700)
        settings = f'wb = "{a.wb}"\n' if a.wb != "auto" else ""
        (self.bot_state / "settings.toml").write_text(settings)
        paths = ["--data-dir", str(self.botdata), "--bridge", str(self.bridge_path), "--control", str(self.control_path),
                 "--settings", str(self.bot_state / "settings.toml"), "--relations", str(self.bot_state / "relations.json"),
                 "--clips-dir", str(self.clips_dir), "--memory-dir", str(self.memory_dir)]
        if self.unit_mode:
            unit_src = REPO / "deploy" / "systemd" / BOT_UNIT
            sh(["sudo", "install", "-m", "0644", str(unit_src), f"/etc/systemd/system/{BOT_UNIT}"], check=True)
            # The same unit, with our freshly built binary (~/aiddnet/bin/ddnet-ai is what production runs), a private data dir and
            # explicit paths, and ReadWritePaths on the run's private dir (+ data/run: the single-instance locks).
            dropin = unit_exec_override(unit_src, self.bin(), paths, self.bot_state / "last-report.json", [self.botdata, DATA / "run"])
            sh(["sudo", "mkdir", "-p", f"/etc/systemd/system/{BOT_UNIT}.d"], check=True)
            sh(["sudo", "tee", f"/etc/systemd/system/{BOT_UNIT}.d/soak.conf"], input=dropin, check=True)
            sh(["sudo", "systemctl", "daemon-reload"], check=True)
            self.start_journal_tail()
            sh(["sudo", "systemctl", "start", BOT_UNIT], check=True)
        else:
            cmd = [self.bin(), "play", "--server", SERVER_ADDR, "--name", a.bot_name, "--brain", "hybrid", "--duration",
                   str(a.duration + 900), "--console", *paths, "--report", str(self.run / "bot-report.json")]
            if a.wb != "auto":
                cmd += ["--wb", a.wb]
            self.bot_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=open(self.run / "bot-console.log", "wb"),
                                             stderr=subprocess.PIPE, env=dict(os.environ, **BOT_ENV),
                                             start_new_session=True)
            self.start_tailer(self.bot_proc.stderr)
        self.ev("bot_started", unit=self.unit_mode, wb=a.wb)

    def start_journal_tail(self):
        p = subprocess.Popen(["sudo", "journalctl", "-u", BOT_UNIT, "-f", "-o", "cat", "-n", "0", "--no-pager"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
        self.procs["journal"] = p
        self.start_tailer(p.stdout)

    def start_tailer(self, stream):
        def run():
            with open(self.run / "bot.log", "a", buffering=1) as out:
                for raw in iter(stream.readline, b""):
                    line = an.strip_ansi(raw.decode(errors="replace"))
                    out.write(line if line.endswith("\n") else line + "\n")
                    with self.botlog_lock:
                        self.botlog.feed(line)
        th = threading.Thread(target=run, daemon=True, name="soak-tail")
        th.start()
        self.threads.append(th)

    def bot_in_game(self):
        with self.botlog_lock:
            return self.botlog.counts.get("in_game", 0)

    def bot_pid(self):
        if self.unit_mode:
            return unit_pid(BOT_UNIT)
        return self.bot_proc.pid if self.bot_proc and self.bot_proc.poll() is None else 0

    def bot_alive(self):
        if self.unit_mode:
            return sh(["systemctl", "is-active", BOT_UNIT]).stdout.strip() == "active"
        return self.bot_proc is not None and self.bot_proc.poll() is None

    def console(self, line, wait=3.0):
        if self.unit_mode or not self.bot_alive():
            return
        path = self.run / "bot-console.log"
        before = path.stat().st_size
        try:
            self.bot_proc.stdin.write((line + "\n").encode())
            self.bot_proc.stdin.flush()
        except OSError:
            self.ev("console_failed", line=line)
            return
        time.sleep(wait)
        with open(path, "rb") as fh:
            fh.seek(before)
            reply = fh.read(1500).decode(errors="replace").strip()
        self.ev("console", line=line, reply=reply)

    # ---- scripted bots
    def scripted(self, name, life, gap, extra, persistent=False):
        rng = random.Random(f"{self.a.seed}-{name}")
        n = 0
        sdata = self.run / "sdata" / name
        sdata.mkdir(parents=True, exist_ok=True)
        if not (sdata / "maps").exists():
            (sdata / "maps").symlink_to(DATA / "maps")
        while not self.stop.is_set():
            while self.churn_pause.is_set() and not self.stop.is_set():
                time.sleep(1)
            if self.stop.is_set():
                break
            n += 1
            remaining = self.a.duration - self.t() + 600
            dur = int(remaining if persistent else min(rng.uniform(*life), max(60, self.a.duration - self.t() - 30)))
            cmd = [self.bin(), "play", "--server", SERVER_ADDR, "--name", name, "--brain", "scripted", "--duration", str(max(dur, 30)),
                   "--data-dir", str(sdata), "--no-bridge", "--no-console", "--no-memory", "--no-settings", "--no-autoclip", *extra]
            logf = open(self.run / "scripted" / f"{name}-{n}.log", "wb")
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=dict(os.environ, **BOT_ENV),
                                 start_new_session=True)
            self.procs[name] = p
            self.ev("scripted_start", name=name, n=n, seconds=dur)
            while p.poll() is None and not self.stop.is_set():
                time.sleep(1)
            if p.poll() is None:
                break  # the run ended: the main thread stops everything
            logf.close()
            self.ev("scripted_exit", name=name, n=n, code=p.returncode)
            if persistent and not self.stop.is_set():
                time.sleep(5)
                continue
            self.sleep_stop(rng.uniform(*gap))

    def sleep_stop(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self.stop.is_set():
            time.sleep(0.5)

    # ---- sampling
    def scan_clips(self):
        names = {}
        if self.clips_dir.exists():
            for f in self.clips_dir.iterdir():
                if f.suffix == ".clip":
                    try:
                        names[f.name] = f.stat().st_size
                    except OSError:
                        pass
        cur = set(names)
        self.clips_written += len(cur - self.clip_names)
        self.clips_pruned += len(self.clip_names - cur)
        self.clip_names = cur
        kinds = {}
        auto = 0
        for n in cur:
            if n.startswith("manual-"):
                continue
            auto += 1
            kind = re.sub(r"-\d+(-s\d+)?\.clip$", "", n)
            kinds[kind] = kinds.get(kind, 0) + 1
        return {"files": len(cur), "auto": auto, "kinds": kinds, "bytes": sum(names.values()),
                "written": self.clips_written, "pruned": self.clips_pruned}

    def scan_memory(self):
        files = {}
        if self.memory_dir.exists():
            for f in self.memory_dir.iterdir():
                try:
                    files[f.name] = f.stat().st_size
                except OSError:
                    pass
        return {"files": files, "bytes": sum(files.values())}

    def proc_sample(self, name, pid):
        r = read_proc(pid) if pid else None
        if not r:
            return None
        w = time.monotonic()
        prev = self.cpu_prev.get((name, pid))
        self.cpu_prev[(name, pid)] = (r["ticks"], w)
        cpu = None
        if prev and w > prev[1]:
            cpu = round((r["ticks"] - prev[0]) / CLK_TCK / (w - prev[1]) * 100.0, 1)
        r.pop("ticks")
        r["cpu"] = cpu
        return r

    def sample(self, bridge, viewer, i):
        procs = {}
        pids = {"server": unit_pid(SERVER_UNIT), "bot": self.bot_pid()}
        if self.unit_mode:
            pids["web"] = unit_pid(WEB_UNIT)
        else:
            w = self.procs.get("web")
            pids["web"] = w.pid if w and w.poll() is None else 0
        for name in ("soak-s1", "soak-s2", "soak-s3"):
            p = self.procs.get(name)
            pids["s" + name[-1]] = p.pid if p and p.poll() is None else 0
        for name, pid in pids.items():
            r = self.proc_sample(name, pid)
            if r:
                procs[name] = r
        with self.botlog_lock:
            bl = {
                "ev": dict(self.botlog.counts),
                "lat": json.loads(json.dumps(self.botlog.lat)),
                "stats": dict(self.botlog.stats),
                "slots": dict(self.botlog.slots),
                "margin": {k: v for k, v in self.botlog.margin.items() if not k.startswith("_")},
                "log_lines": self.botlog.lines,
            }
        b = bridge.snapshot()
        bl["status"] = {k: b["status"].get(k) for k in ("tick", "mode", "brain", "alive", "frozen", "blocks", "blocked_by",
                                                       "self_kills", "decisions", "collapsed", "decide_p99_us", "overhead_p99_us")}
        bl["frozen_edges"], bl["death_edges"] = b["frozen_edges"], b["death_edges"]
        bl["frames"], bl["bridge_reconnects"] = b["frames"], b["reconnects"]
        if b["map"]:
            bl["map"] = {"name": b["map"].get("name"), "sha256": b["map"].get("sha256")}
        if i % 4 == 0:
            rc, out = self.econ("status", timeout=30)
            if rc == 0:
                self.players = len(re.findall(r"name='", out))
        return {
            "t": self.t(),
            "wall": now_iso(),
            "load": [float(x) for x in open("/proc/loadavg").read().split()[:3]],
            "mem": meminfo(),
            "players": self.players,
            "procs": procs,
            "bot": bl,
            "clips": self.scan_clips(),
            "memory": self.scan_memory(),
            "viewer": viewer.snapshot(),
        }

    # ---- the timeline
    def timeline(self):
        D, a = self.a.duration, self.a
        f = lambda x: round(D * x)  # noqa: E731
        items = []
        if not self.unit_mode:
            items += [
                (f(0.02), lambda: self.console("!help")),
                (f(0.05), lambda: self.console("!stats")),
                (f(0.06), lambda: self.console("!where")),
                (f(0.07), lambda: self.console("hello everyone")),  # no prefix: must go nowhere, never to the chat
                (f(0.075), lambda: self.console("!say hello everyone")),  # not ported: refused, never to the chat
                (f(0.12), lambda: self.console("!clip soak-a")),
                (f(0.2), lambda: self.console("!stats")),
                (f(0.28), lambda: self.console("!brain planner")),
                (f(0.30), lambda: self.console("!stats")),
                (f(0.32), lambda: self.console("!brain hybrid")),
                (f(0.34), lambda: self.console("!stats")),
                (f(0.24), lambda: self.console("!where")),
                (f(0.50), lambda: self.console("!clip soak-b")),
                (f(0.80), lambda: self.console("!stats")),
                (f(0.85), lambda: self.console("!clip soak-c")),
                (f(0.95), lambda: self.console("!stats")),
            ]
        if not a.no_restart:
            items.append((f(0.20), self.server_restart))  # early: the memory baseline (minute 25) must be quiet
        if not a.no_mapchange:
            items.append((f(0.63), lambda: self.map_change(a.other_map)))
            items.append((f(0.70), lambda: self.map_change(HOME_MAP)))
            if not self.unit_mode:
                items.append((f(0.63) + 25, lambda: self.console("!where")))
                items.append((f(0.70) + 25, lambda: self.console("!where")))
        items.sort(key=lambda x: x[0])
        return items

    def server_restart(self):
        self.churn_pause.set()
        self.ev("server_restart", how="systemctl restart (graceful, NETMSG_CLOSE Server shutdown)")
        t = time.monotonic()
        sh(["sudo", "systemctl", "restart", SERVER_UNIT], timeout=90)
        self.ev("server_restarted", seconds=round(time.monotonic() - t, 1))
        threading.Timer(60, self.churn_pause.clear).start()

    def map_change(self, name):
        rc, out = self.econ("change_map", name)
        self.ev("map_change", map=name, rc=rc)

    def watch_bot(self):
        if not self.bot_alive():
            self.ev("bot_died")
            self.meta["bot_died_early"] = True
            self.abort.set()

    def main_loop(self, bridge, viewer):
        D = self.a.duration
        items = self.timeline()
        next_sample = 0.0
        i = 0
        jf = open(self.run / "journal.jsonl", "a", buffering=1)
        while not self.abort.is_set():
            t = self.t()
            if t >= D:
                break
            if t >= next_sample:
                next_sample += self.a.sample
                try:
                    jf.write(json.dumps(self.sample(bridge, viewer, i)) + "\n")
                except Exception as e:  # a sampling hiccup must never end a soak
                    self.ev("sample_error", error=str(e)[:200])
                i += 1
            while items and items[0][0] <= t:
                _, fn = items.pop(0)
                threading.Thread(target=fn, daemon=True).start()
            self.watch_bot()
            time.sleep(0.5)
        jf.write(json.dumps(self.sample(bridge, viewer, i)) + "\n")
        jf.close()
        return self.t()

    # ---- whole run
    def preflight(self):
        a = self.a
        if sh(["systemctl", "is-active", SERVER_UNIT]).stdout.strip() != "active":
            raise SystemExit(f"{SERVER_UNIT} is not active")
        if not Path(self.bin()).exists():
            raise SystemExit(f"binary {self.bin()} not found (soak.sh --build)")
        m = self.server_map()
        if m != HOME_MAP:
            print(f"server map is {m!r}: changing to {HOME_MAP!r}")
            self.econ("change_map", HOME_MAP)
            time.sleep(5)
        rc, out = self.econ("status")
        if re.search(r"name='", out):
            raise SystemExit("someone is on the local server: refusing to start (it is ours alone for this task)")
        if self.unit_mode and sh(["systemctl", "is-active", BOT_UNIT]).stdout.strip() == "active":
            raise SystemExit(f"{BOT_UNIT} is already running")

    def run_all(self):
        a = self.a
        self.run.mkdir(parents=True)
        (self.run / "scripted").mkdir()
        self.events_fh = open(self.run / "events.jsonl", "a")
        self.preflight()
        self.botdata.mkdir()
        if not self.unit_mode:
            (self.botdata / "maps").symlink_to(DATA / "maps")  # the shared map cache (process mode only; the unit gets a private one)
        self.prod_bot_before = snapshot_dir(DATA / "bot")
        self.meta = {
            "label": a.label, "mode": "unit" if a.unit else "process", "wb": a.wb, "brain": "hybrid", "duration": a.duration,
            "sample_s": a.sample, "baseline_s": WARMUP_S, "home_map": HOME_MAP, "other_map": a.other_map,
            "binary": self.bin(), "others_names": self.others, "name_logs": ["bot.log"], "extra_logs": ["web.log"],
            "memory": True, "started": now_iso(), "bot_died_early": False, "git_head": sh(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).stdout.strip(),
        }
        (self.run / "meta.json").write_text(json.dumps(self.meta, indent=1))
        bridge = viewer = None
        try:
            self.start_web()
            self.start_bot()
            for _ in range(240):  # the bot must be in game within a minute
                if self.bot_in_game() or not self.bot_alive():
                    break
                time.sleep(0.25)
            if not self.bot_in_game():
                raise SystemExit("the bot never reached the game")
            self.t0, self.t0_epoch = time.monotonic(), time.time()
            self.meta["t0_epoch"] = self.t0_epoch
            self.ev("measurement_start")
            bridge = BridgeReader(self.bridge_path, self.stop)
            bridge.start()
            viewer = Viewer(a.web_port, self.web_password, self.stop)
            viewer.start()
            plan = [("soak-s1", (0, 0), (0, 0), [], True, 10), ("soak-s2", (300, 540), (30, 120), ["--wb", "off"], False, 25),
                    ("soak-s3", (180, 420), (45, 150), ["--wb", "off"], False, 40)]
            for name, life, gap, extra, persistent, delay in plan:
                def start(name=name, life=life, gap=gap, extra=extra, persistent=persistent, delay=delay):
                    self.sleep_stop(delay)
                    self.scripted(name, life, gap, extra, persistent)
                th = threading.Thread(target=start, daemon=True, name=f"soak-{name}")
                th.start()
                self.threads.append(th)
            t_end = self.main_loop(bridge, viewer)
            self.meta["t_end"] = t_end
            self.ev("measurement_end")
        except KeyboardInterrupt:
            self.ev("interrupted")
            self.meta["t_end"] = self.t()
        finally:
            self.finish(bridge, viewer)

    def finish(self, bridge, viewer):
        a = self.a
        self.stop.set()
        time.sleep(1)
        for name in ("soak-s1", "soak-s2", "soak-s3"):
            stop_proc(self.procs.get(name))
        code = None
        if self.unit_mode:
            sh(["sudo", "systemctl", "stop", BOT_UNIT], timeout=60)
            code = sh(["systemctl", "show", "-p", "ExecMainStatus", "--value", BOT_UNIT]).stdout.strip()
            code = int(code) if code.lstrip("-").isdigit() else None
            time.sleep(1)
            jp = self.procs.get("journal")
            if jp:
                stop_proc(jp)
            # what the owner would read: the journal of the unit, nicknames included if any leaked
            r = sh(["sudo", "journalctl", "-u", BOT_UNIT, "--no-pager", "-o", "short-iso", "--since", self.meta["started"].replace("T", " ").replace("Z", " UTC")], timeout=60)
            (self.run / "unit-journal.log").write_text(r.stdout)
            self.meta["name_logs"] = ["bot.log", "unit-journal.log"]
            wl = sh(["sudo", "journalctl", "-u", WEB_UNIT, "--no-pager", "-o", "cat"], timeout=30).stdout
            (self.run / "web.log").write_text(wl)
            sh(["sudo", "systemctl", "stop", WEB_UNIT], timeout=30)
            rp = self.bot_state / "last-report.json"
            if rp.exists():
                shutil.copy(rp, self.run / "bot-report.json")
        else:
            if self.bot_proc and self.bot_proc.poll() is None:
                self.bot_proc.send_signal(signal.SIGTERM)
                try:
                    self.bot_proc.wait(30)
                except subprocess.TimeoutExpired:
                    self.bot_proc.kill()
            code = self.bot_proc.returncode if self.bot_proc else None
            stop_proc(self.procs.get("web"))
        self.meta["bot_exit_code"] = code
        self.ev("bot_stopped", exit_code=code)
        for th in self.threads:
            th.join(timeout=5)
        if bridge:
            bridge.join(timeout=3)
        if viewer:
            viewer.join(timeout=3)
            self.meta["viewer"] = viewer.snapshot()
        # freeze-memory and clip listings, for the report (names only)
        self.meta["memory_files"] = self.scan_memory()["files"]
        self.meta["clip_files"] = sorted(self.clip_names)
        if self.unit_mode:
            # remove the soak's unit again (only the files this run installed); the run's private botdata stays for the analysis
            sh(["sudo", "rm", "-rf", f"/etc/systemd/system/{BOT_UNIT}.d", f"/etc/systemd/system/{BOT_UNIT}"])
            sh(["sudo", "systemctl", "daemon-reload"])
        # ~/aiddnet/data/bot is production state: it must be exactly as it was (listing, sizes, modes, mtimes)
        after = snapshot_dir(DATA / "bot")
        diff = diff_snapshots(self.prod_bot_before, after)
        self.meta["prod_bot_dir"] = {"unchanged": not diff, "entries": len((self.prod_bot_before or {}).get("entries", {})), "diff": diff[:20]}
        self.ev("prod_bot_dir_checked", unchanged=not diff, diff=diff[:5])
        shutil.rmtree(self.run / "web-data" / "secrets", ignore_errors=True)
        # the server: back to the home map, read it back
        self.churn_pause.clear()
        rc, _ = self.econ("change_map", HOME_MAP)
        time.sleep(3)
        got = self.server_map()
        self.meta["server_map_after"] = got
        self.ev("server_map_restored", sv_map=got, ok=got == HOME_MAP)
        (self.run / "meta.json").write_text(json.dumps(self.meta, indent=1))
        self.events_fh.close()
        if self.meta.get("t0_epoch"):
            try:
                res, text = an.analyze(self.run)
                print(text)
                print(f"\nrun dir: {self.run}\nverdict: {'PASS' if res.ok else 'FAIL'}")
            except Exception as e:  # the raw data is on disk either way
                print(f"analysis failed: {e!r}; re-run: tools/e2e/soak_analyze.py {self.run}")


def stop_proc(p, grace=15):
    if p is None or p.poll() is not None:
        return
    try:
        p.send_signal(signal.SIGTERM)
        p.wait(grace)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(5)
    except ProcessLookupError:
        pass


def web_sandbox_props(unit_file, rw_path):
    """`-p Key=Value` arguments for systemd-run: the [Service] hardening of the real web unit, with the writable path swapped
    for the scratch data dir."""
    keep = {"NoNewPrivileges", "ProtectSystem", "ProtectHome", "PrivateTmp", "PrivateDevices", "ProtectKernelTunables",
            "ProtectKernelModules", "ProtectKernelLogs", "ProtectControlGroups", "ProtectClock", "ProtectHostname",
            "RestrictNamespaces", "RestrictRealtime", "RestrictSUIDSGID", "LockPersonality", "MemoryDenyWriteExecute",
            "RemoveIPC", "CapabilityBoundingSet", "AmbientCapabilities", "RestrictAddressFamilies", "MemoryMax",
            "IPAddressAllow", "IPAddressDeny"}
    props = []
    for line in unit_file.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() in keep:
                props += ["-p", f"{k.strip()}={v.strip()}"]
    props += ["-p", f"ReadWritePaths={rw_path}"]
    return props


def unit_exec_override(unit_file, binary, paths, report, rw_paths):
    """A drop-in for the bot unit: `binary` with the unit's own ExecStart arguments, except that `--data-dir` and `--report` are
    replaced by the soak's private ones (`paths` holds `--data-dir` and the explicit socket/state paths), and ReadWritePaths is
    reset to `rw_paths`."""
    for line in unit_file.read_text().splitlines():
        if line.startswith("ExecStart="):
            argv = shlex.split(line[len("ExecStart="):])[1:]
            kept, i = [], 0
            while i < len(argv):
                if argv[i] in ("--data-dir", "--report"):
                    i += 2
                    continue
                kept.append(argv[i])
                i += 1
            cmd = shlex.join([binary, *kept, *paths, "--report", str(report)])
            return (f"[Service]\nExecStart=\nExecStart={cmd}\nReadWritePaths=\nReadWritePaths="
                    + " ".join(str(p) for p in rw_paths) + "\n")
    raise SystemExit("no ExecStart in the bot unit")


def snapshot_dir(path):
    """A listing of a directory tree for the 'untouched' check: type, size, mode and mtime of the directory and of everything in it
    (None if the path does not exist)."""
    path = Path(path)
    if not path.exists():
        return {"exists": False, "entries": {}}
    entries = {}
    for root, dirs, files in os.walk(path):
        for name in [""] + dirs + files:
            p = Path(root) / name if name else Path(root)
            try:
                st = p.lstat()
            except OSError:
                continue
            entries[str(p.relative_to(path))] = [oct(st.st_mode), st.st_size, st.st_mtime_ns]
    return {"exists": True, "entries": entries}


def diff_snapshots(a, b):
    if a is None or b is None:
        return ["no snapshot"]
    out = []
    if a["exists"] != b["exists"]:
        out.append(f"exists {a['exists']} -> {b['exists']}")
    for k in sorted(set(a["entries"]) | set(b["entries"])):
        if k not in b["entries"]:
            out.append(f"removed {k}")
        elif k not in a["entries"]:
            out.append(f"added {k}")
        elif a["entries"][k] != b["entries"][k]:
            out.append(f"changed {k}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="run name, e.g. A-wbauto")
    ap.add_argument("--duration", type=int, default=3600, help="measured seconds (default 3600)")
    ap.add_argument("--wb", default="auto", choices=["auto", "off", "left", "right"])
    ap.add_argument("--unit", action="store_true", help="run our bot through deploy/systemd/ddnet-ai-bot.service (console off)")
    ap.add_argument("--bin", default=str(REPO / "target" / "release" / "ddnet-ai"))
    ap.add_argument("--web-port", type=int, default=7790)
    ap.add_argument("--sample", type=float, default=15.0)
    ap.add_argument("--other-map", default="BlmapChill")
    ap.add_argument("--bot-name", default="soak-bot")
    ap.add_argument("--seed", default="44")
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--no-mapchange", action="store_true")
    ap.add_argument("--out", default=str(DATA / "logs" / "4.4"))
    a = ap.parse_args()
    s = Soak(a)

    def on_signal(signum, frame):
        s.ev("signal", signum=signum)
        s.abort.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    s.run_all()


if __name__ == "__main__":
    main()
