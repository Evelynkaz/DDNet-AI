#!/usr/bin/env python3
"""Task 4.4: the soak's bot-log parser and the acceptance analysis. python3 stdlib only.

`soak.py` imports `BotLog` (the live parser behind the journal) and runs `analyze()` at the end of a run;
the same analysis can be re-run on a finished run directory:

    tools/e2e/soak_analyze.py <run-dir> [--baseline <s>]   # prints the report, writes analysis.txt + analysis.json
    tools/e2e/soak_analyze.py --selftest         # synthetic runs: every acceptance check must be able to fail

A run directory (written by soak.py, under ~/aiddnet/data/logs/4.4/) holds journal.jsonl (one sample every 10-30 s),
events.jsonl (what the harness did and when), bot.log (the bot's stderr: tracing lines), meta.json, and the bot's
--report JSON. Nothing here prints a nickname or a password.
"""
import datetime
import json
import math
import re
import statistics
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
HDR = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z)\s+(TRACE|DEBUG|INFO|WARN|ERROR)\s+(\S+?):(?:\s(.*))?$")
KV = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')
LAT = re.compile(r"^(total|brain|overhead|pick|queue|wire): n=(\d+) p50=(\d+)us p90=(\d+)us p99=(\d+)us max=(\d+)us")
SLOTS = re.compile(r"^slots: (.*)$")
STATS = re.compile(r"stats=BotStats \{(.*?)\}")
NUM = re.compile(r"(\w+): (\d+)")

KILL_COOLDOWN_TICKS = 500
WB_KILL_COOLDOWN_TICKS = 100
MAX_AUTO_CLIPS = 24
MAX_CLIPS_PER_KIND = 16
MAX_INGAME_RECONNECTS = 3
# The lead's gate (task 4.4, review round 1): growth from the post-warm-up baseline (minute 25) is judged per process as at most
# max(10% of the base, 1 MiB), on RssAnon + VmSwap: 1 MiB is the size of one allocator step on the 6 MiB web.
ABS_MEM_TOL_MIB = 1.0
# Task 4.5 (the >= 6 h run): the slope of the anonymous footprint over the last 5 hours, projected to 7 days, must stay under 50 MiB.
# The bot prunes its own old automatic clips (at most 24, 16 a kind): in the real data dir a clip left by an earlier run may go; never a manual one.
# Exactly what `ddai_clip::store::parse_auto_name` accepts (`<kind>-<tick>-s<severity>`, not `manual-*`): nothing else is the bot's to prune.
PRUNED_AUTOCLIP = re.compile(r"^clips/(?!manual-)[^/]+-[0-9]+-s[0-9]+\.clip$")
# A Cl_Kill must end the life (a new life starts) within this many ticks, or it did nothing (sv_kill_protection, E-011).
KILL_EFFECT_TICKS = 50
MAX_DEAD_KILLS_IN_A_ROW = 3
# Task 4.6 (D-078): the one chat-channel message the bot may send, as the outgoing audit labels it (`ddai_client::session::SERVER_COMMAND_KILL_LABEL`).
KILL_COMMAND_LABEL = "Cl_Say(/kill)"
# Task 4.9 (D-094): the owner's website chat, as the outgoing audit labels it (`ddai_client::session::OWNER_SAY_LABEL`). Allowed only
# against the report's own `owner_chat` numbers (see `analyze_chat`). Task 4.9b: a line the owner typed that starts with "/" (a server
# command: /spec, /emote, /w, even /kill) is such a line too and carries this label, never the fallback's: the audit knows the path
# that sent it, not the text, and the text is never in a report.
OWNER_SAY_LABEL = "Cl_Say(owner)"
# Task 4.10 (D-100): the DDNet timeout code `/timeout <code>` the bot sends once per join (`ddai_client::session::TIMEOUT_CODE_LABEL`): on the
# wire at most once per "in game" line of the log, plus the repeats while a same-name ghost exists (owner's decision of 2026-10-06: every
# 30 s, at most 35 per ghost, each logged as "timeout code re-sent while a same-name player is present"), never refused (see `analyze_chat`).
TIMEOUT_MAX_RESENDS = 35
TIMEOUT_CODE_LABEL = "Cl_Say(/timeout)"
SLOPE_WINDOW_S = 5 * 3600.0
SLOPE_LIMIT_MIB_WEEK = 50.0
INGAME_RECONNECT_WINDOW_S = 600
# What the bot may put on the wire: the join's one-offs plus Cl_Kill / Cl_SetTeam (docs/formats.md 21.6).
ALLOWED_OUTGOING = {
    "Cl_Say(/kill)",  # task 4.6, D-078: the typed /kill fallback; every other Cl_Say stays a failure
    "Cl_Say(owner)",  # task 4.9, D-094: a line the owner typed on the website; judged against report["owner_chat"]
    "Cl_Say(/timeout)",  # task 4.10, D-100: the stock timeout code, once per join; judged against the log's "in game" lines
    "Cl_StartInfo",
    "Cl_IsDDNetLegacy",
    "Cl_ShowDistance",
    "Cl_ShowOthers",
    "Cl_EnableSpectatorCount",
    "Cl_CameraInfo",
    "Cl_Kill",
    "Cl_SetTeam",
}


def parse_ts(text):
    return datetime.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def strip_ansi(s):
    return ANSI.sub("", s)


def fields(text):
    out = {}
    for k, v in KV.findall(text):
        out[k] = v[1:-1] if v.startswith('"') else v
    return out


def as_int(d, k, default=0):
    try:
        return int(d.get(k, default))
    except (TypeError, ValueError):
        return default


class BotLog:
    """Incremental parser of the bot's tracing output (stderr, a log file or `journalctl -o cat`)."""

    # The messages that are expected during a healthy soak (restart, map change, reconnect); any other WARN or
    # ERROR is listed in the report.
    EXPECTED_WARN = ("disconnected", "reconnecting")

    def __init__(self):
        self.events = []  # (epoch, kind, dict) in log order, for the analysis
        self.counts = {}
        self.lat = {}
        self.slots = {}
        self.stats = {}
        self.margin = {}
        self.status_blocks = []  # (epoch, lat, stats): every "bot status" report
        self.other_warnings = {}
        self.panics = 0
        self.lines = 0
        self.last_ts = None
        self._cur_block = None

    def count(self, kind):
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def feed(self, raw):
        line = strip_ansi(raw.rstrip("\r\n"))
        if not line:
            return
        self.lines += 1
        if "panicked at" in line or line.startswith("thread '") and "panicked" in line:
            self.panics += 1
            self.count("panic")
        m = HDR.match(line)
        if not m:
            self._continuation(line)
            return
        ts, level, target, msg = m.groups()
        msg = msg or ""
        epoch = parse_ts(ts)
        self.last_ts = epoch
        if self._cur_block is not None:
            self._flush_block()
        self._header(epoch, level, target, msg)

    def _header(self, epoch, level, target, msg):
        f = fields(msg)
        if msg.startswith("bot status") or msg.startswith("the bot stopped"):
            self._cur_block = {"epoch": epoch, "lat": {}, "stats": {}}
            self._continuation(msg.split("\n", 1)[-1] if "\n" in msg else "")
            if msg.startswith("the bot stopped"):
                self.events.append((epoch, "stopped", f))
            return
        if msg.startswith("unstick: Cl_Kill"):
            self.count("killed")
            self.events.append((epoch, "kill", {"tick": as_int(f, "tick"), "reason": f.get("reason", "?")}))
        elif msg.startswith("requesting Cl_Kill"):
            self.count("kill_requested")
        elif msg.startswith("requesting /kill"):
            # task 4.6 (D-078): the typed `/kill` fallback after a Cl_Kill that had no effect
            self.count("kill_command")
            self.events.append((epoch, "killcmd", {"tick": as_int(f, "tick")}))
        elif msg.startswith("kill protection: a /kill ended a life"):
            self.count("kill_protection_learned")
            self.events.append((epoch, "learned", {"minutes": f.get("life_minutes", "?")}))
        elif msg.startswith("blocked by"):
            self.count("blocked_by")
        elif msg.startswith("block "):
            self.count("block")
        elif msg.startswith("life started"):
            self.count("life_started")
            self.events.append((epoch, "life", {"tick": as_int(f, "tick")}))
        elif msg.startswith("clip saved"):
            self.count("clip_saved")
            self.events.append((epoch, "clip", f))
        elif msg.startswith("target "):
            self.count("target_changed")
        elif msg.startswith("input margin"):
            self.margin = {k: v for k, v in f.items()}
            self.margin["_epoch"] = epoch
        elif msg.startswith("game tick went backwards"):
            self.count("tick_reset")
            self.events.append((epoch, "tick_reset", f))
        elif msg.startswith("map changing"):
            self.count("map_changing")
            self.events.append((epoch, "map_changing", {"map": msg.split("map=", 1)[-1].strip()}))
        elif msg.startswith("map ready"):
            self.count("map_ready")
            mm = re.search(r"map=(.*?) w=(\d+) h=(\d+)", msg)
            self.events.append((epoch, "map_ready", {"map": mm.group(1) if mm else "?"}))
        elif msg.startswith("timeout code re-sent while a same-name player is present"):
            self.count("timeout_resent")
        elif msg.startswith("in game"):
            self.count("in_game")
            self.events.append((epoch, "in_game", {}))
        elif msg.startswith("disconnected"):
            self.count("disconnected")
            self.events.append((epoch, "disconnected", f))
        elif msg.startswith("reconnecting"):
            self.count("reconnecting")
            self.events.append((epoch, "reconnecting", f))
        elif msg.startswith("starting the bot"):
            self.events.append((epoch, "start", f))
        if level in ("WARN", "ERROR"):
            self.count(level.lower())
            expected = level == "WARN" and msg.startswith(self.EXPECTED_WARN)
            if not expected:
                key = (level + " " + target + ": " + re.sub(r"\d+", "N", msg))[:160]
                self.other_warnings[key] = self.other_warnings.get(key, 0) + 1
                self.events.append((epoch, "warn", {"level": level, "msg": msg[:200]}))

    def _continuation(self, line):
        if self._cur_block is None:
            return
        b = self._cur_block
        m = LAT.match(line)
        if m:
            name, n, p50, p90, p99, mx = m.groups()
            b["lat"][name] = {"n": int(n), "p50": int(p50), "p90": int(p90), "p99": int(p99), "max": int(mx)}
        m = SLOTS.match(line)
        if m:
            b["slots"] = {k: int(v) for k, v in NUM_EQ.findall(m.group(1).split(" stats=")[0])}
        m = STATS.search(line)
        if m:
            b["stats"] = {k: int(v) for k, v in NUM.findall(m.group(1))}
            self._flush_block()

    def _flush_block(self):
        b, self._cur_block = self._cur_block, None
        if b is None or not b["lat"]:
            return
        self.lat = b["lat"]
        if b.get("slots"):
            self.slots = b["slots"]
        if b["stats"]:
            self.stats = b["stats"]
        self.status_blocks.append((b["epoch"], b["lat"], b["stats"]))

    def finish(self):
        if self._cur_block is not None:
            self._flush_block()


NUM_EQ = re.compile(r"(\w+)=(\d+)")


def parse_log_file(path):
    log = BotLog()
    with open(path, errors="replace") as fh:
        for line in fh:
            log.feed(line)
    log.finish()
    return log


# ------------------------------------------------------------------------------------------------ statistics


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def window_vals(points, t0, t1):
    return [v for (t, v) in points if t0 <= t <= t1 and v is not None]


def theil_sen(points):
    """Median of pairwise slopes (per second) over at most ~400 points."""
    pts = points
    if len(pts) > 400:
        step = len(pts) / 400.0
        pts = [pts[int(i * step)] for i in range(400)]
    slopes = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dt = pts[j][0] - pts[i][0]
            if dt > 0:
                slopes.append((pts[j][1] - pts[i][1]) / dt)
    return statistics.median(slopes) if slopes else 0.0


def fmt(x, nd=0):
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def table(headers, rows):
    cols = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(c[i]) for c in cols) for i in range(len(headers))]
    out = []
    for k, r in enumerate(cols):
        out.append("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |")
        if k == 0:
            out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ the analysis


class Result:
    def __init__(self):
        self.checks = []  # (name, ok, detail)
        self.sections = []

    def check(self, name, ok, detail):
        self.checks.append((name, bool(ok), detail))

    def section(self, title, body):
        self.sections.append((title, body))

    @property
    def ok(self):
        return all(ok for _, ok, _ in self.checks)


def load_jsonl(path):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def proc_segments(samples, name, t_end):
    """Contiguous samples of one process (same pid), as {pid, start, end, pts}."""
    segs = []
    for s in samples:
        if s["t"] > t_end:
            continue
        p = s.get("procs", {}).get(name)
        if not p or not p.get("pid"):
            continue
        if segs and segs[-1]["pid"] == p["pid"]:
            segs[-1]["pts"].append((s["t"], p))
        else:
            segs.append({"pid": p["pid"], "pts": [(s["t"], p)]})
    for seg in segs:
        seg["start"] = seg["pts"][0][0]
        seg["end"] = seg["pts"][-1][0]
    return segs


def analyze_processes(res, samples, meta, t_end):
    baseline = meta["baseline_s"]
    width = max(60.0, meta["sample_s"] * 3)
    min_seg = baseline + 2 * width
    rows = []
    names = ["bot", "web", "s1", "server", "pweb"]  # pweb: the production web unit (real-data runs), reported, not gated
    slope_rows = []
    for name in names:
        segs = proc_segments(samples, name, t_end)
        if name in ("bot", "web"):
            res.check(
                f"the {name} process ran the whole time (one pid, never restarted by a crash)",
                len(segs) == 1,
                f"{len(segs)} pid segment(s): {[(sg['pid'], round(sg['start']), round(sg['end'])) for sg in segs]}",
            )
        for seg in segs:
            pts = seg["pts"]
            dur = seg["end"] - seg["start"]
            # The base window opens after the warm-up (the latency rings fill in RING / 25 Hz = 22 min; meta baseline_s = 25 min) and
            # never earlier than 10 minutes into this process (a server restarted mid-run has its own warm-up).
            b0 = max(baseline, seg["start"] + 600)
            # The VM is shared: under memory pressure the kernel drops file-backed pages (the binary's text) from RSS and swaps
            # anonymous pages out, so the resident size alone falls and rises without anything being allocated. The judged size
            # is therefore the anonymous footprint (heap + stacks): RssAnon + VmSwap, when the journal has them; else RSS.
            has_anon = all(p.get("anon_kb") is not None for _, p in pts)
            size = (lambda p: p["anon_kb"] + (p.get("swap_kb") or 0)) if has_anon else (lambda p: p["rss_kb"])  # noqa: E731
            has_swap = has_anon
            base = window_vals([(t, size(p)) for t, p in pts], b0, b0 + width)
            end = window_vals([(t, size(p)) for t, p in pts], seg["end"] - width, seg["end"])
            rbase = window_vals([(t, p["rss_kb"]) for t, p in pts], b0, b0 + width)
            rend = window_vals([(t, p["rss_kb"]) for t, p in pts], seg["end"] - width, seg["end"])
            thr = [(t, p["thr"]) for t, p in pts]
            fds = [(t, p["fds"]) for t, p in pts]
            cpu = [p["cpu"] for t, p in pts if p.get("cpu") is not None]
            judged = seg["end"] - b0 >= 600 and base and end
            growth = None
            rgrowth = None
            if judged:
                growth = (med(end) - med(base)) / med(base) * 100.0
                rgrowth = (med(rend) - med(rbase)) / med(rbase) * 100.0
            thr_b = med(window_vals(thr, b0, b0 + width))
            thr_e = med(window_vals(thr, seg["end"] - width, seg["end"]))
            fd_b = med(window_vals(fds, b0, b0 + width))
            fd_e = med(window_vals(fds, seg["end"] - width, seg["end"]))
            rows.append(
                (
                    name,
                    seg["pid"],
                    dur,
                    med(base),
                    med(end),
                    growth,
                    max(size(p) for _, p in pts),
                    thr_b,
                    thr_e,
                    max(v for _, v in thr),
                    fd_b,
                    fd_e,
                    max(v for _, v in fds),
                    med(cpu),
                    judged,
                    rgrowth,
                    max((p.get("swap_kb") or 0) for _, p in pts),
                )
            )
            if not judged or name == "pweb":
                continue
            tag = f"{name} pid {seg['pid']}"
            limit = 10.0
            what = "anon+swap" if has_swap else "RSS"
            abs_mib = (med(end) - med(base)) / 1024
            allowed = max(limit / 100.0 * med(base) / 1024, ABS_MEM_TOL_MIB)
            res.check(
                f"{what} growth <= max(10%, {ABS_MEM_TOL_MIB:.0f} MiB) ({tag})",
                abs_mib <= allowed,
                f"{med(base) / 1024:.1f} -> {med(end) / 1024:.1f} MiB ({growth:+.1f}%, {abs_mib:+.1f} MiB) "
                f"between t={b0:.0f}s and t={seg['end']:.0f}s (allowed {allowed:+.1f} MiB)"
                + (f"; resident alone {rgrowth:+.1f}%" if has_swap else ""),
            )
            tol_thr = max(2, 0.05 * (thr_b or 0))
            res.check(
                f"no thread growth ({tag})",
                thr_e <= thr_b + tol_thr,
                f"median {thr_b:.0f} -> {thr_e:.0f}, max {max(v for _, v in thr)}",
            )
            tol_fd = max(2, 0.05 * (fd_b or 0))
            res.check(
                f"no fd growth ({tag})",
                fd_e <= fd_b + tol_fd,
                f"median {fd_b:.0f} -> {fd_e:.0f}, max {max(v for _, v in fds)}",
            )
            if name in ("bot", "web"):
                if seg["end"] - b0 >= SLOPE_WINDOW_S:
                    wpts = [(t, size(p) / 1024.0) for t, p in pts if t >= seg["end"] - SLOPE_WINDOW_S]
                    slope_h = theil_sen(wpts) * 3600.0
                    week = slope_h * 168.0
                    head = med([v for _, v in wpts[:120]])
                    tail = med([v for _, v in wpts[-120:]])
                    slope_rows.append((name, f"{wpts[0][0] / 3600:.2f}-{wpts[-1][0] / 3600:.2f}", len(wpts), fmt(head, 2), fmt(tail, 2), f"{slope_h:+.3f}", f"{week:+.1f}"))
                    res.check(
                        f"memory slope over the last 5 h projects to < {SLOPE_LIMIT_MIB_WEEK:.0f} MiB per 7 days ({tag})",
                        week < SLOPE_LIMIT_MIB_WEEK,
                        f"Theil-Sen {slope_h:+.3f} MiB/h over {len(wpts)} samples -> {week:+.1f} MiB per 7 days "
                        f"(first / last 30 min of the window: {head:.2f} / {tail:.2f} MiB)",
                    )
                else:
                    res.section(
                        f"Memory slope not judged ({name})",
                        f"only {(seg['end'] - b0) / 3600:.2f} h after the baseline; the 5 h slope gate needs {SLOPE_WINDOW_S / 3600:.0f} h (a >= 5.5 h run)",
                    )
    if not any(r[14] for r in rows):
        res.section(
            "Memory gate not judged",
            f"no process has 10 minutes after the baseline at t={baseline:.0f} s (the run is shorter than {baseline / 60 + 10:.0f} min): "
            "growth would be warm-up, so it is not gated here",
        )
    body = table(
        ["proc", "pid", "span s", "mem base MiB", "mem end MiB", "growth %", "mem max MiB", "swap max MiB", "thr b/e/max", "fds b/e/max", "cpu% med"],
        [
            (
                r[0],
                r[1],
                fmt(r[2]),
                fmt(r[3] / 1024 if r[3] else None, 1),
                fmt(r[4] / 1024 if r[4] else None, 1),
                fmt(r[5], 1) if r[5] is not None else "(short)",
                fmt(r[6] / 1024, 1),
                fmt(r[16] / 1024, 1),
                f"{fmt(r[7])}/{fmt(r[8])}/{r[9]}",
                f"{fmt(r[10])}/{fmt(r[11])}/{r[12]}",
                fmt(r[13], 1),
            )
            for r in rows
        ],
    )
    res.section("Processes: memory (anonymous resident + swap; RSS where the journal has no RssAnon), threads, fds, CPU (base = minute 25 of the run, end = last minute)", body)
    if slope_rows:
        res.section(
            "Memory slope over the last 5 h (anonymous MiB, Theil-Sen; 7 days = 168 h)",
            table(["proc", "window h", "samples", "start MiB", "end MiB", "slope MiB/h", "MiB per 7 days"], slope_rows),
        )


def analyze_creep(res, samples, events, meta, t_end):
    """Informational: the slope of the anonymous footprint between the disturbances (restart, map change and back), so a staircase
    of allocator steps can be told from a creep. Not a gate: the spec's gate is the minute-10-to-end growth above."""
    marks = [e["t"] for e in events if e.get("kind") in ("server_restart", "map_change")]
    if not marks:
        return
    bounds = [0.0] + sorted(marks) + [t_end]
    edges = [(bounds[i] + 120, bounds[i + 1] - 5) for i in range(len(bounds) - 1)]
    rows = []
    for name in ("bot", "web"):
        for a, b in edges:
            pts = []
            for smp in samples:
                p = (smp.get("procs") or {}).get(name)
                if p and a <= smp["t"] <= b and p.get("anon_kb") is not None:
                    pts.append((smp["t"], (p["anon_kb"] + (p.get("swap_kb") or 0)) / 1024.0))
            if len(pts) < 8 or pts[-1][0] - pts[0][0] < 300:
                continue
            rows.append((name, f"{a:.0f}-{b:.0f}", f"{med([v for _, v in pts[:4]]):.2f}", f"{med([v for _, v in pts[-4:]]):.2f}",
                         f"{theil_sen(pts) * 3600:+.2f}"))
    if rows:
        res.section(
            "Creep between the disturbances (anonymous MiB; Theil-Sen slope in MiB/h; informational)",
            table(["proc", "t s", "start MiB", "end MiB", "slope MiB/h"], rows),
        )


def buckets(points, t_end, n=6):
    size = t_end / n
    out = []
    for i in range(n):
        out.append((i * size, (i + 1) * size, [v for (t, v) in points if i * size <= t < (i + 1) * size and v is not None]))
    return out


def analyze_latency(res, samples, meta, log, t0_epoch, t_end):
    # The latency trend has its own baseline (minute 10, or 40% of a short run): it does not need the memory warm-up of 25 minutes.
    baseline = min(600.0, 0.4 * t_end)
    series = {}
    for epoch, lat, stats in log.status_blocks:
        t = epoch - t0_epoch
        if t < 0 or t > t_end:
            continue
        for name in ("total", "brain", "overhead", "wire", "queue", "pick"):
            if name in lat:
                series.setdefault(name, []).append((t, lat[name]["p99"]))
                series.setdefault(name + "50", []).append((t, lat[name]["p50"]))
    loads = [(s["t"], s["load"][0]) for s in samples if s.get("load") and s["t"] <= t_end]
    rows = []
    n = 6
    size = t_end / n
    for i in range(n):
        a, b = i * size, (i + 1) * size
        row = [f"{a / 60:.0f}-{b / 60:.0f}"]
        for name in ("overhead50", "overhead", "brain50", "brain", "total", "wire", "queue"):
            row.append(fmt(med(window_vals(series.get(name, []), a, b))))
        row.append(fmt(med(window_vals(loads, a, b)), 1))
        rows.append(row)
    res.section(
        "Latency by time bucket (median of the 10 s reports, us; p99 are trailing-ring percentiles; load = 1-min load average on 8 vCPU)",
        table(["min", "ovh p50", "ovh p99", "brain p50", "brain p99", "total p99", "wire p99", "queue p99", "load1"], rows),
    )
    ov = series.get("overhead", [])
    span = t_end - baseline
    first = med(window_vals(ov, baseline, baseline + 0.3 * span))
    last = med(window_vals(ov, t_end - 0.3 * span, t_end))
    slope = theil_sen([(t, v) for t, v in ov if t >= baseline]) * 3600 if ov else 0.0
    if first is None or last is None:
        res.check("overhead p99 stable", False, "no latency reports in the bot log")
        return
    ok = last <= max(1.25 * first, first + 150)
    res.check(
        "overhead p99 stable (last 30% <= max(1.25x, +150 us) of first 30% after minute 10)",
        ok,
        f"first {first:.0f} us, last {last:.0f} us, Theil-Sen slope {slope:+.0f} us/h",
    )
    # The decision budget (D-042) and the hot ratio of late inputs are reported, not gated: they depend on the load.
    over = [v for _, v in ov if v > 500]
    res.section(
        "Overhead against the D-042 target",
        f"overhead p99 > 0.5 ms in {len(over)} of {len(ov)} reports; max report {max(v for _, v in ov):.0f} us"
        if ov
        else "no data",
    )


def analyze_kills(res, log):
    kills = [(e, d) for (e, k, d) in log.events if k == "kill"]
    last = None
    bad = []
    gaps = []
    for epoch, kind, d in log.events:
        if kind in ("tick_reset", "map_changing", "disconnected"):
            last = None
        elif kind == "kill":
            tick, reason = d["tick"], d["reason"]
            if last is not None:
                gap = tick - last
                gaps.append((gap, reason))
                need = WB_KILL_COOLDOWN_TICKS if reason == "WayBlockLying" else KILL_COOLDOWN_TICKS
                if gap < need:
                    bad.append((tick, reason, gap))
            last = tick
    by_reason = {}
    for _, d in kills:
        by_reason[d["reason"]] = by_reason.get(d["reason"], 0) + 1
    res.check(
        "every Cl_Kill within the cooldowns (500 ticks; WayBlockLying 100)",
        not bad,
        f"{len(kills)} kills {by_reason}; min gap {min((g for g, _ in gaps), default='-')} ticks; violations {bad[:5]}",
    )
    analyze_kill_liveness(res, log)
    return len(kills)


def analyze_kill_liveness(res, log):
    """Task 4.5 (review F7; task 4.6 adds the `/kill` fallback): a `Cl_Kill` the server drops is invisible to every other check. DDNet refuses it silently once a life is
    older than `sv_kill_protection` minutes (20 by default): the bot then sits frozen and asks every 10 s for hours (the 6 h rehearsal:
    93 minutes, 548 of 677 kills). Each kill must be followed by a new life within KILL_EFFECT_TICKS (the respawn the kill asks for);
    a kill whose `/kill` fallback (task 4.6) was followed by a new life within the same window took effect too.
    MAX_DEAD_KILLS_IN_A_ROW or more that did nothing in a row fail. The last kill of the log has no later tick to judge by and is not counted."""
    evs = log.events
    judged = dead = longest = run = 0
    runs = []
    command_effects = 0
    commands_sent = sum(1 for _, k, _ in evs if k == "killcmd")
    for idx, (epoch, kind, d) in enumerate(evs):
        if kind in ("tick_reset", "map_changing", "disconnected"):
            if run:
                runs.append(run)
            run = 0
            continue
        if kind != "kill":
            continue
        later_kill = any(k == "kill" for _, k, _ in evs[idx + 1 :])
        if not later_kill and not any(k == "life" for _, k, _ in evs[idx + 1 :]):
            break  # the final kill: the log ended before its effect could show
        t = d["tick"]
        ok = False
        command_at = None  # a `/kill` after this decision (task 4.6): the life may start within the window of that instead
        for _, k2, d2 in evs[idx + 1 :]:
            if k2 == "killcmd" and command_at is None:
                command_at = d2["tick"]
            elif k2 == "life":
                ok = 0 <= d2["tick"] - t <= KILL_EFFECT_TICKS
                if not ok and command_at is not None and 0 <= d2["tick"] - command_at <= KILL_EFFECT_TICKS:
                    ok = True
                    command_effects += 1
                break
            elif k2 in ("kill", "tick_reset", "map_changing", "disconnected"):
                break
        judged += 1
        if ok:
            if run:
                runs.append(run)
            run = 0
        else:
            dead += 1
            run += 1
    if run:
        runs.append(run)
    longest = max(runs, default=0)
    res.check(
        f"every Cl_Kill takes effect (a new life within {KILL_EFFECT_TICKS} ticks); fewer than {MAX_DEAD_KILLS_IN_A_ROW} that did nothing in a row",
        longest < MAX_DEAD_KILLS_IN_A_ROW,
        f"{judged} kills judged, {dead} did nothing, longest run {longest}; the /kill fallback was sent {commands_sent} time(s), "
        f"{command_effects} kill(s) took effect only through it (the others at once, by the server's notice)"
        + ("; the bot is stuck where the server will not let it die (sv_kill_protection: a life older than 20 minutes)" if longest >= MAX_DEAD_KILLS_IN_A_ROW else ""),
    )


def analyze_chat(res, log, report, console_text):
    out = (report or {}).get("outgoing_game_messages")
    if out is None:
        res.check("0 chat in the outgoing audit", False, "no --report: the outgoing audit is missing")
        return
    labels = sorted(out)
    chat = [k for k in labels if ("Say" in k or "Chat" in k) and k not in (KILL_COMMAND_LABEL, OWNER_SAY_LABEL, TIMEOUT_CODE_LABEL)]
    unknown = [k for k in labels if k not in ALLOWED_OUTGOING]
    refused = {k: v["refused"] for k, v in out.items() if v["refused"]}
    res.check(
        "0 chat in the outgoing audit except the allowlisted /kill, the timeout code and the owner's own lines",
        not chat and not unknown and not refused,
        "outgoing: " + ", ".join(f"{k} x{out[k]['accepted']}" for k in labels) + f"; chat {chat}; unknown {unknown}; refused {refused}",
    )
    # The owner's lines (task 4.9): the bot says them only when the owner types them on the website, so a run with none must show none,
    # and a run with some must account for each: the wire count is at most what the runner handed to the client (`owner_chat.sent`;
    # a line can still be dropped by a session that left the game in between), and none was refused by the allow-list. A server command
    # the owner typed (task 4.9b, `/kill` included) is one of these lines and not the bot's own `/kill` fallback, which is judged below.
    owner = out.get(OWNER_SAY_LABEL)
    if owner is not None:
        counts = (report or {}).get("owner_chat")
        sent = counts.get("sent") if isinstance(counts, dict) else None
        res.check(
            "every Cl_Say(owner) on the wire was handed over by the owner chat (wire <= owner_chat.sent, none refused)",
            sent is not None and owner["accepted"] <= sent and owner["refused"] == 0,
            f"wire {owner['accepted']}, refused {owner['refused']}, owner_chat {counts}",
        )
    # The timeout code (task 4.10): one per join (and one more after a map change, as the official client does), plus the repeats while a
    # same-name ghost exists: at most 1 + 35 per join ("in game" line), and every send beyond the joins' own has its own log line that says a
    # same-name player was present. None refused by the allow-list.
    timeout_code = out.get(TIMEOUT_CODE_LABEL)
    if timeout_code is not None:
        joins = log.counts.get("in_game", 0)
        resent = log.counts.get("timeout_resent", 0)
        res.check(
            "every Cl_Say(/timeout) on the wire is a join's send or a logged repeat for a same-name ghost (wire <= joins + repeats, repeats <= 35 per join, none refused)",
            timeout_code["accepted"] <= joins + resent and resent <= TIMEOUT_MAX_RESENDS * joins and timeout_code["refused"] == 0,
            f"wire {timeout_code['accepted']}, refused {timeout_code['refused']}, in game lines {joins}, repeat lines {resent}",
        )
    kills_sent = out.get("Cl_Kill", {}).get("accepted", 0)
    kill_ticks = (report or {}).get("kill_ticks", [])
    res.check(
        "Cl_Kill on the wire == the bot's own kill decisions",
        kills_sent == len(kill_ticks),
        f"wire {kills_sent}, decisions {len(kill_ticks)}",
    )
    cmds_sent = out.get(KILL_COMMAND_LABEL, {}).get("accepted", 0)
    cmd_ticks = (report or {}).get("kill_command_ticks", [])
    # Each /kill answers a Cl_Kill decision of the bot's own: one at or before it, no more than the 50-tick wait plus the cooldown
    # (500) earlier. The counts alone prove nothing (both come from the same place): this is the property.
    orphans = [c for c in cmd_ticks if not any(0 <= c - k <= KILL_EFFECT_TICKS + KILL_COOLDOWN_TICKS for k in kill_ticks)]
    res.check(
        "every /kill on the wire is the bot's own fallback decision (answers a Cl_Kill decision, none refused)",
        cmds_sent == len(cmd_ticks) and out.get(KILL_COMMAND_LABEL, {}).get("refused", 0) == 0 and not orphans,
        f"wire {cmds_sent}, decisions {len(cmd_ticks)}, without a Cl_Kill decision before them {orphans[:5]}",
    )


def analyze_reconnects(res, log, events):
    drops = [(e, d) for (e, k, d) in log.events if k == "disconnected"]
    ingame = [e for (e, k, d) in log.events if k == "in_game"]
    rows = []
    for e, d in drops:
        nxt = next((x for x in ingame if x > e), None)
        rows.append((e, nxt - e if nxt else None, d.get("reason", "?"), d.get("by_peer", "?")))
    window_bad = False
    for i, (e, _, _, _) in enumerate(rows):
        n = sum(1 for (x, _, _, _) in rows if e <= x < e + INGAME_RECONNECT_WINDOW_S)
        if n > MAX_INGAME_RECONNECTS:
            window_bad = True
    unrecovered = [r for r in rows if r[1] is None]
    restart_events = [ev for ev in events if ev.get("kind") == "server_restart"]
    res.check(
        "reconnects within the in-game budget (<= 3 drops per 600 s) and every drop recovered",
        not window_bad and not unrecovered,
        f"{len(rows)} drop(s); recovery seconds {[round(r[1], 1) if r[1] is not None else None for r in rows]}",
    )
    if restart_events:
        res.check(
            "the server restart was noticed and the bot is back in game",
            len(rows) >= 1 and not unrecovered,
            f"restarts {len(restart_events)}, drops {len(rows)}",
        )
    return rows


def analyze_maps(res, log, events, samples, meta):
    seq = [(e, k, d) for (e, k, d) in log.events if k in ("map_changing", "map_ready")]
    names = [d["map"] for (_, k, d) in seq if k == "map_ready"]
    changes = [ev for ev in events if ev.get("kind") == "map_change"]
    shas = {}
    for s in samples:
        m = (s.get("bot") or {}).get("map")
        if m and m.get("sha256"):
            shas[m["name"]] = m["sha256"]
    mem_files = set()
    for s in samples:
        mem_files.update((s.get("memory") or {}).get("files", {}).keys())
    mem_files.update((meta.get("memory_files") or {}).keys())
    if changes:
        other = changes[0].get("map")
        ok = other in names and names and names[-1] == meta.get("home_map")
        res.check(
            "map change to another block map and back",
            ok,
            f"maps loaded in order: {names}",
        )
        res.check(
            "memory is keyed per map sha256 (one file per map, named by its sha256)",
            len(shas) >= 2 and all(f"{sha}.json" in mem_files for sha in shas.values()),
            f"{len(mem_files)} memory file(s) {sorted(n[:12] for n in mem_files)}; maps seen by the bridge "
            f"{ {k: v[:12] for k, v in shas.items()} }",
        )
        # WB state reset: a console `!where` on the other map says it has no wayblock; back home it holds again
        wheres = [e for e in events if e.get("kind") == "console" and e.get("line") == "!where"]
        t_other = changes[0]["t"]
        t_back = changes[1]["t"] if len(changes) > 1 else None
        if wheres:
            on_other = [e for e in wheres if e["t"] > t_other + 5 and (t_back is None or e["t"] < t_back)]
            at_home = [e for e in wheres if t_back is not None and e["t"] > t_back + 5]
            ok_other = bool(on_other) and all("has none" in e.get("reply", "") for e in on_other)
            ok_home = bool(at_home) and all("has none" not in e.get("reply", "") and "WB:" in e.get("reply", "") for e in at_home)
            res.check(
                "WB state reset on the map change (no wayblock on the other map, held again at home)",
                ok_other and ok_home,
                f"on the other map: {[e.get('reply', '')[:70] for e in on_other]}; back: {[e.get('reply', '')[:70] for e in at_home]}",
            )
        else:
            res.section("WB reset", "not checked: the console is off in this run (the bot's own unit tests cover it)")
    else:
        res.section("Map change", "not performed in this run")
    starts = [d for (e, k, d) in log.events if k == "start"]
    if starts and "wb" in starts[0]:
        res.check(
            "the bot started with the requested wayblock mode",
            starts[0]["wb"] == meta["wb"],
            f"log says wb={starts[0]['wb']}, requested {meta['wb']}",
        )


def analyze_clips(res, samples, t_end):
    cl = [(s["t"], s["clips"]) for s in samples if s.get("clips") and s["t"] <= t_end]
    if not cl:
        res.check("clips bounded", False, "no clip samples")
        return
    max_files = max(c["files"] for _, c in cl)
    max_auto = max(c.get("auto", c["files"]) for _, c in cl)
    max_kind = max((max(c.get("kinds", {}).values(), default=0) for _, c in cl), default=0)
    written = cl[-1][1].get("written", 0)
    pruned = cl[-1][1].get("pruned", 0)
    res.check(
        f"clips bounded by pruning (auto <= {MAX_AUTO_CLIPS}, per kind <= {MAX_CLIPS_PER_KIND})",
        max_auto <= MAX_AUTO_CLIPS and max_kind <= MAX_CLIPS_PER_KIND,
        f"files max {max_files} (auto max {max_auto}, per-kind max {max_kind}); written {written}, pruned {pruned}; "
        f"bytes max {max(c['bytes'] for _, c in cl) / 1024:.0f} KiB",
    )


def analyze_viewer(res, samples, report, t_end):
    v = [(s["t"], s["viewer"]) for s in samples if s.get("viewer") and s["t"] <= t_end]
    if not v:
        res.check("web unit shows the live game", False, "no viewer samples")
        return
    last = v[-1][1]
    frames = last.get("frames", 0)
    res.check(
        "web unit reads the bridge for the whole run (live frames and bot status reached a WebSocket viewer)",
        frames > 25 * 60 and last.get("bot_msgs", 0) > 0,
        f"{frames} live frames, {last.get('bot_msgs', 0)} bot-status messages, {last.get('reconnects', 0)} viewer reconnect(s), "
        f"longest gap between frames {last.get('max_gap_s', 0):.1f} s",
    )
    dropped = (report or {}).get("bridge_clients_dropped")
    b = [(s["t"], s.get("bot", {})) for s in samples if s["t"] <= t_end]
    breconn = max((x.get("bridge_reconnects", 0) for _, x in b), default=0)
    res.check(
        "no bridge reader was dropped for being slow",
        breconn == 0,
        f"the journal's own bridge reader reconnected {breconn} time(s)",
    )


def analyze_flow(res, samples, t_end):
    """Decisions per second, blocks, deaths, freezes by bucket."""
    rows = []
    n = 6
    size = t_end / n
    ss = [s for s in samples if s["t"] <= t_end and s.get("bot")]
    for i in range(n):
        a, b = i * size, (i + 1) * size
        win = [s for s in ss if a <= s["t"] < b]
        if len(win) < 2:
            continue
        first, lastw = win[0], win[-1]
        dt = lastw["t"] - first["t"]
        def d(key, src="stats"):
            x0 = (first["bot"].get(src) or {}).get(key)
            x1 = (lastw["bot"].get(src) or {}).get(key)
            return (x1 - x0) if x0 is not None and x1 is not None else None
        dec = d("decisions")
        rows.append(
            [
                f"{a / 60:.0f}-{b / 60:.0f}",
                fmt(dec / dt, 1) if dec is not None and dt else "-",
                d("collapsed"),
                d("deaths"),
                d("self_kills"),
                d("hammer_fires"),
                d("hooks_fired"),
                (lastw["bot"].get("ev") or {}).get("block", 0) - (first["bot"].get("ev") or {}).get("block", 0),
                (lastw["bot"].get("ev") or {}).get("blocked_by", 0) - (first["bot"].get("ev") or {}).get("blocked_by", 0),
                lastw["bot"].get("frozen_edges", 0) - first["bot"].get("frozen_edges", 0),
            ]
        )
    res.section(
        "Play by time bucket (deltas inside the bucket)",
        table(["min", "dec/s", "collapsed", "deaths", "kills", "hammer", "hooks", "blocks", "blocked by", "freezes"], rows),
    )


def analyze_log_hygiene(res, log, meta, extra_logs):
    res.check(
        "no panic in any log",
        log.panics == 0 and not any(extra_logs.get(k, {}).get("panics") for k in extra_logs),
        f"bot log {log.panics}; other logs {{{', '.join(f'{k}: {v.get('panics', 0)}' for k, v in extra_logs.items())}}}",
    )
    errs = log.counts.get("error", 0)
    res.check("no ERROR line in the bot log", errs == 0, f"{errs} ERROR line(s)")
    names = meta.get("others_names", [])
    leaks = {k: v for k, v in extra_logs.items() if k.startswith("names:") and v.get("hits")}
    res.check(
        "no nickname of another player in the bot's log / journal",
        not leaks,
        f"searched for {len(names)} nickname(s) in {', '.join(k[6:] for k in extra_logs if k.startswith('names:')) or 'the bot log'}: "
        f"{leaks if leaks else '0 hits'}",
    )
    if log.other_warnings:
        body = "\n".join(f"{n:5d} x {k}" for k, n in sorted(log.other_warnings.items(), key=lambda kv: -kv[1])[:20])
    else:
        body = "none"
    res.section("WARN / ERROR lines other than disconnected / reconnecting", body)


def analyze(run_dir, baseline=None):
    run = Path(run_dir)
    meta = json.loads((run / "meta.json").read_text())
    if baseline is not None:
        meta["baseline_s"] = float(baseline)  # re-analysing an older run with a different memory baseline
    samples = load_jsonl(run / "journal.jsonl")
    events = load_jsonl(run / "events.jsonl")
    log = parse_log_file(run / "bot.log") if (run / "bot.log").exists() else BotLog()
    report = None
    rp = run / "bot-report.json"
    if rp.exists():
        try:
            report = json.loads(rp.read_text())
        except json.JSONDecodeError:
            report = None
    res = Result()
    t_end = meta.get("t_end") or (samples[-1]["t"] if samples else 0)
    t0_epoch = meta["t0_epoch"]

    res.section(
        "Run",
        f"label {meta['label']}, mode {meta['mode']}, brain hybrid, wb {meta['wb']}, duration {t_end:.0f} s "
        f"({t_end / 60:.1f} min), {len(samples)} journal samples every {meta['sample_s']} s, "
        f"baseline minute {meta['baseline_s'] / 60:.1f}; binary {meta.get('binary', '?')}",
    )
    # exit and panics
    exit_code = (report or {}).get("exit_code", meta.get("bot_exit_code"))
    gave_up = (report or {}).get("gave_up")
    res.check(
        "no unexpected exit: the bot ended by our stop request, exit code 0, never gave up",
        exit_code == 0 and not gave_up and not meta.get("bot_died_early"),
        f"exit code {exit_code}, gave up {gave_up}, early death {meta.get('bot_died_early', False)}",
    )
    scripted_bad = [e for e in events if e.get("kind") == "scripted_exit" and e.get("code") not in (0, None, -15)]
    res.section(
        "Scripted bots",
        f"{sum(1 for e in events if e.get('kind') == 'scripted_start')} starts, "
        f"{sum(1 for e in events if e.get('kind') == 'scripted_exit')} exits; non-zero exits (excluding our SIGTERM): "
        f"{[(e['name'], e['code']) for e in scripted_bad] or 'none'}",
    )
    extra = {}
    for logname in meta.get("extra_logs", []):
        p = run / logname
        if p.exists():
            txt = p.read_text(errors="replace")
            extra[logname] = {"panics": txt.count("panicked at")}
    # nicknames of the others: searched in the bot log and in the journal text, if any
    names = meta.get("others_names", [])
    for logname in ["bot.log"] + [x for x in meta.get("name_logs", [])]:
        p = run / logname
        if p.exists():
            txt = p.read_text(errors="replace")
            hits = {n: txt.count(n) for n in names if n and n in txt}
            extra["names:" + logname] = {"hits": hits}
    prod = meta.get("prod_bot_dir")
    if prod and prod.get("real"):
        # runs recorded before the pruning rule: a pruned automatic clip is the bot's doing, not a removal by the harness
        pruned = [d for d in prod["diff"] if d.startswith("removed ") and PRUNED_AUTOCLIP.match(d[len("removed "):])]
        prod = dict(prod, diff=[d for d in prod["diff"] if d not in pruned])
        prod["unchanged"] = not prod["diff"] and not any("relations.json" in d for d in prod["diff"])
        if pruned:
            prod["relations"] = f"{prod.get('relations', '?')}; {len(pruned)} old automatic clip(s) pruned by the bot itself"
        res.check(
            "~/aiddnet/data/bot (REAL data dir, production layout): nothing that was there before was removed or had its mode changed; "
            "relations.json is as before (absent, or the same lists)",
            prod["unchanged"],
            f"{prod['entries']} entries before; differences: {prod['diff'] or 'none'}; relations.json: {prod.get('relations', '?')}",
        )
    elif prod:
        res.check(
            "~/aiddnet/data/bot (production state) untouched: listing, modes, sizes and mtimes as before the run",
            prod["unchanged"],
            f"{prod['entries']} entries before; differences: {prod['diff'] or 'none'}",
        )
    else:
        res.section("Production data/bot", "not recorded: a run made before the harness got a private data dir (those runs used and moved data/bot)")
    analyze_log_hygiene(res, log, meta, extra)
    analyze_processes(res, samples, meta, t_end)
    mem = [(s["t"], s["mem"]) for s in samples if s.get("mem") and s["t"] <= t_end]
    if mem:
        res.section(
            "Machine memory (shared VM)",
            f"MemAvailable min {min(m['avail_mb'] for _, m in mem)} MiB; swap in use {min(m['swap_used_mb'] for _, m in mem)}"
            f"-{max(m['swap_used_mb'] for _, m in mem)} MiB over the run",
        )
    analyze_creep(res, samples, events, meta, t_end)
    analyze_latency(res, samples, meta, log, t0_epoch, t_end)
    analyze_flow(res, samples, t_end)
    analyze_kills(res, log)
    analyze_chat(res, log, report, "")
    rows = analyze_reconnects(res, log, events)
    analyze_maps(res, log, events, samples, meta)
    analyze_clips(res, samples, t_end)
    analyze_viewer(res, samples, report, t_end)

    # margin
    mg = [(s["t"], s["bot"].get("margin")) for s in samples if s.get("bot") and s["bot"].get("margin") and s["t"] <= t_end]
    if mg:
        lastm = mg[-1][1]
        res.section(
            "Input timing (last periodic report of the driver)",
            f"inputs {lastm.get('count')}, late {lastm.get('late')} ({float(lastm.get('late_fraction', 0)) * 100:.3f}%), "
            f"stalls {lastm.get('stalls')}, margin {lastm.get('margin_ms')} ms (adaptive {lastm.get('adaptive')}, "
            f"{lastm.get('changes')} changes), time_left p1/p50/p99 {lastm.get('min_ms')}/{lastm.get('p50_ms')}/{lastm.get('p99_ms')} ms",
        )
    if report and report.get("input_margin"):
        m = report["input_margin"]
        res.section(
            "Input timing (final report, last connection)",
            f"inputs {m['count']}, late {m['late']} ({m['late_fraction'] * 100:.3f}%), margin {m['margin_ms']} ms, "
            f"{m['margin_changes']} changes, time_left p50 {m['p50_ms']} ms, p99 {m['p99_ms']} ms",
        )
    if report:
        lat = report.get("latency_us", {})
        res.section(
            "Whole-run latency (the bot's final ring: the last 32 768 samples, about 22 minutes)",
            table(
                ["series", "n", "p50 us", "p90 us", "p99 us", "max us"],
                [(k, v["n"], v["p50_us"], v["p90_us"], v["p99_us"], v["max_us"]) for k, v in lat.items() if k != "slots"],
            ),
        )

    # render
    lines = [f"SOAK ANALYSIS  {meta['label']}", ""]
    for title, body in res.sections[:1]:
        lines += [title + ": " + body, ""]
    lines.append("ACCEPTANCE")
    for name, ok, detail in res.checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    lines.append("")
    lines.append("VERDICT: " + ("PASS" if res.ok else "FAIL"))
    lines.append("")
    for title, body in res.sections[1:]:
        lines += [f"## {title}", body, ""]
    text = "\n".join(lines)
    (run / "analysis.txt").write_text(text + "\n")
    (run / "analysis.json").write_text(
        json.dumps({"ok": res.ok, "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in res.checks]}, indent=1)
    )
    return res, text


# ------------------------------------------------------------------------------------------------ self test


def _synthetic_run(tmp, *, bot_base_kb=300_000, rss_growth=0.0, fd_growth=0, p99_growth=0.0, bad_kill=False, chat=False, panic=False, clips=10, drops=1, viewer=25, file_growth=0.0, swapped_growth=0.0, prod_dir_ok=True, duration=3600, leak_mib_h=0.0, real_dir=False, dead_kills=0, cmd_kills=0, dead_cmds=False, stray_cmd=False, orphan_cmd=False, owner_wire=0, owner_refused=0, owner_report=None, timeout_wire=0, timeout_refused=0, timeout_resent_lines=0):
    """A fake run (60 minutes by default; `duration` for a long one) whose numbers we control, to prove that every acceptance check can fail."""
    import random

    rng = random.Random(7)
    run = Path(tmp)
    run.mkdir(parents=True, exist_ok=True)
    t0 = 1_700_000_000.0
    D = duration
    sc = D / 3600.0  # the scenario times (restart, map change, reconnects) scale with the run
    meta = {
        "label": "selftest",
        "mode": "process",
        "wb": "auto",
        "t0_epoch": t0,
        "t_end": D,
        "baseline_s": 1500,
        "sample_s": 15,
        "home_map": "Copy Love Box",
        "others_names": ["soak-s1", "soak-s2", "soak-s3"],
        "name_logs": [],
        "bot_exit_code": 0,
        "prod_bot_dir": {"unchanged": prod_dir_ok, "entries": 3, "diff": [] if prod_dir_ok else ["changed relations.json"], **({"real": True, "relations": "absent"} if real_dir else {})},
    }
    (run / "meta.json").write_text(json.dumps(meta))
    with open(run / "journal.jsonl", "w") as fh:
        for i in range(0, D, 15):
            frac = max(0.0, (i - 1500) / (D - 1500))
            procs = {}
            for name, base in (("bot", bot_base_kb), ("web", 40_000), ("s1", 250_000), ("server", 60_000)):
                is_bot = name == "bot"
                noise = 1 + rng.uniform(-0.005, 0.005)
                # anonymous memory 80% of the size, file pages 20%; growth cases: anon grows (rss grows), file pages grow
                # (rss grows, anon flat), or the anon growth sits in swap (rss flat)
                anon = (base * 0.8 * (1 + (rss_growth if is_bot else 0.0) * frac) + (leak_mib_h * 1024.0 * i / 3600.0 if is_bot else 0.0)) * noise
                swap = base * 0.8 * swapped_growth * frac if is_bot else 0.0
                file_kb = base * 0.2 * (1 + (file_growth if is_bot else 0.0) * frac)
                procs[name] = {
                    "pid": 100 + len(name),
                    "rss_kb": int(anon + file_kb),
                    "anon_kb": int(anon),
                    "swap_kb": int(swap),
                    "cpu": 20.0 + rng.uniform(-2, 2),
                    "thr": 14,
                    "fds": 30 + (int(fd_growth * frac) if name == "bot" else 0),
                }
            fh.write(
                json.dumps(
                    {
                        "t": i,
                        "load": [8.0, 8, 8],
                        "procs": procs,
                        "bot": {
                            "map": {"name": "Copy Love Box" if (i < 2300 * sc or i >= 2500 * sc) else "BlmapChill", "sha256": ("a" if (i < 2300 * sc or i >= 2500 * sc) else "b") * 64},
                            "stats": {"decisions": i * 25, "collapsed": i // 100, "deaths": i // 60, "self_kills": 0},
                            "ev": {"block": i // 40, "blocked_by": i // 90},
                            "frozen_edges": i // 30,
                            "bridge_reconnects": 0,
                        },
                        "clips": {"files": clips, "auto": clips, "kinds": {"death": min(clips, 8)}, "bytes": 1000, "written": clips, "pruned": 0},
                        "memory": {"files": {"a" * 64 + ".json": 100, "b" * 64 + ".json": 100}, "bytes": 200},
                        "viewer": {"frames": i * viewer, "bot_msgs": i * 5, "reconnects": 0, "max_gap_s": 0.1},
                    }
                )
                + "\n"
            )
    with open(run / "events.jsonl", "w") as fh:
        fh.write(json.dumps({"t": 720 * sc, "kind": "server_restart"}) + "\n")
        fh.write(json.dumps({"t": 2300 * sc, "kind": "map_change", "map": "BlmapChill"}) + "\n")
    lines = []

    def ts(sec):
        return datetime.datetime.fromtimestamp(t0 + sec, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    for i in range(10, D, 10):
        frac = max(0.0, (i - 600) / (D - 600))
        ov = int(200 * (1 + p99_growth * frac))
        lines.append(f"{ts(i)}  INFO ddai_bot::runner: bot status")
        lines.append(f"total: n=30000 p50=90us p90=200us p99={ov + 3000}us max=9000us")
        lines.append(f"brain: n=30000 p50=30us p90=100us p99=2900us max=8000us")
        lines.append(f"overhead: n=30000 p50=60us p90=100us p99={ov}us max=900us")
        lines.append(f"pick: n=30000 p50=1us p90=1us p99=2us max=9us")
        lines.append(f"queue: n=30000 p50=1us p90=1us p99=2us max=9us")
        lines.append(f"wire: n=30000 p50=9000us p90=15000us p99=19000us max=20000us")
        lines.append("slots: decisions=1 first_slot=1 missed_first_slot=0 as_predicted=1 later_than_predicted=0 earlier_than_predicted=0 stats=BotStats { snapshots: 5, decisions: 4 }")
    lines.append(f"{ts(1)}  INFO ddai_bot::runner: starting the bot server=127.0.0.1:8303 brain=\"hybrid\" mode=\"fight\" wb=\"auto\" strong=false")
    lines.append(f"{ts(100)}  INFO ddai_bot::runner: unstick: Cl_Kill tick=1000 reason=Overdue")
    lines.append(f"{ts(100)}  INFO ddai_bot::runner: life started tick=1002")
    lines.append(f"{ts(120)}  INFO ddai_bot::runner: unstick: Cl_Kill tick={1100 if bad_kill else 1600} reason=Overdue")
    lines.append(f"{ts(120)}  INFO ddai_bot::runner: life started tick={1102 if bad_kill else 1602}")
    lines.append(f"{ts(130)}  INFO ddai_bot::runner: unstick: Cl_Kill tick=1700 reason=WayBlockLying")
    lines.append(f"{ts(130)}  INFO ddai_bot::runner: life started tick=1702")
    for n in range(dead_kills):  # kills the server dropped: no new life follows
        lines.append(f"{ts(200 + 10 * n)}  INFO ddai_bot::runner: unstick: Cl_Kill tick={2300 + 500 * n} reason=Overdue")
    for n in range(cmd_kills):  # kills the server dropped, ended by the /kill fallback 50 ticks later
        base = 4000 + 500 * n
        lines.append(f"{ts(300 + 10 * n)}  INFO ddai_bot::runner: unstick: Cl_Kill tick={base} reason=Overdue")
        lines.append(f"{ts(301 + 10 * n)}  INFO ddai_bot::runner: requesting /kill (fallback: the protocol Cl_Kill had no effect) tick={base + 50}")
        lines.append(f"{ts(302 + 10 * n)}  INFO ddai_bot::runner: life started tick={base + 52}")
    if dead_cmds:  # the server ignored the /kill too
        for n in range(dead_kills):
            lines.append(f"{ts(201 + 10 * n)}  INFO ddai_bot::runner: requesting /kill (fallback: the protocol Cl_Kill had no effect) tick={2350 + 500 * n}")
    lines.append(f"{ts(400)}  INFO ddai_bot::runner: life started tick=9000")
    for k in range(drops):
        lines.append(f"{ts(1500 * sc + 30 * k)}  WARN ddai_bot::runner: disconnected reason=ServerShutdown by_peer=true")
        lines.append(f"{ts(1512 * sc + 30 * k)}  INFO ddai_bot::runner: in game")
    for k in range(timeout_resent_lines):
        lines.append(f"{ts(1600 * sc + 30 * k)}  INFO ddai_client::session: timeout code re-sent while a same-name player is present (len 16, resend {k + 1} of 35)")
    lines.append(f"{ts(2300 * sc)}  INFO ddai_bot::runner: map changing map=BlmapChill")
    lines.append(f"{ts(2300 * sc + 5)}  INFO ddai_bot::runner: map ready map=BlmapChill w=100 h=100")
    lines.append(f"{ts(2500 * sc)}  INFO ddai_bot::runner: map changing map=Copy Love Box")
    lines.append(f"{ts(2500 * sc + 5)}  INFO ddai_bot::runner: map ready map=Copy Love Box w=387 h=250")
    if panic:
        lines.append("thread 'ddai-bot' panicked at crates/x.rs:1:1:")
    (run / "bot.log").write_text("\n".join(lines) + "\n")
    out = {"Cl_StartInfo": {"accepted": 1, "refused": 0}, "Cl_Kill": {"accepted": 3 + dead_kills + cmd_kills, "refused": 0}}
    n_cmds = cmd_kills + (dead_kills if dead_cmds else 0)
    if n_cmds or stray_cmd or orphan_cmd:
        out["Cl_Say(/kill)"] = {"accepted": n_cmds + int(stray_cmd) + int(orphan_cmd), "refused": 0}
    if chat:
        out["Cl_Say"] = {"accepted": 1, "refused": 0}
    if timeout_wire or timeout_refused:
        out["Cl_Say(/timeout)"] = {"accepted": timeout_wire, "refused": timeout_refused}
    if owner_wire or owner_refused:
        out["Cl_Say(owner)"] = {"accepted": owner_wire, "refused": owner_refused}
    (run / "bot-report.json").write_text(
        json.dumps({"exit_code": 0, "gave_up": None, "outgoing_game_messages": out,
                    **({"owner_chat": owner_report} if owner_report is not None else {}), "kill_ticks": [1000, 1600, 1700] + [2300 + 500 * n for n in range(dead_kills)] + [4000 + 500 * n for n in range(cmd_kills)],
                    "kill_command_ticks": [4050 + 500 * n for n in range(cmd_kills)] + ([2350 + 500 * n for n in range(dead_kills)] if dead_cmds else []) + ([99_999] if orphan_cmd else [])})
    )
    return run


def selftest():
    import tempfile

    cases = [
        ("clean run passes", {}, None),
        ("anonymous memory growth fails", {"rss_growth": 0.25}, "anon+swap growth"),
        ("a 5 MiB process growing 0.9 MiB (18%) passes on the 1 MiB floor", {"bot_base_kb": 5_000, "rss_growth": 0.22}, None),
        ("a 5 MiB process growing 1.5 MiB fails", {"bot_base_kb": 5_000, "rss_growth": 0.38}, "growth"),
        ("resident growth from file pages alone is not growth (anon flat)", {"file_growth": 1.0}, None),
        ("anonymous growth that sits in swap (rss flat) is caught", {"swapped_growth": 0.3}, "anon+swap growth"),
        ("production data/bot touched fails", {"prod_dir_ok": False}, "untouched"),
        ("real data dir, nothing changed, passes", {"real_dir": True}, None),
        ("real data dir: a removed or changed entry fails", {"real_dir": True, "prod_dir_ok": False}, "REAL data dir"),
        ("a 6 h run with a 0.2 MiB/h creep (34 MiB per 7 days) passes", {"duration": 21600, "leak_mib_h": 0.2}, None),
        ("a 6 h run with a 0.5 MiB/h leak (84 MiB per 7 days) fails the slope gate", {"duration": 21600, "leak_mib_h": 0.5}, "memory slope over the last 5 h"),
        ("a 6 h flat run passes (and the slope gate ran)", {"duration": 21600, "leak_mib_h": 0.0}, None),
        ("fd growth fails", {"fd_growth": 12}, "fd growth"),
        ("p99 trend fails", {"p99_growth": 4.0}, "overhead p99"),
        ("kill cooldown fails", {"bad_kill": True}, "Cl_Kill within"),
        ("two kills the server dropped pass", {"dead_kills": 2}, None),
        ("three kills the server dropped in a row fail (sv_kill_protection)", {"dead_kills": 3}, "takes effect"),
        ("three kills the /kill fallback ended pass (task 4.6)", {"cmd_kills": 3}, None),
        ("three kills whose /kill the server ignored too fail", {"dead_kills": 3, "dead_cmds": True}, "takes effect"),
        ("a /kill on the wire the bot never decided on fails", {"stray_cmd": True}, "/kill on the wire"),
        ("a /kill with no Cl_Kill decision before it fails (counts equal, tick orphaned)", {"orphan_cmd": True}, "/kill on the wire"),
        ("chat fails", {"chat": True}, "0 chat"),
        ("the owner's lines, all handed over by the owner chat, pass (task 4.9)", {"owner_wire": 3, "owner_report": {"accepted": 3, "sent": 3, "refused": 0, "dropped": 0}}, None),
        ("an owner line the runner handed over but the session dropped (wire < sent) passes", {"owner_wire": 2, "owner_report": {"accepted": 3, "sent": 3, "refused": 0, "dropped": 0}}, None),
        ("more owner lines on the wire than the owner chat sent fail", {"owner_wire": 4, "owner_report": {"accepted": 3, "sent": 3, "refused": 0, "dropped": 0}}, "Cl_Say(owner)"),
        ("an owner line the allow-list refused fails", {"owner_wire": 2, "owner_refused": 1, "owner_report": {"accepted": 3, "sent": 3, "refused": 0, "dropped": 0}}, "Cl_Say(owner)"),
        ("owner lines with no owner_chat numbers in the report fail", {"owner_wire": 1}, "Cl_Say(owner)"),
        ("an owner's own /kill is an owner line next to the fallback's /kill: both pass (task 4.9b)", {"owner_wire": 2, "cmd_kills": 1, "owner_report": {"accepted": 2, "sent": 2, "refused": 0, "dropped": 0}}, None),
        ("an owner's /kill counted as the fallback's (no Cl_Kill decision of the bot behind it) fails (task 4.9b)", {"owner_wire": 1, "stray_cmd": True, "owner_report": {"accepted": 1, "sent": 1, "refused": 0, "dropped": 0}}, "/kill on the wire"),
        ("another Cl_Say label beside the owner's still fails", {"owner_wire": 1, "chat": True, "owner_report": {"accepted": 1, "sent": 1, "refused": 0, "dropped": 0}}, "0 chat"),
        ("the timeout code once per join passes (task 4.10)", {"timeout_wire": 1}, None),
        ("more timeout codes on the wire than joins fail (task 4.10)", {"timeout_wire": 5}, "Cl_Say(/timeout)"),
        ("repeats for a ghost, each logged, pass (owner's decision of 2026-10-06)", {"timeout_wire": 4, "timeout_resent_lines": 3}, None),
        ("repeats on the wire that the log does not show as ghost repeats fail", {"timeout_wire": 4, "timeout_resent_lines": 1}, "Cl_Say(/timeout)"),
        ("more than 35 logged repeats for one join fail", {"timeout_wire": 40, "timeout_resent_lines": 40}, "Cl_Say(/timeout)"),
        ("a timeout code the allow-list refused fails (task 4.10)", {"timeout_wire": 1, "timeout_refused": 1}, "Cl_Say(/timeout)"),
        ("the timeout code beside the owner's lines and another Cl_Say label still fails on the other label", {"timeout_wire": 1, "owner_wire": 1, "chat": True, "owner_report": {"accepted": 1, "sent": 1, "refused": 0, "dropped": 0}}, "0 chat"),
        ("panic fails", {"panic": True}, "no panic"),
        ("unbounded clips fail", {"clips": 60}, "clips bounded"),
        ("reconnect budget fails", {"drops": 4}, "reconnects within"),
        ("dead web viewer fails", {"viewer": 0}, "web unit reads"),
    ]
    failed = 0
    for title, kw, expect in cases:
        with tempfile.TemporaryDirectory() as tmp:
            run = _synthetic_run(Path(tmp) / "r", **kw)
            res, _ = analyze(run)
            failing = [n for n, ok, _ in res.checks if not ok]
            if expect is None:
                good = res.ok
            else:
                good = any(expect in n for n in failing)
            print(("ok   " if good else "FAIL ") + title + (f"  (failing checks: {failing})" if not good or failing else ""))
            failed += 0 if good else 1
    print("selftest:", "PASS" if failed == 0 else f"{failed} case(s) FAILED")
    return failed == 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "--selftest":
        return 0 if selftest() else 1
    baseline = None
    if len(argv) == 4 and argv[2] == "--baseline":
        baseline = float(argv[3])
        argv = argv[:2]
    if len(argv) != 2:
        print(__doc__)
        return 2
    res, text = analyze(argv[1], baseline)
    print(text)
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
