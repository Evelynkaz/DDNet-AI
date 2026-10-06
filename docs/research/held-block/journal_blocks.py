#!/usr/bin/env python3
"""Task 3.10, live diagnosis, journal part: what the bot's journal says about the blocks of one session.

Input: the journal of the bot unit (`journalctl -u ddnet-ai-bot -o short-iso`, read only) as a text file. A session is the
span between two `starting the bot` lines; `--server` keeps the sessions of one server. For every `block tick=T victim="<tag>"`
line it reads the `target tick=N target=...` lines and the next blocks of the same victim:

* `repeat`   - the same victim was frozen again within 250 ticks: the first block did not hold (it thawed, and was caught again);
* `left`     - the bot's target was no longer the victim `LEFT_TICKS` (50) ticks after the block (it let go);
* `kept`     - still the target 250 ticks after the block.

Victims are printed as tags (`c<id>-<hash>`), never as nicknames.
"""
import argparse
import re
import sys
from collections import Counter, defaultdict

LINE = re.compile(r"^(\S+) \S+ [^:]*: (?:\S+Z\s+)?(?:INFO|WARN)\s+(\S+): (.*)$")
BLOCK = re.compile(r'^block tick=(-?\d+) victim="([^"]+)"')
BLOCKED = re.compile(r'^blocked by tick=(-?\d+) by="([^"]+)"')
TARGET = re.compile(r'^target tick=(-?\d+) target=(None|Some\("([^"]+)"\))')
START = re.compile(r'^starting the bot server=(\S+) brain="([^"]+)"')
LIFE = re.compile(r"^life started tick=(-?\d+)")
LEFT_TICKS = 50
HOLD_TICKS = 250


def sessions(path, server):
    cur = None
    for raw in open(path, errors="replace"):
        m = re.match(r"^(\S+) \S+ [^:]*: (.*)$", raw.rstrip("\n"))
        if not m:
            continue
        body = m.group(2)
        body = re.sub(r"^\d{4}-\d\d-\d\dT\S+Z\s+(?:INFO|WARN|ERROR)\s+", "", body)
        # body now `module: message`
        mm = re.match(r"^(\S+): (.*)$", body)
        if not mm:
            continue
        mod, msg = mm.groups()
        s = START.match(msg)
        if s and mod == "ddai_bot::runner":
            if cur:
                yield cur
            cur = {"server": s.group(1), "brain": s.group(2), "start": m.group(1), "events": []}
            continue
        if cur is None:
            continue
        for kind, rx in (("block", BLOCK), ("blocked_by", BLOCKED), ("target", TARGET), ("life", LIFE)):
            mt = rx.match(msg)
            if mt:
                cur["events"].append((kind, mt))
                break
    if cur:
        yield cur


def analyse(ev):
    blocks, targets = [], []
    for kind, mt in ev:
        if kind == "block":
            blocks.append((int(mt.group(1)), mt.group(2)))
        elif kind == "target":
            targets.append((int(mt.group(1)), mt.group(3)))
    def target_at(t):
        cur = None
        for tt, tag in targets:
            if tt > t:
                break
            cur = tag
        return cur
    out = []
    for i, (t, v) in enumerate(blocks):
        again = next((t2 for t2, v2 in blocks[i + 1:] if v2 == v and 0 < t2 - t <= HOLD_TICKS), None)
        out.append({
            "tick": t, "victim": v,
            "repeat": None if again is None else again - t,
            "left": target_at(t + LEFT_TICKS) != v,
            "kept_hold": target_at(t + HOLD_TICKS) == v,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("journal")
    ap.add_argument("--server", default="45.141.57.35:8308")
    a = ap.parse_args()
    tot = Counter()
    gaps = []
    for s in sessions(a.journal, a.server):
        if s["server"] != a.server:
            continue
        res = analyse(s["events"])
        n_blocked_by = sum(1 for k, _ in s["events"] if k == "blocked_by")
        c = Counter()
        for r in res:
            c["blocks"] += 1
            c["repeat"] += r["repeat"] is not None
            c["left"] += r["left"]
            c["kept_hold"] += r["kept_hold"]
            c["left_and_no_repeat"] += r["left"] and r["repeat"] is None
            if r["repeat"] is not None:
                gaps.append(r["repeat"])
        c["blocked_by"] = n_blocked_by
        print(s["start"], s["server"], s["brain"], dict(c))
        tot.update(c)
    print("TOTAL", dict(tot))
    if gaps:
        gaps.sort()
        print("repeat gaps (ticks): n=%d p10=%d p50=%d p90=%d" % (len(gaps), gaps[len(gaps) // 10], gaps[len(gaps) // 2], gaps[len(gaps) * 9 // 10]))


if __name__ == "__main__":
    main()
