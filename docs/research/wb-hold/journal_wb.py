#!/usr/bin/env python3
"""Task 3.18, live diagnosis of the wayblock hold, journal part: every block of a session, its fate, and what the bot did with the victim.

Input: the journal of the bot unit (`journalctl -u ddnet-ai-bot`, read only; any journalctl output format) as a text file. A session is
the span between two `starting the bot` lines (tick numbers restart there). For every `block tick=T victim="<tag>"` line it reads the
`block escaped ... after=N` / `block held ...` line of the same victim and the `target tick=N target=...` lines in between, and prints:

* `fate`       - `escaped` (free again `after` ticks after the block) / `held` (out for 250 ticks or dead) / `?` (the session ended first);
* `targeted`   - the victim was the target at the moment of the block (a victim that was frozen as a by-product is not);
* `on`         - the share of the ticks between the block and the fate during which the victim was the target;
* `class`      - `natural` (escaped 140..156 ticks after the block: the 3 s freeze ran out), `late` (> 156: the timer was renewed once, the
                 victim left later), `early` (< 140: somebody freed it: a hammer hit, a rope), `held`;
* `kind`       - `never` (never the target), `dropped` (the target for less than half of the time), `kept` (half or more).

Victims are printed as tags (`c<id>-<hash>`), never as nicknames.

    python3 journal_wb.py journal.txt [--csv]
"""
import argparse
import re
import sys
from collections import Counter

EV = re.compile(
    r"(?P<kind>block escaped|block held|block|blocked by|target|life started|starting the bot)\b[^\n]*?"
    r"(?:tick=(?P<tick>-?\d+))?(?: (?:victim|by|target)=(?:Some\()?\"?(?P<who>[^\")\s]*))?"
)
TICK = re.compile(r"\btick=(-?\d+)")
WHO = re.compile(r"\b(?:victim|by|target)=(?:Some\()?(?:\"([^\"]*)\"|None)")
AFTER = re.compile(r"\bafter=(\d+)")
LOGGER = re.compile(r"ddai_bot::runner: (block escaped|block held|block|blocked by|target|life started|starting the bot)\b")


def events(path):
    session = 0
    for raw in open(path, errors="replace"):
        m = LOGGER.search(raw)
        if not m:
            continue
        kind = m.group(1)
        body = raw[m.end():]
        if kind == "starting the bot":
            session += 1
            continue
        t = TICK.search(body)
        if not t:
            continue
        w = WHO.search(body)
        a = AFTER.search(body)
        yield {
            "session": session,
            "kind": kind,
            "tick": int(t.group(1)),
            "who": (w.group(1) if w and w.group(1) else None) if w else None,
            "after": int(a.group(1)) if a else None,
        }


def analyse(evs):
    rows = []
    for k, e in enumerate(evs):
        if e["kind"] != "block":
            continue
        s, t, v = e["session"], e["tick"], e["who"]
        fate, end, after = "?", t + 250, None
        for f in evs[k + 1 :]:
            if f["session"] != s:
                break
            if f["who"] == v and f["kind"] in ("block escaped", "block held"):
                fate = "escaped" if f["kind"] == "block escaped" else "held"
                end, after = f["tick"], f["after"]
                break
        cur = None
        for f in evs[:k]:
            if f["session"] == s and f["kind"] == "target" and f["tick"] <= t:
                cur = f["who"]
        timeline = [(t, cur)] + [(f["tick"], f["who"]) for f in evs if f["session"] == s and f["kind"] == "target" and t < f["tick"] <= end]
        total = on = 0
        for i, (a, c) in enumerate(timeline):
            b = timeline[i + 1][0] if i + 1 < len(timeline) else end
            total += b - a
            on += (b - a) if c == v else 0
        frac = on / max(total, 1)
        if fate == "held":
            cls = "held"
        elif fate == "escaped":
            cls = "natural" if 140 <= (after or 0) <= 156 else ("late" if (after or 0) > 156 else "early")
        else:
            cls = "?"
        kind = "never" if on == 0 else ("dropped" if frac < 0.5 else "kept")
        rows.append(
            {"session": s, "tick": t, "victim": v, "fate": fate, "after": after, "targeted": cur == v, "on": round(frac, 2), "class": cls, "kind": kind}
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("journal")
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()
    evs = sorted(events(a.journal), key=lambda e: (e["session"], e["tick"]))
    # `evs` must stay in journal order for the "before the block" lookups; sort is stable per (session, tick).
    rows = analyse(evs)
    if a.csv:
        print("session,tick,victim,fate,after,targeted,on,class,kind")
        for r in rows:
            print(",".join(str(r[k]) for k in ("session", "tick", "victim", "fate", "after", "targeted", "on", "class", "kind")))
        return
    for r in rows:
        print(
            f"s{r['session']} tick {r['tick']:>6} {r['victim']:<14} {r['fate']:<8} after {str(r['after']):>4}  targeted {str(r['targeted']):<5} on {r['on']:.2f}  {r['class']:<7} {r['kind']}"
        )
    esc = [r for r in rows if r["fate"] == "escaped"]
    print()
    print(f"blocks {len(rows)}: held {sum(r['fate'] == 'held' for r in rows)}, escaped {len(esc)}, unknown {sum(r['fate'] == '?' for r in rows)}")
    print("escaped by class:", dict(Counter(r["class"] for r in esc)))
    print("escaped by what the bot did:", dict(Counter(r["kind"] for r in esc)))
    if esc:
        print(f"mean share of the time the victim was the target (escaped): {sum(r['on'] for r in esc) / len(esc):.2f}")
    bys = Counter(r["victim"] for r in rows)
    print("blocks per victim tag:", dict(bys.most_common()))
    sys.exit(0)


if __name__ == "__main__":
    main()
