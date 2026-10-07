#!/usr/bin/env python3
"""E-029 (task 8.6): tables of the critical-decision analysis (`train es critical` files).

  critical_tables.py <forward.json|reverse.json>

A swap FLIPS when the held-block outcome differs from the unswapped run's. Forward (the fly plays, the planner's action is swapped in) the
interesting flips are not-held -> held; reverse (the planner plays, the fly's action is swapped in) held -> not-held. Decisions are counted from
the freeze (2 ticks each).
"""
import json
import math
import sys
from collections import Counter, defaultdict

COMPONENTS = ["Direction", "Jump", "Hook", "Fire", "Aim"]
BUCKETS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 126)]
IDLE = 0


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def bucket(k):
    for lo, hi in BUCKETS:
        if lo <= k < hi:
            return f"{lo}-{hi - 1}"
    return "?"


def main(path):
    direction, fly, spec, res = json.load(open(path))
    print(f"{path}: direction {direction}, {len(res)} starts of class {res[0]['class']}")
    base_held = [r for r in res if r["base_held"]]
    base_not = [r for r in res if not r["base_held"]]
    print(f"unswapped run held in {len(base_held)}/{len(res)} starts ({100*len(base_held)/len(res):.1f}%)")
    # The population whose flips are interesting.
    pop = base_not if direction == "forward" else base_held
    print(f"analysed population (starts where a flip is possible): {len(pop)}")

    def flipped(r, s):
        return s["held"] != r["base_held"]

    # 1. Singles: critical decisions.
    tot = Counter()
    flips = Counter()
    first = []
    per_start = []
    for r in pop:
        crit = [s["from"] for s in r["singles"] if flipped(r, s)]
        per_start.append(len(crit))
        if crit:
            first.append(min(crit))
        for s in r["singles"]:
            b = bucket(s["from"])
            tot[b] += 1
            flips[b] += flipped(r, s)
    n_any = sum(1 for c in per_start if c)
    print(f"\n1) one decision swapped (whole action): starts with at least one critical decision: {n_any}/{len(pop)} ({100*n_any/max(1,len(pop)):.1f}%)")
    print("   decision (from the freeze)   swaps   flips   rate")
    for lo, hi in BUCKETS:
        b = f"{lo}-{hi - 1}"
        if tot[b]:
            print(f"   {b:>8s}                   {tot[b]:6d} {flips[b]:6d}  {100*flips[b]/tot[b]:5.1f}%")
    if first:
        fs = sorted(first)
        print(f"   first critical decision: median {fs[len(fs)//2]}, share at decision <4: {100*sum(1 for x in fs if x<4)/len(fs):.0f}%, <16: {100*sum(1 for x in fs if x<16)/len(fs):.0f}%")

    # 2. Windows.
    print("\n2) the alternative really plays for a window of decisions [from, from+len):")
    wt = defaultdict(lambda: [0, 0])
    for r in pop:
        for w in r["windows"]:
            key = (w["from"], w["len"])
            wt[key][0] += 1
            wt[key][1] += w["held"]
    for key in sorted(wt, key=lambda k: (k[1], k[0])):
        n, k = wt[key]
        print(f"   from {key[0]:3d} len {key[1]:3d}: outcome held in {k}/{n} ({100*k/n:.1f}%)")

    # 3. How the actions differ at the critical decisions vs elsewhere.
    def diffs(step):
        m, a = step["main"], step["alt"]
        if a is None:
            return None
        d = {
            "Direction": m["dir"] != a["dir"],
            "Jump": m["jump"] != a["jump"],
            "Hook": m["hook"] != a["hook"],
            "Fire": m["fire"] != a["fire"],
        }
        d["Hook press (alt presses, main not)"] = (not m["hook"]) and a["hook"]
        d["Hook release (alt releases, main holds)"] = m["hook"] and (not a["hook"])
        d["aim_err_deg"] = math.degrees(abs(wrap(m["aim"] - a["aim"])))
        d["throw"] = (m["hook"] and step["state"] == IDLE) or (a["hook"] and step["state"] == IDLE)
        d["dir_pair"] = (m["dir"], a["dir"])
        return d

    crit_rows, other_rows = [], []
    for r in pop:
        crit_k = {s["from"] for s in r["singles"] if flipped(r, s)}
        tested = {s["from"] for s in r["singles"]}
        for st in r["shadow"]:
            if st["k"] not in tested:
                continue
            d = diffs(st)
            if d is None:
                continue
            (crit_rows if st["k"] in crit_k else other_rows).append(d)
    print(f"\n3) action components (main vs alternative at the same state), critical decisions (n={len(crit_rows)}) vs the other tested ones (n={len(other_rows)}):")
    keys = ["Direction", "Jump", "Hook", "Hook press (alt presses, main not)", "Hook release (alt releases, main holds)", "Fire"]
    for k in keys:
        a = sum(d[k] for d in crit_rows) / max(1, len(crit_rows))
        b = sum(d[k] for d in other_rows) / max(1, len(other_rows))
        print(f"   {k:42s} critical {100*a:5.1f}%   other {100*b:5.1f}%")
    for name, rows in [("critical", crit_rows), ("other", other_rows)]:
        thr = sorted(d["aim_err_deg"] for d in rows if d["throw"])
        if thr:
            print(f"   aim difference at throws ({name}, n={len(thr)}): median {thr[len(thr)//2]:.1f} deg, share > 15 deg {100*sum(1 for x in thr if x>15)/len(thr):.0f}%")
    c = Counter(d["dir_pair"] for d in crit_rows if d["Direction"])
    o = Counter(d["dir_pair"] for d in other_rows if d["Direction"])
    print("   direction when it differs (main -> alt), critical:", {f"{a}->{b}": n for (a, b), n in sorted(c.items())}, " other:", {f"{a}->{b}": n for (a, b), n in sorted(o.items())})

    # 4. Component attribution.
    print("\n4) one component swapped alone at a critical decision (first critical decisions of each start):")
    ct = defaultdict(lambda: [0, 0])
    any_single = 0
    tot_crit = 0
    for r in pop:
        by_k = defaultdict(dict)
        for c in r["components"]:
            by_k[c["from"]][c["components"][0]] = flipped(r, c)
        for k, d in by_k.items():
            tot_crit += 1
            any_single += any(d.values())
            for comp, fl in d.items():
                ct[comp][0] += 1
                ct[comp][1] += fl
    for comp in COMPONENTS:
        n, k = ct[comp]
        if n:
            print(f"   {comp:10s}: alone flips the outcome at {k}/{n} critical decisions ({100*k/n:.0f}%)")
    if tot_crit:
        print(f"   at least one single component suffices at {any_single}/{tot_crit} ({100*any_single/tot_crit:.0f}%); the rest need two or more")


if __name__ == "__main__" and sys.argv[1] != "--agreement":
    main(sys.argv[1])


def agreement(path, first=16):
    """Shadow-only file(s): how often the main brain's action equals the alternative's at the same state, per component, in the first
    `first` decisions after the freeze and overall; the hook split by who presses/releases and by the observed own hook state."""
    direction, fly, spec, res = json.load(open(path))
    print(f"{path}: {len(res)} starts, {direction}; agreement of the main brain with the alternative (planner) at the same states")
    for label, lo, hi in [(f"first {first} decisions", 0, first), ("decisions 16-125", first, 10**6)]:
        rows = [(st, r) for r in res for st in r["shadow"] if lo <= st["k"] < hi and st["alt"] is not None]
        n = len(rows)
        if not n:
            continue
        agree = lambda f: 100 * sum(1 for st, _ in rows if f(st["main"], st["alt"])) / n
        print(f"  {label} (n={n}):")
        print(f"    direction {agree(lambda m, a: m['dir'] == a['dir']):5.1f}% | jump {agree(lambda m, a: m['jump'] == a['jump']):5.1f}% | hook key {agree(lambda m, a: m['hook'] == a['hook']):5.1f}% | fire {agree(lambda m, a: m['fire'] == a['fire']):5.1f}%")
        thr = [abs(wrap(st["main"]["aim"] - st["alt"]["aim"])) for st, _ in rows if st["state"] == IDLE and st["main"]["hook"] and st["alt"]["hook"]]
        if thr:
            thr.sort()
            print(f"    aim when both throw (n={len(thr)}): median difference {math.degrees(thr[len(thr)//2]):.1f} deg, within 15 deg {100*sum(1 for x in thr if math.degrees(x) <= 15)/len(thr):.0f}%")
        for name, sel in [("state idle (press decision)", lambda st: st["state"] == IDLE), ("state out (hold/release decision)", lambda st: st["state"] != IDLE)]:
            sub = [st for st, _ in rows if sel(st)]
            if sub:
                mh = sum(1 for st in sub if st["main"]["hook"]) / len(sub)
                ah = sum(1 for st in sub if st["alt"]["hook"]) / len(sub)
                ag = sum(1 for st in sub if st["main"]["hook"] == st["alt"]["hook"]) / len(sub)
                print(f"    hook key, {name}: n={len(sub)}, main {100*mh:.1f}% / planner {100*ah:.1f}%, agree {100*ag:.1f}%")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--agreement":
    agreement(sys.argv[2])
