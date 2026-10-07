#!/usr/bin/env python3
"""E-029 (task 8.6): tables and paired tests from `train es hook-eval` files (and the planner's, as the reference).

  hook_tables.py <set: train-val|holdout> <first.json> [<other.json> ...]

Prints, per file and start class (V/B/H/all): hook start rate (key pressed while the observed own hook state is idle), the same in the first
four decisions, the share of starts with a press within 8 / 24 / 50 ticks of the freeze, the median first-press tick, and the aim error at the
throw; then, for every file after the first, the paired difference against the first (exact McNemar on the starts) for "pressed within 8 ticks"
and "opening throw" and a paired sign test on the first-press tick. Files must come from the same `--starts` of the same bank.
"""
import json
import math
import sys

IDLE = 0
OPEN = 4


def mcnemar(a, b):
    """Exact two-sided McNemar p for paired booleans; returns (only_a, only_b, p)."""
    oa = sum(1 for x, y in zip(a, b) if x and not y)
    ob = sum(1 for x, y in zip(a, b) if y and not x)
    n = oa + ob
    if n == 0:
        return oa, ob, 1.0
    k = min(oa, ob)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return oa, ob, min(1.0, 2 * p)


def load(path, which):
    label, sets = json.load(open(path))
    for name, recs in sets:
        if name == which:
            return label, recs
    raise SystemExit(f"{path}: no set {which}")


def per_start(rec):
    d = rec["decisions"]
    first = next((x["tick"] for x in d if x["hook"]), None)
    return {
        "press8": first is not None and first <= 8,
        "press24": first is not None and first <= 24,
        "open_throw": any(x["hook"] and x["state"] == IDLE for x in d[:OPEN]),
        "first": first,
    }


def table(label, recs):
    print(f"\n{label}")
    for cls in ["V", "B", "H", "all"]:
        rs = [r for r in recs if cls == "all" or r["class"] == cls]
        if not rs:
            continue
        idle = [x for r in rs for x in r["decisions"] if x["state"] == IDLE]
        start = sum(x["hook"] for x in idle) / max(1, len(idle))
        op = [x for r in rs for x in r["decisions"][:OPEN]]
        open_key = sum(x["hook"] for x in op) / max(1, len(op))
        ps = [per_start(r) for r in rs]
        p8 = sum(p["press8"] for p in ps) / len(ps)
        p24 = sum(p["press24"] for p in ps) / len(ps)
        firsts = sorted(p["first"] for p in ps if p["first"] is not None)
        med = firsts[len(firsts) // 2] if firsts else float("nan")
        errs = sorted(x["throw_err"] for r in rs for x in r["decisions"] if x.get("throw_err") is not None)
        aim = math.degrees(errs[len(errs) // 2]) if errs else float("nan")
        w15 = sum(1 for e in errs if math.degrees(e) <= 15) / max(1, len(errs))
        held = sum(r["outcome_held"] for r in rs) / len(rs)
        out = sum(r["outcome_self_out"] for r in rs) / len(rs)
        print(
            f"  {cls:>3} n={len(rs):4d} | start rate (idle) {100*start:5.1f}% | opening key {100*open_key:5.1f}% | press<=8t {100*p8:5.1f}% "
            f"<=24t {100*p24:5.1f}% | median first {med:5.1f}t | throw aim median {aim:5.1f} deg, <=15deg {100*w15:4.0f}% (n={len(errs)}) | held {100*held:4.1f}% own-out {100*out:4.1f}%"
        )


def main():
    which = sys.argv[1]
    files = sys.argv[2:]
    loaded = [load(f, which) for f in files]
    for label, recs in loaded:
        table(label, recs)
    if len(loaded) > 1:
        l0, r0 = loaded[0]
        for l1, r1 in loaded[1:]:
            assert len(r0) == len(r1)
            print(f"\npaired {l0} -> {l1} ({which})")
            for cls in ["V", "B", "H", "all"]:
                idx = [i for i, r in enumerate(r0) if cls == "all" or r["class"] == cls]
                if not idx:
                    continue
                a = [per_start(r0[i]) for i in idx]
                b = [per_start(r1[i]) for i in idx]
                for key in ["press8", "open_throw"]:
                    xa, xb = [p[key] for p in a], [p[key] for p in b]
                    oa, ob, p = mcnemar(xa, xb)
                    print(f"  {cls:>3} {key:10s}: {100*(sum(xb)-sum(xa))/len(idx):+5.1f} pp (only first {oa}, only second {ob}, McNemar p = {p:.4f})")
                both = [(x["first"], y["first"]) for x, y in zip(a, b) if x["first"] is not None and y["first"] is not None]
                earlier = sum(1 for x, y in both if y < x)
                later = sum(1 for x, y in both if y > x)
                n = earlier + later
                p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(earlier, later) + 1)) / 2**n)
                print(f"  {cls:>3} first press tick: second earlier on {earlier}, later on {later} of {len(both)} starts that press in both (sign test p = {p:.4f})")


if __name__ == "__main__":
    main()
