#!/usr/bin/env python3
"""E-028: the go criterion of the opponent-input predictor, from the JSONL files of two arena runs (the arms, same seeds and layouts).

  compare.py --base BASE.jsonl [BASE2.jsonl ...] --model MODEL.jsonl [MODEL2.jsonl ...] [--pool "3/0,3/1,3/2"] [--guard "3/3"]

Several files per arm are several seed sets: their games are kept apart (paired by file position and game index) and pooled.
A condition is named "<arm> <we>/<they>" (a lag pair such as `3/0`). Per lag pair: credited wins of both arms with Wilson intervals, the difference in
percentage points, the games only the model arm won / only the base arm won (credited, paired by game index) and the exact McNemar p. Then the pool of
the lag pairs of `--pool` (the sum of the paired counts, the pooled difference) and the guard pair (`--guard`).
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e017"))
from stats import mcnemar_exact, pct  # noqa: E402


def load(paths):
    arms = defaultdict(dict)
    for pi, p in enumerate(paths):
        for f in sorted(glob.glob(p)) if not os.path.isdir(p) else sorted(glob.glob(os.path.join(p, "*.jsonl"))):
            for line in open(f):
                if line.strip():
                    try:
                        v = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    arms[v["condition"]][(pi, v["game"])] = (v["result"], bool(v["credited"]), bool(v.get("held", False)))
    return arms


def lag_of(name):
    m = re.search(r"(\d+/\d+)\s*$", name)
    return m.group(1) if m else name


def win(g):
    return g[0] == "W" and g[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--pool", default="3/0,3/1,3/2")
    ap.add_argument("--guard", default="3/3")
    a = ap.parse_args()
    base = {lag_of(k): v for k, v in load(a.base).items()}
    model = {lag_of(k): v for k, v in load(a.model).items()}
    print("| lag (we/they) | games | base credited | model credited | diff pp | only model / only base | McNemar p | decided W/(W+L) base -> model |")
    print("|---|---:|---|---|---:|---|---:|---|")
    tot = {}

    def row(lag):
        b, m = base[lag], model[lag]
        idx = sorted(set(b) & set(m))
        wb = sum(win(b[i]) for i in idx)
        wm = sum(win(m[i]) for i in idx)
        om = sum(win(m[i]) and not win(b[i]) for i in idx)
        ob = sum(win(b[i]) and not win(m[i]) for i in idx)
        n = len(idx)

        def dec(g):
            w = sum(g[i][0] == "W" for i in idx)
            l = sum(g[i][0] == "L" for i in idx)
            return f"{100*w/(w+l):.1f}%" if w + l else "-"

        print(f"| {lag} | {n} | {pct(wb, n)} | {pct(wm, n)} | {100*(wm-wb)/n:+.1f} | {om} / {ob} | {mcnemar_exact(om, ob):.3g} | {dec(b)} -> {dec(m)} |")
        return n, wb, wm, om, ob

    for lag in a.pool.split(","):
        if lag in base and lag in model:
            tot[lag] = row(lag)
    if len(tot) > 1:
        n = sum(t[0] for t in tot.values())
        wb = sum(t[1] for t in tot.values())
        wm = sum(t[2] for t in tot.values())
        om = sum(t[3] for t in tot.values())
        ob = sum(t[4] for t in tot.values())
        print(f"| **pooled {a.pool}** | {n} | {pct(wb, n)} | {pct(wm, n)} | **{100*(wm-wb)/n:+.1f}** | {om} / {ob} | **{mcnemar_exact(om, ob):.3g}** | |")
    for lag in a.guard.split(","):
        if lag in base and lag in model:
            row(lag)


if __name__ == "__main__":
    main()
