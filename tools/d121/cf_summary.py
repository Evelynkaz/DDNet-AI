#!/usr/bin/env python3
"""Task 3.23 (E-038): pooled counterfactual table of `pm_counterfactual` JSONL files (one per arm) by start offset.

usage: cf_summary.py base=cf/base.jsonl counter=cf/counter.jsonl ...
Prints, per arm and per start offset (ticks before the decisive freeze), how often we froze first (`we` or `both`) of the runs, and the
paired difference to the first arm (same clip, offset and seed).
"""
import json
import sys
from collections import defaultdict


def load(path):
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[(r["clip"], r["offset"], r["seed"], r["cell"])] = r
    return rows


def lost(r):
    return r["first"] in ("we", "both")


def main():
    arms = [a.split("=", 1) for a in sys.argv[1:]]
    data = {name: load(path) for name, path in arms}
    base_name = arms[0][0]
    base = data[base_name]
    offsets = sorted({k[1] for k in base}, reverse=True)
    print("| arm | " + " | ".join(f"-{o}" for o in offsets) + " | all |")
    print("|---|" + "---:|" * (len(offsets) + 1))
    for name, _ in arms:
        d = data[name]
        cells = []
        tot = n_tot = 0
        for o in offsets:
            ks = [k for k in d if k[1] == o]
            n = len(ks)
            c = sum(lost(d[k]) for k in ks)
            tot += c
            n_tot += n
            cells.append(f"{c}/{n}")
        print(f"| {name} | " + " | ".join(cells) + f" | {tot}/{n_tot} = {100 * tot / max(1, n_tot):.1f}% |")
    print()
    print("paired against", base_name, "(runs lost only in the arm / only in the base; McNemar exact p)")
    from math import comb

    for name, _ in arms[1:]:
        d = data[name]
        for o in offsets + [None]:
            ks = [k for k in base if (o is None or k[1] == o) and k in d]
            a = sum(lost(d[k]) and not lost(base[k]) for k in ks)  # worse in arm
            b = sum(lost(base[k]) and not lost(d[k]) for k in ks)  # better in arm
            n = a + b
            p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(a, b) + 1)) / 2**n) if n else 1.0
            print(f"{name:>20} {'all' if o is None else '-' + str(o):>4}: worse {a:3d} better {b:3d}  p={p:.3f}")


main()
