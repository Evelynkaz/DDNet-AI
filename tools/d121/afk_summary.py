#!/usr/bin/env python3
"""Task 3.23 (E-038): paired statistics of the AFK confirmation (`duel_stats --jsonl` of configs/arena/d121-afk-confirm.toml).

usage: afk_summary.py afk.jsonl
Conditions are "box <rate> <arm>"; game g has the same seed and start in every condition of a rate. Prints blocks (credited wins W) per rate and arm with the Wilson
95% interval, the mean game length, and the paired difference of `afk` to `base` with the exact McNemar test.
"""
import json
import math
import sys
from collections import defaultdict

Z = 1.959963984540054


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = p + Z * Z / (2 * n)
    m = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


rows = defaultdict(dict)
for line in open(sys.argv[1]):
    r = json.loads(line)
    rows[r["condition"]][r["game"]] = r
rates = []
for cond in rows:
    rate = cond.split()[1]
    if rate not in rates:
        rates.append(rate)
print("| rate | arm | blocks | % [95%] | mean game ticks | T |")
print("|---|---|---:|---|---:|---:|")
for rate in rates:
    for arm in ("base", "afk"):
        g = rows[f"box {rate} {arm}"]
        n = len(g)
        w = sum(r["credited"] for r in g.values())
        t = sum(r["result"] == "T" for r in g.values())
        lo, hi = wilson(w, n)
        mean = sum(r["end_tick"] for r in g.values()) / n
        print(f"| {rate} | {arm} | {w}/{n} | {100 * w / n:.1f} [{100 * lo:.1f}; {100 * hi:.1f}] | {mean:.0f} | {t} |")
print()
for rate in rates:
    a, b = rows[f"box {rate} afk"], rows[f"box {rate} base"]
    only_a = sum(a[k]["credited"] and not b[k]["credited"] for k in a)
    only_b = sum(b[k]["credited"] and not a[k]["credited"] for k in a)
    print(f"{rate}: afk-only {only_a}, base-only {only_b}, p = {mcnemar(only_a, only_b):.4f}")
