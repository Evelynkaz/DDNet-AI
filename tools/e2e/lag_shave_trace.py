#!/usr/bin/env python3
"""Task 3.16: the effective input lag of every decision from the input traces of a `lag_shave.sh` run (`*.trace.jsonl`, focal bot only).

  lag_shave_trace.py RUN_DIR

Joins each snapshot record (`k:a`: its tick `snap`, the last input sent `pred`) with the decision that followed it (`k:d`, `first == pred + 1`):
lag_plan = exp - snap - 1 (the ticks the brain's world was rolled: the horizon, the arena's `lag`), lag_sent = tick - snap - 1 (the lag the input really got).
Decisions of the brain only (`brain == 1`). Prints per cell (margin, budget): the share of decisions by lag_sent, the mean, and the share sent later than planned.
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict


def cell(path):
    rows = []
    last_a = None
    for line in open(path):
        if '"k":"a"' in line:
            last_a = json.loads(line)
        elif '"k":"d"' in line and last_a is not None:
            d = json.loads(line)
            if d["brain"] == 1 and d["first"] == last_a["pred"] + 1 and last_a["slack_us"] >= 0:
                rows.append((d["exp"] - last_a["snap"] - 1, d["tick"] - last_a["snap"] - 1, last_a["slack_us"] / 1000))
            last_a = None
    return rows


def main():
    d = sys.argv[1]
    cells = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(d, "*.trace.jsonl"))):
        lab = os.path.basename(f)[: -len(".trace.jsonl")]
        p = lab.split("-")
        cells[(p[0][1:], p[1][1:])].extend(cell(f))
    print("| margin | budget | decisions | lag <= 1 | lag 2 | lag 3 | lag >= 4 | mean lag | mean planned lag | sent later than planned | slack p50 ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(cells, key=lambda k: (int(k[0]) if k[0].isdigit() else 999, int(k[1]))):
        r = cells[k]
        n = len(r)
        if not n:
            continue
        c = Counter(min(x[1], 4) for x in r)
        sh = lambda e: 100 * c.get(e, 0) / n
        sl = sorted(x[2] for x in r)
        print(
            f"| {k[0]} | {k[1]} | {n} | {sh(0) + sh(1):.1f}% | {sh(2):.1f}% | {sh(3):.1f}% | {sh(4):.1f}% | {sum(x[1] for x in r) / n:.2f} | {sum(x[0] for x in r) / n:.2f} | "
            f"{100 * sum(x[1] > x[0] for x in r) / n:.1f}% | {sl[n // 2]:.1f} |"
        )


if __name__ == "__main__":
    main()
