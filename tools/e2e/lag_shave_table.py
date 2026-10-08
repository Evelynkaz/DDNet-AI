#!/usr/bin/env python3
"""Task 3.16: the table of a `tools/e2e/lag_shave.sh` run -- one row per (margin, budget) cell, medians over the rounds [min-max].

  lag_shave_table.py RUN_DIR [--md]

Columns: brain decisions that missed their first slot (share), horizon ticks p50/p90 (the ticks the brain's world was predicted past the snapshot:
effective lag minus one), slack p50 (arrival -> the first input due) and ready p50 (what the decision was aimed to need), brain p50/p99, the late
fraction of the inputs (INPUTTIMING < 0) and the margin in force at the end.
"""
import glob
import json
import os
import statistics
import sys
from collections import defaultdict


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def rng(xs):
    xs = [x for x in xs if x is not None]
    return f"[{min(xs):.3g}-{max(xs):.3g}]" if len(xs) > 1 else ""


def main():
    d = sys.argv[1]
    cells = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        if f.endswith(".server.json"):
            continue
        j = json.load(open(f))
        lab = os.path.basename(f)[:-5]
        parts = lab.split("-")
        if len(parts) < 3 or not parts[0].startswith("m"):
            continue
        cells[(parts[0][1:], parts[1][1:])].append(j)

    def key(k):
        return (999 if k[0] == "a" else int(k[0]), int(k[1]))

    rows = []
    for k in sorted(cells, key=key):
        runs = cells[k]
        def col(fn):
            out = []
            for j in runs:
                try:
                    out.append(fn(j))
                except (KeyError, TypeError, ZeroDivisionError):
                    out.append(None)
            return out
        lat = lambda j: j["latency_us"]
        miss = col(lambda j: lat(j)["slots"]["brain_missed_first_slot"] / lat(j)["slots"]["brain_decisions"])
        later = col(lambda j: lat(j)["slots"]["brain_later_than_predicted"] / lat(j)["slots"]["brain_decisions"])
        h50 = col(lambda j: lat(j)["horizon_ticks"]["p50"])
        h90 = col(lambda j: lat(j)["horizon_ticks"]["p90"])
        sl = col(lambda j: lat(j)["slack"]["p50"] / 1000)
        rd = col(lambda j: lat(j)["ready"]["p50"] / 1000)
        b50 = col(lambda j: lat(j)["brain"]["p50"] / 1000)
        b99 = col(lambda j: lat(j)["brain"]["p99"] / 1000)
        late = col(lambda j: j["input_margin"]["late_fraction"] * 100)
        mg = col(lambda j: j["input_margin"]["margin_ms"])
        n = col(lambda j: lat(j)["slots"]["brain_decisions"])
        rows.append((k, len(runs), med(n), med(miss), rng([x and x * 100 for x in miss]), med(later), med(h50), med(h90), med(sl), med(rd), med(b50), med(b99), med(late), med(mg)))
    print("| margin | budget | runs | brain decisions | missed first slot | (range) | later than predicted | horizon p50 / p90 | slack p50 ms | ready p50 ms | brain p50 / p99 ms | late inputs % | margin end |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    f = lambda x, p=1: "-" if x is None else f"{x:.{p}f}"
    for (k, nr, n, miss, mr, later, h50, h90, sl, rd, b50, b99, late, mg) in rows:
        print(f"| {k[0]} | {k[1]} | {nr} | {f(n, 0)} | {f(miss and miss * 100)}% | {mr} | {f(later and later * 100)}% | {f(h50, 0)} / {f(h90, 0)} | {f(sl)} | {f(rd)} | {f(b50)} / {f(b99)} | {f(late, 2)} | {f(mg, 0)} |")


if __name__ == "__main__":
    main()
