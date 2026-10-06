#!/usr/bin/env python3
"""Task 3.11: the A/B table of live-timing runs.

    live_timing_table.py RUNS_DIR LABEL_PREFIX_A [LABEL_PREFIX_B ...]

Every `<prefix>*.json` (the harness's report) with its `<label>.server.json` (the teehistorian join, if present) is one run; a
row is the **median** over the runs of a prefix (the runs of the variants are interleaved, the machine is shared, and the spread
between runs of one variant is larger than most differences, so the min-max is printed too).
"""

import glob
import json
import os
import statistics
import sys


def load(dirname, prefix):
    runs = []
    for f in sorted(glob.glob(os.path.join(dirname, prefix + "*.json"))):
        if f.endswith(".server.json"):
            continue
        r = json.load(open(f))
        sv = f[: -len(".json")] + ".server.json"
        r["server"] = json.load(open(sv)) if os.path.exists(sv) else None
        runs.append(r)
    return runs


def col(runs, getter):
    v = []
    for r in runs:
        try:
            x = getter(r)
        except (KeyError, TypeError, ZeroDivisionError):
            x = None
        if x is not None:
            v.append(x)
    return v


def fmt(v, scale=1.0, nd=2, pct=False):
    if not v:
        return "-"
    m = statistics.median(v) * scale
    lo, hi = min(v) * scale, max(v) * scale
    if pct:
        return f"{m*100:.1f}% [{lo*100:.1f}-{hi*100:.1f}]"
    return f"{m:.{nd}f} [{lo:.{nd}f}-{hi:.{nd}f}]"


ROWS = [
    ("runs", lambda rs: str(len(rs))),
    ("load1 before", lambda rs: fmt(col(rs, lambda r: r["load1_before"]), 1, 1)),
    ("decisions", lambda rs: fmt(col(rs, lambda r: r["slots"]["decisions"]), 1, 0)),
    ("thread CPU s: ddai-client (all driver threads)", lambda rs: fmt(col(rs, lambda r: r["thread_cpu_s"]["ddai-client"]), 1, 2)),
    ("missed first slot", lambda rs: fmt(col(rs, lambda r: r["slots"]["missed_first_slot_share"]), pct=True)),
    ("as predicted (tick the world assumed)", lambda rs: fmt(col(rs, lambda r: r["slots"]["as_predicted_share"]), pct=True)),
    ("later than predicted", lambda rs: fmt(col(rs, lambda r: r["slots"]["later_than_predicted"] / r["slots"]["decisions"]), pct=True)),
    ("BRAIN decisions: missed first slot", lambda rs: fmt(col(rs, lambda r: r["latency_us"]["slots"]["brain_missed_first_slot"] / r["latency_us"]["slots"]["brain_decisions"]), pct=True)),
    ("BRAIN decisions: later than predicted", lambda rs: fmt(col(rs, lambda r: r["latency_us"]["slots"]["brain_later_than_predicted"] / r["latency_us"]["slots"]["brain_decisions"]), pct=True)),
    ("BRAIN decisions (count)", lambda rs: fmt(col(rs, lambda r: r["latency_us"]["slots"]["brain_decisions"]), 1, 0)),
    ("server: BRAIN applied == expected", lambda rs: fmt(col(rs, lambda r: r["server"]["brain_exact_share_vs_expected"]), pct=True)),
    ("input late at the server (INPUTTIMING<0)", lambda rs: fmt(col(rs, lambda r: r["input_margin"]["late_fraction"]), pct=True)),
    ("server: applied == expected (decisions)", lambda rs: fmt(col(rs, lambda r: r["server"]["exact_share_vs_expected"]), pct=True)),
    ("server: applied == intended (changes)", lambda rs: fmt(col(rs, lambda r: r["server"]["exact_share_vs_intended"]), pct=True)),
    ("total ms p50", lambda rs: fmt(col(rs, lambda r: r["latency_us"]["total"]["p50"]), 1e-3)),
    ("total ms p95", lambda rs: fmt(col(rs, lambda r: r["latency_us"]["total"]["p95"]), 1e-3)),
    ("total ms p99", lambda rs: fmt(col(rs, lambda r: r["latency_us"]["total"]["p99"]), 1e-3)),
    ("brain ms p50", lambda rs: fmt(col(rs, lambda r: r["latency_us"]["brain"]["p50"]), 1e-3)),
    ("brain ms p99", lambda rs: fmt(col(rs, lambda r: r["latency_us"]["brain"]["p99"]), 1e-3)),
    ("overhead ms p99", lambda rs: fmt(col(rs, lambda r: r["latency_us"]["overhead"]["p99"]), 1e-3)),
    ("queue ms p50 / p99", lambda rs: fmt(col(rs, lambda r: r["latency_us"]["queue"]["p50"]), 1e-3) + " / " + fmt(col(rs, lambda r: r["latency_us"]["queue"]["p99"]), 1e-3)),
    ("pickup ms p50 / p99", lambda rs: fmt(col(rs, lambda r: r["latency_us"]["pickup"]["p50"]), 1e-3) + " / " + fmt(col(rs, lambda r: r["latency_us"]["pickup"]["p99"]), 1e-3)),
    ("send lag ms p50 / p99", lambda rs: fmt(col(rs, lambda r: r["input_margin"]["send_lag_us"]["p50"]), 1e-3) + " / " + fmt(col(rs, lambda r: r["input_margin"]["send_lag_us"]["p99"]), 1e-3)),
    ("wire ms p50", lambda rs: fmt(col(rs, lambda r: r["latency_us"]["wire"]["p50"]), 1e-3, 1)),
    ("wire ms p99", lambda rs: fmt(col(rs, lambda r: r["latency_us"]["wire"]["p99"]), 1e-3, 1)),
    ("margin ms (final)", lambda rs: fmt(col(rs, lambda r: r["input_margin"]["margin_ms"]), 1, 1)),
    ("horizon ticks p50", lambda rs: fmt(col(rs, lambda r: r["latency_us"]["horizon_ticks"]["p50"]), 1, 0)),
    ("hooks fired", lambda rs: fmt(col(rs, lambda r: r["stats"]["hooks_fired"]), 1, 0)),
]


def main():
    d, prefixes = sys.argv[1], sys.argv[2:]
    sets = [load(d, p) for p in prefixes]
    w = max(len(n) for n, _ in ROWS) + 2
    print("".ljust(w) + " | ".join(p.ljust(26) for p in prefixes))
    for name, f in ROWS:
        cells = []
        for rs in sets:
            try:
                cells.append(f(rs))
            except Exception:
                cells.append("-")
        print(name.ljust(w) + " | ".join(c.ljust(26) for c in cells))


if __name__ == "__main__":
    main()
