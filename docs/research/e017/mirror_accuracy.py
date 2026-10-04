#!/usr/bin/env python3
"""E-017: how well does the opponent model's first predicted input match what the opponent really sends next? From `truth.jsonl` files of
`hybrid_losses --mode truth` run with a hybrid whose opponent model is on (`hybrid.mirror_live_first` in each decision) over whole games.

  mirror_accuracy.py label=truth.jsonl ...

Per label: decisions with a prediction, direction and hook agreement of the model and of "the opponent keeps its input" (hold), and the share of
the model being right among the decisions where exactly one of the two is.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import pct  # noqa: E402


def same(a, b):
    return a[0] == b[0] and (a[2] != 0) == (b[2] != 0)


for spec in sys.argv[1:]:
    label, _, path = spec.rpartition("=")
    n = dm = dh = hm = hh = one = mine = 0
    for line in open(path):
        g = json.loads(line)
        for d in g["decisions"]:
            o = d["opp"]
            live = d["hybrid"].get("mirror_live_first")
            if not (o["actual"] and o["hold"] and live):
                continue
            a, h = o["actual"], o["hold"]
            n += 1
            dm += a[0] == live[0]
            dh += a[0] == h[0]
            hm += a[2] == live[2]
            hh += a[2] == h[2]
            tm, th = same(a, live), same(a, h)
            if tm != th:
                one += 1
                mine += tm
    print(f"{label or path}: {n} decisions; direction model {pct(dm, n)} hold {pct(dh, n)}; hook model {pct(hm, n)} hold {pct(hh, n)}; "
          f"model right when exactly one is: {pct(mine, one)}")
